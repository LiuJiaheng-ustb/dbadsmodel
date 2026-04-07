import os
import errno
import logging
import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch
from ocpmodels.common.registry import registry
from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS
from ocpmodels.models.scn.smearing import GaussianSmearing
from ocpmodels.common.utils import conditional_grad, _report_incompat_keys

from models.encoders import *
from models.freeze_layers import frozen_layer_name, conv_name


def atom_height(data):
    z_norm_vecs = torch.tensor([], device=data.pos.device)
    for cell in data.cell:
        a = cell[0]
        b = cell[1]
        z_vec = torch.cross(a, b)
        z_norm_vec = F.normalize(z_vec, dim=-1).unsqueeze(0)
        z_norm_vecs = torch.cat((z_norm_vecs, z_norm_vec), 0)
    surface_batch = data.batch
    z_norm_vecs = z_norm_vecs[surface_batch].unsqueeze(1)
    pos = data.pos.unsqueeze(2)
    z_atoms = torch.bmm(z_norm_vecs, pos)
    z_atoms = torch.squeeze(z_atoms)
    z_max = scatter(z_atoms, surface_batch, dim_size=len(data), reduce='max')
    z_min = scatter(z_atoms, surface_batch, dim_size=len(data), reduce='min')
    z_max = z_max[surface_batch]
    z_min = z_min[surface_batch]
    h = (z_atoms - z_min) / (z_max - z_min)

    assert not torch.isnan(h).any()
    assert not torch.isinf(h).any() 
    return h


def get_graph_encoder(name, kwargs):
    # assert name in [], f"Graph Encoder '{name}' is unavailable!"
    encoder = globals().get(name)
    if encoder:
        return encoder(**kwargs)
    else:
        raise ValueError(f"Graph Encoder '{name}' not found!")


def _build_token_batch(data, key):
    """
    Build per-token graph index for custom attributes (e.g. ads_pos).
    Uses Batch._slice_dict when available, and falls back to a single graph.
    """
    value = getattr(data, key)
    if not torch.is_tensor(value):
        raise ValueError(f"{key} must be a tensor")

    num_tokens = value.size(0)
    if num_tokens == 0:
        return torch.zeros(0, device=value.device, dtype=torch.long)

    if hasattr(data, "_slice_dict") and key in data._slice_dict:
        slices = data._slice_dict[key]
        if torch.is_tensor(slices):
            slices = slices.to(value.device)
            counts = slices[1:] - slices[:-1]
            return torch.repeat_interleave(
                torch.arange(counts.numel(), device=value.device), counts
            )

    return torch.zeros(num_tokens, device=value.device, dtype=torch.long)


class AdsGraphEncoder(nn.Module):
    """
    Adsorbate token encoder: atom embedding + position embedding + self-attention.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        embeddings = KHOT_EMBEDDINGS
        self.embedding = torch.zeros(100, len(embeddings[1]))
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])

        self.atom_embedding = nn.Linear(len(embeddings[1]), hidden_dim)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_attns = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        ads_pos = data.ads_pos
        ads_z = data.ads_atom_numb
        if ads_z.dim() > 1:
            ads_z = ads_z.view(-1)
        ads_z = ads_z.long()

        if self.embedding.device != ads_pos.device:
            self.embedding = self.embedding.to(ads_pos.device)

        ads_batch = _build_token_batch(data, "ads_pos")
        x = self.embedding[ads_z - 1]
        token_emb = self.atom_embedding(x) + self.pos_embedding(ads_pos)

        dense, mask = to_dense_batch(token_emb, ads_batch)
        for attn in self.self_attns:
            out = attn(dense, dense, dense, key_padding_mask=~mask)
            dense = self.norm(dense + out[0])
        token_emb = dense[mask]
        return token_emb, ads_batch


class CrossModal(nn.Module):
    def __init__(
        self,
        vec_emb_dim: int = 128,  # kept for config compatibility
        node_emb_dim: int = 128,  # kept for config compatibility
        hidden_dim: int = 128,
        out_channels: int = 1,
        num_gaussians: int = 50,
        num_heads: int = 4,
        attn_layers: int = 1,
        mlp_layers: int = 3,
        dropout: float = 0,
        act: str = "silu",
        norm: str = None,
    ) -> None:

        super(CrossModal, self).__init__()
        self.attn_layers = attn_layers
        self.mlp_layers = mlp_layers

        self.node_pos_expansion = GaussianSmearing(
            start=0.0,
            stop=1.0,
            num_gaussians=num_gaussians,
        )
        self.fc_pos_exp = nn.Linear(num_gaussians, hidden_dim)
        self.surf_proj = nn.Linear(node_emb_dim, hidden_dim)
        self.ads_proj = nn.Linear(vec_emb_dim, hidden_dim)

        # adsorbate -> surface cross attention
        self.cross_attn_as = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(attn_layers)
            ]
        )
        # surface -> adsorbate cross attention
        self.cross_attn_sa = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(attn_layers)
            ]
        )

        self.self_attn_ads = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(attn_layers)
            ]
        )
        self.self_attn_surf = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(attn_layers)
            ]
        )
        self.norm_ads = nn.LayerNorm(hidden_dim)
        self.norm_surf = nn.LayerNorm(hidden_dim)

        self.mlp = MLP(
            in_channels=hidden_dim * 2,
            hidden_channels=hidden_dim*2,
            out_channels=out_channels,
            num_layers=mlp_layers,
            dropout=dropout,
            act=act,
            norm=norm,
        )

    def forward(self, ads_emb, node_emb, ads_batch, surface, need_weights=False):

        # atom positional encoding along z direction for surface atoms
        batch = surface.batch
        atom_h = atom_height(surface)
        node_pos_emb = self.node_pos_expansion(atom_h)
        node_pos_emb = self.fc_pos_exp(node_pos_emb)

        surface_tokens = self.surf_proj(node_emb) + node_pos_emb
        ads_tokens = self.ads_proj(ads_emb)
        s_dense, s_mask = to_dense_batch(surface_tokens, batch)
        a_dense, a_mask = to_dense_batch(ads_tokens, ads_batch)

        # Cross attention: ads -> surface, and surface -> ads.
        cross_weights = []
        for attn_as, attn_sa in zip(self.cross_attn_as, self.cross_attn_sa):
            out_as = attn_as(
                a_dense,
                s_dense,
                s_dense,
                key_padding_mask=~s_mask,
                need_weights=need_weights,
            )
            out_sa = attn_sa(
                s_dense,
                a_dense,
                a_dense,
                key_padding_mask=~a_mask,
                need_weights=False,
            )

            a_dense = self.norm_ads(a_dense + out_as[0])
            s_dense = self.norm_surf(s_dense + out_sa[0])
            cross_weights.append(
                self.get_surface_weights(out_as, a_mask, s_mask)
            )

        # Optional intra-graph refinement after cross conditioning.
        self_weights = []
        for attn_ads, attn_surf in zip(self.self_attn_ads, self.self_attn_surf):
            out_ads = attn_ads(a_dense, a_dense, a_dense, key_padding_mask=~a_mask)
            out_surf = attn_surf(s_dense, s_dense, s_dense, key_padding_mask=~s_mask)
            a_dense = self.norm_ads(a_dense + out_ads[0])
            s_dense = self.norm_surf(s_dense + out_surf[0])
            if need_weights:
                self_weights.append(out_ads[1])

        z_a = self.masked_mean(a_dense, a_mask)
        z_s = self.masked_mean(s_dense, s_mask)
        energy = self.mlp(torch.cat([z_a, z_s], dim=1)).view(-1)

        if need_weights:
            return energy, cross_weights, self_weights
        else:
            return energy

    def masked_mean(self, dense, mask):
        mask_f = mask.unsqueeze(-1).float()
        total = (dense * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return total / denom

    def get_surface_weights(self, attn_out, ads_mask, surface_mask):
        out = []
        if attn_out[1] is None:
            return out
        # attn_out[1]: [B, n_ads_tokens, n_surface_tokens]
        attn_w = attn_out[1]
        for i in range(attn_w.size(0)):
            valid_ads = ads_mask[i]
            valid_surface = surface_mask[i]
            if valid_ads.sum() == 0:
                out.append(torch.zeros(valid_surface.sum(), device=attn_w.device))
                continue
            sample_weights = attn_w[i][valid_ads][:, valid_surface].mean(dim=0)
            out.append(sample_weights)
        return out


@registry.register_model("adsmt_arch")
class AdsMT_ARCH(nn.Module):
    """pyg implementation."""

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc: bool = True,
        otf_graph: bool = True,
        regress_forces: bool = False,
        pretrain: bool = False,
        ckpt_path: str = None,
        freeze_nblock: int = 0,
        graph_encoder: str = 'adsgt',
        graph_encoder_args: dict = None,
        ads_encoder_args: dict = None,
        desc_layers: int = 1,
        desc_hidden_dim: int = 128,
        cross_modal_args: dict = None,
    ):
        super().__init__()

        self.use_pbc = True
        self.otf_graph = True
        self.regress_forces = False

        # graph encoder for surface graph.
        self.surface_encoder = get_graph_encoder(graph_encoder, graph_encoder_args)

        if ads_encoder_args is None:
            ads_encoder_args = {
                "hidden_dim": desc_hidden_dim,
                "num_layers": max(desc_layers, 1),
                "num_heads": cross_modal_args.get("num_heads", 4) if cross_modal_args else 4,
                "dropout": cross_modal_args.get("dropout", 0.0) if cross_modal_args else 0.0,
            }
        self.ads_encoder = AdsGraphEncoder(**ads_encoder_args)

        # cross-modal encoder for fusing the embeddings of surface graph and adsorbate descriptors
        self.cross_encoder = CrossModal(**cross_modal_args)

        # load pretrained model from ckpt file
        if pretrain:
            assert ckpt_path is not None
            self.from_pretrain_ckpt(ckpt_path)
            # freeze some layer parameters
            if freeze_nblock > 0:
                self.freeze_layers(graph_encoder, freeze_nblock)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, need_weights=False):

        node_emb = self.surface_encoder(data)
        ads_emb, ads_batch = self.ads_encoder(data)

        if need_weights:
            energy, cross_weights, self_weights = self.cross_encoder(
                ads_emb, node_emb, ads_batch, data, need_weights
            )
            return energy, cross_weights, self_weights
        else:
            energy = self.cross_encoder(ads_emb, node_emb, ads_batch, data)
            return energy

    def from_pretrain_ckpt(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", ckpt_path
            )
        else:
            logging.info(f"Loading checkpoint from: {ckpt_path}")
            device = next(self.parameters()).device
            checkpoint = torch.load(ckpt_path, map_location=device)

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        incompat_keys = self.load_state_dict(new_dict, strict=False)
        return _report_incompat_keys(self, incompat_keys, strict=True)

    def freeze_layers(self, ge_name, nblock=0):
        frozen_layers = frozen_layer_name[ge_name]
        for name, child in self.surface_encoder.named_children():
            if name not in frozen_layers:
                continue
            for param in child.parameters():
                param.requires_grad = False

        for _name in conv_name[ge_name]:
            conv_layers = getattr(self.surface_encoder, _name)
            for i in range(nblock):
                for param in conv_layers[i].parameters():
                    param.requires_grad = False

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
