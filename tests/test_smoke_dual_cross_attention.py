import numpy as np
import pytest
import torch
from torch import nn


pytest.importorskip("torch_geometric")
pytest.importorskip("torch_scatter")

from torch_geometric.data import Batch, Data

from models.adsmt_arch import AdsMT_ARCH, _build_token_batch
from utils.site_accuracy import calc_accuracy


class DummySurfaceEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(3, hidden_dim)

    def forward(self, data):
        return self.proj(data.pos)


class DummyAdsEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(3, hidden_dim)

    def forward(self, data):
        ads_batch = _build_token_batch(data, "ads_pos")
        return self.proj(data.ads_pos), ads_batch


def _build_single_graph_sample(n_surface_atoms=8, ads_atom_numbs=None):
    if ads_atom_numbs is None:
        ads_atom_numbs = [6, 6, 1, 1, 8]
    n_ads_atoms = len(ads_atom_numbs)

    return Data(
        pos=torch.randn(n_surface_atoms, 3),
        cell=torch.eye(3).view(1, 3, 3),
        atomic_numbers=torch.randint(1, 20, (n_surface_atoms,), dtype=torch.long),
        ads_pos=torch.randn(n_ads_atoms, 3),
        ads_atom_numb=torch.tensor(ads_atom_numbs, dtype=torch.long),
        # Keep some active (non-zero) tags so site scoring is valid.
        gmae_tags=torch.tensor(
            [1] * min(5, n_surface_atoms) + [0] * max(0, n_surface_atoms - 5),
            dtype=torch.long,
        ),
        site=torch.tensor([0, 2], dtype=torch.long),
        y_relaxed=-1.0,
        natoms=n_surface_atoms,
    )


def test_smoke_forward_cross_weights_and_site_accuracy():
    hidden_dim = 32
    model = AdsMT_ARCH(
        num_atoms=0,
        bond_feat_dim=0,
        num_targets=1,
        graph_encoder="adsgt",
        graph_encoder_args={
            "node_features": hidden_dim,
            "edge_features": hidden_dim,
            "conv_layers": 1,
            "node_layer_head": 4,
            "cutoff": 6.0,
            "max_neighbors": 8,
        },
        desc_layers=2,
        desc_hidden_dim=hidden_dim,
        cross_modal_args={
            "vec_emb_dim": hidden_dim,
            "node_emb_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "out_channels": 1,
            "num_gaussians": 16,
            "num_heads": 4,
            "attn_layers": 1,
            "mlp_layers": 2,
            "dropout": 0.0,
            "act": "silu",
        },
    )

    # Replace heavy encoders so this test focuses on fusion/output wiring.
    model.surface_encoder = DummySurfaceEncoder(hidden_dim)
    model.ads_encoder = DummyAdsEncoder(hidden_dim)
    model.eval()

    batch = Batch.from_data_list([_build_single_graph_sample()])

    with torch.no_grad():
        energy, cross_weights, _ = model(batch, need_weights=True)

    assert energy.shape == (1,)
    assert len(cross_weights) == 1
    assert len(cross_weights[0]) == 1
    assert cross_weights[0][0].ndim == 1
    assert cross_weights[0][0].numel() == batch.pos.size(0)

    sample = batch.to_data_list()[0]
    sample.cross_weights = cross_weights[-1][0]
    site_acc = calc_accuracy([sample])

    assert isinstance(site_acc, (float, np.floating))
    assert 0.0 <= float(site_acc) <= 1.0


def test_smoke_two_graph_batch_token_mapping_and_outputs():
    hidden_dim = 32
    model = AdsMT_ARCH(
        num_atoms=0,
        bond_feat_dim=0,
        num_targets=1,
        graph_encoder="adsgt",
        graph_encoder_args={
            "node_features": hidden_dim,
            "edge_features": hidden_dim,
            "conv_layers": 1,
            "node_layer_head": 4,
            "cutoff": 6.0,
            "max_neighbors": 8,
        },
        desc_layers=2,
        desc_hidden_dim=hidden_dim,
        cross_modal_args={
            "vec_emb_dim": hidden_dim,
            "node_emb_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "out_channels": 1,
            "num_gaussians": 16,
            "num_heads": 4,
            "attn_layers": 1,
            "mlp_layers": 2,
            "dropout": 0.0,
            "act": "silu",
        },
    )

    model.surface_encoder = DummySurfaceEncoder(hidden_dim)
    model.ads_encoder = DummyAdsEncoder(hidden_dim)
    model.eval()

    sample_1 = _build_single_graph_sample(
        n_surface_atoms=8, ads_atom_numbs=[6, 1, 1, 8]
    )
    sample_2 = _build_single_graph_sample(
        n_surface_atoms=10, ads_atom_numbs=[6, 6, 1, 1, 1, 8]
    )
    batch = Batch.from_data_list([sample_1, sample_2])

    # Verify that ads tokens are mapped back to their graph id correctly.
    ads_batch = _build_token_batch(batch, "ads_pos")
    expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    assert torch.equal(ads_batch.cpu(), expected)

    with torch.no_grad():
        energy, cross_weights, _ = model(batch, need_weights=True)

    assert energy.shape == (2,)
    assert len(cross_weights) == 1
    assert len(cross_weights[0]) == 2
    assert cross_weights[0][0].shape == (sample_1.pos.size(0),)
    assert cross_weights[0][1].shape == (sample_2.pos.size(0),)
