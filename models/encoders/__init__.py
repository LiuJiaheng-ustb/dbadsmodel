from importlib import import_module


def _optional_alias(module_path, class_name):
    try:
        module = import_module(module_path)
        return getattr(module, class_name)
    except ModuleNotFoundError:
        return None


schnet = _optional_alias("models.encoders.schnet", "SchNetEncoder")
cgcnn = _optional_alias("models.encoders.cgcnn", "CGCNNEncoder")
dpp = _optional_alias("models.encoders.dpp", "DppEncoder")
gemnet_oc = _optional_alias("models.encoders.gemnet_oc", "GemNetOCEncoder")
escn = _optional_alias("models.encoders.escn", "eSCNEncoder")
et = _optional_alias("models.encoders.et", "ETEncoder")
adsgt = _optional_alias("models.encoders.adsgt", "AdsGT")
