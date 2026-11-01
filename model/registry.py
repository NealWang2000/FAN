# Copyright (c) SITP
"""FAN model registry and wrapper modules."""

from importlib import import_module

from torch import nn

from util.misc import NestedTensor


FAN_MODEL_NAMES = (
    "fan_base_transformer",
    "fan_base_mamba",
    "fan_sc_transformer",
    "fan_sc_mamba",
    "fan_hc_transformer",
    "fan_hc_mamba",
)


class FANModel(nn.Module):
    """Adapter that exposes a unified forward API for all FAN model variants."""

    def __init__(self, backbone: nn.Module, uses_hc_branch: bool = False):
        """Create a FAN model.

        Args:
            backbone: Network that predicts the methane enhancement map.
            uses_hc_branch: Whether the model uses the hard-constraint branch.
        """
        super().__init__()
        self.backbone = backbone
        self.uses_hc_branch = uses_hc_branch

    def forward(self, auxiliary_samples: NestedTensor, spectral_samples: NestedTensor):
        """Predict methane enhancement from spectral inputs.

        Args:
            auxiliary_samples: Auxiliary single-channel tensor batch.
            spectral_samples: Main spectral tensor batch.

        Returns:
            Tuple containing the predicted map and attention weights.
        """
        auxiliary = auxiliary_samples.tensors
        spectra = spectral_samples.tensors
        if self.uses_hc_branch:
            return self.backbone(spectra, auxiliary)
        return self.backbone(spectra)


def build_model(args):
    """Build a FAN model from command line arguments."""
    model_name = getattr(args, "model", "fan_base_transformer")
    registry = {
        "fan_base_transformer": ("model.fan_base_transformer", "FANBaseTransformer", False),
        "fan_base_mamba": ("model.fan_base_mamba", "FANBaseMamba", False),
        "fan_sc_transformer": ("model.fan_base_transformer", "FANBaseTransformer", False),
        "fan_sc_mamba": ("model.fan_base_mamba", "FANBaseMamba", False),
        "fan_hc_transformer": ("model.fan_hc_transformer", "FANHCTransformer", True),
        "fan_hc_mamba": ("model.fan_hc_mamba", "FANHCMamba", True),
    }
    if model_name not in registry:
        available = ", ".join(FAN_MODEL_NAMES)
        raise ValueError(f"Unknown FAN model '{model_name}'. Available models: {available}")

    module_name, class_name, uses_hc_branch = registry[model_name]
    backbone_cls = getattr(import_module(module_name), class_name)
    return FANModel(backbone_cls(), uses_hc_branch=uses_hc_branch)


