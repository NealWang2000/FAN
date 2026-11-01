# Model

This directory contains the FAN model registry and network definitions.

- `registry.py` builds the selected FAN variant from command line arguments.
- `fan_base_transformer.py` and `fan_base_mamba.py` define the FAN-Base backbones.
- `fan_hc_transformer.py` and `fan_hc_mamba.py` define the FAN-HC backbones.
- `common_blocks.py`, `swin_layers.py`, and `mamba_layers.py` contain shared building blocks used by the model variants.

FAN-SC uses the FAN-Base architecture and is selected through the training engine by choosing a `fan_sc_*` model name.
