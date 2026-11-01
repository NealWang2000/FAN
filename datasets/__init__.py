# Copyright (c) SITP
"""Dataset builders for FAN."""

import torch.utils.data


def build_dataset(image_set, args):
    """Build a dataset split from command line arguments."""
    if args.dataset_name == "fan":
        from .fan_dataset import build_fan_dataset

        return build_fan_dataset(image_set, args)
    raise ValueError(f"dataset {args.dataset_name} is not supported")
