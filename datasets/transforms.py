# Copyright (c) SITP
"""Array transforms used by the FAN dataset."""

import numpy as np
import torch


class ToTensorArray:
    """Convert a NumPy array to a float tensor."""

    def __call__(self, image, target=None):
        return torch.from_numpy(image.astype("float32")), target


class RemoveMaskedArea:
    """Replace invalid masked pixels with zeros."""

    def __init__(self, mask_val=-9999):
        self.mask_val = mask_val

    def __call__(self, image, target=None):
        image = np.array(image, copy=True)
        image[image <= self.mask_val] = 0.0
        return image, target


class Compose:
    """Apply transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self):
        transform_lines = [f"  {transform}" for transform in self.transforms]
        return f"{self.__class__.__name__}(\n" + "\n".join(transform_lines) + "\n)"
