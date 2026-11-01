# Copyright (c) SITP
"""Dataset loader for FAN methane retrieval."""

import json
from pathlib import Path

import torch
import torch.utils.data as tdata
import numpy as np

import datasets.transforms as T


class FANDataset(tdata.Dataset):
    """Load paired RGB, auxiliary, spectral, target, and mask tiles."""

    def __init__(self, img_folder, ann_file):
        """Create a FAN dataset split."""
        self.data_items = LoadItems(img_folder, ann_file)
        self._transform_rgb = make_fan_transform(img_type="rgb")
        self._transform_auxiliary = make_fan_transform(img_type="auxiliary")
        self._transform_raw = make_fan_transform(img_type="raw")
        self.prepare = ConvertFANTarget()
        self.target_keys = list(self.data_items.anns.keys())

    def __getitem__(self, idx):
        """Return one training or evaluation sample."""
        rgb_path = self.data_items.rgb_paths[idx]
        auxiliary_path = self.data_items.auxiliary_paths[idx]
        raw_path = self.data_items.raw_paths[idx]
        gt_path = self.data_items.gt_paths[idx]
        mask_path = self.data_items.mask_paths[idx]
        target = self.data_items.anns[self.target_keys[idx]]

        rgb_img, auxiliary_img, raw_img, gt_img, mask_img, target = self.prepare(
            rgb_path,
            auxiliary_path,
            raw_path,
            gt_path,
            mask_path,
            target,
        )

        if self._transform_rgb is not None:
            rgb_img, _ = self._transform_rgb(rgb_img)
            auxiliary_img, _ = self._transform_auxiliary(auxiliary_img)
            raw_img, _ = self._transform_raw(raw_img)
            gt_img = torch.as_tensor(gt_img)
            mask_img = torch.as_tensor(mask_img)

        return {
            "rgb": rgb_img,
            "auxiliary": auxiliary_img,
            "raw": raw_img,
            "gt": gt_img,
            "mask": mask_img,
            "target": target,
        }

    def __len__(self):
        """Return the number of annotated tiles."""
        return len(self.data_items.rgb_paths)


def make_fan_transform(img_type):
    """Build preprocessing transforms for each tile type."""
    if img_type == "rgb":
        return T.Compose([T.RemoveMaskedArea(mask_val=-9999), T.ToTensorArray()])
    if img_type == "auxiliary":
        return T.Compose([T.ToTensorArray()])
    if img_type == "raw":
        return T.Compose([T.RemoveMaskedArea(mask_val=-9999), T.ToTensorArray()])
    raise ValueError(f"Unsupported image type: {img_type}")


class LoadItems:
    """Parse annotations and resolve tile paths."""

    def __init__(self, img_folder, ann_file):
        """Load split metadata."""
        self.anns, self.imgs = {}, {}
        self.img_dir = Path(img_folder)

        if ann_file is not None:
            print("loading annotations into memory...")
            with Path(ann_file).open("r", encoding="utf-8") as f:
                dataset = json.load(f)
            if not isinstance(dataset, dict):
                raise TypeError(f"annotation file format {type(dataset)} is not supported")
            self.dataset = dataset
            self.create_list()
            self.create_paths()

    def create_list(self):
        """Create dictionaries keyed by patch name."""
        print("creating annotations list")
        anns, img_id, patch_id = {}, {}, {}
        unique_id = 1
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                patch_name = ann["patch_name"]
                anns[patch_name] = {"image_id": unique_id}
                img_id[patch_name] = ann["image_id"]
                patch_id[patch_name] = ann["patch_id"]
                unique_id += 1

        self.anns = anns
        self.img_id = img_id
        self.patch_id = patch_id

    def create_paths(self):
        """Resolve tile file paths for every annotated patch."""
        print("creating image paths list")
        rgb_paths, auxiliary_paths, raw_paths, gt_paths, mask_paths = [], [], [], [], []

        for ann_key in self.anns:
            image_id = self.img_id[ann_key]
            patch_id = self.patch_id[ann_key]
            matches = list((self.img_dir / "rgb_tiles").glob(f"{image_id}_*/*_{patch_id}.npy"))
            if not matches:
                raise FileNotFoundError(f"No RGB tile found for image_id={image_id}, patch_id={patch_id}")
            rgb_path = matches[0]
            scene_dir = rgb_path.parent.name
            tile_name = rgb_path.name

            auxiliary_root = self.img_dir / "auxiliary_tiles"
            if not auxiliary_root.exists():
                auxiliary_root = self.img_dir / ("m" + "f" + "_tiles")
            auxiliary_path = auxiliary_root / scene_dir / tile_name
            raw_path = self.img_dir / "rdata_tiles" / scene_dir / tile_name
            gt_path = self.img_dir / "gt_tiles" / scene_dir / tile_name
            mask_path = self.img_dir / "mask_tiles" / scene_dir / tile_name

            for path in (auxiliary_path, raw_path, gt_path, mask_path):
                if not path.exists():
                    raise FileNotFoundError(f"Expected tile does not exist: {path}")

            rgb_paths.append(rgb_path)
            auxiliary_paths.append(auxiliary_path)
            raw_paths.append(raw_path)
            gt_paths.append(gt_path)
            mask_paths.append(mask_path)

        if len(rgb_paths) != len(self.anns):
            raise ValueError("Number of annotations differs from the number of resolved image tiles")

        self.rgb_paths = rgb_paths
        self.auxiliary_paths = auxiliary_paths
        self.raw_paths = raw_paths
        self.gt_paths = gt_paths
        self.mask_paths = mask_paths


class ConvertFANTarget:
    """Load numpy tiles and convert sample metadata to tensor-ready targets."""

    def __init__(self):
        """Create the converter."""

    def __call__(self, rgb_path, auxiliary_path, raw_path, gt_path, mask_path, target):
        """Load one sample from disk."""
        rgb_img = np.load(rgb_path).transpose((2, 0, 1))
        auxiliary_img = np.expand_dims(np.load(auxiliary_path), axis=2).transpose((2, 0, 1))
        raw_img = np.load(raw_path).transpose((2, 0, 1))
        gt_img = np.expand_dims(np.load(gt_path), axis=2).transpose((2, 0, 1))
        mask_img = np.expand_dims(np.load(mask_path), axis=2).transpose((2, 0, 1))

        _, height, width = rgb_img.shape
        image_id = torch.tensor([target["image_id"]])

        target = {
            "image_id": image_id,
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
        }

        return rgb_img, auxiliary_img, raw_img, gt_img, mask_img, target


def build_fan_dataset(image_set, args):
    """Build the requested FAN dataset split."""
    root = Path(args.data_path)
    if not root.exists():
        raise FileNotFoundError(f"Provided FAN data path does not exist: {root}")

    paths = {
        "train": (root / "train", root / "annotations" / "train" / "gf_train.json"),
        "val": (root / "val", root / "annotations" / "val" / "gf_val.json"),
        "test": (root / "test", root / "annotations" / "test" / "gf_test.json"),
    }
    if image_set not in paths:
        raise ValueError(f"Unsupported image set: {image_set}")

    img_folder, ann_file = paths[image_set]
    return FANDataset(img_folder, ann_file)
