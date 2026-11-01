# Copyright (c) SITP
"""Training and evaluation routines for FAN."""

import math
import sys
from typing import Iterable

import numpy as np
import torch
from torch import nn

import util.misc as utils


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    args=None,
):
    """Train FAN for one epoch and return averaged loss statistics."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="\t")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    loss_fn = nn.L1Loss()

    for _, constraint_img, raw_img, gt_img, _, _ in metric_logger.log_every(data_loader, 10, header):
        constraint_img = constraint_img.to(device)
        raw_img = raw_img.to(device)
        gt_img = gt_img.to(device)

        predictions, _ = model(constraint_img, raw_img)
        targets = gt_img.tensors

        loss = loss_fn(predictions, targets)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate FAN and return per-split metric statistics."""
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="\t")
    header = "Test:"
    loss_fn = nn.L1Loss()

    rmse_all = []
    plume_rmse_all = []
    background_mean = []
    background_std = []
    plume_mean = []
    plume_std = []

    for _, constraint_img, raw_img, gt_img, mask_img, _ in metric_logger.log_every(data_loader, 10, header):
        constraint_img = constraint_img.to(device)
        raw_img = raw_img.to(device)
        gt_img = gt_img.to(device)
        mask_img = mask_img.to(device)

        predictions, _ = model(constraint_img, raw_img)
        gt_tensor = gt_img.tensors
        mask_tensor = mask_img.tensors

        l1_loss = loss_fn(predictions, gt_tensor)
        metric_logger.update(loss=l1_loss)

        for idx in range(gt_tensor.size(0)):
            pred = predictions[idx]
            gt = gt_tensor[idx]
            mask = mask_tensor[idx].bool()

            residual = pred - gt
            rmse = residual.pow(2).mean().sqrt()
            rmse_all.append(rmse.item())

            bg_mask = ~mask
            if bg_mask.any():
                bg_vals = pred[bg_mask]
                background_mean.append(bg_vals.mean().item())
                background_std.append(bg_vals.std().item())

            if mask.any():
                plume_vals = pred[mask]
                plume_residual = residual[mask]
                plume_rmse = plume_residual.pow(2).mean().sqrt()
                plume_rmse_all.append(plume_rmse.item())
                plume_mean.append(plume_vals.mean().item())
                plume_std.append(plume_vals.std().item())

    stats = {
        "rmse": float(np.mean(rmse_all)) if rmse_all else float("nan"),
        "plume_rmse": float(np.mean(plume_rmse_all)) if plume_rmse_all else float("nan"),
        "background_mean": float(np.mean(background_mean)) if background_mean else float("nan"),
        "background_std": float(np.mean(background_std)) if background_std else float("nan"),
        "plume_mean": float(np.mean(plume_mean)) if plume_mean else float("nan"),
        "plume_std": float(np.mean(plume_std)) if plume_std else float("nan"),
    }

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for name, value in stats.items():
        print(f"  {name}: {value:.6f}" if not np.isnan(value) else f"  {name}: nan")
    return stats
