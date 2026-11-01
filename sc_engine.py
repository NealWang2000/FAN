# Copyright (c) SITP
"""Soft-constraint training and evaluation routines for FAN-SC."""

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

import util.misc as utils


def _resolve_signature_path(args):
    """Resolve the spectral signature file used by the soft constraint."""
    candidates = []
    if getattr(args, "sc_signature", ""):
        candidates.append(Path(args.sc_signature))
    if getattr(args, "data_path", ""):
        candidates.append(Path(args.data_path) / "sc_signature.npy")
    candidates.append(Path("sc_signature.npy"))

    for path in candidates:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "FAN-SC requires a spectral signature file for the soft physical loss. "
        f"Provide --sc_signature or place one of: {searched}"
    )


def load_sc_signature(args, device):
    """Load the spectral signature vector for FAN-SC."""
    signature_path = _resolve_signature_path(args)
    signature = np.load(signature_path).astype("float32")
    signature = np.squeeze(signature)
    if signature.ndim != 1:
        raise ValueError(f"Soft-constraint signature must be one-dimensional: {signature_path}")
    return torch.from_numpy(signature).to(device)


def _align_spectra_and_signature(spectra, signature):
    """Align spectral tensors to the signature length."""
    channels = min(spectra.shape[1], signature.numel())
    return spectra[:, :channels, :, :], signature[:channels]


def _matrix_pinv(matrix):
    """Compute a pseudo-inverse across PyTorch 1.x versions."""
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "pinv"):
        return torch.linalg.pinv(matrix)
    return torch.pinverse(matrix)


def soft_constraint_loss(predictions, targets, spectra, mask, signature, block_width=4, eps=1e-4):
    """Compute the FAN-SC physical consistency term."""
    spectra, signature = _align_spectra_and_signature(spectra, signature)
    batch_size, channels, height, width = spectra.shape
    block_width = max(1, min(int(block_width), width))
    block_count = width // block_width
    if block_count == 0:
        block_count = 1
        block_width = width

    signature = signature.view(1, 1, channels)
    physical_loss = spectra.new_tensor(0.0)

    for block_idx in range(block_count):
        start = block_idx * block_width
        end = width if block_idx == block_count - 1 else start + block_width
        spectra_block = spectra[:, :, :, start:end].permute(0, 2, 3, 1).contiguous()
        target_block = targets[:, :, :, start:end].contiguous()
        pred_block = predictions[:, :, :, start:end].contiguous()
        mask_block = mask[:, :, :, start:end].contiguous()

        pixel_count = spectra_block.shape[1] * spectra_block.shape[2]
        spectra_flat = spectra_block.view(batch_size, pixel_count, channels)
        target_flat = target_block.view(batch_size, pixel_count)
        pred_flat = pred_block.view(batch_size, pixel_count)
        mask_flat = mask_block.view(batch_size, pixel_count).to(torch.bool)

        background_mu = []
        for batch_idx in range(batch_size):
            background = ~mask_flat[batch_idx]
            if not torch.any(background):
                background = torch.ones_like(background, dtype=torch.bool)
            background_mu.append(torch.mean(spectra_flat[batch_idx][background], dim=0))

        mu = torch.stack(background_mu, dim=0).view(batch_size, 1, channels)
        scaled_mu = mu * signature

        target_residual = spectra_flat - target_flat.view(batch_size, pixel_count, 1) * scaled_mu - mu
        covariance = torch.bmm(target_residual, target_residual.transpose(1, 2))
        identity = torch.eye(pixel_count, device=spectra.device, dtype=spectra.dtype).unsqueeze(0)
        covariance_inv = _matrix_pinv(covariance + eps * identity)

        pred_residual = spectra_flat - pred_flat.view(batch_size, pixel_count, 1) * scaled_mu - mu
        physical = torch.bmm(torch.bmm(pred_residual.transpose(1, 2), covariance_inv), pred_residual)
        physical_loss = physical_loss + torch.mean(torch.abs(physical))

    return physical_loss


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    args=None,
):
    """Train FAN-SC for one epoch and return averaged loss statistics."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="\t")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    data_loss_fn = nn.L1Loss()
    signature = load_sc_signature(args=args, device=device)

    for _, auxiliary_img, raw_img, gt_img, mask_img, _ in metric_logger.log_every(data_loader, 10, header):
        auxiliary_img = auxiliary_img.to(device)
        raw_img = raw_img.to(device)
        gt_img = gt_img.to(device)
        mask_img = mask_img.to(device)

        predictions, _ = model(auxiliary_img, raw_img)
        targets = gt_img.tensors
        spectra = raw_img.tensors
        mask = mask_img.tensors

        data_loss = data_loss_fn(predictions, targets)
        physical_loss = soft_constraint_loss(
            predictions,
            targets,
            spectra,
            mask,
            signature,
            block_width=args.sc_block_width,
        )
        physical_weight = args.sc_physical_weight
        loss = data_loss + physical_weight * physical_loss

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss,
            data_loss=data_loss,
            physical_loss=physical_weight * physical_loss,
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    """Evaluate FAN-SC and optionally save per-sample predictions."""
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="\t")
    header = "Test:"
    data_loss_fn = nn.L1Loss()
    signature = load_sc_signature(args=args, device=device)

    rmse_all = []
    plume_rmse_all = []
    background_mean = []
    background_std = []
    plume_mean = []
    plume_std = []

    for _, auxiliary_img, raw_img, gt_img, mask_img, _ in metric_logger.log_every(data_loader, 10, header):
        auxiliary_img = auxiliary_img.to(device)
        raw_img = raw_img.to(device)
        gt_img = gt_img.to(device)
        mask_img = mask_img.to(device)

        predictions, _ = model(auxiliary_img, raw_img)
        targets_tensor = gt_img.tensors
        spectra = raw_img.tensors
        mask_tensor = mask_img.tensors

        data_loss = data_loss_fn(predictions, targets_tensor)
        physical_loss = soft_constraint_loss(
            predictions,
            targets_tensor,
            spectra,
            mask_tensor,
            signature,
            block_width=args.sc_block_width,
        )
        metric_logger.update(
            loss=data_loss + args.sc_physical_weight * physical_loss,
            data_loss=data_loss,
            physical_loss=args.sc_physical_weight * physical_loss,
        )

        for idx in range(targets_tensor.size(0)):
            pred = predictions[idx]
            gt = targets_tensor[idx]
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
