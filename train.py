# Copyright (c) SITP
"""Training entry point for FAN."""

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_dataset
from model import FAN_MODEL_NAMES, build_model

torch.multiprocessing.set_sharing_strategy("file_system")


def get_args_parser():
    """Build the command line parser used for FAN training."""
    parser = argparse.ArgumentParser("FAN training", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr_drop", default=150, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float)
    parser.add_argument(
        "--model",
        default="fan_base_transformer",
        choices=FAN_MODEL_NAMES,
        help="FAN model variant to train.",
    )
    parser.add_argument("--sc_signature", default="", type=str, help="Spectral signature file for FAN-SC.")
    parser.add_argument("--sc_physical_weight", default=1e-19, type=float, help="FAN-SC physical loss weight.")
    parser.add_argument("--sc_block_width", default=4, type=int, help="FAN-SC block width for covariance estimation.")
    parser.add_argument("--dataset_name", default="fan")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", default="cuda", help="Device used for training and evaluation.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="Checkpoint path used to resume model weights.")
    parser.add_argument("--output_dir", default="", help="Directory used for checkpoints and metric logs.")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    return parser


def load_checkpoint(model, checkpoint_path, device):
    """Load matching model weights from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {len(unexpected)}")
    return checkpoint


def get_engine(args):
    """Return the training/evaluation engine for the selected FAN method."""
    if args.model.startswith("fan_sc_"):
        import sc_engine as engine
    else:
        import engine
    return engine


def main(args):
    """Run the FAN training loop."""
    print(utils.get_sha())
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(args)
    model.to(device)
    model_without_ddp = model
    engine = get_engine(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.lr_drop, gamma=0.5)

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fan_data,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fan_data,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.resume:
        checkpoint = load_checkpoint(model_without_ddp, args.resume, device="cpu")
        if all(k in checkpoint for k in ("optimizer", "lr_scheduler", "epoch")):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    metric_history = []
    checkpoint_interval = 50
    best_val_rmse = float("inf")

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = engine.train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            args=args,
        )
        lr_scheduler.step()

        if args.model.startswith("fan_sc_"):
            val_stats = engine.evaluate(model, data_loader_val, device, args)
        else:
            val_stats = engine.evaluate(model, data_loader_val, device)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        metric_history.append(log_stats)

        if output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            epoch_number = epoch + 1
            is_final_epoch = epoch_number == args.epochs
            if is_final_epoch:
                utils.save_on_master(checkpoint, output_dir / "checkpoint_final.pth")
            elif epoch_number % checkpoint_interval == 0:
                utils.save_on_master(checkpoint, output_dir / f"checkpoint_epoch_{epoch_number:04d}.pth")
            current_val_rmse = val_stats.get("rmse", float("inf"))
            if current_val_rmse < best_val_rmse:
                best_val_rmse = current_val_rmse
                utils.save_on_master(checkpoint, output_dir / "checkpoint_best.pth")
            if utils.is_main_process():
                with (output_dir / "log.txt").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    if output_dir and metric_history:
        np.save(output_dir / "metrics.npy", metric_history)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FAN training script", parents=[get_args_parser()])
    parsed_args = parser.parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
