# Copyright (c) SITP
"""Evaluation entry point for FAN."""

import argparse

import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_dataset
from train import get_engine, load_checkpoint
from model import FAN_MODEL_NAMES, build_model


def get_args_parser():
    """Build the command line parser used for FAN testing."""
    parser = argparse.ArgumentParser("FAN testing", add_help=False)
    parser.add_argument(
        "--model",
        default="fan_base_transformer",
        choices=FAN_MODEL_NAMES,
        help="FAN model variant to test.",
    )
    parser.add_argument("--sc_signature", default="", type=str, help="Spectral signature file for FAN-SC.")
    parser.add_argument("--sc_physical_weight", default=1e-19, type=float, help="FAN-SC physical loss weight.")
    parser.add_argument("--sc_block_width", default=4, type=int, help="FAN-SC block width for covariance estimation.")
    parser.add_argument("--dataset_name", default="fan")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", default="cuda", help="Device used for testing.")
    parser.add_argument("--resume", required=True, help="Checkpoint path used to load model weights.")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    return parser


def main(args):
    """Run FAN evaluation from a trained checkpoint."""
    device = torch.device(args.device)

    model = build_model(args)
    model.to(device)
    load_checkpoint(model, args.resume, device="cpu")
    engine = get_engine(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    dataset_test = build_dataset(image_set="test", args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(
        dataset_test,
        args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=utils.collate_fan_data,
        num_workers=args.num_workers,
    )

    print("Start evaluation")
    if args.model.startswith("fan_sc_"):
        stats = engine.evaluate(model, data_loader_test, device, args)
    else:
        stats = engine.evaluate(model, data_loader_test, device)
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FAN testing script", parents=[get_args_parser()])
    parsed_args = parser.parse_args()
    main(parsed_args)
