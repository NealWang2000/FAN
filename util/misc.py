# Copyright (c) SITP
"""Utility functions used by FAN training and evaluation."""

import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict, deque
from typing import List, Optional

import torch
import torch.distributed as dist
import torchvision
from packaging import version
from torch import Tensor

if version.parse(torchvision.__version__) < version.parse("0.7"):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue:
    """Track scalar values and expose smoothed statistics."""

    def __init__(self, window_size=20, fmt=None):
        """Create a smoothed value tracker."""
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        """Add a value to the tracker."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Synchronize count and total across distributed workers."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        self.count = int(t[0].item())
        self.total = t[1].item()

    @property
    def median(self):
        """Return the median over the moving window."""
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        """Return the mean over the moving window."""
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        """Return the average over all observed values."""
        return self.total / self.count if self.count else 0.0

    @property
    def max(self):
        """Return the maximum over the moving window."""
        return max(self.deque)

    @property
    def value(self):
        """Return the latest observed value."""
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """Print and synchronize training metrics."""

    def __init__(self, delimiter="\t"):
        """Create a metric logger."""
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Update one or more named metrics."""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.meters[key].update(value)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = [f"{name}: {meter}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronize every tracked meter."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Register a preconfigured meter."""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Yield from an iterable while periodically printing timing and metrics."""
        i = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        mb = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / mb,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def get_sha():
    """Return a compact git revision string."""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        diff = "has uncommitted changes" if _run(["git", "diff-index", "HEAD"]) else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    return f"sha: {sha}, status: {diff}, branch: {branch}"



def collate_fan_data(batch):
    """Collate FAN samples into nested tensors."""
    rgb_data, auxiliary_data, raw_data, gt_data, mask_data, targets = [], [], [], [], [], []
    for data in batch:
        rgb_data.append(data["rgb"])
        auxiliary_data.append(data["auxiliary"])
        raw_data.append(data["raw"])
        gt_data.append(data["gt"])
        mask_data.append(data["mask"])
        targets.append(data["target"])

    return (
        nested_tensor_from_tensor_list(rgb_data),
        nested_tensor_from_tensor_list(auxiliary_data),
        nested_tensor_from_tensor_list(raw_data),
        nested_tensor_from_tensor_list(gt_data),
        nested_tensor_from_tensor_list(mask_data),
        targets,
    )


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    """Tensor batch paired with a padding mask."""

    def __init__(self, tensors, mask: Optional[Tensor]):
        """Create a nested tensor."""
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        """Move tensors and mask to a device."""
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        """Return the tensor and mask."""
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """Pad a list of CxHxW tensors into a batch."""
    if tensor_list[0].ndim != 3:
        raise ValueError("Only 3D tensors are supported")
    if torchvision._is_tracing():
        return _onnx_nested_tensor_from_tensor_list(tensor_list)

    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    batch, _, height, width = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((batch, height, width), dtype=torch.bool, device=device)
    for img, pad_img, sample_mask in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        sample_mask[: img.shape[1], : img.shape[2]] = False
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    """ONNX-compatible nested tensor construction."""
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        mask = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(mask, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    return NestedTensor(torch.stack(padded_imgs), mask=torch.stack(padded_masks))


def setup_for_distributed(is_master):
    """Disable printing in non-master processes."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """Return whether torch.distributed is ready."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Return distributed rank."""
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    """Return whether this process is rank zero."""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Call torch.save only on the main process."""
    if is_main_process():
        torch.save(*args, **kwargs)


