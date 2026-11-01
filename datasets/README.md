# Datasets

This directory contains dataset construction and preprocessing utilities for FAN.

- `fan_dataset.py` builds the training, validation, and test datasets.
- `transforms.py` contains lightweight preprocessing transforms applied before batching.

The dataset implementation expects the data root passed through `--data_path`.
By default, `--data_path` points to the project-level `data/` directory.
