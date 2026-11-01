# Full-Spectrum Attention Network (FAN)

The repository provides training and evaluation pipelines for three FAN variants:

- **FAN-Base**: a data-driven baseline network.
- **FAN-SC**: the FAN-Base architecture with soft constraints.
- **FAN-HC**: the FAN-Base architecture with hard constraints.

Each variant supports Transformer and Mamba implementations.

## Repository Layout

```text
FAN/
  train.py              # Training entry point
  test.py               # Evaluation entry point
  engine.py             # Standard training and evaluation loops
  sc_engine.py          # Soft-constraint training and evaluation loops
  model/                # Model registry and FAN network definitions
  datasets/             # Dataset builders and preprocessing transforms
  util/                 # Metric logging, batching, and checkpoint helpers
  data/                 # Data directory
```

## Model Variants

Use `--model` to select a model:

```text
fan_base_transformer
fan_base_mamba
fan_sc_transformer
fan_sc_mamba
fan_hc_transformer
fan_hc_mamba
```

## Data

The `data/` directory is the default location for the FAN dataset. Place the training, validation, and test splits under this directory before running the scripts. See [data/README.md](data/README.md) for the high-level expected organization.

For FAN-SC, place the soft-constraint signature file at `data/sc_signature.npy` or pass a custom path with `--sc_signature`.

## Training

Install the base dependencies before running Transformer models:

```bash
pip install -r requirements.txt
```

For Mamba models, install the optional Linux CUDA dependency:

```bash
pip install -r requirements-mamba.txt
```

Example:

```bash
python train.py \
  --data_path ./data \
  --dataset_name fan \
  --model fan_base_transformer \
  --output_dir ./exps/fan_base_transformer \
  --batch_size 8 \
  --num_workers 10 \
  --lr 5e-5 \
  --lr_drop 200 \
  --epochs 200 \
  --device cuda:0
```

For FAN-SC:

```bash
python train.py \
  --data_path ./data \
  --dataset_name fan \
  --model fan_sc_transformer \
  --sc_signature ./data/sc_signature.npy \
  --output_dir ./exps/fan_sc_transformer \
  --batch_size 8 \
  --num_workers 10 \
  --lr 5e-5 \
  --lr_drop 200 \
  --epochs 200 \
  --device cuda:0
```

The training script saves fixed-interval checkpoints, the final checkpoint, and the best checkpoint selected on the validation split.

## Evaluation

Example:

```bash
python test.py \
  --data_path ./data \
  --dataset_name fan \
  --model fan_base_transformer \
  --resume ./exps/fan_base_transformer/checkpoint_best.pth \
  --batch_size 8 \
  --device cuda:0
```

## Notes

- Mamba models require `mamba_ssm`, which is typically installed in a Linux CUDA environment.


## License

This project is released under the [MIT License](LICENSE).
