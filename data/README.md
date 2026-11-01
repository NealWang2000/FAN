# Data

This directory is the default location for FAN dataset files used by training, validation, and testing.

Expected layout:

```text
data/
  sc_signature.npy
  train/
  val/
  test/
  annotations/
```

- `train/`: training data split.
- `val/`: validation data split.
- `test/`: held-out test data split used by `test.py`.
- `annotations/`: split annotation files consumed by the dataset loader.
- `sc_signature.npy`: optional spectral signature used by FAN-SC.

Keep the internal organization of each split consistent with the dataset loader.
