# Annotations

This directory stores annotation files for the training, validation, and test splits.

Expected high-level layout:

```text
annotations/
  train/
  val/
  test/
```

The annotation files provide the sample index used by the FAN dataset loader.
Each record should contain the fields required to identify one sample in the corresponding split.
Keep the annotation split consistent with the corresponding data split.
