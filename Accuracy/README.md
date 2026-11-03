# Accuracy

For sampling, follow the [README](../README.md#accuracy). Keep the seed and sampling steps the same when comparing methods.

## Checkpoints

Paths below are relative to `Accuracy/`:

| Model | Checkpoint |
|---|---|
| Vanilla / CipherDM | `Acceleration-Comparison-CipherDM-VanillaModel/checkpoints/Silu.msgpack` |
| PrismDM | `Acceleration-PrismDM-AlignedModel/checkpoints/FA-CTA-PlanA-silu-epoch-50-model.msgpack` |

The `.msgpack` files use Python pickle serialization; only load trusted checkpoints.

## Dataset generation

Install the additional dependencies from the repository root:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dataset.txt
```

Run either command block from the repository root.

CipherDM:

```bash
cd Accuracy/Acceleration-Comparison-CipherDM-VanillaModel
PYTHONPATH=. python scripts/flax_fid_prepare_generated_method_SettingB_dataset.py
```

PrismDM:

```bash
cd Accuracy/Acceleration-PrismDM-AlignedModel
PYTHONPATH=. python scripts/flax_fid_prepare_generated_method_SettingE_dataset.py
```

These scripts download MNIST and select digits 0 and 1. Generated images go to `FID_dataset/` in each experiment directory.

Checkpoint paths, operators and the 50-step DDIM setting are configured in each script's `__main__` block. For configurable sampling, use `scripts/sample.py` from the repository root.
