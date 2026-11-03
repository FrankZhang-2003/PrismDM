# Efficiency

Use the Python 3.10 environment from the [README](../README.md#installation). From the repository root:

```bash
pip install -r requirements-mpc.txt
```

## Start the parties

Run in a separate terminal:

```bash
cd Efficiency/PrismDM
python nodectl.py -c 3pc.json up
```

The configuration uses ABY3 with `FM64` and three local parties. Both implementations share the same ports, so run one set of services at a time. Stop them with Ctrl-C.

## PrismDM

From `Efficiency/PrismDM`:

```bash
PYTHONPATH=. python scripts/flax_sample_silu.py --backend prismdm --sampler ddim --ddim-steps 50
PYTHONPATH=. python scripts/flax_sample_silu.py --backend vanilla --sampler ddim --ddim-steps 50
```

Use `--backend cpu` for local inference or `--ddim-steps 2` for a quick check.

## CipherDM

From `Efficiency/Comparsion-CipherDM`, with the parties running:

```bash
PYTHONPATH=. python scripts/flax_sample_hack.py --backend cipherdm --sampler ddim --ddim-steps 50
```

This script enables the SiLU approximation and leaves the Softmax patch disabled. The accuracy scripts use their own operator configurations.

## Settings

- `--model_path`, `--save_dir` and `--config` override the default paths.
- Each run evaluates one class-1 sample; `--num_samples` does not change the batch size.
- Runtime includes any compilation during the call. SPU profiling is enabled in `3pc.json`.

## Docker

Build and start from the repository root:

```bash
docker build -t prismdm:local .
docker run --rm -it --name prismdm prismdm:local
```

Inside the container:

```bash
cd /workspace/PrismDM/Efficiency/PrismDM
python nodectl.py -c 3pc.json up
```

In a second host terminal:

```bash
docker exec -it prismdm bash
cd /workspace/PrismDM/Efficiency/PrismDM
PYTHONPATH=. python scripts/flax_sample_silu.py --backend prismdm --sampler ddim --ddim-steps 50
```

## LAN emulation

Start the container with network administration enabled:

```bash
docker run --rm -it --cap-add=NET_ADMIN --name prismdm prismdm:local
```

From `Efficiency/PrismDM` inside the container:

```bash
bash simulate_lan.sh lan
# Run the experiment.
bash simulate_lan.sh del
```

This sets loopback traffic to 1 Gbit/s with 0.05 ms one-way delay. Run it only inside the container, without `--network=host`, since it changes the loopback qdisc.
