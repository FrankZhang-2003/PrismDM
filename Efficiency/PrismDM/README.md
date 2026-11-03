# PrismDM Secure Inference Example

See [the shared efficiency guide](../README.md) for dependencies, node services, Docker and optional LAN emulation. Build the Dockerfile at the repository root.

With the Python environment activated and the node services running:

```bash
PYTHONPATH=. python scripts/flax_sample_silu.py --backend prismdm --sampler ddim --ddim-steps 50
```

Use `--backend cpu --sampler ddim --ddim-steps 2` for a local execution check without nodes.
