#!/usr/bin/env python3
"""Sample the supplied MNIST-01 model locally, without downloading MNIST."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "vanilla": ("Acceleration-Comparison-CipherDM-VanillaModel", "Silu.msgpack", "CPU"),
    "cipherdm": ("Acceleration-Comparison-CipherDM-VanillaModel", "Silu.msgpack", "CDM"),
    "prismdm": ("Acceleration-PrismDM-AlignedModel", "FA-CTA-PlanA-silu-epoch-50-model.msgpack", "CTA"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=CASES, default="prismdm")
    parser.add_argument("--num-samples", type=int, default=4, help="Samples per class (0 and 1).")
    parser.add_argument("--steps", type=int, default=50, choices=[2, 4, 5, 10, 20, 25, 50, 100])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if args.num_samples < 1 or args.seed < 0:
        parser.error("--num-samples must be positive and --seed must be nonnegative")
    # Local sampling defaults to CPU, but an explicit user setting takes precedence.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    folder, checkpoint_name, method = CASES[args.method]
    experiment = ROOT / "Accuracy" / folder
    sys.path.insert(0, str(experiment))

    import flax
    from flax import linen as nn
    from flax.training import train_state
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from PIL import Image
    from flax_ddpm import approximation_methods as approximations
    from flax_ddpm.mish_function import mish
    from flax_ddpm.script_utils import get_args, get_diffusion_from_args

    # Obtain the artifact's architecture defaults without passing launcher flags.
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        model_args = get_args()
    finally:
        sys.argv = saved_argv
    functions = {"nn.silu": nn.silu, "nn.softmax": nn.softmax, "mish": mish}
    functions.update({name: getattr(approximations, name) for name in dir(approximations)
                      if name.startswith("approximate_")})
    config = experiment / "unlinear_dicts_silu" / f"unlinear_dict_{method}.json"
    operators = json.loads(config.read_text())
    model_args.unlinear_function_dict = {method: {k: functions[v] for k, v in operators.items()}}
    diffusion = get_diffusion_from_args(model_args, method)
    key = jax.random.PRNGKey(0)
    variables = diffusion.init(rngs=key, rng=key,
                              x=jax.random.normal(key, (1, 28, 28, 1)),
                              y=jnp.ones(1, dtype=jnp.int32))
    target = train_state.TrainState.create(apply_fn=diffusion.apply, params=variables["params"],
                                          tx=optax.adam(model_args.learning_rate))
    checkpoint = experiment / "checkpoints" / checkpoint_name
    # The supplied .msgpack files contain pickled Flax state dictionaries.
    with checkpoint.open("rb") as stream:
        state = flax.serialization.from_state_dict(target, pickle.load(stream))

    # Keep the original DDIM implementation and 1,000-step training schedule.
    # Only the number of reverse sampling steps is configurable here.
    def sample(key, label):
        return diffusion.apply({"params": state.params}, batch_size=1, y=label,
                               use_ema=False, rng_key=key, ddim_timesteps=args.steps,
                               method=diffusion.sample_ddim)

    destination = args.output.resolve() / args.method
    destination.mkdir(parents=True, exist_ok=True)
    grid = Image.new("L", (28 * args.num_samples, 56))
    start = time.perf_counter()
    for label in (0, 1):
        class_dir = destination / f"class_{label}"
        class_dir.mkdir(exist_ok=True)
        for index in range(args.num_samples):
            seed = args.seed + index
            values = np.asarray(sample(jax.random.PRNGKey(seed), jnp.array([label], dtype=jnp.int32)))[0, :, :, 0]
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite samples: method={args.method}, class={label}, seed={seed}")
            # Match the artifact's per-image min/max visualization, with a constant-image guard.
            span = float(values.max() - values.min())
            pixels = np.zeros_like(values, dtype=np.uint8) if span == 0 else (
                (values - values.min()) * (255.0 / span)).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            image.save(class_dir / f"seed_{seed:06d}.png")
            grid.paste(image, (index * 28, label * 28))
            print(f"{args.method}: class={label}, seed={seed}", flush=True)
    grid.save(destination / "grid.png")
    elapsed = time.perf_counter() - start
    manifest = {"method": args.method, "checkpoint": str(checkpoint.relative_to(ROOT)),
                "operator_config": str(config.relative_to(ROOT)), "ddim_steps": args.steps,
                "training_timesteps": 1000, "samples_per_class": args.num_samples,
                "classes": [0, 1], "seed": args.seed, "elapsed_seconds": elapsed,
                "execution": "plaintext", "jax_version": jax.__version__,
                "flax_version": flax.__version__}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {2 * args.num_samples} images and grid.png to {destination}")


if __name__ == "__main__":
    main()
