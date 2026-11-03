"""Regression checks for the supplementary Eq. (2) exponential and softmax."""
import importlib.util
from pathlib import Path
import unittest

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ApproximationTests(unittest.TestCase):
    def test_exponential_and_softmax(self):
        paths = [*ROOT.glob("Accuracy/*/flax_ddpm/approximation_methods.py"),
                 ROOT / "Efficiency/PrismDM/flax_ddpm/approximation_methods.py"]
        for index, path in enumerate(paths):
            with self.subTest(path=str(path.relative_to(ROOT))):
                spec = importlib.util.spec_from_file_location(f"approximations_{index}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                suffix = "PDM" if "Efficiency" in path.parts else "CTA"
                exponential = getattr(module, f"chexp_{suffix}")
                softmax = getattr(module, f"approximate_softmax_{suffix}")
                # Both branches must approximate exp on their respective intervals.
                x = jnp.array([-4.0, -1.0, 0.0])
                np.testing.assert_allclose(exponential(x), np.exp(np.asarray(x)), atol=0.03, rtol=0)
                logits = jnp.array([[0.0, -1.0, -4.0], [2.0, 1.0, 2.0]])
                values = np.asarray(softmax(logits))
                self.assertTrue(np.isfinite(values).all())
                np.testing.assert_allclose(values.sum(axis=-1), 1.0, atol=1e-6)
                np.testing.assert_allclose(softmax(logits + 17), values, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
