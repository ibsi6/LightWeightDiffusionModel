import jax
import jax.numpy as jnp
from jax import random, jit, grad
import numpy as np
import matplotlib.pyplot as plt

class Sampler:
    """
    Handles the reverse diffusion (sampling) process.
    """

    def __init__(self, model, diffusion_process, key):
        """
        Initialize the sampler.

        Args:
            model (DiffusionModel): The trained diffusion model.
            diffusion_process (DiffusionProcess): The diffusion process.
            key (jax.random.PRNGKey): PRNG key.
        """
        self.model = model
        self.diffusion_process = diffusion_process
        self.key = key

    def sample(self, params, num_samples):
        key = self.key
        x = random.normal(key, shape=(num_samples, 2)).astype(jnp.float32)  # Ensure float32

        epsilon = 1e-8
        for t in reversed(range(self.diffusion_process.T)):
            t_batch = jnp.full((num_samples,), t)
            t_normalized = t_batch[:, None] / self.diffusion_process.T

            t_int = int(t)
            beta_t = self.diffusion_process.betas[t_int]
            alpha_t = self.diffusion_process.alphas[t_int]
            alpha_bar_t = self.diffusion_process.alphas_cumprod[t_int]
            sqrt_alpha_t = jnp.sqrt(alpha_t + epsilon)
            sqrt_one_minus_alpha_bar_t = jnp.sqrt(1 - alpha_bar_t + epsilon)
            sqrt_recip_alpha_t = 1 / sqrt_alpha_t

            # Predict noise
            pred_noise = self.model.forward(params, x, t_normalized)
            if jnp.isnan(pred_noise).any():
                print(f"NaNs detected in pred_noise at timestep {t}")
                break  # Exit the loop if NaNs are detected

            # Reverse diffusion step
            x = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise)

            # Add noise if t > 0
            if t > 0:
                key, subkey = random.split(key)
                noise = random.normal(subkey, shape=x.shape).astype(jnp.float32)
                x += jnp.sqrt(beta_t + epsilon) * noise

        return x