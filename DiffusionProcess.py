import jax
import jax.numpy as jnp
from jax import random, jit, grad
import numpy as np
import matplotlib.pyplot as plt

class DiffusionProcess:

    def __init__(self, T, beta_start=1e-4, beta_end=0.02):
        """
        Initialize the diffusion process.

        Args:
            T (int): Number of diffusion steps.
            beta_start (float): Starting beta value.
            beta_end (float): Ending beta value.
        """
        self.T = T
        self.betas = jnp.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion (adding noise) at timestep t.

        Args:
            x0 (jnp.array): Original data.
            t (jnp.array): Time steps.
            noise (jnp.array): Noise to add.

        Returns:
            jnp.array: Noisy data at time t.
        """
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha_cumprod_t[:, None] * x0 + sqrt_one_minus_alpha_cumprod_t[:, None] * noise
