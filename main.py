import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax

import jax.numpy as jnp
from jax import random, jit, grad
import numpy as np
import matplotlib.pyplot as plt

from DiffusionModel import DiffusionModel
from DiffusionProcess import DiffusionProcess 
from Trainer import Trainer
from Sampler import Sampler

def main():
    # Initialize PRNG key
    key = random.PRNGKey(0)

    # Generate synthetic 2D Gaussian data
    def get_data(num_samples=10000):
        mean = jnp.array([0.0, 0.0])
        cov = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        data = np.random.multivariate_normal(mean, cov, num_samples)
        return data

    data = get_data()

    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title('Input Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

    T = 1000  # Number of diffusion steps
    diffusion_process = DiffusionProcess(T)

    # Initialize model
    input_dim = 2  # Dimensionality of the data
    hidden_dim = 128
    output_dim = 2
    key, subkey = random.split(key)
    model = DiffusionModel(input_dim, hidden_dim, output_dim, subkey)

    # Initialize trainer
    batch_size = 128
    trainer = Trainer(model, diffusion_process, data, key, batch_size=batch_size)

    # Train the model
    num_epochs = 10
    trainer.train(num_epochs)

    # Initialize sampler
    key, subkey = random.split(trainer.key)
    sampler = Sampler(model, diffusion_process, subkey)

    # Generate samples
    num_samples = 1000
    samples = sampler.sample(trainer.opt_state['params'], num_samples)
    samples = np.array(samples)
    print("Sampling completed.")
    print("Sample shape:", samples.shape)
    print("Sample mean:", np.mean(samples, axis=0))
    print("Sample std:", np.std(samples, axis=0))
    print("Any NaNs in samples:", np.isnan(samples).any())
    print("Any Infs in samples:", np.isinf(samples).any())

    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title('Generated Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Input Data')
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Generated Data')
    plt.title('Input vs. Generated Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
