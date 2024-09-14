import jax
import jax.numpy as jnp
from jax import random, jit, grad
import numpy as np
import matplotlib.pyplot as plt

class DiffusionModel:

    def __init__(self, input_dim, hidden_dim, output_dim, key):
        """
        Args:
            input_dim (int): Dimension of input data.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of output data.
            key (jax.random.PRNGKey): PRNG key for parameter initialization.
        """
        self.params = self.init_params(input_dim, hidden_dim, output_dim, key)

    def init_params(self, input_dim, hidden_dim, output_dim, key):
        """
        Initialize network parameters.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output dimension.
            key (jax.random.PRNGKey): PRNG key.

        Returns:
            dict: Dictionary containing initialized weights and biases.
        """
        keys = random.split(key, 6)
        params = {
            'W1': random.normal(keys[0], (input_dim, hidden_dim)) * jnp.sqrt(2 / input_dim),
            'b1': jnp.zeros(hidden_dim),
            'W2': random.normal(keys[1], (hidden_dim, hidden_dim)) * jnp.sqrt(2 / hidden_dim),
            'b2': jnp.zeros(hidden_dim),
            'W3': random.normal(keys[2], (hidden_dim, hidden_dim)) * jnp.sqrt(2 / hidden_dim),
            'b3': jnp.zeros(hidden_dim),
            'W4': random.normal(keys[3], (hidden_dim, output_dim)) * jnp.sqrt(2 / hidden_dim),
            'b4': jnp.zeros(output_dim),
            'W_t': random.normal(keys[4], (1, hidden_dim)) * jnp.sqrt(2),
            'b_t': jnp.zeros(hidden_dim),
        }
        return params

    def forward(self, params, x, t):
        """
        Forward pass of the network.

        Args:
            params (dict): Network parameters.
            x (jnp.array): Input data of shape [batch_size, input_dim].
            t (jnp.array): Time embedding of shape [batch_size, 1].

        Returns:
            jnp.array: Output predictions of shape [batch_size, output_dim].
        """
        # Time embedding
        t_emb = jax.nn.relu(jnp.dot(t, params['W_t']) + params['b_t'])  # Shape: [batch_size, hidden_dim]

        h1 = jax.nn.relu(jnp.dot(x, params['W1']) + params['b1'] + t_emb)
        h2 = jax.nn.relu(jnp.dot(h1, params['W2']) + params['b2'])
        h3 = jax.nn.relu(jnp.dot(h2, params['W3']) + params['b3'])
        output = jnp.dot(h3, params['W4']) + params['b4']
        return output