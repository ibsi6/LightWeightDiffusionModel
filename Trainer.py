import jax
import jax.numpy as jnp
from jax import random, jit, grad
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    """
    Handles training of the diffusion model.
    """

    def __init__(self, model, diffusion_process, data, key, learning_rate=1e-3, batch_size=128):
        """
        Initialize the trainer.

        Args:
            model (DiffusionModel): The diffusion model.
            diffusion_process (DiffusionProcess): The diffusion process.
            data (np.array): Training data.
            key (jax.random.PRNGKey): PRNG key.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
        """
        self.model = model
        self.diffusion_process = diffusion_process
        self.data = data
        self.key = key
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.opt_state = self.init_optimizer(model.params)

    def init_optimizer(self, params):
        """
        Initialize optimizer state.

        Args:
            params (dict): Model parameters.

        Returns:
            dict: Optimizer state.
        """
        opt_state = {
            'params': params,
            'm': {k: jnp.zeros_like(v) for k, v in params.items()},
            'v': {k: jnp.zeros_like(v) for k, v in params.items()},
            't': 0
        }
        return opt_state

    def update_optimizer(self, grads, opt_state):
        """
        Update parameters using Adam optimizer.

        Args:
            grads (dict): Gradients of parameters.
            opt_state (dict): Optimizer state.

        Returns:
            dict: Updated optimizer state.
        """
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        opt_state['t'] += 1
        t = opt_state['t']
        updated_params = {}
        for k in opt_state['params'].keys():
            g = grads[k]
            m = beta1 * opt_state['m'][k] + (1 - beta1) * g
            v = beta2 * opt_state['v'][k] + (1 - beta2) * (g ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            updated_params[k] = opt_state['params'][k] - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
            opt_state['m'][k] = m
            opt_state['v'][k] = v
        opt_state['params'] = updated_params
        return opt_state

    def loss_fn(self, params, x0, t, noise):
        """
        Compute the loss.

        Args:
            params (dict): Model parameters.
            x0 (jnp.array): Original data batch.
            t (jnp.array): Time steps batch.
            noise (jnp.array): Noise added to data.

        Returns:
            float: Loss value.
        """
        # Get noisy data
        xt = self.diffusion_process.q_sample(x0, t, noise)
        t_normalized = t[:, None] / self.diffusion_process.T  # Normalize time steps
        # Predict noise
        pred_noise = self.model.forward(params, xt, t_normalized)
        # Compute MSE loss
        loss = jnp.mean((pred_noise - noise) ** 2)
        return loss

    def train_step(self, opt_state, x0_batch):
        """
        Perform a single training step.

        Args:
            opt_state (dict): Optimizer state.
            x0_batch (np.array): Batch of original data.

        Returns:
            tuple: Updated optimizer state, loss value.
        """
        key, subkey = random.split(self.key)
        batch_size = x0_batch.shape[0]
        # Sample random time steps
        t = random.randint(subkey, shape=(batch_size,), minval=0, maxval=self.diffusion_process.T)
        t = t.astype(jnp.int32)
        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=x0_batch.shape)
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(self.loss_fn)(opt_state['params'], x0_batch, t, noise)
        # Update optimizer state
        opt_state = self.update_optimizer(grads, opt_state)
        self.key = key
        return opt_state, loss

    def train(self, num_epochs):
	    num_batches = self.data.shape[0] // self.batch_size
	    for epoch in range(num_epochs):
	        # Shuffle data
	        perm = np.random.permutation(self.data.shape[0])
	        data_shuffled = self.data[perm]
	        epoch_loss = 0
	        for i in range(num_batches):
	            x0_batch = data_shuffled[i * self.batch_size:(i + 1) * self.batch_size]
	            self.opt_state, loss = self.train_step(self.opt_state, x0_batch)
	            if jnp.isnan(loss):
	                print("Loss became NaN during training")
	                return  # Stop training if loss is NaN
	            epoch_loss += loss
	        epoch_loss /= num_batches
	        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

