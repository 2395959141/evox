import evoxlib as exl
import jax
import jax.numpy as jnp


@exl.jit_class
class GaussianMutation(exl.Operator):
    def __init__(self, stdvar=1.0):
        self.stdvar = stdvar

    def setup(self, key):
        return exl.State(key=key)

    def __call__(self, state, x):
        key, subkey = jax.random.split(state.key)
        perturbation = jax.random.normal(subkey, x.shape) * self.stdvar
        return exl.State(key=key), x + perturbation