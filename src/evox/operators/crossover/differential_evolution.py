<<<<<<< HEAD
import jax
=======
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
import jax.numpy as jnp
from jax import jit, random, vmap
from evox import jit_class

<<<<<<< HEAD
from evox import jit_class, Operator, State
# def _random_scaling(key, x):
#     batch, dim = x.shape
#     candidates = jnp.tile(x, (3, 1))  # shape: (3*batch, dim)
#     candidates = jax.random.permutation(key, candidates, axis=0)
#     return candidates.reshape(batch, 3, dim)
=======

def _random_scaling(key, x):
    batch, dim = x.shape
    candidates = jnp.tile(x, (3, 1))  # shape: (3*batch, dim)
    candidates = random.permutation(key, candidates, axis=0)
    return candidates.reshape(batch, 3, dim)
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9


def _de_mutation(x1, x2, x3, F):
    # use DE/rand/1
    # parents[0], parents[1] and parents[2] may be the same with each other, or with the original individual
    mutated_pop = x1 + F * (x2 - x3)
    return mutated_pop


def _de_crossover(key, new_x, x, CR):
    batch, dim = x.shape
<<<<<<< HEAD
    random_crossover = jax.random.uniform(key, shape=(batch, dim))
    mask = random_crossover < cr
    return jnp.where(mask, new_x, x)


@jit_class
class DECrossover(Operator):
    def __init__(self, F=0.5, cr=1):
=======
    random_crossover = random.uniform(key, shape=(batch, dim))
    mask = random_crossover <= CR
    return jnp.where(mask, new_x, x)


@jit
def differential_evolve(key, x, F, CR):
    scaling_key, crossover_key = random.split(key, 2)
    scaled = _random_scaling(scaling_key, x)
    mutated_individual = vmap(_de_mutation)(scaled, F)
    children = _de_crossover(crossover_key, mutated_individual, x, CR)
    return children


@jit_class
class DifferentialEvolve:
    def __init__(self, F=0.5, CR=0.7):
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
        """
        Parameters
        ----------
        F
            The scaling factor
        CR
            The probability of crossover
        """
        self.F = F
        self.CR = CR

<<<<<<< HEAD
    def setup(self, key):
        return State(key=key)

    def __call__(self, state, *args):
        key = state.key
        x1, x2, x3 = args
        key, de_key = jax.random.split(key)
        mutated_pop = _de_mutation(x1, x2, x3, self.F)

        children = _de_crossover(de_key, mutated_pop, x1, self.cr)

        return children, State(key=key)
=======
    def __call__(self, key, x):
        return differential_evolve(key, x, self.F, self.CR)
>>>>>>> e9d1a7a9a7ff3bb82fc0c14c4cd4180929c822b9
