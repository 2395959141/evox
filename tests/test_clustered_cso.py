import evoxlib as exl
import jax
import jax.numpy as jnp
import pytest


@exl.jit_class
class Pipeline(exl.Module):
    def __init__(self):
        # create a clustered CSO
        self.algorithm = exl.algorithms.ClusterdAlgorithm(
            base_algorithm=exl.algorithms.CSO,
            dim=100,
            num_cluster=10,
            lb=jnp.full(shape=(10,), fill_value=-32),
            ub=jnp.full(shape=(10,), fill_value=32),
            # the base algorithm is CSO
            pop_size=100,
        )
        # choose a problem
        self.problem = exl.problems.classic.Ackley()

    def setup(self, key):
        # record the min fitness
        return exl.State({"min_fitness": 1e9})

    def step(self, state):
        # one step
        state, pop = self.algorithm.ask(state)
        state, fitness = self.problem.evaluate(state, pop)
        state = self.algorithm.tell(state, fitness)
        return state | {"min_fitness": jnp.minimum(state["min_fitness"], jnp.min(fitness))}

    def get_min_fitness(self, state):
        return state, state["min_fitness"]

# disable this test for now
def test_clustered_cso():
    # create a pipeline
    pipeline = Pipeline()
    # init the pipeline
    key = jax.random.PRNGKey(42)
    state = pipeline.init(key)

    # run the pipeline for 300 steps
    for i in range(300):
        state = pipeline.step(state)

    state, min_fitness = pipeline.get_min_fitness(state)
    assert min_fitness < 1