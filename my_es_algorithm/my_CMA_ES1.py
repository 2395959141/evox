# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Completely Derandomized Self-Adaptation in Evolution Strategies
# Link: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf
#
# Title: A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity (SepCMAES)
# Link: https://inria.hal.science/inria-00287367/document
#
# Title: A Restart CMA Evolution Strategy With Increasing Population Size (IPOPCMAES)
# Link: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf
#
# Title: Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed (BIPOPCMAES)
# Link: https://inria.hal.science/inria-00382093/document
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

import math

import jax
import jax.numpy as jnp
from jax import lax

import evox
from evox import Algorithm, State
from evox.algorithms.so.es_variants.sort_utils import sort_by_key



@evox.jit_class
class my_CMAES1(Algorithm):
    def __init__(
        self, center_init, init_stdev, lb, ub,pop_size=None, recombination_weights=None, cm=1, 
    ):
        """
        This implementation follows `The CMA Evolution Strategy: A Tutorial <https://arxiv.org/pdf/1604.00772.pdf>`_.

        .. note::
            CMA-ES involves eigendecomposition,
            which introduces relatively large numerical error,
            and may lead to non-deterministic behavior on different hardware backends.
        """
        #获取上下界
        self.lb = lb
        self.ub = ub
        #self.Pd = Pd    

        self.center_init = center_init
        assert init_stdev > 0, "Expect variance to be a non-negative float"
        self.init_stdev = init_stdev
        self.dim = center_init.shape[0]
        self.cm = cm
        if pop_size is None:
            # auto
            self.pop_size = 4 + math.floor(3 * math.log(self.dim))
        else:
            self.pop_size = pop_size

        if recombination_weights is None:
            # auto
            self.mu = self.pop_size // 2
            self.weights = jnp.log(self.mu + 0.5) - jnp.log(jnp.arange(1, self.mu + 1))
            self.weights = self.weights / sum(self.weights)
        else:
            assert (
                recombination_weights[1:] <= recombination_weights[:-1]
            ).all(), "recombination_weights must be non-increasing"
            assert (
                jnp.abs(jnp.sum(recombination_weights) - 1) < 1e-6
            ), "sum of recombination_weights must be 1"
            assert (
                recombination_weights > 0
            ).all(), "recombination_weights must be positive"
            self.mu = recombination_weights.shape[0]
            assert self.mu <= self.pop_size
            self.weights = recombination_weights

        self.mueff = jnp.sum(self.weights) ** 2 / jnp.sum(self.weights**2)
        # time constant for cumulation for C
        self.cc = (4 + self.mueff / self.dim) / (
            self.dim + 4 + 2 * self.mueff / self.dim
        )

        # t-const for cumulation for sigma control
        self.cs = (2 + self.mueff) / (self.dim + self.mueff + 5)

        # learning rate for rank-one update of C
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)

        # learning rate for rank-μ update of C
        # convert self.dim to float first to prevent overflow
        self.cmu = min(
            1 - self.c1,
            (
                2
                * (self.mueff - 2 + 1 / self.mueff)
                / ((float(self.dim) + 2) ** 2 + self.mueff)
            ),
        )

        # damping for sigma
        self.damps = (
            1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        )

        self.chiN = self.dim**0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.decomp_per_iter = 1 / (self.c1 + self.cmu) / self.dim / 10
        self.decomp_per_iter = max(jnp.floor(self.decomp_per_iter).astype(jnp.int32), 1)

    def setup(self, key):
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        B = jnp.eye(self.dim)
        D = jnp.ones((self.dim,))
        C = B @ jnp.diag(D) @ B.T
        return State(
            pc=pc,
            ps=ps,
            B=B,
            D=D,
            C=C,
            count_eigen=0,
            count_iter=0,
            invsqrtC=C,
            mean=self.center_init,
            sigma=self.init_stdev,
            key=key,
            population=jnp.empty((self.pop_size, self.dim)),
        )

    def ask(self, state):

        def adjust_solution(solution):
        # 循环调整每个解的值，直到它们的和等于 Pd
            def cond_fun(state):
                sol, sum_sol = state
                return jnp.abs(sum_sol - 10500) > 1e-2

            def body_fun(state):
                sol, sum_sol = state
                diff = sum_sol - 10500
                sol = sol - diff / self.dim  # 平均分配调整
                sol = jnp.clip(sol, self.lb, self.ub)  # 确保不超出上下限
                return sol, jnp.sum(sol)

            sum_solution = jnp.sum(solution)
            solution, _ = jax.lax.while_loop(cond_fun, body_fun, (solution, sum_solution))
            return solution


        key, sample_key = jax.random.split(state.key)
        noise = jax.random.normal(sample_key, (self.pop_size, self.dim))
        population = state.mean + state.sigma * (state.D * noise) @ state.B.T


        #对种群批量调整，满足输出功率的等式约束
        population = jax.vmap(adjust_solution)(population)

        # 对生成的解应用反射边界约束
        population = jnp.where(population > self.ub, 2 * self.ub - population, population)
        population = jnp.where(population < self.lb, 2 * self.lb - population, population)

        
        #population  = adjust_solution(population)

        #population = jnp.clip(population, self.lb, self.ub)

        new_state = state.replace(
            population=population, count_iter=state.count_iter + 1, key=key
        )
        return population, new_state

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)

        mean = self._update_mean(state.mean, population)
        delta_mean = mean - state.mean

        ps = self._update_ps(state.ps, state.invsqrtC, state.sigma, delta_mean)

        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - self.cs) ** (2 * state.count_iter))
            < (1.4 + 2 / (self.dim + 1)) * self.chiN
        )
        pc = self._update_pc(state.pc, ps, delta_mean, state.sigma, hsig)
        C = self._update_C(state.C, pc, state.sigma, population, state.mean, hsig)
        sigma = self._update_sigma(state.sigma, ps)

        B, D, invsqrtC = lax.cond(
            state.count_iter % self.decomp_per_iter == 0,
            self._decomposition_C,
            lambda _C: (state.B, state.D, state.invsqrtC),
            C,
        )

        return state.replace(
            mean=mean, ps=ps, pc=pc, C=C, sigma=sigma, B=B, D=D, invsqrtC=invsqrtC
        )

    def _update_mean(self, mean, population):
        update = self.weights @ (population[: self.mu] - mean)
        return mean + self.cm * update

    def _update_ps(self, ps, invsqrtC, sigma, delta_mean):
        return (1 - self.cs) * ps + jnp.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * invsqrtC @ delta_mean / sigma

    def _update_pc(self, pc, ps, delta_mean, sigma, hsig):
        return (1 - self.cc) * pc + hsig * jnp.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * delta_mean / sigma

    def _update_C(self, C, pc, sigma, population, old_mean, hsig):
        y = (population[: self.mu] - old_mean) / sigma
        return (
            (1 - self.c1 - self.cmu) * C
            + self.c1 * (jnp.outer(pc, pc) + (1 - hsig) * self.cc * (2 - self.cc) * C)
            + self.cmu * (y.T * self.weights) @ y
        )

    def _update_sigma(self, sigma, ps):
        return sigma * jnp.exp(
            (self.cs / self.damps) * (jnp.linalg.norm(ps) / self.chiN - 1)
        )

    def _decomposition_C(self, C):
        C = jnp.triu(C) + jnp.triu(C, 1).T  # enforce symmetry
        D, B = jnp.linalg.eigh(C)
        D = jnp.sqrt(D)
        invsqrtC = (B / D) @ B.T
        return B, D, invsqrtC