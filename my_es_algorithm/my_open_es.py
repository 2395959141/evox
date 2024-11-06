# --------------------------------------------------------------------------------------
# This code implements algorithms described in the following papers:
#
# Title: Exponential Natural Evolution Strategies (XNES)
# Link: https://dl.acm.org/doi/abs/10.1145/1830483.1830557
#
# Title: Natural Evolution Strategies (SeparableNES)
# Link: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import optax

from evox import Algorithm, State, jit_class, use_state, utils
from jax import random


@jit_class
class my_OpenES(Algorithm):
    def __init__(
        self,
        center_init,
        pop_size,
        learning_rate,
        noise_stdev,
        lb,  # 新增：解的下界
        ub,  # 新增：解的上界
        optimizer=None,
        mirrored_sampling=True,
    ):
        """
        Implement the algorithm described in "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
        from https://arxiv.org/abs/1703.03864
        """
        assert noise_stdev > 0
        assert learning_rate > 0
        assert pop_size > 0

        if mirrored_sampling is True:
            assert (
                pop_size % 2 == 0
            ), "When mirrored_sampling is True, pop_size must be a multiple of 2."

        self.dim = center_init.shape[0]
        self.center_init = center_init
        self.pop_size = pop_size
        self.learning_rate = learning_rate
        self.noise_stdev = noise_stdev
        self.mirrored_sampling = mirrored_sampling
        self.lb = lb  # 新增
        self.ub = ub  # 新增

        if optimizer == "adam":
            self.optimizer = utils.OptaxWrapper(
                optax.adam(learning_rate=learning_rate), center_init
            )
        else:
            self.optimizer = None

    def setup(self, key):
        # placeholder
        population = jnp.tile(self.center_init, (self.pop_size, 1))
        noise = jnp.tile(self.center_init, (self.pop_size, 1))
        return State(
            population=population, center=self.center_init, noise=noise, key=key
        )

    def ask(self, state):
        key, noise_key = jax.random.split(state.key)
        if self.mirrored_sampling:
            noise = jax.random.normal(noise_key, shape=(self.pop_size // 2, self.dim))
            noise = jnp.concatenate([noise, -noise], axis=0)
        else:
            noise = jax.random.normal(noise_key, shape=(self.pop_size, self.dim))
        population = state.center[jnp.newaxis, :] + self.noise_stdev * noise

        # # 新增：如果指定了上下界，则对 population 进行裁剪
        # if self.lb is not None and self.ub is not None:
        #     population = jnp.clip(population, self.lb, self.ub)

        if self.lb is not None and self.ub is not None:     # 对生成的解应用反射边界约束
            population = jnp.where(population > self.ub, 2 * self.ub - population, population)
            population = jnp.where(population < self.lb, 2 * self.lb - population, population)

        # 循环调整 population，使每个解的所有维度之和等于 Pd
        Pd = 10500  # 设置目标总和值
        tolerance = 1e-2  # 设置容差，确保浮点数不会无限循环
    
        def adjust_solution(solution):
            # 循环调整每个解的值，直到它们的和等于 Pd
            def cond_fun(state):
                sol, sum_sol = state
                return jnp.abs(sum_sol - Pd) > tolerance

            def body_fun(state):
                sol, sum_sol = state
                diff = sum_sol - Pd
                sol = sol - diff / self.dim  # 平均分配调整
                sol = jnp.clip(sol, self.lb, self.ub)  # 确保不超出上下限
                return sol, jnp.sum(sol)

            sum_solution = jnp.sum(solution)
            solution, _ = jax.lax.while_loop(cond_fun, body_fun, (solution, sum_solution))
            return solution

    # 对 population 中的每个解进行调整
        population = jax.vmap(adjust_solution)(population)

        return population, state.replace(population=population, key=key, noise=noise)

    def tell(self, state, fitness):
        grad = state.noise.T @ fitness / self.pop_size / self.noise_stdev
        if self.optimizer is None:
            center = state.center - self.learning_rate * grad
        else:
            updates, state = use_state(self.optimizer.update)(state, state.center)
            center = optax.apply_updates(state.center, updates)

    #    # 新增：如果指定了上下界，则对 center 进行裁剪
    #     if self.lb is not None and self.ub is not None:
    #         center = jnp.clip(center, self.lb, self.ub)    

        return state.replace(center=center)
