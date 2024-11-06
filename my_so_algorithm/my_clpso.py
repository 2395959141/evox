# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Comprehensive learning particle swarm optimizer for global optimization of multimodal functions
# Link: https://ieeexplore.ieee.org/document/1637688
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.utils import *
from evox import Algorithm, State, jit_class


# CL-PSO: Comprehensive Learning PSO
@jit_class
class my_CLPSO(Algorithm):
    def __init__(
        self,
        lb,  # lower bound of problem
        ub,  # upper bound of problem
        Pd,  # 发电机组满足的等式约束
        pop_size,  # population size
        inertia_weight,  # w
        const_coefficient,  # c
        learning_probability,  # P_c. shape:(pop_size,). It can be different for each particle
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.Pd = Pd
        self.pop_size = pop_size
        self.w = inertia_weight
        self.c = const_coefficient # 学习因子c
        self.P_c = learning_probability  # 学习概率P_c，每个粒子可以有不同的学习概率

    def setup(self, key):
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
        length = self.ub - self.lb
        population = jax.random.uniform(init_pop_key, shape=(self.pop_size, self.dim))
        population = population * length + self.lb
        velocity = jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim))
        velocity = velocity * length * 2 - length

        return State(
            population=population,
            velocity=velocity,
            pbest_position=population,
            pbest_fitness=jnp.full((self.pop_size,), jnp.inf),
            gbest_position=population[0],
            gbest_fitness=jnp.array([jnp.inf]),
            key=state_key,
        )

    def ask(self, state):
        """返回当前种群位置供适应度评估
        
        Args:
            state: 算法当前状态
            
        Returns:
            tuple: (当前种群位置, 当前状态)
        """
        return state.population, state
    

    def adjust_solution(self, solution):
        """
        调整解的所有元素使得它们的和等于 Pd。
        """
        def cond_fun(state):
            sol, sum_sol = state
            return jnp.abs(sum_sol - self.Pd) > 1e-2

        def body_fun(state):
            sol, sum_sol = state
            diff = sum_sol - self.Pd
            sol = sol - diff / self.dim  # 平均分配调整
            sol = jnp.clip(sol, self.lb, self.ub)  # 确保不超出上下限
            return sol, jnp.sum(sol)

        sum_solution = jnp.sum(solution)
        solution, _ = jax.lax.while_loop(cond_fun, body_fun, (solution, sum_solution))
        return solution


    def tell(self, state, fitness):

        key, random_coefficient_key, rand1_key, rand2_key, rand_key = jax.random.split(
            state.key, num=5
        )

        random_coefficient = jax.random.uniform(
            random_coefficient_key, shape=(self.pop_size, self.dim)
        )

        # ----------------- Update pbest -----------------
        compare = state.pbest_fitness > fitness # 比较后得到条件数组
        #如果新位置更好,就更新pbest_position;否则保持原来的pbest_position
        pbest_position = jnp.where(
            compare[:, jnp.newaxis], state.population, state.pbest_position
        )
        #更新个体历史最优适应度值
        pbest_fitness = jnp.minimum(state.pbest_fitness, fitness)

        # ----------------- Update gbest -----------------
        gbest_position, gbest_fitness = min_by(
            [state.gbest_position[jnp.newaxis, :], state.population],
            [state.gbest_fitness, fitness],
        )
        gbest_fitness = jnp.atleast_1d(gbest_fitness)

        # ------------------ Choose pbest ----------------------

        rand1_index = jnp.floor(
            jax.random.uniform(
                rand1_key, shape=(self.pop_size,), minval=0, maxval=self.pop_size
            )
        ).astype(int)
        rand2_index = jnp.floor(
            jax.random.uniform(
                rand2_key, shape=(self.pop_size,), minval=0, maxval=self.pop_size
            )
        ).astype(int)
        learning_index = jnp.where(
            pbest_fitness[rand1_index] < pbest_fitness[rand2_index],
            rand1_index,
            rand2_index,
        )
        learning_pbest = state.pbest_position[learning_index, :]
        rand_possibility = jax.random.uniform(rand_key, shape=(self.pop_size,))
        rand_possibility = jnp.broadcast_to(
            rand_possibility[:, jnp.newaxis], shape=(self.pop_size, self.dim)
        )
        P_c = jnp.broadcast_to(
            self.P_c[:, jnp.newaxis], shape=(self.pop_size, self.dim)
        )
        pbest = jnp.where(rand_possibility < P_c, learning_pbest, state.pbest_position)

        # ------------------------------------------------------

        velocity = self.w * state.velocity + self.c * random_coefficient * (
            pbest - state.population
        )
        population = state.population + velocity

        # 调用adjust_solution函数来调整population
        population = jax.vmap(self.adjust_solution)(population)
        
        population = jnp.clip(population, self.lb, self.ub)
        return state.replace(
            population=population,
            velocity=velocity,
            pbest_position=pbest_position,
            pbest_fitness=pbest_fitness,
            gbest_position=gbest_position,
            gbest_fitness=gbest_fitness,
            key=key,
        )
