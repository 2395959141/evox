# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A new optimizer using particle swarm theory
# Link: https://ieeexplore.ieee.org/document/494215
# --------------------------------------------------------------------------------------

from functools import partial

import jax
import jax.numpy as jnp
import copy

from evox.utils import *
from evox import Algorithm, State, jit_class

#from evox.utils.common import max_by, min_by


@jit_class
class my_PSO_Model_Swarm(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        Pd, #功率约束
        pop_size,
        inertia_weight=0.6,
        cognitive_coefficient=2.5,
        social_coefficient=0.8,
        repulsion_coefficient=0.05,  # 添加新的系数用于最差解的排斥
        stagnation_threshold = 2000,
        mean=None,
        stdev=None,
        velocity_decay_rate=0.90,  # 速度衰减率
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.w = inertia_weight
        self.phi_p = cognitive_coefficient
        self.phi_g = social_coefficient
        self.phi_w = repulsion_coefficient  # 存储新的排斥系数
        self.mean = mean
        self.stdev = stdev
        self.Pd = Pd  # 功率约束目标值
        self.stagnation_threshold = stagnation_threshold  # 存储停滞阈值
        self.velocity_decay_rate = velocity_decay_rate
        #self.tolerance = tolerance #满足等式约束的处理的精度


    def setup(self, key):
        state_key, init_pop_key, init_v_key,v_keys = jax.random.split(key, 4)
        if self.mean is not None and self.stdev is not None:
            population = self.stdev * jax.random.normal(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
            # velocity = self.stdev * jax.random.normal(
            #     init_v_key, shape=(self.pop_size, self.dim)
            # )
        else:
            length = self.ub - self.lb
            population = jax.random.uniform(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = population * length + self.lb
            # velocity = jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim))
            # velocity = velocity * length * 2 - length
            ##上面两行是原始的PSO的初始化速度公式

            ##下面两行是改进的PSO的初始化速度公式
            # 修改速度初始化部分
            #init_v_key ,v_keys = jax.random.split(key, 2)
            def init_velocity(population, key):
                # 生成两个随机索引
                idx1 = jax.random.randint(key, shape=(), minval=0, maxval=self.pop_size)
                key2 = jax.random.split(key)[0]
                idx2 = jax.random.randint(key2, shape=(), minval=0, maxval=self.pop_size)
                
                # 计算两个随机位置的差作为初始速度
                return population[idx1] - population[idx2]

            # # 为每个粒子生成随机密钥
            v_keys = jax.random.split(init_v_key, self.pop_size)

            # 使用vmap批量初始化速度
            velocity = jax.vmap(init_velocity, in_axes=(None, 0))(population, v_keys)
            # # 使用vmap批量初始化速度
            # velocity = jax.vmap(init_velocity)(
            #     population,
            #     v_keys
            # )

        return State(
            population=population,
            velocity=velocity,
            local_best_location=population,
            local_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            global_best_location=population[0],
            global_best_fitness=jnp.array([jnp.inf]),
            global_worst_location=population[0],  # 添加全局最差位置
            global_worst_fitness=jnp.array([-jnp.inf]),  # 添加全局最差适应度
            stagnation_counter=jnp.zeros(self.pop_size),  # 添加停滞计数器
            velocity_weight=jnp.array(1.0),  # 初始速度权重为1.0
            key=state_key,
        )

    

    def ask(self, state):
        return state.population, state
    
    # def max_by(
    #     values: Union[jax.Array, list[jax.Array]],
    #     keys: Union[jax.Array, list[jax.Array]],
    # ):
    #     if isinstance(values, list):
    #         values = jnp.concatenate(values)
    #         keys = jnp.concatenate(keys)

    #     max_index = jnp.argmax(keys)
    #     return values[max_index], keys[max_index]


    def tell(self, state, fitness):

        def max_by(
            values: Union[jax.Array, list[jax.Array]],
            keys: Union[jax.Array, list[jax.Array]],
        ):
            if isinstance(values, list):
                values = jnp.concatenate(values)
                keys = jnp.concatenate(keys)

            max_index = jnp.argmax(keys)
            return values[max_index], keys[max_index]

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



        key, rg_key, rp_key,rw_key = jax.random.split(state.key, 4)

        rg = jax.random.uniform(rg_key, shape=(self.pop_size, self.dim))
        rp = jax.random.uniform(rp_key, shape=(self.pop_size, self.dim))
        rw = jax.random.uniform(rw_key, shape=(self.pop_size, self.dim))  # 新的随机数用于排斥


        compare = state.local_best_fitness > fitness
        local_best_location = jnp.where(
            compare[:, jnp.newaxis], state.population, state.local_best_location
        )
        local_best_fitness = jnp.minimum(state.local_best_fitness, fitness)

        global_best_location, global_best_fitness = min_by(
            [state.global_best_location[jnp.newaxis, :], state.population],
            [state.global_best_fitness, fitness],
        )

        global_best_fitness = jnp.atleast_1d(global_best_fitness)


         # 更新全局最优和最差解
        global_best_location, global_best_fitness = min_by(
            [state.global_best_location[jnp.newaxis, :], state.population],
            [state.global_best_fitness, fitness],
        )
        
        global_worst_location, global_worst_fitness = max_by(  # 添加最差解的更新
            [state.global_worst_location[jnp.newaxis, :], state.population],
            [state.global_worst_fitness, fitness],
        )

        global_best_fitness = jnp.atleast_1d(global_best_fitness)
        global_worst_fitness = jnp.atleast_1d(global_worst_fitness)

        velocity = (
            self.w * state.velocity
            + self.phi_p * rp * (local_best_location - state.population)
            + self.phi_g * rg * (global_best_location - state.population)
            - self.phi_w * rw * (global_worst_location - state.population)
        )
        population = state.population + state.velocity_weight * velocity

        # 更新停滞计数器
        stagnation_counter = jnp.where(
            compare,
            jnp.zeros_like(state.stagnation_counter),  # 如果更新了最佳位置，重置计数器
            state.stagnation_counter + 1  # 否则增加计数器
        )

        # 检查是否需要重置粒子位置
        reset_mask = (stagnation_counter >= self.stagnation_threshold)
        population = jnp.where(
            reset_mask[:, jnp.newaxis],
            state.local_best_location,  # 重置到个人最佳位置
            state.population  # 保持当前位置
        )

        #对种群批量调整，满足输出功率的等式约束
        population = jax.vmap(adjust_solution)(population)
        #population  = adjust_solution(population)

        population = jnp.clip(population, self.lb, self.ub)

        return state.replace(
            population=population,
            velocity=velocity,
            local_best_location=local_best_location,
            local_best_fitness=local_best_fitness,
            global_best_location=global_best_location,
            global_best_fitness=global_best_fitness,
            global_worst_location=global_worst_location,  # 添加最差位置更新
            global_worst_fitness=global_worst_fitness,    # 添加最差适应度更新
            stagnation_counter=stagnation_counter,  # 更新停滞计数器
            velocity_weight=state.velocity_weight * self.velocity_decay_rate,  # 更新速度权重
            key=key,
        )

