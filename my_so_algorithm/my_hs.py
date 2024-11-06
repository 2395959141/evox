from functools import partial

import jax
import jax.numpy as jnp
import copy

from evox.utils import *
from evox import Algorithm, State, jit_class

import jax
import jax.numpy as jnp
from jax import random
from evox import Algorithm, State, jit_class, use_state

from functools import partial
import jax
import jax.numpy as jnp
from jax import random
from evox import Algorithm, State, jit_class, use_state

@jit_class
class my_HarmonySearch(Algorithm):
    def __init__(
        self,
        lb,          # 变量下界
        ub,          # 变量上界
        HMS,         # 和谐记忆大小
        nNew,        # 新生成的解数量
        HMCR,        # 和谐记忆考虑率
        PAR,         # 音高调整率
        BW           # 调整带宽
    ):
        self.dim = lb.shape[0]  # 解的维度
        self.lb = lb            # 下界
        self.ub = ub            # 上界
        self.HMS = HMS          # 种群大小
        self.nNew = nNew        # 新解数量
        self.HMCR = HMCR        # 和谐记忆考虑率
        self.PAR = PAR          # 音高调整率
        self.BW = BW            # 调整带宽

    def setup(self, key):
        key, subkey = random.split(key)
        population = random.uniform(subkey, shape=(self.HMS, self.dim), minval=self.lb, maxval=self.ub)
        return State(
            population=population,
            key=key
        )

    def ask(self, state):
        return state.population, state  # 返回当前种群

    def adjust_solution(self, solution):
        """
        调整解的所有元素，使得它们的和等于指定的目标值（例如 10500）。
        """
        target_sum = 10500  # 设置目标总和值
        tolerance = 1e-2  # 设置容差，确保浮点数不会无限循环

        # 循环调整每个解的值，直到它们的和等于 target_sum
        def cond_fun(state):
            sol, sum_sol = state
            return jnp.abs(sum_sol - target_sum) > tolerance

        def body_fun(state):
            sol, sum_sol = state
            diff = sum_sol - target_sum
            sol = sol - diff / self.dim  # 平均分配调整
            sol = jnp.clip(sol, self.lb, self.ub)  # 确保不超出上下限
            return sol, jnp.sum(sol)

        sum_solution = jnp.sum(solution)
        solution, _ = jax.lax.while_loop(cond_fun, body_fun, (solution, sum_solution))
        return solution

    def tell(self, state, new_harmonies):
        key, subkey = random.split(state.key)
        
        # 生成随机数用于向量化逻辑
        harmony_rand = random.uniform(subkey, shape=(self.nNew, self.dim))
        pitch_rand = random.uniform(subkey, shape=(self.nNew, self.dim))
        idx_rand = random.randint(subkey, (self.nNew, self.dim), 0, self.HMS)

        # 和谐记忆条件掩码 (True 表示使用和谐记忆中的值)
        harmony_mask = harmony_rand < self.HMCR
        pitch_mask = pitch_rand < self.PAR

        # 从现有种群中随机选择和谐记忆
        selected_harmonies = state.population[idx_rand, jnp.arange(self.dim)]

        # 使用和谐记忆和音高调整进行向量化操作
        new_harmonies = jnp.where(
            harmony_mask, 
            selected_harmonies, 
            random.uniform(subkey, shape=(self.nNew, self.dim), minval=self.lb, maxval=self.ub)
        )
        
        # 音高调整步骤（随机带宽调整）
        pitch_adjustment = new_harmonies + pitch_mask * self.BW * random.normal(subkey, shape=new_harmonies.shape)

        # 对生成的解进行裁剪，确保其在上下限之间
        new_harmonies = jnp.clip(pitch_adjustment, self.lb, self.ub)

        # 调整生成的解，使其满足总和值等于10500的约束
        new_harmonies = jax.vmap(self.adjust_solution)(new_harmonies)

        # 合并当前种群与新生成的和谐
        merged_population = jnp.vstack([state.population, new_harmonies])

        # 随机打乱并选择前 HMS 个解
        idx_shuffle = random.permutation(key, merged_population.shape[0])
        best_population = merged_population[idx_shuffle][:self.HMS]

        return state.replace(population=best_population, key=key)

