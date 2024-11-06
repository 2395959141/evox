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
from scipy.stats import qmc


@jit_class
class my_PSO(Algorithm):
    def __init__(
        ## 带阀点效应的ELD问题
        self,
        # lb,
        # ub,
        pop_size,
        problem_type='traditional_eld',  # 新增：问题类型
        problem_params=None,  #问题特定参数
        pso_params=None,  # PSO特定参数
        init_method='uniform',
    ):
        
        ## 基本参数设置
        #self.dim = lb.shape[0]
        # self.lb = lb
        # self.ub = ub
        self.pop_size = pop_size
        self.problem_type = problem_type #设置ELD问题类型
        self.problem_params = problem_params #不同ELD问题相关参数
        self.pso_params = pso_params #PSO参数
        
        self.init_method = init_method #种群初始化方法
        # self.heat_demand = Hd
        # self.chp_params = chp_params
        # self.heat_params = heat_params

        # 设置默认的PSO参数
        default_pso_params = {
            'inertia_weight': 0.6,
            'cognitive_coefficient': 2.5,
            'social_coefficient': 0.8,
            'mean': None,
            'stdev': None
        }
        # 更新PSO参数
        self.pso_params = default_pso_params
        if pso_params is not None:
            self.pso_params.update(pso_params)

        # 将PSO参数作为对象属性
        self.w = default_pso_params['inertia_weight']
        self.phi_p = default_pso_params['cognitive_coefficient']
        self.phi_g = default_pso_params['social_coefficient']    

        # 根据问题类型设置特定参数
        self._init_problem_params(problem_params)

        
    def _init_problem_params(self, problem_params):
        """根据问题类型初始化特定参数"""
        if self.problem_type == 'traditional_eld':
            self._init_traditional_eld_params(problem_params)
        elif self.problem_type == 'chp_24_eld':
            self._init_chp_24_eld_params(problem_params)
        else:
            raise ValueError(f"不支持的问题类型: {self.problem_type}")

    def _init_traditional_eld_params(self, params):
        """传统经济负荷调度问题参数初始化"""
        default_params = {
            'Pd': None,  # 功率需求
            'lb': None,  #下界
            'ub': None,  #上界
        }
        
        if params is not None:
            default_params.update(params)
        
        required_params = ['Pd', 'lb', 'ub']
        for param in required_params:
            if default_params[param] is None:
                raise ValueError(f"传统ELD问题需要设置参数: {param}")

        self.Pd = default_params['Pd']
        self.lb = default_params['lb']
        self.ub = default_params['ub']
        self.dim = self.lb.shape[0]


    def _init_chp_24_eld_params(self, params):
        """热电联产24小时调度问题参数初始化"""
        default_params = {
            'Pd': None,          # 功率需求
            'Hd': None,          # 热力需求
            'chp_units': None,   # 热电联产机组参数
            'power_units': None, # 常规发电机组参数
            'heat_units': None   # 供热机组参数
        }    
        if params is not None:
            default_params.update(params)
            
        # 验证必要参数
        required_params = ['Pd', 'Hd', 'chp_units','power_units','heat_units']
        for param in required_params:
            if default_params[param] is None:
                raise ValueError(f"CHP-24-ELD问题需要设置参数: {param}")

        # 允许在运行时动态设置问题参数
        for key, value in default_params.items():
            setattr(self, key, value)




    def setup(self, key):
        """初始化种群"""
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
        if self.problem_type == 'traditional_eld':
            return self._init_tradition_eld_setup(key)
        elif self.problem_type == 'chp_24_eld':
            return self._init_chp_24_eld_setup(key)
        else:
            raise ValueError(f"不支持的任务类型: {self.problem_type}")


    def _init_tradition_eld_setup(self, key):
        """初始化种群"""
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
        if self.init_method == 'gassian':
            population = jax.random.uniform(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
            velocity = self.stdev * jax.random.normal(
                init_v_key, shape=(self.pop_size, self.dim)
            )
        elif self.init_method == 'uniform':
            length = self.ub - self.lb
            population = jax.random.uniform(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = population * length + self.lb
            velocity = jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim))
            velocity = velocity * length * 2 - length
        elif self.init_method == 'latin_hypercube':
            # 拉丁超立方体采样初始化 
            sampler = qmc.LatinHypercube(d=self.dim)
            population = sampler.random(n=self.pop_size)
            population = population * (self.ub - self.lb) + self.lb
            velocity = jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim))
            velocity = velocity * (self.ub - self.lb) * 2 - (self.ub - self.lb)

        return State(
            population=population,
            velocity=velocity,
            local_best_location=population,
            local_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            global_best_location=population[0],
            global_best_fitness=jnp.array([jnp.inf]),
            key=state_key,
        )

    # def _init_chp_24_eld_setup(self, key):
    #     """初始化种群"""
    #     state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
    #     if self.init_method == 'uniform':
        


    def ask(self, state):
        return state.population, state

    def tell(self, state, fitness):

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



        key, rg_key, rp_key = jax.random.split(state.key, 3)

        rg = jax.random.uniform(rg_key, shape=(self.pop_size, self.dim))
        rp = jax.random.uniform(rp_key, shape=(self.pop_size, self.dim))

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

        
        velocity = (
            self.w  * state.velocity
            + self.phi_p * rp  * (local_best_location - state.population)
            + self.phi_g * rg  * (global_best_location - state.population)
        )
        population = state.population + velocity

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
            key=key,
        )

