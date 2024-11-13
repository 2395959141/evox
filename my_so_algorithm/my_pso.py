# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A new optimizer using particle swarm theory
# Link: https://ieeexplore.ieee.org/document/494215
# --------------------------------------------------------------------------------------

 

import jax
import jax.numpy as jnp
import copy

from evox.utils import *
from evox import Algorithm, State, jit_class
from scipy.stats import qmc
from jax import lax


@jit_class
class my_PSO(Algorithm):
    def __init__(
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
        self.pop_size = pop_size
        self.problem_type = problem_type #设置ELD问题类型
        self.problem_params = problem_params #不同ELD问相关参数
        self.pso_params = pso_params #PSO参数
        
        self.init_method = init_method #种群初始化方法

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
        init_method = {
            'traditional_eld': self._init_traditional_eld_params,
            'chp_24_eld': self._init_chp_24_eld_params,
        }

        try : 
            init_method[self.problem_type](problem_params)
        except KeyError:
            raise ValueError(f"不支持的问题类型: {self.problem_type}")

        
    def _init_traditional_eld_params(self, params):
        """传统经济负荷调度问题参数初始化"""
        default_params = {
            'Pd': None,  # 功率需求
            'lb_P': None,  #功率下界
            'ub_P': None,  #功率上界
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
        """24台机组的热电联产调度问题参数初始化"""
        default_params = {
            'Pd': None,          # 功率需求
            'Hd': None,          # 热力需求
            'chp_units': None,   # 热电联产机组参数
            'power_units': None, # 常规发电机组参数
            'heat_units': None,   # 供热机组参数
    
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
        
        # 根据不同类型机组设置上下界
            # 1. 常规发电机组上下界
        self.Pmin = default_params['power_units']['Pmin']
        self.Pmax = default_params['power_units']['Pmax'] 

            # 2. 热电联产机组上下界（功率和热力）
        self.PRmin = default_params['chp_units']['PRmin']
        self.PRmax = default_params['chp_units']['PRmax']
        self.HRmin = default_params['chp_units']['HRmin']
        self.HRmax = default_params['chp_units']['HRmax']

            # 3. 产热机组的热能上下界
        self.Hmin = default_params['heat_units']['Hmin']
        self.Hmax = default_params['heat_units']['Hmax']
        
        #功率维度;热点机组的热能和功率维度；热力维度
        #self.dim_P = len(self.Pmin)
        self.dim_P = len(self.Pmin)
        assert self.dim_P == 13, f"P电力机组功率维度应为13,实际为{self.dim_P}"
        # self.dim_H = len(self.Hmin)   
        self.dim_PH = len(self.PRmin)
        assert self.dim_PH == len(self.PRmin), f"热点混合机组的功率和热力维度应都为6,实际为{self.dim_PH}"
        # self.dim_PH = len(self.PRmin) 
        self.dim_H = len(self.Hmin)
        assert self.dim_H == len(self.Hmin), f"热能机组的热能维度应为5,实际为{self.dim_H}"

        # 4. 计算总维度
        self.dim = self.dim_P + 2*self.dim_PH + self.dim_H
        assert self.dim == 30, f"总维度应为30,实际为{self.dim}"

        # assert self.HRmax.shape == (self.dim_PH), f"HRmax维度应为{self.dim_PH}，实际为{self.HRmax.shape}"
        # assert self.HRmin.shape == (self.dim_PH), f"HRmin维度应为{self.dim_PH}，实际为{self.HRmin.shape}"

        # 5. 其他必要参数
        self.Pd = params['Pd']  # 总电力需求
        self.Hd = params['Hd']  # 总热力需求


    def setup(self, key):
        """初始化种群"""
        # 定义任务类型与对应方法的映射
        setup_methods = {
            'traditional_eld': self._setup_tradition_eld_setup,
            'chp_24_eld': self._setup_chp_24_eld_setup,
        }
        

        # 根据任务类型调用对应方法
        if self.problem_type not in setup_methods:
            raise ValueError(f"不支持的任务类型: {self.problem_type}")

        return setup_methods[self.problem_type](key)
        


    def _setup_tradition_eld_setup(self, key):
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


    def _setup_chp_24_eld_setup(self, key):
        """初始化种群"""
        state_key, init_pop_key, init_v_key = jax.random.split(key, 3)
         # 初始化种群和速度
        population = jnp.zeros((self.pop_size, self.dim))
        velocity = jnp.zeros((self.pop_size, self.dim))

         # 验证种群维度
        # print(f"Population shape: {population.shape}")
        # print(f"Expected shape: ({self.pop_size}, {self.dim})")
       
        if self.init_method == 'uniform':
            # 1.初始化纯电机组功率
            length = self.Pmax - self.Pmin
            population = population.at[:, :self.dim_P].set(
                jax.random.uniform(init_pop_key, shape=(self.pop_size, self.dim_P)) * length + self.Pmin
            )
            # 初始化纯电机组功率维度的速度
            velocity = velocity.at[:, :self.dim_P].set(
                jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim_P)) * length * 2 - length
            )

            # 2.初始化热电联产机组的功率输出
            length_PR = self.PRmax - self.PRmin
            assert length_PR.shape == (self.dim_PH,), f"PRmax/PRmin维度应为{self.dim_PH}，实际为{len(length_PR)}"
            population = population.at[:, self.dim_P:self.dim_P+self.dim_PH].set(
                jax.random.uniform(init_pop_key, shape=(self.pop_size, self.dim_PH)) * length_PR + self.PRmin
            )
            # 初始化热电联产机组功率维度的速度
            velocity = velocity.at[:, self.dim_P:self.dim_P+self.dim_PH].set(
                jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim_PH)) * length_PR * 2 - length_PR
            )

            # 根据热电联产机组初始化功率输出来初始化其热力输出（19:25）
                # 获取CHP机组的功率输出
            chp_power = population[:, self.dim_P:self.dim_P+self.dim_PH]
            # 将 chp_power 扩展到 (1000, 6) 的形状
            # chp_power = jnp.tile(chp_power, (population.shape[0], 1))
            rand = jax.random.uniform(init_pop_key, shape=(self.pop_size, self.dim_PH))
            ## !! 这里需要优化，使用for循环不会被jit编译
            for k in range(self.dim_PH):
                # CHP unit 14 & 16 (第1和第3个)
                if k in [0, 2]:
                    heat = jnp.where(
                        (chp_power[:, k] >= 81) & (chp_power[:, k] <= 98.8),
                        rand[:, k] * (188/335 + 524/89) * (chp_power[:, k]-81) + 104.8 - 524/89 * (chp_power[:, k]-81),
                        jnp.where(
                            (chp_power[:, k] > 98.8) & (chp_power[:, k] <= 215),
                            rand[:, k] * (104.8 + 188/335 * (chp_power[:, k]-81)),
                            jnp.where(
                                (chp_power[:, k] > 215) & (chp_power[:, k] <= 247),
                                rand[:, k] * (-45/8) * (chp_power[:, k]-247),
                                0.0
                            )
                        )
                    )
                    #population = population.at[:,self.dim_P+self.dim_PH+k].set(heat)

                # CHP unit 15 & 17 (第2和第4个)
                elif k in [1, 3]:
                    heat = jnp.where(
                        (chp_power[:, k] >= 40) & (chp_power[:, k] <= 44),
                        rand[:, k] * (101/117 + 591/40) * (chp_power[:, k]-40) + 75 - 591/40 * (chp_power[:, k]-40),
                        jnp.where(
                            (chp_power[:, k] > 44) & (chp_power[:, k] <= 110.2),
                            rand[:, k] * (75 + 101/117 * (chp_power[:, k]-40)),
                            jnp.where(
                                (chp_power[:, k] > 110.2) & (chp_power[:, k] <= 125.8),
                                rand[:, k] * (32.4 - 86/13 * (chp_power[:, k]-125.8)),
                                0.0
                            )
                        )
                    )
                    #population = population.at[:,self.dim_P+self.dim_PH+k].set(heat)


                elif k == 4:# CHP unit 18 (第5个)
                    heat = jnp.where(
                        (chp_power[:, 4] >= 10) & (chp_power[:, 4] <= 20),
                        rand[:, 4] * (3/7 + 4) * (chp_power[:, 4]-10) + 40 - 4 * (chp_power[:, 4]-10),
                        jnp.where(
                            (chp_power[:, 4] > 20) & (chp_power[:, 4] <= 45),
                            rand[:, 4] * (40 + 3/7 * (chp_power[:, 4]-10)),
                            jnp.where(
                                (chp_power[:, 4] > 45) & (chp_power[:, 4] <= 60),
                                rand[:, 4] * (-11/3) * (chp_power[:, 4]-60),
                                0.0
                            )
                        )
                    )
                    #population = population.at[:,self.dim_P+self.dim_PH+4].set(heat)

                elif k == 5:# CHP unit 19 (第6个)
                    heat = jnp.where(
                        (chp_power[:, 5] >= 35) & (chp_power[:, 5] <= 90),
                        rand[:, 5] * (20 + 5/11 * (chp_power[:, 5]-35)),
                        jnp.where(
                            (chp_power[:, 5] > 90) & (chp_power[:, 5] <= 105),
                            rand[:, 5] * (-5/3 * (chp_power[:, 5]-105)),
                            0.0
                        )
                    )
                population = population.at[:,self.dim_P+self.dim_PH+k].set(heat)

            # 初始化热电联产机组热力维度的速度
            length_HR = self.HRmax - self.HRmin

            # 确保维度正确
            assert length_HR.shape == (self.dim_PH,), f"HRmax/HRmin维度应为{self.dim_PH}，实际为{length_HR.shape}"
            # # 方案1：使用 jnp.broadcast_to
            # length_HR = jnp.broadcast_to(length_HR, (self.pop_size, self.dim_PH))  # 形状为 (1000, 6)
            
            velocity = velocity.at[:, self.dim_P+self.dim_PH:self.dim_P+2*self.dim_PH].set(
                jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim_PH)) * length_HR * 2 - length_HR
            )


            # 4. 初始化纯热力机组的热力输出（25:30）
            length_H = self.Hmax - self.Hmin
            population = population.at[:, self.dim_P+2*self.dim_PH:].set(
                jax.random.uniform(init_pop_key, shape=(self.pop_size, self.dim_H)) * length_H + self.Hmin
            )
            

            # 初始化纯热力机组热力维度的速度
            velocity = velocity.at[:, self.dim_P+2*self.dim_PH:].set(
                jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim_H)) * length_H * 2 - length_H
            )
            

        return State(
            population=population,
            velocity=velocity,
            local_best_location=population,
            local_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            global_best_location=population[0],
            global_best_fitness=jnp.array([jnp.inf]),
            key=state_key,
        )


    # PSO算法在tell中只根据state.population和state来更新参数
    def ask(self, state):
        return state.population, state



    def tell(self, state, fitness):
        tell_methods = {
                'traditional_eld': self._traditional_eld_tell,
                'chp_24_eld': self._chp_24_eld_tell
            }
            
        tell_method = tell_methods.get(self.problem_type)
            
        if tell_method is None:
            raise ValueError(f"不支持的任务类型: {self.problem_type}")
            
        return tell_method(state, fitness)
           

    # 40台带阈点效应的传统ELD问题
    def _traditional_eld_tell(self, state, fitness):

        def adjust_solution(solution):
        # 循环调整每个解的值，直到它们的和等于 Pd
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


    # 24台混合热电联产的ELD问题
    def _chp_24_eld_tell(self, state, fitness):
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

      

        #population = jax.vmap(self.adjust_power_heat_limitation)(population)
        population = jax.vmap(self.adjust_power_heat_limitation)(population)
        population = jax.vmap(self.adjust_power_balance)(population)
        population = jax.vmap(self.adjust_heat_balance)(population)
        #population = jax.vmap(self.adjust_power_heat_limitation)(population)
        #population = jax.vmap(self.adjust_balance)(population)

        # population = jax.vmap(self.adjust_power_balance)(population)

        return state.replace(
            population=population,
            velocity=velocity,
            local_best_location=local_best_location,
            local_best_fitness=local_best_fitness,
            global_best_location=global_best_location,
            global_best_fitness=global_best_fitness,
            key=key,
        )
    
    
    def update_chp_unit_heat_all(self, power_chp, heat):
        """
        更新热电联产机组的热能输出

        参数:
            power_chp: 热电联产机组的功率输出数组 (dim_PH,)
            heat: 热能输出数组 (dim_PH,)

        返回:
            更新后的热能输出数组
        """
        # 将功率和热能输出转换为 float32 类型
        p = power_chp.astype(jnp.float32)
        h = heat.astype(jnp.float32)

        # 定义机组类型的掩码
        indices = jnp.arange(self.dim_PH)

         # 创建掩码
        mask_14_16 = (indices == 0) | (indices == 2)
        mask_15_17 = (indices == 1) | (indices == 3)
        mask_18 = (indices == 4)
        mask_19 = (indices == 5)
        #mask_default = ~(mask_14_16 | mask_15_17 | mask_18 | mask_19)

         # 扩展掩码以匹配形状
        mask_14_16 = mask_14_16[jnp.newaxis, :]
        mask_15_17 = mask_15_17[jnp.newaxis, :]
        mask_18 = mask_18[jnp.newaxis, :]
        mask_19 = mask_19[jnp.newaxis, :]

        # 初始化 adjusted_heat
        adjusted_heat = jnp.zeros_like(heat)

        # 计算 chp_14_16 的 lower_bound 和 upper_bound
        cond1_14_16 = (p >= 81) & (p <= 98.8)
        cond2_14_16 = (p > 98.8) & (p <= 215)
        cond3_14_16 = (p > 215) & (p <= 247)

        lower_bound_14_16 = jnp.where(cond1_14_16, 104.8 - (524 / 89) * (p - 81), 0.0)
        upper_bound_14_16 = jnp.where(
            cond1_14_16 | cond2_14_16,
            104.8 + (188 / 335) * (p - 81),
            jnp.where(cond3_14_16, - (45 / 8) * (p - 247), 0.0)
        )
        adjusted_14_16 = jnp.clip(h, lower_bound_14_16, upper_bound_14_16)
        adjusted_heat = jnp.where(mask_14_16, adjusted_14_16, adjusted_heat)

        # 计算 chp_15_17 的 lower_bound 和 upper_bound
        cond1_15_17 = (p >= 40) & (p <= 44)
        cond2_15_17 = (p > 44) & (p <= 110.2)
        cond3_15_17 = (p > 110.2) & (p <= 125.8)

        lower_bound_15_17 = jnp.where(cond1_15_17, 75 - (591 / 40) * (p - 40), 0.0)
        upper_bound_15_17 = jnp.where(
            cond1_15_17 | cond2_15_17,
            75 + (101 / 117) * (p - 40),
            jnp.where(cond3_15_17, 32.4 - (86 / 13) * (p - 125.8), 0.0)
        )
        adjusted_15_17 = jnp.clip(h, lower_bound_15_17, upper_bound_15_17)
        adjusted_heat = jnp.where(mask_15_17, adjusted_15_17, adjusted_heat)

        # 计算 chp_18 的 lower_bound 和 upper_bound
        cond1_18 = (p >= 10) & (p <= 20)
        cond2_18 = (p > 20) & (p <= 45)
        cond3_18 = (p > 45) & (p <= 60)

        lower_bound_18 = jnp.where(cond1_18, 40 - 4 * (p - 10), 0.0)
        upper_bound_18 = jnp.where(
            cond1_18 | cond2_18,
            40 + (3 / 7) * (p - 10),
            jnp.where(cond3_18, - (11 / 3) * (p - 60), 0.0)
        )
        adjusted_18 = jnp.clip(h, lower_bound_18, upper_bound_18)
        adjusted_heat = jnp.where(mask_18, adjusted_18, adjusted_heat)
        
        # 计算 chp_19 的 lower_bound 和 upper_bound
        cond1_19 = (p >= 35) & (p <= 90)
        cond2_19 = (p > 90) & (p <= 105)

        lower_bound_19 = 0.0
        upper_bound_19 = jnp.where(
            cond1_19,
            20 + (5 / 11) * (p - 35),
            jnp.where(cond2_19, - (5 / 3) * (p - 105), 0.0)
        )
        adjusted_19 = jnp.clip(h, lower_bound_19, upper_bound_19)
        adjusted_heat = jnp.where(mask_19, adjusted_19, adjusted_heat)
   
        adjusted_heat = jnp.squeeze(adjusted_heat, axis=0)  # 去除第0维

        return adjusted_heat

        
    
    def update_chp_unit_heat_once(self, pi, hi, r):
        """
        调整热能输出基于输入的 r 值。

        参数:
            pi: 功率输出 (标量或数组)
            hi: 热能输出 (标量或数组)
            r: 指定要进行哪种调整的标识符

        返回:
            调整后的热能输出
        """
        # 将 pi 和 hi 转换为 float32 类型
        pi = jnp.array(pi, dtype=jnp.float32)
        hi = jnp.array(hi, dtype=jnp.float32)

        # 定义计算热能输出的辅助函数
        def compute_heat(hi, lower_bound, upper_bound):
            return jnp.clip(hi, lower_bound, upper_bound)

        # 定义各个调整函数
        def chp_14_16():
            cond1 = (pi >= 81) & (pi <= 98.8)
            cond2 = (pi > 98.8) & (pi <= 215)
            cond3 = (pi > 215) & (pi <= 247)

            lower_bound = jnp.where(cond1, 104.8 - (524 / 89) * (pi - 81), 0.0)
            upper_bound = jnp.where(
                cond1 | cond2,
                104.8 + (188 / 335) * (pi - 81),
                jnp.where(cond3, - (45 / 8) * (pi - 247), 0.0)
            )
            return compute_heat(hi, lower_bound, upper_bound)

        def chp_15_17():
            cond1 = (pi >= 40) & (pi <= 44)
            cond2 = (pi > 44) & (pi <= 110.2)
            cond3 = (pi > 110.2) & (pi <= 125.8)

            lower_bound = jnp.where(cond1, 75 - (591 / 40) * (pi - 40), 0.0)
            upper_bound = jnp.where(
                cond1 | cond2,
                75 + (101 / 117) * (pi - 40),
                jnp.where(cond3, 32.4 - (86 / 13) * (pi - 125.8), 0.0)
            )
            return compute_heat(hi, lower_bound, upper_bound)

        def chp_18():
            cond1 = (pi >= 10) & (pi <= 20)
            cond2 = (pi > 20) & (pi <= 45)
            cond3 = (pi > 45) & (pi <= 60)

            lower_bound = jnp.where(cond1, 40 - 4 * (pi - 10), 0.0)
            upper_bound = jnp.where(
                cond1 | cond2,
                40 + (3 / 7) * (pi - 10),
                jnp.where(cond3, - (11 / 3) * (pi - 60), 0.0)
            )
            return compute_heat(hi, lower_bound, upper_bound)

        def chp_19():
            cond1 = (pi >= 35) & (pi <= 90)
            cond2 = (pi > 90) & (pi <= 105)

            lower_bound = 0.0
            upper_bound = jnp.where(
                cond1,
                20 + (5 / 11) * (pi - 35),
                jnp.where(cond2, - (5 / 3) * (pi - 105), 0.0)
            )
            return compute_heat(hi, lower_bound, upper_bound)

        def default_case():
            return hi

        # 根据 r 的值选择对应的调整函数
        adjusted_hi = lax.cond(
            (r == 0) | (r == 2),
            chp_14_16,
            lambda: lax.cond(
                (r == 1) | (r == 3),
                chp_15_17,
                lambda: lax.cond(
                    r == 4,
                    chp_18,
                    lambda: lax.cond(
                        r == 5,
                        chp_19,
                        default_case
                    )
                )
            )
        )

        return adjusted_hi

    
    def adjust_power_heat_limitation(self,solution):

        """对三种不同的机组应用功率和热能约束。"""
        # 定义解数组中不同部分的索引
        power_indices = slice(0, self.dim_P)
        power_chp_indices = slice(self.dim_P, self.dim_P + self.dim_PH)
        power_heat_indices = slice(self.dim_P + self.dim_PH, self.dim_P + 2 * self.dim_PH)
        thermal_indices = slice(self.dim_P + 2 * self.dim_PH, None)

        # 对纯电机组应用功率约束
        adjusted_power = jnp.clip(solution[power_indices], self.Pmin, self.Pmax)

        # 对联合热电机组（电功率部分）应用功率约束
        adjusted_power_chp = jnp.clip(solution[power_chp_indices], self.PRmin, self.PRmax)

        # 对联合热电机组（热功率部分）应用热能约束
        adjusted_power_heat = self.update_chp_unit_heat_all(adjusted_power_chp, solution[power_heat_indices])
        
        # 对纯热机组应用热能约束
        adjusted_thermal = jnp.clip(solution[thermal_indices], self.Hmin, self.Hmax)
         # 合并所有调整后的部分
        adjusted_solution = jnp.concatenate([
            adjusted_power,
            adjusted_power_chp,
            adjusted_power_heat,
            adjusted_thermal
        ])
        return adjusted_solution
    


    def adjust_power_balance(self,solution):
        """调整功率输出以满足功率等式约束"""
        # 定义解数组中功率部分的索引
        power_indices = slice(0, self.dim_P + self.dim_PH)
        power_chp_indices = slice(self.dim_P, self.dim_P + self.dim_PH)
        heat_chp_indices = slice(self.dim_P + self.dim_PH, self.dim_P + 2 * self.dim_PH)
        
        # 计算当前总功率与目标功率的差值
        total_power = solution[power_indices]
        delta_power = jnp.sum(total_power) - self.Pd

        def condition(state):
            delta_power, power = state
            return jnp.abs(delta_power) > 1e-2
        
        # 定义循环体函数
        def body(state):
            delta_power, power = state
            # 计算一个调整量而不是每次都选择使用这个差值进行计算
            adjustment = delta_power / (self.dim_P + self.dim_PH)
            power = power - adjustment

            # 第一部分：dim_P 元素
            power_P = power[:self.dim_P]
            power_P = jnp.clip(power_P, self.Pmin, self.Pmax)
            # 第二部分：dim_PH 元素
            power_PH = power[self.dim_P:]
            power_PH = jnp.clip(power_PH, self.PRmin, self.PRmax)
            # 合并调整后的功率部分
            power = jnp.concatenate([power_P, power_PH])
            # 重新计算 delta_power
            new_delta_power = jnp.sum(power) - self.Pd
            return new_delta_power, power
        
        # 获取初始热能状态
        initial_heat = solution[heat_chp_indices]

        ## 执行循环确保功率平衡
        delta_power, total_power = jax.lax.while_loop(
            condition,
            body,
            (delta_power, total_power)
        )
        ## 将更新后的功率写回解中
        #total_power = solution.at[power_indices].set(total_power)
        solution = solution.at[power_indices].set(total_power)

        ## 更新热能输出
        total_heat = self.update_chp_unit_heat_all(total_power[power_chp_indices], initial_heat)
        solution = solution.at[heat_chp_indices].set(total_heat)

        return solution




    def adjust_heat_balance(self,solution):
        # 定义热能输出部分的索引
        heat_indices = slice(self.dim_P + self.dim_PH, None)
        # 定义热电联产机组热功率部分的索引
        heat_chp_indices = slice(self.dim_P + self.dim_PH, self.dim_P + 2 * self.dim_PH)
        # 定义纯热电机组热功率部分的索引
        heat_thermal_indices = slice(self.dim_P + 2 * self.dim_PH, None)

        ## 先根据之前热电机组的电功率来热功率输出
        power_chp_indices = slice(self.dim_P, self.dim_P + self.dim_PH)
        # 获取热电联产机组的功率输出
        power_chp = solution[power_chp_indices]
        # 获取热电联产机组的热能输出
        heat_chp = solution[heat_chp_indices]
        # 获取所有的热能输出
        total_heat = solution[heat_indices]
   
        delta_heat = jnp.sum(total_heat) - self.Hd
        
         # 热能平衡约束循环
        def heat_condition(state):
            delta_heat, heat = state
            return jnp.abs(delta_heat) > 1e-2
        
        # 定义循环体函数
        def heat_body(state):
            delta_heat, heat = state
            adjustment_heat = delta_heat / (self.dim_PH + self.dim_H)

            # 调整热电联产机组
            heat_chp = heat[:self.dim_PH]
            power_chp = solution[power_chp_indices]
            indices_chp = jnp.arange(self.dim_PH)

            def adjust_chp_unit(p, h, idx):
                new_heat = h - adjustment_heat
                new_heat = self.update_chp_unit_heat_once(p, new_heat, idx)
                return new_heat

            # 向量化调整热电联产机组的热能输出
            heat_chp_updated = jax.vmap(adjust_chp_unit, in_axes=(0, 0, 0))(power_chp, heat_chp, indices_chp)

            # 调整纯热机组
            heat_thermal = heat[self.dim_PH : ]
            heat_thermal_adjusted = heat_thermal - adjustment_heat
            heat_thermal_updated = jnp.clip(heat_thermal_adjusted, self.Hmin, self.Hmax)

            # 合并更新后的热能输出
            heat_updated = jnp.concatenate([heat_chp_updated, heat_thermal_updated])

            # 重新计算 delta_heat
            delta_heat = jnp.sum(heat_updated) - self.Hd

            return delta_heat, heat_updated

        # 初始化循环状态
        state = (delta_heat, total_heat)

        # 执行循环，调整热能输出以满足热能平衡约束
        delta_heat, total_heat = jax.lax.while_loop(
            heat_condition,
            heat_body,
            state
        )

        # 更新 solution 中的热能输出部分
        solution = solution.at[heat_indices].set(total_heat)

        return solution

    


    def adjust_balance(self, solution):
        """调整功率和热能输出以满足功率和热能平衡约束"""
        # 定义功率输出部分的索引
        power_indices = slice(0, self.dim_P + self.dim_PH)
        power_P_indices = slice(0, self.dim_P)
        power_PH_indices = slice(self.dim_P, self.dim_P + self.dim_PH)

        # 定义热能输出部分的索引
        heat_indices = slice(self.dim_P + self.dim_PH, None)
        heat_chp_indices = slice(self.dim_P + self.dim_PH, self.dim_P + 2 * self.dim_PH)
        heat_thermal_indices = slice(self.dim_P + 2 * self.dim_PH, None)

        ## 调整功率平衡约束
        total_power = solution[power_indices]
        delta_power = jnp.sum(total_power) - self.Pd

        # 定义功率平衡约束的条件函数
        def power_condition(state):
            delta_power, power = state
            return jnp.abs(delta_power) > 1e-2

        # 定义功率平衡约束的循环体函数
        def power_body(state):
            delta_power, power = state

            # 计算调整量
            adjustment = delta_power / (self.dim_P + self.dim_PH)
            power = power - adjustment

            # 调整功率单元的输出并进行剪切
            power_P = power[power_P_indices]
            power_P = jnp.clip(power_P, self.Pmin, self.Pmax)

            power_PH = power[power_PH_indices]
            power_PH = jnp.clip(power_PH, self.PRmin, self.PRmax)

            # 合并调整后的功率部分
            power = jnp.concatenate([power_P, power_PH])

            # 重新计算 delta_power
            new_delta_power = jnp.sum(power) - self.Pd

            return new_delta_power, power

        # 执行循环以确保功率平衡
        delta_power, total_power = jax.lax.while_loop(
            power_condition,
            power_body,
            (delta_power, total_power)
        )

        # 将更新后的功率写回解中
        solution = solution.at[power_indices].set(total_power)

        ## 调整热能平衡约束

        # 获取热电联产机组的功率输出
        power_chp = solution[power_PH_indices]

        # 获取热电联产机组的热能输出
        heat = solution[heat_indices]
        heat_chp = heat[:self.dim_PH]

        # 根据功率输出更新热电联产机组的热能输出
        heat_chp_updated = self.update_chp_unit_heat_all(power_chp, heat_chp)

        # 将更新后的热电联产机组热能输出写回解中
        solution = solution.at[heat_chp_indices].set(heat_chp_updated)
        heat = solution[heat_indices]

        # 获取所有的热能输出
        heat_thermal = heat[self.dim_PH:]
        total_heat = jnp.concatenate([heat_chp_updated, heat_thermal])

        delta_heat = jnp.sum(total_heat) - self.Hd

        # 定义热能平衡约束的条件函数
        def heat_condition(state):
            delta_heat, heat = state
            return jnp.abs(delta_heat) > 1e-2

        # 定义热能平衡约束的循环体函数
        def heat_body(state):
            delta_heat, heat = state
            adjustment_heat = delta_heat / (self.dim_PH + self.dim_H)

            # 调整热电联产机组的热能输出
            heat_chp = heat[:self.dim_PH]
            power_chp = solution[power_PH_indices]
            indices_chp = jnp.arange(self.dim_PH)

            def adjust_chp_unit(p, h, idx):
                new_heat = h - adjustment_heat
                new_heat = self.update_chp_unit_heat_once(p, new_heat, idx)
                return new_heat

            # 向量化调整热电联产机组的热能输出
            heat_chp_updated = jax.vmap(adjust_chp_unit, in_axes=(0, 0, 0))(power_chp, heat_chp, indices_chp)

            # 调整纯热机组的热能输出
            heat_thermal = heat[self.dim_PH:]
            heat_thermal_adjusted = heat_thermal - adjustment_heat
            heat_thermal_updated = jnp.clip(heat_thermal_adjusted, self.Hmin, self.Hmax)

            # 合并更新后的热能输出
            heat_updated = jnp.concatenate([heat_chp_updated, heat_thermal_updated])

            # 重新计算 delta_heat
            delta_heat = jnp.sum(heat_updated) - self.Hd

            return delta_heat, heat_updated

        # 初始化循环状态
        state = (delta_heat, total_heat)

        # 执行循环，调整热能输出以满足热能平衡约束
        delta_heat, total_heat = jax.lax.while_loop(
            heat_condition,
            heat_body,
            state
        )

        # 更新 solution 中的热能输出部分
        solution = solution.at[heat_indices].set(total_heat)

        return solution
