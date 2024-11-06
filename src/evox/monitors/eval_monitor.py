import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.sharding import SingleDeviceSharding

from evox import Monitor
from evox.vis_tools import plot

from ..operators import non_dominated_sort


class EvalMonitor(Monitor):
    """评估监视器
    用于单目标和多目标优化工作流程的监控。
    在评估过程中进行监控，可以监控:
    1. 后代个体
    2. 对应的适应度值
    3. 评估计数
    4. 记录最优解或实时更新的帕累托前沿

    参数说明
    ----------
    full_fit_history : bool
        是否记录完整的适应度值历史。默认为True。
        设置为False可以减少内存使用。
    
    full_sol_history : bool
        是否记录完整的解决方案历史。默认为False。
        设置为True会增加内存使用，并增加GPU到CPU的数据传输开销。
    
    topk : int
        仅影响单目标优化。记录的精英解决方案数量。
        默认为1，即只记录最优个体。
    
    calc_pf : bool
        仅影响多目标优化。是否在运行期间持续更新帕累托前沿（存档）。
        默认为False。设置为True将使监视器维护一个无限大小的帕累托前沿，
        可能会影响性能。
    """

    def __init__(
        self, full_fit_history=True, full_sol_history=False, topk=1, calc_pf=False
    ):
        # 初始化监视器的配置参数
        self.full_fit_history = full_fit_history
        self.full_sol_history = full_sol_history
        self.topk = topk
        self.calc_pf = calc_pf
        
        # 初始化存储数据的容器
        self.fitness_history = []  # 存储适应度历史
        self.solution_history = []  # 存储解决方案历史
        self.topk_fitness = None   # 存储前k个最优适应度
        self.topk_solutions = None # 存储前k个最优解决方案
        self.pf_solutions = None   # 存储帕累托前沿的解决方案
        self.pf_fitness = None     # 存储帕累托前沿的适应度
        self.latest_solution = None # 存储最新的解决方案
        self.latest_fitness = None  # 存储最新的适应度
        self.eval_count = 0        # 评估计数器
        self.opt_direction = 1     # 优化方向，默认为最小化问题

    def hooks(self):
        """定义监视器的钩子函数"""
        return ["post_eval"]

    def set_opt_direction(self, opt_direction):
        """设置优化方向（最大化或最小化）"""
        self.opt_direction = opt_direction

    def post_eval(self, _state, cand_sol, _transformed_cand_sol, fitness):
        """评估后的处理函数
        
        处理新评估的候选解和其适应度值，根据问题类型（单目标/多目标）
        调用相应的记录函数
        """
        # 确保监控在单个设备上进行
        monitor_device = SingleDeviceSharding(jax.devices()[0])
        
        # 根据适应度维度判断是单目标还是多目标问题
        if fitness.ndim == 1:  # 单目标问题
            if self.full_sol_history:
                cand_fit = None
            else:
                # 当不记录完整解决方案历史时，只传送topk解决方案到主机以节省带宽
                rank = jnp.argsort(fitness)
                topk_rank = rank[: self.topk]
                cand_sol = cand_sol[topk_rank]
                cand_fit = fitness[topk_rank]

            io_callback(
                self.record_fit_single_obj,
                None,
                cand_sol,
                cand_fit,
                fitness,
                sharding=monitor_device,
            )
        else:  # 多目标问题
            io_callback(
                self.record_fit_multi_obj,
                None,
                cand_sol,
                fitness,
                sharding=monitor_device,
            )

    def record_fit_single_obj(self, cand_sol, cand_fit, fitness):
        """记录单目标优化的适应度和解决方案
        
        更新历史记录并维护topk最优解
        """
        if cand_fit is None:
            cand_fit = fitness

        # 更新历史记录
        if self.full_sol_history:
            self.solution_history.append(cand_sol)
        if self.full_fit_history:
            self.fitness_history.append(fitness)

        # 处理topk=1的特殊情况（性能优化）
        if self.topk == 1:
            current_min_fit = jnp.min(cand_fit, keepdims=True)
            if self.topk_fitness is None or self.topk_fitness > current_min_fit:
                self.topk_fitness = current_min_fit
                individual_index = jnp.argmin(cand_fit)
                self.topk_solutions = cand_sol[individual_index : individual_index + 1]
        else:
            # 处理topk>1的一般情况
            if self.topk_fitness is None:
                self.topk_fitness = cand_fit
                self.topk_solutions = cand_sol
            else:
                # 合并并更新topk解决方案
                self.topk_fitness = jnp.concatenate([self.topk_fitness, cand_fit])
                self.topk_solutions = jnp.concatenate(
                    [self.topk_solutions, cand_sol], axis=0
                )
                # 选择前k个最优解
                rank = jnp.argsort(self.topk_fitness)
                topk_rank = rank[: self.topk]
                self.topk_solutions = self.topk_solutions[topk_rank]
                self.topk_fitness = self.topk_fitness[topk_rank]

    def record_fit_multi_obj(self, cand_sol, fitness):
        """记录多目标优化的适应度和解决方案
        
        更新历史记录并维护帕累托前沿
        """
        # 更新历史记录
        if self.full_sol_history:
            self.solution_history.append(cand_sol)
        if self.full_fit_history:
            self.fitness_history.append(fitness)

        # 更新帕累托前沿
        if self.calc_pf:
            if self.pf_fitness is None:
                self.pf_fitness = fitness
                self.pf_solutions = cand_sol
            else:
                # 合并新的解决方案到现有帕累托前沿
                self.pf_fitness = jnp.concatenate([self.pf_fitness, fitness], axis=0)
                self.pf_solutions = jnp.concatenate(
                    [self.pf_solutions, cand_sol], axis=0
                )
            
            # 使用非支配排序更新帕累托前沿
            rank = non_dominated_sort(self.pf_fitness)
            pf = rank == 0  # 选择等级为0的解（非支配解）
            self.pf_fitness = self.pf_fitness[pf]
            self.pf_solutions = self.pf_solutions[pf]

        # 更新最新的解决方案和适应度
        self.latest_fitness = fitness
        self.latest_solution = cand_sol

    # 以下是各种获取监控数据的接口方法
    def get_latest_fitness(self):
        """获取最新的适应度值（考虑优化方向）"""
        return self.opt_direction * self.latest_fitness

    def get_latest_solution(self):
        """获取最新的解决方案"""
        return self.latest_solution

    def get_pf_fitness(self):
        """获取帕累托前沿的适应度值（考虑优化方向）"""
        return self.opt_direction * self.pf_fitness

    def get_pf_solutions(self):
        """获取帕累托前沿的解决方案"""
        return self.pf_solutions

    def get_topk_fitness(self):
        """获取topk的适应度值（考虑优化方向）"""
        return self.opt_direction * self.topk_fitness

    def get_topk_solutions(self):
        """获取topk的解决方案"""
        return self.topk_solutions

    def get_best_solution(self):
        """获取最优解决方案"""
        return self.topk_solutions[0]

    def get_best_fitness(self):
        """获取最优适应度值（考虑优化方向）"""
        return self.opt_direction * self.topk_fitness[0]

    def get_history(self):
        """获取适应度历史记录（考虑优化方向）"""
        return [self.opt_direction * fit for fit in self.fitness_history]

    def plot(self, problem_pf=None, **kwargs):
        """绘制优化过程的可视化图表
        
        根据目标数量选择适当的可视化方法：
        - 1D：单目标优化过程图
        - 2D：二维目标空间图
        - 3D：三维目标空间图
        """
        if not self.fitness_history:
            warnings.warn("No fitness history recorded, return None")
            return

        # 确定目标维度
        if self.fitness_history[0].ndim == 1:
            n_objs = 1
        else:
            n_objs = self.fitness_history[0].shape[1]

        # 根据目标维度选择合适的绘图方法
        if n_objs == 1:
            return plot.plot_obj_space_1d(self.get_history(), **kwargs)
        elif n_objs == 2:
            return plot.plot_obj_space_2d(self.get_history(), problem_pf, **kwargs)
        elif n_objs == 3:
            return plot.plot_obj_space_3d(self.get_history(), problem_pf, **kwargs)
        else:
            warnings.warn("Not supported yet.")

    def flush(self):
        """确保所有JAX操作都已完成"""
        jax.effects_barrier()

    def close(self):
        """关闭监视器，确保所有操作都已完成"""
        self.flush()
