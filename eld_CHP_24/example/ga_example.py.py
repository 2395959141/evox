
#~~~~~~~~~~~导入相关库~~~~~~~~~~~~~~~~~~~·
import numpy as np
from tqdm import tqdm#进度条设置
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib; matplotlib.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import time


class GaPLoad(object):
    def __init__(self,PD,PH):
        self.PD=PD #单个时刻的负荷
        self.PH=PH #单个时刻的热力负荷

        #~~~~~~~~~~~~初始化参数~~~~~~~~~~~~·
        self.NP = 500  # 种群数量
        self.L = 20  # 二进制数串长度（选择合适长度，不是越长越好）
        self.Pc = 0.5  # 交叉率
        self.Pm = 0.1  # 变异率
        self.G = 100  # 最大遗传代数
        self.N=6#变量个数X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热

    #~~~~~~~~~~~~~目标函数值~~~~~~~~~~~~~~~~~~~~·
    def calc_f(self, X):
        """
        :param X: 十进制格式 X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热
        :return: 总的成本
        """
        TC = 50 * X[0] + 2650 + 14.5 * X[1] + 0.0345 * X[2] * X[2] + 1250 + 36 * X[2] + 0.0435 * X[2] * X[2] + \
             4.2 * X[3] + 0.03 * X[3] * X[3] + 0.031 * X[1] * X[3] + \
             0.6 * X[4] + 0.027 * X[4] * X[4] + 0.011 * X[2] * X[4] + \
             23.4 * X[5]
        return TC

    #~~~~~~~~~~~~~平衡约束惩罚项1~~~~~~~~~~~~~~~~~~~~~·
    def calc_e1(self, X):
        """
            :param X:  十十进制格式 X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热
            :return: 功率平衡约束,热力平衡约束
        """
        dP = self.PD - X[0] - X[1] - X[2]  # 负荷-机组1出力-机组2出力-机组3出力#功率平衡约束
        dP1 = self.PH - X[3] - X[4] - X[5]  # 热力负荷-机组2出热-机组3出热-机组4出热 #热力平衡约束
        return np.abs(dP) + np.abs(dP1)

    #~~~~~~~~~~~~~不等式惩罚项2~~~~~~~~~~~~~~~~~~~~~·
    def calc_e2(self, X):
        """
        :param X:  十十进制格式 X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热
        :return:
        """
        ee = 0
        """机组2发电区间"""
        PG2max = (11115 - 8 * X[3]) / 45  # 机组2P出力上限
        PG2min = np.max([(-2886120 + 134 * X[3]) / 75.2, (10354.24 - 17.8 * X[3]) / 104.8])  # 机组2P出力下限
        if  PG2min > PG2max:  # 如果机组2出力上下限约束条件违法
            ee += np.abs(PG2min - PG2max)
        """机组2发热区间"""
        HG2max = np.min([(11115 - 45 * X[1]) / 8, np.abs((7952 - 75.2 * X[1]) / 134)])  # 机组2出热最大值
        HG2min = 0  # 机组2出热最小值
        if HG2max < HG2min:
            ee += np.abs(HG2max - HG2min)
        """机组3发电区间"""
        PG3max = np.min([(125.8), (13488 - 15.6 * X[4]) / 103.2])  # 机组3出力上限
        PG3min = np.max([(-2841 + 70.2 * X[4]) / 60.6, (2664 - 4 * X[4]) / 59.1, 44])  # 机组3出力下限
        if PG3min < 0 or PG3max < 0 or PG3min > PG3max:  # 如果机组3出力上下限约束条件违法
            ee += np.abs(PG3min - PG3max)
        """机组3发热区间"""
        HG3max = np.min([(13488 - 103.2 * X[2]) / 15.6, (2841 - 60.6 * X[2]) / 70.2])  # 机组3出热上限
        HG3min = 0  # 机组3出热下限
        if HG3max < HG3min:
            ee += np.abs(HG3max - HG3min)
        return np.abs(ee)

    #~~~~~二进制转换为十进制的函数~~~~~·
    def two_to_ten(self, f):
        """
        :param f: 二进制编码 群体或个体
        :return: 实数编码的群体
        """
        x = np.zeros((f.shape[0], self.N))  # 存放种群的实数形式

        for i in range(f.shape[0]):  # 遍历每一个个体
            U = f[i, :]  # 当前个体 shape=(f.shape[0],L)

            for j in range(self.N):  # 遍历每一个变量
                m = 0
                for k in range(self.L):  # 遍历每一个二进制元素
                    m = U[j, k] * np.power(2, k) + m  # np.power(2,j)求2的j次方
                if j == 0:  # 机组1出力 （0-150）随机取
                    x[i, j] = 0 + m * (150 - 0) / (
                                np.power(2, self.L) - 1)  # 已经将二进制编码转换到机组1出力十进制之间的
                if j == 1:  # 机组2出力（81-247）随机取
                    x[i, j] = 81 + m * (247 -81) / (
                                np.power(2, self.L) - 1)  # 已经将二进制编码转换到机组2出力十进制之间的
                if j == 2:  # 机组3出力（40-125.8）随机取
                    x[i, j] = 40 + m * (125.8 - 40) / (
                                np.power(2, self.L) - 1)  # 已经将二进制编码转换到机组3出力十进制之间的

                if j == 3:#机组2产热（0-180）随机取
                    x[i, j] = 0 + m * (180 - 0) / (
                            np.power(2, self.L) - 1)  # 已经将二进制编码转换到机组2出热十进制之间的
                if j==4:#机组3产热（0-135.6）随机取
                    x[i, j] = 0 + m * (135.6 - 0) / (
                            np.power(2, self.L) - 1)  # 已经将二进制编码转换到机组3出热十进制之间的
                if j == 5:  # 机组4产热（0-2695.2）随机取
                    x[i, j] = 0 + m * (2695.2 - 0) / (
                            np.power(2, self.L) - 1)  # 已经将二进制编码转换到机组4出热十进制之间的
        return x

    #~~~~~~~~~~~~~遗传操作方法~~~~~~~~~~~~~~~~~~~
    def select(self, f, fitness):
        """
        :param f: 二进制群体
        :param fitness: 适应度
        :return:选择后的种群 二进制种群
        """
        fitness = np.array(fitness)
        fitness = fitness.reshape(-1, )
        """根据轮盘赌法选择优秀个体"""
        fitness = 1 / fitness  # fitness越小表示越优秀，被选中的概率越大，做 1/fitness 处理
        fitness = fitness / fitness.sum()  # 归一化
        idx = np.array(list(range(f.shape[0])))
        f2_idx = np.random.choice(idx, size=f.shape[0], p=fitness)  # 根据概率选择
        f2 = f[f2_idx, :]
        return f2

    def crossover(self, f, c):
        """
        :param f: 二进制群体
        :param c: 交叉概率
        :return:交叉后的二进制种群
        """
        """按顺序选择2个个体以概率c进行交叉操作"""
        for i in range(0, self.NP, 2):  # 遍历群体个体
            p = np.random.random()  # 生成一个0-1之间的随机数
            if p < c:
                q = np.random.randint(0, 2, (1, self.L))  # 生成一个长度为L的01数组 shape(1,20)
                for j in range(self.L):  # 遍历个体每一位元素
                    if q[:, j] == 1:
                        for k in range(self.N):  # 遍历每一个机组
                            temp = np.int(f[i + 1, k, j])  # 下一个个体(i+1) 的第j个元素
                            f[i + 1, k, j] = f[i, k, j]
                            f[i, k, j] = temp
        return f

    def mutation(self, f, m):
        """
        :param f: 二进制种群
        :param m: 变异概率
        :return: 变异后的群体 （二进制）
        """
        """变异操作"""
        for i in range(np.int(np.round(self.NP * m))):  # 指定变异个数
            h = np.random.randint(0, self.NP, 1)[0]  # 随机选择一个（0-NP）之间的整数
            for j in range(int(np.round(self.L * m))):  # 指定变异元素个数
                g = np.random.randint(0, self.L, 1)[0]  # 随机选择一个(0-L）之间的整数
                for k in range(self.N):  # 遍历每一个变量
                    f[h, k, g] = np.abs(1 - f[h, k, g])  # 将该元素取反
        return f

    #~~~~~~~~~~~~~~~子代和父辈之间的选择操作~~~~~~~~~~~~~~~~~~~~~~·
    def update_best(self, parent, parent_fitness, parent_e, child, child_fitness, child_e):
        """
            判
            :param parent: 父辈个体
            :param parent_fitness:父辈适应度值
            :param parent_e    ：父辈惩罚项
            :param child:  子代个体
            :param child_fitness 子代适应度值
            :param child_e  ：子代惩罚项

            :return: 父辈 和子代中较优者、适应度、惩罚项
            #合理设置惩罚项能接受的阈值。影响效果。可以自己手动调参,设置合理阈值
            """

        # 规则1，如果 parent 和 child 都没有违反约束，则取适应度小的
        if parent_e <= 5 and child_e <= 5:
            if parent_fitness <= child_fitness:
                return parent, parent_fitness, parent_e
            else:
                return child, child_fitness, child_e
        # 规则2，如果child违反约束而parent没有违反约束，则取parent
        if parent_e < 5 and child_e >= 5:
            return parent, parent_fitness, parent_e
        # 规则3，如果parent违反约束而child没有违反约束，则取child
        if parent_e >= 5 and child_e < 5:
            return child, child_fitness, child_e
        # 规则4，如果两个都违反约束，则取适应度值小的
        if parent_fitness <= child_fitness:
            return parent, parent_fitness, parent_e
        else:
            return child, child_fitness, child_e

    #~~~~~~~~~~~~~~~主函数~~~~~~~~~~~~~~~~~~~~~~~·
    def main(self):
        """
        :return: 最优解（十进制），目标函数值，惩罚项1，惩罚项2
        """
        parent = np.random.randint(0, 2, (self.NP, self.N, self.L))  # 随机获得二进制 初始种群f.shape (50,6, 20) .6表示有6个变量
        best_x = []  # 存放最优二进制
        fitneess_value_list = []  # 每一步迭代的最优解
        for i in tqdm(range(self.G)):  # 遍历每一次迭代
            fitness = np.zeros((self.NP, 1))  # 存放更新产生的适应度值
            ee = np.zeros((self.NP, 1))  # 存放更新产生的惩罚项值
            parentfit = np.zeros((self.NP, 1))  # 存放父辈适应度值
            parentee = np.zeros((self.NP, 1))  # 存放父辈惩罚项值
            parentten = self.two_to_ten(parent)  # 转换为十进制

            for j in range(self.NP):  # 遍历每一个群体
                parentfit[j] = self.calc_f(parentten[j])  # 目标函数值
                parentee[j] = self.calc_e1(parentten[j])+self.calc_e2(parentten[j])#惩罚项
            parentfitness = parentfit + parentee  # 计算父辈适应度值   适应度值=目标函数值+惩罚项

            X2 = self.select(parent, parentfitness)  # 选择 X2为二进制形式
            X3 = self.crossover(X2, self.Pc)  # 交叉
            X4 = self.mutation(X3, self.Pm)  # 变异 (子代二进制）

            childten = self.two_to_ten(X4)  # 子代转换为十进制
            childfit = np.zeros((self.NP, 1))  # 子代目标函数值
            childee = np.zeros((self.NP, 1))  # 子代惩罚项

            for j in range(self.NP):  # 遍历每一个群体
                childfit[j] = self.calc_f(childten[j])  # 子代目标函数值
                childee[j] = self.calc_e1(childten[j]) +self.calc_e2(childten[j]) # 子代惩罚项
            childfitness = childfit + childee  # 子代适应度值

            #~~~~~~~~~~更新群体~~~~~~~~~~~~~~·
            for j in range(self.NP):  # 遍历每一个个体
                X4[j], fitness[j], ee[j] = self.update_best(parent[j], parentfitness[j], parentee[j], X4[j],
                                                            childfitness[j],
                                                            childee[j])
            fitneess_value_list.append(fitness.min())
            x = X4[fitness.argmin()]  # 最优二进制

            best_x.append(x)
            parent = X4

        #~~~~~~~~~~··多次迭代后的最终效果~~~~~~~~~~~~~~~·
        best_x_two = best_x[-1]  # 最优二进制
        best_x_two = best_x_two.reshape(-1, self.N, self.L)
        best_xten = self.two_to_ten(best_x_two)  # 转换为十进制格式
        best_xten = best_xten.reshape(-1, 1)

        return best_xten, self.calc_f(best_xten), self.calc_e1(best_xten), self.calc_e2(best_xten)


if __name__=='__main__':
    PD=200
    PH=115
    #X = [0, 160, 40, 40, 75, 0] 文献上的解，成本9257元
    ga = GaPLoad(PD=PD,PH=PH)
    best_xten, fit, ee1, ee2 =ga.main()

    #print('最优十进制解',best_xten)
    print('机组1出力',best_xten[0])
    print('机组2出力', best_xten[1])
    print('机组3出力', best_xten[2])
    print('机组2出热', best_xten[3])
    print('机组3出热', best_xten[4])
    print('机组4出热', best_xten[5])
    print('最优目标函数',fit)
    print('最优惩罚项1：',ee1)
    print('最优惩罚项2：',ee2)


