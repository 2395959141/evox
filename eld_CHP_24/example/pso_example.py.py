
#~~~~~~~导入相关库·~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from tqdm import tqdm#进度条设置
import matplotlib.pyplot as plt
from pylab import *
from tqdm import tqdm#进度条设置
import matplotlib; matplotlib.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


class LiPLoad(object):
    def __init__(self,PD,PH):
        self.PD=PD #单个时刻的负荷
        self.PH=PH#单个时刻的热能
        ####################初始化参数#####################
        self.size = 500  # 种群数量
        self.dim =6 #变量个数 X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热
        self.G = 400  # 最大迭代代数
        self.w = 1  # 惯性因子，一般取1
        self.c1 = 2  # 学习因子，一般取2
        self.c2 = 2  #学习因子，一般取2
        self.max_vel = 0.5  # 限制粒子的最大速度为0.5

    #~~~~~~~~~~目标函数值~~~~~~~~~~~~~~~
    def calc_f(self, X):
        """
        :param X: 十进制格式 X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热
        :return: 总的成本
        """
        TC =50*X[0]+2650+14.5*X[1]+0.0345*X[2]*X[2]+1250+36*X[2]+0.0435*X[2]*X[2]+\
            4.2*X[3]+0.03*X[3]*X[3]+0.031*X[1]*X[3]+\
            0.6*X[4]+0.027*X[4]*X[4]+0.011*X[2]*X[4]+\
            23.4*X[5]
        return TC

    # ~~~~~~~~~~~~~平衡约束惩罚项1~~~~~~~~~~~~~~~~~~~~~·
    def calc_e1(self, X):
        """
            :param X:  十十进制格式 X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热
            :return: 功率平衡约束热力平衡约束
        """

        dP = self.PD -X[0]-X[1]-X[2]  #负荷-机组1出力-机组2出力-机组3出力
        dP1 = self.PH - X[3] - X[4] - X[5]  # 热力负荷-机组2出热-机组3出热-机组4出热 #热力平衡约束

        return np.abs(dP) + np.abs(dP1)


    def calc_e2(self, X):
        """
        :param X:  十十进制格式 X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热
        :return:
        """
        ee = 0
        """机组2发电区间"""
        PG2max = (11115 - 8 * X[3]) / 45  # 机组2P出力上限
        PG2min = np.max([(-2886120 + 134 * X[3]) / 75.2, (10354.24 - 17.8 * X[3]) / 104.8])  # 机组2P出力下限
        if PG2min<0 or PG2max<0 or PG2min > PG2max:  # 如果机组2出力上下限约束条件违法
            ee += np.abs(PG2min - PG2max)
        """机组2发热区间"""
        HG2max = np.min([(11115 - 45 * X[1]) / 8, np.abs((7952 - 75.2 * X[1]) / 134)])  # 机组2出热最大值
        HG2min = 0  # 机组2出热最小值
        if  HG2max < HG2min:
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

    #~~~~~~~~~~~~~速度更新~~~~~~~~~~~~~~~~~·
    def velocity_update(self,V, X, pbest, gbest):
        """
        根据速度更新公式更新每个粒子的速度
         种群size=50
        :param V: 粒子当前的速度矩阵，50*6 的矩阵
        :param X: 粒子当前的位置矩阵，50*6 的矩阵
        :param pbest: 每个粒子历史最优位置，50*6 的矩阵
        :param gbest: 种群历史最优位置，1*6 的矩阵
        """
        r1 = np.random.random((self.size, 1))
        r2 = np.random.random((self.size, 1))
        V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)  # 直接对照公式写就好了
        # 防止越界处理
        V[V < -self.max_vel] = -self.max_vel
        V[V > self.max_vel] = self.max_vel
        return V

    #~~~~~~~~~~~~~位置更新~~~~~~~~~~~~~~~~~~~~
    def position_update(self,X, V):
        """
        根据公式更新粒子的位置
        X[0] 机组1出力，X[1] 机组2出力 ;X[2]机组3出力,X[3]机组2产热，x[4]机组3产热，X[5]机组4产热
        :param X: 粒子当前的位置矩阵，维度是 50*6
        :param V: 粒子当前的速度举着，维度是 50*6
        """
        X = X + V  # 更新位置
        size = np.shape(X)[0]  # 种群大小
        for i in range(size):  # 遍历每一个例子
            """机组1出力"""
            if X[i][0] <0 or X[i][0] >= 150:  #机组1出力的上下限约束
                X[i][0] = np.random.uniform(0, 150, 1)[0]  # 则在0到150随机生成一个数
            """机组2"""
            PG2max=(11115-8*X[i][3])/45   #机组2出力上限
            PG2min=np.max([(-2886120+134*X[i][3])/75.2,(10354.24-17.8*X[i][3])/104.8])#机组2P出力下限
            if PG2min<0 or  PG2max<0 or PG2min>PG2max:    #如果机组2出力上下限约束条件违法
                X[i][3]=np.random.uniform(0, 180, 1)[0]
                PG2max = (11115 - 8 * X[i][3]) / 45  #机组2P出力上限
                PG2min = np.max([(-2886120 + 134 * X[i][3]) / 75.2, (10354.24 - 17.8 * X[i][3]) / 104.8])
            if  X[i][1] < PG2min or X[i][1] >= PG2max:  # 机组2出力的上下限约束
                X[i][1] = np.random.uniform(PG2min, PG2max, 1)[0]
            """机组2出热"""
            HG2max=np.min([(11115-45*X[i][1])/8,np.abs((7952-75.2*X[i][1])/134)])#机组2出热最大值
            HG2min=0 #机组2出热最小值
            if HG2max<HG2min:
                X[i][1]= np.random.uniform(PG2min, PG2max, 1)[0]
            if X[i][3] < HG2min or X[i][3] >= HG2max:  # 机组2出热的上下限约束
                X[i][3]=np.random.uniform(HG2min, HG2max, 1)[0]

            """机组3出力"""
            PG3max=np.min([(125.8),(13488-15.6*X[i][4])/103.2])#机组3出力上限
            PG3min=np.max([(-2841+70.2*X[i][4])/60.6,(2664-4*X[i][4])/59.1,44])#机组3出力下限
            if X[i][2] < PG3min or X[i][2] >= PG3max:  # 机组3出力的上下限约束
                X[i][2] = np.random.uniform(PG3min, PG3max, 1)[0]

            """机组3出热"""
            HG3max=np.min([(13488-103.2*X[i][2])/15.6,(2841-60.6*X[i][2])/70.2])#机组3出热上限
            HG3min=0#机组3出热下限
            if X[i][4] < HG3min or X[i][4] >= HG3max:  # 机组3出热的上下限约束
                X[i][4] = np.random.uniform(HG3min, HG3max, 1)[0]
            if X[i][5] < 0 or X[i][5] >=2695.2:  # 机组4出热的上下限约束
                X[i][5] = np.random.uniform(0, 2695.2, 1)[0]
        return X

    #~~~~~~~~~~~~~~~更新粒子的历史最优位置~~~~~~~~~~~~~~~~~~~~~~~·
    def update_pbest(self,pbest, pbest_fitness, pbest_e, xi, xi_fitness, xi_e):
        """
        判断是否需要更新粒子的历史最优位置
        :param pbest: 历史最优位置
        :param pbest_fitness: 历史最优位置对应的适应度值
        :param pbest_e: 历史最优位置对应的约束惩罚项
        :param xi: 当前位置
        :param xi_fitness: 当前位置的适应度函数值
        :param xi_e: 当前位置的约束惩罚项
        :return:
        """
        # 下面的 0.1 是考虑到计算机的数值精度位置，值等同于0  这个自己设置合理的阈值
        # 规则1，如果 pbest 和 xi 都没有违反约束，则取适应度小的
        if pbest_e <= 5 and xi_e <= 5:
            if pbest_fitness <= xi_fitness:
                return pbest, pbest_fitness, pbest_e
            else:
                return xi, xi_fitness, xi_e
        # 规则2，如果当前位置违反约束而历史最优没有违反约束，则取历史最优
        if pbest_e < 5 and xi_e >= 5:
            return pbest, pbest_fitness, pbest_e
        # 规则3，如果历史位置违反约束而当前位置没有违反约束，则取当前位置
        if pbest_e >= 5 and xi_e < 5:
            return xi, xi_fitness, xi_e
        # 规则4，如果两个都违反约束，则取适应度值小的
        if pbest_fitness <= xi_fitness:
            return pbest, pbest_fitness, pbest_e
        else:
            return xi, xi_fitness, xi_e

    #~~~~~~~主函数~~~~~~~~~~~~~~~~~·
    def main1(self):
        """
        :return:
        # 初始化一个矩阵 info, 记录：
        # 0、种群每个粒子的历史最优位置对应的适应度，
        # 1、历史最优位置对应的惩罚项，
        # 2、当前适应度，
        # 3、当前目标函数值，
        # 4、约束1惩罚项，
        # 5、约束2惩罚项，
        # 6、惩罚项的和
        # 所以列的维度是7
        """
        info = np.zeros((self.size, 7))

        #~~~~~初始化种群的各个粒子的位置~~~~~~~~~~~~~~~~·
        # 用一个 NP*N 的矩阵表示种群，每行表示一个粒子
        X = np.random.uniform(10, 1000, size=(self.size, self.dim))

        #~~~~~~初始化种群的各个粒子的速度~~~~~~~~~~~~~~~~·
        V = np.random.uniform(-0.5, 0.5, size=(self.size, self.dim))
        fitneess_value_list = []  # 记录每次迭代过程中的种群适应度值变化

        #~~~~~初始化粒子历史最优位置为当当前位置~~~~~~~~~~~~~~~~~·
        pbest = X
        #~~~~计算每个粒子的适应度~~~~~·
        for i in range(self.size):
            info[i, 3] = self.calc_f(X[i])  # 目标函数值
            info[i, 4] = self.calc_e1(X[i])  # 第一个约束的惩罚项
            info[i, 5] = self.calc_e2(X[i])  # 第二个约束的惩罚项
        info[:, 2] = info[:, 3] + info[:, 4] +  info[:, 5]  # 适应度值
        info[:, 6] =  info[:, 4] +  info[:, 5]  # 惩罚项的加权求和

        #~~~~~历史最优~~~~~~·
        info[:, 0] = info[:, 2]  # 粒子的历史最优位置对应的适应度值
        info[:, 1] = info[:, 6]  # 粒子的历史最优位置对应的惩罚项值

        #~~~~~~~全局最优~~~~~~~~~~~·
        gbest_i = info[:, 0].argmin()  # 全局最优对应的粒子编号
        gbest = X[gbest_i]  # 全局最优粒子的位置
        gbest_fitness = info[gbest_i, 0]  # 全局最优位置对应的适应度值
        gbest_e = info[gbest_i, 1]  # 全局最优位置对应的惩罚项

        #~~~~~~记录迭代过程的最优适应度值~~~~~~~~~·
        fitneess_value_list.append(gbest_fitness)


        #~~~~~~接下来开始迭代~~~~~~~~·
        for j in tqdm(range(self.G)):
            #~~~更新速度~~~
            V = self.velocity_update(V, X, pbest=pbest, gbest=gbest)
            #~~~更新位置~~~
            X = self.position_update(X, V)

            #~~~计算每个粒子的目标函数和约束惩罚项~~~
            for i in range(self.size):
                info[i, 3] = self.calc_f(X[i])  # 目标函数值
                info[i, 4] = self.calc_e1(X[i])  # 第一个约束的惩罚项
                info[i, 5] = self.calc_e2(X[i])  # 第二个约束的惩罚项

            info[:, 2] = info[:, 3] + info[:, 4] + info[:, 5]  # 适应度值
            info[:, 6] =  info[:, 4] + info[:, 5]  # 惩罚项的加权求和

            #~~~更新历史最优位置~~~
            for i in range(self.size):
                pbesti = pbest[i]
                pbest_fitness = info[i, 0]
                pbest_e = info[i, 1]
                xi = X[i]
                xi_fitness = info[i, 2]
                xi_e = info[i, 6]

                #~~~计算更新个体历史最优~~~
                pbesti, pbest_fitness, pbest_e = \
                    self.update_pbest(pbesti, pbest_fitness, pbest_e, xi, xi_fitness, xi_e)
                pbest[i] = pbesti
                info[i, 0] = pbest_fitness
                info[i, 1] = pbest_e

            #~~~更新全局最优位置~~~·
            for i in range(self.size):
                pbesti=pbest[i]
                pbest_fitness = info[i, 0]
                pbest_e = info[i, 1]
                gbest, gbest_fitness, gbest_e = \
                    self.update_pbest(gbest, gbest_fitness, gbest_e,pbesti, pbest_fitness, pbest_e)

            #~~~记录当前迭代全局之硬度~~~
            fitneess_value_list.append(gbest_fitness)

        #~~~最后绘制适应度值曲线~~~·
        print('迭代最优结果是：%.5f' % self.calc_f(gbest))
        print('迭代最优解',gbest)
        print('惩罚项1',self.calc_e1(gbest))
        print('惩罚项2',self.calc_e2(gbest))

        #~~~~~~绘图~~~~~~·
        plt.plot(fitneess_value_list[: 30], color='r')
        plt.title('迭代过程')
        plt.show()

if __name__=="__main__":
    liziqun=LiPLoad(200,115)
    liziqun.main1()

