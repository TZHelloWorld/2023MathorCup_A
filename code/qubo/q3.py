# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/16 15:38
@Author : tz
@File : q3_2.py
@Description : 使用贪心+qubo模型
"""

import pandas as pd
import time
import re

from pyqubo import Array, Constraint, Binary
# from neal import SimulatedAnnealingSampler  # 使用pyqubo里面的采样器
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np
from pyqubo import Placeholder


# 在字典中找到value的key
def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def main():
    # 定义一些变量
    L = 1000000  # 贷款资金
    I = 0.08  # 利息收入率

    # 迭代次数
    Iter = 10

    # 选取第几列的信用卡,初始化最开始选择的数据
    card1, threshold1, card2, threshold2, card3, threshold3 = 1, 0, 2, 0, 3, 0

    # 定义最大的数
    max_value = -1
    # 读取数据
    df = pd.read_csv('../data/附件1：data_100.csv', header=0)

    for iter in range(Iter):  # 迭代次数，其实这里还可以根据几次迭代算出的总值来提前结束迭代
        # 设置更新迭代,不使用临时变量
        card1, threshold1, card2, threshold2, card3, threshold3 = card2, threshold2, card3, threshold3, card1, threshold1

        # 获取第一张卡和第二张卡的通过率和坏账率
        t1 = df[f"t_{card1}"].iloc[threshold1]
        t2 = df[f"t_{card2}"].iloc[threshold2]
        h1 = df[f"h_{card1}"].iloc[threshold1]
        h2 = df[f"h_{card2}"].iloc[threshold2]

        # 获取第三张卡能够选取的通过率与坏账率的矩阵

        choose_index = [i for i in range(1, 101) if i not in (card1, card2)]
        card3_data = df.drop([f"t_{card1}", f"h_{card1}", f"t_{card2}", f"h_{card2}"], axis=1).values

        # 获取T矩阵和H矩阵
        T = card3_data[:, ::2]
        H = card3_data[:, 1::2]

        # 定义二进制决策变量,10 * 98
        x = Array.create('x', shape=(10, 98), vartype='BINARY')
        # 定义惩罚项M
        M = Placeholder('M')
        M = 50000

        # 定义哈密顿量,也就是对应的目标函数
        P = t1 * t2 * np.sum(np.multiply(x, T))
        Q = (h1 + h2 + np.sum(np.multiply(x, H))) / 3
        H = - (L * I * P * (1 - Q) - L * P * Q) + M * Constraint((np.sum(x) - 1) ** 2, label='sum(x_i_j) = 1')

        # 编译哈密顿量得到一个模型
        model = H.compile()
        bqm = model.to_bqm()
        # 记录开始退火时间
        start = time.time()

        #
        sa = SimulatedAnnealingSampler()

        sampleset = sa.sample(bqm, seed=666, beta_range=[10e-10, 50], num_sweeps=999, beta_schedule_type='geometric',
                              num_reads=50)

        # 对数据进行筛选，对选取的数据进行选取最优的
        decoded_samples = model.decode_sampleset(sampleset)  # 将上述采样最好的num_reads组数据变为模型可读的样本数据
        best_sample = min(decoded_samples, key=lambda x: x.energy)  # 将能量值最低的样本统计出来，表示BQM的最优解

        end = time.time()
        # print(f"退火时间花费：{end - start} s")

        # 统计决策变量为1的所有数据,并对第一个数据进行拆解，获得对应的下标
        data_1_list = get_key(best_sample.sample, 1)
        index_i_j = re.findall(r'\d+', data_1_list[0])
        index_i = int(index_i_j[0])

        if max_value < - best_sample.energy:
            card3 = choose_index[int(index_i_j[1])]
            threshold3 = index_i
            max_value = - best_sample.energy

        print(
            f"第{iter + 1}次迭代，最大值：{max_value},卡选择:{card1}_{threshold1 + 1},{card2}_{threshold2 + 1},{card3}_{threshold3 + 1}")

    print(f"最大值：{max_value},卡选择:{card1}_{threshold1 + 1},{card2}_{threshold2 + 1},{card3}_{threshold3 + 1}")


if __name__ == '__main__':
    for i in range(100):
        print(f"==============================={i + 1}====================================")
        start = time.time()
        main()
        end = time.time()
        print(f"时间花费： {end - start}s")
