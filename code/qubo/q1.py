# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 14:18
@Author : tz
@File : main.py
@Description : 可以考虑使用其他优化器来实现，用的模拟退火，调参调好了，时间13s左右
"""
import time
from pyqubo import Array, Constraint
from dwave.samplers import SimulatedAnnealingSampler

import numpy as np
from pyqubo import Placeholder


# 在字典中找到value的key
def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


# 主函数,问题一的主要求解函数
def main():
    # 定义一些变量
    L = 1000000  # 贷款资金
    I = 0.08  # 利息收入率

    # 读取数据
    data = np.genfromtxt('../data/附件1：data_100.csv', delimiter=',', skip_header=1)

    # 获取T矩阵和H矩阵
    T = data[:, ::2]
    H = data[:, 1::2]

    # 定义二进制决策变量,10 * 100
    x = Array.create('x', shape=(10, 100), vartype='BINARY')

    # 定义惩罚项M
    M = Placeholder('M')
    M = 100000

    # 定义哈密顿量,也就是对应的目标函数
    # 注意这里不是总通过率和总坏账率，是中间变量，将最终决策目标的连加符号放到里面整出来的
    P = np.sum(np.multiply(x, T))
    Q = np.sum(np.multiply(x, H))
    H = - (L * I * P * (1 - Q) - L * P * Q) + M * Constraint((np.sum(x) - 1) ** 2, label='sum(x_i_j) = 1')

    # 编译哈密顿量得到一个模型
    model = H.compile()

    # 将Qubo模型输出为BinaryQuadraticModel，BQM来求解
    bqm = model.to_bqm()

    # 记录开始退火时间
    start = time.time()

    # 模拟退火
    sa = SimulatedAnnealingSampler()
    sampleset = sa.sample(bqm, seed=666, beta_range=[10e-10, 50], num_sweeps=10000, beta_schedule_type='geometric',
                          num_reads=10)


    # 对数据进行筛选，对选取的数据进行选取最优的
    decoded_samples = model.decode_sampleset(sampleset)  # 将上述采样最好的num_reads组数据变为模型可读的样本数据
    best_sample = min(decoded_samples, key=lambda x: x.energy)  # 将能量值最低的样本统计出来，表示BQM的最优解

    end = time.time()
    print(f"退火时间花费：{end - start} s")

    # 统计决策变量为1的所有数据
    data_1_list = get_key(best_sample.sample, 1)

    print(f"对应的取1的决策变量有：{data_1_list}(注意是索引，从0开始),对应的能量为(最大最终收入):{- best_sample.energy}")


if __name__ == '__main__':
    for i in range(100):
        print(f"==============================={i + 1}====================================")
        main()
        print(f"==================================================================")
