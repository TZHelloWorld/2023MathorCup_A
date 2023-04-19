# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 14:18
@Author : tz
@File : main.py
@Description :可以考虑使用其他优化器来实现，用的模拟退火，调参调好了，时间60s左右
该方法使用的是创建x[10][10][10]来进行决策，其中x[i][j][k]表示是否采取：第一张卡选取第i个阈值，第二张卡选取第j个阈值，第三张卡选取第k个阈值，
"""
import time

from pyqubo import Array, Constraint
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

    # 表示选取的卡号
    card1, card2, card3 = 1, 2, 3

    # 读取数据
    data = np.genfromtxt('../data/附件1：data_100.csv', delimiter=',', skip_header=1)

    # 获取T矩阵和H矩阵
    T = data[:, ::2]
    H = data[:, 1::2]

    # 定义二进制决策变量,10 * 100
    x = Array.create('x', shape=(10, 10, 10), vartype='BINARY')

    # 定义惩罚项M
    M = Placeholder('M')
    M = 30000

    # 定义哈密顿量,也就是对应的目标函数
    # 注意这里不是总通过率和总坏账率，是中间变量，将最终决策目标的连加符号放到里面整出来的
    T1 = T[:, card1 - 1]
    T2 = T[:, card2 - 1]
    T3 = T[:, card3 - 1]

    H1 = H[:, card1 - 1]
    H2 = H[:, card2 - 1]
    H3 = H[:, card3 - 1]

    # 计算三种组合后的总通过率和总坏账率,通过广播机制来实现
    T = T1[:, None, None] * T2[None, :, None] * T3[None, None, :]
    H = (H1[:, None, None] + H2[None, :, None] + H3[None, None, :]) / 3

    P = np.sum(np.multiply(x, T))
    Q = np.sum(np.multiply(x, H))
    H = - (L * I * P * (1 - Q) - L * P * Q) + M * Constraint((np.sum(x) - 1) ** 2, label='sum(x_i_j) = 1')

    # 编译哈密顿量得到一个模型
    model = H.compile()

    # 将Qubo模型输出为BinaryQuadraticModel，BQM来求解
    bqm = model.to_bqm()

    print("开始模拟退火")

    start = time.time()
    sa = SimulatedAnnealingSampler()
    sampleset = sa.sample(bqm, seed=888, beta_range=[10e-12, 60], beta_schedule_type='geometric', num_reads=50)

    # 对数据进行筛选
    decoded_samples = model.decode_sampleset(sampleset)  # 将上述采样最好的num_reads组数据变为模型可读的样本数据
    best_sample = min(decoded_samples, key=lambda x: x.energy)  # 将能量值最低的样本统计出来，表示BQM的最优解

    # print(f"验证约束条件M：{best_sample.constraints()}")
    end = time.time()
    print(f"退火时间：{end - start} s")

    # print(best_sample.sample)

    # 统计决策变量为1的所有数据
    data_1_list = get_key(best_sample.sample, 1)

    print(f"对应的取1的决策变量有：{data_1_list}(表示[第一张卡的阈值][第二张卡的阈值][第三张卡的阈值]，注意是索引，从0开始),对应的能量为{-best_sample.energy}")


if __name__ == '__main__':
    main()
# for i in range(100):
#     print(f"==============================={i + 1}====================================")
#     main()
#     print(f"==================================================================")
