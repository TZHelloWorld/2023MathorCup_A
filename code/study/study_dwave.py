# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 19:12
@Author : tz
@File : study_dwave.py
@Description : 
"""

from pyqubo import Binary

import dimod
from dwave.samplers import SimulatedAnnealingSampler

if __name__ == '__main__':
    # 定义哈密顿量
    x1, x2, x3, x4 = Binary("x1"), Binary("x2"), Binary("x3"), Binary("x4")
    H = -5 * x1 - 3 * x2 - 8 * x3 - 6 * x4 + 4 * x1 * x2 + 8 * x1 * x3 + 2 * x2 * x3 + 10 * x3 * x4

    # 编译哈密顿量得到一个模型
    model = H.compile()

    # 求解模型
    # 将Qubo模型输出为BinaryQuadraticModel，BQM来求解
    bqm = model.to_bqm()

    # 创建采样器，类似于取穷举所有决策变量的取值
    sampler = SimulatedAnnealingSampler()

    # 使用默认温度计划的模拟退火进行采样
    sampleset = sampler.sample(bqm)

    # 定制一个采样器，各种参数设置，具体的可以去看源码里面的设置
    # beta_range 指定了温度参数的范围
    # beta_schedule_type 指定了温度参数的调度方式，这里使用的是线性调度方式
    # num_reads 指定了采样器运行的次数,也就是生成多少个样本
    # 该方法返回一个样本集合，
    sampleset = sampler.sample(bqm, beta_range=[.1, 4.2], beta_schedule_type='linear', num_reads=30)

    # 采样器的格式是：x1 x2 x3 x4 energy num_oc. xi表示bqm模型中的取值，energy表示能量，num_oc不知道是啥

    # sampleset = sampler.sample(bqm, beta_range=[.1, 4.2], beta_schedule_type='linear')

    print(sampleset)

    # 对sampleset进行筛选，因为要转换回原来的格式类型，当然也可以自己写相关的东西

