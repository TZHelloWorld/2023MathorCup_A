# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 14:18
@Author : tz
@File : main.py
@Description : 
"""

from pyqubo import Binary
import neal

# 定义哈密顿量
x1, x2, x3, x4, x5 = Binary("x1"), Binary("x2"), Binary("x3"), Binary("x4"), Binary("x5")
# H = -5 * x1 - 3 * x2 - 8 * x3 - 6 * x4 + 4 * x1 * x2 + 8 * x1 * x3 + 2 * x2 * x3 + 10 * x3 * x4

H = x1 * x2 * x3 * x4 * x5
# 编译哈密顿量得到一个模型
model = H.compile()

# 调用'to_qubo()'获取QUBO系数
# 其中，offset表示下面目标函数中的常数值
# qubo是系数，字典dict类型,{('x2', 'x2'): -3.0,...}表示其系数
qubo, offset = model.to_qubo()

# print(qubo)
# print(offset)

# 求解qubo模型
# 将Qubo模型输出为BinaryQuadraticModel，BQM来求解
bqm = model.to_bqm()
print(bqm)
print(len(bqm))

sa = neal.SimulatedAnnealingSampler()  # 定义采样器
sampleset = sa.sample(bqm, num_reads=10)  # , num_reads=10
decoded_samples = model.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)

print(best_sample.sample)

# 在使用惩罚项系数的时候，可以使用Placeholder占位符，先占用一个位置，然后设置的时候可以不需要重新编译（.compile）
