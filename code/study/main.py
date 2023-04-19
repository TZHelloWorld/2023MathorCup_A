# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/18 21:40
@Author : tz
@File : 1.py
@Description : 
"""
from pyqubo import Binary
import neal

# 定义哈密顿量
x1, x2, x3, x4 = Binary("x1"), Binary("x2"), Binary("x3"), Binary("x4")
H = -5 * x1 - 3 * x2 - 8 * x3 - 6 * x4 + 4 * x1 * x2 + 8 * x1 * x3 + 2 * x2 * x3 + 10 * x3 * x4

# 编译哈密顿量得到一个模型
model = H.compile()

# 调用'to_qubo()'获取QUBO系数
# 其中，offset表示下面目标函数中的常数值
# qubo是系数，字典dict类型,{('x2', 'x2'): -3.0,...}表示其系数
qubo, offset = model.to_qubo()

print(qubo)
print(offset)

# 求解qubo模型
# 将Qubo模型输出为BinaryQuadraticModel，BQM来求解
bqm = model.to_bqm()

# 定义采样器，这里使用模拟退火采样器
sa = neal.SimulatedAnnealingSampler()

# 进行采样，由于该采样器机制，需要设置高一点的采样个数，这样能够确保能够获得最优值
# 对应的设置越高，时间花费越久。
sampleset = sa.sample(bqm, num_reads=10)
decoded_samples = model.decode_sampleset(sampleset)

# 将采样出来的结果按照目标函数进行排序，求出最佳（小）的采样结果
best_sample = min(decoded_samples, key=lambda x: x.energy)

# 输出采样结果
print(f"采样取值：{best_sample.sample}，对应的结果为:{best_sample.energy}")
