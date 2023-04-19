安装pyqubo：`pip install pyqubo` 教程参考:[GitHub](https://github.com/recruit-communications/pyqubo) ,[Getting Started — pyqubo 1.0.5 documentation](https://pyqubo.readthedocs.io/en/latest/getting_started.html)，[pdf](https://pyqubo.readthedocs.io/_/downloads/en/latest/pdf/)

安装dwave： `pip install dwave-samplers`，教程参考： [GitHub - dwavesystems/dwave-samplers：求解二元二次模型的经典算法](https://github.com/dwavesystems/dwave-samplers)



使用说明：使用`pyqubo`将`哈密顿量`转换成qubo类型，或者说BinaryQuadraticModel（BQM）类型来求解

BQM的python类型是：`class <'dimod.binary.binary_quadratic_model.BinaryQuadraticModel'>`

转换成BQM类型后直接使用`dwave`中的一系列采样器来求解该BQM模型，并获得结果



dwave采样器有很多种，每种都具有一定参数设置，可以考虑找到最优参数去调整。
