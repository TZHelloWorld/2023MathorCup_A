# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 9:48
@Author : tz
@File : q1.py
@Description : 贪心+ 暴力求解第三题
"""
import pandas as pd

if __name__ == '__main__':
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

        # 从固定的信用卡1和信用卡2外数据中选一个，构建可以选择的卡的集合
        choose_list = {i for i in range(1, 101) if i not in (card1, card2)}

        # 获取第一张卡和第二张卡的通过率和坏账率，否则在循环里重复获取了
        t1 = df[f"t_{card1}"].iloc[threshold1]
        t2 = df[f"t_{card2}"].iloc[threshold2]
        h1 = df[f"h_{card1}"].iloc[threshold1]
        h2 = df[f"h_{card2}"].iloc[threshold2]

        # 固定第一张卡,和第二张卡，然后找到最佳的第三张卡
        for c3 in choose_list:
            for th3 in range(0, 10):
                # 总通过率
                A = t1 * t2 * df[f"t_{c3}"].iloc[th3]

                # 总坏账率
                B = (h1 + h2 + df[f"h_{c3}"].iloc[th3]) / 3

                # 此时的最终收入
                temp_value = L * I * A * (1 - B) - L * A * B

                if temp_value > max_value:
                    max_value = temp_value
                    card3 = c3
                    threshold3 = th3

    print(f"最大值：{max_value},卡选择:{card1}_{threshold1 + 1},{card2}_{threshold2 + 1},{card3}_{threshold3 + 1}")
