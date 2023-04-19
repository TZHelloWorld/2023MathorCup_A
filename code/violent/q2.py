# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 9:48
@Author : tz
@File : q1.py
@Description : 
"""
import pandas as pd

if __name__ == '__main__':
    # 定义一些变量
    L = 1000000  # 贷款资金
    I = 0.08  # 利息收入率

    # 选取第几列的信用卡
    card1 = 1
    card2 = 2
    card3 = 3

    df = pd.read_csv('../data/附件1：data_100.csv', header=0)
    # 定义最大的数
    max_value = -1

    # 定义最大数字的索引下标
    max_i = max_j = max_k = -1

    for i in range(0, 10):
        for j in range(0, 10):
            for k in range(0, 10):
                # 总通过率
                P = df[f"t_{card1}"].iloc[i] * df[f"t_{card2}"].iloc[j] * df[f"t_{card3}"].iloc[k]

                # 总坏账率
                Q = (df[f"h_{card1}"].iloc[i] + df[f"h_{card2}"].iloc[j] + df[f"h_{card3}"].iloc[k]) / 3

                # 此时的最终收入
                temp_value = L * I * P * (1 - Q) - L * P * Q

                # print(f"选卡1阈值：[{i + 1}]，选卡1阈值：[{j + 1}]，选卡1阈值：[{k + 1}]，对应最终收入{temp_value},之前的最大收入:{max_value}")

                if temp_value > max_value:
                    max_value = temp_value
                    max_i = i
                    max_j = j
                    max_k = k

        # print(f"第{i + 1}个对应的最大值：{max_value},三个阈值选取为:{max_i + 1},{max_j + 1},{max_k + 1},其中，阈值编号[1-10]")
        print(f"[{max_i + 1},{max_j + 1},{max_k + 1}]-->")


    print(f"最大值：{max_value},三个阈值选取为:{max_i + 1},{max_j + 1},{max_k + 1},其中，阈值编号[1-10]")
