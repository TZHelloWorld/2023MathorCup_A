# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 9:44
@Author : tz
@File : q3.py
@Description : 一个for循环暴力破解
"""

import pandas as pd

if __name__ == '__main__':
    # 定义一些变量
    L = 1000000  # 贷款资金
    I = 0.08  # 利息收入率

    df = pd.read_csv('./data/附件1：data_100.csv', header=0)

    # # 根据下标获取对应的T矩阵和H矩阵
    # df_T = df.iloc[:, [i % 2 == 0 for i in range(len(df.columns))]]
    # df_H = df.iloc[:, [i % 2 == 1 for i in range(len(df.columns))]]
    # count = 0

    # 定义最大的数
    max_value = -1

    # 定义最大数字的索引下标
    max_i = max_j = max_k = max_m = max_n = max_l = -1

    for i in range(1, 101):
        for j in range(i + 1, 101):
            for k in range(j + 1, 101):
                for m in range(0, 10):
                    for n in range(0, 10):
                        for l in range(0, 10):
                            # 总通过率
                            A = df[f"t_{i}"].iloc[m] * df[f"t_{j}"].iloc[n] * df[f"t_{k}"].iloc[l]

                            # 总坏账率
                            B = (df[f"h_{i}"].iloc[m] + df[f"h_{j}"].iloc[n] + df[f"h_{k}"].iloc[l]) / 3

                            # 此时的最终收入
                            temp_value = L * I * A * (1 - B) - L * A * B

                            print(
                                f"卡1[{i},{m}]，卡2[{j},{n}]，卡3[{k},{l}]，最终收入{temp_value};卡1[{max_i},{max_m}]，卡2[{max_j},{max_n}]，卡3[{max_k},{max_l}]，最大收入:{max_value}")
                            if temp_value > max_value:
                                max_value = temp_value
                                max_i = i
                                max_j = j
                                max_k = k
                                max_m = m
                                max_n = n
                                max_l = l

    print(f"最大值：{max_value}")
    print(f"对应的选取卡1：第{max_i}张{max_m + 1}个阈值")
    print(f"对应的选取卡1：第{max_j}张{max_n + 1}个阈值")
    print(f"对应的选取卡1：第{max_k}张{max_l + 1}个阈值")
    print("其中，卡取值[1-100],阈值取值[1-10]")
