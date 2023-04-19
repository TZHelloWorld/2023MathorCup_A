# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 9:48
@Author : tz
@File : q1.py
@Description : 
"""
import time

import pandas as pd

if __name__ == '__main__':
    # 定义一些变量
    L = 1000000  # 贷款资金
    I = 0.08  # 利息收入率

    df = pd.read_csv('./data/附件1：data_100.csv', header=0)

    # 定义最大的数
    max_value = -1

    # 定义最大数字的索引下标
    max_i = max_j = -1

    start = time.time()

    for i in range(1, 101):
        for j in range(0, 10):

            # 总通过率
            P = df[f"t_{i}"].iloc[j]

            # 总坏账率
            Q = df[f"h_{i}"].iloc[j]

            # 此时的最终收入
            temp_value = L * I * P * (1 - Q) - L * P * Q

            # print(f"选卡[{i},{j + 1}]，对应最终收入{temp_value},最大收入:{max_value}")

            if temp_value > max_value:
                max_value = temp_value
                max_i = i
                max_j = j

    end = time.time()
    print(f"暴力时间：{end - start} s")

    print(f"最大值：{max_value}")
    print(f"对应的选取卡1：第{max_i}张{max_j + 1}个阈值")
