# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/14 9:44
@Author : tz
@File : q3.py
@Description : 
"""
import time
import pandas as pd
from threading import Thread

# 定义一些变量
L = 1000000  # 贷款资金
I = 0.08  # 利息收入率

df = pd.read_csv('../data/附件1：data_100.csv', header=0)

# 全局变量，用于存储所有线程中的最大值
max_value = -1
# 定义最大数字的索引下标
max_i = max_j = max_k = max_m = max_n = max_l = -1


def computer_result(begin, end):
    # 使用全局变量
    global max_value, max_i, max_j, max_k, max_m, max_n, max_l

    for i in range(begin, end):
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

                            if temp_value > max_value:
                                max_value, max_i, max_j, max_k, max_m, max_n, max_l = temp_value, i, j, k, m, n, l


if __name__ == '__main__':

    start = time.time()
    print(start)

    # 创建线程并启动它们
    threads = [Thread(target=computer_result, args=(i, i + 1)) for i in range(1, 101)]
    for thread in threads:
        thread.start()

    # 等待所有线程执行完毕
    for thread in threads:
        thread.join()

    end = time.time()
    print(f"时间花费：{end - start} s")
    print(f"最大值：{max_value},对应的选取卡1：第{max_i}张{max_m + 1}个阈值,"
          f"对应的选取卡2：第{max_j}张{max_n + 1}个阈值,"
          f"对应的选取卡3：第{max_k}张{max_l + 1}个阈值,其中，卡取值[1-100],阈值取值[1-10]")
