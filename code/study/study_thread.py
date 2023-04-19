# -*- coding: UTF-8 -*-
r"""
@time : 2023/4/19 16:41
@Author : tz
@File : study_thread.py
@Description : 
"""

import threading
import random

# 全局变量，用于存储所有线程中的最大值
max_value = None


# 线程函数，生成一组随机数并找到其中的最大值
def find_max():
    global max_value
    values = [random.randint(1, 100) for i in range(10)]
    local_max = max(values)
    if max_value is None or local_max > max_value:
        max_value = local_max


if __name__ == '__main__':
    # 创建三个线程并启动它们
    threads = [threading.Thread(target=find_max) for i in range(3)]
    for thread in threads:
        thread.start()

    # 等待所有线程执行完毕
    for thread in threads:
        thread.join()

    # 输出所有线程中的最大值
    print("Max value: ", max_value)
