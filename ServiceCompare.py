import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

latency_data = pd.read_excel('Evaluation.xlsx', sheet_name='Sheet1')

latency_data = latency_data.iloc[1:6, 1:7]
print(latency_data)


def ou_distance(x, y):
    # 定义欧式距离的计算
    x = x[1:-1]
    y = y[1:-1]
    return np.sqrt(sum(np.square(x - y)))


def request_sum(request, deployment_edge):
    """

    :param request: 所有ES对该模块的请求
    :param deployment_edge: 放置了该模块的ES
    :return: 该模块的所有请求频率

    等下这样子做 不对的 原因是 不一定是给最近的ES 是看整个应用结构的 所以该模块从属的应用网络延迟加起来就好了

    """


def moduleReuqest(r1, r2, r3, r4, r5, r6):
    return [r1, r2, r3, r4, r5, r6, r1, r2, r3, r4, r5, r6, r1 + r4 + r5, r1 + r2 + r3 + r4 + r6, r1 + r5, r2 + r3 + r6,
            r2 + r4, r5, r6, r3]


# 生成服务模块请求
latency_data[
    ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
     's18', 's19', 's20']] = latency_data.apply(
    lambda x: pd.Series(moduleReuqest(x.a1, x.a2, x.a3, x.a4, x.a5, x.a6, )), axis=1)

latency_data = latency_data.iloc[:, 6:26]
print(latency_data)
# 保存 保存到已用的excel中 需要多一步
wb = openpyxl.load_workbook('Evaluation.xlsx')
write = pd.ExcelWriter('Evaluation.xlsx', engine='openpyxl')
write.book = wb  # 没有这句话会覆盖

latency_data.to_excel(write, sheet_name='service_latency', index=False)
write.save()
write.close()
