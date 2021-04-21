'''
    随机生成部署方案 作为对比实验
    1.完全随机
    2.知道冗余部署情况后， 再随机 √ 只能是这个吧

    四个对比算法 生成部署方案 并保存会后续作比较

'''
import pandas as pd
import openpyxl
import random
import numpy as np

from Service import Service
from Application import Application
from User import User
from EdgeServer import Edge
from Node import Node
from queue import Queue
import copy
from DAG import DAG

budget = 2000

edge_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='edge_data1')

# 且不能超过其能力
capacity = edge_data['capacity'].to_list()
edge_list = []
for i in range(40):
    temp = edge_data.loc[i].values[0:-1]
    temp_edge = Edge(no=i + 1, latitude=temp[1], longitude=temp[2], capacity=temp[3])
    edge_list.append(temp_edge)

a0 = Application(0, [])
a1 = Application(1, ['1', '13', '14', '15', '7'])
a2 = Application(2, ['2', '14', '16', '17', '8'])
a3 = Application(3, ['3', '14', '16', '20', '9'])
a4 = Application(4, ['4', '13', '14', '17', '10'])
a5 = Application(5, ['5', '13', '15', '18', '11'])
a6 = Application(6, ['6', '14', '16', '19', '12'])

app_list = [a0, a1, a2, a3, a4, a5, a6]

user_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='user_data2')
user_list = []
for i in range(len(user_data)):
    temp = user_data.loc[i].values[0:11]
    temp_user = User(i + 1, temp[1], temp[2], temp[3], temp[5:])
    user_list.append(temp_user)

redundancy = [4, 6, 4, 6, 6, 4, 4, 6, 4, 6, 6, 4, 7, 9, 5, 7, 7, 6, 4, 4]
capacity = [8, 8, 6, 7, 6, 5, 5, 8, 7, 7, 7, 6, 7, 5, 6, 8, 6, 7, 8, 7, 5, 6, 8, 7, 7, 7, 7, 6, 7, 7, 8, 6, 7, 6, 7, 6,
            6, 5, 7, 7]
service_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='rank_result')
price = service_data['price'].to_list()
edge_location = edge_data[['index', 'LATITUDE', 'LONGITUDE']].values


def random_plan():
    plan = [[] for i in range(40)]

    for i in range(1, 21):
        j = redundancy[i - 1]
        for k in range(j):
            # 开始生成j个随机数
            tmp = random.randint(0, 39)
            # print("要把" + str(i) + "放到服务器", tmp)
            while len(plan[tmp]) > capacity[tmp]:
                # 超过了所以要变tmp
                # print("超过了")
                tmp = random.randint(0, 39)
            plan[tmp].append(i)

    return plan


def avg_random_plan():
    redundancy2 = []
    total = sum(price)
    n = budget / total
    for i in range(20):
        redundancy2.append(n)

    plan = [[] for i in range(40)]

    for i in range(1, 21):
        for k in range(n):
            # 开始生成j个随机数
            tmp = random.randint(0, 39)
            # print("要把" + str(i) + "放到服务器", tmp)
            while len(plan[tmp]) > capacity[tmp]:
                # 超过了所以要变tmp
                # print("超过了")
                tmp = random.randint(0, 39)
            plan[tmp].append(i)

    return plan


def ou_distance(x, y):
    # 定义欧式距离的计算
    x = x[1:-1]
    y = y[1:-1]
    return np.sqrt(sum(np.square(x - y)))


def nearest_spare_edge(index, plan, capacity):
    """
    为当前边缘服务器找到最近空闲服务器
    :param index: 当前ES编号
    :param plan: 目前的部署方案
    :param capacity: 每个ES的容量
    :param edge_location: 每个ES的地理信息位置 二维列表
    :return: 最近的空闲ES编号
    """
    # 首先按照距离近远进行排序 sort sorted cmp func真的把人整吐了 直接换最笨的
    distance = []
    print("进入判断最近空闲ES函数中")
    for i in range(len(edge_location)):
        x = edge_location[index]
        y = edge_location[i]
        distance.append(ou_distance(x, y))

    edge_location2 = np.c_[edge_location, distance]

    edge_location3 = edge_location2[edge_location2[:, 3].argsort()]

    res = -1
    for i in range(len(edge_location3)):
        mark = int(edge_location3[i, 0])
        if len(plan[mark]) < capacity[mark]:
            print(len(plan[mark]))
            print(capacity[mark])
            res = int(edge_location3[mark, 0])
            break

    print("找到离" + str(index) + "最近的ES是")
    print(res)
    return res


def request_first_plan():
    # 根据冗余部署模块 和每个ES对于该模块请求频率最大的上面
    # 根据edge_data
    plan = [[] for i in range(40)]
    for i in range(1, 21):
        need = redundancy[i - 1]
        # 找到排名前need个且空闲的ES进行部署
        cur = 's' + str(i)
        data = edge_data[['index', cur]].values
        index = data[:, 1].argsort()
        index = index[::-1]
        cnt, j = 0, 0
        while cnt < need:
            if len(plan[index[j]]) < capacity[index[j]]:
                plan[index[j]].append(i)
                cnt += 1
            j += 1
    return plan


def latency_first_plan():
    plan = [[] for i in range(40)]
    # 先建立距离矩阵
    dis = np.zeros([40, 40], dtype=float)
    for i in range(0, 40):
        for j in range(i, 40):
            temp = ou_distance(edge_location[i], edge_location[j])
            dis[i][j] = dis[j][i] = temp

    for i in range(1, 21):
        need = redundancy[i - 1]
        # 开始创建该模块的请求频率矩阵
        column = 's' + str(i)
        df = edge_data[column]
        request = np.array(df).reshape(40, 1)
        res = np.dot(dis, request)
        res = np.array(res).flatten()
        # 选择排名前need个的位置进行存在

        edge_location2 = np.c_[edge_location, res]
        # edge_location3 = edge_location2[edge_location2[:, 3].argsort()]
        # print(edge_location3)
        index = edge_location2[:, 3].argsort()

        cnt, j = 0, 0
        # cnt遍历要放的个数 j遍历排名的ES
        while cnt < need:
            if len(plan[index[j]]) < capacity[index[j]]:
                plan[index[j]].append(i)
                cnt += 1
            j += 1
    # print(plan)
    return plan


def load_balance_plan():
    plan = [[] for i in range(40)]
    # 那不就是按照顺序 1 2 3 4 的放进去
    service = []
    for i in range(20):
        tmp = redundancy[i]
        for j in range(tmp):
            service.append(i + 1)
    print(service)
    index = 0
    j = 0
    while index < len(service):
        plan[j].append(service[index])
        j += 1
        j %= 40
        index += 1
    return plan


def ga_plan():
    # 通过遗传算法获得的最优个体 是一维的变成二维Plan
    ga = [16, 16, 17, -1, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, 4, 7, 10, 14, 16, 17, 2, -1, -1, -1, -1, -1,
          -1,
          11, 14, 15, 18, -1, -1, -1, -1, -1, -1, -1, 4, 9, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1,
          -1,
          -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 6, 7, 9, 11, 13, 16, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
          -1, 2,
          8, -1, -1, -1, 1, 4, 7, 10, 13, 14, 5, -1, -1, -1, -1, -1, -1, -1, 1, 3, 11, 14, 18, -1, 12, 15, -1, -1, -1,
          -1,
          -1, 16, 17, -1, -1, -1, -1, -1, -1, 1, 4, 6, 7, 10, -1, -1, 5, 13, -1, -1, -1, 2, 4, 8, 10, 13, 15, -1, -1,
          -1,
          -1, -1, -1, -1, -1, 2, 12, 15, 16, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, 19, 20, -1, -1, -1, -1, -1, 3, 6,
          12,
          19, -1, -1, -1, 2, 8, 17, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 13, 14, 16, 18, -1, -1, 17, -1, -1, -1,
          -1,
          -1, -1, -1, 1, 3, 6, 8, 12, 19, 3, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, 5, 9, 11, 13, 18, -1, -1, 4,
          10,
          14, 17, -1, -1, 14, 20, -1, -1, -1, -1, 15, 18, -1, -1, -1, 5, 11, 13, 18, -1, -1, -1, 2, 8, 14, 17, -1, -1,
          -1]
    plan = []
    idx = 0
    for i in range(len(capacity)):
        tmp = ga[idx:idx+capacity[i]]
        plan.append(tmp)
        idx = idx+capacity[i]
    return plan

def translate(plan):
    for i in range(len(plan)):
        if len(plan[i]) < capacity[i]:
            # 没填满就填上-1
            for j in range(capacity[i] - len(plan[i])):
                plan[i].append(-1)
    return plan


if __name__ == '__main__':
    # print("随机算法")
    # plan = translate(random_plan())
    # print(plan)
    # print(len(plan))
    # compute = DAG(plan=plan, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # res = compute.run(user_list=user_list)
    # print("该方案总延迟为", res)
    #
    # print("随机平均算法")
    # plan = translate(random_plan())
    # print(plan)
    # print(len(plan))
    # compute = DAG(plan=plan, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # res = compute.run(user_list=user_list)
    # print("该方案总延迟为", res)
    #
    # print("最大请求频率算法")
    # plan = translate(request_first_plan())
    # print(plan)
    # print(len(plan))
    # compute = DAG(plan=plan, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # res = compute.run(user_list=user_list)
    # print("该方案总延迟为", res)
    #
    # print("最低延迟算法")
    # plan = translate((latency_first_plan()))
    # print(plan)
    # print(len(plan))
    # compute = DAG(plan=plan, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # res = compute.run(user_list=user_list)
    # print("该方案总延迟为", res)
    # print("负载均衡算法")
    # plan = translate((load_balance_plan()))
    # print(plan)
    # print(len(plan))
    # compute = DAG(plan=plan, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # res = compute.run(user_list=user_list)
    # print("该方案总延迟为", res)

    plan = ga_plan()
    print(plan)

# 开始计算总的网络延迟
