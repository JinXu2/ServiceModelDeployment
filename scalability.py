"""
    新增对比实验：修改用户和边缘服务器的大小 设置总体的算法流程
"""
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
from sklearn.cluster import AffinityPropagation

'''
设定原始数据
'''
# 设置应用列表
a0 = Application(0, [])
a1 = Application(1, ['1', '13', '14', '15', '7'])
a2 = Application(2, ['2', '14', '16', '17', '8'])
a3 = Application(3, ['3', '14', '16', '20', '9'])
a4 = Application(4, ['4', '13', '14', '17', '10'])
a5 = Application(5, ['5', '13', '15', '18', '11'])
a6 = Application(6, ['6', '14', '16', '19', '12'])
app_list = [a0, a1, a2, a3, a4, a5, a6]

# 根据U设置用户列表
U = 273
user_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='user_data2')
user_data = user_data.sample(n=U, random_state=None, replace=True)
user_list = []
for i in range(U):
    temp = user_data.loc[i].values[0:10]
    temp_user = User(i + 1, temp[1], temp[2], temp[3], temp[4:])
    user_list.append(temp_user)

# 根据E设置服务器列表
E = 40
edge_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='edge_data1')
edge_data = edge_data.sample(n=E, random_state=None, replace=True)
edge_list = []
for i in range(40):
    temp = edge_data.loc[i].values[0:-1]
    temp_edge = Edge(no=i + 1, latitude=temp[1], longitude=temp[2], capacity=temp[3])
    edge_list.append(temp_edge)

budget = 2000  # 不变的
price = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 50, 30, 20, 20, 40, 20, 20, 15]
capacity = edge_data['capacity'].to_list()  # 这个可以直接获取
edge_location = edge_data[['index', 'LATITUDE', 'LONGITUDE']].values

'''
对用户请求进行 模块总和 和聚类计算 离散度计算 获取每个模块的优先级
'''
cluster_result = []
sd_result = []
request_sum = []
# AP聚类、请求频率总和 和离散度计算
for i in range(1, 21):
    print("当前处理模块", i)
    coloumn = 's' + str(i)
    # 读取对应列数的数据
    data = user_data[['Latitude', 'Longitude', coloumn]]
    mean = data[coloumn].mean()
    # 请求总和
    request_sum.append(data[coloumn].sum())

    print("平均值为", mean)
    # 对数据进行筛选 只有超过 平均值 才参与聚类 否则不参与
    selected_data = data[(data[coloumn] > mean)]
    selected_num = len(selected_data)
    print("筛选后数据共有", selected_num)
    data = data[(data[coloumn] <= mean)]

    # 筛选后数据可能为 0
    if selected_num == 0:
        sd_result.append(-1)
        cluster_result.append(0)
        print("modeul " + coloumn + "无需聚类,筛选后无数据")
        continue
    else:
        # 标准差
        avg_x = selected_data['Latitude'].mean()
        avg_y = selected_data['Longitude'].mean()

        sum = 0
        for j in range(len(selected_data)):
            sum += pow(abs(selected_data.iloc[j][0] - avg_x) + abs(selected_data.iloc[j][1] - avg_y), 2)

        sd_result.append(pow(sum / (selected_num - 1), 0.5))

        # 进行AP聚类
        weight = selected_data[coloumn]
        selected_data = selected_data[['Latitude', 'Longitude']].values
        '''
        这个参数总算是调对了！
        '''
        p = [x / 10000 * -0.085 for x in weight]
        ap = AffinityPropagation(damping=0.5, max_iter=5000, convergence_iter=30, preference=p).fit(selected_data)

        # 聚类结果
        cluster_centers_indices = ap.cluster_centers_indices_
        if len(cluster_centers_indices) == 0:
            cluster_result.append(0)
            print("module " + coloumn + "无法收敛")
            continue
        else:
            cluster_result.append(len(cluster_centers_indices))

# 优先级计算

# 设定公式 优先级等于前面三者的总和
weight = []
for i in range(20):
    weight.append(cluster_result[i] + request_sum[i] / 1000 * 1.1 + (0 if (sd_result[i] == -1) else sd_result[i] * 100))

# 根据优先级计算 每个模块的rank值 通过建立dataframe来实现

# index列
index = []
for i in range(1, 21):
    index.append("s" + str(i))
c = {'service': index, 'cluster_num': cluster_result, 'request_sum': request_sum, 'sta_dev': sd_result,
     'weight': weight, 'price': price}

rank_result = pd.DataFrame(c)
rank_result['rank'] = rank_result['weight'].rank(method="first", ascending=True)

'''
根据模块优先级 分配部署成本 获取模块冗余个数
'''

df = rank_result[['service', 'cluster_num', 'rank', 'price']]


# 分配部署成本
def allocation(df):
    """

    :param df: 数据集 包括任务编号，聚类个数，优先级，部署价格
    :return: redundant[]列表 表示每个模块冗余部署的个数
    """

    n = 4  # 默认冗余部署个数

    # 先按照优先级进行排序
    res = df.sort_values(by='rank', ascending=False)
    # 转换为列表
    DC = res['price'].to_list()
    rank = res['rank'].to_list()
    rank_sum = sum(rank)
    ap = res['cluster_num'].to_list()

    redundant = []
    func = lambda x, y: x * y

    for i in range(20):

        if i == 0:
            pre_sum = 0
        else:
            pre = map(func, DC[0:i], redundant[0:i])
            pre_sum = sum(list(pre))

        # 当前剩余预算
        # print("之前花了多少钱", pre_sum)

        SSB = budget - pre_sum - sum(DC[i:]) * n
        # print("未来要花", sum(DC[i:]) * n)
        # print("现在还剩多少钱", SSB)
        # AF预算因子
        if SSB >= 0:
            AF = DC[i] * 1.1 / sum(DC[0:])
        else:
            AF = 0

        # 当前预算
        CSB = DC[i] * (n + ap[i]) + SSB * AF
        # print("当前预算", CSB)
        Ni = int(CSB / DC[i])
        redundant.append(Ni)
    res['redundant'] = redundant
    res.sort_index(inplace=True)
    redundant = res['redundant'].values.tolist()
    return redundant


# 获取冗余部署个数结果
redundancy = allocation(df)
print(redundancy)


'''
对边缘服务器进行处理 计算分配到它身上的 有重复的话怎么办呢 对于用户而言最近的ES还是没有变的 重复采样的时候Index还是不变的 好像也无所谓？跟index
这样好了 我的实验一开始的参数变小。。好像也不行
'''


'''
根据冗余结果 和 边缘服务器的容量情况 进行方案初始化生成
'''


# 欧式距离函数
def ou_distance(x, y):
    # 定义欧式距离的计算
    x = np.array(x[1:])
    y = np.array(y[1:])
    return np.sqrt(sum(np.square(x - y)))


# 查找最近空余服务器
def nearest_spare_edge(index, cur_plan):
    """
    为当前边缘服务器找到最近空闲服务器  这好像有一点问题。。。plan传进来了吗
    :param index: 当前ES编号
    :param cur_plan: 目前的部署方案
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
        if len(cur_plan[mark]) < capacity[mark]:
            print(len(cur_plan[mark]))
            print(capacity[mark])
            # res = int(edge_location3[mark, 0])
            res = mark
            break

    print("找到离" + str(index) + "最近的空闲ES是")
    print(res)
    return res


# 未部署填充-1
def translate(plan):
    for i in range(len(plan)):
        if len(plan[i]) < capacity[i]:
            # 没填满就填上-1
            for j in range(capacity[i] - len(plan[i])):
                plan[i].append(-1)
    return plan


# 将生成的初始方案组进行编码 三维变二维 可以进行遗传
def encode(plan_group):
    """
    将三维plan  变成二维 编码
    :param plan: 生成的初始方案
    :return:
    """
    pop = []
    for list_i in plan_group:
        new = [k for a in list_i for k in a]
        pop.append(new)
    return pop


# GA方案生成
pop_size = 20
def ga_plan():
    plan_group = []
    for j in range(pop_size):
        print("生成第" + str(j) + "个个体")
        plan = [[] for i in range(E)]
        for i in range(1, 21):
            # print("当前处理模块", i)
            column = 's' + str(i)
            # 根据redundant获得的冗余部署模块数
            k_num = redundancy[i - 1]
            # 读取对应列数的数据
            data = edge_data[['index', 'LATITUDE', 'LONGITUDE', column]]
            mean = data[column].mean()
            total = data[column].sum()
            # print("平均值为", mean)

            # 对数据进行筛选 只有超过 平均值 才参与聚类 否则不参与
            selected_data = data[(data[column] > mean)]
            selected_num = len(selected_data)
            high_total = selected_data[column].sum()
            proportion = high_total / total
            # 算出低频高频能分到多少个
            high_k_num = int(k_num * proportion)
            low_k_num = k_num - high_k_num

            data = data[(data[column] <= mean)]
            num = len(data)

            selected_data = selected_data[['index', 'LATITUDE', 'LONGITUDE']].values
            data = data[['index', 'LATITUDE', 'LONGITUDE']].values

            # print(type(selected_data))
            # print(selected_data.shape)
            # 获得要部署该模块的位置

            test_one = KMediod(n_points=selected_num, k_num_center=high_k_num, data=selected_data)
            centroids1 = test_one.run()
            centroids1 = centroids1[:, 0].tolist()
            # print(centroids1)
            test_two = KMediod(n_points=len(data), k_num_center=low_k_num, data=data)
            centroids2 = test_two.run2()
            centroids2 = centroids2[:, 0].tolist()
            # print(centroids2)
            centroids = centroids1 + centroids2
            # print("当前模块" + str(i) + "需要部署在以下ES上")
            # print(centroids)
            # plt.savefig("./temp_redundancy{}.png".format(i))
            # plt.clf()
            # plt.show()
            # 进行部署
            for j in centroids:
                j = int(j)
                if len(plan[j]) < capacity[j]:
                    plan[j].append(i)
                else:
                    # print(str(j) + "位置已满")
                    # print(len(plan[j]))
                    # print(capacity[j])
                    plan[nearest_spare_edge(j)].append(i)

            # check1 = [0 for i in range(20)]
            # check2 = []
            # for i in plan:
            #     check2.append(len(i))
            #     for j in i:
            #         check1[j - 1] = check1[j - 1] + 1
        print(plan)
        plan_changed = translate(plan)
        plan_group.append(plan_changed)

    pop = encode(plan_group)
    print(pop)


# 随机方案生成
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


# 随机平均方案生成
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
