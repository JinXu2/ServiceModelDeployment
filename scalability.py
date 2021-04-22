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
from KM import KMediod

'''
设定原始数据 从data.xlsx读取想要的数据
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

# 关于模块和成本
budget = 2000  # 不变的
price = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 50, 30, 20, 20, 40, 20, 20, 15]

# 选择用户数据
U = input("请选择用户数据表")
sheet_name = "user_data" + U
user_data = pd.read_excel('data.xlsx', sheet_name=sheet_name)

# 选择边缘服务器数据
E = input("请选择边缘服务器数据表")
sheet_name = "edge_data" + E
edge_data = pd.read_excel('data.xlsx', sheet_name=sheet_name)
capacity = edge_data['capacity'].to_list()  # 这个可以直接获取
edge_location = edge_data[['id', 'LATITUDE', 'LONGITUDE']].values

'''
对用户请求进行 模块请求总和 和AP聚类计算 离散度计算 获取每个模块的优先级
'''
cluster_result = []  # 存放AP数
sd_result = []  # 存放离散度
request_sum = []  # 存在请求总和
service_weight = []  # 模块权值 上述三者之和


# AP聚类、请求频率总和 和离散度计算
def service_rank():
    for i in range(1, 21):
        # 进行AP聚类和离散度计算
        coloumn = 's' + str(i)
        # 读取对应列数的数据
        data = user_data[['Latitude', 'Longitude', coloumn]]
        mean = data[coloumn].mean()
        # 请求总和
        request_sum.append(data[coloumn].sum())

        # 对数据进行筛选 只有超过 平均值 才参与聚类 否则不参与
        selected_data = data[(data[coloumn] > mean)]
        selected_num = len(selected_data)

        data = data[(data[coloumn] <= mean)]
        # 筛选后数据可能为 0
        if selected_num == 0:
            sd_result.append(-1)
            cluster_result.append(0)

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

                continue
            else:
                cluster_result.append(len(cluster_centers_indices))

    # 优先级计算
    # 设定公式 优先级等于前面三者的总和
    for i in range(20):
        service_weight.append(float(
            cluster_result[i] + request_sum[i] / 1000 * 1.1 + (
                0 if (sd_result[i] == -1) else float(sd_result[i]) * 100)))


service_rank()
# 根据优先级计算 每个模块的rank值 通过建立dataframe来实现

# index列
index = []
for i in range(1, 21):
    index.append("s" + str(i))
c = {'service': index, 'cluster_num': cluster_result, 'request_sum': request_sum, 'sta_dev': sd_result,
     'weight': service_weight, 'price': price}

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
print("redundancy")
print(redundancy)

'''
根据冗余结果 和 边缘服务器的容量情况 进行方案初始化生成

首先找到离到每个用户最近的ES 记录下index
统计分配该ES上的应用请求之和 和模块请求之和

'''

# 离到每个用户最近的ES 记录下index
nearest = []


def find_nearest():
    for i in range(len(user_data)):
        min = 10
        temp = -1
        for j in range(len(edge_data)):
            distance = abs(user_data.iloc[i, 1] - edge_data.iloc[j, 1]) + abs(
                user_data.iloc[i, 2] - edge_data.iloc[j, 2])
            if distance < min:
                temp = j
                min = distance
        nearest.append(temp)


find_nearest()
# 把这列加到 user_data中去
print("nearest")
user_data['nearest'] = nearest

# 统计分配该ES上的 模块请求之和
request_sum = np.zeros((len(edge_data), 20))


def compute_edge_request():
    for i in range(len(user_data)):
        edge_index = user_data.iloc[i][30]
        for j in range(20):
            request_sum[edge_index][j] += user_data.iloc[i, j + 10]


compute_edge_request()
request_sum = request_sum.tolist()
df = pd.DataFrame(request_sum,
                  columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                           's15', 's16', 's17', 's18', 's19', 's20'])

edge_data = pd.concat([edge_data, df], axis=1)
'''
生成各种部署方案
'''
edge_len = len(edge_data)


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


# GA 方案生成 相对来说很复杂 而且时间很慢很慢 只能生成初始化的种群
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
    plan = [[] for i in range(edge_len)]

    for i in range(1, 21):
        j = redundancy[i - 1]
        for k in range(j):
            # 开始生成j个随机数
            tmp = random.randint(0, edge_len - 1)
            # print("要把" + str(i) + "放到服务器", tmp)
            while len(plan[tmp]) > capacity[tmp]:
                # 超过了所以要变tmp
                # print("超过了")
                tmp = random.randint(0, 39)
            plan[tmp].append(i)

    return plan


CR_plan = random_plan()
print(CR_plan)


# 随机平均方案生成
def avg_random_plan():
    redundancy2 = []
    total = sum(price)
    n = int(budget / total)
    for i in range(20):
        redundancy2.append(n)

    plan = [[] for i in range(edge_len)]

    for i in range(1, 21):
        for k in range(n):
            # 开始生成j个随机数
            tmp = random.randint(0, edge_len - 1)
            # print("要把" + str(i) + "放到服务器", tmp)
            while len(plan[tmp]) > capacity[tmp]:
                # 超过了所以要变tmp
                # print("超过了")
                tmp = random.randint(0, edge_len - 1)
            plan[tmp].append(i)

    return plan


AR_plan = avg_random_plan()
print(AR_plan)


# 请求频率优先
def request_first_plan():
    # 根据冗余部署模块 和每个ES对于该模块请求频率最大的上面
    # 根据edge_data
    plan = [[] for i in range(edge_len)]
    for i in range(1, 21):
        need = redundancy[i - 1]
        # 找到排名前need个且空闲的ES进行部署
        cur = 's' + str(i)
        # 这里index 和 id 有很大不同的地方
        data = edge_data[cur].values
        index = data.argsort()
        index = index[::-1]
        cnt, j = 0, 0
        # 优先放在排名前面，如果还有空位就放上去，同时count+1
        while cnt < need:
            if len(plan[index[j]]) < capacity[index[j]]:
                plan[index[j]].append(i)
                cnt += 1
            j += 1
    return plan


MRF_plan = request_first_plan()
print(MRF_plan)


# 网络延迟最低优先  对于每个模块而言 如果只部署在这个ES上面的话，与其他ES的距离则是所有的网络延迟 再找到排名前R个的 矩阵相乘
def latency_first_plan():
    plan = [[] for i in range(edge_len)]
    # 先建立距离矩阵
    dis = np.zeros([edge_len, edge_len], dtype=float)
    for i in range(0, edge_len):
        for j in range(i, edge_len):
            temp = ou_distance(edge_location[i], edge_location[j])
            dis[i][j] = dis[j][i] = temp

    for i in range(1, 21):
        need = redundancy[i - 1]
        # 开始创建该模块的请求频率矩阵
        column = 's' + str(i)
        df = edge_data[column]
        request = np.array(df).reshape(edge_len, 1)
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


MLF_plan = latency_first_plan()
print(MLF_plan)


# 负载均衡方案生成
def load_balance_plan():
    plan = [[] for i in range(edge_len)]
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
        j %= edge_len
        index += 1
    return plan


LB_plan = load_balance_plan()
print(LB_plan)


def evaluate(cur_plan, cur_edge, cur_user):
