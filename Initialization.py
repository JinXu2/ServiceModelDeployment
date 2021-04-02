import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from KM import KMediod
import functools

'''
根据任务优先级排序后的结果
以及各自所需的成本

进行成本初次分配 解决 需要冗余部署的模块能部署几个的问题
'''

# 事先设定的值
budget = 2000
n = 4


# 读取数据
service_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='rank_result')
df = service_data[['service', 'cluster_num', 'rank', 'price']]



def allocation(df):
    """

    :param df: 数据集 包括任务编号，聚类个数，优先级，部署价格
    :return: redundant[]列表 表示每个模块冗余部署的个数
    """

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
            AF = DC[i] *1.1 / sum(DC[0:])
        else:
            AF = 0

        # 当前预算
        CSB = DC[i] * (n + ap[i]) + SSB * AF
        # print("当前预算", CSB)
        Ni = int(CSB / DC[i])
        redundant.append(Ni)
    '''
      但是这样子超过了 边缘服务器异构的数量不保证的撒
      要么总的不看了？？？？总归不可能放不下的把
      直接开始想冗余的模块怎么部署了
      只是 一个边缘服务器上不能放的超过5个？？ 那不就还是自己编的么 反正就是成本不多
      '''
    # 为检验
    # 我靠 出现大错误 这个redundant是按照优先级排序下来每个模块应该冗余部署的情况 而在后面的时候 变成了从1开始的 已解决

    res['redundant']=redundant
    res.sort_index(inplace=True)
    redundant = res['redundant'].values.tolist()

    # print(sum(redundant))
    # print(sum(list(map(func, DC, redundant))))
    # print(redundant)
    return redundant


# 获取冗余部署个数结果
redundancy = allocation(df)
print(redundancy)
edge_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='edge_data1')


# 且不能超过其能力
capacity = edge_data['capacity'].to_list()
edge_location = edge_data[['index', 'LATITUDE', 'LONGITUDE']].values


def ou_distance(x, y):
    # 定义欧式距离的计算
    x = x[1:-1]
    y = y[1:-1]
    return np.sqrt(sum(np.square(x - y)))


def nearest_spare_edge(index):
    """
    为当前边缘服务器找到最近空闲服务器  这好像有一点问题。。。plan传进来了吗
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
            # res = int(edge_location3[mark, 0])
            res = mark
            break

    print("找到离" + str(index) + "最近的空闲ES是")
    print(res)
    return res


def translate(plan):
    for i in range(len(plan)):
        if len(plan[i]) < capacity[i]:
            # 没填满就填上-1
            for j in range(capacity[i] - len(plan[i])):
                plan[i].append(-1)

    # 再把二维变成一维 在遗传里面变好了
    # new_plan = [k for a in plan for k in a]
    # return new_plan
    return plan


def encode(plan):
    """
    将三维plan  变成二维 编码
    :param plan: 生成的初始方案
    :return:
    """
    pop = []
    for list_i in plan:
        new = [k for a in list_i for k in a]
        pop.append(new)
    return pop



def check(plan):
    D = df['price'].values
    print(D)
    r = [0 for i in range(20)]
    for list_i in plan:
        for tmp in list_i:
            if tmp == -1 : continue
            else:
                r[tmp-1] +=1

    func = lambda x, y: x * y
    print(r)
    print(sum(list(map(func, D, r))))
    return
'''
要对plan进行一些操作 首先capacity有空位的 要用-1来填充 相应的在DAG中创建节点时要将-1忽视
之后将二维列表Plan变成一维列表

同时这个初始化要生成很多plan 相应的在KM算法中的质心要随机生成

'''

# def create_plan(num):
#     plan_group = []
#     for j in range(1,int(num)+1):
#         print("生成第" + str(j) + "个个体")
#         plan = [[] for i in range(40)]
#         for i in range(1, 21):
#             # print("当前处理模块", i)
#             column = 's' + str(i)
#             # 根据redundant获得的冗余部署模块数
#             k_num = redundancy[i - 1]
#             # 读取对应列数的数据
#             data = edge_data[['index', 'LATITUDE', 'LONGITUDE', column]]
#             mean = data[column].mean()
#             total = data[column].sum()
#             # print("平均值为", mean)
#
#             # 对数据进行筛选 只有超过 平均值 才参与聚类 否则不参与
#             selected_data = data[(data[column] > mean)]
#             selected_num = len(selected_data)
#             high_total = selected_data[column].sum()
#             proportion = high_total / total
#             # 算出低频高频能分到多少个
#             high_k_num = int(k_num * proportion)
#             low_k_num = k_num - high_k_num
#
#             data = data[(data[column] <= mean)]
#             num = len(data)
#
#             selected_data = selected_data[['index', 'LATITUDE', 'LONGITUDE']].values
#             data = data[['index', 'LATITUDE', 'LONGITUDE']].values
#
#             # print(type(selected_data))
#             # print(selected_data.shape)
#             # 获得要部署该模块的位置
#
#             test_one = KMediod(n_points=selected_num, k_num_center=high_k_num, data=selected_data)
#             centroids1 = test_one.run()
#             centroids1 = centroids1[:, 0].tolist()
#             # print(centroids1)
#             test_two = KMediod(n_points=len(data), k_num_center=low_k_num, data=data)
#             centroids2 = test_two.run2()
#             centroids2 = centroids2[:, 0].tolist()
#             # print(centroids2)
#             centroids = centroids1 + centroids2
#             # print("当前模块" + str(i) + "需要部署在以下ES上")
#             # print(centroids)
#             # plt.savefig("./temp_redundancy{}.png".format(i))
#             # plt.clf()
#             # plt.show()
#             # 进行部署
#             for j in centroids:
#                 j = int(j)
#                 if len(plan[j]) < capacity[j]:
#                     plan[j].append(i)
#                 else:
#                     # print(str(j) + "位置已满")
#                     # print(len(plan[j]))
#                     # print(capacity[j])
#                     plan[nearest_spare_edge(j,plan)].append(i)
#
#             # check1 = [0 for i in range(20)]
#             # check2 = []
#             # for i in plan:
#             #     check2.append(len(i))
#             #     for j in i:
#             #         check1[j - 1] = check1[j - 1] + 1
#         # print(plan)
#         plan_changed = translate(plan)
#         plan_group.append(plan_changed)
#     print("生成方案")
#     print(plan_group)
#     return plan_group


if __name__ == '__main__':
    # 生成部署方案plan 二维列表
    plan_group = []
    for j in range(20):
        print("生成第"+str(j)+"个个体")
        plan = [[] for i in range(40)]
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