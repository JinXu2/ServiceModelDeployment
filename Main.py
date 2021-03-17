from mimetypes import init

import pandas as pd
import openpyxl
import random
import numpy as np


class Service:
    '服务模块'

    def __init__(self, cost, type, rank):
        self.cost = cost
        self.type = type
        self.rank = rank


# # 生成各类服务模块
# s1 = Service(10, "start1", 6)
# s2 = Service(10, "start2", 8)
# s3 = Service(10, "start3", 18)
# s4 = Service(10, "start4", 10)
# s5 = Service(10, "start5", 12)
# s6 = Service(10, "start6", 15)
# s7 = Service(10, "end1", 7)
# s8 = Service(10, "end2", 9)
# s9 = Service(10, "end3", 19)
# s10 = Service(10, "end4", 11)
# s11 = Service(10, "end5", 13)
# s12 = Service(10, "end6", 16)
# s13 = Service(50, "Recognition", 2)
# s14 = Service(30, "Rendering", 1)
# s15 = Service(20, "Authentication", 4)
# s16 = Service(20, "Music", 3)
# s17 = Service(40, "VR", 5)
# s18 = Service(20, "Pay", 14)
# s19 = Service(20, "Social", 17)
# s20 = Service(15, "Other", 20)
# # 服务模块对象数组
# servicelist = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20]
#
#
# # 根据区域划分+应用需求度情况 生成每个用户对于不同应用的请求频率
#
#
class Application:
    '应用'

    def __init__(self, no, services):
        self.no = no
        self.services = services


# 生成应用
# a1 = Application(1, ['1', '13', '14', '15', '7'])
a1 = Application(1, ['1', '2', '3', '4'])
a2 = Application(2, ['2', '14', '16', '17', '8'])
a3 = Application(3, ['s3', 's14', 's16', 's20', 's9'])
a4 = Application(4, ['s4', 's13', 's14', 's17', 's10'])
a5 = Application(5, ['s5', 's13', 's15', 's18', 's11'])
a6 = Application(6, ['s6', 's14', 's16', 's19', 's12'])

app_list = [a1, a2, a3, a4, a5, a6]

'服务器'


# 边缘服务器 还是40个吧 但是注意异构一下 即每个ES能接受的模块个数不同
class Edge:
    def __init__(self, no, latitude, longitude, capacity):
        self.no = no
        self.latitude = latitude
        self.longitude = longitude
        self.capacity = capacity


# 创建服务器对象列表

edge_list = []
for i in range(40):
    temp = edge_data.loc[i].values[0:-1]
    temp_edge = Edge(i, temp[0], temp[1], temp[2])
    edge_list.append(temp_edge)

'用户'


# 保存的应该是用户对于应用的请求频率？
class User:
    def __init__(self, no, latitude, longitude, area, request):
        self.area = area
        self.no = no
        self.latitude = latitude
        self.longitude = longitude
        self.request = request


user_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='user_data2')
user_list = []
for i in range(len(user_data)):
    temp = user_data.loc[i].values[0:10]
    temp_user = User(i, temp[1], temp[2], temp[3], temp[4:])
    user_list.append(temp_user)

print(user_list[3].request)

'''
适应函数 计算网络延迟和传输成本的
user_list: 用户列表包括地理位置，和对每一应用的请求频率
plan: 二维数组，每个边缘服务器部署的模块
实现难点在于，如何选择冗余部署的模块 重新建图
'''


class Node:
    def __init__(self, no, latitude, longitude, type, edge_no):
        '''
        :param no: 在所有冗余部署的模块中的 独有的编号
        :param latitude: 纬度
        :param longitude: 经度
        :param type: 服务编号
        :param edge_no: 所属服务器编号
        '''
        self.no = no
        self.latitude = latitude
        self.longitude = longitude
        self.type = type
        self.edge_no = edge_no

    def __str__(self):
        return 'type: %s  edge_no: %s' % (self.type, self.edge_no)


class Point:
    def __init__(self, node):
        self.node = node
        self.next = None


def CreateDAG(plan, no, edge_list):
    '''

    :param plan: 二维数组，每个边缘服务器上部署的模块
    :param no: 应用编号
    :param edge_list: 边缘服务器列表
    :return: 有重复模块的该应用的DAG图
    '''
    # 建立哈希表，服务类型：[节点]
    service_dict = [[] for i in range(20)]

    # 建立列表，模块：
    node_list = []
    node_list.append("null")
    # 根据plan情况 先构造所有节点 并存入对应的哈希表中
    index = 1
    for i in range(len(plan)):
        for j in range(len(plan[i])):
            node = Node(index, edge_list[i].latitude, edge_list[i].longitude, plan[i][j], i)
            node_list.append(node)
            service_dict[plan[i][j]].append(node)
            index += 1

    # for i in range(len(service_dict)):
    #     for j in range(len(service_dict[i])):
    #         print(service_dict[i][j])

    # if no == 1:
    # 1 13 14 15 7 建一个属于该应用1的DAG图 1 2 3 4  app_list[no].services =  [1,13,14,15,7]
    # count = 0
    # for i in range(5):
    #     count += len(service_dict[app_list[no][i]])

    # next = 1
    # 建立邻接表 93个节点
    graph = [[] for i in range(94)]
    # print(type(graph[0]))
    services_len = len(app_list[no - 1].services)
    # services =
    for i in range(services_len - 1):
        nodes = service_dict[int(app_list[no - 1].services[i])]
        for node in nodes:
            index = node.no
            point = None
            nodes_next = service_dict[int(app_list[no - 1].services[i + 1])]
            for node_next in nodes_next:
                graph[index].append(node_next)

    for i in range(10):
        points = graph[i]
        print(i, end=':')
        for point in points:
            print(point.no, end='->')
        print()

    edges = [[] for i in range(94)]
    for i in range(94):
        edges[i] = [-1 for j in range(94)]
    for i in range(94):
        nodes = graph[i]
        if len(nodes) != 0:
            head = node_list[i]
            latitude = head.latitude
            longitude = head.longitude
            # edges[i] = [-1 for j in range(94)]
            for node in nodes:
                # edges[node.no] = [-1 for j in range(94)]
                new_latitude = node.latitude
                new_longitude = node.longitude
                result = abs(latitude - new_latitude) + abs(longitude - new_longitude)
                edges[i][node.no] = result
                # if len(edges[node.no]) == 0:
                #     edges[node.no] = [-1 for j in range(94)]
                edges[node.no][i] = result

    for e in edges:
        print(e)

    # BFS
    from queue import Queue
    import copy
    q = Queue(maxsize=0)
    for i in range(len(app_list[no - 1].services)):
        service = app_list[no - 1].services[i]
        # print(q.queue)
        # 队列初始化
        if i == 0:
            nodes = service_dict[int(service)]
            for node in nodes:
                element = [[], 0]
                element[0].append(node.no)
                q.put(element)
        else:
            size = q.qsize()
            nodes = service_dict[int(service)]
            for j in range(size):
                top = q.get()
                last = top[0][-1]
                for node in nodes:
                    # print("top:"+str(top))
                    tmp = copy.deepcopy(top)
                    # print("last:"+str(last))
                    # print("k.no:"+str(k.no))
                    tmp[1] += edges[last][node.no]
                    tmp[0].append(node.no)
                    q.put(tmp)
    # print(q.queue)
    res = q.get()
    while q.empty() == False:
        top = q.get()
        if res[1] > top[1]:
            res = top
    print(res)


def Distance(user, request, DAG):
    '''

    :param user: 用户的地理位置
    :param request: 用户对于当前应用的请求频率
    :param DAG: 当前应用的DAG示意图
    :return: 
    '''
    return


def Fitness(user_list, plan):
    '''

    :param user_list: 用户列表，用户地理位置，以及对各个应用的请求频率
    :param plan: 部署方案 二维数组，每个边缘服务器上部署的模块
    :return:
    '''

    # 首先生成每个应用的DAG图
    DAG = []
    for i in range(6):
        DAG.append(CreateDAG(plan, i + 1))

    sum_latency = 0
    for i in range(len(user_list)):
        for j in range(6):
            sum_latency += Distance(user_list[i][1:3], user_list[i][4 + i], DAG[i])

    # 获得传输距离后 计算网络延迟和传输成本 权值函数
    return


if __name__ == '__main__':
    plan = [[1, 2], [2, 3], [3, 4], [2, 4]]
    CreateDAG(plan, 1, edge_list)
