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


class DAG:
    """
    图的建立
    """

    def __init__(self, plan, service_type_sum, edge_list, app_list):
        self.weight = None
        self.plan = plan
        self.service_dict = [[] for k in range(service_type_sum + 1)]
        self.node_list = [Node(0, -1, -1, -1, -1)]
        self.edge_list = edge_list
        self.app_list = app_list

    def create_node(self):
        """
        创建service map 和 node_list
        :return:
        """
        index = 1
        for i in range(len(self.plan)):
            for j in range(len(self.plan[i])):
                if self.plan[i][j] == -1:  # 不用创建节点
                    continue
                node = Node(no=index, latitude=self.edge_list[i].latitude, longitude=self.edge_list[i].longitude,
                            type=self.plan[i][j],
                            edge_no=self.edge_list[i].no)
                # print(node)
                self.node_list.append(node)
                self.service_dict[self.plan[i][j]].append(node)
                index += 1

    def create_weight(self):
        """
        创建权值矩阵
        :return:
        """
        # 初始化
        length = len(self.node_list)
        self.weight = [[] for i in range(length)]
        for i in range(length):
            self.weight[i] = [-1 for j in range(length)]

        # 对称图
        for i in range(1, length):
            now_node = self.node_list[i]
            for j in range(i, length):
                temp_node = self.node_list[j]
                self.weight[j][i] = self.weight[i][j] = (abs(now_node.latitude - temp_node.latitude)
                                                         + abs(now_node.longitude - temp_node.longitude))

    def BFS(self, user, service_no):
        q = Queue(maxsize=0)
        # 先要把用户也创建成一个node 创不创也无所谓的 在i=0的时候初始化了的
        user_node = Node(-1, user.latitude, user.longitude, -1, -1)

        for i in range(len(self.app_list[service_no].services)):
            service = self.app_list[service_no].services[i]
            if i == 0:
                nodes = self.service_dict[int(service)]
                for node in nodes:
                    element = [[], 0]
                    element[0].append(node.no)
                    element[1] = (abs(user_node.latitude - node.latitude)
                                  + abs(user_node.longitude - node.longitude))
                    q.put(element)
            else:
                size = q.qsize()
                nodes = self.service_dict[int(service)]
                for j in range(size):
                    top = q.get()
                    last = top[0][-1]
                    for node in nodes:
                        tmp = copy.deepcopy(top)
                        tmp[1] += self.weight[last][node.no]
                        tmp[0].append(node.no)
                        q.put(tmp)

        res = q.get()
        while not q.empty():
            top = q.get()
            if res[1] > top[1]:
                res = top

        # print(user.no, service_no, res[0], res[1])
        # if user.no == 1 and service_no == 1:
        #     manul = (abs(user.latitude - self.node_list[res[0][0]].latitude)+
        #             abs(user.longitude - self.node_list[res[0][0]].longitude))
        #
        #     for i in range(4):
        #         manul = manul + self.weight[res[0][i]][res[0][i+1]]
        #     print(manul)

        return res

    def sBFS(self, user_list, service_no):
        path = []
        path_sum = 0
        for i in range(len(user_list)):
            res = self.BFS(user_list[i], service_no)
            path.append(res[0])
            path_sum += res[1]
        return path_sum

    def run(self, user_list):
        self.create_node()
        self.create_weight()
        total = 0
        app_number = len(self.app_list)  # 省略了0的情况的
        for i in range(1, app_number):
            print("应用" + str(i) + "的总网络延迟是")
            path_sum = self.sBFS(user_list, i)
            print(path_sum)
            total += path_sum
        # print(total)
        return total


edge_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='edge_data1')
edge_list = []
for i in range(40):
    temp = edge_data.loc[i].values[0:-1]
    temp_edge = Edge(no=i + 1, latitude=temp[1], longitude=temp[2], capacity=temp[3])
    edge_list.append(temp_edge)

a0 = Application(0, [])
a1 = Application(1, ['1', '13', '14', '15', '7'])
# a1 = Application(1, ['1', '2', '3', '4'])
a2 = Application(2, ['2', '14', '16', '17', '8'])
a3 = Application(3, ['3', '14', '16', '20', '9'])
a4 = Application(4, ['4', '13', '14', '17', '10'])
a5 = Application(5, ['5', '13', '15', '18', '11'])
a6 = Application(6, ['6', '14', '16', '19', '12'])

app_list = [a0, a1, a2, a3, a4, a5, a6]

user_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='user_data2')
user_list = []
for i in range(len(user_data)):
    temp = user_data.loc[i].values[0:10]
    temp_user = User(i + 1, temp[1], temp[2], temp[3], temp[4:])
    user_list.append(temp_user)

if __name__ == '__main__':
    plan = [[2, 8, 16, 17], [1, 3], [1, 3, 4, 12, 16, 19], [2, 6, 7, 8, 9, 12, 14], [5, 11], [1, 2, 3, 6, 7],
            [1, 3, 4, 6, 9], [1, 3], [1, 2, 3, 8], [15], [4, 5, 6, 7, 9, 11, 14], [5, 15, 18], [2, 8],
            [2, 8, 12, 19, 20], [2, 10, 16, 17, 19, 20, 20], [5], [5, 8, 9, 14, 17], [], [], [4], [10, 13, 13, 15],
            [4, 10, 13], [], [16], [], [1, 3, 16], [1, 3], [], [], [11, 18], [13, 17], [6, 7, 9, 14, 15, 17], [1, 3],
            [6, 7, 9, 14], [5, 11, 18], [4], [], [], [], [4, 13, 15]]
    test_one = DAG(plan=plan, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    print("DAG构建成功")
    total = test_one.run(user_list=user_list)
    print(total)
    print("test succeeded")
