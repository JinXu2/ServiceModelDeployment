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

'''
网络延迟DAG 跟之前不同的地方在于
1.添加最后的用户节点  完成
2.RTT算法 5+0.02*distance 
3.用户上传和下载的 latency要重新定义的 要考虑datasize  但这样就不对 那岂不是用户的位置都不用考虑了啊 把两个都写上去好了


'''
gauge = 111000
transmission_rate = 300000000

bandwidth = 2500  # kb/s 2.5m/s
app_data = [0, 10, 20, 5, 15, 5, 15]

# 是不是没有加请求频率啊

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

        # 在这里加上最后用户的信息
        size = q.qsize()
        for j in range(size):
            top = q.get()
            last = top[0][-1]
            node = self.node_list[last]
            tmp = copy.deepcopy(top)
            tmp[1] += (abs(user_node.latitude - node.latitude)
                       + abs(user_node.longitude - node.longitude))
            tmp[0].append(-1)
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
        # print("当前用户最短路径是")
        # print(res[0])
        return res

    def sBFS(self, user_list, app_no):
        path = []
        path_sum = 0
        user_len = len(user_list)
        for i in range(user_len):
            res = self.BFS(user_list[i], app_no)
            path.append(res[0])
            path_sum += res[1]
        return path_sum

    def run(self, user_list):
        self.create_node()
        self.create_weight()
        user_len = len(user_list)
        # print("一共有用户", user_len)
        proportion_total = 0
        transmission_total = 0
        app_number = len(self.app_list)  # 省略了0的情况的

        for i in range(1, app_number):
            path_sum = self.sBFS(user_list, i)
            # print(path_sum)
            print("应用" + str(i) + "的总传播延迟是")
            proportion = path_sum * gauge / transmission_rate * 1000
            print(proportion)

            # 这里加上上传和下载的延迟
            down = app_data[i] / bandwidth * user_len
            up = app_data[i] / bandwidth * user_len
            print("应用" + str(i) + "的总发送延迟是")
            print(down + up)

            print("总网络延迟为")
            print(proportion + down + up)

            proportion_total += proportion
            transmission_total += down + up
        # print(total)
        return proportion_total, transmission_total


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
    planCR = [[2, 3, 11, 13, 18, 18, -1, -1], [15, -1, -1, -1, -1, -1, -1, -1], [14, -1, -1, -1, -1, -1],
              [6, 6, -1, -1, -1, -1, -1], [2, 4, 16, -1, -1, -1], [16, -1, -1, -1, -1], [7, 15, 17, -1, -1],
              [-1, -1, -1, -1, -1, -1, -1, -1], [15, 16, 18, 19, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1],
              [1, 8, -1, -1, -1, -1, -1], [2, 11, 14, 19, 19, 20], [3, 4, 8, -1, -1, -1, -1], [4, 10, -1, -1, -1],
              [4, 5, 17, -1, -1, -1], [12, 17, -1, -1, -1, -1, -1, -1], [4, 13, -1, -1, -1, -1],
              [8, 8, 20, -1, -1, -1, -1], [1, 11, 11, 16, 20, -1, -1, -1], [5, 9, 10, 10, 12, 13, -1],
              [7, 14, 15, -1, -1], [13, 14, 20, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1],
              [3, 5, 10, 10, 14, 17, -1], [14, 15, -1, -1, -1, -1, -1], [4, 12, 16, -1, -1, -1, -1],
              [7, 9, 10, 11, -1, -1, -1], [2, 2, -1, -1, -1, -1], [5, 14, 17, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1, -1], [1, 7, 9, 18, -1, -1, -1, -1], [5, 6, -1, -1, -1, -1],
              [2, 12, 13, 19, -1, -1, -1], [6, 13, 17, -1, -1, -1], [8, 9, 16, -1, -1, -1, -1], [5, 11, 13, -1, -1, -1],
              [3, 14, 16, 17, 18, -1], [1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [8, 14, 18, -1, -1, -1, -1]]
    planAR = [[7, 8, 12, 17, -1, -1, -1, -1], [10, 12, 13, 14, -1, -1, -1, -1], [5, 6, 10, 13, -1, -1],
              [19, 20, -1, -1, -1, -1, -1], [14, -1, -1, -1, -1, -1], [17, -1, -1, -1, -1], [11, 16, 16, 19, 20],
              [-1, -1, -1, -1, -1, -1, -1, -1], [14, 15, 18, -1, -1, -1, -1], [12, 16, 20, -1, -1, -1, -1],
              [3, 17, 17, -1, -1, -1, -1], [6, 11, 13, -1, -1, -1], [4, 10, 18, -1, -1, -1, -1], [5, 8, -1, -1, -1],
              [5, 11, 14, -1, -1, -1], [8, -1, -1, -1, -1, -1, -1, -1], [1, 14, 16, -1, -1, -1],
              [1, 10, 15, -1, -1, -1, -1], [1, 8, 8, 15, -1, -1, -1, -1], [2, 6, -1, -1, -1, -1, -1],
              [15, -1, -1, -1, -1], [4, 5, 7, 13, -1, -1], [17, 19, -1, -1, -1, -1, -1, -1], [2, 4, 16, 17, -1, -1, -1],
              [3, 20, -1, -1, -1, -1, -1], [5, 5, 11, -1, -1, -1, -1], [8, 13, 14, 16, -1, -1, -1],
              [4, 14, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [2, 3, 12, 16, 17, 18, -1],
              [14, -1, -1, -1, -1, -1, -1, -1], [2, 4, 18, 19, -1, -1], [4, 11, 18, -1, -1, -1, -1],
              [15, -1, -1, -1, -1, -1], [11, -1, -1, -1, -1, -1, -1], [3, 9, 10, -1, -1, -1], [6, 7, 7, 18, -1, -1],
              [9, -1, -1, -1, -1], [2, 2, 9, 13, -1, -1, -1], [9, 10, 13, 14, -1, -1, -1]]
    planMRF = [[16, 17, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1], [4, 10, 12, 13, 14, 16],
               [14, 16, 17, 20, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [2, 8, 9, 10, 12], [1, 3, 4, 6, 7],
               [-1, -1, -1, -1, -1, -1, -1, -1], [2, 8, 9, 10, 12, 13, 14], [-1, -1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1], [5, 11, 13, 15, 18, 20], [-1, -1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1], [1, 2, 3, 4, 6, 7], [-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1], [1, 2, 3, 6, 7, 8, 9, 10], [4, 10, 17, -1, -1, -1, -1],
               [5, 11, 15, 18, -1], [1, 3, 4, 6, 7, 9], [-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [5, 11, 13, 15, 18, 19, 20],
               [2, 8, 12, 14, 16, 17], [-1, -1, -1, -1, -1, -1, -1], [5, 11, 13, 14, 15, 18, 19],
               [-1, -1, -1, -1, -1, -1, -1, -1], [14, 16, 19, 20, -1, -1], [-1, -1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1], [2, 8, 14, 16, 17, -1, -1], [4, 10, 13, 14, 17, 19], [-1, -1, -1, -1, -1, -1],
               [5, 11, 13, 15, 18], [5, 11, 18, -1, -1, -1, -1], [8, 14, 16, 17, -1, -1, -1]]
    planMLF = [[13, 15, 16, 20, -1, -1, -1, -1], [15, -1, -1, -1, -1, -1, -1, -1], [14, 17, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1], [5, 11, 18, -1, -1, -1], [4, 10, -1, -1, -1], [-1, -1, -1, -1, -1],
               [15, -1, -1, -1, -1, -1, -1, -1], [2, 8, 13, 14, 16, 17, 19], [-1, -1, -1, -1, -1, -1, -1],
               [5, 11, 18, -1, -1, -1, -1], [5, 11, 18, -1, -1, -1], [1, 2, 3, 6, 7, 8, 9], [-1, -1, -1, -1, -1],
               [15, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1],
               [5, 11, 18, -1, -1, -1, -1], [1, 2, 3, 6, 7, 8, 9, 12], [4, 10, 14, 17, -1, -1, -1], [5, 11, 18, -1, -1],
               [4, 10, 14, 17, -1, -1], [4, 10, -1, -1, -1, -1, -1, -1], [1, 2, 3, 6, 7, 8, 9],
               [15, -1, -1, -1, -1, -1, -1], [5, 11, 18, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1],
               [12, 13, 14, 16, 19, 20], [4, 10, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1, -1], [13, 14, 16, 17, 19, 20], [4, 10, -1, -1, -1, -1, -1],
               [13, 14, 16, 17, -1, -1], [1, 2, 3, 6, 7, 8, 9], [-1, -1, -1, -1, -1, -1], [12, 13, 14, 16, 19, 20],
               [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [2, 8, 12, 13, 14, 16, 17]]
    planLB = [[1, 9, 15, -1, -1, -1, -1, -1], [1, 9, 16, -1, -1, -1, -1, -1], [1, 9, 16, -1, -1, -1],
              [1, 9, 16, -1, -1, -1, -1], [2, 10, 16, -1, -1, -1], [2, 10, 16, -1, -1], [2, 10, 16, -1, -1],
              [2, 10, 16, -1, -1, -1, -1, -1], [2, 10, 17, -1, -1, -1, -1], [2, 10, 17, -1, -1, -1, -1],
              [3, 11, 17, -1, -1, -1, -1], [3, 11, 17, -1, -1, -1], [3, 11, 17, -1, -1, -1, -1], [3, 11, 17, -1, -1],
              [4, 11, 17, -1, -1, -1], [4, 11, 18, -1, -1, -1, -1, -1], [4, 12, 18, -1, -1, -1],
              [4, 12, 18, -1, -1, -1, -1], [4, 12, 18, -1, -1, -1, -1, -1], [4, 12, 18, -1, -1, -1, -1],
              [5, 13, 18, -1, -1], [5, 13, 19, -1, -1, -1], [5, 13, 19, -1, -1, -1, -1, -1],
              [5, 13, 19, -1, -1, -1, -1], [5, 13, 19, -1, -1, -1, -1], [5, 13, 20, -1, -1, -1, -1],
              [6, 13, 20, -1, -1, -1, -1], [6, 14, 20, -1, -1, -1], [6, 14, 20, -1, -1, -1, -1],
              [6, 14, -1, -1, -1, -1, -1], [7, 14, -1, -1, -1, -1, -1, -1], [7, 14, -1, -1, -1, -1],
              [7, 14, -1, -1, -1, -1, -1], [7, 14, -1, -1, -1, -1], [8, 14, -1, -1, -1, -1, -1],
              [8, 14, -1, -1, -1, -1], [8, 15, -1, -1, -1, -1], [8, 15, -1, -1, -1], [8, 15, -1, -1, -1, -1, -1],
              [8, 15, -1, -1, -1, -1, -1]]
    planGA = [[16, 16, 17, -1, -1, -1, -1, -1], [20, -1, -1, -1, -1, -1, -1, -1], [4, 7, 10, 14, 16, 17],
              [2, -1, -1, -1, -1, -1, -1], [11, 14, 15, 18, -1, -1], [-1, -1, -1, -1, -1], [4, 9, 10, 14, -1],
              [-1, -1, -1, -1, -1, -1, -1, -1], [8, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1],
              [5, 6, 7, 9, 11, 13, 16], [11, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [2, 8, -1, -1, -1],
              [1, 4, 7, 10, 13, 14], [5, -1, -1, -1, -1, -1, -1, -1], [1, 3, 11, 14, 18, -1],
              [12, 15, -1, -1, -1, -1, -1], [16, 17, -1, -1, -1, -1, -1, -1], [1, 4, 6, 7, 10, -1, -1],
              [5, 13, -1, -1, -1], [2, 4, 8, 10, 13, 15], [-1, -1, -1, -1, -1, -1, -1, -1], [2, 12, 15, 16, 19, -1, -1],
              [-1, -1, -1, -1, -1, -1, -1], [19, 20, -1, -1, -1, -1, -1], [3, 6, 12, 19, -1, -1, -1],
              [2, 8, 17, 20, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [5, 13, 14, 16, 18, -1, -1],
              [17, -1, -1, -1, -1, -1, -1, -1], [1, 3, 6, 8, 12, 19], [3, -1, -1, -1, -1, -1, -1],
              [9, -1, -1, -1, -1, -1], [5, 9, 11, 13, 18, -1, -1], [4, 10, 14, 17, -1, -1], [14, 20, -1, -1, -1, -1],
              [15, 18, -1, -1, -1], [5, 11, 13, 18, -1, -1, -1], [2, 8, 14, 17, -1, -1, -1]]

    # test_one = DAG(plan=planCR, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # print("进入完全随机CR算法")
    # p_total, t_total = test_one.run(user_list=user_list)
    # print("CR总传播网络延迟")
    # print(p_total)
    # print("CR总传输网络延迟")
    # print(t_total)
    #
    # test_two = DAG(plan=planAR, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # print("进入平均随机AR算法")
    # p_total, t_total = test_two.run(user_list=user_list)
    # print("AR总传播网络延迟")
    # print(p_total)
    # print("AR总传输网络延迟")
    # print(t_total)
    #
    # test_three = DAG(plan=planMRF, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # print("进入最大请求优先MRF算法")
    # p_total, t_total = test_three.run(user_list=user_list)
    # print("MRF总传播网络延迟")
    # print(p_total)
    # print("MRF总传输网络延迟")
    # print(t_total)
    #
    # test_four = DAG(plan=planMLF, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # print("进入最小延迟优先MLF算法")
    # p_total, t_total = test_four.run(user_list=user_list)
    # print("MLF总传播网络延迟")
    # print(p_total)
    # print("MLF总传输网络延迟")
    # print(t_total)
    #
    # test_five = DAG(plan=planLB, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    # print("进入负载均衡LB算法")
    # p_total, t_total = test_five.run(user_list=user_list)
    # print("LB总传播网络延迟")
    # print(p_total)
    # print("LB总传输网络延迟")
    # print(t_total)

    test_six = DAG(plan=planGA, service_type_sum=20, edge_list=edge_list, app_list=app_list)
    print("进入遗传最优算法")
    p_total, t_total = test_six.run(user_list=user_list)
    print("GA总传播网络延迟")
    print(p_total)
    print("GA总传输网络延迟")
    print(t_total)