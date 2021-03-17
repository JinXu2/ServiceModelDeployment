# -*- coding: utf-8 -*-
# @Time    : 18-12-6
# @Author  : lin

from sklearn.datasets import make_blobs
from matplotlib import pyplot
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


class KMediod():
    """
    实现简单的k-medoid算法
    """

    def __init__(self, n_points, k_num_center, data):
        self.n_points = n_points
        self.k_num_center = k_num_center
        self.data = data

    def get_test_data(self):
        """
        产生测试数据, n_samples表示多少个点, n_features表示几维, centers
        得到的data是n个点各自坐标
        target是每个坐标的分类比如说我规定好四个分类，target长度为n范围为0-3，主要是画图颜色区别
        :return: none
        """
        # self.data, target = make_blobs(n_samples=self.n_points, n_features=2, centers=self.n_points)
        # print(self.data)
        np.put(self.data, [self.n_points, 0], 500, mode='clip')
        np.put(self.data, [self.n_points, 1], 500, mode='clip')
        pyplot.scatter(self.data[:, 0], self.data[:, 1], c='blue')
        # 画图
        pyplot.show()

    def ou_distance(self,  x, y):
        # 定义欧式距离的计算
        x = x[1:-1]
        y = y[1:-1]
        return np.sqrt(sum(np.square(x - y)))

    def run_k_center(self, func_of_dis):
        """
        选定好距离公式开始进行训练
        :param func_of_dis:
        :return:
        """
        print('初始化', self.k_num_center, '个中心点')
        indexs = list(range(len(self.data)))
        random.shuffle(indexs)  # 随机选择质心
        init_centroids_index = indexs[:self.k_num_center]
        centroids = self.data[init_centroids_index, :]  # 初始中心点
        # print(centroids)
        # 确定种类编号
        levels = list(range(self.k_num_center))
        print('开始迭代')
        sample_target = []
        if_stop = False
        while (not if_stop):
            if_stop = True
            classify_points = [[centroid] for centroid in centroids]
            sample_target = []
            # 遍历数据
            for sample in self.data:
                # 计算距离，由距离该数据最近的核心，确定该点所属类别
                distances = [func_of_dis(sample, centroid) for centroid in centroids]
                cur_level = np.argmin(distances)
                sample_target.append(cur_level)
                # 统计，方便迭代完成后重新计算中间点
                classify_points[cur_level].append(sample)
            # 重新划分质心
            for i in range(self.k_num_center):  # 几类中分别寻找一个最优点
                distances = [func_of_dis(point_1, centroids[i]) for point_1 in classify_points[i]]
                now_distances = sum(distances)  # 首先计算出现在中心点和其他所有点的距离总和
                for point in classify_points[i]:
                    distances = [func_of_dis(point_1, point) for point_1 in classify_points[i]]
                    new_distance = sum(distances)
                    # 计算出该聚簇中各个点与其他所有点的总和，若是有小于当前中心点的距离总和的，中心点去掉
                    if new_distance < now_distances:
                        now_distances = new_distance
                        centroids[i] = point  # 换成该点
                        if_stop = False
        print('结束')

        # 想要获取中心点的index
        print(centroids)
        return sample_target, centroids

    def run(self):
        """
        先获得数据，由传入参数得到杂乱的n个点，然后由这n个点，分为m个类
        :return:
        """
        # self.get_test_data()
        # 需要的是 中心点的结果
        predict, centroids = self.run_k_center(self.ou_distance)
        pyplot.scatter(self.data[:, 1], self.data[:, 2], c='red', alpha=0.5)
        pyplot.scatter(centroids[:, 1], centroids[:, 2], c='red', marker="*")
        # pyplot.show()

    def run2(self):
        """
        先获得数据，由传入参数得到杂乱的n个点，然后由这n个点，分为m个类
        :return:
        """
        # self.get_test_data()
        # 需要的是 中心点的结果
        predict, centroids = self.run_k_center(self.ou_distance)
        pyplot.scatter(self.data[:, 1], self.data[:, 2], c='blue', alpha=0.5)
        pyplot.scatter(centroids[:, 1], centroids[:, 2], c='blue', marker="+")
        # pyplot.show()


edge_data = pd.read_excel('数据处理/data_sheets.xlsx', sheet_name='edge_data1')
for i in range(1, 2):
    print("当前处理模块", i)
    column = 's' + str(i)
    # 根据redundant获得的冗余部署模块数
    k_num = 9 # 测试写9
    # 读取对应列数的数据
    data = edge_data[['index', 'LATITUDE', 'LONGITUDE', column]]
    mean = data[column].mean()
    total = data[column].sum()
    print("平均值为", mean)

    # 对数据进行筛选 只有超过 平均值 才参与聚类 否则不参与
    selected_data = data[(data[column] > mean)]
    selected_num = len(selected_data)
    high_total = selected_data[column].sum()
    proportion = high_total/total
    # 算出低频高频能分到多少个
    high_k_num = int(k_num * proportion)
    low_k_num = k_num - high_k_num

    data = data[(data[column] <= mean)]
    num = len(data)

    selected_data = selected_data[['index', 'LATITUDE', 'LONGITUDE']].values
    data = data[['index', 'LATITUDE', 'LONGITUDE']].values

    # print(type(selected_data))
    # print(selected_data.shape)
    test_one = KMediod(n_points=selected_num, k_num_center=high_k_num, data=selected_data)
    test_one.run()

    test_two = KMediod(n_points=len(data), k_num_center=low_k_num, data=data)
    test_two.run2()
    pyplot.show()


