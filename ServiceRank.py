import pandas as pd
import  openpyxl
import  random
import matplotlib.pyplot as plt
'''
根据各模块聚类后的结果，以及请求情况进行优先级排序
怎么想都感觉用聚类结果来描述广泛性是不对
是聚类出来的个数越多，越广泛呢；还是聚类出来的个数越少，越广泛？太矛盾了

我觉得还是用 面积 来表示分布的是否广泛 但其实呢 所有都是分布到的 只是选了高于平均数的
对selected_data进行 分布上的量化分析 

或者是把0看成无穷大，其实0就是 无法收敛 0的话设置一个数好了 

'''

# 读取处理后的数据源
service_cluster_num = pd.read_excel('..\数据处理\data_sheets.xlsx', sheet_name='cluster_result')
