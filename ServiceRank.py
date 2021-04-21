import pandas as pd
import openpyxl
import random
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
service_cluster_num = pd.read_excel('数据处理\data_sheets.xlsx', sheet_name='cluster_result')

# index列
index = []
for i in range(1, 21):
    index.append("s" + str(i))

# 第一列 聚类结果 没问题
list1 = service_cluster_num.values.tolist()
cluster_num = list1[0]

# 第二列 请求频率汇总 修改了
request_num = pd.read_excel('数据处理\data_sheets.xlsx', sheet_name='user_data2')
request_num = request_num.iloc[:, 11:31]
request_sum = []
for i in range(20):
    request_sum.append(request_num.iloc[:, i].sum())
print(request_sum)

# 第三列 高频率分布的离散度 -> 标准差 二维随机向量的标准差 没问题

sta_dev = list1[1]

# 设定公式 优先级等于前面三者的总和
weight = []
for i in range(20):
    weight.append(cluster_num[i] + request_sum[i] / 1000 * 1.1 + (0 if (sta_dev[i] == -1) else sta_dev[i] * 100))

c = {'service': index, 'cluster_num': cluster_num, 'request_sum': request_sum, 'sta_dev': sta_dev, 'weight': weight}
rank_result = pd.DataFrame(c)
# 进行排序
# rank_result.sort_values(by="weight",axis=0,ascending=False,inplace=True)
# print(rank_result)

rank_result['rank'] = rank_result['weight'].rank(method="first", ascending=True)

wb = openpyxl.load_workbook('数据处理/data_sheets.xlsx')
write = pd.ExcelWriter('数据处理/data_sheets.xlsx', engine='openpyxl')
write.book = wb  # 没有这句话会覆盖

rank_result.to_excel(write, sheet_name='rank_result', index=False)
write.save()
write.close()
