import pandas as pd
import  openpyxl
import random as rd
import numpy as np
'''
对边缘服务器数据集进行处理
1.生成异构性的模块容量
2.统计从属于自己的用户的 对于各自模块的请求频率
'''
edge_data = pd.read_excel('data_sheets.xlsx', sheet_name='edge_data')
user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data2')
random_capacity = []
for i in range(40):
    random_capacity.append(rd.randint(5,8))

edge_data['capacity'] = random_capacity

request_sum = np.zeros((40,20))


for i in range(len(user_data)):
    edge_index = user_data.iloc[i][4]

    for j in range(20):
        request_sum[edge_index][j] = request_sum[edge_index][j] + user_data.iloc[i, j+11]

request_sum = request_sum.tolist()
df = pd.DataFrame(request_sum, columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20'])


edge_data = pd.concat([edge_data, df], axis=1)

# print(edge_data)


wb = openpyxl.load_workbook('./data_sheets.xlsx')
write = pd.ExcelWriter('./data_sheets.xlsx', engine='openpyxl')
write.book = wb # 没有这句话会覆盖

edge_data.to_excel(write,sheet_name='edge_data1', index=False)
write.save()
write.close()