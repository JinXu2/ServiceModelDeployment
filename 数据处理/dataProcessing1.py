import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series, DataFrame

'''
对数据源进行处理 只挑选400个用户和50个服务器
并进行可视化
'''

# 读取数据
user_data = pd.read_csv('users-melbcbd-generated.csv')
edge_data = pd.read_csv('site-optus-melbCBD.csv')

# user_data = user_data.sample(n=500, replace=True)#随机选500个用户
#
# edge_data = edge_data.sample(n=40,replace=True)#随机选40个ES

# user_data = user_data.reset_index(drop=True)
# edge_data = edge_data.reset_index(drop=True)

# 只选取位置信息
# edge_data = edge_data.iloc[0:40,1:3]
# user_data = user_data.iloc[0:400]
# print(edge_data)
# print(edge_data['LATITUDE'])

# 对随机选取后的数据进行保存
# write = pd.ExcelWriter('./data_sheets.xlsx') #创建数据存放路径
# user_data.to_excel(write,sheet_name='user_data')
# edge_data.to_excel(write,sheet_name='edge_data')
# write.save()
# write.close()

plt.title('User&ES Distribution')
Label = ['User', 'Edge Server']
colors = ['b', 'r']

plt.xlabel('Latitude')
plt.ylabel('Longitude')

x1 = edge_data['LATITUDE']
y1 = edge_data['LONGITUDE']
x2 = user_data['Latitude']
y2 = user_data['Longitude']

plt.figure(figsize=(100, 100))

plt.scatter(x=x1, y=y1, c=colors[0], alpha=0.5)
plt.scatter(x=x2, y=y2, c=colors[1], alpha=0.5)

# for (x, y) in zip(x2, y2):
#     plt.text(x, y, ['%0.6f' % x, '%0.6f' % y], fontdict={'fontsize': 4})

plt.savefig('User&ES Distribution2.png')
plt.show()
