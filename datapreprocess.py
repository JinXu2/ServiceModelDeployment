import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series, DataFrame
import random
import openpyxl

#读取数据
user_data = pd.read_csv('users-melbcbd-generated.csv')
edge_data = pd.read_csv('site-optus-melbCBD.csv')

'''
进行数据筛选 获取想要的数据 并保存 一共有549个符合要求的用户，之后这个文件不再运行，因为genaralarea的用户是随机进行删选的，每次结果会不同
同时划分用户的区域 生成对应的应用请求频率 和 模块请求频率（没用
随机生成服务器的容量
'''

# 初始可视化
# plt.title('User&ES Distribution')
# Label = ['User','Edge Server']
# colors = ['b','r']
#
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
#
# x1 = edge_data['LATITUDE']
# y1 = edge_data['LONGITUDE']
# x2 = user_data['Latitude']
# y2 = user_data['Longitude']
#
# plt.figure(figsize=(100,100))
#
# plt.scatter(x=x1,y=y1,c=colors[0],alpha=0.5)
# plt.scatter(x=x2,y=y2,c=colors[1],alpha=0.5)
#
#
#
# plt.savefig('User&ES Distribution.png')
# plt.show()

# 判断区域


# def zoning(x, y):
#     if -37.820869 <= x <= -37.816800 and 144.96200 > y >= 144.953784:
#         return "commercialArea"
#     elif -37.812355 <= x <= -37.807937 and 144.972941 >= y >= 144.9662:
#         return "touristArea"
#     elif -37.816248 <= x <= -37.81300 and 144.978 > y >= 144.9685:
#         return "livingArea"
#     elif -37.815482 <= x <= -37.8113 and 144.959775 > y >= 144.953920:
#         return "livingArea"
#     else:
#         return "generalArea"
#
#
# user_data['area'] = user_data.apply(lambda x: zoning(x.Latitude, x.Longitude), axis=1)
# print(user_data)
# print(len(user_data))
#
# delete_list = []
# for i in range(len(user_data)):
#     if user_data.iloc[i, 2] == "generalArea":
#         a = random.randint(1, 6)
#         if a <= 4:
#             delete_list.append(i)
#
# user_data = user_data.drop(labels=delete_list)

# 划分区域后可视化
# for index, row in user_data.iterrows():
#     if row['area'] == 'commercialArea':
#         plt.scatter(row['Latitude'], row['Longitude'], color='r')
#     elif row['area'] == 'livingArea':
#         plt.scatter(row['Latitude'], row['Longitude'], color='g')
#     elif row['area'] == 'touristArea':
#         plt.scatter(row['Latitude'], row['Longitude'], color='blue')
#     else:
#         plt.scatter(row['Latitude'], row['Longitude'], color='black')
#
# plt.savefig('User&ES Distribution2.png')
# plt.show()
# print(len(user_data))

#保存 保存到已有的excel中 需要多一步

# 生成对应的应用请求频率和模块频率
#生成应用请求
# def request(area):
#     if area == "commercialArea":
#         return [15,1,5,1,20,10]
#     elif area == "touristArea":
#         return [15,1,5,20,1,10]
#     elif area == 'livingArea':
#         return [15,20,5,1,1,10]
#     else: #generalArea
#         return [15,1,5,1,1,10]
#
# user_data[['r1','r2','r3','r4','r5','r6']]=user_data.apply(lambda x:pd.Series(request(x.area)),axis=1)
#
# # 生成服务模块请求
# user_data[['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20']]=user_data.apply(lambda x:pd.Series(moduleReuqest(x.r1,x.r2,x.r3,x.r4,x.r5,x.r6,)),axis=1)
# # 保存 保存到已用的excel中 需要多一步
#
#
# wb = openpyxl.load_workbook('./data.xlsx')
# write = pd.ExcelWriter('./data.xlsx',engine='openpyxl')
# write.book = wb #没有这句话会覆盖
#
# print(type(user_data))
# user_data.to_excel(write,sheet_name='user_data',index=False)
# write.save()
# write.close()
# print("输入成功")


# 随机生成edge_data 的容量
# edge_data = pd.read_excel('data.xlsx', sheet_name='edge_data')
# random_capacity = []
# for i in range(len(edge_data)):
#     random_capacity.append(random.randint(5,8))
#
# edge_data['capacity'] = random_capacity
# wb = openpyxl.load_workbook('./data.xlsx')
# write = pd.ExcelWriter('./data.xlsx',engine='openpyxl')
# write.book = wb #没有这句话会覆盖
#
#
# edge_data.to_excel(write,sheet_name='edge_data',index=False)
# write.save()
# write.close()
# print("输入成功")