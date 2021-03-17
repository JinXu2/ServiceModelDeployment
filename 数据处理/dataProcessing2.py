import pandas as pd
import openpyxl
import random
import matplotlib.pyplot as plt

'''
数据预处理：
对用户进行区域划分，对其他区域人数进行概率减少
选择离每个用户最近的ES
'''

# 读取处理后的数据源
user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data')
edge_data = pd.read_excel('data_sheets.xlsx', sheet_name='edge_data')

'''
各区域的坐标位置为
商业：-37.820869 -37.816800 144.96200 144.953784
旅游：-37.812355 -37.807937 144.972941 144.9662
生活上：-37.816248 -37.813000 144.978 144.9685
生活下：-37.815482 -37.8113 144.959775 144.953920
'''


# 判断区域
def zoning(x, y):
    if -37.820869 <= x <= -37.816800 and 144.96200 > y >= 144.953784:
        return "commercialArea"
    elif -37.812355 <= x <= -37.807937 and 144.972941 >= y >= 144.9662:
        return "touristArea"
    elif -37.816248 <= x <= -37.81300 and 144.978 > y >= 144.9685:
        return "livingArea"
    elif -37.815482 <= x <= -37.8113 and 144.959775 > y >= 144.953920:
        return "livingArea"
    else:
        return "generalArea"


user_data['area'] = user_data.apply(lambda x: zoning(x.Latitude, x.Longitude), axis=1)

# user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data1')
# 一般区域的用户太多了 按概率随机删除了80% 构造区域的聚集情况

# 多行删除
delete_list = []
for i in range(400):
    if user_data.iloc[i, 3] == "generalArea":
        a = random.randint(1, 6)
        if a <= 4:
            delete_list.append(i)

user_data = user_data.drop(labels=delete_list)

# 进行划分区域后的可视化
for index, row in user_data.iterrows():
    if row['area'] == 'commercialArea':
        plt.scatter(row['Latitude'], row['Longitude'], color='r')
    elif row['area'] == 'livingArea':
        plt.scatter(row['Latitude'], row['Longitude'], color='g')
    elif row['area'] == 'touristArea':
        plt.scatter(row['Latitude'], row['Longitude'], color='blue')
    else:
        plt.scatter(row['Latitude'], row['Longitude'], color='black')

plt.savefig('User&ES Distribution2.png')
plt.show()

user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data1')
edge_data = pd.read_excel('data_sheets.xlsx', sheet_name='edge_data')

nearest = []

for i in range(len(user_data)):
    min = 10
    temp = -1
    for j in range(40):
        distance = abs(user_data.iloc[i, 1] - edge_data.iloc[j, 1]) + abs(user_data.iloc[i, 2] - edge_data.iloc[j, 2])
        if distance < min:
            temp = j
            min = distance
    nearest.append(temp)

print(nearest)
# 把这列加到 user_data1中去

user_data['nearest'] = nearest

#保存 保存到已有的excel中 需要多一步
wb = openpyxl.load_workbook('./data_sheets.xlsx')
write = pd.ExcelWriter('./data_sheets.xlsx',engine='openpyxl')
write.book = wb #没有这句话会覆盖

user_data.to_excel(write,sheet_name='user_data1',index=False)
write.save()
write.close()
