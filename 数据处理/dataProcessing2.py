import pandas as pd
import  openpyxl
import  random
'''
数据预处理：
对用户进行区域划分，以及生成每个用户对每个应用的请求频率
'''

# 读取处理后的数据源
user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data',index_col=0)
edge_data = pd.read_excel('data_sheets.xlsx', sheet_name='edge_data')

'''
各区域的坐标位置为
商业：-37.820869 -37.816800 144.961355 144.953784
旅游：-37.812355 -37.807937 144.972941 144.9662
生活上：-37.816248 -37.813000 144.974443 144.9685
生活下：-37.815482 -37.8113 144.959775 144.953920
'''

#判断区域
def zoning(x, y):
    if -37.820869 <= x <= -37.816800 and 144.961355 >= y >= 144.953784:
        return "commercialArea"
    elif -37.812355 <= x <= -37.807937 and 144.972941 >= y >= 144.9662:
        return "touristArea"
    elif -37.816248 <= x <= -37.81300 and 144.974443 > y >= 144.9685:
        return "livingArea"
    elif -37.815482 <= x <= -37.8113 and 144.959775 > y >= 144.953920:
        return "livingArea"
    else:
        return "generalArea"



#生成请求
def request(area):
    if area == "commercialArea":
        return [15,1,5,1,20,10]
    elif area == "touristArea":
        return [15,1,5,20,1,10]
    elif area == 'livingArea':
        return [15,10,5,1,1,10]
    else: #generalArea
        return [15,1,5,1,1,10]


user_data['area'] = user_data.apply(lambda x: zoning(x.Latitude, x.Longitude), axis=1)

#一般区域的用户太多了 按概率随机删除一半
length = len(user_data)

for i in range(400):
    print(i)
    if user_data.iloc[i,2] == "generalArea":
        a = random.randint(1,10)
        if a<=5:
            user_data.drop(i,inplace=True)


print(user_data)
user_data[['r1','r2','r3','r4','r5','r6']]=user_data.apply(lambda x:pd.Series(request(x.area)),axis=1)

#保存 保存到已用的excel中 需要多一步
wb = openpyxl.load_workbook('./data_sheets.xlsx')
write = pd.ExcelWriter('./data_sheets.xlsx',engine='openpyxl')
write.book = wb #没有这句话会覆盖

user_data.to_excel(write,sheet_name='user_data1',index=False)
write.save()
write.close()


