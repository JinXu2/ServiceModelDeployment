import pandas as pd
import  openpyxl

'''
生成每个用户对每个应用的请求频率
根据用户对每一个应用的请求
计算出 不同用户对于同一个服务模块的 请求总和分布情况
为后续聚类进行准备
'''

'''
各应用组成情况：
a1 = Application(['s1','s13','s14','s15','s7'])
a2 = Application(['s2','s14','s16','s17','s8'])
a3 = Application(['s3','s14','s16','s20','s9'])
a4 = Application(['s4','s13','s14','s17','s10'])
a5 = Application(['s5','s13','s15','s18','s11'])
a6 = Application(['s6','s14','s16','s19','s12'])
'''

def moduleReuqest(r1,r2,r3,r4,r5,r6):
    return [r1,r2,r3,r4,r5,r6,r1,r2,r3,r4,r5,r6,r1+r4+r5,r1+r2+r3+r4+r6,r1+r5,r2+r3+r6,r2+r4,r5,r6,r3]

user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data1')

#生成应用请求
def request(area):
    if area == "commercialArea":
        return [15,1,5,1,20,10]
    elif area == "touristArea":
        return [15,1,5,20,1,10]
    elif area == 'livingArea':
        return [15,20,5,1,1,10]
    else: #generalArea
        return [15,1,5,1,1,10]

user_data[['r1','r2','r3','r4','r5','r6']]=user_data.apply(lambda x:pd.Series(request(x.area)),axis=1)

# 生成服务模块请求
user_data[['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20']]=user_data.apply(lambda x:pd.Series(moduleReuqest(x.r1,x.r2,x.r3,x.r4,x.r5,x.r6,)),axis=1)
# 保存 保存到已用的excel中 需要多一步
wb = openpyxl.load_workbook('./data_sheets.xlsx')
write = pd.ExcelWriter('./data_sheets.xlsx',engine='openpyxl')
write.book = wb # 没有这句话会覆盖

user_data.to_excel(write,sheet_name='user_data2',index=False)
write.save()
write.close()