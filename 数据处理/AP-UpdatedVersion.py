import pandas as pd
import numpy as np
import  openpyxl
from sklearn.cluster import AffinityPropagation
from sklearn import metrics


import matplotlib.pyplot as plt

user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data2',index_col=0)

#尝试对第13模块进行AP聚类
#需要进行筛选
#对数据进行筛选 只有超过平均数才参与聚类 否则不参与
weight = user_data['s13'].tolist()
data = user_data[['Latitude','Longitude','s13']]

data = data[(data['s13']>3.38)]
print(data)

print(data.iloc[:,2].mean())

data = data.values
simi = []
for m in data:
    temp = []
    for n in data:
        s = -np.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)
        temp.append(s)
    simi.append(temp)

print(np.min(simi))
print(np.median(simi))
p = [x/10000 * -1 for x in weight]
print(p)
ap = AffinityPropagation(damping=0.5,max_iter=5000,convergence_iter=30,preference=p).fit(data)

cluster_centers_indices = ap.cluster_centers_indices_
print(cluster_centers_indices)
# for idx in cluster_centers_indices:
#     print(data[idx])

class_cen = cluster_centers_indices

##根据聚类中心划分数据
c_list = []
for m in data:
    temp = []
    for j in class_cen:
        n = data[j]
        d = -np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
        temp.append(d)
    ##按照是第几个数字作为聚类中心进行分类标识
    c = class_cen[temp.index(np.max(temp))]
    c_list.append(c)

##画图
colors = ['red','blue','black','green','yellow','purple','pink']


##列表才有index属性 要转换一下
class_cen = class_cen.tolist()
print(class_cen)

for i in range(68):
    d1 = data[i]
    d2 = data[c_list[i]]
    c = class_cen.index(c_list[i])
    plt.plot([d2[0],d1[0]],[d2[1],d1[1]],color=colors[c],linewidth=1)
    #if i == c_list[i] :
    #    plt.scatter(d1[0],d1[1],color=colors[c],linewidth=3)
    #else :
    #    plt.scatter(d1[0],d1[1],color=colors[c],linewidth=1)
plt.show()