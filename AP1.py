from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt

## 生成的测试数据的中心点
centers = [[1, 1], [-1, -1], [1, -1]]
##生成数据
Xn, labels_true = make_blobs(n_samples=150, centers=centers, cluster_std=0.5,
                            random_state=0)

print(Xn)

simi = []
for m in Xn:
    ##每个数字与所有数字的相似度列表，即矩阵中的一行
    temp = []
    for n in Xn:
         ##采用负的欧式距离计算相似度
        s =-np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
        temp.append(s)
    simi.append(temp)

p=-50   ##3个中心
#p = np.min(simi)  ##9个中心，
#p = np.median(simi)  ##13个中心

ap = AffinityPropagation(damping=0.5,max_iter=500,convergence_iter=30,
                         preference=p).fit(Xn)

cluster_centers_indices = ap.cluster_centers_indices_
print(cluster_centers_indices)
for idx in cluster_centers_indices:
    print(Xn[idx])

class_cen = cluster_centers_indices
##根据聚类中心划分数据
c_list = []
for m in Xn:
    temp = []
    for j in class_cen:
        n = Xn[j]
        d = -np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
        temp.append(d)
    ##按照是第几个数字作为聚类中心进行分类标识
    c = class_cen[temp.index(np.max(temp))]
    c_list.append(c)
##画图
colors = ['red','blue','black','green','yellow']
plt.figure(figsize=(8,6))
plt.xlim([-3,3])
plt.ylim([-3,3])

##列表才有index属性 要转换一下
class_cen = class_cen.tolist()
for i in range(150):
    d1 = Xn[i]
    d2 = Xn[c_list[i]]
    c = class_cen.index(c_list[i])
    plt.plot([d2[0],d1[0]],[d2[1],d1[1]],color=colors[c],linewidth=1)
    #if i == c_list[i] :
    #    plt.scatter(d1[0],d1[1],color=colors[c],linewidth=3)
    #else :
    #    plt.scatter(d1[0],d1[1],color=colors[c],linewidth=1)
plt.show()