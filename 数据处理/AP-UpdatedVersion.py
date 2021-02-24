import pandas as pd
import numpy as np
import  openpyxl
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

import matplotlib.pyplot as plt

user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data2',index_col=0)

cluster_result = []

for i in range(1,21):
    print("当前处理模块",i)
    coloumn = 's'+str(i)
    # 读取对应列数的数据
    data = user_data[['Latitude', 'Longitude', coloumn]]
    mean = data[coloumn].mean()
    print("平均值为",mean)
    # 对数据进行筛选 只有超过 平均值 才参与聚类 否则不参与
    selected_data = data[(data[coloumn] > mean )]
    selected_num = len(selected_data)
    print("筛选后数据共有",selected_num)
    data = data[(data[coloumn] <= mean )]
    # 筛选后数据可能为 0
    if selected_num == 0:
        cluster_result.append(0);
        print("modeul "+coloumn+"无需聚类,筛选后无数据")
        continue;
    else:
        # 首先进行可视化展示
        # 对于该模块请求大于 平均值 的 在全用户分布对比下的 分布情况
        plt.title('Request Distribution'+coloumn)
        Label = ['SelectedUser', 'User']
        colors = ['blue', 'red']
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')

        x1 = selected_data['Latitude']
        y1 = selected_data['Longitude']
        x2 = data['Latitude']
        y2 = data['Longitude']

        plt.scatter(x=x1, y=y1, c=colors[0], alpha=0.5)
        plt.scatter(x=x2, y=y2, c=colors[1], alpha=0.5)
        print("生成筛选后示意图")
        plt.savefig("./temp{}.png".format(i))
        plt.clf()

        # 进行AP聚类
        weight = selected_data[coloumn]
        selected_data = selected_data[['Latitude','Longitude']].values
        #bprint(selected_data)
        '''
        这个参数总算是调对了！
        '''
        p = [x/10000 * -0.085 for x in weight]
        ap = AffinityPropagation(damping=0.5,max_iter=5000,convergence_iter=30,preference=p).fit(selected_data)

        # 聚类结果
        cluster_centers_indices = ap.cluster_centers_indices_
        # print("聚类结果的中心的index")
        # print(ap.cluster_centers_indices_)

        if len(cluster_centers_indices) == 0:
            cluster_result.append(0)
            print("moduel "+coloumn+"无法收敛")
            continue
        else:

            cluster_result.append(len(cluster_centers_indices))
            class_cen = cluster_centers_indices

            ##根据聚类中心划分数据
            c_list = []
            for m in selected_data:
                temp = []
                for j in class_cen:

                    n = selected_data[j]
                    # print(n)
                    d = -np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
                    temp.append(d)
                ##按照是第几个数字作为聚类中心进行分类标识
                c = class_cen[temp.index(np.max(temp))]
                c_list.append(c)

            ##画图
            colors = ['red','blue','green','yellow','purple','pink']


            ##列表才有index属性 要转换一下
            class_cen = class_cen.tolist()
            # print(class_cen)
            plt.clf()
            for i in range(selected_num):
                d1 = selected_data[i]
                d2 = selected_data[c_list[i]]
                c = class_cen.index(c_list[i])
                plt.plot([d2[0],d1[0]],[d2[1],d1[1]],color=colors[c],linewidth=1)
                # if i == c_list[i] :
                #    plt.scatter(d1[0],d1[1],color=colors[c],alpha=1)
                # else :
                #    plt.scatter(d1[0],d1[1],color=colors[c],alpha=0.5)

            plt.scatter(x=x2, y=y2, c="black", alpha=0.5)
            plt.savefig("./temp_cluster{}.png".format(coloumn))
            plt.clf()
            print("moduel " + coloumn + "聚类成功")

#保存聚类结果
print(cluster_result)
list = []
list.append(cluster_result)
df = pd.DataFrame(list, columns=['s1', 's2', 's3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20'])
wb = openpyxl.load_workbook('./data_sheets.xlsx')
write = pd.ExcelWriter('./data_sheets.xlsx',engine='openpyxl')
write.book = wb #没有这句话会覆盖

df.to_excel(write,sheet_name='cluster_result',index=False)
write.save()
write.close()