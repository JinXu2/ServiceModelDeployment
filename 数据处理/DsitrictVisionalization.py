import pandas as pd
import numpy as np
import  openpyxl
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

import matplotlib.pyplot as plt

user_data = pd.read_excel('data_sheets.xlsx', sheet_name='user_data1',index_col=0)

xn = user_data['Latitude']
yn = user_data['Longitude']
labels = user_data['area']
color =['b','r','g','y']

for i in range(len(xn)):
    if labels[i] == "livingArea":
        plt.scatter(xn[i],yn[i],color='g')
    elif  labels[i] == "commercialArea":
        plt.scatter(xn[i], yn[i], color='r')
    elif  labels[i] == "touristArea":
        plt.scatter(xn[i], yn[i], color='blue')
    else:
        plt.scatter(xn[i], yn[i], color='black')

plt.show()