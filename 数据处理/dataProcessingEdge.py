import pandas as pd
import  openpyxl
import random as rd
edge_data = pd.read_excel('data_sheets.xlsx', sheet_name='edge_data')

random_capacity = []
for i in range(40):
    random_capacity.append(rd.randint(2,3))

edge_data['capacity']=random_capacity

wb = openpyxl.load_workbook('./data_sheets.xlsx')
write = pd.ExcelWriter('./data_sheets.xlsx',engine='openpyxl')
write.book = wb #没有这句话会覆盖

edge_data.to_excel(write,sheet_name='edge_data1',index=False)
write.save()
write.close()