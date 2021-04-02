class Edge:
    def __init__(self, no, latitude, longitude, capacity):
        self.no = no
        self.latitude = latitude
        self.longitude = longitude
        self.capacity = capacity

import numpy as np
if __name__ == '__main__':
    a = [1,2,3]
    a1 =[[1,2,3],[4,5,6]]
    b = [a]
    # print(b.reshape(1,-1))
    print(a1)
    # print(np.append(a1,b,axis = 0))
    c = np.concatenate((a1,b),axis = 0)
    print(c)
    for row in c:
        print(row[0:0+1])