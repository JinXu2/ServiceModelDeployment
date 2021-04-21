class Node:
    def __init__(self, no, latitude, longitude, type, edge_no):
        """
        :param no: 在所有冗余部署的模块中的 独有的编号 从1开始的
        :param latitude: 纬度
        :param longitude: 经度
        :param type: 服务编号 从1开始的
        :param edge_no: 所属服务器编号 从1开始
        """
        self.no = no
        self.latitude = latitude
        self.longitude = longitude
        self.type = type
        self.edge_no = edge_no

    def __str__(self):
        return 'type: %s  edge_no: %s latitude: %s longitude: %s' % (self.type, self.edge_no,self.latitude,self.longitude)



