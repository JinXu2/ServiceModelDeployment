# 生成用户信息

class Service:
    '服务模块'
    def __init__(self,cost,type):
        self.cost=cost
        self.type=type

    def isBelong(self,App):
        '判断是否属于该应用'

    def computeRequest(self):
        '计算请求频率'


# 生成各类服务模块
s1 = Service(10,"start1")
s2 = Service(10,"start2")
s3 = Service(10,"start3")
s4 = Service(10,"start4")
s5 = Service(10,"start5")
s6 = Service(10,"start6")
s7 = Service(10,"end1")
s8 = Service(10,"end2")
s9 = Service(10,"end3")
s10 = Service(10,"end4")
s11 = Service(10,"end5")
s12 = Service(10,"end6")
s13 = Service(10,"Recognition")
s14 = Service(10,"Rendering")
s15 = Service(10,"Authentication")
s16 = Service(10,"Music")
s17 = Service(10,"VR")
s18 = Service(10,"Pay")
s19 = Service(10,"Social")
s20 = Service(10,"Other")

#根据区域划分+应用需求度情况 生成每个用户对于不同应用的请求频率



class Application:
    '应用'
    def __init__(self,services):
        self.services=services

#生成应用
a1 = Application(['s1','s13','s14','s15','s7'])
a2 = Application(['s2','s14','s16','s17','s8'])
a3 = Application(['s3','s14','s16','s20','s9'])
a4 = Application(['s4','s13','s14','s17','s10'])
a5 = Application(['s5','s13','s15','s18','s11'])
a6 = Application(['s6','s14','s16','s19','s12'])