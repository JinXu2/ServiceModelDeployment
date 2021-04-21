class User:
    def __init__(self, no, latitude, longitude, area, request):
        self.area = area
        self.no = no
        self.latitude = latitude
        self.longitude = longitude
        self.request = request

from Logger import Logger
import sys
sys.stdout = Logger('E:\ServiceModelDeployment\est.txt')
if __name__ == '__main__':
    for i in range(2000):
        print(i)