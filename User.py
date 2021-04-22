class User:
    def __init__(self, no, latitude, longitude, area, request):
        self.area = area
        self.no = no
        self.latitude = latitude
        self.longitude = longitude
        self.request = request

    def __str__(self):
        return "%s no. user, area is %s,latitude is %s, longitude is %s, request is" % (self.no, self.area,self.latitude,self.longitude) + str(self.request)
