class User:
    def __init__(self, no, latitude, longitude, area, request):
        self.area = area
        self.no = no
        self.latitude = latitude
        self.longitude = longitude
        self.request = request
