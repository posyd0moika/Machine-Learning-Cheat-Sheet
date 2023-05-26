from random import shuffle


class Boundary:
    pass


class Kernel:
    pass


class DBSCAN2D:

    def __init__(self, M, E):
        self.M = M
        self.E = E
        self.clasters = []

    def fit(self, points):
        shuffle(points)
        self.points = points
        self.visited = {}

        for item in points:
            if item in self.visited:
                continue

            cl = Claster2D(self)
            cl._find(item)
            if len(cl) > 1:
                self.clasters.append(cl)




class Claster2D:

    def __init__(self, data: DBSCAN2D):
        self.data = data
        self.claster_point = {}

    def __length_vec(self, point1, point2):
        res = 0
        res += (point1[0] - point2[0]) ** 2
        res += (point1[1] - point2[1]) ** 2

        return res ** (0.5)

    def __len__(self):
        return len([
            k for k, v in self.claster_point.items()
            if v is not None
        ])

    def _find(self, point):
        if self.claster_point.get(point, False) is not False:
            return
        arr = [
            (point, i)
            for i in self.data.points
            if point != i and self.__length_vec(point, i) <= self.data.E
        ]

        if len(arr) >= self.data.M:
            self.claster_point[point] = Kernel
            for i in arr:
                if self.claster_point.get(i, False) is False or self.claster_point[i] is Boundary:
                    self._find(i[1])
        else:
            self.claster_point[point] = None

        self.data.visited[point] = True

