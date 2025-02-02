import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Node:

    def __init__(self, x, y, data=None) -> None:

        self.x = x
        self.y = y
        self.data = data

    def __repr__(self):
        return repr([self.x, self.y])


class boundingBox:

    def __init__(self, x_center, y_center, half_dimension):

        self.center = Node(x_center, y_center)
        self.half_dimension = half_dimension

    def containsPoint(self, point: Node):

        return ((- self.half_dimension
                 <= point.x - self.center.x
                 < self.half_dimension)
                and
                (- self.half_dimension
                 <= point.y - self.center.y
                 < self.half_dimension)
                )

    def intersectsBox(self, box):

        x_intersection = [max(box.center.x-box.half_dimension,
                              self.center.x-self.half_dimension),
                          min(box.center.x+box.half_dimension,
                              self.center.x+self.half_dimension)]

        y_intersection = [max(box.center.y-box.half_dimension,
                              self.center.y-self.half_dimension),
                          min(box.center.y+box.half_dimension,
                              self.center.y+self.half_dimension)]

        return ((x_intersection[0] <= x_intersection[1])
                and
                (y_intersection[0] <= y_intersection[1]))


class quadTree:

    def __init__(self, boundary: boundingBox):

        self.boundary = boundary

        self.capacity = 4
        self.points = []

        self.northWest = None
        self.northEast = None
        self.southWest = None
        self.southEast = None

    def insert(self, point: Node):

        if not self.boundary.containsPoint(point):
            return False

        if (len(self.points) < self.capacity) and (self.northWest is None):
            self.points.append(point)
            return True

        if self.northWest is None:
            self.subdivide()

        if (self.northWest.insert(point)):
            return True
        if (self.northEast.insert(point)):
            return True
        if (self.southWest.insert(point)):
            return True
        if (self.southEast.insert(point)):
            return True

        return False

    def subdivide(self):

        self.northWest = quadTree(boundingBox(
            self.boundary.center.x - self.boundary.half_dimension/2,
            self.boundary.center.y + self.boundary.half_dimension/2,
            self.boundary.half_dimension/2
        ))
        self.northEast = quadTree(boundingBox(
            self.boundary.center.x + self.boundary.half_dimension/2,
            self.boundary.center.y + self.boundary.half_dimension/2,
            self.boundary.half_dimension/2
        ))
        self.southWest = quadTree(boundingBox(
            self.boundary.center.x - self.boundary.half_dimension/2,
            self.boundary.center.y - self.boundary.half_dimension/2,
            self.boundary.half_dimension/2
        ))
        self.southEast = quadTree(boundingBox(
            self.boundary.center.x + self.boundary.half_dimension/2,
            self.boundary.center.y - self.boundary.half_dimension/2,
            self.boundary.half_dimension/2
        ))

    def queryRange(self, box: boundingBox):

        points_in_range = []

        if not self.boundary.intersectsBox(box):
            return points_in_range

        for i in range(len(self.points)):

            if box.containsPoint(self.points[i]):
                points_in_range.append(self.points[i])

        if self.northWest is not None:

            points_in_range.extend(self.northWest.queryRange(box))
            points_in_range.extend(self.northEast.queryRange(box))
            points_in_range.extend(self.southWest.queryRange(box))
            points_in_range.extend(self.southEast.queryRange(box))

        return points_in_range

    def visualize(self, ax):

        rect = patches.Rectangle(
            (self.boundary.center.x-self.boundary.half_dimension,
             self.boundary.center.y-self.boundary.half_dimension),
            width=2*self.boundary.half_dimension,
            height=2*self.boundary.half_dimension,
            facecolor=(1,1,1,0),
            edgecolor='blue',
            linewidth=1)

        ax.add_patch(rect)

        for point in self.points:
            plt.plot(point.x, point.y, 'r.')

        if self.northWest is not None:

            self.northWest.visualize(ax)
            self.northEast.visualize(ax)
            self.southWest.visualize(ax)
            self.southEast.visualize(ax)


if __name__ == '__main__':

    import numpy as np

    lim = [-100, 100]
    points = np.random.uniform(low=lim[0], high=lim[1], size=(10000, 2))

    center = [(lim[0]+lim[1])/2, (lim[0]+lim[1])/2]

    qt = quadTree(boundary=boundingBox(
        center[0], center[1], (lim[1]-lim[0])/2))
    for point in points:
        qt.insert(Node(point[0], point[1]))

    fig, ax = plt.subplots()

    plt.xlim(-101, 101)
    plt.ylim(-101, 101)

    qt.visualize(ax)


    bx = boundingBox(20, 30, 25)

    rect = patches.Rectangle(
        (bx.center.x-bx.half_dimension,
         bx.center.y-bx.half_dimension),
        width=2*bx.half_dimension,
        height=2*bx.half_dimension,
        facecolor=(1,1,1,0),
        edgecolor='g',
        linewidth=1)

    ax.add_patch(rect)

    pts = qt.queryRange(bx)

    for pt in pts:
        plt.plot(pt.x, pt.y, 'g.')

    plt.show()
