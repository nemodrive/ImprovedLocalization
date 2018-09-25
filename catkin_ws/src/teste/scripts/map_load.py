#!/usr/bin/env python
import common.proto_utils as proto_utils
from modules.map.proto import map_pb2

from enum import Enum
from scipy.spatial import distance

MAP_FILE_NAME = '/apollo/modules/map/data/test/base_map.txt'

class LaneBoundaryType(Enum):
    left = 1
    right = 2

class Map:
    def __init__(self, map_file):
        self.map_pb = map_pb2.Map()
        proto_utils.get_pb_from_file(MAP_FILE_NAME, self.map_pb)

        self.points = []

        self.load_points()

    def _dist(self, p1, p2):
        return distance.euclidean(p1, p2)

    def get_closest_points(self, ref):
        dist = [(self._dist(p[0], ref), p[0], p[1]) for p in self.points]
        return sorted(dist, key = lambda x: (x[0]))

    def load_points(self):
        for lane in self.map_pb.lane:
            for segment in lane.left_boundary.curve.segment:
                for point in segment.line_segment.point:
                    p = (point.x, point.y, point.z)
                    self.points.append((p, LaneBoundaryType.left))
            for segment in lane.right_boundary.curve.segment:
                for point in segment.line_segment.point:
                    p = (point.x, point.y, point.z)
                    self.points.append((p, LaneBoundaryType.right))

if __name__ == '__main__':
    m = Map(MAP_FILE_NAME)

    print(m.get_closest_points((424213.15, 4920674.05, 0))[0:5  ])
