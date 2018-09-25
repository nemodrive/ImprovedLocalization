#!/usr/bin/env python2

import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

from numpy import identity
import numpy as np

import occup_grid

# apollo input
RAW_LANES_TOPIC = '/republish/raw_lanes'
POSE_TOPIC = '/republish/pose'

# middle topics
POINTCLOUD_TOPIC = '/cloud_in'

# AMCL topics
AMCL_SCAN_TOPIC = '/scan'
AMCL_MAP_TOPIC = '/map'
AMCL_INITIALPOSE = '/initialpose'

# debug

# AMCL useful
COV_MATRIX = identity(6)

class PoseEstimation():

    def __init__(self):
        rospy.init_node("pose_estimation")

        # useful vars
        self.occup_grid_msg = occup_grid.get_occup_grid()
        self.last_pose = None
        self.last_pcl_points = None

        # amcl publishers
        self.scan_pub = rospy.Publisher(AMCL_SCAN_TOPIC, LaserScan, queue_size=10)
        self.initialpose_pub = rospy.Publisher(AMCL_INITIALPOSE, PoseWithCovarianceStamped, queue_size=10)
        self.map_pub = rospy.Publisher(AMCL_MAP_TOPIC, OccupancyGrid, queue_size=10)
        
        # pcl_pub used for pointcloud -> laserscan conversion
        self.pcl_pub = rospy.Publisher(POINTCLOUD_TOPIC, PointCloud2, queue_size=10)

        # ros input
        rospy.Subscriber(RAW_LANES_TOPIC, String, self.raw_lanes_callback, queue_size=10)
        rospy.Subscriber(POSE_TOPIC, String, self.pose_callback, queue_size=10)

        rospy.loginfo("Started...")

    # returns arr [x, y, z] from string '(x, y, z)'
    def get_point(self, s):
        return [float(c) for c in s[1:-1].replace(',', '').split()]
    
    def _to_pointcloud_msg(self, points):
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = 'laser'
        msg = pcl2.create_cloud_xyz32(h, points)
        return msg

    def raw_lanes_callback(self, data):
        points = [self.get_point(p) for p in data.data.split(';')[:-1]]

        m = self._to_pointcloud_msg(points)
        self.pcl_pub.publish(m)

        occup_grid.update_time(self.occup_grid_msg)
        self.map_pub.publish(self.occup_grid_msg)

    def pose_callback(self, data):
        if self.last_pose is None:
            self.last_pose = PoseWithCovarianceStamped()
            self.last_pose.pose.covariance = COV_MATRIX.flatten().tolist()
        arr = [float(s) for s in data.split()]
        self.last_pose.pose.pose.position.x = arr[0]
        self.last_pose.pose.pose.position.y = arr[1]
        self.last_pose.pose.pose.position.z = arr[2]
        self.last_pose.pose.pose.orientation.x = arr[3]
        self.last_pose.pose.pose.orientation.y = arr[4]
        self.last_pose.pose.pose.orientation.z = arr[5]
        self.last_pose.pose.pose.orientation.w = arr[6]
        self.last_pose.header.stamp = rospy.Time.now()
        self.last_pose.header.frame_id = 'amcl'

def main():
    PoseEstimation()
    rospy.spin()

if __name__ == '__main__':
    main()