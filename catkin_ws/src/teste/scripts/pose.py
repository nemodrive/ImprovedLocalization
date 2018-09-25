#!/usr/bin/env python

import rospy
from pb_msgs.msg import LocalizationEstimate
from pb_msgs.msg import PerceptionObstacles
from pb_msgs.msg import LaneMarkers
from pb_msgs.msg import LaneMarker

from math import radians

'''
LaneMarkers {
  optional LaneMarker left_lane_marker = 1;
  optional LaneMarker right_lane_marker = 2;
}
'''

class Test:
    start_x = 424271.47
    start_y = 4920682.90

    heading_start = radians(220)

    pose_topic = '/apollo/localization/pose'

    def __init__(self):
        self.pose_pub = None

    def run(self):
        rospy.init_node('test_node', anonymous=True)
        self.pose_pub = rospy.Publisher(self.pose_topic, LocalizationEstimate, queue_size=10)

        r = rospy.Rate(10)

        msg = LocalizationEstimate()

        msg.pose.position.x = self.start_x
        msg.pose.position.y = self.start_y
        msg.pose.position.z = 0.0
        
        msg.pose.heading = self.heading_start

        for _ in range(0, 1000):
            self.pose_pub.publish(msg)
            r.sleep()
            if rospy.is_shutdown():
                break


if __name__ == '__main__':
    t = Test()
    t.run()