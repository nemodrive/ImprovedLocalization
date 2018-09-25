#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from modules.localization.proto.localization_pb2 import LocalizationEstimate

RAW_LANES_IN = '/apollo/raw_lanes'
RAW_LANES_OUT = '/republish/raw_lanes'
RAW_LANES_TYPE = String

POSE_IN = '/apollo/localization/pose'
POSE_OUT = '/republish/pose'
POSE_TYPE = LocalizationEstimate



def create_republisher(topic_in, topic_out, in_topic_type, out_topic_type=None, tf=None):
    if not out_topic_type is None and tf is None:
        print("Error: Can't change msg type with transform func")
        return
    out_topic_type = in_topic_type if out_topic_type is None else out_topic_type
    pub = rospy.Publisher(topic_out, out_topic_type, queue_size=10)
    cb = callback
    vs = [pub]
    if not tf is None:
        cb = callback_tf
        vs.append(tf)
    rospy.Subscriber(topic_in, in_topic_type, cb, vs, queue_size=10)

def callback(data, vs):
    vs[0].publish(data)

def callback_tf(data, vs):
    pub = vs[0]
    tf = vs[1]
    d = tf(data)
    pub.publish(d)

def pose_to_str(data):
    s = String()
    s.data = "{0} {1} {2} {3} {4} {5} {6}".format(
        data.pose.position.x,
        data.pose.position.y,
        data.pose.position.z,
        data.pose.orientation.x,
        data.pose.orientation.y,
        data.pose.orientation.z,
        data.pose.orientation.w
    )
    return s

def main():
    rospy.init_node('republisher', anonymous=True)

    create_republisher(RAW_LANES_IN, RAW_LANES_OUT, RAW_LANES_TYPE)
    create_republisher(POSE_IN, POSE_OUT, in_topic_type=POSE_TYPE, out_topic_type=String)
    
    rospy.spin()

if __name__ == '__main__':
    main()
