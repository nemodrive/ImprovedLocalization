import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData

import numpy as np
import cv2

INPUT_IMAGE = 'upb.png'

def get_occup_grid():
    img = cv2.imread(INPUT_IMAGE, -1)

    # these 3 lines were used when the image was white backgorund with black lines
    # THRESHOLD = 180
    # np.place(img, img <= THRESHOLD, 100)
    # np.place(img, img > THRESHOLD, 0)

    metadata = MapMetaData()
    metadata.map_load_time = rospy.Time.now()
    metadata.resolution = 0.6
    metadata.width = img.shape[1]
    metadata.height = img.shape[0]
    metadata.origin.position.x = 0.0
    metadata.origin.position.y = 0.0
    metadata.origin.position.z = 0.0
    # heading = 0
    metadata.origin.orientation.x = 0
    metadata.origin.orientation.y = 0
    metadata.origin.orientation.z = 0
    metadata.origin.orientation.w = 0
    
    msg = OccupancyGrid()
    msg.info = metadata
    msg.data = img.flatten().tolist()

    return img

def update_time(msg):
    msg.info.map_load_time = rospy.Time.now()