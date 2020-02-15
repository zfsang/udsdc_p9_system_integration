#!/usr/bin/env python

import rospy
import cv2
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageExtractor:
    def __init__(self):
        rospy.init_node('image_extractor')
        
        self.T_sample = 1.0
        self.logpath = '/home/ryan/practice_images'
        
        self.img_sub = rospy.Subscriber('/image_color', Image, self.img_cb, queue_size=1)
        self.bridge = CvBridge()
        
        self.t_last = rospy.Time.now()
        
        rospy.loginfo('READY!')
        
        rospy.spin()
        
    def img_cb(self, msg):
        t_img = rospy.Time.now()
        dtee = (t_img - self.t_last).to_sec()
        if dtee < self.T_sample:
            return
        
        self.t_last = t_img
        
        img = self.bridge.imgmsg_to_cv2(msg)
        img = cv2.resize(img, (384, 288))
        
        t_proc = rospy.Time.now()
        fname = str(t_proc.secs) + '_' + str(t_proc.nsecs) + '.png'
        
        rospy.loginfo('%s', fname)
        cv2.imwrite(os.path.join(self.logpath, fname), img)
        
if __name__ == "__main__":
    ix = ImageExtractor()
