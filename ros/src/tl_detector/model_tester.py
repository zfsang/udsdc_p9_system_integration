#!/usr/bin/env python

import keras
from keras.models import load_model
import tensorflow as tf
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import Image

import numpy as np
import os

class ModelTester:
    def __init__(self):
        rospy.init_node('model_tester')
        rospy.loginfo(os.getcwd())
        
        keras.backend.clear_session()
        self.model = load_model('model_simple84.h5')
        global graph
        graph = tf.get_default_graph()
        self.model._make_predict_function()
        self.model.summary()
        
        img_sub = rospy.Subscriber('image_color', Image, self.img_cb, queue_size=1)
        
        self.bridge = CvBridge()
        
        rospy.spin()
        
    def img_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img = np.float32(img)/255.0
        img = np.expand_dims(img, axis=0)
        
        rospy.loginfo(' '.join([str(e) for e in img.shape]))
        
        global graph
        with graph.as_default():
            predict = self.model.predict(img)
            rospy.loginfo(' '.join([str(e) for e in predict[0]]))
        
        
if __name__ == "__main__":
    mt = ModelTester()
