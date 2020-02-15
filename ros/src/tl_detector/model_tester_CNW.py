#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import rospy
import cv2
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ModelTesterCNW:
    def __init__(self):
        rospy.init_node('model_tester_cnw')
        
        self.xscale = 800/384.
        self.yscale = 600/288.
        self.t_detect = 0.20 # 5 Hz
        self.t_last = rospy.Time.now()
        
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber('image_color', Image, self.img_cb, queue_size=1)
        self.detect_pub = rospy.Publisher('image_detect_cnw', Image, queue_size=1)
        
        graph_file = rospy.get_param('~graph_file', 'logs_sim/ckpt-1-9170.pb')
        module_path = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(module_path, graph_file)
        
        self.sess = tf.Session()
        with tf.gfile.FastGFile(graph_path, 'rb') as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        
        fetch_names = ['CBNOnet/light_state:0', 'CBNOnet/light_position:0']
        self.fetch_nodes = [self.sess.graph.get_tensor_by_name(name) for name in fetch_names]
        self.fetch_dict = {node.name.split(':')[0].split('/')[-1]: node for node in self.fetch_nodes}
        
        #print self.fetch_dict
        self.input_node = self.sess.graph.get_tensor_by_name('CBNOnet/images:0')
        
        rospy.loginfo("READY!")
        
        rospy.spin()
        
    def predict(self, images):
        return self.sess.run(self.fetch_dict, {self.input_node: images})
    
    def save_pb(self, *args):
        pass
    
    def img_cb(self, msg):
        t_in = rospy.Time.now()
        dt_detect = (t_in - self.t_last ).to_sec()
        if dt_detect < self.t_detect:
            return
        
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img = cv2.resize(img, (384, 288))
        preds = self.predict(np.expand_dims(img, 0))
        
        state = preds['light_state']
        poses = preds['light_position']
        
        rospy.loginfo('estimate %d', state)
        
        spotspec = (0, 0, 0)
        if state == 0:
            spotspec = (255, 0, 0)
        elif state == 1:
            spotspec = (255, 255, 0)
        elif state == 2:
            spotspec = (0, 255, 0)
            
        for pose in poses:
            #img = cv2.circle(img, (int(pose[0]*self.xscale), int(pose[1]*self.yscale)), 5, spotspec, 2)
            img = cv2.circle(img, (pose[1], pose[0]), 5, spotspec, -1)
        img = cv2.circle(img, (192, 144), 5, (255, 255, 255), 1)
            
        self.detect_pub.publish(self.bridge.cv2_to_imgmsg(img, 'rgb8'))
        t_out = rospy.Time.now()
        self.t_last = t_out
        dtee = (t_out - t_in).to_sec()
        #rospy.loginfo('inference time %.3f, max frequency %.3f', dtee, 1.0/dtee)

if __name__ == "__main__":
    mtc = ModelTesterCNW()
        
