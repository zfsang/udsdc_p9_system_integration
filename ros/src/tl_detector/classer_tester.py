#!/usr/bin/env python

import tensorflow
import numpy as np
import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ClasserTester:
    def __init__(self):
        rospy.init_node('classer_tester')
        
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber('image_color', Image, self.img_cb, queue_size=1)
        self.detect_pub = rospy.Publisher('image_detect', Image, queue_size=1)
        
        graph_file = rospy.get_param('~graph_file', 
                                     'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb')
        self.confidence_cutoff = rospy.get_param('~confidence_cutoff', 0.1)
        self.detect_rate = 1.0 #Hz
        
        # Load graph
        self.graph = tensorflow.Graph()
        with self.graph.as_default():
            od_graph_def = tensorflow.GraphDef()
            with tensorflow.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tensorflow.import_graph_def(od_graph_def, name='')
                
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        
        self.t_prev = rospy.Time.now()
        
        rospy.loginfo('READY')
        
        rospy.spin()
        
    def img_cb(self, msg):
        t_now = rospy.Time.now()
        dtee = (t_now - self.t_prev).to_sec()
        
        if dtee < 1.0/self.detect_rate:
            return
        
        self.t_prev = t_now
        rospy.loginfo('ping! %f, %f', dtee, 1.0/dtee)
        
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img = np.expand_dims(img, 0)
        
        with tensorflow.Session(graph=self.graph) as sess:
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores,
                                                 self.detection_classes],
                                                feed_dict={self.image_tensor: img})
            
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            
            boxes, scores, classes = self.filter_boxes(self.confidence_cutoff,
                                                       boxes, scores, classes)
            
            tl_boxes = []
            tl_scores = []
            tl_classes = []
            
            for i in range(len(classes)):
                if classes[i] == 10:
                    tl_boxes.append(boxes[i])
                    tl_scores.append(scores[i])
                    tl_classes.append(classes[i])
                    
            if len(tl_classes) > 0:
                height, width = img.shape[1:3]
                box_coords = self.to_image_coords(tl_boxes, height, width)
                
                img_detect = img[0]
                for bc in box_coords:
                    img_detect = cv2.rectangle(img_detect, (bc[1], bc[0]), (bc[3], bc[2]),
                                               (255, 0, 0, 4))
                self.detect_pub.publish(self.bridge.cv2_to_imgmsg(img_detect, "rgb8"))
                
    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        boxes = np.array(boxes)
        #rospy.loginfo('%s', ' '.join([str(e) for e in box_coords.shape]))
        #rospy.loginfo('%s', ' '.join([str(e) for e in boxes.shape]))
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords

if __name__ == "__main__":
    ct = ClasserTester()
