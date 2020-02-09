#!/usr/bin/env python

import keras
from keras.models import load_model
import tensorflow as tf
from cv_bridge import CvBridge

import rospy
from sensor_msgs.msg import Image
import os
import cv2

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np

import os, sys

module_path = os.path.dirname(os.path.abspath(__file__))
ssd_keras_path = os.path.join(module_path, 'ssd_keras')
sys.path.insert(0, ssd_keras_path)

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

class ModelTesterKeras:
    def __init__(self):
        rospy.init_node('model_tester_keras')
        
        self.t_detect = 1.5 # minimum time between inferences
        self.t_last_detect = rospy.Time.now()
        
        self.img_pub = rospy.Publisher('image_detect', Image, queue_size=1)
        
        img_height = 300 # Height of the input images
        img_width = 300 # Width of the input images
        img_channels = 3 # Number of color channels of the input images
        subtract_mean = [123, 117, 104] # The per-channel mean of the images in the dataset
        swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
        n_classes = 8 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
        scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
        # scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
        aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
        two_boxes_for_ar1 = True
        steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
        offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
        clip_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
        variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
        normalize_coords = True
        
        keras.backend.clear_session()
        self.model = ssd_300(image_size=(img_height, img_width, img_channels),
                            n_classes=n_classes,
                            mode='inference',
                            l2_regularization=0.0005,
                            scales=scales,
                            aspect_ratios_per_layer=aspect_ratios,
                            two_boxes_for_ar1=two_boxes_for_ar1,
                            steps=steps,
                            offsets=offsets,
                            clip_boxes=clip_boxes,
                            variances=variances,
                            normalize_coords=normalize_coords,
                            subtract_mean=subtract_mean,
                            divide_by_stddev=None,
                            swap_channels=swap_channels,
                            confidence_thresh=0.5,
                            iou_threshold=0.45,
                            top_k=200,
                            nms_max_output_size=400,
                            return_predictor_sizes=False)
        
        self.model.load_weights(os.path.join(module_path, 'ssdx_wt.h5'), by_name=True)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        
        # keras voodoo 
        global graph
        graph = tf.get_default_graph()
        self.model._make_predict_function()
        
        img_sub = rospy.Subscriber('image_color', Image, self.img_cb, queue_size=1)
        self.bridge = CvBridge()
        
        rospy.loginfo("READY")
        
        rospy.spin()
        
    def img_cb(self, msg):
        t_in = rospy.Time.now()
        dt_detect = (t_in - self.t_last_detect).to_sec()
        if dt_detect < self.t_detect:
            return
        
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img = img[::2, 100:700:2, :]
        img = np.expand_dims(img, axis=0)
        
        global graph
        with graph.as_default():
            predict = self.model.predict(img)
            objects = [prospect for prospect in predict[0] if prospect[1] > 0.25]
            
            img_detect = img[0].copy()
            #rospy.loginfo(type(img_detect))
            if len(objects) < 1:
                rospy.loginfo('nothing here')
            for obj in objects:
                #rospy.loginfo('cls %d, conf %.2f', obj[0], obj[1])
                rospy.loginfo(' '.join([str(e) for e in obj]))
                x1 = int(obj[2])
                y1 = int(obj[3])
                x2 = int(obj[4])
                y2 = int(obj[5])
                image_detect = cv2.rectangle(img_detect, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            #img_detect = cv2.rectangle(img_detect, (5, 5), (295, 295), (0, 0, 255), 10)
            #rospy.loginfo(type(img_detect))
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(img_detect, "rgb8"))
                
        t_out = rospy.Time.now()
        self.t_last_detect = t_out
        dtee = (t_out - t_in).to_sec()
        #rospy.loginfo('period %.4f, freq %.4f', dtee, 1.0/dtee)
        
        
if __name__ == "__main__":
    mtk = ModelTesterKeras()
