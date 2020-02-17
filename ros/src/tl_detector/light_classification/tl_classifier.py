import os

import cv2
import numpy as np

from styx_msgs.msg import TrafficLight

import tensorflow as tf


class TLClassifier(object):
    def __init__(self, graph_path):
        # load classifier
        
        self.sess = tf.Session()
        with tf.gfile.FastGFile(graph_path, 'rb') as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        
        fetch_names = ['CBNOnet/light_state:0', 'CBNOnet/light_position:0']
        self.fetch_nodes = [self.sess.graph.get_tensor_by_name(name) for name in fetch_names]
        self.fetch_dict = {node.name.split(':')[0].split('/')[-1]: node for node in self.fetch_nodes}
        
        self.input_node = self.sess.graph.get_tensor_by_name('CBNOnet/images:0')
        self.light_map = [TrafficLight.RED,     # 0
                          TrafficLight.YELLOW,  # 1
                          TrafficLight.GREEN]   # 2

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.resize(image, (384, 288))
        images = np.expand_dims(image, 0)
        ret = self.sess.run(self.fetch_dict, {self.input_node: images})
        state = ret['light_state']
        if state in self.light_map:
            return self.light_map[state]
        else:
            return TrafficLight.UNKNOWN
