#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.train_flag = False
        self.img_count = 4
        self.img_skip = 3
        self.last_image_stamp = None
        
        self.blind = rospy.get_param('~blind', False)
        if self.blind:
            rospy.loginfo("TL detector operating blind")
        
        self.collect_samples = rospy.get_param('~collect_samples', False)    
        # minimum distance in wp index for collecting non-tl images
        self.min_landscape_idx = rospy.get_param('~min_landscape_idx', 200)
        # maximum distance in wp index for collecting tl images
        self.max_light_idx = rospy.get_param('~max_light_idx', 50)
        self.samples_path = rospy.get_param('~samples_path', '~/samples/')
        self.sample_period = 1.0 # how many seconds between landscape samples
        # consequently, at distances between min_landscape_idx and max_light_idx, no image will be collected

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, 
                                queue_size=1)
        if not self.blind:
            sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        
        if not self.waypoint_tree:
            waypoints_2D = [[wp.pose.pose.position.x, wp.pose.pose.position.y]\
                            for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(waypoints_2D)

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if self.blind:
            light_wp, state = self.process_traffic_lights()
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.upcoming_red_light_pub.publish(Int32(light_wp))        

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if self.img_count < self.img_skip:
            self.img_count += 1
            return

        if self.last_image_stamp == None:
            self.last_image_stamp = rospy.Time.now()
        
        self.img_count = 0
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement classification
        #---------------------------------------------------------
        #if(not self.has_image):
            #self.prev_light_loc = None
            #return False

        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        #return self.light_classifier.get_classification(cv_image)
        #---------------------------------------------------------
        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose and self.waypoint_tree:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                tmp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                
                d = tmp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = tmp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)

            if self.collect_samples:
                idx_dist = line_wp_idx - car_wp_idx
                t_process = rospy.Time.now()
                dt_sample = (t_process - self.last_image_stamp).to_sec()
                if idx_dist > self.min_landscape_idx and dt_sample > self.sample_period:
                    self.last_image_stamp = t_process
                    im = self.bridge.imgmsg_to_cv2(self.camera_image)
                    fname = self.samples_path + '4__' + str(t_process.secs) + '-' \
                        + str(t_proess.nsecs) + 'png'
                    cv2.imwrite(fname, im)
                elif idx_dist < self.max_light_idx:
                    self.last_image_stamp = t_process
                    label = '4__'
                    if state == TrafficLight.RED:
                        label = '0__'
                    elif state == TrafficLight.YELLOW:
                        label = '1__'
                    elif state == TrafficLight.GREEN:
                        label = '2__'
                    im = self.bridge.imgmsg_to_cv2(self.camera_image)
                    fname = self.samples_path + label + str(t_process.secs) + '_' \
                        + str(t_process.nsecs) + 'png'
                    cv2.imwrite(fname, im)

            return line_wp_idx, state
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
