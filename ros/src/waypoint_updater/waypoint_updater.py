#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 5.0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2D = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        self.loop()
        
    
    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                self.publish_waypoints()
            rate.sleep()
                
                
    def get_closest_wp_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        closest_coord = self.waypoints_2D[closest_idx]
        prev_coord = self.waypoints_2D[closest_idx-1]
        
        close_vec = np.array(closest_coord)
        prev_vec = np.array(prev_coord)
        pose_vec = np.array([x, y])
        
        val = np.dot(close_vec - prev_vec, pose_vec - close_vec)
        
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2D)
        
        return closest_idx
    
    def publish_waypoints(self):
        final_lane = self.generate_lane()        
        self.final_waypoints_pub.publish(final_lane)
        
    def generate_lane(self):
        lane = Lane()
        
        closest_idx = self.get_closest_wp_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        
        remainder = farthest_idx - len(self.base_waypoints.waypoints)
        if remainder > 0:
            base_waypoints += self.base_waypoints.waypoints[0:remainder]

        # add stamps to base_waypoints so we can plot them with rqt_plot
        for i in range(len(base_waypoints)):
            base_waypoints[i].pose.header.stamp = rospy.Time.now()
            base_waypoints[i].twist.header.stamp = rospy.Time.now()
        
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            #rospy.loginfo('attempt stop at wp %d, now at %d', self.stopline_wp_idx, closest_idx)
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
            
        return lane
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        wps = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            #if vel < 10:
            #    rospy.loginfo('%d, v: %f', i+closest_idx, vel)
            if vel < 0.05:
                vel = 0.0
                p.twist.twist.linear.x = vel
                wps.append(p)
                break
                
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            wps.append(p)
        return wps
        

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2D:
            self.waypoints_2D = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]\
                                 for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2D)
            

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data
        

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
