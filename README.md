# Self-Driving Car System Integration

## Team Members

- Ryan Arya Pratama, ryan.arya.pratama@gmail.com
- Zhifan Sang, zfsang@gmail.com
- Sabari Rajan, sabari.rajan90@gmail.com
- Ching Nian Wong, drah2easy@gmail.com

## Externs

Starter code from [Udacity](https://github.com/udacity/CarND-Capstone)

Traffic light detector-classifier network as implemented by [CN Wong](https://github.com/almightybobo/tl_detector)

## Overview
###  waypoint updater node
- input: 
	- base waypoint topic (the ground truth of the track)
	- current pose topic (the current position / speed of the car)
	- traffic waypoint topic
- output:
	- final waypoint topic
- method:
	- KD tree (used to find closest waypoint for the car to follow)


### drive-by-wire (dbw) node
- input:
	- current velocity topic
	- twist cmd topic
	- dbw enabled topi
- output:
	- throttle, brake, steering of the car (published by Controller class imported from twist_controller.py)
- method:
	- PID filter (pid.py)
	- low pass filter (lowpass.py)
	- yaw controller (convert target linear and angular velocity to steering commands.) 

### traffic light detection node
- input:
	- base waypoint topic
	- current pose topic
	- image color topic
	- traffic light topic (3D position of the traffic light)
- output:
	- traffic waypoint topic
- method: 
	- object detection with CNN (Tensorflow)


## Dependencies

Requires `dbw_mkz_msgs`. If you are on the Udacity workspace, copy over from the provided starter code. Otherwise, install `ros-{melodic, kinetic}-dbw-mkz-msgs` with apt.

## Usage

Use `ros/launch/styx.launch` if you have ROS Kinetic (or using Udacity workspace), use `ros/launch/styx_melodic.launch` if you have ROS melodic.

**Original README from starter below**

---

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
