# Udacity SDC Capstone Project Checklist


## 08-02-12020

- [x] prep local environment
- [x] test keras model integration with resampled ssd (but not retrained)
    - some surprise with opencv, but it works, low accuracy aside
- [ ] retrain vgg16 ssd with udacity data (pierluigi preparation)
    - [ ] test performance on practice bag
- [x] figure out retraining with partially frozen weights
    - so long as you're using keras, it's just setting layer.trainable = False
- [ ] retrain vgg16 ssd with new label udacity data (each light status gets different label)
    - [ ] test performance on practice bag
- [ ] collect image bag from simulation
- [ ] make summary to present to team
    - even if it doesn't work well, the point is to get them thinking along this line

## Post-break checklist

- [x] look at the practice bag
    - 2.25 minutes of driving back-and-forth in front of a traffic light
    - colours are a bit washed out
    - possible strategy:
        - write node that takes the images and save one out of N to images (say, N=3)
        - write labelling utility
        - distribute utility and parts of the samples to the team
        - get labelled dataset, check the "collect training data" line below
- [x] explore object detection lab more
    - [x] reduce confidence threshhold
        - at about 0.2 it can detect traffic light in simulation screenie
        - unfortunately, the state isn't classified. It's just traffic light, not green, yellow, or red
    - [x] try against practice bag
        - very slow, 1 Hz is borderline
    - [x] try against simulation dataset
- [x] figure out loading trainable checkpoint
    - probably same as odlab, frozen inference graph means the definition, not the weights, right?
- [x] figure out output shape on single image input
    - as configured in odlab: 100 boxes, each classified
    - that is why we need box filtering, paring the 100 proposals into good inferences

---

- [ ] get udacity dataset
    - [ ] adapt udacity dataset; give each colour traffic light different class labels
- [ ] retrain ssd with udacity dataset
    - [ ] use ss300 with vgg-16 base
    - [ ] use mobilenet base if vgg-16 is too slow
- [ ] test retrained ssd with traffic lights
    - [ ] in simulation
    - [ ] on practice bag
- [ ] integrate retrained model with rest of system
- [ ] test system integration
    - [ ] melodic on local machine
    - [ ] kinetic on workspace
- [ ] prep submission
- [ ] submit project
- [ ] wait for feedback
- [ ] graduate
- [ ] celebrate

## Team checklist

- [x] prep starter pack for ZFS
- [ ] coordinate second meeting

## Personal checklist

- [x] setup github repo
- [~] figure out running with camera enabled without lag
    - [x] enable camera in sim but disable callback (baseline perf)
        - >> lag, but only if the ros nodes are running. Manual drive without any ros nodes works OK
    - [x] figure out which node causes lag
        - waypoint_updater and tl_detector, of course
    - [x] process only 1 out of N image (or manipulate rate)
        - 1 out of 3, and limit wp update to 5 Hz, barely keeping up/slightly lagging
    - [x] use VM on Teglon, run sim on host, nodes on vm
        - >> not helping
    - [x] use `taskset -c 0-1` for nodes, `2-7` for simulator
        - still some lag, but minimised, I think
    - [/] write c++ node for image processing
    - [x] write blind mode for TL detector
        - use vehicle/traffic_lights messages to update stopline waypoint
- [x] fix dbw behaviour
    - pre-fix: throttle remains constant until sudden brake at stopline
    - [x] check waypoint_updater
        - make sure there is a last waypoint with zero speed before break statement in deceleration function
    - [x] check dbw_node
    - [x] check pure pursuit
        - next waypoint search has a minimum distance (at least 6 [m]), but if there isn't any that far, the last one is picked
        - instead of filling post-stopline waypoints with identical waypoints, one should be enough. It'll pick the last one anyway, and next cycle the list will be empty
        - [x] change argument in getCmdVelocity from 0 to num_of_next_waypoint_
            - in current behaviour, the speed reference is always the speed of the first waypoint, so if we have a list of waypoints, with deceleration at the tail end, the speed reference might not reflect that
    - [x] test changes; desired outcome:
        - no lag
        - stop smoothly at stoplines
        - never get stuck at greenlight
        - no missing waypoint warning from pure_pursuit (and not because I just suppress the warning)
- [x] tune training data collection parameters
    - far threshhold must be far enough (can't see lights)
    - close threshhold must be close enough (see lights clearly)
    - tune the volume of data collected (raw volume and proportion between classes)
- [x] write training data collector
- [x] collect training data for traffic light classifier
    - [x] recollect training data
        - I messed up and overwrote the no-light images
        - collect at least 1000 images, no-light should have > 2 [s] interval
        - label the waypoint distance in the filename too, help us filter things later
        - if not lag-inducing, process more images
    - [x] collect more constrained data
        - 20-30 waypoint from lights, to be cropped to contain the middle light
- [x] make waypoint_updater loop around the track
    - currently stops at final waypoint, but can be induced to loop with a little manual control
- [x] check in with the mentor about graduation requirement on sim-only
    - classifier is needed
- [x] prep for in-workspace training
    - [x] spin notebook into module
    - [x] run local module test
    - [x] upload samples
    - [x] run in-workspace tests
        - it would train, but net_simple accuracy is only 78% after 4 epochs
- [x] build, train  model for traffic light classifier (in-simulation)
    - [x] explore data
    - [x] write batch generator
    - [x] write model builder
        - add resize layer
        - rebalance sample proportion
        - rename label 4 to 3 (economise categories)
    - [x] train model
        - one epoch wtih naïve adaptation of net_nvidia gets us 50% accuracy
        - a simple one can get 90%, some misclassing on green though
        - [x] check green recall
            - 50ish percent with extant model
        - [x] try greyscale input
            - improvement! Colourblind people drive too!
        - [x] try top-bottom crop
        - [x] escalate model from very simple structures
            - [x] dense only
            - [x] one conv layer
            - [x] two conv layer
    - [x] test inference
    - [x] integrate to tl_detector
- [x] fix system integration
    - [x] verify shape of bridge output (want: (1, 600, 800, 3))
    - [x] verify data type of bridge ouput (want: rgb, float32, (0.0, 1.0))
    - [x] record bag with camera topic for off-sim development
    - it's about thread and tf graph, make sure to use the same graph in every thread
- [x] look into `pure_pursuit` waypoint following behaviour
    - see above point on 'fix dbw behaviour'
- [x] split kinetic-melodic utilities
- [x] smoothen out inference
    - enforce label consistence over 3 inferences
    - train several more epochs on closer views
    - reverse encoding at bridge
- [ ] check out keras' functional API
    - you should be able to make multiscale models with it
    - it compiles, but immediately kills the kernel
    - better start with something smaller, mvp and such
- [~] read the ssd paper
- [ ] figure out how to reuse other people's models
    - revise that lesson before the behavioural cloning project
    - figure out how to load and adjust MobileNet with keras
    - reusing pure keras or pure tf is pretty straightforward, but the ssd example is in tf and I'd like to use keras API
- [ ] train traffic light detector-classifier for use in-sim
- [ ] harmonise modeller.py with notebook
- [x] test system integration (melodic)
- [ ] test system integration (kinetic, on workspace)
- [ ] scout for capstone team
- [ ] submit project (possibly individually, on-sim only)
- [ ] graduate

# Training data collector design notes

- use rosparam switch
- classes:
    - 3: no light
    - 2: green
    - 1: yellow
    - 0: red
- classes match styx_msgs to minimise confusion
- collection rule:
    - if near traffic light, sample every skip (2 or 3 images)
    - if far from light, sample at most every second
    - name sample with timestamp and label
- write filename parser to help with serving data during training
