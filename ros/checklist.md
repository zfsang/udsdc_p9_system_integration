# Udacity SDC Capstone Project Checklist

- [ ] setup github repo
- [ ] figure out running with camera enabled without lag
    - [x] enable camera in sim but disable callback (baseline perf)
        - >> lag, but only if the ros nodes are running. Manual drive without any ros nodes works OK
    - [x] figure out which node causes lag
        - waypoint_updater and tl_detector, of course
    - [x] process only 1 out of N image (or manipulate rate)
        - 1 out of 3, and limit wp update to 5 Hz, barely keeping up/slightly lagging
    - [ ] use VM on Teglon, run sim on host, nodes on vm
    - [ ] write c++ node for image processing
- [ ] collect training data for traffic light classifier
- [ ] build, train  model for traffic light classifier
- [ ] look into `pure_pursuit` waypoint following behaviour
- [ ] test system integration
- [ ] scout for capstone team
- [ ] submit project (possibly individually, on-sim only)
- [ ] graduate
