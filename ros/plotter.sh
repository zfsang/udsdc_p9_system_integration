#!/usr/bin/env bash
rqt_plot /vehicle/brake_cmd/pedal_cmd&
rqt_plot /vehicle/throttle_cmd/pedal_cmd&
rqt_plot /twist_cmd/twist/linear/x /current_velocity/twist/linear/x&
