# Archived behavior-generation reference

This directory contains the former behavior-generation scripts and their dedicated research assets. It is retained for historical comparison only.

These files are not supported runtime nodes, are not referenced by any launch file, and are not installed by the package. The active simulator uses the scripts under `scripts/` instead.

The two-lane runtime has since been restored as `scripts/car_following_double_lane_behavior_generation.py`, with JSON initialization and matrices under `config/behavior_generation/`. This directory remains an unchanged historical code/data reference; its scripts are not launched.

Archived scripts deliberately have no execute permission so ROS node lookup cannot select them instead of active scripts with the same filename.
