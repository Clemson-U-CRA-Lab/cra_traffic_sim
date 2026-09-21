# cra_traffic_sim

Human-CAV traffic simulation platform for the Clemson CRA Lab.

## Supported runtime

The supported car-following runtimes are the ordinary driving-cycle simulations:

- `roslaunch cra_traffic_sim cra_traffic_sim_human_adaptive_acc.launch`
- `roslaunch cra_traffic_sim cra_traffic_sim_human_adaptive_acc_double_lane.launch`
- `roslaunch cra_traffic_sim cra_traffic_sim_cmi.launch`

The single-lane and double-lane nodes use speed-profile motion, optional preview, bounded vehicle states, and IDM/CBF follower safety for the double-lane case. Their ROS topics and message interfaces are unchanged.

Behavior generation is not a supported runtime feature. There are no behavior-generation launch files, triggers, parameters, or active Koopman/CasADi behavior-generation dependencies.

## Archived reference material

The former behavior-generation scripts and their dedicated model, matrix, and analysis files are retained under `reference/behavior_generation/` for historical comparison only. They are not installed, launched, or maintained as runnable tools.
