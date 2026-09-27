# cra_traffic_sim

Human-CAV traffic simulation platform for the Clemson CRA Lab.

## Supported runtime

The supported car-following runtimes are the ordinary driving-cycle simulations:

- `roslaunch cra_traffic_sim cra_traffic_sim_human_adaptive_acc.launch`
- `roslaunch cra_traffic_sim cra_traffic_sim_human_adaptive_acc_double_lane.launch`
- `roslaunch cra_traffic_sim cra_traffic_sim_cmi.launch`

The single-lane and double-lane nodes use speed-profile motion, optional preview, bounded vehicle states, and IDM/CBF follower safety for the double-lane case. Their ROS topics and message interfaces are unchanged.

Behavior generation is not a supported runtime feature. There are no behavior-generation launch files, triggers, parameters, or active Koopman/CasADi behavior-generation dependencies.

## Double-lane maps and scenario

The current double-lane launch uses independent lane centerlines, as the archived
behavior-generation version did:

- `site_map_0` → `/map_0`: `CMI_outerloop_mach_e.csv` (ego lane).
- `site_map_1` → `/map_1`: `CMI_outerloop_mach_e_wider_lane.csv` (side lane).

Override them together with a scenario file as needed:

```bash
roslaunch cra_traffic_sim cra_traffic_sim_human_adaptive_acc_double_lane.launch \
  site_map_0:=CMI_outerloop_mach_e.csv \
  site_map_1:=CMI_outerloop_mach_e_wider_lane.csv \
  scenario_file:=/absolute/path/scenario.json
```

Map filenames resolve under `maps/`; absolute paths also work. The single
`site_map` argument has been replaced by `site_map_0` and `site_map_1` for this
launch. Other launches retain their existing map settings.

JSON `lane_id` selects the map. Each file supplies its own position, elevation,
yaw, and pitch; no additional lane-width offset is applied. Before Start, ego is
projected onto both maps to align their distance origins. This alignment is frozen
at Start. `S_v_s` remains in the common lane-0 longitudinal frame for the existing
controllers; conversion to each map's local distance happens during placement.
As in the archived runtime, both lanes use equal traveled-distance increments.
The merging vehicle starts on map 1, tracks a target on map 0, and uses its
controller pose throughout the maneuver.

The default scenario is `config/double_lane_scenario.json`:

```json
{
  "ego_vehicle": {"lane_id": 0, "initial_distance": 0.0},
  "vehicles": [
    {"id": 0, "type": 0, "lane_id": 0, "initial_distance": 8.0},
    {"id": 1, "type": 0, "lane_id": 1, "initial_distance": 18.0},
    {"id": 2, "type": 0, "lane_id": 1, "initial_distance": 10.0},
    {"id": 3, "type": 1, "lane_id": 1, "initial_distance": 138.0}
  ]
}
```

`initial_distance` is a longitudinal scenario coordinate in meters. Placement is
relative to ego: `vehicle.initial_distance - ego.initial_distance`, measured on
the selected map from ego's projection. `type` selects appearance; the last ID is
stationary regardless of type. Ego has no ID or type in the JSON.

The existing motion logic requires 4–12 objects with contiguous IDs starting at
zero. Ego and ID 0 occupy lane 0, with ID 0 ahead. ID 1 leads the moving side-lane
vehicles; ID 2 merges, with optional further followers ordered front-to-back by
increasing ID. The final ID is stationary ahead of ID 1 in lane 1. Initial lane
ordering and predecessors are calculated, validated, and logged. Invalid layouts,
extra/missing fields, equal same-lane distances, and nonfinite distances are
rejected. JSON list order does not matter.

JSON controls count and starting positions; `/num_vehicles`,
`/stationary_side_vehicle_distance`, and `/side_lane_leader_distance_offset` no
longer configure this node. Controller and speed-profile parameters remain ROS
parameters. Relative scenario paths resolve against the package directory.

Once a valid ego pose arrives, the configured objects and types publish before
joystick Start. The scene follows ego while staged, with zero simulation time
and traffic speed. Start anchors it to the road and starts motion. Stop/Start
pauses and resumes the same state, including lane changes. Relaunch to reset or
reload configuration.

Run tests without a ROS master or Unity from this package directory:

```bash
source /home/cra/mach_e_ws/devel/setup.bash
python3 -B -m unittest discover -s tests -v
```

## Archived reference material

The former behavior-generation scripts and their dedicated model, matrix, and analysis files are retained under `reference/behavior_generation/` for historical comparison only. They are not installed, launched, or maintained as runnable tools.
