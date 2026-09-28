# cra_traffic_sim

Human-CAV traffic simulation platform for the Clemson CRA Lab.

## Supported runtime

The supported car-following runtimes are the ordinary driving-cycle simulations:

- `roslaunch cra_traffic_sim cra_traffic_sim_human_adaptive_acc.launch`
- `roslaunch cra_traffic_sim cra_traffic_sim_human_adaptive_acc_double_lane.launch`
- `roslaunch cra_traffic_sim cra_traffic_sim_cmi.launch`

The single-lane and double-lane nodes use speed-profile motion, optional preview, bounded vehicle states, and IDM/CBF follower safety for the double-lane case. Their ROS topics and message interfaces are unchanged.

The separate double-lane behavior-generation runtime is also supported (see below). Ordinary driving-cycle nodes do not import its CasADi optimizer.

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

## Two-lane behavior generation (no lane changes)

```bash
roslaunch cra_traffic_sim cra_traffic_sim_behavior_generation_double_lane.launch
```

This restores the original front-vehicle behavior generation, return to driving
cycle, and stop-at-distance modes. The side-lane leader uses IDM/CBF against a
virtual leader derived from the front vehicle; side followers remain in lane 1.
There is no lane-changing controller or stationary obstacle in this runtime.
The existing ordinary double-lane launch continues to provide the obstacle and
lane-change scenario.

The new launch defaults to `config/double_lane_behavior_generation_scenario.json`.
It uses the same `ego_vehicle` and `vehicles` JSON fields and the same pre-Start
staging and pause/resume behavior. All entries represent moving vehicles,
regardless of their `type`. IDs start at zero: ID 0 is ahead of ego in lane 0;
IDs 1 onward are lane-1 vehicles ordered front-to-back. Configure 2–12 vehicles.
The default five-vehicle scene preserves the old launch's 8 m front gap, 1 m
side-leader offset, and 8 m side-follower spacing. JSON defines count and initial
placement; it does not configure behavior-generation timing or control gains.

Override `scenario_file`, `site_map_0`, and `site_map_1` as in the ordinary
launch. The same two CMI map files are used by default. Do not run both traffic
launches simultaneously: they publish the same traffic topics.

Joystick button indices follow the old script: 5 starts/resumes, 4 pauses, and
3 requests front-vehicle behavior generation on a rising edge. Automatic
triggering is enabled by default: fresh `/control_target_cmd` messages with
human acceleration below the configured threshold start the randomized quiet
period. Set `auto_behavior_generation_enable:=false` for manual triggers only.
Pausing freezes the active behavior's simulation timer and resets the automatic
quiet-period timer. `front_vehicle_travel_distance` defaults to 1000 m; set it
to zero to disable the final stop target.

This runtime needs Python `casadi` and `numpy`. The restored optimizer and
runtime A/B/C matrices live in `scripts/behavior_generation_optimizer.py` and
`config/behavior_generation/`. It does not load archived training scripts or
PyTorch models. `koopman_lift_method:=auto` selects the 16-state model, matching
the old behavior. `behavior_generation_solver:=ipopt` is the default because
local FATROP smoke tests failed and crashed; IPOPT passed the same optimization
problem. The solver is configurable, but FATROP has not passed local validation.

For tests including the actual optimizer:

```bash
source /home/cra/mach_e_ws/devel/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B -m unittest discover -s tests -v
```

## Archived reference material

The former behavior-generation scripts and their dedicated model, matrix, and analysis files are retained under `reference/behavior_generation/` for historical comparison only. The archived copies are not launched. The restored two-lane runtime and its required matrices are maintained separately under `scripts/` and `config/`.
