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
Vehicles with lane-changing permission steer toward the other map (0 to 1 or
1 to 0), while retaining their original map's elevation profile.

Both scenario JSON files use per-vehicle behavior settings, for example:

```json
{
  "ego_vehicle": {"lane_id": 0, "initial_distance": 0.0},
  "vehicles": [
    {"id": 0, "type": 0, "lane_id": 0, "initial_distance": 9.0,
     "operation_mode": "driving", "lane_layout": "single_lane"},
    {"id": 1, "type": 2, "lane_id": 1, "initial_distance": 15.0,
     "operation_mode": "driving", "lane_layout": "single_lane"},
    {"id": 2, "type": 3, "lane_id": 1, "initial_distance": 7.0,
     "operation_mode": "driving", "lane_layout": "double_lane"},
    {"id": 3, "type": 1, "lane_id": 1, "initial_distance": 35.0,
     "operation_mode": "parking", "lane_layout": "single_lane"}
  ]
}
```

`operation_mode: parking` holds a vehicle at its road position after Start,
regardless of its ID or appearance `type`. `driving` enables motion.
`lane_layout: single_lane` keeps that vehicle in its assigned lane, following
and stopping behind traffic. `double_lane` uses symmetric MOBIL to decide whether
to change into lane `1 - lane_id`, including when following slower moving traffic.
The full incentive is own acceleration gain plus politeness times the combined
acceleration changes of the old and new followers. It must exceed the incentive
threshold, and the new follower's predicted braking must be below the safe
braking limit. The existing front/rear gap check also remains mandatory. Parking
takes priority over lane-change permission. These are permissions per object;
`single_lane` does not remove the second map from the scene.

MOBIL uses unclipped predictions from this project's IDM, before CBF and jerk
limiting. Ego is included using an IDM prediction only; its actual controller is
not changed. This does not add cooperative gap creation by single-lane vehicles.
Parked followers contribute zero acceleration gain. Steering, source-path height,
and IDM/CBF motion control remain unchanged. A vehicle cannot decide again during
a maneuver; after completion, a simulation-time cooldown prevents immediate
reversals. There is no preferred lane or mandatory return to the original lane.
Legacy JSON without behavior fields still uses the original fixed-role logic.

Both double-lane launches expose:

| Launch argument | Default | Meaning |
| --- | --- | --- |
| `mobil_politeness` | 0.3 | Weight given to affected followers |
| `mobil_acceleration_threshold` | 0.2 | Minimum weighted acceleration gain, m/s² |
| `mobil_safe_braking` | 4.0 | Maximum imposed follower braking, m/s²; strictly between 0 and 6 |
| `mobil_cooldown` | 3.0 | Seconds after completion before another decision; pauses freeze it |

Model reference: [Kesting, Treiber, and Helbing, MOBIL](https://www.mtreiber.de/publications/MOBIL_TRB.pdf).

`initial_distance` is a shared longitudinal coordinate in meters. Placement is
relative to ego: `vehicle.initial_distance - ego.initial_distance`. `type` only
selects appearance. Ego has only lane and distance fields.

Explicit configurations accept 1–12 objects with contiguous IDs starting at zero,
in either lane, and any number of parked objects. ID 0 remains the nominal
speed-profile/behavior-generation vehicle and the preview topic's vehicle;
its parking and following constraints take priority. Other driving vehicles use
IDM/CBF against their nearest current leader, or accelerate toward the side-lane
speed limit when the lane is clear. Initial ordering includes ego and is logged.
Invalid modes, lane IDs outside 0/1, duplicate same-lane positions, nonfinite
distances, and missing/extra fields are rejected. JSON list order does not matter.

Old JSON files omitting both new fields on **every** vehicle retain the legacy
fixed-role controller and layout validation. When using the new fields, specify
both on every vehicle; mixed schemas are rejected.

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

This retains front-vehicle behavior generation, return to driving cycle, and
stop-at-distance modes. The launch defaults to
`config/double_lane_behavior_generation_scenario.json`, whose vehicles are all
`driving` and `single_lane`, so its default scene has no lane changes. You can
assign parking or double-lane permission using the same schema above. Both
runtimes share initialization, following, lane-changing, staging, and pause/resume
handling for explicit configurations. Behavior-generation timing and control
gains remain ROS parameters.

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
