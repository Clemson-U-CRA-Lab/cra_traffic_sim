"""Validated initial layouts and lifecycle for the existing double-lane runtime.

This module has no ROS dependencies. Vehicle IDs retain their existing runtime
roles; initial predecessors describe the layout, not a new motion controller.
"""

import json
import math


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("{} must contain exactly {}".format(label, ", ".join(expected)))


def _integer(value, label):
    if type(value) is not int or not 0 <= value <= 2147483647:
        raise ValueError("{} must be a nonnegative int32 integer".format(label))


def _distance(value, label):
    if type(value) not in (int, float):
        raise ValueError("{} must be a finite distance in meters".format(label))
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("{} must be a finite distance in meters".format(label))


class DoubleLaneScenario:
    def __init__(self, data):
        _fields(data, ("ego_vehicle", "vehicles"), "scenario")
        ego = data["ego_vehicle"]
        _fields(ego, ("lane_id", "initial_distance"), "ego_vehicle")
        _integer(ego["lane_id"], "ego_vehicle.lane_id")
        _distance(ego["initial_distance"], "ego_vehicle.initial_distance")
        vehicles = data["vehicles"]
        if not isinstance(vehicles, list) or not 4 <= len(vehicles) <= 12:
            raise ValueError("vehicles must contain 4–12 objects (at least 3 moving and 1 stationary)")
        for index, vehicle in enumerate(vehicles):
            label = "vehicles[{}]".format(index)
            _fields(vehicle, ("id", "type", "lane_id", "initial_distance"), label)
            for field in ("id", "type", "lane_id"):
                _integer(vehicle[field], label + "." + field)
            _distance(vehicle["initial_distance"], label + ".initial_distance")
        self.vehicles = sorted((dict(v) for v in vehicles), key=lambda v: v["id"])
        if [v["id"] for v in self.vehicles] != list(range(len(vehicles))):
            raise ValueError("traffic IDs must be unique and contiguous starting at 0")
        self.ego = dict(ego)
        self.stationary_id = len(vehicles) - 1
        if ego["lane_id"] != 0 or self.vehicles[0]["lane_id"] != 0:
            raise ValueError("ego and front vehicle ID 0 must occupy lane 0")
        if any(v["lane_id"] != 1 for v in self.vehicles[1:]):
            raise ValueError("traffic IDs 1 onward must occupy lane 1")

        self.lane_order = {}
        self.predecessors = {}
        for lane in (0, 1):
            objects = [(v["initial_distance"], v["id"])
                       for v in self.vehicles if v["lane_id"] == lane]
            if ego["lane_id"] == lane:
                objects.append((ego["initial_distance"], "ego"))
            objects.sort(key=lambda item: item[0])
            if len({distance for distance, _ in objects}) != len(objects):
                raise ValueError("lane {} contains duplicate initial distances".format(lane))
            order = [identifier for _, identifier in objects]
            self.lane_order[lane] = order  # rear to front
            for index, identifier in enumerate(order):
                self.predecessors[identifier] = order[index + 1] if index + 1 < len(order) else None

        self.ego_leader = self.predecessors["ego"]
        if self.ego_leader != 0:
            raise ValueError("front vehicle ID 0 must be ahead of ego")
        expected_side_order = list(range(self.stationary_id - 1, 0, -1)) + [self.stationary_id]
        if self.lane_order[1] != expected_side_order:
            raise ValueError("lane 1 must have stationary last ID ahead of ID 1, "
                             "with moving IDs ordered front-to-back as 1, 2, ...")
        self.offsets = [v["initial_distance"] - ego["initial_distance"] for v in self.vehicles]
        for index, offset in enumerate(self.offsets):
            _distance(offset, "vehicle {} offset from ego".format(index))
        self.front_gap = self.offsets[0]
        self.leader_offset = self.offsets[1] - self.offsets[0]
        _distance(self.leader_offset, "side leader offset from front vehicle")

    @classmethod
    def load(cls, filename):
        try:
            with open(filename, encoding="utf-8") as stream:
                return cls(json.load(stream))
        except (OSError, ValueError) as error:
            raise ValueError("Cannot load double-lane scenario '{}': {}".format(filename, error)) from error

    def initialize(self, manager, ego_s, initial_speed=0.0):
        for vehicle, offset in zip(self.vehicles, self.offsets):
            manager.traffic_initialization(
                ego_s, offset, vehicle["lane_id"], vehicle["id"], 0,
                initial_speed=0.0 if vehicle["id"] == self.stationary_id else initial_speed,
                initial_acceleration=0.0,
                vehicle_type=vehicle["type"],
            )


class ScenarioSession:
    """Separate ego-relative staging from a road-anchored, pausable simulation."""

    def __init__(self, scenario, initial_speed):
        self.scenario = scenario
        self.initial_speed = initial_speed
        self.started = False
        self.running = False
        self.ready = False
        self.sim_t = 0.0
        self.anchor_s = None
        self.previous_time = None

    def update(self, manager, ego_s, pose_ready, requested_start, now):
        elapsed = 0.0 if self.previous_time is None else max(now - self.previous_time, 0.0)
        self.previous_time = now
        was_running = self.running
        self.ready = bool(pose_ready)
        self.running = self.ready and bool(requested_start)
        if not self.ready:
            return 0.0
        if not self.started:
            self.scenario.initialize(manager, ego_s)
            if self.running:
                self.scenario.initialize(manager, ego_s, self.initial_speed)
                self.anchor_s = ego_s
                self.started = True
            return 0.0
        dt = elapsed if self.running and was_running else 0.0
        self.sim_t += dt
        return dt


class LaneMapGeometry:
    """Place common longitudinal states onto two separately surveyed centerlines.

    traffic_s remains in the lane-0 longitudinal frame for existing controllers.
    Each map uses its own ego projection as the origin while staging. Their
    difference is frozen at Start, so paused objects stay anchored to the road.
    As in the archived runtime, equal traveled distances are used on both maps.
    """

    def __init__(self, lane_maps):
        self.lane_maps = lane_maps
        self.side_distance_offset = 0.0

    def align_to_ego(self, ego_pose, ego_s):
        side_s, _, _ = self.lane_maps[1].find_ego_vehicle_distance_reference(ego_pose)
        offset = side_s - ego_s
        if not math.isfinite(offset):
            return False
        self.side_distance_offset = offset
        return True

    def pose(self, distance, lane_id):
        if lane_id not in (0, 1):
            raise ValueError("Map placement requires lane 0 or 1; merging vehicles use their controller pose")
        lane = int(lane_id)
        map_distance = distance + (self.side_distance_offset if lane == 1 else 0.0)
        # Each file already describes its lane: never add a synthetic lane-width offset.
        return self.lane_maps[lane].find_traffic_vehicle_poses(map_distance, lane_id=0)
