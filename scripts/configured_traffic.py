"""Per-object driving and lane permissions for JSON double-lane scenarios.

Positions use the shared lane-0 longitudinal frame. Changing vehicles occupy
both lanes until they have cleared the source lane.
"""
import math

import numpy as np

from sim_env_manager import lateral_vehicle_controller


class ConfiguredTraffic:
    def __init__(self, scenario, geometry, idm, get_param):
        self.scenario, self.geometry, self.idm = scenario, geometry, idm
        self.lanes = [v['lane_id'] for v in scenario.vehicles]
        self.height_lanes = list(self.lanes)
        self.changes = {}
        self.controllers = {}
        self.source_cleared = set()
        self.speed_limit = float(get_param('/side_lane_speed_limit', 35.0 * .44704))
        self.front_speed_limit = float(get_param('/front_vehicle_speed_limit', 35.0)) * .44704
        self.front_acc_min = float(get_param('/front_vehicle_acceleration_lower_limit', -6.0))
        self.front_acc_max = float(get_param('/front_vehicle_acceleration_upper_limit', 4.0))
        self.cbf_enable = str(get_param('/side_lane_cbf_enable', True)).lower() in ('true', '1', 'yes', 'on')
        self.cbf_alpha = float(get_param('/side_lane_cbf_alpha', 1.0))
        self.emergency_margin = float(get_param('/side_lane_cbf_emergency_decel_margin', 1.0))
        self.acc_min = float(get_param('/side_lane_acceleration_lower_limit', -6.0))
        self.acc_max = float(get_param('/side_lane_acceleration_upper_limit', 3.0))
        self.clearance = float(get_param('/side_lane_cbf_s0', 5.0))
        self.headway = float(get_param('/side_lane_cbf_T', .8))
        self.lookahead = float(get_param('/min_lookahead_distance', 15.0))
        self.steering_rate = float(get_param('/max_steering_rate', 2.0))
        self.jerk = float(get_param('/max_jerk', 5.0))
        self.car_length = float(get_param('/car_length', 3.5))
        for name in ('speed_limit', 'clearance', 'lookahead', 'car_length'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be finite and positive')
        for name in ('headway', 'steering_rate', 'jerk'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if not math.isfinite(self.acc_min) or self.acc_min >= 0 or not math.isfinite(self.acc_max) or self.acc_max <= 0:
            raise ValueError('side lane acceleration limits must bracket zero')
        self.mobil_politeness = float(get_param('/mobil_politeness', .3))
        self.mobil_threshold = float(get_param('/mobil_acceleration_threshold', .2))
        self.mobil_safe_braking = float(get_param('/mobil_safe_braking', 4.0))
        self.mobil_cooldown = float(get_param('/mobil_cooldown', 3.0))
        for name in ('mobil_politeness', 'mobil_threshold', 'mobil_cooldown'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        # Keep permitted follower braking below the existing IDM actuation
        # limit of 6 m/s^2. Predictions below use unclipped IDM demand.
        if not math.isfinite(self.mobil_safe_braking) or not 0 < self.mobil_safe_braking < 6.:
            raise ValueError('mobil_safe_braking must be finite and strictly between 0 and 6 m/s^2')
        self.cooldowns = [0.] * len(self.lanes)
        self.front_constrained = False

    def occupied(self, i):
        if i in self.changes:
            return {self.changes[i]} if i in self.source_cleared else {self.lanes[i], self.changes[i]}
        return {self.lanes[i]}

    def actors(self, manager, ego_s):
        actors = [(i, float(manager.traffic_s[i]), float(manager.traffic_v[i]), self.occupied(i))
                  for i in range(len(self.lanes))]
        actors.append(('ego', float(ego_s), max(0., float(manager.ego_v)), {self.scenario.ego['lane_id']}))
        return actors

    @staticmethod
    def ahead(i, s, lanes, actors):
        candidates = [a for a in actors if a[0] != i and a[3] & lanes and a[1] >= s]
        return min(candidates, key=lambda a: a[1]) if candidates else None

    def safe_gap(self, follower_v, leader_v):
        closing = max(0., follower_v - leader_v)
        return self.clearance + self.headway * follower_v + closing ** 2 / (2 * -self.acc_min)

    def can_merge(self, i, s, v, target, actors):
        for other, other_s, other_v, lanes in actors:
            if other == i or target not in lanes:
                continue
            gap = other_s - s
            required = self.safe_gap(v, other_v) if gap >= 0 else self.safe_gap(other_v, v)
            if abs(gap) <= required:
                return False
        return True

    @staticmethod
    def behind(i, s, lane, actors):
        candidates = [a for a in actors if a[0] != i and lane in a[3] and a[1] < s]
        return max(candidates, key=lambda a: a[1]) if candidates else None

    def mobil_acceleration(self, follower, leader):
        """Hypothetical acceleration from the existing IDM, before CBF/jerk.

        Ego uses the same IDM as a prediction only; its controls are untouched.
        Parked actors cannot accelerate and contribute zero utility.
        """
        if follower is None or follower[0] in self.scenario.parking_ids:
            return 0.
        if leader is None:
            return float(self.idm.a * (1. - (follower[2] / self.idm.v0) ** 4))
        if leader[1] - follower[1] <= self.idm.s0 + 1e-3:
            return -float('inf')
        return float(self.idm.raw_IDM_acceleration(
            front_v=leader[2], ego_v=follower[2], front_s=leader[1], ego_s=follower[1]))

    def mobil_wants_change(self, i, target, actors):
        """Full symmetric MOBIL incentive, plus new-follower braking safety.

        gain = own delta + p * (new follower delta + old follower delta).
        Gap safety is additionally enforced by can_merge at the call site.
        """
        vehicle = next(a for a in actors if a[0] == i)
        source = self.lanes[i]
        others = [a for a in actors if a[0] != i]
        old_leader = self.ahead(i, vehicle[1], {source}, others)
        new_leader = self.ahead(i, vehicle[1], {target}, others)
        old_follower = self.behind(i, vehicle[1], source, others)
        new_follower = self.behind(i, vehicle[1], target, others)
        new_after = self.mobil_acceleration(new_follower, vehicle)
        if new_after <= -self.mobil_safe_braking:
            return False
        if not all(math.isfinite(self.mobil_acceleration(vehicle, leader))
                   for leader in (old_leader, new_leader)):
            return False
        own_gain = (self.mobil_acceleration(vehicle, new_leader)
                    - self.mobil_acceleration(vehicle, old_leader))
        # With finite-duration merges, a follower can occupy both lanes. Find
        # its actual nearest leader in all occupied lanes before/after insertion.
        def follower_gain(follower, after_actors):
            if follower is None:
                return 0.
            before = self.ahead(follower[0], follower[1], follower[3], actors)
            after = self.ahead(follower[0], follower[1], follower[3], after_actors)
            return self.mobil_acceleration(follower, after) - self.mobil_acceleration(follower, before)
        hypothetical = others + [(i, vehicle[1], vehicle[2], {target})]
        affected = {a[0]: a for a in (old_follower, new_follower) if a is not None}
        follower_delta = sum(follower_gain(a, hypothetical) for a in affected.values())
        return (math.isfinite(follower_delta)
                and own_gain + self.mobil_politeness * follower_delta > self.mobil_threshold)

    def pose(self, manager, i):
        if i in self.controllers:
            return self.controllers[i].get_traffic_pose()
        pose = list(self.geometry.pose(manager.traffic_s[i], self.lanes[i]))
        height = self.geometry.pose(manager.traffic_s[i], self.height_lanes[i])
        pose[2], pose[4] = height[2], height[4]
        return pose

    def step(self, manager, dt, ego_s, front_proposal):
        """Apply a nominal (s,v,a) for ID 0, subject to its JSON and traffic.

        The proposal comes from the existing cycle/behavior generator. Once
        constrained, track speed continuously rather than snapping to its s.
        """
        if dt <= 0:
            return
        remaining = float(dt)
        while remaining > 1e-9:
            h = min(.05, remaining)
            self._step(manager, h, ego_s, front_proposal)
            remaining -= h
        if (0 not in self.scenario.parking_ids and not self.front_constrained
                and 0 not in self.controllers):
            manager.traffic_s[0], manager.traffic_v[0], manager.traffic_alon[0] = front_proposal

    def _step(self, manager, dt, ego_s, front_proposal):
        self.cooldowns = [max(0., value - dt) for value in self.cooldowns]
        actors = self.actors(manager, ego_s)
        # Reserve both lanes immediately so simultaneous changes cannot select
        # the same gap. Decisions and accelerations otherwise use one snapshot.
        for vehicle in self.scenario.vehicles:
            i = vehicle['id']
            if i in self.scenario.parking_ids or vehicle['lane_layout'] != 'double_lane' or i in self.changes or self.cooldowns[i] > 0.:
                continue
            s, v = manager.traffic_s[i], manager.traffic_v[i]
            target = 1 - self.lanes[i]
            if self.can_merge(i, s, v, target, actors) and self.mobil_wants_change(i, target, actors):
                self.source_cleared.discard(i)
                self.changes[i] = target
                pose = self.pose(manager, i)
                controller = lateral_vehicle_controller(
                    x_init=pose[0], y_init=pose[1], z_init=pose[2], yaw_init=pose[3], pitch_init=pose[4],
                    car_length=self.car_length, max_steering_rate=self.steering_rate, max_jerk=self.jerk)
                controller.v, controller.acc = v, manager.traffic_alon[i]
                self.controllers[i] = controller
                actors = self.actors(manager, ego_s)
                if i == 0:
                    self.front_constrained = True

        for vehicle in self.scenario.vehicles:
            i = vehicle['id']
            if i in self.scenario.parking_ids:
                manager.traffic_v[i] = manager.traffic_alon[i] = 0.
                continue
            s, v = float(manager.traffic_s[i]), max(0., float(manager.traffic_v[i]))
            desired_v = max(0., front_proposal[1]) if i == 0 else self.speed_limit
            acc_min = self.front_acc_min if i == 0 else self.acc_min
            acc_max = self.front_acc_max if i == 0 else self.acc_max
            acceleration = float(np.clip((desired_v - v) / max(dt, .2), acc_min, acc_max))
            if i in self.controllers:
                previous_acc = manager.traffic_alon[i]
                acceleration = float(np.clip(acceleration, previous_acc-self.jerk*dt, previous_acc+self.jerk*dt))
            leader = self.ahead(i, s, self.occupied(i), actors)
            if leader is not None:
                follow = self.idm.safe_IDM_acceleration(front_v=leader[2], ego_v=v,
                    front_s=leader[1], ego_s=s, fallback_acc=acc_min)
                follow, _, _, _ = self.idm.CBF_acceleration_filter(
                    commanded_acc=follow, front_v=leader[2], ego_v=v, front_s=leader[1], ego_s=s,
                    acc_min=acc_min, acc_max=acc_max, cbf_enable=self.cbf_enable, cbf_alpha=self.cbf_alpha,
                    cbf_s0=self.clearance, cbf_T=self.headway, emergency_decel_margin=self.emergency_margin)
                acceleration = min(acceleration, float(follow))
                if i == 0:
                    self.front_constrained = True
            acceleration = float(np.clip(acceleration, acc_min, acc_max))
            # Stop at zero speed; never integrate backwards during a long tick.
            move_dt = min(dt, v / -acceleration) if acceleration < 0 else dt
            next_s = s + v * move_dt + .5 * acceleration * move_dt ** 2
            next_v = float(np.clip(v + acceleration * dt, 0., self.speed_limit if i else self.front_speed_limit))
            if leader is not None:
                # Do not reverse an object already inside the requested gap.
                limit = max(s, leader[1] - self.clearance)
                if next_s >= limit:
                    next_s, next_v = limit, 0.
            if i in self.controllers:
                controller = self.controllers[i]
                target = self.changes.get(i, self.lanes[i])
                goal = self.geometry.pose(s + max(self.lookahead, v * .6), target)
                controller.pure_pursuit_controller(goal)
                # Longitudinal following has priority over comfort jerk limits.
                old_x, old_y, old_yaw = controller.x, controller.y, controller.yaw
                controller.update_vehicle_state(acc=acceleration, z=controller.z, pitch=controller.pitch, dt=dt)
                pose = controller.get_traffic_pose()
                projected, _, _ = self.geometry.lane_maps[0].find_ego_vehicle_distance_reference(
                    np.array([[pose[0]], [pose[1]], [pose[2]]]))
                allowed = max(0., next_s - s)
                advance = max(0., float(projected) - s)
                if advance > allowed + 1e-8:
                    fraction = allowed / advance
                    controller.x, controller.y = old_x, old_y
                    controller.yaw = old_yaw + (controller.yaw - old_yaw) * fraction
                    controller.x += (pose[0] - old_x) * fraction
                    controller.y += (pose[1] - old_y) * fraction
                    projected = s + allowed
                next_s = max(s, float(projected))
                controller.v, controller.acc = next_v, acceleration
                height = self.geometry.pose(next_s, self.height_lanes[i])
                controller.z, controller.pitch = height[2], height[4]
                target_pose = self.geometry.pose(next_s, target)
                source_pose = self.geometry.pose(next_s, self.lanes[i])
                if i in self.changes:
                    separation = math.hypot(target_pose[0] - source_pose[0], target_pose[1] - source_pose[1])
                    distance = math.hypot(controller.x - target_pose[0], controller.y - target_pose[1])
                    progress = min(1., max(0., 1. - distance / max(separation, .001)))
                    if progress >= .65:
                        self.source_cleared.add(i)
                    manager.traffic_l[i] = self.lanes[i] + (target - self.lanes[i]) * progress
                    angle = math.atan2(math.sin(controller.yaw-target_pose[3]), math.cos(controller.yaw-target_pose[3]))
                    if distance < .3 and abs(angle) < .1:
                        self.lanes[i] = target
                        del self.changes[i]
                        self.cooldowns[i] = self.mobil_cooldown
                        manager.traffic_l[i] = target
            manager.traffic_s[i], manager.traffic_v[i] = next_s, next_v
            manager.traffic_alon[i] = acceleration if next_v > 0 else 0.
