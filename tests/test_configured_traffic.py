"""JSON behavior and bidirectional maneuver regressions; no ROS master needed."""
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import test_double_lane_scenario as support
from double_lane_scenario import DoubleLaneScenario, BehaviorGenerationScenario, LaneMapGeometry

try:
    from configured_traffic import ConfiguredTraffic
    from utils import IDM
except ImportError:
    ConfiguredTraffic = None


def data(lane=0, permission='double_lane'):
    return {'ego_vehicle': {'lane_id': 0, 'initial_distance': -100.}, 'vehicles': [
        {'id': 0, 'type': 1, 'lane_id': lane, 'initial_distance': 0.,
         'operation_mode': 'driving', 'lane_layout': permission},
        {'id': 1, 'type': 0, 'lane_id': lane, 'initial_distance': 30.,
         'operation_mode': 'parking', 'lane_layout': 'double_lane'}]}


class SchemaTests(unittest.TestCase):
    def test_modes_independent_of_type_id_and_runtime(self):
        for cls in (DoubleLaneScenario, BehaviorGenerationScenario):
            scenario = cls(data())
            self.assertEqual(scenario.parking_ids, {1})
            manager = support.MemoryTraffic()
            scenario.initialize(manager, -100., 5.)
            self.assertEqual(manager.traffic_v[:2], [5., 0.])
            self.assertEqual(scenario.predecessors[0], 1)

    def test_bad_modes_lanes_and_partial_settings(self):
        for key, value in [('operation_mode', 'park'), ('operation_mode', True),
                           ('lane_layout', 'two'), ('lane_layout', 2), ('lane_id', 2)]:
            config = data()
            config['vehicles'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                DoubleLaneScenario(config)
        config = data()
        del config['vehicles'][1]['operation_mode']
        with self.assertRaises(ValueError):
            DoubleLaneScenario(config)


@unittest.skipIf(ConfiguredTraffic is None, 'Source ROS workspace')
class MotionTests(unittest.TestCase):
    def setup_scene(self, config):
        class Road:
            def __init__(self, lane):
                self.lane = lane
            def find_traffic_vehicle_poses(self, s, lane_id):
                return [s, 3.5*self.lane, self.lane * (2. + .01*s), 0., .01*self.lane]
            def find_ego_vehicle_distance_reference(self, pose):
                return float(pose[0, 0]), 0., 0.
        scenario = DoubleLaneScenario(config)
        manager = support.MemoryTraffic()
        manager.traffic_alon = [0.] * 12
        manager.ego_v = 0.
        scenario.initialize(manager, config['ego_vehicle']['initial_distance'], 5.)
        runtime = ConfiguredTraffic(scenario, LaneMapGeometry([Road(0), Road(1)]),
                                    IDM(a=4,b=6,s0=6,v0=35,T=.8), lambda k,d: d)
        return manager, runtime

    def test_mobil_allows_overtaking_moving_leader(self):
        config = data()
        config['vehicles'][1]['operation_mode'] = 'driving'
        manager, runtime = self.setup_scene(config)
        manager.traffic_v[1] = 1.
        self.assertTrue(runtime.mobil_wants_change(0, 1, runtime.actors(manager, -100.)))
        runtime.step(manager, .05, -100., (0., 5., 0.))
        self.assertIn(0, runtime.changes)

    def test_mobil_rejects_empty_equally_attractive_lanes(self):
        manager, runtime = self.setup_scene(data())
        actors = [(0, 0., 5., {0})]
        self.assertFalse(runtime.mobil_wants_change(0, 1, actors))

    def test_mobil_rejects_excessive_follower_braking(self):
        manager, runtime = self.setup_scene(data())
        actors = [(0, 0., 5., {0}), (1, 30., 0., {0}), (2, -20., 20., {1})]
        self.assertLess(runtime.mobil_acceleration(actors[2], actors[0]), -6.)
        self.assertFalse(runtime.mobil_wants_change(0, 1, actors))
        # Ego participates in exactly the same safety prediction.
        actors[2] = ('ego', -20., 20., {1})
        self.assertFalse(runtime.mobil_wants_change(0, 1, actors))

    def test_mobil_politeness_includes_both_followers(self):
        manager, runtime = self.setup_scene(data())
        actors = [(0, 0., 5., {0}), (1, 30., 0., {0}),
                  (2, -20., 5., {1}), (3, 60., 5., {1}), (4, -20., 5., {0})]
        predictions = {(0, 1): 0., (0, 3): 1., (2, 3): 2.,
                       (2, 0): -1., (4, 0): 0., (4, 1): 3.}
        def acceleration(follower, leader):
            if follower is None:
                return 0.
            return predictions[(follower[0], leader[0])]
        runtime.mobil_politeness = .8
        with patch.object(runtime, 'mobil_acceleration', side_effect=acceleration):
            # Own gain 1, target follower loses 3, source follower gains 3.
            self.assertTrue(runtime.mobil_wants_change(0, 1, actors))
            self.assertFalse(runtime.mobil_wants_change(0, 1, actors[:-1]))
            runtime.mobil_politeness = 0.
            self.assertTrue(runtime.mobil_wants_change(0, 1, actors[:-1]))
            runtime.mobil_threshold = 1.
            self.assertFalse(runtime.mobil_wants_change(0, 1, actors[:-1]))

    def test_mobil_cooldown_uses_only_running_time(self):
        manager, runtime = self.setup_scene(data())
        runtime.cooldowns[0] = 1.
        runtime.step(manager, 0., -100., (0., 5., 0.))
        self.assertEqual(runtime.cooldowns[0], 1.)
        with patch.object(runtime, 'mobil_wants_change', return_value=True):
            runtime.step(manager, .5, -100., (0., 5., 0.))
            self.assertNotIn(0, runtime.changes)
            runtime.step(manager, .6, -100., (0., 5., 0.))
            self.assertIn(0, runtime.changes)

    def test_mobil_launch_parameters_and_validation(self):
        for filename in ('cra_traffic_sim_human_adaptive_acc_double_lane.launch',
                         'cra_traffic_sim_behavior_generation_double_lane.launch'):
            root = ET.parse(support.ROOT / 'launch' / filename).getroot()
            args = {n.attrib['name']: n.attrib['default'] for n in root.findall('arg')}
            params = {n.attrib['name']: n.attrib['value'] for n in root.findall('param')}
            for key in ('mobil_politeness', 'mobil_acceleration_threshold', 'mobil_safe_braking', 'mobil_cooldown'):
                self.assertGreaterEqual(float(args[key]), 0.)
                self.assertEqual(params[key], '$(arg ' + key + ')')
        manager, runtime = self.setup_scene(data())
        for key, value in (('mobil_politeness', -1.), ('mobil_acceleration_threshold', float('nan')),
                           ('mobil_safe_braking', 6.), ('mobil_safe_braking', 0.), ('mobil_cooldown', -1.)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ConfiguredTraffic(runtime.scenario, runtime.geometry, runtime.idm,
                                  lambda k,d: value if k == '/' + key else d)

    def test_single_lane_stops_without_jumping_or_moving_parked(self):
        manager, runtime = self.setup_scene(data(permission='single_lane'))
        positions = []
        for step in range(1000):
            runtime.step(manager, .05, -100., (step*.25, 5., 0.))
            positions.append(manager.traffic_s[0])
            self.assertEqual(manager.traffic_s[1], 30.)
        self.assertFalse(runtime.controllers)
        self.assertEqual(manager.traffic_v[0], 0.)
        self.assertTrue(all(b >= a for a,b in zip(positions, positions[1:])))
        self.assertLess(manager.traffic_s[0], 30.)
        self.assertAlmostEqual(positions[-1], positions[-100])

    def test_changes_both_directions_preserving_source_height(self):
        for lane in (0, 1):
            with self.subTest(lane=lane):
                manager, runtime = self.setup_scene(data(lane))
                for step in range(1200):
                    runtime.step(manager, .05, -100., (step*.25, 5., 0.))
                    pose = runtime.pose(manager, 0)
                    self.assertAlmostEqual(pose[2], lane * (2. + .01*manager.traffic_s[0]))
                self.assertEqual(runtime.lanes[0], 1-lane)
                self.assertGreater(manager.traffic_s[0], 30.)
                self.assertEqual(manager.traffic_s[1], 30.)

    def test_target_lane_ego_blocks_merge(self):
        config = data(1)
        config['vehicles'][1]['initial_distance'] = 20.
        manager, runtime = self.setup_scene(config)
        for step in range(20):
            runtime.step(manager, .05, manager.traffic_s[0], (0., 5., 0.))
        self.assertFalse(runtime.controllers)

    def test_fast_rear_vehicle_blocks_cut_in(self):
        config = data(1)
        config['vehicles'].append({'id': 2, 'type': 0, 'lane_id': 0,
            'initial_distance': -15., 'operation_mode': 'driving', 'lane_layout': 'single_lane'})
        manager, runtime = self.setup_scene(config)
        manager.traffic_v[2] = 15.
        runtime.step(manager, .05, -100., (0., 5., 0.))
        self.assertNotIn(0, runtime.controllers)

    def test_pause_during_maneuver_freezes_pose(self):
        manager, runtime = self.setup_scene(data(1))
        for step in range(20):
            runtime.step(manager, .05, -100., (0., 5., 0.))
        self.assertIn(0, runtime.controllers)
        pose = list(runtime.pose(manager, 0))
        position = manager.traffic_s[0]
        runtime.step(manager, 0., -90., (100., 5., 0.))
        self.assertEqual(runtime.pose(manager, 0), pose)
        self.assertEqual(manager.traffic_s[0], position)
        runtime.step(manager, .05, -90., (100., 5., 0.))
        self.assertGreater(manager.traffic_s[0], position)
        self.assertLess(manager.traffic_s[0] - position, 1.)

    def test_parked_front_never_follows_cycle(self):
        config = data()
        config['vehicles'][0]['operation_mode'] = 'parking'
        manager, runtime = self.setup_scene(config)
        for step in range(10):
            runtime.step(manager, .2, -100., (100.+step, 15., 3.))
        self.assertEqual(manager.traffic_s[0], 0.)
        self.assertEqual(manager.traffic_v[0], 0.)
        self.assertFalse(runtime.controllers)

    def test_both_ros_runtimes_publish_parked_front_and_hold_pause(self):
        import car_following_double_lane_behavior_generation as behavior
        config = data()
        config['vehicles'][0]['operation_mode'] = 'parking'
        for module in (None, behavior):
            with self.subTest(runtime=module):
                messages = support.RuntimeTests.run_scene(self,
                    [(0, None, True), (1, 100, False), (2, 110, False), (3, 110, True),
                     (4, 110, True), (5, 120, False), (10, 120, True), (11, 120, True)],
                    config, module, {'/auto_behavior_generation_enable': False, '/pv_states_dt': .5, '/track_style': 'Rally', '/runDirection': 0, '/front_vehicle_travel_distance': 1000.})
                traffic = messages['/traffic_sim_info_mache']
                holo = messages['/virtual_sim_info_mache']
                self.assertEqual(traffic[0].S_v_s[0], 200.)
                self.assertEqual(traffic[1].S_v_s[0], 210.)
                for msg in traffic[2:]:
                    self.assertEqual(msg.S_v_s[0], 210.)
                    self.assertEqual(msg.S_v_sv[0], 0.)
                self.assertEqual(holo[0].S_v_type[0], 1)
                self.assertEqual(traffic[-1].sim_T, 2.)
