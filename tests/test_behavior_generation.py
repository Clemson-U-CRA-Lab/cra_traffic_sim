"""Configuration, actual optimizer, and ROS-loop checks for the restored runtime."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_double_lane_scenario as support
from test_double_lane_scenario import ROOT, MemoryTraffic
from double_lane_scenario import BehaviorGenerationScenario

try:
    import car_following_double_lane_behavior_generation as behavior
except ImportError:
    behavior = None


class BehaviorScenarioTests(unittest.TestCase):
    def test_all_vehicles_move_and_keep_types(self):
        data = json.loads((ROOT / 'config/double_lane_behavior_generation_scenario.json').read_text())
        data['vehicles'][-1]['type'] = 1
        scenario = BehaviorGenerationScenario(data)
        self.assertIsNone(scenario.stationary_id)
        self.assertEqual(scenario.predecessors, {'ego': 0, 0: None, 4: 3, 3: 2, 2: 1, 1: None})
        manager = MemoryTraffic()
        scenario.initialize(manager, 100, 2)
        self.assertEqual(manager.traffic_v[:5], [2] * 5)
        self.assertEqual(manager.traffic_type[4], 1)
        data['vehicles'] = data['vehicles'][:2]
        self.assertEqual(len(BehaviorGenerationScenario(data).vehicles), 2)

    def test_obstacle_layout_is_not_silently_reinterpreted(self):
        with self.assertRaises(ValueError):
            BehaviorGenerationScenario.load(ROOT / 'config/double_lane_scenario.json')


@unittest.skipIf(behavior is None, 'Source ROS workspace and install CasADi')
class BehaviorRuntimeTests(unittest.TestCase):
    def run_scene(self, frames, trigger_frames=(), extra_params=None):
        data = json.loads((ROOT / 'config/double_lane_behavior_generation_scenario.json').read_text())
        data['vehicles'][-1]['type'] = 1
        params = {'/track_style': 'Rally', '/pv_states_dt': 0.5, '/runDirection': 0,
                  '/front_vehicle_travel_distance': 1000.0, '/auto_behavior_generation_enable': False}
        params.update(extra_params or {})
        return support.RuntimeTests.run_scene(self, frames, data, behavior, params, trigger_frames)

    def test_staging_and_triggered_generation_without_lane_changes(self):
        frames = [(0, None, True), (1, 100, False), (2, 120, False), (3, 120, True),
                  (3.1, 120, True), (3.2, 120, False), (50, 125, False),
                  (51, 125, True), (51.1, 125, True), (54.1, 125, True), (54.2, 125, True)]
        with patch.object(behavior.preceding_vehicle_spd_profile_generation,
                          'perform_nonlinear_optimization_for_reward_tracking',
                          autospec=True, wraps=None) as solve:
            # Deterministic acceleration isolates lifecycle/mode transitions; real solve tested below.
            def optimize(instance, **kwargs):
                instance.reward_tracking_u_opt = [1.0] * 7
            solve.side_effect = optimize
            messages = self.run_scene(frames, trigger_frames=(4,))
        self.assertGreaterEqual(solve.call_count, 2)
        traffic = messages['/traffic_sim_info_mache']
        holograms = messages['/virtual_sim_info_mache']
        self.assertEqual(len(traffic), len(frames) - 1)
        self.assertEqual(list(traffic[0].S_v_s[:5]), [108, 109, 101, 93, 85])
        self.assertEqual(list(holograms[0].S_v_x[:5]), [8, 9, 1, -7, -15])
        self.assertEqual(list(holograms[0].S_v_y[:5]), [0, -7, -7, -7, -7])
        self.assertEqual(holograms[0].S_v_type[4], 1)
        self.assertEqual(list(traffic[0].S_v_sv[:5]), [0] * 5)
        self.assertEqual(traffic[2].sim_T, 0)
        self.assertEqual(traffic[2].S_v_s[0], 128)
        self.assertEqual(traffic[2].S_v_sv[4], 2)
        self.assertGreater(traffic[3].S_v_acc[0], 0)
        for index in (4, 5, 6):
            self.assertAlmostEqual(traffic[index].sim_T, 0.1)
            self.assertEqual(traffic[index].S_v_s, traffic[3].S_v_s)
        for message in traffic:
            self.assertEqual(list(message.S_v_l[:5]), [0, 1, 1, 1, 1])

    def test_stop_at_distance_and_hold(self):
        messages = self.run_scene([(0, 100, True), (1, 100, True), (2, 100, True), (3, 100, True)],
                                  extra_params={'/front_vehicle_travel_distance': 1.0})
        traffic = messages['/traffic_sim_info_mache']
        self.assertLessEqual(traffic[-1].S_v_s[0], 109)
        self.assertEqual(traffic[-1].S_v_sv[0], 0)

    def test_ros_resolves_only_active_runtime(self):
        import roslib.packages
        paths = roslib.packages.find_node('cra_traffic_sim',
                                          'car_following_double_lane_behavior_generation.py')
        self.assertEqual({Path(path).resolve() for path in paths},
                         {(ROOT / 'scripts/car_following_double_lane_behavior_generation.py').resolve()})

    def test_manual_trigger_is_edge_triggered(self):
        with patch.object(behavior.rospy, 'Publisher'), patch.object(behavior.rospy, 'Subscriber'):
            manager = behavior.BehaviorTrafficManager(12, 5, True)
        msg = SimpleNamespace(buttons=[0, 0, 0, 1, 0, 1])
        manager.joy_callback(msg)
        self.assertTrue(manager.sim_start)
        self.assertTrue(manager.consume_behavior_generation_request())
        manager.joy_callback(msg)
        self.assertFalse(manager.consume_behavior_generation_request())
        msg.buttons[3] = 0
        manager.joy_callback(msg)
        msg.buttons[3] = 1
        manager.joy_callback(msg)
        self.assertTrue(manager.consume_behavior_generation_request())

    def test_automatic_trigger_requires_fresh_quiet_input(self):
        with patch.object(behavior.rospy, 'Subscriber'), patch.object(behavior.rospy, 'loginfo'), \
             patch.object(behavior.rospy, 'loginfo_throttle'), \
             patch.object(behavior.rospy, 'logwarn_throttle'), \
             patch.object(behavior, 'time') as clock:
            clock.time.return_value = 0
            monitor = behavior.HumanInterventionMonitor(0.05, 0.5, 0.5, 1.0, 1)
            self.assertFalse(monitor.should_trigger())
            monitor.control_target_callback(SimpleNamespace(human_acceleration_command=0.0))
            self.assertFalse(monitor.should_trigger())
            clock.time.return_value = 0.6
            self.assertTrue(monitor.should_trigger())
            clock.time.return_value = 2
            self.assertFalse(monitor.should_trigger())
            monitor.control_target_callback(SimpleNamespace(human_acceleration_command=0.2))
            self.assertFalse(monitor.should_trigger())

    def test_actual_optimizer_default_model(self):
        import numpy as np
        model = behavior.preceding_vehicle_spd_profile_generation(8, 0.5)
        model.load_matrices_from_file(str(ROOT / 'config/behavior_generation'))
        model.update_ego_vehicle_state(0, 5, 0, 0, 5, 15)
        model.perform_nonlinear_optimization_for_reward_tracking(
            Q=1, R=10, reward_target=[0, 5, 10, 15, 20, 20, 20, 20],
            a_max=4, a_min=-6, v_max=10, v_min=-5, R_du=100)
        self.assertEqual(model.reward_tracking_u_opt.shape, (7,))
        self.assertTrue(np.all(np.isfinite(model.reward_tracking_u_opt)))


if __name__ == '__main__':
    unittest.main()
