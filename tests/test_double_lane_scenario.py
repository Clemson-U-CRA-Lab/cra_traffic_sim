"""Run with python3 -m unittest discover -s tests -v.

Source the catkin workspace first to also run ROS-message integration checks.
No ROS master, joystick, or network connection is used.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from double_lane_scenario import DoubleLaneScenario, ScenarioSession, LaneMapGeometry


def fixture_scenario():
    # Keep regressions reproducible when the user edits the live scenario JSON.
    return {
        'ego_vehicle': {'lane_id': 0, 'initial_distance': 0.0},
        'vehicles': [
            {'id': 0, 'type': 0, 'lane_id': 0, 'initial_distance': 8.0},
            {'id': 1, 'type': 0, 'lane_id': 1, 'initial_distance': 18.0},
            {'id': 2, 'type': 0, 'lane_id': 1, 'initial_distance': 10.0},
            {'id': 3, 'type': 1, 'lane_id': 1, 'initial_distance': 138.0},
        ],
    }


class MemoryTraffic:
    def __init__(self):
        self.traffic_s = [0.0] * 12
        self.traffic_v = [0.0] * 12
        self.traffic_l = [0.0] * 12
        self.traffic_type = [0] * 12

    def traffic_initialization(self, ego_s, gap, lane, vehicle_id, lane_index,
                               initial_speed=0.0, initial_acceleration=0.0, vehicle_type=0):
        self.traffic_s[vehicle_id] = ego_s + gap
        self.traffic_v[vehicle_id] = initial_speed
        self.traffic_l[vehicle_id] = lane
        self.traffic_type[vehicle_id] = vehicle_type


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture_scenario()

    def test_default_order_and_unordered_json_entries(self):
        self.data['vehicles'].reverse()
        scenario = DoubleLaneScenario(self.data)
        self.assertEqual(scenario.predecessors, {'ego': 0, 0: None, 2: 1, 1: 3, 3: None})
        self.assertEqual(scenario.offsets, [8, 18, 10, 138])
        self.assertEqual(scenario.front_gap, 8)
        self.assertEqual(scenario.leader_offset, 10)

    def test_nonzero_ego_coordinate_and_type_is_only_metadata(self):
        self.data['ego_vehicle']['initial_distance'] += 100
        for vehicle in self.data['vehicles']:
            vehicle['initial_distance'] += 100
            vehicle['type'] = 7
        scenario = DoubleLaneScenario(self.data)
        manager = MemoryTraffic()
        scenario.initialize(manager, 200, initial_speed=4)
        self.assertEqual(manager.traffic_s[:4], [208, 218, 210, 338])
        self.assertEqual(manager.traffic_type[:4], [7] * 4)
        self.assertEqual(manager.traffic_v[:4], [4, 4, 4, 0])

    def test_invalid_layouts_and_fields(self):
        changes = [
            lambda d: d.update(extra=True),
            lambda d: d['ego_vehicle'].update(id='ego'),
            lambda d: d['vehicles'][0].update(control=True),
            lambda d: d['vehicles'][0].pop('type'),
            lambda d: d['vehicles'][0].update(id=True),
            lambda d: d['vehicles'][0].update(type=1.5),
            lambda d: d['vehicles'][0].update(type=2**31),
            lambda d: d['vehicles'][0].update(initial_distance='8'),
            lambda d: d['vehicles'][0].update(initial_distance=True),
            lambda d: d['vehicles'][0].update(initial_distance=float('nan')),
            lambda d: d['vehicles'][0].update(initial_distance=float('inf')),
            lambda d: d['vehicles'][0].update(initial_distance=10**400),
            lambda d: d['vehicles'][1].update(id=0),
            lambda d: d['vehicles'][3].update(id=4),
            lambda d: d['ego_vehicle'].update(lane_id=1),
            lambda d: d['vehicles'][0].update(lane_id=1),
            lambda d: d['vehicles'][2].update(lane_id=0),
            lambda d: d['vehicles'][0].update(initial_distance=-1),
            lambda d: d['vehicles'][0].update(initial_distance=0),
            lambda d: d['vehicles'][2].update(initial_distance=18),
            lambda d: d['vehicles'][2].update(initial_distance=20),
            lambda d: d['vehicles'][3].update(initial_distance=15),
            lambda d: d['vehicles'].pop(),
            lambda d: d.update(vehicles=d['vehicles'] * 4),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                data = copy.deepcopy(self.data)
                change(data)
                with self.assertRaises(ValueError):
                    DoubleLaneScenario(data)

    def test_extra_followers_and_capacity(self):
        self.data['vehicles'] = self.data['vehicles'][:3] + [
            {'id': i, 'type': 0, 'lane_id': 1, 'initial_distance': 10 - 8 * (i - 2)}
            for i in range(3, 11)
        ] + [{'id': 11, 'type': 1, 'lane_id': 1, 'initial_distance': 138}]
        scenario = DoubleLaneScenario(self.data)
        self.assertEqual(scenario.predecessors[10], 9)
        self.assertEqual(scenario.predecessors[1], 11)

    def test_load_error_includes_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for contents in (None, '{broken', '{}'):
                if contents is not None:
                    path.write_text(contents)
                with self.assertRaisesRegex(ValueError, 'bad.json'):
                    DoubleLaneScenario.load(path)

    def test_staging_start_pause_resume(self):
        session = ScenarioSession(DoubleLaneScenario(self.data), initial_speed=2)
        traffic = MemoryTraffic()
        session.update(traffic, 0, False, True, 0)
        self.assertFalse(session.started)
        session.update(traffic, 100, True, False, 1)
        self.assertEqual(traffic.traffic_s[:4], [108, 118, 110, 238])
        self.assertEqual(traffic.traffic_type[:4], [0, 0, 0, 1])
        self.assertEqual(traffic.traffic_v[:4], [0] * 4)
        session.update(traffic, 120, True, False, 10)
        self.assertEqual(traffic.traffic_s[0], 128)
        self.assertEqual(session.update(traffic, 120, True, True, 20), 0)
        self.assertEqual(session.anchor_s, 120)
        self.assertEqual(traffic.traffic_v[:4], [2, 2, 2, 0])
        self.assertEqual(session.update(traffic, 125, True, True, 21), 1)
        self.assertEqual(session.sim_t, 1)
        traffic.traffic_s[2] = 135  # Stand in for a controller's evolved state.
        traffic.traffic_l[2] = 0.5
        session.update(traffic, 140, True, False, 22)
        session.update(traffic, 150, True, False, 500)
        self.assertEqual(session.update(traffic, 150, True, True, 501), 0)
        self.assertEqual(session.sim_t, 1)
        self.assertEqual(traffic.traffic_s[2], 135)
        self.assertEqual(traffic.traffic_l[2], 0.5)
        self.assertEqual(session.update(traffic, 150, True, True, 502), 1)


try:
    import car_following_double_lane as runtime
    from nav_msgs.msg import Odometry
except ImportError:
    runtime = None


@unittest.skipIf(runtime is None, 'Source the ROS/catkin workspace for integration checks')
class RuntimeTests(unittest.TestCase):
    def run_scene(self, frames, data=None, runtime_module=None, extra_params=None, trigger_frames=(), side_grade=0.0):
        """Run the real node loop/controllers/message builders with fake ROS I/O and road."""
        node = runtime_module or runtime
        manager_name = "BehaviorTrafficManager" if runtime_module else "CMI_traffic_sim"
        published = {}
        state = SimpleNamespace(index=0, manager=None)
        manager_class = getattr(node, manager_name)

        class Publisher:
            def __init__(self, topic, *args, **kwargs):
                self.topic = topic
                published[topic] = []

            def publish(self, msg):
                published[self.topic].append(copy.deepcopy(msg))

        class Road:
            def __init__(self, filename=None, *args, **kwargs):
                filename = filename or kwargs["map_filename"]
                self.side = Path(filename).name == 'lane1.csv'
                self.origin = 1000.0 if self.side else 0.0

            def read_map_data(self):
                pass

            def read_speed_profile(self):
                pass

            def find_speed_profile_information(self, sim_t):
                return 2.0, 50.0 + 2.0 * sim_t, 0.0

            def find_front_vehicle_predicted_state(self, dt, sim_t):
                return self.find_speed_profile_information(sim_t + dt)

            def find_ego_vehicle_distance_reference(self, pose):
                return float(pose[0, 0]) + self.origin, 0.0, 0.0

            def find_traffic_vehicle_poses(self, distance, lane_id):
                if lane_id != 0:
                    raise AssertionError("Independent lane maps must not add a synthetic lane offset")
                return [distance - self.origin, 7.0 if self.side else 0.0,
                        2.0 + side_grade * (distance - self.origin) if self.side else 0.0,
                        0.1 if self.side else 0.0,
                        0.02 if self.side else 0.0]

            def find_ego_frenet_pose(self, ego_poses, ego_yaw, vy, vx):
                return 0.0, 0.0, vy, vx

        def make_manager(*args, **kwargs):
            state.manager = manager_class(*args, **kwargs)
            return state.manager

        def is_shutdown():
            if state.index >= len(frames):
                return True
            now, pose, start = frames[state.index]
            state.manager.sim_start = start
            if state.index in trigger_frames:
                state.manager.behavior_generation_requested = True
            if pose is not None:
                msg = Odometry()
                msg.pose.pose.position.x = pose
                state.manager.odom_callback(msg)
            return False

        def sleep():
            state.index += 1

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'scenario.json'
            filename.write_text(json.dumps(data if data is not None else fixture_scenario()))
            params = {'~scenario_file': str(filename), '/map_0': 'lane0.csv',
                      '/map_1': 'lane1.csv', '/spd_map': 'unused',
                      '/run_sim': True, '/use_preview': True, '/use_acceleration_pitch': False}
            params.update(extra_params or {})
            with patch.multiple(node.rospy, init_node=lambda *a: None,
                                Publisher=Publisher, Subscriber=lambda *a, **kw: None,
                                Rate=lambda *a: SimpleNamespace(sleep=sleep),
                                get_param=lambda key, default=None: params.get(key, default),
                                is_shutdown=is_shutdown, loginfo=lambda *a: None,
                                logwarn_throttle=lambda *a: None), \
                 patch.object(node, manager_name, make_manager), \
                 patch.object(node, 'road_reader', Road), \
                 patch.object(node, 'time', SimpleNamespace(monotonic=lambda: frames[state.index][0])):
                if runtime_module:
                    node.main_double_lane_behavior_generation()
                else:
                    node.main_double_lane_following()
        return published

    def test_messages_before_start_and_profile_origin_and_pause(self):
        messages = self.run_scene([(0, None, True), (1, 100, False), (2, 120, False),
                                   (3, 120, True), (4, 120, True), (5, 125, False),
                                   (100, 130, False), (101, 130, True), (102, 130, True)])
        traffic = messages['/traffic_sim_info_mache']
        holograms = messages['/virtual_sim_info_mache']
        previews = messages['/front_v_traj_seq_v0']
        self.assertEqual(len(traffic), 8)  # No publication before pose arrives.
        self.assertEqual(list(traffic[0].S_v_s[:4]), [108, 118, 110, 238])
        self.assertEqual(list(traffic[0].S_v_type[:4]), [0, 0, 0, 1])
        self.assertEqual(list(holograms[0].S_v_type[:4]), [0, 0, 0, 1])
        self.assertEqual(list(holograms[0].S_v_vx[:4]), [0] * 4)
        self.assertEqual(list(holograms[0].S_v_x[:4]), [8, 18, 10, 138])
        self.assertEqual(list(holograms[0].S_v_y[:4]), [0, -7, -7, -7])
        self.assertEqual(list(holograms[0].S_v_z[:4]), [0, 2, 2, 2])
        self.assertEqual(list(holograms[0].S_v_yaw[:4]), [0, -0.1, -0.1, -0.1])
        self.assertEqual(list(holograms[0].S_v_pitch[:4]), [0, -0.02, -0.02, -0.02])
        self.assertEqual(list(holograms[1].S_v_x[:4]), [8, 18, 10, 138])
        self.assertEqual(traffic[2].sim_T, 0)
        self.assertEqual(traffic[2].S_v_s[0], 128)  # No jump by profile's initial 50m.
        self.assertEqual(previews[2].front_s[1], 129)
        self.assertEqual(traffic[3].S_v_s[0], 130)
        for index in (4, 5, 6):
            self.assertEqual(traffic[index].sim_T, 1)
            self.assertEqual(traffic[index].S_v_s, traffic[3].S_v_s)
        self.assertEqual(holograms[5].S_v_x[0], 0)  # Frozen at road s=130, ego moved to 130.
        self.assertEqual(traffic[7].sim_T, 2)
        self.assertEqual(traffic[7].S_v_s[0], 132)

    def test_start_requested_before_pose(self):
        messages = self.run_scene([(0, None, True), (50, 100, True), (51, 100, True)])
        traffic = messages['/traffic_sim_info_mache']
        self.assertEqual(len(traffic), 2)
        self.assertEqual(traffic[0].sim_T, 0)
        self.assertEqual(traffic[0].S_v_s[0], 108)
        self.assertEqual(traffic[1].sim_T, 1)

    def test_lane_change_state_survives_pause(self):
        data = fixture_scenario()
        data['vehicles'][0]['initial_distance'] = 30
        data['vehicles'][3]['initial_distance'] = 26
        messages = self.run_scene([(0, 100, True), (0.1, 100, True), (0.2, 100, False),
                                   (50, 105, False), (51, 105, True), (51.1, 105, True)], data)
        traffic = messages['/traffic_sim_info_mache']
        holograms = messages['/virtual_sim_info_mache']
        # The merge starts on the real side map and targets the lane-0 centerline.
        self.assertEqual(holograms[0].S_v_y[2], -7)
        self.assertEqual(holograms[0].S_v_yaw[2], -0.1)
        # The target lane is at z=0, but the original side-lane path stays at z=2.
        for message in holograms:
            self.assertEqual(message.S_v_z[2], 2)
            self.assertEqual(message.S_v_pitch[2], -0.02)
        self.assertLess(traffic[1].S_v_l[2], 1)
        for index in (2, 3, 4):
            self.assertEqual(traffic[index].S_v_l[2], traffic[1].S_v_l[2])
            self.assertEqual(traffic[index].S_v_s[2], traffic[1].S_v_s[2])
            self.assertEqual(holograms[index].S_v_yaw[2], holograms[1].S_v_yaw[2])
        self.assertLess(traffic[5].S_v_l[2], traffic[1].S_v_l[2])

    def test_lane_change_tracks_original_path_elevation(self):
        data = fixture_scenario()
        data['vehicles'][0]['initial_distance'] = 30
        data['vehicles'][3]['initial_distance'] = 26
        for grade in (-0.05, 0.05):
            with self.subTest(grade=grade):
                messages = self.run_scene(
                    [(0, 100, True), (0.1, 100, True), (0.2, 100, True),
                     (0.3, 100, False), (10, 100, False), (11, 100, True),
                     (11.1, 100, True)], data, side_grade=grade)
                traffic = messages['/traffic_sim_info_mache']
                holograms = messages['/virtual_sim_info_mache']
                for state, pose in zip(traffic, holograms):
                    # Ego's height/pitch are zero in the fixture, so local z equals world z.
                    self.assertAlmostEqual(pose.S_v_z[2], 2.0 + grade * state.S_v_s[2])
                    self.assertEqual(pose.S_v_pitch[2], -0.02)
                self.assertNotEqual(holograms[0].S_v_z[2], holograms[2].S_v_z[2])
                self.assertEqual(holograms[2].S_v_z[2], holograms[4].S_v_z[2])


    def test_obstacle_following_stays_active_through_stop_and_resume(self):
        data = fixture_scenario()
        data['vehicles'][0]['initial_distance'] = 30
        data['vehicles'][3]['initial_distance'] = 26
        frames = [(i * 0.1, 100, True) for i in range(101)]
        frames += [(50, 105, False), (100, 110, False), (101, 110, True), (101.1, 110, True)]
        messages = self.run_scene(frames, data)
        traffic = messages['/traffic_sim_info_mache']
        positions = [state.S_v_s[1] for state in traffic]
        self.assertEqual(len(traffic), len(frames))
        for before, after in zip(positions, positions[1:]):
            self.assertGreaterEqual(after - before, -1e-9)
            self.assertLessEqual(after - before, 0.201)
        self.assertEqual(traffic[-1].S_v_sv[1], 0)
        self.assertLessEqual(positions[-1], 120)  # Truck at 126, clearance 6m.
        self.assertEqual(positions[-5:], [positions[-5]] * 5)
        self.assertGreater(traffic[-1].S_v_s[0], positions[-1])

    def test_stationary_braking_integrates_only_until_rest(self):
        from traffic_runtime import update_vehicle_against_stationary
        manager = SimpleNamespace(traffic_s=[32.0], traffic_v=[0.1], traffic_alon=[0.0])
        idm = SimpleNamespace(
            safe_IDM_acceleration=lambda **kw: -6.0,
            CBF_acceleration_filter=lambda **kw: (kw['commanded_acc'], None, None, None),
        )
        for _ in range(5):
            self.assertTrue(update_vehicle_against_stationary(
                manager, idm, 0, 40.0, 1.0, 15.0, -6.0, 3.0, following_active=True))
            self.assertAlmostEqual(manager.traffic_s[0], 32.0 + 0.1 ** 2 / 12.0)
            self.assertEqual(manager.traffic_v[0], 0)

    def test_real_map_files_use_independent_centerlines(self):
        import numpy as np
        readers = [runtime.road_reader(
            str(ROOT / 'maps' / name), str(ROOT / 'speed_profile/I85_ftp1_ITIC_HAMPC.csv'),
            closed_track=False,
        ) for name in ('CMI_outerloop_mach_e.csv', 'CMI_outerloop_mach_e_wider_lane.csv')]
        for reader in readers:
            reader.read_map_data()
        geometry = LaneMapGeometry(readers)
        initial_pose = readers[0].find_traffic_vehicle_poses(100.0, lane_id=0)
        ego_pose = np.array(initial_pose[:3]).reshape(3, 1)
        ego_s = readers[0].find_ego_vehicle_distance_reference(ego_pose)[0]
        side_s = readers[1].find_ego_vehicle_distance_reference(ego_pose)[0]
        self.assertTrue(geometry.align_to_ego(ego_pose, ego_s))
        np.testing.assert_allclose(geometry.pose(ego_s + 18, 1),
                                   readers[1].find_traffic_vehicle_poses(side_s + 18, lane_id=0))
        np.testing.assert_allclose(geometry.pose(ego_s + 8, 0),
                                   readers[0].find_traffic_vehicle_poses(ego_s + 8, lane_id=0))
        synthetic_pose = readers[0].find_traffic_vehicle_poses(ego_s + 18, lane_id=1)
        self.assertFalse(np.allclose(geometry.pose(ego_s + 18, 1)[:2], synthetic_pose[:2]))


if __name__ == '__main__':
    unittest.main()
