"""Exercise TrafficVehicle without importing the module's ROS/CasADi runtime.

Only the actual class definition is compiled, using the standard-library math
module. These are unit checks, not ROS integration tests.
"""

import ast
import math
from pathlib import Path
import unittest


source_path = Path(__file__).resolve().parents[1] / "scripts" / "sim_env_manager.py"
source_tree = ast.parse(source_path.read_text())
vehicle_definition = next(
    node for node in source_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "TrafficVehicle"
)
exec(compile(ast.Module(body=[vehicle_definition], type_ignores=[]), str(source_path), "exec"))


class TrafficVehicleTests(unittest.TestCase):
    def test_defaults_and_independent_instances(self):
        vehicle = TrafficVehicle(1)
        other = TrafficVehicle(2)
        self.assertEqual(vehicle.get_pose(), [0.0] * 5)
        self.assertEqual((vehicle.s, vehicle.l, vehicle.sv, vehicle.lv), (0.0,) * 4)
        self.assertEqual((vehicle.speed, vehicle.acceleration, vehicle.steering), (0.0,) * 3)
        self.assertFalse(vehicle.brake_status)
        self.assertEqual((vehicle.wheelbase, vehicle.max_steering_angle), (3.5, 0.5))
        self.assertEqual((vehicle.min_acceleration, vehicle.max_acceleration), (-6.0, 4.0))
        vehicle.assign_acceleration(2)
        vehicle.step(0.5)
        self.assertEqual(vehicle.speed, 1)
        self.assertEqual(other.speed, 0)
        self.assertEqual(other.acceleration, 0)

    def test_invalid_construction(self):
        invalid = [
            {"vehicle_id": value} for value in (-1, 1.5, True, "1")
        ] + [
            {"speed": -1}, {"speed": math.nan}, {"speed": math.inf},
            {"wheelbase": 0}, {"wheelbase": -1}, {"wheelbase": math.inf},
            {"max_steering_angle": 0}, {"max_steering_angle": math.pi / 2},
            {"max_steering_angle": math.nan}, {"min_acceleration": 1},
            {"max_acceleration": -1}, {"min_acceleration": -math.inf},
            {"max_acceleration": math.nan},
            {"min_acceleration": 0, "max_acceleration": 0},
            {"acceleration": 5}, {"acceleration": -7}, {"acceleration": math.nan},
            {"steering": 0.6}, {"steering": -0.6}, {"steering": math.inf},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrafficVehicle(**dict({"vehicle_id": 0}, **values))

    def test_acceleration_assignment(self):
        vehicle = TrafficVehicle(0, min_acceleration=-3, max_acceleration=2)
        for command, expected in [(-10, -3), (-3, -3), (-1, -1), (0, 0), (2, 2), (10, 2)]:
            self.assertEqual(vehicle.assign_acceleration(command), expected)
            self.assertEqual(vehicle.acceleration, expected)
        for value in (math.nan, math.inf, -math.inf):
            before = vars(vehicle).copy()
            with self.assertRaises(ValueError):
                vehicle.assign_acceleration(value)
            self.assertEqual(vars(vehicle).copy(), before)

    def test_pure_pursuit(self):
        vehicle = TrafficVehicle(0)
        self.assertEqual(vehicle.compute_steering([10, 0, 0, 0, 0]), 0)
        left = vehicle.compute_steering([10, 1, 0, 0, 0])
        self.assertAlmostEqual(left, math.atan(7 / 101))
        self.assertAlmostEqual(vehicle.compute_steering([10, -1, 0, 0, 0]), -left)
        self.assertEqual(vehicle.compute_steering([0, 1, 0, 0, 0]), 0.5)
        self.assertEqual(vehicle.compute_steering([0, -1, 0, 0, 0]), -0.5)
        self.assertEqual(vehicle.compute_steering([0, 0, 0, 0, 0]), 0)
        rotated = TrafficVehicle(1, x=5, y=7, yaw=math.pi / 2)
        self.assertAlmostEqual(rotated.compute_steering([4, 17, 0, 0, 0]), left)
        shorter = TrafficVehicle(2, wheelbase=2)
        self.assertAlmostEqual(shorter.compute_steering([10, 1, 0, 0, 0]), math.atan(4 / 101))
        before = vars(rotated).copy()
        rotated.compute_steering([8, 20, 1, 0.1, 0.2])
        after = vars(rotated).copy()
        before.pop("steering")
        after.pop("steering")
        self.assertEqual(before, after)

    def test_invalid_targets_do_not_mutate(self):
        vehicle = TrafficVehicle(0, steering=0.2)
        targets = [None, 1, [], [1, 2], [1] * 6, ["a"] * 5]
        targets += [[1, 2, 3, 4, value] for value in (math.nan, math.inf, -math.inf)]
        for target in targets:
            with self.subTest(target=target), self.assertRaises(ValueError):
                vehicle.compute_steering(target)
            self.assertEqual(vehicle.steering, 0.2)

    def test_motion_and_external_state(self):
        vehicle = TrafficVehicle(0, speed=2, z=3, pitch=0.1,
                                 s=10, l=1, sv=2, lv=0.5, brake_status=True)
        vehicle.assign_acceleration(2)
        vehicle.step(0.5)
        self.assertEqual(vehicle.get_pose(), [1, 0, 3, 0, 0.1])
        self.assertEqual(vehicle.speed, 3)
        vehicle.steering = 0.2
        vehicle.step(0.5, z=4, pitch=0.3)
        self.assertAlmostEqual(vehicle.x, 2.5)
        self.assertAlmostEqual(vehicle.y, 0)
        self.assertAlmostEqual(vehicle.yaw, 3 * math.tan(0.2) / 3.5 * 0.5)
        self.assertEqual((vehicle.z, vehicle.pitch, vehicle.speed), (4, 0.3, 4))
        self.assertEqual((vehicle.s, vehicle.l, vehicle.sv, vehicle.lv), (10, 1, 2, 0.5))
        self.assertTrue(vehicle.brake_status)

    def test_step_clips_direct_controls_and_prevents_reverse(self):
        vehicle = TrafficVehicle(0, speed=1)
        vehicle.acceleration = -100
        vehicle.steering = 100
        vehicle.step(1)
        self.assertEqual((vehicle.acceleration, vehicle.steering, vehicle.speed), (-6, 0.5, 0))
        stopped_pose = vehicle.get_pose()
        vehicle.step(1)
        self.assertEqual(vehicle.get_pose(), stopped_pose)
        self.assertEqual(vehicle.speed, 0)

    def test_invalid_steps_are_atomic(self):
        vehicle = TrafficVehicle(0, speed=2, acceleration=1)
        inputs = [{"dt": value} for value in (0, -1, math.nan, math.inf)]
        inputs += [{"dt": 0.1, key: value} for key in ("z", "pitch")
                   for value in (math.nan, math.inf, -math.inf)]
        for kwargs in inputs:
            before = vars(vehicle).copy()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                vehicle.step(**kwargs)
            self.assertEqual(vars(vehicle).copy(), before)
        for field in ("acceleration", "steering"):
            vehicle = TrafficVehicle(0, speed=2)
            setattr(vehicle, field, math.inf)
            before = vars(vehicle).copy()
            with self.assertRaises(ValueError):
                vehicle.step(0.1)
            self.assertEqual(vars(vehicle).copy(), before)


if __name__ == "__main__":
    unittest.main()
