"""Rate-limit regression checks; source the ROS workspace before running."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from sim_env_manager import lateral_vehicle_controller


class LateralControllerTests(unittest.TestCase):
    def controller(self):
        return lateral_vehicle_controller(0, 0, 0, 0, 0, 3.5,
                                          max_steering_rate=0.4, max_jerk=2.0)

    def test_rate_limits_in_both_directions(self):
        controller = self.controller()
        controller.control_signal_update(0.5, 5, 0.1)
        self.assertAlmostEqual(controller.steering, 0.04)
        self.assertAlmostEqual(controller.acc, 0.2)
        controller.control_signal_update(-0.5, -5, 0.2)
        self.assertAlmostEqual(controller.steering, -0.04)
        self.assertAlmostEqual(controller.acc, -0.2)

    def test_pursuit_does_not_bypass_limits_and_reports_applied_acceleration(self):
        controller = self.controller()
        controller.v = 5.0
        controller.pure_pursuit_controller([6, 4, 0, 0, 0])
        self.assertEqual(controller.steering, 0)
        controller.update_vehicle_state(5, 0, 0, 0.1)
        self.assertAlmostEqual(controller.steering, 0.04)
        self.assertAlmostEqual(controller.acc, 0.2)
        self.assertAlmostEqual(controller.v, 5.02)
        state = (controller.x, controller.y, controller.yaw, controller.v,
                 controller.steering, controller.acc)
        controller.update_vehicle_state(-5, 0, 0, 0)
        self.assertEqual(state, (controller.x, controller.y, controller.yaw, controller.v,
                                 controller.steering, controller.acc))

    def test_straight_target_and_target_reached_without_overshoot(self):
        controller = self.controller()
        controller.pure_pursuit_controller([15, 0, 0, 0, 0])
        self.assertEqual(controller.target_steering, 0)
        controller.control_signal_update(0.01, 0.05, 0.1)
        self.assertAlmostEqual(controller.steering, 0.01)
        self.assertAlmostEqual(controller.acc, 0.05)

    def test_launch_exposes_parameters(self):
        root = ET.parse(ROOT / 'launch/cra_traffic_sim_human_adaptive_acc_double_lane.launch').getroot()
        args = {node.attrib['name']: node.attrib['default'] for node in root.findall('arg')}
        params = {node.attrib['name']: node.attrib['value'] for node in root.findall('param')}
        for name, default in [('max_steering_rate', '2.0'), ('max_jerk', '5.0'),
                              ('min_lookahead_distance', '15.0')]:
            self.assertEqual(args[name], default)
            self.assertEqual(params[name], '$(arg ' + name + ')')


if __name__ == '__main__':
    unittest.main()
