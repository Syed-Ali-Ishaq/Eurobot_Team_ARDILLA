#!/usr/bin/env python3
"""
MISSION CONTROLLER - EUROBOT 2026 (Robust Patch Version)
Compatible with production ArUco perception node
"""

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from eurobot_interfaces.msg import CrateDetectionArray


HOME_X = 0.30
HOME_Y = 0.30
HOME_YAW = 0.0


class MissionController(Node):

    MAX_NAV_RETRIES = 3

    def __init__(self):
        super().__init__('mission_controller')

        self._declare_parameters()
        self._load_parameters()

        # FSM
        self.state = 'INIT'
        self.state_start = self.get_clock().now()

        # Perception
        self.current_detection = None
        self.last_detection_time = None

        # Nav
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._nav_goal_handle = None
        self._nav_result = None
        self._nav_retries = 0

        # Actuators
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.grip_pub = self.create_publisher(Float64MultiArray,
                                              '/gripper_controller/commands', 10)

        self.create_subscription(
            CrateDetectionArray,
            '/crate/detections',
            self._on_detection,
            10
        )

        self.create_timer(0.1, self._tick)

        self._log("Mission controller ready")

    # ---------------- PARAMETERS ----------------

    def _declare_parameters(self):
        self.declare_parameter('team_color', 'yellow')
        self.declare_parameter('grasp_distance', 0.15)

        self.declare_parameter('align_threshold_deg', 5.0)

        self.declare_parameter('exit_speed', 0.2)
        self.declare_parameter('exit_duration', 1.5)

        self.declare_parameter('search_speed', 0.4)

        self.declare_parameter('linear_gain', 0.8)
        self.declare_parameter('angular_gain', 1.5)

        self.declare_parameter('linear_max', 0.3)
        self.declare_parameter('angular_max', 0.45)

        self.declare_parameter('detection_timeout', 0.5)
        self.declare_parameter('confidence_threshold', 0.3)

    def _load_parameters(self):
        g = self.get_parameter

        self.team_color = g('team_color').value
        self.grasp_distance = g('grasp_distance').value

        self.align_threshold = math.radians(g('align_threshold_deg').value)

        self.exit_speed = g('exit_speed').value
        self.exit_duration = g('exit_duration').value

        self.search_speed = g('search_speed').value

        self.linear_gain = g('linear_gain').value
        self.angular_gain = g('angular_gain').value

        self.linear_max = g('linear_max').value
        self.angular_max = g('angular_max').value

        self.detection_timeout = g('detection_timeout').value
        self.confidence_threshold = g('confidence_threshold').value

    # ---------------- UTILS ----------------

    def _time(self):
        return (self.get_clock().now() - self.state_start).nanoseconds / 1e9

    def _change(self, s):
        self.get_logger().info(f"{self.state} → {s}")
        self.state = s
        self.state_start = self.get_clock().now()

    def _move(self, v=0.0, w=0.0):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._move(0.0, 0.0)

    def _has_detection(self):
        if self.current_detection is None or self.last_detection_time is None:
            return False
        age = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
        return age < self.detection_timeout

    def _angle_rad(self):
        # ArUco gives degrees → convert
        return math.radians(self.current_detection.angle)

    def _clamp(self, x, a, b):
        return max(a, min(b, x))

    # ---------------- DETECTION ----------------

    def _on_detection(self, msg):
        valid = [
            d for d in msg.detections
            if d.color == self.team_color and d.confidence >= self.confidence_threshold
        ]

        if not valid:
            return

        best = min(valid, key=lambda d: d.distance)

        self.current_detection = best
        self.last_detection_time = self.get_clock().now()

    # ---------------- FSM ----------------

    def _tick(self):

        # INIT
        if self.state == 'INIT':
            self._stop()
            if self._time() > 1.0:
                self._change('EXIT')

        # EXIT
        elif self.state == 'EXIT':
            if self._time() < self.exit_duration:
                self._move(v=self.exit_speed)
            else:
                self._stop()
                self._change('SEARCH')

        # SEARCH
        elif self.state == 'SEARCH':

            if self._has_detection():
                self._stop()
                self._change('ALIGN')
                return

            # clean rotation search
            self._move(w=self.search_speed)

        # ALIGN
        elif self.state == 'ALIGN':

            if not self._has_detection():
                self._stop()
                self._change('SEARCH')
                return

            err = self._angle_rad()

            if abs(err) < self.align_threshold:
                self._stop()
                self._change('APPROACH')
                return

            w = self._clamp(-err * self.angular_gain,
                            -self.angular_max, self.angular_max)

            self._move(w=w)

        # APPROACH
        elif self.state == 'APPROACH':

            if not self._has_detection():
                self._stop()
                self._change('SEARCH')
                return

            d = self.current_detection

            dist = d.distance
            err = self._angle_rad()

            if dist <= self.grasp_distance:
                self._stop()
                self._change('GRAB')
                return

            # safety: misalignment escape
            if abs(err) > math.radians(25):
                self._stop()
                self._change('ALIGN')
                return

            v = self._clamp((dist - self.grasp_distance) * self.linear_gain,
                            0.02, self.linear_max)

            w = self._clamp(-err * 0.6,
                            -self.angular_max * 0.5,
                            self.angular_max * 0.5)

            self._move(v=v, w=w)

        # GRAB
        elif self.state == 'GRAB':
            self._stop()
            self._change('SEARCH')  # placeholder (gripper logic already in your full system)

    # ---------------- LOG ----------------

    def _log(self, msg):
        self.get_logger().info(f"[{self.state}] {msg}")


def main():
    rclpy.init()
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()