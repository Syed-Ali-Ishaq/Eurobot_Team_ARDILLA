#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

from eurobot_interfaces.msg import CrateDetection, CrateDetectionArray

class CratePerception(Node):
    def __init__(self):
        super().__init__('crate_perception')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('visualize', True)
        self.declare_parameter('min_area', 2000)
        self.declare_parameter('known_crate_width', 0.12)  # meters
        self.declare_parameter('focal_length_px', 600.0)   # camera calibration
        self.declare_parameter('camera_center_offset', 0.0)  # optional, meters offset if camera not centered

        self.camera_topic = self.get_parameter('camera_topic').value
        self.visualize = self.get_parameter('visualize').value
        self.min_area = self.get_parameter('min_area').value
        self.crate_width = self.get_parameter('known_crate_width').value
        self.focal_length = self.get_parameter('focal_length_px').value
        self.camera_offset = self.get_parameter('camera_center_offset').value

        # Color ranges in HSV
        self.color_ranges = {
            'blue': ([95, 80, 40], [135, 255, 255]),
            'yellow': ([20, 100, 100], [35, 255, 255])
        }

        # Topics
        self.sub = self.create_subscription(Image, self.camera_topic, self.image_cb, 10)
        self.pub = self.create_publisher(CrateDetectionArray, '/crate/detections', 10)
        if self.visualize:
            self.debug_pub = self.create_publisher(Image, '/crate/image_debug', 1)

        self.get_logger().info(f"Subscribed to: {self.camera_topic}")
        self.get_logger().info("CratePerception Node started")

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detections = []
        debug = frame.copy()

        frame_center_x = frame.shape[1] / 2  # Image center in pixels

        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w / 2
                cy = y + h / 2

                # -------------------- MODIFIED SECTION START -------------------- #
                # Distance estimation (simple pinhole model)
                distance = (self.crate_width * self.focal_length) / w  # meters

                # Horizontal offset from image center (pixels)
                offset_px = cx - frame_center_x

                # Convert pixel offset to radians (approx.)
                angle_rad = math.atan2(offset_px, self.focal_length)

                # Convert to robot-relative coordinates (meters)
                rel_x = distance * math.cos(angle_rad) + self.camera_offset  # forward direction
                rel_y = distance * math.sin(angle_rad)                      # lateral direction

                # Confidence (relative to area)
                confidence = min(1.0, area / (frame.shape[0]*frame.shape[1]*0.1))
                # -------------------- MODIFIED SECTION END -------------------- #

                # Fill message
                det = CrateDetection()
                det.color = color
                det.x = float(rel_x)        # in meters (robot-relative)
                det.y = float(rel_y)        # in meters (robot-relative)
                det.distance = float(distance)
                det.angle = float(math.degrees(angle_rad))
                det.confidence = float(confidence)
                detections.append(det)

                if self.visualize:
                    cv2.rectangle(debug, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(debug, f"{color} {distance:.2f}m ({rel_x:.2f},{rel_y:.2f})",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if detections:
            msg_out = CrateDetectionArray()
            msg_out.detections = detections
            msg_out.header = msg.header
            self.pub.publish(msg_out)

        if self.visualize:
            img_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            img_msg.header = msg.header
            self.debug_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CratePerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#____________________Old One____________________
"""import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from eurobot_interfaces.msg import CrateDetection, CrateDetectionArray  # adjust package name

class CratePerception(Node):
    def __init__(self):
        super().__init__('crate_perception')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('visualize', True)
        self.declare_parameter('min_area', 2000)
        self.declare_parameter('known_crate_width', 0.12)  # meters
        self.declare_parameter('focal_length_px', 600.0)   # camera calibration

        self.camera_topic = self.get_parameter('camera_topic').value
        self.visualize = self.get_parameter('visualize').value
        self.min_area = self.get_parameter('min_area').value
        self.crate_width = self.get_parameter('known_crate_width').value
        self.focal_length = self.get_parameter('focal_length_px').value

        # Color ranges in HSV
        self.color_ranges = {
            'blue': ([95, 80, 40], [135, 255, 255]),
            'yellow': ([20, 100, 100], [35, 255, 255])
        }

        # Topics
        self.sub = self.create_subscription(Image, self.camera_topic, self.image_cb, 10)
        self.pub = self.create_publisher(CrateDetectionArray, '/crate/detections', 10)
        if self.visualize:
            self.debug_pub = self.create_publisher(Image, '/crate/image_debug', 1)

        self.get_logger().info(f"Subscribed to: {self.camera_topic}")
        self.get_logger().info("CratePerception node started")

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detections = []
        debug = frame.copy()

        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w / 2
                cy = y + h / 2

                # Distance estimation (simple pinhole model)
                distance = (self.crate_width * self.focal_length) / w
                angle = np.degrees(np.arctan2(cx - frame.shape[1]/2, self.focal_length))
                confidence = min(1.0, area / (frame.shape[0]*frame.shape[1]*0.1))

                det = CrateDetection()
                det.color = color
                det.x = float(cx)
                det.y = float(cy)
                det.distance = float(distance)
                det.angle = float(angle)
                det.confidence = float(confidence)
                detections.append(det)

                if self.visualize:
                    cv2.rectangle(debug, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(debug, f"{color} {distance:.2f}m", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if detections:
            msg_out = CrateDetectionArray()
            msg_out.detections = detections
            self.pub.publish(msg_out)

        if self.visualize:
            img_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            img_msg.header = msg.header
            self.debug_pub.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CratePerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
