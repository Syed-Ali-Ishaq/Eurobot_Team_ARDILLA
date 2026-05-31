#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from collections import deque

from eurobot_interfaces.msg import CrateDetection, CrateDetectionArray

class CratePerception(Node):
    def __init__(self):
        super().__init__('crate_perception')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('visualize', True)
        self.declare_parameter('min_area', 2000)
        self.declare_parameter('max_area', 50000)  # NEW: Filter out too-large detections
        self.declare_parameter('known_crate_width', 0.12)
        self.declare_parameter('focal_length_px', 600.0)
        self.declare_parameter('camera_center_offset', 0.0)
        self.declare_parameter('min_aspect_ratio', 0.6)  # NEW: Filter non-square objects
        self.declare_parameter('max_aspect_ratio', 1.4)
        self.declare_parameter('temporal_filter_size', 5)  # NEW: Smooth detections
        self.declare_parameter('max_detection_distance', 3.0)  # NEW: meters

        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.visualize = self.get_parameter('visualize').value
        self.min_area = self.get_parameter('min_area').value
        self.max_area = self.get_parameter('max_area').value
        self.crate_width = self.get_parameter('known_crate_width').value
        self.focal_length = self.get_parameter('focal_length_px').value
        self.camera_offset = self.get_parameter('camera_center_offset').value
        self.min_aspect = self.get_parameter('min_aspect_ratio').value
        self.max_aspect = self.get_parameter('max_aspect_ratio').value
        self.filter_size = self.get_parameter('temporal_filter_size').value
        self.max_distance = self.get_parameter('max_detection_distance').value

        # Camera calibration (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_received = False

        # Color ranges in HSV (consider making these parameters for easy tuning)
        self.color_ranges = {
            'blue': ([95, 80, 40], [135, 255, 255]),
            'yellow': ([20, 100, 100], [35, 255, 255])
        }

        # Temporal filtering: store recent detections
        self.detection_history = {
            'blue': deque(maxlen=self.filter_size),
            'yellow': deque(maxlen=self.filter_size)
        }

        # Topics
        self.sub = self.create_subscription(Image, self.camera_topic, self.image_cb, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_cb, 10
        )
        self.pub = self.create_publisher(CrateDetectionArray, '/crate/detections', 10)
        
        if self.visualize:
            self.debug_pub = self.create_publisher(Image, '/crate/image_debug', 1)

        self.get_logger().info(f"Subscribed to: {self.camera_topic}")
        self.get_logger().info("CratePerception node started (improved version)")

    def camera_info_cb(self, msg):
        """Extract camera calibration from camera_info topic"""
        if not self.calibration_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.focal_length = self.camera_matrix[0, 0]  # fx
            self.calibration_received = True
            self.get_logger().info(f"Camera calibration received. Focal length: {self.focal_length:.2f}px")

    def filter_detection(self, color, x, y, distance):
        """Apply temporal filtering to smooth jittery detections"""
        self.detection_history[color].append({'x': x, 'y': y, 'distance': distance})
        
        if len(self.detection_history[color]) < 2:
            return x, y, distance
        
        # Simple moving average
        avg_x = np.mean([d['x'] for d in self.detection_history[color]])
        avg_y = np.mean([d['y'] for d in self.detection_history[color]])
        avg_dist = np.mean([d['distance'] for d in self.detection_history[color]])
        
        return avg_x, avg_y, avg_dist

    def validate_detection(self, w, h, distance):
        """Validate detection based on geometry and distance"""
        # Check aspect ratio (crates should be roughly square)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < self.min_aspect or aspect_ratio > self.max_aspect:
            return False
        
        # Check if distance is reasonable
        if distance > self.max_distance or distance < 0.1:
            return False
        
        return True

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Apply Gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detections = []
        debug = frame.copy()
        frame_center_x = frame.shape[1] / 2

        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Improved morphological operations
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the best (largest valid) detection for this color
            best_detection = None
            best_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > self.max_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                
                # Distance estimation
                distance = (self.crate_width * self.focal_length) / w
                
                # Validate detection
                if not self.validate_detection(w, h, distance):
                    continue

                # Keep track of best detection
                if area > best_area:
                    best_area = area
                    cx = x + w / 2
                    cy = y + h / 2
                    
                    # Horizontal offset and angle
                    offset_px = cx - frame_center_x
                    angle_rad = math.atan2(offset_px, self.focal_length)
                    
                    # Robot-relative coordinates
                    rel_x = distance * math.cos(angle_rad) + self.camera_offset
                    rel_y = distance * math.sin(angle_rad)
                    
                    # Apply temporal filter
                    rel_x, rel_y, distance = self.filter_detection(color, rel_x, rel_y, distance)
                    
                    # Confidence based on multiple factors
                    size_confidence = min(1.0, area / (frame.shape[0] * frame.shape[1] * 0.1))
                    aspect_confidence = 1.0 - abs(1.0 - (w/h))
                    confidence = (size_confidence + aspect_confidence) / 2.0
                    
                    best_detection = {
                        'color': color,
                        'x': rel_x,
                        'y': rel_y,
                        'distance': distance,
                        'angle': math.degrees(angle_rad),
                        'confidence': confidence,
                        'bbox': (x, y, w, h),
                        'area': area
                    }

            # Add best detection if found
            if best_detection:
                det = CrateDetection()
                det.color = best_detection['color']
                det.x = float(best_detection['x'])
                det.y = float(best_detection['y'])
                det.distance = float(best_detection['distance'])
                det.angle = float(best_detection['angle'])
                det.confidence = float(best_detection['confidence'])
                detections.append(det)

                if self.visualize:
                    x, y, w, h = best_detection['bbox']
                    cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    label = (f"{best_detection['color']} d={best_detection['distance']:.2f}m "
                            f"({best_detection['x']:.2f}, {best_detection['y']:.2f}) "
                            f"conf={best_detection['confidence']:.2f}")
                    cv2.putText(debug, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish detections
        if detections:
            msg_out = CrateDetectionArray()
            msg_out.detections = detections
            msg_out.header = msg.header
            self.pub.publish(msg_out)

        # Publish debug image
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