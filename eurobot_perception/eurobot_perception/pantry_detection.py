#!/usr/bin/env python3
"""
Pantry Detection Node - Camera-based detection with home location tracking
Detects pantries by green rectangle outlines on the floor with persistent IDs
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

from eurobot_interfaces.msg import PantryDetection, PantryDetectionArray

class PantryPerception(Node):
    def __init__(self):
        super().__init__('pantry_perception')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('visualize', True)
        self.declare_parameter('min_contour_area', 200)
        self.declare_parameter('min_rect_area', 1000)
        self.declare_parameter('known_pantry_width', 0.20)
        self.declare_parameter('focal_length_px', 600.0)
        self.declare_parameter('camera_center_offset', 0.0)
        self.declare_parameter('home_detection_timeout', 5.0)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.visualize = self.get_parameter('visualize').value
        self.min_contour_area = self.get_parameter('min_contour_area').value
        self.min_rect_area = self.get_parameter('min_rect_area').value
        self.pantry_width = self.get_parameter('known_pantry_width').value
        self.focal_length = self.get_parameter('focal_length_px').value
        self.camera_offset = self.get_parameter('camera_center_offset').value
        self.home_timeout = self.get_parameter('home_detection_timeout').value

        # Home position tracking
        self.home_position = None
        self.home_captured = False
        self.start_time = self.get_clock().now()

        # Green color range for pantry markers
        self.green_lower = np.array([35, 60, 60])
        self.green_upper = np.array([85, 255, 255])
        
        # Confidence and filtering parameters
        self.declare_parameter('min_detection_confidence', 0.3)
        self.declare_parameter('clustering_distance_ratio', 0.35)  # Increased for better grouping
        self.min_confidence = self.get_parameter('min_detection_confidence').value
        self.cluster_ratio = self.get_parameter('clustering_distance_ratio').value

        # Topics
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_cb, 10
        )
        self.img_sub = self.create_subscription(
            Image, self.camera_topic, self.image_cb, 10
        )
        self.pub = self.create_publisher(PantryDetectionArray, '/pantry/detections', 10)
        
        if self.visualize:
            self.debug_pub = self.create_publisher(Image, '/pantry/image_debug', 1)

        # Persistent pantry tracking
        self.pantry_id_counter = 0
        self.detected_pantries = []  # List of {id, x, y}
        self.last_detection_time = {}
        
        self.get_logger().info(f"Subscribed to camera: {self.camera_topic}")
        self.get_logger().info(f"Subscribed to odom: {self.odom_topic}")
        self.get_logger().info("Waiting to capture home position...")

    def odom_cb(self, msg):
        """Capture home position from initial odometry"""
        if not self.home_captured:
            self.home_position = {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            }
            self.home_captured = True
            # Log only once
            self.get_logger().info(
                f"Home position captured: x={self.home_position['x']:.2f}, "
                f"y={self.home_position['y']:.2f}"
            )
            

    def find_or_create_pantry_id(self, rel_x, rel_y, distance):
        """
        Find existing pantry ID or create new one
        Returns (id, is_new) tuple
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        match_threshold = 0.7  # Increased to 70cm to handle position variations
        
        # Check if this matches an existing pantry
        best_match = None
        best_dist = float('inf')
        
        for pantry in self.detected_pantries:
            dx = pantry['x'] - rel_x
            dy = pantry['y'] - rel_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < match_threshold and dist < best_dist:
                best_match = pantry
                best_dist = dist
        
        if best_match is not None:
            # Update the position with weighted average (favor stability)
            weight = 0.3  # 30% new, 70% old position
            best_match['x'] = best_match['x'] * (1 - weight) + rel_x * weight
            best_match['y'] = best_match['y'] * (1 - weight) + rel_y * weight
            self.last_detection_time[best_match['id']] = current_time
            return best_match['id'], False
        
        # New pantry - create ID
        if distance > 0.3:  # Only add if not too close to robot
            new_id = self.pantry_id_counter
            self.pantry_id_counter += 1
            
            self.detected_pantries.append({
                'id': new_id,
                'x': rel_x,
                'y': rel_y
            })
            self.last_detection_time[new_id] = current_time
            
            self.get_logger().info(
                f"✓ New pantry ID={new_id} at ({rel_x:.2f}, {rel_y:.2f}) - distance: {distance:.2f}m"
            )
            return new_id, True
        
        return None, False

    def group_nearby_rectangles(self, rectangles, img_shape):
        """
        Group rectangles that are close together (belong to same pantry)
        Uses adaptive clustering based on rectangle sizes
        """
        if not rectangles:
            return []
        
        # More aggressive clustering to handle broken boundaries
        # Use larger threshold - 35% of image width by default
        cluster_threshold = img_shape[1] * self.cluster_ratio
        
        grouped = []
        used = set()
        
        for i, rect1 in enumerate(rectangles):
            if i in used:
                continue
            
            # Start a new group with this rectangle
            group = [rect1]
            used.add(i)
            
            # Adaptive threshold based on rectangle size
            # Larger rectangles can be further apart
            adaptive_threshold = max(cluster_threshold, 
                                    (rect1['w'] + rect1['h']) * 0.8)
            
            # Find all rectangles close to this one
            for j, rect2 in enumerate(rectangles):
                if j in used or i == j:
                    continue
                
                # Calculate distance between centers
                dx = rect1['cx'] - rect2['cx']
                dy = rect1['cy'] - rect2['cy']
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Also check if rectangles are roughly aligned (same pantry)
                # Check vertical alignment (similar Y position)
                y_diff = abs(rect1['cy'] - rect2['cy'])
                are_aligned = y_diff < img_shape[0] * 0.15  # Within 15% of image height
                
                if dist < adaptive_threshold or (are_aligned and dist < cluster_threshold * 1.5):
                    group.append(rect2)
                    used.add(j)
            
            # Merge the group into a single rectangle
            if len(group) > 1:
                merged = self.merge_rectangles(group)
                grouped.append(merged)
            else:
                grouped.append(rect1)
        
        return grouped
    
    def merge_rectangles(self, rectangles):
        """Merge multiple rectangles into one by finding bounding box"""
        min_x = min(r['x'] for r in rectangles)
        min_y = min(r['y'] for r in rectangles)
        max_x = max(r['x'] + r['w'] for r in rectangles)
        max_y = max(r['y'] + r['h'] for r in rectangles)
        
        w = max_x - min_x
        h = max_y - min_y
        
        return {
            'x': min_x,
            'y': min_y,
            'w': w,
            'h': h,
            'cx': min_x + w / 2,
            'cy': min_y + h / 2,
            'area': w * h,
            'contour': None
        }

    def detect_green_rectangles(self, frame, hsv):
        """Detect green rectangular outlines on the floor and group them"""
        # Create mask for green color
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # More aggressive morphological operations to bridge gaps
        kernel_large = np.ones((7, 7), np.uint8)
        kernel_small = np.ones((3, 3), np.uint8)
        
        # Close gaps first with large kernel
        mask = cv2.dilate(mask, kernel_large, iterations=4)
        mask = cv2.erode(mask, kernel_small, iterations=2)
        
        # Additional closing to connect broken boundaries
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_rectangles = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            
            # Filter by rectangle area
            if rect_area < self.min_rect_area:
                continue
            
            raw_rectangles.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + w / 2,
                'cy': y + h / 2,
                'area': rect_area,
                'contour': cnt
            })
        
        # Group nearby rectangles (same pantry) - more aggressive
        rectangles = self.group_nearby_rectangles(raw_rectangles, frame.shape)
        
        return rectangles, mask

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detections = []
        debug = frame.copy()
        frame_center_x = frame.shape[1] / 2

        # Always include home pantry if captured
        if self.home_captured and self.home_position is not None:
            home_det = PantryDetection()
            home_det.id = -1
            home_det.x = float(self.home_position['x'])
            home_det.y = float(self.home_position['y'])
            home_det.distance = 0.0
            home_det.angle = 0.0
            home_det.confidence = 1.0
            detections.append(home_det)

        # Detect green rectangles
        rectangles, mask = self.detect_green_rectangles(frame, hsv)

        for rect in rectangles:
            x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
            cx, cy = rect['cx'], rect['cy']

            # Distance estimation using rectangle width
            distance = (self.pantry_width * self.focal_length) / w if w > 0 else 0

            # Horizontal offset
            offset_px = cx - frame_center_x
            angle_rad = math.atan2(offset_px, self.focal_length)

            # Robot-relative coordinates
            rel_x = distance * math.cos(angle_rad) + self.camera_offset
            rel_y = distance * math.sin(angle_rad)

            confidence = min(1.0, rect['area'] / (frame.shape[0] * frame.shape[1] * 0.1))

            # Skip low confidence detections (likely noise or partial views)
            if confidence < self.min_confidence:
                continue

            # Find or create pantry ID
            pantry_id, is_new = self.find_or_create_pantry_id(rel_x, rel_y, distance)
            
            if pantry_id is None:
                continue  # Skip if too close to robot

            # Fill message
            det = PantryDetection()
            det.id = pantry_id
            det.x = float(rel_x)
            det.y = float(rel_y)
            det.distance = float(distance)
            det.angle = float(math.degrees(angle_rad))
            det.confidence = float(confidence)
            detections.append(det)

            if self.visualize:
                # Draw detected rectangle (merged bounding box)
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 255), 3)
                cv2.circle(debug, (int(cx), int(cy)), 8, (255, 0, 255), -1)
                
                # Draw text with background for better visibility
                text1 = f"Pantry ID={pantry_id}: {distance:.2f}m"
                text2 = f"Pos: ({rel_x:.2f}, {rel_y:.2f})"
                
                # Text background
                (w1, h1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(debug, (x, y-h1-25), (x+w1+5, y-5), (0, 0, 0), -1)
                
                cv2.putText(debug, text1, (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(debug, text2, (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Add home indicator and stats
        if self.visualize:
            if self.home_captured:
                cv2.putText(debug, f"HOME: ({self.home_position['x']:.2f}, {self.home_position['y']:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug, f"Unique Pantries: {len(self.detected_pantries)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug, f"Current frame: {len(rectangles)} detections",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if detections:
            msg_out = PantryDetectionArray()
            msg_out.detections = detections
            msg_out.header = msg.header
            self.pub.publish(msg_out)

        if self.visualize:
            # Show mask in corner
            mask_small = cv2.resize(mask, (160, 120))
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            debug[0:120, 0:160] = mask_bgr
            
            img_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            img_msg.header = msg.header
            self.debug_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PantryPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()