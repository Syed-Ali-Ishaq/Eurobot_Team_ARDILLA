# ArUco Crate Perception Node — Design Documentation

**Package:** `eurobot_perception`  
**Node name:** `aruco_crate_perception`  
**File:** `aruco_crate_perception.py`  
**Team:** Team Yellow (ARDILLA) — Eurobot 2026

---

## 1. Purpose

This node is the robot's "eyes" for finding hazelnut crates on the competition field. It reads frames from a USB camera, detects ArUco markers attached to the crates, estimates each crate's position relative to the robot, applies temporal filtering to suppress noise, and publishes reliable crate detections to the rest of the system.

---

## 2. High-Level Data Flow

```
/camera  (sensor_msgs/Image)
    │
    ▼
[image_callback]
    │
    ├─► detect_markers()
    │       │  ArUco detection (OpenCV)
    │       │  Pose estimation (solvePnP)
    │       │  camera_to_base_link() transform
    │       │  Distance + angle computation
    │       │  Outlier rejection
    │       │  Confidence scoring
    │       └─► updates TrackedCrate objects
    │
    ├─► get_filtered_detections()
    │       └─► weighted moving average over TrackedCrate history
    │
    ├─► Publish  /crate/detections  (CrateDetectionArray)
    │
    └─► Publish  /crate/debug       (Image, if visualize=True)

/camera_info  (sensor_msgs/CameraInfo)  [optional]
    └─► camera_info_callback()
            └─► replaces default pinhole model with calibrated K + D
```

---

## 3. Key Classes

### 3.1 `TrackedCrate` (dataclass)

Maintains a short history of detections for a single marker ID so that temporal filtering can be applied.

| Field | Type | Description |
|---|---|---|
| `marker_id` | int | ArUco ID of this crate |
| `color` | str | `'blue'`, `'yellow'`, or `'empty'` |
| `positions` | `deque(maxlen=N)` | Recent `(x, y)` positions in base_link (m) |
| `distances` | `deque(maxlen=N)` | Recent Euclidean distances (m) |
| `angles` | `deque(maxlen=N)` | Recent bearing angles (rad) |
| `confidences` | `deque(maxlen=N)` | Recent confidence scores [0–1] |
| `last_seen` | float | ROS time (s) of latest detection |
| `detection_count` | int | Total detections since first seen |

**Filtering method:** all `get_filtered_*()` methods apply a **linearly increasing weight** (`np.linspace(0.5, 1.0, N)`) so the most recent reading has twice the influence of the oldest. This is a simple but effective approximation of a low-pass filter without the state-estimation complexity of a Kalman filter.

**Stability gate:** `is_stable(min_detections)` — a tracked crate is only forwarded to the mission controller once it has been seen at least `min_stable_detections` times (default 2). This prevents single-frame ghost detections from triggering robot motion.

---

### 3.2 `ArucoCratePerception` (ROS 2 Node)

The main node class. Responsibilities are split into clearly labelled sections:

| Section | Methods | What it does |
|---|---|---|
| Initialization | `_declare_parameters`, `_load_parameters`, `_validate_parameters` | ROS parameter system setup and sanity checks |
| ArUco setup | `_setup_aruco` | Builds the dictionary, detector parameters, and the marker→color mapping |
| Camera model | `_setup_camera_model`, `camera_info_callback` | Constructs a default pinhole matrix; overrides it with calibrated values if `/camera_info` is received |
| Coordinate transform | `camera_to_base_link` | Rotates the ArUco translation vector from camera frame into robot base_link frame |
| Detection | `detect_markers` | Per-frame detection pipeline: ArUco → pose → transform → filter → confidence |
| Temporal filtering | `get_filtered_detections`, `_cleanup_old_tracks` | Aggregates per-marker histories; prunes stale tracks |
| Confidence | `compute_confidence`, `_corner_quality` | Scores each detection 0–1 |
| Visualization | `draw_detection` | Draws bounding boxes, axes, labels, and arrows on the debug image |
| Main callback | `image_callback` | Orchestrates all of the above on every incoming frame |

---

## 4. Detection Pipeline (per frame)

```
image_callback()
│
├── 1. Convert ROS Image → OpenCV BGR frame (cv_bridge)
│
├── 2. detect_markers(frame, timestamp)
│       a. Convert to grayscale
│       b. aruco.detectMarkers() → corners, ids, rejected
│       c. aruco.estimatePoseSingleMarkers() → rvecs, tvecs  (one per marker)
│       d. For each detected marker ID:
│           i.   Skip if ID not in ARUCO_IDS map
│           ii.  camera_to_base_link(tvec) → (x_base, y_base)
│           iii. Compute distance = √(x² + y²)
│           iv.  Compute bearing angle = atan2(y, x)
│           v.   Skip if distance outside [min_distance, max_distance]
│           vi.  is_outlier() — skip if position jumped > max_position_jump
│           vii. compute_confidence() — skip if below min_confidence
│           viii.Update TrackedCrate history
│           ix.  Append to raw_detections list
│
├── 3. get_filtered_detections()   (if enable_filtering=True)
│       — iterate TrackedCrate dict
│       — skip unstable tracks (detection_count < min_stable_detections)
│       — return weighted-average position/distance/angle/confidence
│
├── 4. Publish CrateDetectionArray on /crate/detections
│       — angle is published in DEGREES (converted from radians here)
│
└── 5. Publish debug image on /crate/debug  (if visualize=True)
```

---

## 5. Coordinate Transform

The camera is mounted on the front of the robot with a downward pitch (`camera_pitch ≈ 0.7 rad ≈ 40°`).

**Frame conventions:**

| Frame | X | Y | Z |
|---|---|---|---|
| Camera (OpenCV) | right | down | forward (optical axis) |
| Robot base_link | forward | left | up |

**Transform steps in `camera_to_base_link(tvec)`:**

1. **Pitch rotation** around the camera's X-axis (removes the tilt):
   ```
   y_int = y_cam·cos(pitch) + z_cam·sin(pitch)
   z_int = −y_cam·sin(pitch) + z_cam·cos(pitch)
   x_int = x_cam
   ```

2. **Axis remap** to base_link:
   ```
   x_base = z_int + camera_x_offset   (camera forward → robot forward)
   y_base = −x_int                    (camera right  → robot left = negative Y)
   ```

The `camera_x_offset` (default 0.10 m) accounts for the camera being mounted in front of the robot's geometric centre.

---

## 6. Confidence Scoring

`compute_confidence()` returns a value in [0, 1] as a weighted sum of four sub-scores:

| Sub-score | Weight | What it measures |
|---|---|---|
| Corner quality | 0.30 | How square/regular the detected corners are (ratio of opposite sides and diagonals) |
| Distance score | 0.30 | Inverse-normalized distance; closer = higher |
| Angle score | 0.20 | How frontal the view is; 0° bearing = 1.0, ±90° = 0.0 |
| Size score | 0.20 | Marker perimeter as a fraction of image perimeter; larger = more reliable |

A detection is only forwarded if `confidence ≥ min_confidence` (default 0.30).

---

## 7. Outlier Rejection

Before updating the tracker, `is_outlier()` computes the Euclidean jump from the last known position. If `jump > max_position_jump` (default 0.50 m), the detection is discarded and a throttled warning is logged. This guards against momentary false positives caused by reflections or partial occlusions.

---

## 8. ROS Interfaces

### Subscriptions

| Topic | Type | Purpose |
|---|---|---|
| `/camera` (configurable) | `sensor_msgs/Image` | Input camera frames |
| `/camera_info` | `sensor_msgs/CameraInfo` | Optional calibrated intrinsics |

### Publications

| Topic | Type | Description |
|---|---|---|
| `/crate/detections` | `eurobot_interfaces/CrateDetectionArray` | Stable, filtered crate list |
| `/crate/debug` | `sensor_msgs/Image` | Annotated debug frame (if `visualize=True`) |

### `CrateDetection` message fields

| Field | Unit | Notes |
|---|---|---|
| `color` | string | `'blue'`, `'yellow'`, `'empty'` |
| `x` | m | Forward distance in base_link |
| `y` | m | Lateral offset in base_link (positive = left) |
| `distance` | m | Euclidean range |
| `angle` | **degrees** | Bearing from robot heading |
| `confidence` | [0–1] | Detection quality score |

> ⚠️ Note: `angle` is stored internally in **radians** but **published in degrees**. Consumers of `/crate/detections` must expect degrees.

---

## 9. Parameters Reference

| Parameter | Default | Description |
|---|---|---|
| `camera_topic` | `/camera` | Image subscription topic |
| `image_width` / `image_height` | 640 / 480 | Expected resolution |
| `focal_length` | 277.0 px | Default pinhole focal length |
| `marker_size` | 0.04 m | Physical ArUco marker side length (40 mm) |
| `aruco_dict` | `DICT_4X4_50` | ArUco dictionary |
| `min_distance` | 0.10 m | Minimum valid detection range |
| `max_distance` | 6.0 m | Maximum valid detection range |
| `min_confidence` | 0.30 | Minimum confidence to keep a detection |
| `min_marker_perimeter` | 30 px | Smallest accepted marker in pixels |
| `camera_pitch` | 0.7 rad | Camera tilt angle (downward positive) |
| `camera_x_offset` | 0.10 m | Camera forward offset from base_link origin |
| `camera_z_offset` | 0.20 m | Camera height (informational; not used in current transform) |
| `base_frame` | `base_link` | Output frame ID for published detections |
| `enable_filtering` | True | Enable temporal filtering |
| `filter_history_size` | 5 | Deque length per tracked crate |
| `min_stable_detections` | 2 | Detections required before publishing |
| `tracking_timeout` | 1.0 s | Stale track removal threshold |
| `enable_outlier_rejection` | True | Enable position-jump filter |
| `max_position_jump` | 0.50 m | Maximum inter-frame position change |
| `visualize` | True | Publish annotated debug image |
| `show_fps` | True | Log FPS periodically |

---

## 10. Marker ID Mapping

| ArUco ID | Crate colour |
|---|---|
| 36 | Blue |
| 47 | Yellow |
| 41 | Empty |

All other IDs detected in the frame are silently ignored.

---

## 11. Design Decisions & Trade-offs

**Weighted moving average instead of Kalman filter.** A full Kalman filter would give better noise rejection but requires tuning process/measurement noise matrices and adds complexity. The linearly-weighted deque is sufficient for the slow-moving crates in Eurobot and easier to debug on hardware.

**No TF publishing.** Positions are published as fields in the custom message rather than as TF transforms. This keeps the perception node self-contained and avoids TF tree complexity during the competition.

**Calibrated vs. default camera model.** The node starts with a simple pinhole model (square pixels, principal point at image centre). If `/camera_info` is received, it silently upgrades to the calibrated matrix and distortion coefficients. This makes the node usable even without a calibration file.

**Angle published in degrees.** The conversion happens at publish time (`math.degrees(det['angle'])`). Internal tracking always uses radians. This is a deliberate choice for human-readability in `ros2 topic echo` but requires all downstream consumers (e.g. the mission controller) to treat the angle field as degrees.

---

## 12. Performance Monitoring

- `frame_count` and `detection_count` are accumulated every callback.
- Every second (when `show_fps=True`) the node logs FPS and total detections at INFO level.
- A 0.5 s timer calls `_cleanup_old_tracks()` to prune stale entries from `tracked_crates`, preventing unbounded memory growth.
