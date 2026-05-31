# Eurobot Navigation Package — Design Documentation

**Package:** `eurobot_navigation`  
**Team:** Team Yellow (ARDILLA) — Eurobot 2026  
**Files covered:** `MissionController.py`, `scan_frame_fixer.py`, `mapper_params_online_async.yaml`

---

## 1. Package Overview

The navigation package contains three components that work together to give the robot autonomous mobility and mission execution:

| Component | Role |
|---|---|
| `SimpleMissionController` | Top-level FSM — decides what the robot does and when |
| `ScanFrameFixer` | Thin adapter that corrects the LiDAR frame ID for SLAM Toolbox |
| `mapper_params_online_async.yaml` | SLAM Toolbox configuration for online async mapping |

---

## 2. System Architecture

```
[LiDAR driver]
    │  /scan  (wrong frame_id)
    ▼
[ScanFrameFixer]
    │  /scan_fixed  (frame_id = 'lidar_link')
    ▼
[SLAM Toolbox]  ←── mapper_params_online_async.yaml
    │  /map  +  TF: map → odom → base_link
    ▼
[Nav2 stack]  ←── NavigateToPose action server
    ▲
    │  navigate_to_pose action
    │
[SimpleMissionController]
    ├── /cmd_vel  ──────────────────► [robot base]
    ├── /gripper_controller/commands ► [gripper]
    └── /crate/detections ◄─────────── [ArucoCratePerception]
```

---

## 3. `ScanFrameFixer` Node

### Purpose

Some LiDAR drivers publish `/scan` with a generic or incorrect `frame_id` (e.g. `laser` or empty). SLAM Toolbox requires the scan's `frame_id` to match a known TF frame. This node is a **minimal relay** that rewrites the header and republishes.

### How it works

```
/scan  ──► [overwrite header.frame_id = 'lidar_link'] ──► /scan_fixed
```

Every incoming `LaserScan` message is mutated in place — only `header.frame_id` changes, all scan data passes through unchanged.

### Design rationale

Rather than patching the driver or adding a URDF static transform with the wrong name, this tiny node decouples the driver's output format from the SLAM Toolbox's expectation. It costs essentially zero CPU and can be removed once the driver is fixed upstream.

### ROS Interfaces

| Direction | Topic | Type |
|---|---|---|
| Subscribes | `/scan` | `sensor_msgs/LaserScan` |
| Publishes | `/scan_fixed` | `sensor_msgs/LaserScan` |

---

## 4. SLAM Toolbox Configuration (`mapper_params_online_async.yaml`)

SLAM Toolbox runs in **online asynchronous mapping** mode, building a live occupancy grid from LiDAR scans while the robot moves.

### Frame setup

| Parameter | Value | Meaning |
|---|---|---|
| `odom_frame` | `odom` | Wheel odometry frame |
| `map_frame` | `map` | Global SLAM output frame |
| `base_frame` | `base_link` | Robot centre frame |
| `scan_topic` | `/scan_fixed` | Uses the corrected scan from ScanFrameFixer |
| `mode` | `mapping` | Live SLAM (not localisation-only) |

### Key tuning decisions

**Map resolution — 0.05 m/cell (5 cm)**  
Appropriate for Eurobot's 3×2 m table. Fine enough to detect crates (150×50 mm) while keeping the map small enough to update in real time on the Horizon RDK X3 Pi's ARM CPU.

**Map update interval — 5.0 s**  
The published `/map` topic refreshes every 5 seconds. This is intentionally slow to reduce CPU load; the robot's local costmap (managed by Nav2) updates much faster from the live scan.

**Minimum travel thresholds — 0.5 m / 0.5 rad**  
A new scan is only added to the map graph when the robot has moved at least 0.5 m or rotated at least 0.5 rad since the last keyframe. This prevents the map from growing with redundant scans when the robot is nearly stationary (e.g. during grasping).

**Loop closure — enabled**  
`do_loop_closing: true` with a 3 m search radius. On a 3×2 m table the robot will frequently revisit areas, so loop closure is important for keeping the map consistent. The chain size of 10 means at least 10 keyframes must form the loop before it is applied, avoiding spurious corrections.

**Solver — Ceres with Sparse Normal Cholesky**  
The Ceres solver with LEVENBERG_MARQUARDT trust region is a standard robust choice for pose graph optimisation. `SCHUR_JACOBI` preconditioning accelerates convergence on sparse graphs typical of small-environment SLAM.

**LiDAR range — 0.0–20.0 m (for rastering)**  
These are the image rastering bounds used internally by SLAM Toolbox, not a filter on the incoming scan. Actual useful range on the Eurobot table is ≈ 0.1–3.0 m.

**`restamp_tf: false`**  
Timestamps on the published `map → odom` TF transform are taken directly from the scan stamp, not re-stamped to `now()`. This is important for avoiding TF extrapolation errors that were a known issue in simulation.

### Parameter summary table

| Parameter | Value | Effect |
|---|---|---|
| `resolution` | 0.05 m | Grid cell size |
| `map_update_interval` | 5.0 s | How often the `/map` topic is refreshed |
| `minimum_travel_distance` | 0.5 m | Min movement before new keyframe |
| `minimum_travel_heading` | 0.5 rad | Min rotation before new keyframe |
| `scan_buffer_size` | 10 | Rolling buffer of recent scans |
| `do_loop_closing` | true | Enable loop closure correction |
| `loop_search_maximum_distance` | 3.0 m | Max distance to search for loop candidates |
| `transform_timeout` | 0.2 s | Max wait for a TF lookup |
| `tf_buffer_duration` | 30.0 s | How long TF history is kept |
| `enable_interactive_mode` | true | Allows RViz-based map editing |

---

## 5. `SimpleMissionController` Node

### Purpose

This is the robot's brain. It reads crate detections from the perception node, drives the robot through a simple linear state machine to collect a crate of the team colour, and returns it to the home/deposit zone using Nav2.

### State Machine

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
   ┌────▼────┐   1 s    ┌───────────┐   1.5 s  ┌────────┐        │
   │  INIT   ├─────────►│ EXIT_ZONE ├──────────►│ SEARCH │        │
   └─────────┘          └───────────┘           └───┬────┘        │
  open gripper         drive forward              rotate/nudge    │
                                                    │             │
                              detection seen        │             │
                              ◄─────────────────────┘             │
                              │                                   │
                         ┌────▼────────┐                          │
                         │ GO_TO_CRATE │  lost crate → SEARCH     │
                         └─────┬───────┘                          │
                    proportional servo                            │
                    (linear + angular)                            │
                               │ dist ≤ 0.15 m                   │
                          ┌────▼────┐                             │
                          │  GRAB   │  2 s close gripper          │
                          └────┬────┘                             │
                               │                                  │
                          ┌────▼────┐                             │
                          │ BACKUP  │  1 s reverse                │
                          └────┬────┘                             │
                               │                                  │
                          ┌────▼────┐                             │
                          │ GO_HOME │  Nav2 NavigateToPose        │
                          └────┬────┘  retry on failure           │
                               │ succeeded                        │
                          ┌────▼────┐                             │
                          │ RELEASE │  1.5 s open gripper         │
                          └────┬────┘                             │
                               │                                  │
                          ┌────▼────┐                             │
                          │ SUCCESS │─────────────────────────────┘
                          └─────────┘  (TODO: loop back to SEARCH)
```

### State descriptions

#### `INIT` (1 second)
Opens the gripper to ensure a known starting position. Transitions to `EXIT_ZONE` after 1 s.

#### `EXIT_ZONE` (1.5 seconds)
Drives forward at 0.2 m/s to clear the starting zone boundary. Pure open-loop timed move — no sensing required. Transitions to `SEARCH`.

#### `SEARCH`
Rotates in place at 0.4 rad/s, scanning for crates. This gives a full 360° sweep in approximately 16 s. After each full rotation, nudges forward by 0.15 m/s for one tick to change viewpoint if no crate is found.

Transitions to `GO_TO_CRATE` as soon as `has_recent_detection()` returns true (a detection of the correct team colour within the last 2 s).

#### `GO_TO_CRATE`
Visual servoing loop running at 10 Hz. The robot drives directly toward the closest valid crate using two proportional controllers:

**Angular controller:**
```
angular_speed = −angle_rad × 1.5
clamped to [−0.4, +0.4] rad/s
```
The negative sign corrects the convention: a positive bearing angle (crate to the left) requires negative (clockwise) rotation to centre it.

**Linear controller:**
```
distance_error = distance − grasp_distance   (grasp_distance = 0.15 m)
linear_speed   = distance_error × 0.8
clamped to [0.02, 0.30] m/s
```
The floor of 0.02 m/s prevents the robot from stopping just short of the crate due to proportional controller droop. The 0.15 m grasp distance is chosen so the crate is within gripper reach without the camera losing sight of the marker (camera FOV loss occurs below ~0.20 m).

If the crate is lost mid-approach (`has_recent_detection()` returns false), the robot stops and returns to `SEARCH`.

> ⚠️ **Angle unit note:** The detection message's `angle` field is published in **degrees** by the perception node (see `aruco_crate_perception.py` line 679), but the mission controller treats it as **radians**. This means there is currently a unit mismatch — the angular gain of 1.5 effectively operates on a much larger number than intended. The comment in the code (`# FIX 1`) acknowledges this: if the detector publishes degrees, a `math.radians(d.angle)` conversion should be added here.

#### `GRAB` (2 seconds)
Closes the gripper by publishing `[−0.04, −0.04]` to `/gripper_controller/commands`. After 2 s sets `crate_grabbed = True` and transitions to `BACKUP`.

#### `BACKUP` (1 second)
Drives in reverse at −0.15 m/s to pull the crate away from its original position and give Nav2 room to plan a path. After 1 s transitions to `GO_HOME`.

#### `GO_HOME`
Sends a `NavigateToPose` goal to Nav2 targeting `(HOME_X, HOME_Y, HOME_YAW)` — by default `(0.3, 0.3, 0.0)` in the map frame. This is where the deposit zone is.

The goal is sent **once** on first entry (detected by `_nav_goal_handle is None and _nav_result is None`). The state then waits for the async result callback. On failure, the goal is reset and retried indefinitely (a retry counter is noted as a TODO).

Nav2 feedback (distance remaining) is logged at every feedback tick.

#### `RELEASE` (1.5 seconds)
Opens the gripper by publishing `[+0.04, +0.04]`, depositing the crate. Sets `crate_grabbed = False` after 1.5 s, transitions to `SUCCESS`.

#### `SUCCESS`
Stops the robot and logs mission completion. Currently terminal — a TODO exists to loop back to `SEARCH` if match time remains.

---

### Visual Servoing Controller Summary

| Parameter | Value | Notes |
|---|---|---|
| Angular gain | 1.5 | Applied to angle in radians |
| Angular clamp | ±0.4 rad/s | Prevents over-spinning |
| Linear gain | 0.8 | Applied to distance error |
| Linear clamp | 0.02–0.30 m/s | Floor avoids droop; ceiling avoids collisions |
| Grasp distance | 0.15 m | Stop threshold (within gripper reach) |
| Detection timeout | 2.0 s | Maximum age of a "recent" detection |

---

### Detection Filtering in the Controller

`on_detection()` filters the incoming `CrateDetectionArray` to keep only entries where:
- `color == self.team_color` (default: `'yellow'`)
- `confidence > 0.3`

Among the valid detections it selects the **closest** one by `distance`. This means the robot always pursues the nearest accessible crate of its colour, which is the greedy-optimal strategy for Eurobot scoring.

---

## 6. ROS Interfaces — `SimpleMissionController`

### Subscriptions

| Topic | Type | Purpose |
|---|---|---|
| `/crate/detections` | `eurobot_interfaces/CrateDetectionArray` | Crate position and colour from perception |

### Publications

| Topic | Type | Purpose |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Drive commands to robot base |
| `/gripper_controller/commands` | `std_msgs/Float64MultiArray` | Gripper open/close commands |

### Action Clients

| Action | Type | Purpose |
|---|---|---|
| `navigate_to_pose` | `nav2_msgs/NavigateToPose` | Autonomous navigation to home pose |

---

## 7. Key Configuration Constants

These are hardcoded at the top of `MissionController.py` and should be updated before each match:

| Constant | Default | Description |
|---|---|---|
| `HOME_X` | 0.3 m | X coordinate of deposit zone in map frame |
| `HOME_Y` | 0.3 m | Y coordinate of deposit zone in map frame |
| `HOME_YAW` | 0.0 rad | Robot heading at deposit zone |
| `self.team_color` | `'yellow'` | Which crate colour to collect |
| `self.grasp_distance` | 0.15 m | Stop distance for grasping |

---

## 8. Known Issues & TODOs

| Issue | Location | Notes |
|---|---|---|
| Angle unit mismatch | `GO_TO_CRATE` | Perception publishes degrees; controller treats as radians. Add `math.radians(d.angle)` conversion. |
| No retry limit on GO_HOME | `GO_HOME` | Nav2 failure retries indefinitely. Add a counter and fall back gracefully. |
| No multi-crate loop | `SUCCESS` | After one successful delivery, the robot stops. Loop back to `SEARCH` to collect more crates. |
| Open-loop EXIT_ZONE | `EXIT_ZONE` | 1.5 s forward drive is not distance-controlled; result varies with battery voltage. |
| HOME pose hardcoded | top of file | Should be a ROS parameter or loaded from a config file. |
