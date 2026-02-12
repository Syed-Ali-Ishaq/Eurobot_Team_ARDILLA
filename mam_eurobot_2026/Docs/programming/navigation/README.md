# Navigation & Mission — eurobot_navigation

**Goal**: autonomous SLAM/localization, path planning, and mission control (pick crates → place in pantries), built on ROS 2 Nav2.

## Components
- **SLAM** (slam_toolbox wrapper) → /map occupancy grid  
- **Localization** (AMCL wrapper) → /amcl_pose  
- **Global planner** (Nav2 planner_server wrapper) → /plan  
- **Local planner** (Nav2 controller_server wrapper) → /cmd_vel  
- **Task Manager** (custom): picks nearest crate, then nearest pantry; publishes nav goals.

## Key topic flows
- **/scan → SLAM/AMCL → /amcl_pose** (pose)  
- **Task Manager → /task_manager/nav_goal → Nav2 → /cmd_vel**  
- Integrates **/crate/detections** and **/pantry/detections** from perception.

_Reference: Eurobot Navigation Package Documentation.pdf._
