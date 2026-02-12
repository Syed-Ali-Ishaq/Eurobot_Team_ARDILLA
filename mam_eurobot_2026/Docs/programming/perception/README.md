# Perception — eurobot_perception

**Goal**: detect **blue/yellow crates** and **green pantries** in real time; output robot-relative (x,y) in meters.

## Nodes
- **crate_perception** → publishes `CrateDetectionArray`  
- **pantry_detection_node** → publishes `PantryDetectionArray` (with unique IDs)

## Approach
- HSV color filtering → contour detection → size filtering (min area)  
- Pinhole model for distance; angle from pixel offset → convert to robot (x,y)  
- Pantry duplicate filter within **0.5 m** radius

## Parameters (examples)
- HSV ranges (blue/yellow/green), `min_area`, `focal_length_px`, `camera_topic`, `visualize`.

_Reference: Eurobot Perception Package Documentation.pdf._
