# SINJAB Design — Overview

**SINJAB** is a compact circular robot designed to collect nuts and deliver them to the nest.
This folder hosts the **SINJAB Robot Design Documentation** (PDF) and a short summary.

## PDF
- [SINJAB_Robot_Design_Documentation.pdf](./SINJAB_Robot_Design_Documentation.pdf)

## Summary
- **Robot name:** SINJAB  
- **Goal:** Collect nuts and deliver to nest  
- **CAD:** SolidWorks  
- **Shape:** Circular robot, **Ø 200 mm**; 100 mm parallel jaw grip  
- **Main components:** body, two driven wheels + caster, stem + head, parallel-jaw grip  
- **Material:** 3D-printed plastic (PLA)  
- **Drive:** Differential (2 powered wheels + 1 caster)  
- **Authors (mechanical):** Hussein Youssef, Bashir Awad
_Source: SINJAB_Robot_Design_Documentation.pdf._

### Next actions
- Validate envelopes vs Eurobot general rules (height ≤350 mm; perimeter limits).  
- Reserve mounting for sensors (camera, ToF/LiDAR later), E-stop access, wiring paths.

> Notes for integration:
> - Keep envelopes compliant with Eurobot general rules (height ≤ 350 mm; start area fit; perimeter undeployed ≤ 1200 mm).
> - Ensure end-effector and caster configuration do not violate height/perimeter limits during operation.
> - Mounting interfaces for sensors (Lidar, camera) and future electronics should be reserved.
