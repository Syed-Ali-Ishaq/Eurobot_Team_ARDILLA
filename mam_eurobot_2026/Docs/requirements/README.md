# System Requirements v0.1 — Sinjab (Eurobot 2026)

This page derives concrete constraints from the official rules to guide simulation, mechanical design, and programming.

## 1) Playing Field & Time
- Table: **3000 × 2000 mm**; bordered; two team nests (start/finish).  
- Match duration: **100 s** (no human interaction); **3 min** setup before start.

## 2) Main Robot Constraints
- Perimeter at start: **≤ 1200 mm** (convex hull of top view).
- Perimeter deployed: **≤ 1400 mm**.
- Height: **≤ 350 mm** (≤ 375 mm if E-stop protrudes).
- Must fully fit in team nest (45 × 60 cm area) at setup.
- Safety: Red emergency stop, fuse, low-voltage (≤ 48 V), no dangerous parts.

## 3) SIMA Constraints (for later integration)
- Perimeter start **≤ 600 mm**; deployed **≤ 700 mm**; height start **≤ 150 mm**; mass **≤ 1.5 kg**.
- No beacon mast; independent motion; released from SIMA start zones.

## 4) 2026 Actions to Support (Simulation First)
- **Pantry/Nest (collect & place crates)** — priority.
- **Thermometer cursor** — opportunistic.
- **Fridges with SIMA (empty/fill)** — planned later (after electronics).
- **End in Nest** — must park in team nest at end of match.

## 5) Software Architecture (initial)
- ROS 2 workspace with packages for **world**, **robot description**, **nav**, **vision**, **game logic**, **tools**.
- Sim-first: run missions; avoid borders/opponent; record KPIs.

## 6) Mechanical Interfaces (initial)
- Chassis must remain within height/perimeter constraints.
- Reserved mounts: Lidar (360°), camera, end-effector.
- Serviceability: E-stop reach; access to battery area (later).

## 7) Evidence of Compliance
- Simulation videos and logs (KPIs: time-to-pantry, crates placed, end-in-nest).
- Drawings and STEP models demonstrating envelopes and fit.
