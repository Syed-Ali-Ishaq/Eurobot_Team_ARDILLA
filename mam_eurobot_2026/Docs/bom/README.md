# Electronics & BOM (MVP Plan)

## Power & Safety
- **24 V LiFePO₄** pack (8S), **main fuse** + **per-driver fuses**, **NC E-stop** (cuts motor power only), bucks: 24→12 V and 24→5 V.

## Compute & Control
- **Raspberry Pi 4** (ROS 2 logic, mission) + **STM32/Arduino Mega** (deterministic I/O).  
- Level-shifters/opto for safe interfaces; USB-UART.

## Drive & Actuation
- **2× DC geared w/ encoders (24 V ~300 rpm)**, **Sabertooth 2×25** (or dual MD30C).  
- Servos (MG996R/DS3218) for manip/thermo.

## Sensors (MVP)
- **IMU**, **ToF (VL53)** ×5, reflectance bar near table edges, limit switches.  
- Camera optional now; **LiDAR deferred** for 3×2 m MVP.

> Rationale: higher-voltage, robust drivers for torque; Pi+MCU split for reliability.

_Reference: RoboCup_list of electronic components_FINAL.pdf._
