# Electronics — SINJAB Robot

This folder contains the full electronics architecture for the SINJAB Eurobot robot.

## Canonical Power Architecture (FINAL)
- Battery: **24V LiFePO₄**
- High-power rail: **24V** (motors via Sabertooth 2x25)
- Intermediate rail: **12V** (servos and auxiliary electronics)
- Logic rail: **5V** (Raspberry Pi, MCU, sensors)

Architecture:
24V Battery → Fuse → E-Stop → 24V BUS  
24V BUS → Sabertooth 2x25  
24V BUS → Buck 24→12V → 12V BUS  
12V BUS → Buck 12→5V → 5V BUS  

This decision supersedes earlier 12V-only drafts.

## Folder Structure
- `bom/` – Bill of Materials
- `power/` – Voltage rails, power budget, fuses
- `schematics/` – KiCad schematics
- `interfaces/` – Pi ↔ MCU ↔ drivers communication
