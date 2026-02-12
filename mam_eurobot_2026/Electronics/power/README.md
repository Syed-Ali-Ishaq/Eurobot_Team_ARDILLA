# Power Architecture

## Voltage Rails
- **24V rail**
  - Source: LiFePO₄ battery
  - Consumers: Sabertooth 2x25 (motors)

- **12V rail**
  - Source: Buck converter (24V → 12V)
  - Consumers: servos, auxiliary electronics

- **5V rail**
  - Source: Buck converter (12V → 5V)
  - Consumers: Raspberry Pi, MCU, IMU, logic sensors

## Safety
- Main fuse directly after battery
- Emergency stop cuts 24V rail
- Logic rails collapse automatically when 24V is cut

See `Nominal_voltages_of_components.pdf` for per-component voltages and current estimates.
