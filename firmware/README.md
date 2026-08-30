# STM32 Nucleo Firmware

This directory is reserved for the V2 firmware implementation.

An STM32 project has not been generated yet because several pending decisions
directly affect clock, peripheral, DMA, driver, and USB configuration:

- exact Nucleo and STM32 model;
- analog front end or ADC and its electrical interface with the PCB;
- sampling rate and number of simultaneous channels;
- transport to the computer;
- power, isolation, and safety strategy.

After these items are confirmed, the firmware should use these layers:

```text
board/       board-specific clock, pins, and peripherals
drivers/     ADC/front-end and transport drivers
acquisition/ timing, DMA, buffers, and timestamps
protocol/    serialization, integrity, and versioning
app/         state machine, telemetry, and error handling
tests/       tests for hardware-independent modules
```

The ESP32-S3 firmware from V1 must not be reused automatically.
