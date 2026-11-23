# Energy budget of one decision (order of magnitude, not measured)

## High assumptions

- Front end, 22 electrodes: 220 mJ per decision; 16 electrodes save 60 mJ.
- Streaming the raw 22-channel signal: 26.4 mJ per decision.
- The network on the device: original FP32 23.5 mJ, layer-swapped INT8 0.754 mJ.
- Share of the network in the best on-device total: 0.3%.
- Removing 6 electrodes saves 2.55 times the energy of running the original FP32 network.

## Low assumptions

- Front end, 22 electrodes: 0.44 mJ per decision; 16 electrodes save 0.12 mJ.
- Streaming the raw 22-channel signal: 5.28 mJ per decision.
- The network on the device: original FP32 5.89 mJ, layer-swapped INT8 0.151 mJ.
- Share of the network in the best on-device total: 25.5%.
- Removing 6 electrodes saves 0.0204 times the energy of running the original FP32 network.

## Constants

- front_end_w_per_channel: 1e-05 to 0.005. high: TI ADS1299 datasheet, 'low power: 5 mW/channel'; low: research-grade EEG front ends reported at around 10 uW per channel.
- radio_j_per_bit: 2e-08 to 1e-07. Bluetooth Low Energy: about 20 nJ/bit for the radio alone at 1 Mbit/s (nRF52-class transmit current near 6 mA at 3 V); about 100 nJ/bit once protocol overhead is included.
- mcu_j_per_mac_fp32: 1e-09 to 4e-09. Cortex-M4F-class microcontroller: about 0.1-0.3 nJ per cycle at roughly 0.25-0.5 float MACs per cycle (order of magnitude).
- mcu_j_per_mac_int8: 2e-10 to 1e-09. the same core with CMSIS-NN SIMD int8 kernels, roughly 1-2 MACs per cycle (order of magnitude).

MAC counts from results/supplementary/int8_study/costs.csv. Memory traffic, sleep current and the 2-s acquisition by the converter are not modelled.
