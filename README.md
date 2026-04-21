# Smart Vest: Real-Time Industrial Safety System
### Powered by RT-DETR, ConvLSTM, and Raspberry Pi

This repository contains the implementation of a **Smart Vest** designed for industrial safety. The system combines state-of-the-art object detection with temporal analysis to provide real-time haptic feedback to workers in high-risk environments like mining and construction.

---

## Project Structure
```
├── detection_rtdetr/
│   ├── dataset_tools/      # Scripts for splitting datasets and BBox counting
│   ├── training/           # RT-DETR training pipelines
│   └── optimization/       # INT8 Quantization and inference testing
├── temporal_convlstm/
│   ├── preprocessing/      # Frame cutting and sequence preparation
│   ├── training/           # ConvLSTM model training
│   └── conversion/         # FP32 model conversion and testing
├── feedback_firmware/
│   └── main.cpp            # C++ source for haptic actuator control
└── deployment/
    └── full_system.py      # Main integration script for Raspberry Pi

```

## Objectives
- Hazard Detection: Real-time identification of industrial risks using RT-DETR.
- Dynamic Prediction: Utilizing ConvLSTM to analyze temporal patterns and motion.
- Active Alerting: Providing immediate physical feedback via an integrated haptic vest.
