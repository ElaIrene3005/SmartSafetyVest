# Smart Vest: Real-Time Industrial Safety System
### Powered by RT-DETR, ConvLSTM, and Raspberry Pi

This repository contains the implementation of a **Smart Vest** designed for industrial safety. The system combines state-of-the-art object detection with temporal analysis to provide real-time haptic feedback to workers in high-risk environments like mining and construction.

---

## Project Structure
```
├── RT-DETR/
│   ├── code/      # Scripts for pre-processing, training, testing, and quantization
│   ├── model/           # Model weight from training
├── ConvLSTM/
│   ├── code/      # Scripts for pre-processing, training, testing, and conversion
│   ├── model/           # Model weight from training
├── Feedback_System/
│   └── main.cpp            # C++ source for haptic actuator control and LED feedback
└── deployment/
    └── full_system.py      # Main integration script for Raspberry Pi

```

## Objectives
- Hazard Detection: Real-time identification of industrial risks using RT-DETR.
- Dynamic Prediction: Utilizing ConvLSTM to analyze temporal patterns and motion.
- Active Alerting: Providing immediate physical feedback via an integrated haptic vest.
