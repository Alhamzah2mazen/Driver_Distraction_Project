# CCTV-Based Driver Distraction Detection System

## Overview
An AI-powered system designed to detect driver distraction through vehicle windshields using CCTV feeds. The system utilizes a dual-stage architecture:
1. **Stage 1 (The Filter):** YOLOv10 for vehicle/windshield ROI and ALPR.
2. **Stage 2 (The Brain):** Spatio-Temporal Dual-Encoder for behavioral analysis.

## Current Milestone
- [x] Project Structure & Environment Setup
- [ ] Milestone 1: YOLOv10 & ALPR Implementation
- [ ] Milestone 2: Spatio-Temporal Behavioral Analysis

## Hardware
- **Development:** i7-2860QM (CPU Inference)
- **Training:** Google Colab (T4 GPU)