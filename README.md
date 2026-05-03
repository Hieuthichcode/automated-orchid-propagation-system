# Automated Orchid Propagation System

> **Graduation Thesis (DATN)** — An integrated robotic system for automated orchid shoot cutting, gripping, and branching using a custom 6-DOF robot arm and 2D/3D vision.

---

## Demo

<p align="center">
  <img src="docs/images/system_diagram.png" alt="System Architecture" width="700"/><br/>
  <em>System architecture — Embedded vision computer + 6-DOF robot arm with RGB-D sensing</em>
</p>

<p align="center">
  <img src="docs/images/robot_demo.jpg" alt="Robot Demo" width="700"/><br/>
  <em>Physical prototype — 6-DOF robot arm performing orchid shoot manipulation</em>
</p>

---

## Overview

This project combines a custom-built **6-DOF serial robot arm** with an **Intel RealSense D435i depth camera** and a **YOLOv8 instance segmentation model** to autonomously detect, locate, and cut orchid shoots (*mầm lan*) — replacing manual labor in orchid propagation.

### System Pipeline

```
RealSense D435i Camera
        │
        ▼
YOLOv8m Instance Segmentation (v8m-seg-832.pt)
        │  detect & segment each orchid shoot
        ▼
Skeleton Analysis → Tip Detection → Cutting Point & Angle
        │  3D coordinate extraction from depth map
        ▼
JSON Export (cutting coordinates + orientation)
        │
        ▼
Arduino Mega 2560 — Inverse Kinematics (on-board)
        │  quintic trajectory interpolation
        ▼
6-DOF Stepper Robot → Cut / Grip / Branch
```

---

## Repository Structure

```
├── CAD/                        # SolidWorks 3D models (.SLDPRT, .SLDASM, .STEP)
│   ├── ROBOT_6DOF_ASSEM.SLDASM   # Full robot assembly
│   ├── ROBOT_6DOF_ASSEM.STEP     # Export for other CAD tools
│   ├── base.SLDPRT, J6.SLDPRT … # Individual part files
│   └── …
│
├── Code_robot_6DOF/
│   └── Code_robot_6DOF.ino     # Arduino Mega 2560 firmware
│
└── Vision2D-3D/
    ├── realsense_gui_advanced.py # Main vision GUI application
    ├── test_segment.py           # Standalone YOLO segmentation test
    ├── view_pointcloud.py        # Point cloud viewer utility
    ├── ply_editor.py             # PLY file editor
    ├── draw.py                   # Evaluation plots (speed vs. accuracy)
    ├── v8m-seg-832.pt            # Trained YOLOv8m segmentation model
    ├── roi_config.txt            # Saved ROI configuration
    ├── Output_image/             # Exported result images
    └── Output_pointcloud/        # Exported .pcd/.ply + JSON tip data
```

---

## Hardware

| Component | Specification |
|---|---|
| Robot controller | Arduino Mega 2560 |
| Joints | 6× Stepper motors (TB6600 drivers) |
| Depth camera | Intel RealSense D435i |
| End-effector | Pneumatic gripper (SMC MHZ2-16D) |
| Gripper control | Digital pin D3 (HIGH = close) |
| Task triggers | Digital pins D2, D14, D15 |

### Robot DH Parameters

| Parameter | Value |
|---|---|
| d1 | 231.5 mm |
| a2 | 221.68 mm |
| d3 | 224.75 mm |
| d6 | 202.64 mm |

### Steps Per Revolution (per joint)

| Joint | Steps/Rev |
|---|---|
| J1 | 27 333 |
| J2 | 18 000 |
| J3 | 18 000 |
| J4 | 6 400 |
| J5 | 12 000 |
| J6 | 8 192 |

---

## Software

### 1. Robot Firmware — `Code_robot_6DOF.ino`

- **Inverse Kinematics** computed entirely on-board (no ROS required)
  - Spherical wrist decomposition (analytical IK)
  - DH standard convention (ZYX RPY)
  - Singularity handling + joint limit enforcement
- **Quintic polynomial trajectory** interpolation for smooth motion
- **Coordinated multi-joint motion** — all joints arrive simultaneously
- **Command queue** — up to 100 waypoints (`G` / `GR` commands over Serial)
- **3 motion profiles:** `SMOOTH`, `BALANCED`, `FAST`
- **Speed multiplier** — runtime adjustable (10% – 300%)
- **Hardware task lists** — 3 pre-loadable programs triggered by D2/D14/D15
- **Arc motion** (`GR` command) with configurable steps

#### Serial Command Format

```
G x y z roll pitch yaw [delay] [j6_extra] [d3_state] [gripper_delay]
GR cx cy cz x y z roll pitch yaw [...]   # Arc motion around center
HOME                                      # Return to home position
SPEED <multiplier>                        # e.g. SPEED 1.5
PROFILE <0|1|2>                          # SMOOTH / BALANCED / FAST
STATUS                                    # Print current joint angles
```

---

### 2. Vision Application — `realsense_gui_advanced.py`

**Requirements:** Python 3.8+, see [Dependencies](#dependencies)

```bash
cd Vision2D-3D
python realsense_gui_advanced.py
```

#### Features

- **Real-time RGB + Depth preview** at 1280×720 @ 30 FPS (cropped to 720×720)
- **YOLOv8m Instance Segmentation** — runs every 5 frames for stability
- **Skeleton analysis** of each segmented shoot:
  - Tip detection (endpoints of skeleton branches)
  - Cutting point & cutting angle calculation
  - 2D (image-plane) or 3D (metric) base coordinate modes
- **3D Point Cloud export** (`.pcd` / `.ply`) with configurable quality
  - Outlier removal, smoothing, bilateral filter
  - Per-instance point cloud or combined
- **JSON tip export** — 3D coordinates + orientation for each instance
- **ROI (Region of Interest)** — draw, save, and auto-load detection area
- **Picking frame visualization** — coordinate axes at stem for robot planning

#### Output Files

| File | Description |
|---|---|
| `Output_pointcloud/*.ply` | Point cloud per instance |
| `Output_pointcloud/*_tips.json` | Tip positions (3D) + cut angle |
| `Output_image/*.png` | Annotated result images |

---

## Dependencies

### Python (Vision)

```bash
pip install pyrealsense2 numpy opencv-python Pillow open3d ultralytics scipy scikit-image
```

| Package | Version (tested) |
|---|---|
| pyrealsense2 | ≥ 2.54 |
| ultralytics (YOLOv8) | ≥ 8.0 |
| open3d | ≥ 0.17 |
| opencv-python | ≥ 4.8 |
| scipy | ≥ 1.11 |
| scikit-image | ≥ 0.21 |

### Arduino

- [AccelStepper](https://www.airspayce.com/mikem/arduino/AccelStepper/) library

---

## CAD Models

All mechanical parts are designed in **SolidWorks**. A STEP export (`ROBOT_6DOF_ASSEM.STEP`) is included for compatibility with other CAD tools (Fusion 360, FreeCAD, etc.).

Third-party component models included:
- `stepper 24BYJ-48` — Seeed Studio
- `28BYJ-48` stepper + ULN2003 driver board
- `NEMA17 42-40` stepper
- `Servo SG90`
- `SMC MHZ2-16D` pneumatic gripper
- `TB6600` stepper driver
- V-slot rail wheel (625-2Z bearing)

---

## License

This project is developed as a university graduation thesis. Third-party CAD models are subject to their original terms of use (see `CAD/readme-and-terms-of-use-3d-cad-models.txt`).
