# Player Re-Identification in Sports Footage

## Overview

This project implements a player re-identification system for sports analytics. The goal is to assign and maintain consistent player IDs in a single video feed, even when players leave and re-enter the frame.

## Problem Statement

Given a 15-second sports video clip, the system should:

1. Detect players and the ball using YOLOv11
2. Assign unique IDs to each player
3. Maintain consistent IDs even when players go out of view and return
4. Track player positions and movements

## Approach

- **Detection**: Use provided YOLOv11 model for player/ball detection
- **Tracking**: Implement SORT (Simple Online and Realtime Tracking) algorithm
- **Re-Identification**: Use appearance-based features to re-identify players
- **ID Management**: Maintain consistent player IDs across frames

## Project Structure

```text
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                  # Main execution script
├── player_tracker.py        # Core tracking and re-identification logic
├── detection.py             # YOLOv11 detection wrapper
├── utils.py                 # Utility functions
├── config.py                # Configuration parameters
├── 15sec_input_720p.mp4     # Input video
└── output/                  # Output directory for results
    ├── tracked_video.mp4    # Video with tracking visualization
    └── tracking_results.json # Tracking data
```

## Model Download

**Important:** The YOLOv11 model file (`best.pt`) is not included in this repository due to GitHub's file size limits (186MB > 100MB limit).

To run this project, you need to:

1. **Download the model file** from: [YOLOv11 Model (best.pt)](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
2. **Place `best.pt` in the project root directory** (same level as `main.py`)
3. **Ensure the file is named exactly `best.pt`**

The model file is the same YOLOv11 model provided in the original assignment materials.

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics (for YOLOv11)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py --input 15sec_input_720p.mp4 --model best.pt --output output/
```

## Dependencies

- opencv-python
- torch
- ultralytics
- numpy
- scikit-learn
- matplotlib

## Output

- **Visual Output**: Video with bounding boxes, player IDs, and tracking lines
- **Data Output**: JSON file with frame-by-frame tracking data
- **Metrics**: Tracking accuracy and re-identification performance

## Methodology

1. **Frame Processing**: Extract frames from input video
2. **Object Detection**: Use YOLOv11 to detect players and ball
3. **Tracking**: Apply SORT algorithm for temporal consistency
4. **Re-Identification**: Use appearance features for player matching
5. **Visualization**: Draw bounding boxes, IDs, and tracking paths

## Challenges and Solutions

- **Occlusion**: Handle players being hidden behind others
- **Re-entry**: Match returning players with previous appearances
- **ID Consistency**: Maintain stable IDs across the entire sequence
- **Performance**: Optimize for real-time processing

## Future Improvements

- Multi-camera support
- Advanced appearance modeling
- Real-time processing optimization
- Integration with sports analytics platforms
