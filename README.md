# MOUNT: MOtion-aware Ultrasound video Needle Tracking

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of **MOUNT** (MOtion-aware Ultrasound video Needle Tracking), a robust video instance segmentation model designed for accurate needle tracking in ultrasound-guided interventions.

## Overview

MOUNT is a two-stage instance detection and tracking model developed based on [mmtracking](https://github.com/open-mmlab/mmtracking) framework. It addresses the challenges of needle tracking in ultrasound videos, including:

- Missing needle signals
- Low needle contrast
- Uneven motion between frames
- Needle bending

### Key Features

MOUNT integrates three novel modules:

1. **Uneven Motion Perception (UMP)**
   - Perceives uneven motion between key frames and reference frames
   - Aligns reference frames to the key frame using optical flow
   - Enhances the consistency of the needle target in feature space
2. **Region Proposal Network with Tip-specific Detection (TD-RPN)**
   - Focuses on the tip of the needle which has distinct characteristics
   - Uses polar coordinate system for anchor generation
   - Incorporates tip-specific detection loss
3. **Adjacent Frame Aggregation (AFA)**
   - Aggregates features from adjacent reference frames
   - Filters outlier proposals using IoU, feature similarity, and directional differences
   - Enhances needle signal through weighted aggregation
4. **Post-processing Pipeline**
   - Generates final tracking results from proposals
   - Produces entry point, tip, and shaft of the needle

## Architecture

```
Ultrasound Video Frames
       ↓
   ┌───┴───────────────────────────────────────────┐
   │  Key Frame    Reference Frames (window)       │
   └───┬───────────────────────────────────────────┘
       ↓
   ┌───────────────────────────────────────────────┐
   │          Feature Extractor (shared)           │
   └───┬───────────────────────────────────────────┘
       ↓
   ┌──────────────────┬────────────────────────────┐
   │   Reference      │       UMP Module           │
   │  Feature Maps    │  (Optical Flow + Alignment)│
   └──────────────────┴───────────┬────────────────┘
                                  ↓
                       ┌───────────────────┐
                       │   Aligned Ref     │
                       │   Feature Maps    │
                       └─────────┬─────────┘
                                 ↓
   ┌───────────────────────────────────────────────┐
   │              TD-RPN Module                    │
   │   (Tip-specific Region Proposal Network)      │
   └───────────────────────────┬───────────────────┘
                               ↓
                   ┌───────────────────┐
                   │   Needle Proposals│
                   └─────────┬─────────┘
                             ↓
   ┌───────────────────────────────────────────────┐
   │              AFA Module                        │
   │  (Filtering + Weighted Feature Aggregation)   │
   └───────────────────────────┬───────────────────┘
                               ↓
                   ┌───────────────────┐
                   │  BBox & Mask Heads│
                   └─────────┬─────────┘
                             ↓
                   ┌───────────────────┐
                   │  Post-processing  │
                   └─────────┬─────────┘
                             ↓
              Needle Tracking Result
              (Entry + Shaft + Tip)
```

## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+
- GCC 5+
- mmcv-full 1.2.0+
- mmdet 2.13.0+
- mmtracking (this repository)

### Install Steps

1. Install mmcv-full

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

1. Install mmdetection

```bash
pip install mmdet
```

1. Clone this repository

```bash
git clone https://github.com/your-username/MOUNT.git
cd MOUNT
```

1. Install dependencies

```bash
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## Data Preparation

### Dataset Structure

```
data/
└── UltrasoundNeedle/
    └── dataset_version/
        ├── Data/
        │   ├── video_001/
        │   │   ├── 00001.jpg
        │   │   ├── 00002.jpg
        │   │   └── ...
        │   └── ...
        └── DataSet/
            └── coco_format/
                ├── fold_1/
                │   ├── train.json
                │   ├── val.json
                │   └── test.json
                └── ...
```

### Annotation Format

The annotations follow COCO format with additional needle-specific fields, including:

- Polygon annotations for needle segmentation
- Tip coordinates
- Entry point coordinates
- Shaft direction

### Configuration
Please see [Configurations](./configs/liver_needle/README.md) for more details.

## Usage

### Training

To train MOUNT on the ultrasound needle dataset:

```bash
# Single GPU training
python tools/train.py configs/liver_needle/r50_dc5_1x_c04_needle_MOUNT.py

# Multi-GPU training
bash tools/dist_train.sh configs/liver_needle/r50_dc5_1x_c04_needle_MOUNT.py 8
```

### Testing

To test the trained model:

```bash
# Single GPU testing
python tools/test.py configs/liver_needle/r50_dc5_1x_c04_needle_MOUNT.py \
    work_dirs/your_checkpoint.pth \
    --eval bbox segm

# Multi-GPU testing
bash tools/dist_test.sh configs/liver_needle/r50_dc5_1x_c04_needle_MOUNT.py \
    work_dirs/your_checkpoint.pth \
    8 \
    --eval bbox segm
```

### Inference

To run inference on a single video:

```bash
python demo/demo_vid.py \
    configs/liver_needle/r50_dc5_1x_c04_needle_MOUNT.py \
    work_dirs/your_checkpoint.pth \
    --input path/to/your/video.mp4 \
    --output path/to/output.mp4
```

## Configurations

We provide several configuration files in `configs/liver_needle/`:

| Config File                                                         | Description                       |
| ------------------------------------------------------------------- | --------------------------------- |
| `r50_dc5_1x_c04_needle_MOUNT.py`           | Full MOUNT model with all modules |
| `r50_dc5_1x_c04_needle_UMP_TDRPN.py`     | MOUNT with UMP and TD-RPN only    |
| `r50_dc5_1x_c04_needle_tip_rpn_AFA.py`   | MOUNT with TD-RPN and AFA         |
| `r50_dc5_1x_c04_needle_MOUNT_fold{1-5}.py` | Cross-validation splits           |


## License

This project is released under the Apache 2.0 license.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{mount2025,
  title={Needle Tracking for Free-hand Ultrasound-Guided Percutaneous Liver Tumor Ablations},
  author={Ningtao Liu, Shuwei Xing, Derek W. Cool, Jing Yuan, Luguang Huang, Kun Jiang, Shuiping Gou, and Aaron Fenster},
  year={2025}
}
```

## Acknowledgements

This work is built upon the excellent [mmtracking](https://github.com/open-mmlab/mmtracking) and [mmdetection](https://github.com/open-mmlab/mmdetection) frameworks.

## Contact

For questions or issues, please open an issue or contact the authors.
