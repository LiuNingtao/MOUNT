# MOUNT Model Configurations

This directory contains configuration files for the **MOUNT** (MOtion-aware Ultrasound video Needle Tracking) model.

## Model Architecture

MOUNT consists of three key modules:

### 1. Uneven Motion Perception (UMP)
- Uses optical flow to align reference frames to the key frame
- Handles uneven motion, needle bending, and tissue deformation
- Configuration: `flow_generator`, `flow_encoder`, `flow_neck`

### 2. Tip-specific Detection RPN (TD-RPN)
- Specialized RPN for needle tip detection
- Uses polar coordinate system for anchor generation
- Configuration: `rpn_head.type='RPNNeedleHead'`

### 3. Adjacent Frame Aggregation (AFA)
- Filters outlier proposals using IoU, feature similarity, and directional differences
- Aggregates features from reference frames with weighted combination
- Configuration: `roi_head.type='SelsaRoIHeadBoth'`, `aggregator.type='SelsaIoUAggregator'`

## Configuration Files

### Full MOUNT Models

| File Name | Description |
|-----------|-------------|
| `r50_dc5_1x_c04_needle_MOUNT.py` | Complete MOUNT with UMP, TD-RPN, and AFA |
| `r50_dc5_1x_c04_needle_MOUNT_fold{1,2,3,4,5}.py` | Cross-validation splits |

### Ablation Study Models

| File Name | Description |
|-----------|-------------|
| `r50_dc5_1x_c04_needle_UMP_TDRPN.py` | UMP + TD-RPN (no AFA) |
| `r50_dc5_1x_c04_needle_tip_rpn_AFA.py` | TD-RPN + AFA (no UMP) |
| `r50_dc5_1x_c04_needle_tip_rpn.py` | TD-RPN only |
| `r50_dc5_1x_c04_needle_AFA.py` | AFA (no UMP/TD-RPN) |
| `r50_dc5_1x_c04_needle_flow.py` | UMP (flow) only |
| `r50_dc5_1x_c04_needle_tip.py` | Tip detection only |
| `r50_dc5_1x_c04_needle_only_tip.py` | Only tip detection |
| `r50_dc5_1x_c04_needle_MOUNT_flops.py` | MOUNT with FLOPs counting |
| `r50_dc5_1x_c04_needle_MOUNT_half.py` | MOUNT with half time resolution (1/2 down-sampling)|
| `r50_dc5_1x_c04_needle_MOUNT_RFA.py` | MOUNT with RFA |

### Baseline Models

| File Name | Description |
|-----------|-------------|
| `masktrack_rcnn_r50_fpn_12e_needle.py` | MaskTrack RCNN c04 |
| `bytetrack_yolox_x_needle.py` | ByteTrack c04 |
| `selsa_faster_rcnn_r50_dc5_1x_tip.py` | SELSA c04 |

## Key Configuration Parameters

### Model Configuration

```python
model = dict(
    type='SELSAFlow',  # MOUNT model type
    detector=dict(
        rpn_head=dict(
            type='RPNNeedleHead',  # TD-RPN
            tip_gene_config=dict(
                type='TipGenerator',
                dires=[0.16666666666666666, 0.25, ...],  # Polar directions
            ),
            loss_tip=dict(type='L1Loss', loss_weight=1.0),  # Tip loss
        ),
        roi_head=dict(
            type='SelsaRoIHeadBoth',
            aggregator=dict(
                type='SelsaIoUAggregator',  # AFA aggregator
                filter_component=['IoU', 'slope'],  # Filtering criteria
                aggre_component=['length', 'mask'],  # Aggregation features
                aggre_factor=[0.5, 0.5],  # Aggregation weights
            ),
        ),
    ),
    flow_generator=dict(...),  # UMP: Optical flow generator
    flow_encoder=dict(...),  # UMP: Flow feature encoder
    flow_neck=dict(...),  # UMP: Flow feature neck
)
```

### Dataset Configuration

```python
dataset_type = 'NeedleVideoDataset'
data_root = '/path/to/UltrasoundNeedle/dataset_version/'
data = dict(
    train=dict(
        ref_img_sampler=dict(
            num_ref_imgs=6,  # Number of reference frames
            frame_range=[-10, 10],  # Frame search range
            method='bilateral_uniform',  # Sampling method
        ),
    ),
)
```

### Training Configuration

```python
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='step', warmup='linear', step=[8, 11])
total_epochs = 12
```

## Usage

### Training

```bash
# Train with full MOUNT model
python tools/train.py configs/liver_needle/r50_dc5_1x_c04_needle_MOUNT.py

# Train with cross-validation split
python tools/train.py configs/liver_needle/r50_dc5_1x_c04_needle_MOUNT_fold1.py
```

### Testing

```bash
python tools/test.py configs/liver_needle/r50_dc5_1x_c04_needle_MOUNT.py \
    work_dirs/your_checkpoint.pth \
    --eval bbox segm
```

## Customization

### Changing Reference Frames

Modify `num_ref_imgs` and `frame_range` in the data config:

```python
ref_img_sampler=dict(
    num_ref_imgs=4,  # Use 4 reference frames instead of 6
    frame_range=[-8, 8],  # Narrower frame range
)
```

### Adjusting AFA Weights

Modify `aggre_factor` in the aggregator config:

```python
aggregator=dict(
    aggre_factor=[0.7, 0.3],  # More weight on length, less on mask
)
```

### Changing TD-RPN Anchor Directions

Modify `dires` in the tip generator config:

```python
tip_gene_config=dict(
    dires=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0],
)
```

## Notes

- All configurations assume the dataset is located at `/path/to/UltrasoundNeedle/dataset_version/`
- Modify `data_root` to match your dataset path
- The `fold` parameter controls which cross-validation split is used
- Pre-trained flow generator model is required for UMP module
