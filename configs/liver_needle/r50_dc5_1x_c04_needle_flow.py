_base_ = [
    '../_base_/models/faster_rcnn_r50_dc5.py',
    '../_base_/datasets/coco_vid_detection_fgfa_drchong_modified.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='SELSA',
    detector=dict(
        backbone=dict(
            in_channels=2,
        ),
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_head=dict(
                num_classes=1,
                type='SelsaBBoxHead',
                num_shared_fcs=2,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=512,
                featmap_strides=[16]),  
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=512,
                conv_out_channels=512,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
                )            
        ),
        train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=6000,
            max_per_img=600,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=6000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5)),
        ),   
    )

# dataset
classes = ('Needle', )
dataset_type = 'CocoVideoDataset'
data_root = '/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/'
dataset_version = 'custom'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFileMultiChannel', to_float32=True, color_type='grayscale'),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_mask=True,
        with_track=True),
    # dict(
    #     type='SeqResize',
    #     share_params=True,
    #     img_scale=(600, 525),
    #     keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    # dict(type='SeqNormalize', **img_norm_cfg),
    # dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFileMultiChannel', to_float32=True, color_type='grayscale'),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]

# test_pipeline = [
#     dict(type='LoadMultiImagesFromFileMultiChannel'),
#     # dict(type='SeqResize', img_scale=(800, 700)),
#     dict(type='SeqPad', size_divisor=32),
#     dict(
#         type='VideoCollect',
#         keys=['img'],
#         meta_keys=('num_left_ref_imgs', 'frame_stride')),
#     dict(type='ConcatVideoReferences'),
#     dict(type='MultiImagesToTensor', ref_prefix='ref'),
#     dict(type='ToList')
# ]

fold = 1
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        type=dataset_type,
        pipeline=train_pipeline,
        ann_file=data_root + f'DataSet/coco_flow/{str(fold)}/train.json',
        img_prefix=data_root + 'Data',
        ref_img_sampler=dict(
            num_ref_imgs=6,
            frame_range=[-10, 10],
            filter_key_img=True,
            method='bilateral_uniform')
        ),
    val=dict(
        classes=classes,
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file=data_root + f'DataSet/coco_flow/{str(fold)}/test.json',
        img_prefix=data_root + 'Data',
        ref_img_sampler=dict(
            num_ref_imgs=6,
            frame_range=[-10, 10],
            method='test_with_adaptive_stride')),
    test=dict(
        classes=classes,
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file=data_root + f'DataSet/coco_flow/{str(fold)}/test_p020_08.json',
        img_prefix=data_root + 'Data',
        ref_img_sampler=dict(
            num_ref_imgs=6,
            frame_range=[-10, 10],
            method='test_with_adaptive_stride'))
    )

# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
# runtime settings
total_epochs = 20
evaluation = dict(metric=['bbox'], interval=20)
