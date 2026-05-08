_base_ = [
    '../_base_/models/faster_rcnn_r50_dc5.py',
    '../_base_/datasets/coco_vid_detection_fgfa_drchong_modified.py',
    '../_base_/default_runtime.py'
]
custom_imports = dict(imports=['mmtrack.core.hook.mask_selsa_hook'], allow_failed_imports=False)
desp='after poly updated the annotation March 18'
model = dict(
    type='SELSAFlow',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHeadBoth',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='DecupleBBoxHead',
                in_channels=512,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                num_shared_fcs=2),
            aggregator=dict(
                type='SelsaIoUAggregator',
                in_channels=1024,
                num_attention_blocks=16,
                filter_component=['IoU', 'slope'],
                aggre_component=['length', 'mask'],
                aggre_factor=[0.5, 0.5],
                aux_factor=0.2,
                mode='multiply'),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=512,
                conv_out_channels=512,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        rpn_head=dict(
            type='RPNNeedleHead',
            in_channels=512,
            feat_channels=512,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[2, 4, 8, 16, 32],
                ratios=[0.5, 1.0, 2.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(_delete_=True, type='L1Loss', loss_weight=1.0),
            loss_tip=dict(type='L1Loss', loss_weight=1.0),
            tip_gene_config=dict(
                type='TipGenerator',
                strides=[16],
                scales=[2, 4, 8, 16, 32],
                dires=[
                    0.16666666666666666, 0.25, 0.3333333333333333, 0.5,
                    0.6666666666666666, 0.75, 0.8333333333333334, 1,
                    1.1666666666666667, 1.25, 1.3333333333333333, 1.5, 1.25, 1.75,
                    1.8333333333333333, 2
                ]),
            num_convs=1
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
            debug=False,
            save_at_train=False,)),
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
                mask_thr_binary=0.5,
                save_at_test=False
            )
        ),
    ),    
    flow_generator=dict(
        model=dict(
            upsample=True,
            n_frames=2,
            reduce_dense=True
        ),
        pretrained_model=r'/srv/fenster/people/Ningtao/Project/USVideo/ARFlow/outputs/checkpoints/230613/184057/Needle_Flow_model_best.pth.tar',
        test_shape=[512, 576]
    ),
    flow_encoder=dict(
            type='ResNet',
            in_channels = 2,
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
    flow_neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        out_channels=512,
        kernel_size=3),
)

# dataset
classes = ('Needle', )
dataset_type = 'NeedleVideoDataset'
data_root = '/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/'
dataset_version = 'custom'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_mask=True,
        with_track=True),
    dict(
        type='SeqResize',
        share_params=True,
        img_scale=(600, 525),
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_slopes', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(600, 525), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]

fold = 5
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + f'DataSet/coco_valid_0318_no_occu/{str(fold)}/train.json',
        img_prefix=data_root + 'Data',
        pipeline=train_pipeline,
        ref_img_sampler=dict(
            num_ref_imgs=6,
            frame_range=[-10, 10],
            filter_key_img=True,
            method='bilateral_uniform'
        )
    ),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + f'DataSet/coco_valid_0318_no_occu/{str(fold)}/test.json',
        img_prefix=data_root + 'Data',
        pipeline=test_pipeline,
        ref_img_sampler=dict(
            num_ref_imgs=6,
            frame_range=[-10, 10],
            method='test_with_adaptive_stride'),
    ),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + f'DataSet/coco_valid_0318_no_occu/{str(fold)}/test.json',
        img_prefix=data_root + 'Data',
        pipeline=test_pipeline,
        ref_img_sampler=dict(
            num_ref_imgs=6,
            frame_range=[-10, 10],
            method='test_with_adaptive_stride'),
        
        )
    )

# optimizer
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
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
total_epochs = 12
evaluation = dict(metric=['bbox', 'segm'], interval=1)
resume_from = '/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020_AFA_fold5/epoch_10.pth'

