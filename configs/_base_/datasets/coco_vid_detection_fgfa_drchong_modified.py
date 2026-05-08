dataset_type = 'CocoVideoDataset'
classes = ('nodule', )
data_root = '/srv/fenster/people/Ningtao/Dataset/US3D/DrChongModified/data/'
data_set_root = '/srv/fenster/people/Ningtao/Dataset/US3D/DrChongModified/dataset/Fold0/'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(512, 512), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(512, 512), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_set_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        ref_img_sampler=dict(
                num_ref_imgs=4,
                frame_range=12,
                filter_key_img=True,
                method='bilateral_uniform'
        )
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_set_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        ref_img_sampler=dict(
            num_ref_imgs=30,
            frame_range=[-15, 15],
            stride=1,
            method='test_with_fix_stride'
        )
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_set_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        ref_img_sampler=dict(
            num_ref_imgs=30,
            frame_range=[-15, 15],
            stride=1,
            method='test_with_fix_stride'
        )
    )
)