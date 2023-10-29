dataset_type = 'CocoDataset'
data_root = '/workspace/common_dataset/'
# data_root = '/workspace/DataSets/tabel_character/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        #ann_file=data_root + 'train_4_27.json',
        # ann_file=data_root + 'annotations/train.json',
        # ann_file=data_root + 'annotations/train_crop_border.json',
        # ann_file=data_root + 'annotations/train_crop_border_hangqie.json',
        # ann_file=data_root + 'annotations/train_8_5_add_xianshangyinshua.json',
        # ann_file=data_root + 'annotations/train_8_14.json',
        # ann_file=data_root + 'annotations/train_8_29.json',
        # ann_file=data_root + 'annotations/train_kousuan.json',
        # ann_file=data_root + 'annotations/train_add_excel_formulaset_mix.json',
        ann_file=data_root + 'annotations/train_paisou_9_26_2.json',
        # ann_file = '/workspace/common_dataset/d2_empirical_data/annotations/train.json',
        # ann_file=data_root + 'annotations/train_7_6.json',
        # ann_file=data_root + 'annotations/train_7_8.json',
        
        # img_prefix=data_root + 'all_img/',
        img_prefix=data_root + 'all_img_paisou/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'all_img/',
        # ann_file=data_root + 'annotations/test.json',
        # img_prefix=data_root + 'test_img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'all_img/',
        # ann_file=data_root + 'annotations/test.json',
        # img_prefix=data_root + 'test_img/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
