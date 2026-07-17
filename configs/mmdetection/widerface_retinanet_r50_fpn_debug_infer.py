_base_ = './widerface_retinanet_r50_fpn_debug.py'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='WIDERFaceDataset',
        data_root='data/raw/WIDERFace/',
        ann_file='val.txt',
        data_prefix=dict(img='WIDER_val'),
        test_mode=True,
        pipeline=test_pipeline),
)

val_dataloader = test_dataloader
test_evaluator = dict(_delete_=True, type='VOCMetric', metric='mAP', eval_mode='11points')
val_evaluator = test_evaluator
test_cfg = dict(_delete_=True, type='TestLoop')
val_cfg = dict(_delete_=True, type='ValLoop')
model = dict(test_cfg=dict(score_thr=0.0))
