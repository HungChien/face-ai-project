_base_ = './widerface_retinanet_r50_fpn.py'

default_scope = 'mmdet'

model = dict(
    backbone=dict(init_cfg=None),
)

img_scale = (320, 320)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        pipeline=train_pipeline,
    ),
)

val_dataloader = None
val_evaluator = None
val_cfg = dict(_delete_=True, type='ValLoop')
val_cfg = None

test_dataloader = None
test_evaluator = None
test_cfg = dict(_delete_=True, type='TestLoop')
test_cfg = None

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=2, val_interval=1000)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=1),
)

log_processor = dict(type='LogProcessor', window_size=1, by_epoch=False)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
)

device = 'cpu'
work_dir = 'outputs/mmdetection_widerface/retinanet_r50_fpn_smoke'


