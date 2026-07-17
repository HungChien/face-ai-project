_base_ = './widerface_retinanet_r50_fpn.py'

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
)

test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2),
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
)

device = 'cpu'
work_dir = 'outputs/mmdetection_widerface/retinanet_r50_fpn_cpu_full'