_base_ = './widerface_retinanet_r50_fpn_smoke.py'

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=20, val_interval=1000)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2),
)

work_dir = 'outputs/mmdetection_widerface/retinanet_r50_fpn_debug'
