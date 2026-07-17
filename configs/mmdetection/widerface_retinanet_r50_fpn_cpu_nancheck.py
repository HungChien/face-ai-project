_base_ = './widerface_retinanet_r50_fpn_cpu_full.py'

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=600, val_interval=1000)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=600, max_keep_ckpts=1),
)

work_dir = 'outputs/mmdetection_widerface/retinanet_r50_fpn_cpu_nancheck'
