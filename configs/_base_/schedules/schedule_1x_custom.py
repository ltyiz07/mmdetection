optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None, type='OptimizerHook')
lr_config = dict(
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11],
    type='StepLrUpdaterHook',
    policy='step')
runner = dict(type='EpochBasedRunner', max_epochs=12)