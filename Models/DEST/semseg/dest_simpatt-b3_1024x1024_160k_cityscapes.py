# _base_ = [
#     '../_base_/models/dest_simplemit-b0.py',
#     '../_base_/datasets/cityscapes_1024x1024.py',
#     '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_160k.py'
# ]


_base_ = [
    '../_base_/models/dest_simpatt-b0.py',
    '../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]



evaluation = dict(interval=1000, metric='mIoU')
data = dict(samples_per_gpu=4)
checkpoint_config = dict(by_epoch=False, interval=20000)


# optimizer

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=1.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

embed_dims = [64, 128, 250, 320]
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None, 
    backbone=dict(
        type='SimplifiedTransformer',
        img_size=(1024, 1024),
        in_chans=3,
        num_classes=19,
        embed_dims=embed_dims,
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        depths=[3, 6, 8, 3],
        sr_ratios=[8, 4, 2, 1]),
    decode_head=dict(
        type='DestHead',
        in_channels=embed_dims,
        in_index=[0, 1, 2, 3],
        channels=512, #decoder param
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))





