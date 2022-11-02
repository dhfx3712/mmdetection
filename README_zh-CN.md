(test_openmmlab) admin@bogon mmdetection % python tools/train.py configs/yolo/rr_yolov5_416_coco.py





config/yolov4
base_yolo_head.py
yolo_head : 1 ,pre_map : torch.Size([1, 75, 9, 13])
yolo_head : 1 ,pre_map : torch.Size([1, 75, 18, 26])
yolo_head : 1 ,pre_map : torch.Size([1, 75, 36, 52])
yolo_head : 1 ,pre_map : torch.Size([1, 75, 9, 13])
yolo_head : 1 ,pre_map : torch.Size([1, 75, 18, 26])
yolo_head : 1 ,pre_map : torch.Size([1, 75, 36, 52])



yolo5 
rr_yolov5_head.py类别固定80个类










fasterfpn 结构
demo/MMDet_Tutorial.py提取demo/test_train.py 
(test_openmmlab) admin@bogon mmdetection % python demo/test_train.py
Config:
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
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
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'KittiTinyDataset'
data_root = '/Users/admin/data/kitti_tiny/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='KittiTinyDataset',
        ann_file='train.txt',
        img_prefix='training/image_2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        data_root='/Users/admin/data/kitti_tiny/'),
    val=dict(
        type='KittiTinyDataset',
        ann_file='val.txt',
        img_prefix='training/image_2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root='/Users/admin/data/kitti_tiny/'),
    test=dict(
        type='KittiTinyDataset',
        ann_file='train.txt',
        img_prefix='training/image_2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root='/Users/admin/data/kitti_tiny/'))
evaluation = dict(interval=12, metric='mAP')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=12)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/Users/admin/Downloads/models/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=2)
work_dir = './tutorial_exps'
seed = 0
gpu_ids = range(0, 1)
device = 'cpu'

demo/test_train.py:55: DeprecationWarning: `np.long` is a deprecated alias for `np.compat.long`. To silence this warning, use `np.compat.long` by itself. In the likely event your code does not need to work on Python 2 you can use the builtin `int` for which `np.compat.long` is itself an alias. Doing this will not modify any behaviour and is safe. When replacing `np.long`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  labels=np.array(gt_labels, dtype=np.long),
demo/test_train.py:58: DeprecationWarning: `np.long` is a deprecated alias for `np.compat.long`. To silence this warning, use `np.compat.long` by itself. In the likely event your code does not need to work on Python 2 you can use the builtin `int` for which `np.compat.long` is itself an alias. Doing this will not modify any behaviour and is safe. When replacing `np.long`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  labels_ignore=np.array(gt_labels_ignore, dtype=np.long))
/Users/admin/data/test_project/mmdetection/mmdet/datasets/custom.py:180: UserWarning: CustomDataset does not support filtering empty gt images.
  'CustomDataset does not support filtering empty gt images.')
train_detector : {'samples_per_gpu': 2, 'workers_per_gpu': 2, 'num_gpus': 1, 'dist': False, 'seed': 0, 'runner_type': 'EpochBasedRunner', 'persistent_workers': False}
2022-11-01 11:26:16,632 - mmdet - INFO - Automatic scaling of learning rate (LR) has been disabled.
2022-11-01 11:26:16,774 - mmdet - INFO - load checkpoint from local path: /Users/admin/Downloads/models/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth
2022-11-01 11:26:17,034 - mmdet - WARNING - The model and loaded state dict do not match exactly

size mismatch for roi_head.bbox_head.fc_cls.weight: copying a param with shape torch.Size([81, 1024]) from checkpoint, the shape in current model is torch.Size([4, 1024]).
size mismatch for roi_head.bbox_head.fc_cls.bias: copying a param with shape torch.Size([81]) from checkpoint, the shape in current model is torch.Size([4]).
size mismatch for roi_head.bbox_head.fc_reg.weight: copying a param with shape torch.Size([320, 1024]) from checkpoint, the shape in current model is torch.Size([12, 1024]).
size mismatch for roi_head.bbox_head.fc_reg.bias: copying a param with shape torch.Size([320]) from checkpoint, the shape in current model is torch.Size([12]).
2022-11-01 11:26:17,058 - mmdet - INFO - Start running, host: admin@bogon, work_dir: /Users/admin/data/test_project/mmdetection/tutorial_exps
2022-11-01 11:26:17,059 - mmdet - INFO - Hooks will be executed in the following order:
before_run:





two_stage_backbone_neck : 5,
rpn_head : 2 , img_metas : [{'filename': '/Users/admin/data/kitti_tiny/training/image_2/000003.jpeg', 'ori_filename': '000003.jpeg', 'ori_shape': (375, 1242, 3), 'img_shape': (402, 1333, 3), 'pad_shape': (416, 1344, 3), 'scale_factor': array([1.0732689, 1.072    , 1.0732689, 1.072    ], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}, {'filename': '/Users/admin/data/kitti_tiny/training/image_2/000000.jpeg', 'ori_filename': '000000.jpeg', 'ori_shape': (370, 1224, 3), 'img_shape': (403, 1333, 3), 'pad_shape': (416, 1344, 3), 'scale_factor': array([1.0890523, 1.0891892, 1.0890523, 1.0891892], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}]
max_iou_assigner_input : torch.Size([139671, 4]) torch.Size([1, 4]) None None iof_thr -1
max_iou_assigner : iou torch.Size([1, 139671])  get_bb torch.Size([1, 4])  bb torch.Size([139671, 4])
max_iou_assigner_paramma : {'pos_iou_thr': 0.7, 'neg_iou_thr': 0.3, 'min_pos_iou': 0.3, 'gt_max_assign_all': True, 'ignore_iof_thr': -1, 'ignore_wrt_candidates': True, 'gpu_assign_thr': -1, 'match_low_quality': True, 'iou_calculator': BboxOverlaps2D(scale=1.0, dtype=None)}
max_iou_assigner_input : torch.Size([139671, 4]) torch.Size([1, 4]) None None iof_thr -1
max_iou_assigner : iou torch.Size([1, 139671])  get_bb torch.Size([1, 4])  bb torch.Size([139671, 4])
max_iou_assigner_paramma : {'pos_iou_thr': 0.7, 'neg_iou_thr': 0.3, 'min_pos_iou': 0.3, 'gt_max_assign_all': True, 'ignore_iof_thr': -1, 'ignore_wrt_candidates': True, 'gpu_assign_thr': -1, 'match_low_quality': True, 'iou_calculator': BboxOverlaps2D(scale=1.0, dtype=None)}
max_iou_assigner_input : torch.Size([1000, 5]) torch.Size([1, 4]) None tensor([0]) iof_thr -1
max_iou_assigner : iou torch.Size([1, 1000])  get_bb torch.Size([1, 4])  bb torch.Size([1000, 5])
max_iou_assigner_paramma : {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0.5, 'gt_max_assign_all': True, 'ignore_iof_thr': -1, 'ignore_wrt_candidates': True, 'gpu_assign_thr': -1, 'match_low_quality': False, 'iou_calculator': BboxOverlaps2D(scale=1.0, dtype=None)}
roi_bbox_assigner batch-0, assign_result  <AssignResult(num_gts=1, gt_inds.shape=(1000,), max_overlaps.shape=(1000,), labels.shape=(1000,))>
max_iou_assigner_input : torch.Size([1000, 5]) torch.Size([1, 4]) None tensor([1]) iof_thr -1
max_iou_assigner : iou torch.Size([1, 1000])  get_bb torch.Size([1, 4])  bb torch.Size([1000, 5])
max_iou_assigner_paramma : {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0.5, 'gt_max_assign_all': True, 'ignore_iof_thr': -1, 'ignore_wrt_candidates': True, 'gpu_assign_thr': -1, 'match_low_quality': False, 'iou_calculator': BboxOverlaps2D(scale=1.0, dtype=None)}






mmdet/models/detectors/faster_rcnn.py
class FasterRCNN(TwoStageDetector) 传入两个head（rpnhead，roihead）
    



 img_metas : [{'filename': '/Users/admin/data/kitti_tiny/training/image_2/000003.jpeg', 'ori_filename': '000003.jpeg', 'ori_shape': (375, 1242, 3), 'img_shape': (402, 1333, 3), 'pad_shape': (416, 1344, 3), 'scale_factor': array([1.0732689, 1.072    , 1.0732689, 1.072    ], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}, {'filename': '/Users/admin/data/kitti_tiny/training/image_2/000000.jpeg', 'ori_filename': '000000.jpeg', 'ori_shape': (370, 1224, 3), 'img_shape': (403, 1333, 3), 'pad_shape': (416, 1344, 3), 'scale_factor': array([1.0890523, 1.0891892, 1.0890523, 1.0891892], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}]



rpnhead_loss
mmdet/models/detectors/two_stage.py
rpn_losses, proposal_list = self.rpn_head.forward_train(x,img_metas,gt_bboxes,gt_labels=None,gt_bboxes_ignore=gt_bboxes_ignore,proposal_cfg=proposal_cfg,**kwargs)
       mmdet/models/dense_heads/anchor_head.py
       __init__ 初始化网络结构 （self._init_layers()）
             mmdet/models/dense_heads/base_dense_head.py
             forward_train(self,x,img_metas,gt_bboxes,gt_labels=None,gt_bboxes_ignore=None,proposal_cfg=None,**kwargs):
                    /Users/admin/opt/anaconda3/envs/test_openmmlab/lib/python3.7/site-packages/torch/nn/modules/module.py
                    outs = self(x) (module类中的__call__模块)


roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

mmdet/models/roi_heads/standard_roi_head.py
forward_train(self,x,img_metas,proposal_list,gt_bboxes,gt_labels,gt_bboxes_ignore=None,gt_masks=None,**kwargs):
    mmdet/core/bbox/assigners/max_iou_assigner.py
    self.bbox_assigner = build_assigner(self.train_cfg.assigner)
    assign_result = self.bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],gt_labels[i])





bbox assigner策略
https://zhuanlan.zhihu.com/p/464404563
self.bbox_assigner = build_assigner(self.train_cfg.assigner)
MaxIoUAssigner



mmdet/core/bbox/assigners/max_iou_assigner.py
MaxIoUAssigner


     





mmdet/models/dense_heads/rpn_head.py
losses = super(RPNHead, self).loss(cls_scores,bbox_preds,gt_bboxes,None,img_metas,gt_bboxes_ignore=gt_bboxes_ignore)
        mmdet/models/dense_heads/anchor_head.py
        loss(self,cls_scores,bbox_preds,gt_bboxes,gt_labels,img_metas,gt_bboxes_ignore=None)






   