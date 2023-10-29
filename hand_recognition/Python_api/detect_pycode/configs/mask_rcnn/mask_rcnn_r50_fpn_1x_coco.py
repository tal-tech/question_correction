_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_gn_ws_9_22.py',
    #'../_base_/models/mask_rcnn_r50_fpn_gn_ws.py',
    #'../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
