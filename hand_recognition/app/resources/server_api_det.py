#! -*- coding:utf-8 -*- 

from Python_api.server_api_det import init_detector, process_task
from threading import Lock

class DetReg(object):
    def __init__(self):
        self.config = 'Python_api/detect_pycode/configs/fp16/mask_rcnn_r50_fpn_poly_1x_coco_gnws.py'
        self.check_point = 'Python_api/detect_pycode/output/7_2_poly2mask_hangqie_addanchor/epoch_24.pth'
        self.mode = init_detector(self.config, self.check_point)
        self.lock = Lock()
    def process(self, image):
        with self.lock:
            return process_task(self.mode, image)