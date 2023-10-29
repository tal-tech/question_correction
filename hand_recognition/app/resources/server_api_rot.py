#! -*- coding:utf-8 -*- 

from Python_api.server_api_rot import mainRot, init_model
from threading import Lock

class Rot(object):
    def __init__(self):
        self.check_point = 'Python_api/rot_pycode/checkpoint/crnn_Rec_done_73_993.pth'
        self.mode = init_model(self.check_point)
        self.lock = Lock()
        
    def process(self, image):
        with self.lock:
            return mainRot(self.mode, image)
