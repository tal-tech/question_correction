#! -*- coding:utf-8 -*- 

from threading import Lock
from Python_api.server_api_handreg import init_model, test_batch_base64

class HandRegDet(object):
    def __init__(self):
        self.mode = init_model()
        self.lock = Lock()
    def process(self, images):
        with self.lock:
            return test_batch_base64(self.mode, images)
