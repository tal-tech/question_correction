#! -*- coding:utf-8 -*- 

from threading import Lock
from Python_api.server_api_mathreg import model_init, formula_preds

class MathReg(object):
    def __init__(self):
        self.check_point1 = 'Python_api/formula_pycode/encoder_0728.pt'
        self.check_point2 = 'Python_api/formula_pycode/decoder_0728.pt'
        self.check_point3 = 'Python_api/formula_pycode/dict_test_240.txt'
        self.mode = model_init(self.check_point1, self.check_point2, self.check_point3, 5)
        self.lock = Lock()
    def process(self, image):
        with self.lock:
            return formula_preds(self.mode, image)