#! -*- coding:utf-8 -*-

import numpy as np
import cv2
import base64
from app.resources.handreg import handreg_process_task

def process_task(image_base64):
    image_data = ''
    img = None
    if len(image_base64) != 0:
            image_data = base64.b64decode(image_base64) 
    img_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    x = img.shape[0]#h
    y = img.shape[1]#w
    print('*************')
    if min(x, y) < 10 or max(x, y) > 4096 or (min(x, y) < 16 and (max(x, y) * 1.0 / min(x, y)) > 20):
        return 300042004, None
    ret = handreg_process_task(function, image_base64)
    if ret == -1:
        return -1
    ret = parse_result(ret)
    print(ret)

def parse_result(ret):
    result = {} 
    if 'rotate' in ret.keys():
        result['rotate'] = ret['rotate']
    if 'data' in ret.keys():
        if 'rotate' in ret['data'].keys():
            result['rotate'] = ret['data']['rotate']
        if 'handText' in ret['data'].keys():
            result['line_results'] = ret['data']['handText']
        if 'images' in ret['data'].keys():
            result['images'] = ret['data']['images']
        if 'sheets' in ret['data'].keys():
            result['sheets'] = ret['data']['sheets']
    return result 
    
class ImageService(Resource):
    def __init__(self):
        self._errors = {
            20000:'success',
            300042001:'request body inval',
            300042002:'function format is invalid',
            300042003:'image decode error',
            300042004:'image size inval',
            300042005:'algorithm error',
            300042006:'image download error',
            300042007:'image inval',
            300042008:'image base64 inval', 
            }
    def get(self):
        return 'The service is evaluation service.'

