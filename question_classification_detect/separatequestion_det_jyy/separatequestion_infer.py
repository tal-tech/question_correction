#coding:utf-8
import json
import numpy as np
import time
import cv2
import os
import os.path as osp
import base64
import torch
import torch.optim as optim
import torchvision
# from lenet_model import LeNet2

import argparse

import cv2
import time
import os
import os.path as osp
import sys

#指定用哪块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)
sys.path.append(os.path.join(rootPath[0],rootPath[1]))

#exit(0)
from mmdet.apis import inference_detector, init_detector, show_result
import json


def parse_args():
    conf_path = os.path.join(rootPath[0],rootPath[1], "configs/separate_questions_cascade_mask_rcnn_r50_fpn_1x.py")
    model_path = os.path.join(rootPath[0],rootPath[1], "epoch_12.pth")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default=conf_path, help='test config file path')
    parser.add_argument('--checkpoint',type=str, default=model_path, help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.13, help='bbox score threshold')
    args = parser.parse_args()
    return args



def img_process(img):
    scale = 1
    w = img.shape[1]
    h = img.shape[0]
    short_side = min(w, h)
    if(short_side > 800):
        scale = short_side / 800
        img = cv2.resize(img, (int(w/scale), int(h/scale)))
    scale_w = scale
    scale_h = scale
    #elif(short_side < 100):
    #    scale = short_side / 100
    #    img = cv2.resize(img, (int(w/scale/1.1), int(h/scale)))
    if(img.shape[0] / img.shape[1] > 19):
        scale_h = scale_h * img.shape[0] /(img.shape[1]*19)
        img = cv2.resize(img, (int(w/scale_w), int(h/scale_h)))
    elif(img.shape[1] / img.shape[0] > 19):
        scale_w = scale_w * img.shape[1] / (img.shape[0]*19)
        img = cv2.resize(img, (int(w/scale_w), int(h/scale_h)))

    return img, scale_w, scale_h
    
# args = parse_args()
model = init_detector(
os.path.join(rootPath[0],rootPath[1], "configs/separate_questions_cascade_mask_rcnn_r50_fpn_1x.py"),
    os.path.join(rootPath[0], rootPath[1], "epoch_12.pth"),
    device=torch.device('cuda', 0))

def process(img,im_name):
    '''
    :param img: 图片,图片名
    :return: 检测结果的json串
    '''
    img, scale_w, scale_h = img_process(img)
    result = inference_detector(model, img)
    data = show_result(
        scale_w, scale_h, im_name, img, result, model.CLASSES, score_thr=0.1, wait_time=1)
    return json.dumps(data)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read())
        base64_str = base64_data.decode("utf-8")
        return base64_str
   

def base64_to_image(base64_code):
    '''
    :param base64_code: base64
    :return: opencv格式的图片
    '''
    img_data = base64.b64decode(base64_code)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img

def main(base64_code, token):
    '''
    :param base64_code: base64_code
            token: token
    :return: json结果
    '''
    image=base64_to_image(base64_code)
    image=np.array(image, dtype=np.uint8)
    return process(image, token)

if __name__=="__main__":
    img_path = "./R.jpeg"  # 替换为实际图片路径
    base64_code = image_to_base64(img_path)
    # print(base64_code) 
    result = main(base64_code,"222")
    print(result)

