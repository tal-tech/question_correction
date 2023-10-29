from PIL import Image
import numpy as np
import random
import base64
import torch
from numpy.ma import exp, sin, cos
import json
import time
import sys
import cv2
import os

from .detect_pycode.mmdet.apis import inference_detector, init_detector, show_result_pyplot, show_result

# 设置使用的gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def make_border_height(img):
    height, width = img.shape[:2]
    border = 0
    if (height < width):
        border = int((1.33333 * width - height) / 2)
        img = cv2.copyMakeBorder(img,
                                 border,
                                 border,
                                 0,
                                 0,
                                 cv2.BORDER_CONSTANT,
                                 value=[128, 128, 128])
    return img, border


def process(model, img, im_name):
    write_txt_path = None

    img_border, border = make_border_height(img)

    w = img_border.shape[1]
    h = img_border.shape[0]
    short_side = min(w, h)
    long_side = max(w,h)

    scale_w = 1
    scale_h = 1
    new_img = img_border
    #if (short_side > 800):
    #   scale = short_side / 800
    #    new_img = cv2.resize(img_border, (int(w / scale), int(h / scale)),
    #                         interpolation=cv2.INTER_CUBIC)
    #    scale_w = img_border.shape[1] / new_img.shape[1]
    #    scale_h = img_border.shape[0] / new_img.shape[0]
    if(short_side > 800 or long_side > 1333):
        scale = max(short_side / 800, long_side/1333)
        new_img = cv2.resize(img_border, (int(w/scale), int(h/scale)), interpolation=cv2.INTER_CUBIC)
        scale_w = img_border.shape[1] / new_img.shape[1]
        scale_h = img_border.shape[0] / new_img.shape[0]

    result = inference_detector(model, new_img)
    det_img_shape = new_img.shape

    data = show_result(img,
                       det_img_shape,
                       result,
                       scale_w,
                       scale_h,
                       border,
                       write_txt_path=write_txt_path,
                       score_thr=0.3,
                       token=im_name,
                       is_server=True)

    return json.dumps(data)


def process_task(model, token, base64_code):
    '''
    :param base64_code: base64_code
            token: token
    :return: json结果
    '''
    image = base64_to_image(base64_code)
    image = np.array(image, dtype=np.uint8)
    # print("token:", token)
    # cv2.imwrite(os.path.join("/root/xzy/img/1013_testImg_rot/", token), image)
    return process(model, image, token)


if __name__ == '__main__':
    # -----------------------------------Detection test-----------------------------------
    config = "./detect_pycode/configs/fp16/mask_rcnn_r50_fpn_poly_1x_coco_gnws.py"
    checkpoint = "./detect_pycode/output/7_2_poly2mask_hangqie_addanchor/epoch_24.pth"

    model = init_detector(config, checkpoint)

    imgnn = "./img/test_det.jpg"
    f = open(imgnn, 'rb')
    base64_data = base64.b64encode(f.read())
    import time
    time.sleep(2)
    img_str = base64_data.decode()
    data = process_task(model, "2020", img_str)
    print(data)
