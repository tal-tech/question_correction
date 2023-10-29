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

from formula_pycode.formula_ocr import formula_ocr

# 设置使用的gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


# --------------------------------Formula Reg---------------------------------


def model_init(encoder_path,
               decoder_path,
               dict_path,
               max_batchsize,
               gpu=["0"]):
    assert (encoder_path != "" and decoder_path != "" and dict_path != "")
    return formula_ocr(encoder_path, decoder_path, dict_path, max_batchsize,
                       gpu)


def formula_preds(ocr_engine, base64data):
    try:
        time_s = time.time()
        img_mats = []
        outjs = {}
        data = []
        for base64_code in base64data:
            image = base64_to_image(base64_code)
            image = np.array(image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_mats.append(image)

        feature_total, label_total, imgidx_total, im, ratio = ocr_engine.receive(
            img_mats)
        res, prob = ocr_engine.predict(feature_total,
                                       label_total, imgidx_total, im,
                                       len(imgidx_total), ratio)

        time_e = time.time()
        print("inter time:", time_e - time_s)

        for result, score in zip(res, prob):
            result = result.replace("begin ", "")
            result = result.replace(" end", "")
            data.append({"result":result, "confidence":score})
        outjs["data"] = data
        return json.dumps(outjs)
    except:
        print("error")
        return "{}"


if __name__ == '__main__':
    # ---------------------------------------Formula test--------------------------------------
    ocr_engine = model_init("./formula_pycode/encoder_0728.pt",
                            "./formula_pycode/decoder_0728.pt",
                            "./formula_pycode/dict_test_240.txt", 5)

    ff = open('./img/test_formula.jpg', 'rb')
    base64_data = base64.b64encode(ff.read())
    img_str = base64_data.decode()
    ll = formula_preds(ocr_engine, [img_str])
    print(ll)
