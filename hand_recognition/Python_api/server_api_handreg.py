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
from reg_pycode.crnn_resnet_5780_gru_fpn import REG
import reg_pycode.params_test as params_test

# 设置使用的gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def resize(img):
    image = Image.fromarray(img)
    w, h = image.size
    new_h = 64
    new_w = int(new_h * w / h)

    if new_w < 896:
        temp = image.resize((new_w, new_h), Image.ANTIALIAS)
        image = Image.new("L", (896, 64), (255))
        image.paste(temp, ((896 - new_w) // 2, 0))

    else:
        image = image.resize((896, 64), Image.ANTIALIAS)

    image = np.asarray(image)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(0.588).div_(0.193)
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    image = image.to("cuda")
    return image


def decode(ls, crnn_model):
    outstr = ""
    pos = []
    ls = np.insert(ls, 0, 0)
    ls = np.append(ls, 0)

    for i in range(1, len(ls)):
        if ls[i] != ls[i - 1]:
            outstr += crnn_model.zidian[ls[i]]
            pos.append(i - 1)
    pos = pos[::2]
    return outstr, pos


def RegModel_Init(modelfile, zidianfile, max_batchsize):
    crnn_model = REG(params_test.Height, params_test.nc, params_test.nclass,
                     params_test.nh, zidianfile, max_batchsize)
    print('loading pretrained model from %s' % modelfile)
    crnn_model.load_weights(modelfile)
    crnn_model.CRNN = crnn_model.CRNN.to("cuda")
    for p in crnn_model.CRNN.parameters():
        p.requires_grad = False
    crnn_model.CRNN.eval()
    return crnn_model


def test_batch_base64(crnn_model, imagelist):
    start = time.time()
    imglist = []
    resultlist = []
    batchsize = crnn_model.max_batchsize
    group = len(imagelist) // batchsize
    if len(imagelist) % batchsize > 0:
        group += 1
    for n in range(group):
        imglist = imagelist[n * batchsize:min((n + 1) *
                                              batchsize, len(imagelist))]
        batch = len(imglist)

        imgs = torch.cat([x for x in imglist], 0)
        preds_tabel = crnn_model.CRNN(imgs)

        pro, preds = preds_tabel.max(2)
        pro = exp(pro.cpu().numpy())
        predstemp = preds.cpu().numpy()

        for j in range(preds.shape[0]):
            prob = [pro[j][x] for x in range(227) if predstemp[j][x] != 0]
            temp_result, temp_pos = decode(predstemp[j], crnn_model)
            if np.mean(prob) <= 1 and np.mean(prob) >= 0:
                resultlist.append({
                    "result": temp_result,
                    "confidence": np.mean(prob).item(),
                    "pos": temp_pos
                })

            else:
                resultlist.append({
                    "result": "",
                    "confidence": 0.1,
                    "pos": [
                        227,
                    ]
                })

    end = time.time()

    print("python run time to reg batch:", end - start)
    return resultlist


def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return img


def mainProcess(crnn_model, imagelist):
    js = {}
    reglist = []
    for n in imagelist:
        image = base64_to_image(n)
        image = np.array(image, dtype=np.uint8)
        image = resize(image)
        reglist.append(image)
    js["data"] = test_batch_base64(crnn_model, reglist)
    return json.dumps(js, ensure_ascii=False)


def init_model():
    return RegModel_Init(params_test.crnn, params_test.zidian, 10)


if __name__ == '__main__':

    # ---------------------------------------Reg test--------------------------------------

    crnn_model = RegModel_Init(params_test.crnn, params_test.zidian, 10)
    ff = open('./img/test_reg.jpg', 'rb')
    base64_data = base64.b64encode(ff.read())
    img_str = base64_data.decode()
    ll = test_batch_base64(crnn_model, [img_str])
    print(ll)
