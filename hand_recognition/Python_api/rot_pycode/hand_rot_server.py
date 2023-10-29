import flask
from flask import request, Flask
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
import models.mobilenetv3 as mobilenet
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# 设置使用的gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False

# ------------------------------Rot-------------------------------


def init_model(checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """

    cudnn.benchmark = True

    crnn = mobilenet.MobileNetV3(n_class=4, input_size=512)
    print('loading pretrained model from %s' % checkpoint)
    crnn.load_state_dict(torch.load(checkpoint))

    crnn.to(device)
    crnn.eval()
    return crnn


def processRot(crnn, image, token):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for p in crnn.parameters():
        p.requires_grad = False

    image = Image.fromarray(image)
    out = image
    image = np.asarray(out) / 255.
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)
    preds_tabel = crnn(image)
    preds_tabel = F.softmax(torch.exp(preds_tabel), dim=1)

    pro, preds = preds_tabel.max(1)
    return int(preds[0].cpu().numpy())


def base64_to_image(base64_code, name):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


@app.route("/", methods=['POST'])
def get_frame():
    res = request.get_data()
    res_js = json.loads(res)
    js = {}
    start = time.perf_counter()

    name = res_js["token"]
    image = base64_to_image(res_js["image"], name)
    image = np.array(image, dtype=np.uint8)

    js["data"] = processRot(model, image, name)
    end = time.perf_counter()
    print("time:", end - start)
    js["code"] = "000000"
    print(js)
    return flask.jsonify(js)


if __name__ == '__main__':

    # ---------------------------------------Rot test--------------------------------------
    checkpoint = './checkpoint/crnn_Rec_done_73_993.pth'
    model = init_model(checkpoint)

    app.run("127.0.0.1", port=10182)
