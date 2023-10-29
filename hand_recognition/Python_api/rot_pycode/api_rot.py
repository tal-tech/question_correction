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
import rot_pycode.models.mobilenetv3 as mobilenet
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# 设置使用的gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


# ------------------------------Rot-------------------------------


def init_model(checkpoint=None, device='cuda:1'):
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
    outjs = {}
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
    outjs["data"]=int(preds[0].cpu().numpy())
    return json.dumps(outjs)


def mainRot(model, token, base64_code):
    image = base64_to_image(base64_code)
    image = np.array(image, dtype=np.uint8)
    return processRot(model, image, token)


if __name__ == '__main__':

    # ---------------------------------------Rot test--------------------------------------
    checkpoint = './rot_pycode/checkpoint/crnn_Rec_done_73_993.pth'
    model = init_model(checkpoint)

    imgn = './img/test_rot.jpg'
    f = open(imgn, 'rb')

    base64_data = base64.b64encode(f.read())
    img_str = base64_data.decode()

    data = mainRot(model, '2020', img_str)
    print(data)
