# -*- coding: utf-8 -*-
from app.resources import r_image_service as h
import base64

if __name__ == '__main__':
    with open('./example.webp','rb') as f:
        ls_f = base64.b64encode(f.read())
        image_base64 = str(base64.b64encode(ls_f),encoding='utf-8')
        f.close()
        print(image_base64)
        h.process_task(image_base64)
