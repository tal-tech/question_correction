import sys
import numpy as np
class ConfigPath:
    TEST_GPU_ID=['0']
    THREAD_NUM=3
    # CTPN_COM_LINE='/data/ouyangshizhuang/ExpReg/text-detection-ctpn/ctpn'
    # DETECTION_TXT_PATH='/data/www/ocr/ocrmobi/image/py_out_txt'
    # FORMULA_CTPN_PATH='/data/ouyangshizhuang/ExpReg/text-detection-ctpn'
    # FORMULA_CTPN_PY_PATH='/data/ouyangshizhuang/ExpReg/text-detection-ctpn/ctpn/formula_ctpn.py'
    image_dir='/data/www/ocr/ocrpaper.xueersi.com/image/separate/py_input_img/'
    image_flag_dir='/data/www/ocr/ocrpaper.xueersi.com/image/separate/py_input_flag/'
    txt_dir='/data/www/ocr/ocrpaper.xueersi.com/image/separate/py_out_txt/'
    txt_flag_dir='/data/www/ocr/ocrpaper.xueersi.com/image/separate/py_out_flag/'
    image_save_dir='/data/www/ocr/ocrpaper.xueersi.com/image/separate/py_img_save_dir/'
    middle_image_dir='/data/www/ocr/ocrpaper.xueersi.com/image/separate/middle'
    if_save_middle=True

