#! -*- coding:utf-8 -*- 

from handreg import HandReg, strVector
from app.resources.server_api_det import DetReg
from app.resources.server_api_rot import Rot
from app.resources.server_api_handreg import HandRegDet
from app.resources.server_api_mathreg import MathReg

import json

detReg = DetReg() 
rotReg = Rot() 
handRegDet = HandRegDet() 
mathReg = MathReg() 
def handreg_process_task(function, images_data):
    is_formula = False if function == 1 else True
    try:
        h = HandReg(is_formula)
        out_img = ''
        ret = h.pretreatmentOfRotate(images_data, out_img)
        if len(ret) != 2 or ret[0] != 0:
            raise Exception('pretreatmentOfRotate parse error.')
        out_img = str(ret[1])
        if out_img == None or len(out_img) == 0:
            raise Exception('out_img zero error.')
        out_img = rotReg.process(out_img)
        if out_img == None or len(out_img) == 0:
            raise Exception('rotReg parse error.')
        detectMatStr = ''
        ret = h.pretreatmentOfDetect(out_img, detectMatStr) 
        if len(ret) != 2 or ret[0] != 0:
            raise Exception('pretreatmentOfDetect parse error.')
        detectMatStr = ret[1] 
        if detectMatStr == None or len(detectMatStr) == 0:
            raise Exception('detectmatstr zero error.')
        outputOfDetect = detReg.process(detectMatStr)
        if outputOfDetect == None or len(outputOfDetect) == 0:
            raise Exception('detreg parse error.')
        h.detectStruct(outputOfDetect) 
        allFormulaMats = strVector()
        handMats = strVector()
        h.beforeReg(allFormulaMats, handMats)
        def parse_formula_task():
            if function == 1 or allFormulaMats.size() == 0:
                return None 
            in_mats = []
            for mat in allFormulaMats.iterator():
                in_mats.append(mat)
            out_mats = mathReg.process(in_mats)
            if out_mats == None:
                raise Exception('mathReg.process parse error.')
            h.afterFormulaReg(out_mats)
            return None 
        def parse_text_task():
            if function == 2 or handMats.size() == 0:
                return None
            in_text = [] 
            for text in handMats.iterator():
                in_text.append(text)
            out_text = handRegDet.process(in_text)
            if out_text == None:
                raise Exception('handRegDet.process parse error.')
            h.afterTextReg(out_text) 
            return None 
        
        parse_formula_task()
        parse_text_task() 
        h.afterReg()
        h.operateColumn()
        ret = h.combineJson()
        result = json.loads(ret)
        if result['code'] != 1:
            return -1 
        return result
    except Exception as e:
        raise e 
