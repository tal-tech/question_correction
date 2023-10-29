# 题目批改
批改步骤如下：
1.ocr检测（ocr_common_detect）: 先对图片进行ocr检测
2.手写文字识别（hand_recognition）：然后对ocr检测的手写文字部分进行识别
3.印刷公式识别（print_formula_recognition_high）：同样对印刷文字部分进行识别
4.题目检测（question_classification_detect）：将文字识别的部分与印刷文字识别部分进行综合，便可以对题目进行检测
根据以上步骤便可以一步一步将题目进行批改。
