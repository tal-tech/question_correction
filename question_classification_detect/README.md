\#1.安装anaconda: 运行文件夹中的安装脚本
sh Anaconda3-5.3.1-Linux-x86_64.sh
\#2.搭建虚拟环境：
\# 2.1搭建Python 3.6环境
/root/anaconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
/root/anaconda3/bin/conda config --set show_channel_urls yes
/root/anaconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
/root/anaconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
\#2.2 创建虚拟环境 mmdetection
conda create -n mmdetection python=3.6
conda activate mmdetection
\#2.3 安装所需库
cd separatequestion_det_jyy
pip install -r requirements1.txt
pip install -r requirements2.txt
pip install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-linux_x86_64.whl https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-manylinux1_x86_64.whl
\#2.4退出虚拟环境
conda deactivate
\#3.接口说明：
\# ./separatequestion_det_jyy/separatequestion_infer.py文件中,
def main(base64_code, token)

输入参数

| 参数名称    | 参数类型 | 参数描述           |
| ----------- | -------- | ------------------ |
| token       | string   | 图片名称，唯一标识 |
| Base64_code | string   | 图像base64数据     |

 返回参数和样例：

| 参数名称 | 参数类型 | 参数描述                 |
| -------- | -------- | ------------------------ |
| token    | string   | 图片名称，唯一标识       |
| result   | list[][] | 题目和题号位置信息及类别 |

 返回样例：

{"token:": "2020", "result": [[1, 27, 244, 27, 292, 355, 292, 355, 244], [1, 32, 392, 32, 424, 351, 424, 351, 392], [1, 32, 362, 32, 398, 349, 398, 349, 362], [1, 351, 417, 34, 418, 34, 449, 351, 449], [1, 23, 144, 23, 174, 225, 174, 225, 144], [1, 33, 496, 33, 554, 376, 554, 376, 496], [1, 32, 442, 32, 499, 354, 499, 354, 442], [3, 34, 428, 34, 436, 45, 436, 45, 428], [1, 25, 291, 25, 334, 288, 334, 288, 291], [3, 31, 347, 31, 355, 39, 355, 39, 347], [1, 23, 207, 23, 239, 322, 239, 322, 207], [3, 32, 372, 32, 379, 40, 379, 40, 372], [1, 30, 339, 30, 357, 349, 357, 349, 339], [2, 25, 77, 25, 85, 39, 85, 39, 77], [3, 33, 402, 33, 409, 42, 409, 42, 402], [1, 22, 73, 22, 92, 334, 92, 334, 73], [3, 35, 512, 35, 520, 46, 520, 46, 512], [3, 27, 186, 27, 194, 35, 194, 35, 186], [3, 34, 457, 34, 464, 48, 464, 48, 457], [1, 23, 89, 23, 141, 216, 141, 216, 89], [3, 27, 217, 27, 225, 37, 225, 37, 217], [3, 29, 255, 29, 263, 37, 263, 37, 255], [1, 25, 181, 25, 203, 357, 203, 357, 181], [3, 30, 310, 30, 318, 38, 318, 38, 310], [3, 26, 154, 26, 162, 36, 162, 36, 154], [3, 25, 107, 25, 116, 35, 116, 35, 107], [4, 32, 395, 32, 401, 41, 401, 41, 395], [3, 32, 395, 32, 401, 41, 401, 41, 395]]}

1555573807959.jpg: 测试图片
./separatequestion_det_jyy/separatequestion_infer.py中有输入上述图片的base64，返回结果的示例
需要测资源占用的话，循环调用main函数即可
