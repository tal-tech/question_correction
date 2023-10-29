import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import cv2
import os
import shapely
from shapely.geometry import Polygon,MultiPoint 

from scipy.spatial import distance as dist
import math
import itertools

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    print(a, b)
    print(zip(a, b))
    for a1, b1 in zip(a, b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down
 
 
# this function is confined to rectangle
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
 
 
def order_points_quadrangle(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left and bottom-left coordinate, use it as an
    # base vector to calculate the angles between the other two vectors
 
    vector_0 = np.array(bl-tl)
    vector_1 = np.array(rightMost[0]-tl)
    vector_2 = np.array(rightMost[1]-tl)
 
    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('formula_struct')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result



def merge_diff_class_rect_polygon(boxes, segms, np_classes):
    # np_classes = np.array(classes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    all_del_inds = []
    scores = boxes[:, 4]
    # scores = np.array([boxes_list[-1] for boxes_list in boxes])
    keep_index = []
    keep_classes = []
    # order = scores.argsort()[::-1]

    # print("order:", order)
    num = len(boxes)
    include_dict = {}
    order = np.arange(num)
    # print("order:", order)
    suppressed = np.zeros((num), dtype=np.int)
    inter_index_all = []



    for _i in range(num):
        i = order[_i]
        # print("i:", i)
        # print("_i:", _i)
        if suppressed[i] == 1:# 对于抑制的检测框直接跳过
            continue
        keep_index.append(i)# 保留当前框的索引

        segms1 = segms[i]
        segms1_arr=np.array(segms1).reshape(int(len(segms1)/2), 2)
        
        index = np.where(np_classes == -1)[0]
        # index_same_class = np.where(np_classes == np_classes[])
        # index = np.where(np_classes != np_classes[i])[0]
        xx1 = np.maximum(x1[i], x1[index])
        yy1 = np.maximum(y1[i], y1[index])
        xx2 = np.minimum(x2[i], x2[index])
        yy2 = np.minimum(y2[i], y2[index])
        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        valid_index = index[np.where(inter>0)]
        if(valid_index.shape[0] == 0):
            inter_index_all.append([])
            continue
        valid_index_list = valid_index.tolist()
        inter_index_all.append(valid_index_list)
        # print("valid_index_list:", valid_index_list)
        # exit(0)
        # segms1_arr = np.array([[3, 29], [3, 26], [452, 62], [452, 30]])

        segms1_poly = Polygon(segms1_arr)
        segms1_poly = segms1_poly.convex_hull
        
        area_segms1 = segms1_poly.area    #计算当前检测框面积
        for _j in range(_i + 1, num):   #对剩余的而进行遍历
            j = order[_j]
            if suppressed[i] == 1:
                continue
            if not j in valid_index_list:
                continue

            segms2 = segms[j]
            segms2_arr=np.array(segms2).reshape(int(len(segms2)/2), 2)
            segms2_poly = Polygon(segms2_arr)
            segms2_poly = segms2_poly.convex_hull
            area_segms2 = segms2_poly.area    #计算当前检测框面积
            iou = 0.0
            small_iou = 0.0
            if not segms1_poly.intersects(segms2_poly): #如果两四边形不相交
                iou = 0
            else:
                try:
                    inter_area = segms1_poly.intersection(segms2_poly).area   #相交面积
                    # print(inter_area)
                    union_area = area_segms1 + area_segms2 - inter_area
                    small_iou = float(inter_area) / float(min(area_segms1, area_segms2))
                    #print("small_iou:", small_iou)
                    # print("i:", i)
                    # print("j:", j)
                    # print("small_iou:", small_iou)
                    # union_area = MultiPoint(union_poly).convex_hull.area
                    # print(union_area)
                    if union_area == 0:
                        iou= 0
                    # iou = float(inter_area) / (union_area-inter_area)  #错了
                    iou=float(inter_area) / union_area
                    # print(iou)
                    # exit(0)
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured, iou set to 0')
                    iou = 0
            if(np_classes[i] == np_classes[j]):
                if(small_iou > 0.93):  #对大于设定阈值的检测框进行滤除
                    if(area_segms1 < area_segms2):
                        if(j in include_dict):
                            sub_list = include_dict[j]
                            sub_list.append(i)
                            include_dict[j] = sub_list
                        else:
                            sub_list = [i]
                            include_dict[j] = sub_list
                    else:
                        # suppressed[j] = 1
                        if(i in include_dict):
                            sub_list = include_dict[i]
                            sub_list.append(j)
                            include_dict[i] = sub_list
                        else:
                            sub_list = [j]
                            include_dict[i] = sub_list
    return include_dict, inter_index_all


# def test(sub_list, sub_bboxs):
   
def get_sub_bboxes_y_range(sub_list, bboxes, inter_index_all):
    all_y_length = 0
    used_index = []
    sub_index = []

    # for i in range(len(sub_list)):
    for i in sub_list:
        sub_index = []
        if(i in used_index):
            flag = False
        else:
            flag = True
        used_index.append(i)
        sub_index.append(i)
        # print
        tmp = [val for val in inter_index_all[i] if val in sub_list]
        
        tmp = list(set(tmp).difference(set(used_index)))
        if(len(tmp) == 0):
            if(flag):
                bunch_box = bboxes[sub_index]
                # sub_box_min_x = min(sub_bboxs[:, 0])
                bunch_box_min_y = min(bunch_box[:, 1])
                # bunch_box_max_x = max(sub_bboxs[:, 2])
                bunch_box_max_y = max(bunch_box[:, 3])
                all_y_length += (bunch_box_max_y - bunch_box_min_y)
            continue
        while(len(tmp) != 0):
            all_sub_tmp = []
            # used_index.append(tmp)
            used_index.extend(tmp)
            sub_index.extend(tmp)
            # print("tmp:", tmp)
            # print("used_index:", used_index)
            # if(isinstance(used_index[-1], list)):
            #     used_index = list(itertools.chain.from_iterable(used_index))

            for t in tmp:
                sub_tmp = [val for val in inter_index_all[t] if val in sub_list]
                # print("sub_tmp:", sub_tmp)
                # print("used_index:", used_index)
                sub_tmp = list(set(sub_tmp).difference(set(used_index)))
                all_sub_tmp.extend(sub_tmp)
            tmp = sub_tmp
            #     all_sub_tmp.append(sub_tmp)
            # all_sub_tmp = list(itertools.chain.from_iterable(all_sub_tmp))
        bunch_box = bboxes[sub_index]
        # sub_box_min_x = min(sub_bboxs[:, 0])
        bunch_box_min_y = min(bunch_box[:, 1])
        # bunch_box_max_x = max(sub_bboxs[:, 2])
        bunch_box_max_y = max(bunch_box[:, 3])
        all_y_length += (bunch_box_max_y - bunch_box_min_y)
    return all_y_length







def merge_box(include_dict, inter_index_all, bboxes, segms_list, labels):
    # print("bboxes:", bboxes)
    # print("type(bboxes):", type(bboxes))
    # exit(0)

    #print("inter_index_all:", inter_index_all)

    delete_index = []

    for include_index in include_dict:
        include_box = bboxes[include_index]
        # print("include_box:", include_box)
        # print("type(include_box):", type(include_box))

        # print("segms_list[include_index]:", segms_list[include_index])
        # print("type(segms_list[include_index]):", type(segms_list[include_index]))

        # print("labels:", labels)
        # print("type(labels):", type(labels))
        # exit(0)
        # print("include_box:", include_box)
        sub_list = include_dict[include_index]
        sub_bboxs = bboxes[sub_list]
        # print("sub_bboxs:", sub_bboxs)

        include_box_min_x = include_box[0]
        include_box_min_y = include_box[1]
        include_box_max_x = include_box[2]
        include_box_max_y = include_box[3]
        include_box_score = include_box[4]

        sub_box_min_x = min(sub_bboxs[:, 0])
        sub_box_min_y = min(sub_bboxs[:, 1])
        sub_box_max_x = max(sub_bboxs[:, 2])
        sub_box_max_y = max(sub_bboxs[:, 3])
        sub_bboxs_scores = sub_bboxs[:, 4]


        # sub_box_y_length = get_sub_bboxes_y_range(sub_list, bboxes, inter_index_all)
        # include_box_y_length = (include_box_max_y - include_box_min_y)
        # print("include_box_y_length:", include_box_y_length)
        # print("sub_box_y_length:", sub_box_y_length)


        # print("sub_bboxs:", sub_bboxs)


        sub_area = np.sum((sub_bboxs[:, 2] - sub_bboxs[:, 0]) * (sub_bboxs[:, 3] - sub_bboxs[:, 1]))
        include_area = (include_box_max_y - include_box_min_y) * (include_box_max_x - include_box_min_x)


        # print("sub_area:", sub_area)
        # print("include_area:", include_area)
        # exit(0)

        sub_box_h = sub_box_max_y - sub_box_min_y
        include_box_h = include_box_max_y - include_box_min_y
        sub_box_w = sub_box_max_x - sub_box_min_x
        include_box_w = include_box_max_y - include_box_min_y
        mean_include_box_h = np.mean(sub_bboxs[:, 3] - sub_bboxs[:, 1])
        #print("sub_box_h:", sub_box_h)
        #print("include_box_h:", include_box_h)
        # print("sub_box_w:", sub_box_w)
        # print("include_box_w:", include_box_w)
        #print("mean_include_box_h:", mean_include_box_h)
        #print("sub_box_min_y:", sub_box_min_y)
        #print("include_box_min_y:", include_box_min_y)
        #print("sub_box_max_y:", sub_box_min_y)
        #print("include_box_max_y:", include_box_max_y)
        # exit(0)
        # if(abs(sub_box_min_y - include_box_min_y) / mean_include_box_h >2 and abs(sub_box_max_y - include_box_max_y) / mean_include_box_h >2):

        if(include_box_score > 0.9 and np.mean(sub_bboxs_scores) > 0.9):
            return bboxes, segms_list, labels
        if(len(sub_list) == 1):
            # sub_box_h = sub_box_max_y - sub_box_min_y
            # include_box_h = include_box_max_y - include_box_min_y
            # sub_box_w = sub_box_max_x - sub_box_min_x
            # include_box_w = include_box_max_y - include_box_min_y

            # print("sub_box_h:", sub_box_h)
            # print("include_box_h:", include_box_h)
            # print("sub_box_w:", sub_box_w)
            # print("include_box_w:", include_box_w)
            # print("=============================")
            if(abs(include_box_h - sub_box_h) / include_box_h < 0.5  and abs(include_box_w - sub_box_w) / include_box_w > 0.7):
                if(include_box_score >= np.mean(sub_bboxs_scores)):
                    delete_index.append(sub_list[0])
                else:
                    delete_index.append(include_index)
        elif(len(sub_list) >= 2):
            if(abs(sub_box_min_y - include_box_min_y) / mean_include_box_h >2 and abs(sub_box_max_y - include_box_max_y) / mean_include_box_h >2):
                #print("ooooooooooooooooooooooo")
                above_list = [include_box[0], include_box[1], include_box[2], sub_box_min_y, include_box[4]]
                bboxes[include_index] = np.array(above_list)

                include_segms = segms_list[include_index]
                include_segms_arr=np.array(include_segms).reshape(int(len(include_segms)/2), 2)
                include_segms_arr=order_points_quadrangle(include_segms_arr)
                include_segms_min_x = min(include_segms_arr[:,0])
                include_segms_max_x = max(include_segms_arr[:,0])
                include_segms_min_y = min(include_segms_arr[:,1])
                include_segms_max_y = max(include_segms_arr[:,1])

                # 得到sub中最小y的索引
                sub_min_y_index = np.where(sub_bboxs[:, 1] == sub_box_min_y)[0][0]
                sub_max_y_index = np.where(sub_bboxs[:, 3] == sub_box_max_y)[0][0]

                sub_min_y_segms = segms_list[include_dict[include_index][sub_min_y_index]]
                sub_min_y_segms_arr=np.array(sub_min_y_segms).reshape(int(len(sub_min_y_segms)/2), 2)
                sub_min_y_segms_arr = order_points_quadrangle(sub_min_y_segms_arr)

                include_left_x = min(include_segms_arr[0][0], include_segms_arr[3][0])
                include_right_x = max(include_segms_arr[1][0], include_segms_arr[2][0])

                sub_up_y = max(sub_min_y_segms_arr[0][1], sub_min_y_segms_arr[1][1])
                # sub_down_y = min(sub_min_y_segms_arr[2][1], sub_min_y_segms_arr[3][1])

                segms_list[include_index] = [include_segms_arr[0][0], include_segms_arr[0][1], include_segms_arr[1][0], include_segms_arr[1][1], include_right_x, sub_up_y, include_left_x, sub_up_y]


                sub_max_y_segms = segms_list[include_dict[include_index][sub_max_y_index]]
                sub_max_y_segms_arr=np.array(sub_max_y_segms).reshape(int(len(sub_max_y_segms)/2), 2)
                sub_max_y_segms_arr = order_points_quadrangle(sub_max_y_segms_arr)
                sub_down_y = min(sub_max_y_segms_arr[2][1], sub_max_y_segms_arr[3][1])
                add_list = [include_left_x, sub_down_y, include_right_x, sub_down_y, include_segms_arr[2][0], include_segms_arr[2][1], include_segms_arr[3][0], include_segms_arr[3][1]]
                #print("===============================")
                #print(add_list)
                segms_list.append(add_list)
                labels = np.concatenate((labels,[0]))

                bboxes_rows = np.array([include_box[0], sub_box_max_y, include_box[2], include_box[3], include_box[4]])
                bboxes = np.row_stack((bboxes, bboxes_rows))
                # print("bboxes:", bboxes)
                # print("bboxes_rows:", bboxes_rows)
                # exit(0)

            elif(abs(sub_box_min_y - include_box_min_y) / mean_include_box_h >2):
                above_list = [include_box[0], include_box[1], include_box[2], include_box_min_y, include_box[4]]
                bboxes[include_index] = np.array(above_list)

                include_segms = segms_list[include_index]
                include_segms_arr=np.array(include_segms).reshape(int(len(include_segms)/2), 2)
                include_segms_arr=order_points_quadrangle(include_segms_arr)
                include_segms_min_x = min(include_segms_arr[:,0])
                include_segms_max_x = max(include_segms_arr[:,0])
                include_segms_min_y = min(include_segms_arr[:,1])
                include_segms_max_y = max(include_segms_arr[:,1])

                # 得到sub中最小y的索引
                sub_min_y_index = np.where(sub_bboxs[:, 1] == sub_box_min_y)[0][0]
                sub_max_y_index = np.where(sub_bboxs[:, 3] == sub_box_max_y)[0][0]

                sub_min_y_segms = segms_list[include_dict[include_index][sub_min_y_index]]
                sub_min_y_segms_arr=np.array(sub_min_y_segms).reshape(int(len(sub_min_y_segms)/2), 2)
                sub_min_y_segms_arr = order_points_quadrangle(sub_min_y_segms_arr)

                include_left_x = min(include_segms_arr[0][0], include_segms_arr[3][0])
                include_right_x = max(include_segms_arr[1][0], include_segms_arr[2][0])

                sub_up_y = max(sub_min_y_segms_arr[0][1], sub_min_y_segms_arr[1][1])

                segms_list[include_index] = [include_segms_arr[0][0], include_segms_arr[0][1], include_segms_arr[1][0], include_segms_arr[1][1], include_right_x, sub_up_y, include_left_x, sub_up_y]
            elif(abs(sub_box_max_y - include_box_max_y) / mean_include_box_h >2):
                above_list = [include_box[0], include_box[1], include_box[2], include_box_min_y, include_box[4]]
                bboxes[include_index] = np.array(above_list)

                include_segms = segms_list[include_index]
                include_segms_arr=np.array(include_segms).reshape(int(len(include_segms)/2), 2)
                include_segms_arr=order_points_quadrangle(include_segms_arr)
                include_segms_min_x = min(include_segms_arr[:,0])
                include_segms_max_x = max(include_segms_arr[:,0])
                include_segms_min_y = min(include_segms_arr[:,1])
                include_segms_max_y = max(include_segms_arr[:,1])

                # 得到sub中最小y的索引
                # sub_min_y_index = np.where(sub_bboxs[:, 1] == sub_box_min_y)[0][0]
                sub_max_y_index = np.where(sub_bboxs[:, 3] == sub_box_max_y)[0][0]

                # sub_min_y_segms = segms_list[include_dict[include_index][sub_min_y_index]]
                # sub_min_y_segms_arr=np.array(sub_min_y_segms).reshape(int(len(sub_min_y_segms)/2), 2)
                # sub_min_y_segms_arr = order_points_quadrangle(sub_min_y_segms_arr)

                include_left_x = min(include_segms_arr[0][0], include_segms_arr[3][0])
                include_right_x = max(include_segms_arr[1][0], include_segms_arr[2][0])

                # sub_up_y = max(sub_min_y_segms_arr[0][1], sub_min_y_segms_arr[1][1])
                
                sub_max_y_segms = segms_list[include_dict[include_index][sub_max_y_index]]
                sub_max_y_segms_arr=np.array(sub_max_y_segms).reshape(int(len(sub_max_y_segms)/2), 2)
                sub_max_y_segms_arr = order_points_quadrangle(sub_max_y_segms_arr)
                sub_down_y = min(sub_max_y_segms_arr[2][1], sub_max_y_segms_arr[3][1])

                segms_list[include_index] = [include_left_x, sub_down_y, include_right_x, sub_down_y, include_segms_arr[2][0], include_segms_arr[2][1], include_segms_arr[3][0], include_segms_arr[3][1]]

            else:
                #print("include_box_score:", include_box_score)
                #print("np.mean(sub_bboxs_scores):", np.mean(sub_bboxs_scores))
                # if(include_box_score > 2 * np.mean(sub_bboxs_scores)):
                # if(include_box_score - np.mean(sub_bboxs_scores) > 0.2):
                if((include_box_score > np.mean(sub_bboxs_scores) and include_box_score > 0.94) or (include_box_score > 1.5 *  np.mean(sub_bboxs_scores))):
                    for var in sub_list:
                        delete_index.append(var)
                else:
                    delete_index.append(include_index)
                
                # add_list = [include_left_x, sub_down_y, include_right_x, sub_down_y, include_segms_arr[2][0], include_segms_arr[2][1], include_segms_arr[3][0], include_segms_arr[3][1]]
                # segms_list.append(add_list)


            # print("sub_min_y_segms:", sub_min_y_segms)
            # print("sub_max_y_segms:", sub_max_y_segms)

            # exit(0)



                # min_x = 
                # min_y

                # include_segms = [segms_list[include_index]]

                # pass

            # pass
            # if()


    valid_index_list = [i for i in range(len(bboxes))]
    valid_index_list = list(set(valid_index_list).difference(set(delete_index)))
    keep_index_arr = np.array(valid_index_list)
    new_segms = []
    # new_charses = []
    new_classes = []
    # new_all_index_contours = []
    # new_all_index_contours_label = []
    for i in valid_index_list:
        new_segms.append(segms_list[i])
        # new_charses.append(charses[i])
        # new_classes.append(classes[i])
        # new_all_index_contours.append(all_index_contours[i])
        # new_all_index_contours_label.append(all_index_contours_label[i])
    # print(type(charses))
    # print(type(classes))
    # print(keep_index)
    # print(type(boxes))

    # exit(0)
    # print("keep_index_arr:", keep_index_arr)
    if(keep_index_arr.shape[0] != 0):
        return bboxes[keep_index_arr], new_segms, labels[keep_index_arr]
    else:
        return bboxes, segms_list, labels




# 原始
def merge_diff_class_rect_polygon_o(boxes, segms, np_classes):
    # np_classes = np.array(classes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    all_del_inds = []
    scores = boxes[:, 4]
    # scores = np.array([boxes_list[-1] for boxes_list in boxes])
    keep_index = []
    keep_classes = []
    order = scores.argsort()[::-1]

    # print("order:", order)
    num = len(boxes)
    # order = np.arange(num)
    suppressed = np.zeros((num), dtype=np.int)
    for _i in range(num):
        i = order[_i]
        if suppressed[i] == 1:# 对于抑制的检测框直接跳过
            continue
        keep_index.append(i)# 保留当前框的索引

        segms1 = segms[i]
        segms1_arr=np.array(segms1).reshape(int(len(segms1)/2), 2)
        
        index = np.where(np_classes != -1)[0]
        # index_same_class = np.where(np_classes == np_classes[])
        # index = np.where(np_classes != np_classes[i])[0]
        xx1 = np.maximum(x1[i], x1[index])
        yy1 = np.maximum(y1[i], y1[index])
        xx2 = np.minimum(x2[i], x2[index])
        yy2 = np.minimum(y2[i], y2[index])
        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        valid_index = index[np.where(inter>0)]
        if(valid_index.shape[0] == 0):
            continue
        valid_index_list = valid_index.tolist()
        # segms1_arr = np.array([[3, 29], [3, 26], [452, 62], [452, 30]])

        segms1_poly = Polygon(segms1_arr)
        segms1_poly = segms1_poly.convex_hull
        
        area_segms1 = segms1_poly.area    #计算当前检测框面积
        for _j in range(_i + 1, num):   #对剩余的而进行遍历
            j = order[_j]
            if suppressed[i] == 1:
                continue
            if not j in valid_index_list:
                continue

            segms2 = segms[j]
            segms2_arr=np.array(segms2).reshape(int(len(segms2)/2), 2)
            segms2_poly = Polygon(segms2_arr)
            segms2_poly = segms2_poly.convex_hull
            area_segms2 = segms2_poly.area    #计算当前检测框面积
            iou = 0.0
            small_iou = 0.0
            if not segms1_poly.intersects(segms2_poly): #如果两四边形不相交
                iou = 0
                merge_threshold1 = 0.85
                merge_threshold2 = 0.1

            else:
                try:
                    inter_area = segms1_poly.intersection(segms2_poly).area   #相交面积
                    # print(inter_area)
                    union_area = area_segms1 + area_segms2 - inter_area
                    small_iou = float(inter_area) / float(min(area_segms1, area_segms2))
                    #print("i:", i)
                    #print("j:", j)
                    #print("small_iou:", small_iou)
                    # union_area = MultiPoint(union_poly).convex_hull.area
                    # print(union_area)
                    if union_area == 0:
                        iou= 0
                    # iou = float(inter_area) / (union_area-inter_area)  #错了
                    iou=float(inter_area) / union_area
                    if(abs(iou - small_iou) > 0.8):
                        merge_threshold1 = 0.75
                    else:
                        merge_threshold1 = 0.85
                    if(abs(area_segms1 - area_segms2) > min(area_segms1, area_segms2)):
                        merge_threshold2 = 0.2
                    else:
                        merge_threshold2 = 0.1
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured, iou set to 0')
                    iou = 0
            if(np_classes[i] == np_classes[j]):
                
                if((iou >= merge_threshold1 or small_iou > merge_threshold1) and (scores[i] < 0.96 or scores[j] < 0.96)):  #对大于设定阈值的检测框进行滤除
                    # if(scores[i] > 0.9 and scores[j] < 0.9)
                    if(scores[i] > 0.99):
                        suppressed[j] = 1
                    elif(scores[j] > 0.99):
                        keep_index.remove(i)
                        suppressed[i] = 1
                    elif(scores[i] > 0.94 and scores[i] - scores[j] > merge_threshold2):
                        suppressed[j] = 1
                    elif(scores[j] > 0.94 and scores[j] - scores[i] > merge_threshold2):
                        keep_index.remove(i)
                        suppressed[i] = 1
                    elif(area_segms1 < area_segms2):
                        keep_index.remove(i)
                        suppressed[i] = 1
                    else:
                        suppressed[j] = 1
                elif(scores[i] > 0.96 and scores[j] > 0.96):
                    # print("y1[i]:", y1[i])
                    # print("y1[i]:", y2[i])
                    inter_h = min(y2[i], y2[j]) - max(y1[i], y1[j])
                    union_h = max(y2[i], y2[j]) - min(y1[i], y1[j])

                    #print("inter_h / union_h:", inter_h / union_h)
                    if(inter_h >0 and (inter_h / union_h > 0.9)):
                        if(area_segms1 < area_segms2):
                            keep_index.remove(i)
                            suppressed[i] = 1
                        else:
                            suppressed[j] = 1
                '''
                if(iou >= 0.85 or small_iou > 0.85):
                    if(area_segms1 < area_segms2):
                        keep_index.remove(i)
                        suppressed[i] = 1
                    else:
                        suppressed[j] = 1
                
                '''
    keep_index_arr = np.array(keep_index)
    new_segms = []
    new_classes = []
    for i in keep_index:
        new_segms.append(segms[i])
    if(keep_index_arr.shape[0] != 0):
        return boxes[keep_index_arr], new_segms, np_classes[keep_index_arr]
    else:
        return boxes, segms, np_classes


def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(13, 1. / 3)))
colors = [_to_color(x, base) for x in range(20)]

def npoly_2_4poly(segms):

    for i in range(len(segms)):
        segms[i] = np.array(segms[i]).reshape(int(len(segms[i])/2), 2).astype(np.int32)

        
        # print("before segms[i]:", segms[i])
        # print("type segms[i]:", type(segms[i]))

        # exit(0)
        #array_polygon = np.array(segms[i]).astype("int32").reshape(int(len(segms[i])/2),2)
        #rect = cv2.minAreaRect(array_polygon)
        #box = cv2.boxPoints(rect).astype(np.int32)
        #segms[i] = box
        # print("after segms[i]:", segms[i])
        # print("type segms[i]:", type(segms[i]))

    return segms

def draw_result(scale_w, scale_h, im_name, img, segms_list, labels, bboxes):

    
    include_dict, inter_index_all = merge_diff_class_rect_polygon(bboxes, segms_list, labels)

    bboxes, segms_list, labels = merge_box(include_dict, inter_index_all, bboxes, segms_list, labels)

    bboxes, segms_list, labels = merge_diff_class_rect_polygon_o(bboxes, segms_list, labels)
    # 
    scores = bboxes[:, 4]

    CLASSES_old = ('subject', 'ChineseNum', 'ArabNum', 'ArabNumBrackets', 'RomeNum', 'RomeNumBrackets', 'ChineseNumBrackets', 'ArabNumHalfBrackets', 'littleRomeNum', 'littleRomeNumHalfBrackets', 'ABCD', 'ABCDBrackets', 'halfABCDBrackets', 'littleRomeNumBrackets', 'DiX', 'circle', 'analysis', 'otherCircle')

    CLASSES = ('subject','ChineseNum','ArabNum','ArabNumBrackets','RomeNum','RomeNumBrackets','ArabNumHalfBrackets','littleRomeNum','ChineseNumBrackets','littleRomeNumBrackets','littleRomeNumHalfBrackets','ABCD','ABCDBrackets','halfABCDBrackets','DiX','circle','analysis','otherCircle')
    
    # o_txt = open(path, "w")

    data_result = []
    data = {"token:": im_name, "result": data_result}

    if(len(segms_list) != 0):
        segms_list = npoly_2_4poly(segms_list)

    for i in range(labels.shape[0]):
        cv2.polylines(img,[segms_list[i]],True, colors[labels[i] + 1], 3)
        cv2.putText(img, str(scores[i]), (int(segms_list[i][0][0]), int(segms_list[i][0][1])), cv2.FONT_HERSHEY_COMPLEX, 0.6,(0, 0, 0))
        segms_list[i][:, 0] = segms_list[i][:, 0] * scale_w
        segms_list[i][:, 1] = segms_list[i][:, 1] * scale_h
        segms_list[i] = segms_list[i].reshape(-1)

        segms_list_int = segms_list.copy()
        segms_list_int[i] = segms_list_int[i].astype(int)
        result = segms_list_int[i].tolist()

        result = [int(labels[i] + 1)] + result
        #result.extend([int(labels[i] + 1)])

        data_result.append(result)

        # segms_list[i] = [str(x) for x in segms_list[i]]
        # write_line = str(CLASSES_old.index(CLASSES[labels[i]]) + 1) + ',' + ",".join(segms_list[i]) + "," + str(scores[i])

        # o_txt.write(write_line + "\n")
    # return img
    data["result"] = data_result
    return data

# TODO: merge this method with the one in BaseDetector
# 
# 

def show_result(scale_w,
                scale_h,
                img_name,
                img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # print("segm_result:", segm_result)
    # exit(0)
    # draw segmentation masks
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # print("labels.shape:", labels.shape)
    segms_list = []
    delete_index = []

    if segm_result is not None:
        # print("len(segm_result):", len(segm_result))
        # print("segm_result:", segm_result)
        # exit(0)
        segms = mmcv.concat_list(segm_result)
        # print("len(segms):", len(segms))

        # print("segms:", segms)
        
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        
        # inds = np.where(bboxes[:, -1] > 0.1)[0]
        # print("len(bboxes):", len(bboxes))
        # print("len(segms):", len(segms))
        # print("segms[0]:", segms[0])
        # exit(0)
        # print("inds.shape[0]:", inds.shape[0])

        for index, i in enumerate(inds):

            box = bboxes[i]
            box = list(map(int, box))
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]

            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

            mask = maskUtils.decode(segms[i]).astype(np.bool)
            # print(np.where(mask == True))
            # print()
            # exit(0)
            # print("mask:", mask)

            # print("type(mask):", type(mask))
            # print("type(img):", type(img))
            # print("mask.shape:", mask.shape)
            # print("color_mask:", color_mask)
            mask_img = mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8) * 255

            contours,hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # print("contours[0]:", contours[0])
            # exit(0)
            max_area=0
            #print("len(contours):", len(contours))
            if(len(contours) ==0):
                delete_index.append(index)
                continue
            max_cnt = contours[0]
            for cnt in contours:
                area=cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_cnt = cnt
            perimeter = cv2.arcLength(max_cnt,True)
            epsilon = 0.01*cv2.arcLength(max_cnt,True)
            approx = cv2.approxPolyDP(max_cnt,epsilon,True)
            pts = approx.reshape((-1,2))
            pts[:,0] = pts[:,0] + box[0]
            pts[:,1] = pts[:,1] + box[1]
            segms_single = list(pts.reshape((-1,)))

            segms_single = list(map(int, segms_single))
            if len(segms_single)<6:
                delete_index.append(index)
                pass
            else:
                segms_list.append(segms_single)


        inds = np.delete(inds, delete_index)
        labels = labels[inds]
        bboxes = bboxes[inds]
    img = draw_result(scale_w, scale_h, img_name, img, segms_list, labels, bboxes)
    return img
    #mmcv.imshow_det_bboxes(
    #    img,
    #    bboxes,
    #    labels,
    #    class_names=class_names,
    #    score_thr=score_thr,
    #    show=show,
    #    wait_time=wait_time,
    #    out_file=out_file)
    #if not (show or out_file):
    #    return img


def show_result_pyplot(img,
                       result,
                       class_names,
                       score_thr=0.3,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    img = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))

