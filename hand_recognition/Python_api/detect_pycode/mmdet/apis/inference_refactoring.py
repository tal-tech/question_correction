import warnings
import matplotlib.pyplot as plt

import mmcv
import torch

from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.ops import RoIAlign, RoIPool
import numpy as np
import cv2
import shapely
from shapely.geometry import Polygon,MultiPoint

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
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    # print("model.cfg:", model.cfg)
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
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
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
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

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.savefig("result.jpg")

    # plt.show()


def if_merge_ptext_polygon_common(inter_poly, poly1, poly2, segms1_arr, segms2_arr, score_i, score_j, img_shape, boxes_i, boxes_j):

    # 得到四边形
    # import pdb
    # pdb.set_trace()
    # quadrangle1 = npoly_2_4poly(segms1_arr, img_shape)
    # quadrangle2 = npoly_2_4poly(segms2_arr, img_shape)

    # quadrangle1 = order_points_quadrangle(quadrangle1)
    # quadrangle2 = order_points_quadrangle(quadrangle2)
    
    # if()

    # 得到两个四边形的交集
    # poly1 = Polygon(quadrangle1).convex_hull
    # poly2 = Polygon(quadrangle2).convex_hull
    # # poly1 = Polygon(segms1_arr)
    # # poly2 = Polygon(segms2_arr)
    # intersection_poly = poly1.intersection(poly2)

    # import pdb
    # pdb.set_trace()
    intersection_x = [item[0] for item in inter_poly.exterior.coords]
    intersection_y = [item[1] for item in inter_poly.exterior.coords]
    if(len(intersection_y) == 0):
        # print("intersection_poly.area:", intersection_poly)
        return False
    if(min(intersection_y) == max(intersection_y)):
        return True

    area1 = (boxes_i[3] - boxes_i[1]) * (boxes_i[2] - boxes_i[0])
    area2 = (boxes_j[3] - boxes_j[1]) * (boxes_j[2] - boxes_j[0])

    intersection_x1 = min(intersection_x)
    intersection_x2 = max(intersection_x)

    
    
    if(area1 > area2):
        y1 = min(boxes_i[1], boxes_i[3], boxes_j[1], boxes_j[3])
        y2 = max(boxes_i[1], boxes_i[3], boxes_j[1], boxes_j[3])
        x1 = min(boxes_j[0], boxes_j[2])
        x2 = max(boxes_j[0], boxes_j[2])

        union_poly_arr = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        union_poly = Polygon(union_poly_arr)
        intersection_poly2 = union_poly.intersection(poly1)
        
    else:
        y1 = min(boxes_i[1], boxes_i[3], boxes_j[1], boxes_j[3])
        y2 = max(boxes_i[1], boxes_i[3], boxes_j[1], boxes_j[3])
        x1 = min(boxes_i[0], boxes_i[2])
        x2 = max(boxes_i[0], boxes_i[2])

        union_poly_arr = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        union_poly = Polygon(union_poly_arr)
        intersection_poly2 = union_poly.intersection(poly2)
    
    intersection_y2 = [item[1] for item in intersection_poly2.exterior.coords]

    # # 取两个四边形交集的中心

    # # 得到两个四边形的关系
    # # 得到x最小的四边形
    # x1_arr = segms1_arr[:, 0]
    # y1_arr = segms1_arr[:, 1]

    # x2_arr = segms2_arr[:, 0]
    # y2_arr = segms2_arr[:, 1]

    # x1_arr = segms1_arr[:, 0]
    # y1_arr = segms1_arr[:, 1]

    # x2_arr = segms2_arr[:, 0]
    # y2_arr = segms2_arr[:, 1]

    # min_x = min(min(x1_arr), min(x2_arr))
    # max_x = max(max(x1_arr), max(x2_arr))

    # min_y = min(min(y1_arr), min(y2_arr))
    # max_y = max(max(y1_arr), max(y2_arr))

    # # print()
    # segms1_arr_copy = segms1_arr.copy()
    # segms2_arr_copy = segms2_arr.copy()

    # # 得到四边形的边框

    # # 计算y轴交并比，若大于0.9，则认为可以合在一起
    # segms1_arr[:, 0] = segms1_arr[:, 0] - min_x
    # segms1_arr[:, 1] = segms1_arr[:, 1] - min_y

    # segms2_arr[:, 0] = segms2_arr[:, 0] - min_x
    # segms2_arr[:, 1] = segms2_arr[:, 1] - min_y



    # 取相交的这一段x
    # y_iou = (min(max(y1_arr), max(y2_arr)) - max(min(y1_arr), min(y2_arr))) / (max_y - min_y)
    # y_iou = (min(boxes_i[3], boxes_j[3]) - max(boxes_i[1], boxes_j[1])) / (max(boxes_i[3], boxes_j[3]) - min(boxes_i[1], boxes_j[1]))
    # # y_iou = (min(boxes_i[3], boxes_j[3]) - max(boxes_i[1], boxes_j[1])) / (max(intersection_y) - min(intersection_y))
    y_iou = (max(intersection_y) - min(intersection_y)) / (max(intersection_y2) - min(intersection_y2))
    # print("y_iou:", y_iou)
    if(y_iou < 0.65):
        return False
    return True

# 置0
def set_0(arr, i, j_list):
    for j in j_list:
        arr[i][j] = 0
        arr[j][i] = 0
    return arr


def merge_segms_and_bboxes(merge_index_lists, bboxes, segms_list, labels):
    merge_bboxes = np.array([])
    merge_segms = []
    merge_labels = []

    for merge_index_list in merge_index_lists:
        merge_boxs = bboxes[merge_index_list]
        max_y = int(max(merge_boxs[:, 3]))
        min_y = int(min(merge_boxs[:, 1]))
        max_x = int(max(merge_boxs[:, 2]))
        min_x = int(min(merge_boxs[:, 0]))
        mean_scores = np.mean(merge_boxs[:, 4])
        
        # import pdb
        # pdb.set_trace()
        det_mask = np.zeros([max_y - min_y, max_x - min_x], np.uint8)

        for index in merge_index_list:
            segms_index = segms_list[index]
            segms_index_arr=np.array(segms_index).reshape(int(len(segms_index)/2), 2)
            segms_index_arr[:, 0] = segms_index_arr[:, 0] - min_x
            segms_index_arr[:, 1] = segms_index_arr[:, 1] - min_y

            
            segms_index_arr = segms_index_arr.reshape(-1,2).astype(np.int32)
            # import pdb
            # pdb.set_trace()
            cv2.fillPoly(det_mask, [segms_index_arr], 255)
        # det_mask = cv2.copyMakeBorder(det_mask,5,5,5,5, cv2.BORDER_CONSTANT,value=[0,0,0])
        det_mask = cv2.copyMakeBorder(det_mask,0,0,0,0, cv2.BORDER_CONSTANT,value=[0,0,0])


        contours,hierarchy = cv2.findContours(det_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_area=0
        if(len(contours) ==0):
            continue
        max_cnt = contours[0]
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt
        # # perimeter = cv2.arcLength(max_cnt,True)
        # # epsilon = 0.005*cv2.arcLength(max_cnt,True)
        epsilon = 0.001*cv2.arcLength(max_cnt,True)
        # # epsilon = 0.0000001*cv2.arcLength(max_cnt,True)
        approx = cv2.approxPolyDP(max_cnt,epsilon,True)

        # approx = max_cnt


        pts = approx.reshape((-1,2))
        pts[:,0] = pts[:,0] + min_x
        pts[:,1] = pts[:,1] + min_y

        segms_single = list(pts.reshape((-1,)))

        segms_single = list(map(int, segms_single))
        mrege_box = np.array([[min_x, min_y, max_x, max_y, mean_scores]])
        if(len(merge_bboxes) == 0):
            merge_bboxes = mrege_box
        else:
            merge_bboxes = np.concatenate((merge_bboxes, mrege_box))

        merge_labels.append(labels[merge_index_list[0]])

        merge_segms.append(segms_single)

    return merge_segms, merge_bboxes, merge_labels

def get_new_bbox_and_bboxes(boxes_relation, bboxes, segms_list, labels):
    new_segms_list = []
    new_bboxes = np.array([])
    new_labels_list = []

    merge_index_lists = []
    boxes_relation_copy = boxes_relation.copy()
    
    for i in range(len(boxes_relation[0])):
        if((boxes_relation[i] == 0).all()):
            new_labels_list.append(labels[i])
            new_segms_list.append(segms_list[i])
            # mrege_box = np.array([[min_x, min_y, max_x, max_y, mean_scores]])
            new_box = bboxes[i]
            if(len(new_bboxes) == 0):
                new_bboxes = np.array([new_box])
            else:
                new_bboxes = np.concatenate((new_bboxes, [new_box]))
        elif((boxes_relation[i] == -1).all()):
            pass
        else:
            merge_index_list_i = [i]
            index = np.where(boxes_relation_copy[i] == 1)[0]
            index_list = list(index)
            
            boxes_relation_copy = set_0(boxes_relation_copy, i, index_list)
            while(index_list):
                j = index_list.pop()
                merge_index_list_i.append(j)
                index_j = np.where(boxes_relation_copy[j] == 1)[0]
                if(len(index_j) != 0):
                    index_list.extend(list(index_j))

                    boxes_relation_copy = set_0(boxes_relation_copy, j, list(index_j))
            if(len(merge_index_list_i)>1):
                merge_index_lists.append(merge_index_list_i)
    # import pdb
    # pdb.set_trace()
    merge_segms, merge_bboxes, merge_labels  = merge_segms_and_bboxes(merge_index_lists, bboxes, segms_list, labels)

    merge_bboxes = merge_bboxes.astype("float32")

    if(len(merge_segms) != 0 and len(new_segms_list) != 0):

        new_segms_list.extend(merge_segms)
        new_bboxes = np.concatenate((new_bboxes, merge_bboxes))
        new_labels_list.extend(merge_labels)
    elif(len(merge_segms) != 0 and len(new_segms_list) == 0):
        new_segms_list = merge_segms
        new_bboxes = merge_bboxes
        new_labels_list = merge_labels

    
    return new_segms_list, new_bboxes, np.array(new_labels_list)


class Polygon_info:
    def __init__(self, box_num):
        self.segms_arr = [0] * box_num
        self.polygon = [0] * box_num
        self.polygon_area = [0] * box_num
    def __call__(self, index, segms1):
        # if(self.polygon[index] == 0):
        segms1_arr=np.array(segms1).reshape(int(len(segms1)/2), 2)
        segms1_poly = Polygon(segms1_arr)
        segms1_poly = segms1_poly.convex_hull
        area_segms1 = segms1_poly.area    #计算当前检测框面积
        self.polygon[index] = segms1_poly
        self.polygon_area[index] = area_segms1
        self.segms_arr[index] = segms1_arr
        return segms1_arr, segms1_poly, area_segms1
            


def merge_diff_class_rect_polygon_common_hand_text(boxes, segms, np_classes, img_shape):

    # 框之间关系矩阵
    # 0:不需要改变，1:i,j需要合框，-1:该框需要删除
    boxes_relation = np.zeros((len(boxes), len(boxes)))
    # 多边形框的列表
    polygon_info = Polygon_info(len(boxes))
    polygon_list = [0] * len(boxes)
    CLASSES_common = ['ptext','htext','pformula','hformula','ld','nld','oc','mixformula','graph','excel','p_formula_set','p_up_down','h_formula_set','h_up_down','p_matrix','h_matrix']
    graph_excel_list = ['graph', 'excel']

    # 包含关系
    includable_set = {}
    includable_set['ptext'] = ['p_up_down', 'p_matrix']
    includable_set['htext'] = ['h_up_down', 'h_matrix']
    includable_set['pformula'] = ['p_up_down', 'p_matrix', 'p_formula_set']
    includable_set['hformula'] = ['h_up_down', 'h_matrix', 'h_formula_set']
    # includable_set['ld'] = ['p_up_down', 'p_matrix', 'h_up_down', 'h_matrix']
    # includable_set['nld'] = ['p_up_down', 'p_matrix', 'h_up_down', 'h_matrix']
    # includable_set['oc'] = ['p_up_down', 'p_matrix', 'h_up_down', 'h_matrix']
    includable_set['ld'] = ['p_up_down', 'p_matrix', 'h_up_down', 'h_matrix', 'pformula', 'hformula', 'ptext', 'htext']
    includable_set['nld'] = ['p_up_down', 'p_matrix', 'h_up_down', 'h_matrix', 'pformula', 'hformula', 'ptext', 'htext']
    includable_set['oc'] = ['p_up_down', 'p_matrix', 'h_up_down', 'h_matrix', 'pformula', 'hformula', 'ptext', 'htext']
    includable_set['mixformula'] = ['p_up_down', 'p_matrix', 'h_up_down', 'h_matrix']

    includable_set['excel'] = []
    includable_set['graph'] = []

    includable_set['p_formula_set'] = ['ptext', 'p_up_down', 'p_matrix', 'pformula']
    includable_set['h_formula_set'] = ['htext', 'h_up_down', 'h_matrix', 'hformula']
    includable_set['p_up_down'] = []
    includable_set['h_up_down'] = []
    includable_set['p_matrix'] = ['p_up_down']
    includable_set['h_matrix'] = ['h_up_down']
    # np_classes = np.array(classes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    all_del_inds = []
    scores = boxes[:, 4]
    keep_index = []
    keep_classes = []
    order = scores.argsort()[::-1]
    num = len(boxes)
    # order = np.arange(num)
    suppressed = np.zeros((num), dtype=np.int)
    for _i in range(num):
        i = order[_i]

        # 若i的类别是ptext或htext,则只要两类别在纵轴上有相交，则保留
        # np_classes
        if suppressed[i] == 1:# 对于抑制的检测框直接跳过
            continue
        keep_index.append(i)# 保留当前框的索引
        
        index = np.where(np_classes != -1)[0]
        xx1 = np.maximum(x1[i], x1[index])
        yy1 = np.maximum(y1[i], y1[index])
        xx2 = np.minimum(x2[i], x2[index])
        yy2 = np.minimum(y2[i], y2[index])
        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h



        # 按相交面积排序
        valid_index = index[np.where(inter>0)]
        
        if(valid_index.shape[0] == 0):
            continue
        valid_index_list = valid_index.tolist()
                
        if(polygon_info.polygon[i] == 0):
            segms1 = segms[i]
            segms1_arr, segms1_poly,area_segms1 = polygon_info(i, segms1)
        else:
            segms1_arr, segms1_poly,area_segms1 = polygon_info.segms_arr[i], polygon_info.polygon[i], polygon_info.polygon_area[i]
        
        for _j in range(_i + 1, num):   #对剩余的而进行遍历
            j = order[_j]
            if suppressed[i] == 1:
                continue
            if not j in valid_index_list:
                continue

            if(polygon_info.polygon[j] == 0):
                segms2 = segms[j]
                segms2_arr, segms2_poly,area_segms2 = polygon_info(j, segms2)
            else:
                segms2_arr, segms2_poly,area_segms2 = polygon_info.segms_arr[j], polygon_info.polygon[j], polygon_info.polygon_area[j]
            iou = 0.0
            small_iou = 0.0
            if not segms1_poly.intersects(segms2_poly): #如果两四边形不相交
                iou = 0
            else:
                try:
                    inter_poly = segms1_poly.intersection(segms2_poly)

                    inter_area = inter_poly.area   #相交面积
                    
                    # print(inter_area)
                    union_area = area_segms1 + area_segms2 - inter_area
                    if(float(min(area_segms1, area_segms2) == 0)):
                        continue
                    small_iou = float(inter_area) / float(min(area_segms1, area_segms2))
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
                filter_list = [4, 5, 6,8,9,10,12]
                filter_graph_excel_list = [8,9]

                # if(iou >= 0.95 or small_iou >= 0.8):
                if(iou >= 0.95):
                    # suppressed[j] = 1
                    if(area_segms1 < area_segms2):
                        boxes_relation[i] = -1
                        keep_index.remove(i)
                        suppressed[i] = 1
                    else:
                        suppressed[j] = 1
                        boxes_relation[j] = -1
                # 若类别是印刷文本、手写文本、图则合并
                # 图暂未添加，多个图有问题
                elif((iou > 0.5 or small_iou > 0.5) and (np_classes[i] in filter_list)):
                    if(np_classes[i] in filter_graph_excel_list):
                        if(area_segms1 > 0.3 * area_segms2):
                            suppressed[j] = 1
                            boxes_relation[j] = -1

                    else:
                        suppressed[j] = 1
                        boxes_relation[j] = -1

                
                elif((small_iou > 0.1 and (np_classes[i] == 0 or np_classes[i] == 1)) or small_iou > 0.8):
                    if(small_iou >= 0.95 or (small_iou >= 0.8 and (np_classes[i] == 2 or np_classes[i] == 3))):
                        boxes_relation[i][j] = 1
                        boxes_relation[j][i] = 1

                    elif(if_merge_ptext_polygon_common(inter_poly, segms1_poly, segms2_poly, segms1_arr, segms2_arr,scores[i], scores[j], img_shape,  boxes[i], boxes[j])):
                        boxes_relation[i][j] = 1
                        boxes_relation[j][i] = 1
                
            else:
                class_name_i = CLASSES_common[np_classes[i]]
                class_name_j = CLASSES_common[np_classes[j]]
                # if(class_name_i == "ptext" and class_name_j == "htext"):
                    # print("small_iou:", small_iou)
                    # print("iou:", iou)
                    # print("area_segms1:", area_segms1)
                    # print("area_segms2:", area_segms2)

                includable_list_i = includable_set[class_name_i]
                if(not class_name_j in includable_list_i):
                    if(iou >= 0.8 or small_iou > 0.8):
                        if(class_name_i in graph_excel_list and class_name_j in graph_excel_list):
                            suppressed[j] = 1
                            boxes_relation[j] = -1

                        # elif(area_segms1 > area_segms2):
                        #     suppressed[j] = 1
                        #     boxes_relation[j] = -1

                        else:
                            # pass
                            includable_list_j = includable_set[class_name_j]
                            if(not class_name_i in includable_list_j):
                                if(class_name_j == "pformula" and class_name_i == "ptext"):
                                    suppressed[i] = 1
                                    keep_index.remove(i)
                                    boxes_relation[i] = -1

                                elif((class_name_i == "graph" or class_name_i == "excel") and (class_name_j == "htext" or class_name_j == "ptext")):
                                    if(scores[i] > 0.7):
                                        graph_excel_thr = 1.8
                                    else:
                                        graph_excel_thr = 2.5

                                    if(area_segms1 > graph_excel_thr * area_segms2):
                                        suppressed[j] = 1
                                        boxes_relation[j] = -1
                                        
                                    else:
                                        suppressed[i] = 1
                                        keep_index.remove(i)
                                        boxes_relation[i] = -1
                                elif((class_name_j == "graph" or class_name_j == "excel") and (class_name_i == "htext" or class_name_i == "ptext")):
                                    if(scores[j] > 0.7):
                                        graph_excel_thr = 1.8
                                    else:
                                        graph_excel_thr = 2.5
                                    if(area_segms2 > graph_excel_thr * area_segms1):
                                        suppressed[i] = 1
                                        keep_index.remove(i)
                                        boxes_relation[i] = -1
                                        
                                    else:
                                        suppressed[j] = 1
                                        # keep_index.remove(j)
                                        boxes_relation[j] = -1

                                # elif(class_name_j != "excel" and class_name_j != "graph"):
                                #     suppressed[j] = 1   
                                #     boxes_relation[j] = -1
                                else:
                                    if(area_segms1 < area_segms2):
                                        suppressed[i] = 1
                                        keep_index.remove(i)
                                        boxes_relation[i] = -1

                                    else:
                                        suppressed[j] = 1
                                        boxes_relation[j] = -1

    # boxes_relation = np.zeros((len(boxes), len(boxes)))
    return boxes_relation    

def show_result(img,
                det_img_shape,
                result,
                scale_w,
                scale_h,
                border,
                write_txt_path = None,
                score_thr=0.3,
                token=None,
                is_server=False,
                wait_time=0):
    """
    :type scale: 缩放比例
    :type img_name: str
    :type img: numpy
    :type result
    :rtype: float
    """
    # assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    segms_list = []
    delete_index = []

    # print("labels.shape:", labels.shape)

    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)

        inds = np.where(bboxes[:, -1] > score_thr)[0]

        for index, i in enumerate(inds):

            box = bboxes[i]
            box = list(map(int, box))
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

            # mask = maskUtils.decode(segms[i]).astype(np.bool)
            mask = segms[i]

            mask_img = mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8) * 255
            contours,hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            max_area=0

            if(len(contours) ==0):
                delete_index.append(index)
                continue
            max_cnt = contours[0]
            for cnt in contours:
                area=cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_cnt = cnt
            # print("max_cnt:", max_cnt)
            # perimeter = cv2.arcLength(max_cnt,True)
            # epsilon = 0.0001*cv2.arcLength(max_cnt,True)
            epsilon = 0.001*cv2.arcLength(max_cnt,True)
            approx = cv2.approxPolyDP(max_cnt,epsilon,True)
            # print("approx:", approx)
            # approx = max_cnt
            pts = approx.reshape((-1,2))
            pts[:,0] = pts[:,0] + box[0]
            pts[:,1] = pts[:,1] + box[1]
            segms_single = list(pts.reshape((-1,)))

            segms_single = list(map(int, segms_single))                

            if len(segms_single)<6:
                delete_index.append(index)

            elif(labels[i] == 13 and bboxes[i][4] < 0.6):
                delete_index.append(index)
            elif(labels[i] == 11 and bboxes[i][4] < 0.25):
                delete_index.append(index)
            elif((labels[i] == 9 or labels[i] == 10) and bboxes[i][4] < 0.17):
                delete_index.append(index)
            
            elif(box_h / box_w > 3 ):
                # 若类别是印刷，手写式子，混合式子过滤掉
                if(labels[i] == 2 or labels[i] == 3 or labels[i] == 7):
                    delete_index.append(index)
                # 若ptext,htext是竖长文本，且置信度较低，则过滤掉
                elif(labels[i] == 0 and bboxes[i][4] < 0.1):
                    delete_index.append(index)
                else:
                    segms_list.append(segms_single)
            # 若图片在顶部或底部，且宽占图片大小的大部分，则过滤掉
            elif(labels[i] == 8):
                if(box[3] >0.95*det_img_shape[0] and box_w > 0.8*det_img_shape[1] and box[1] > 0.6 * det_img_shape[0]):
                    delete_index.append(index)
                elif(box[1] <0.05*det_img_shape[0] and box_w > 0.8*det_img_shape[1] and box[3] < 0.4 * det_img_shape[0]):
                    delete_index.append(index)
                else:
                    segms_list.append(segms_single)
            
            else:
                segms_list.append(segms_single)

        inds = np.delete(inds, delete_index)
        labels = labels[inds]
        bboxes = bboxes[inds]

    # import pdb
    # pdb.set_trace()
    boxes_relation = merge_diff_class_rect_polygon_common_hand_text(bboxes, segms_list, labels, det_img_shape[:2])
    # boxes_relation = np.zeros((len(bboxes), len(bboxes)))

    if(len(boxes_relation)!=0):
        segms_list, bboxes, labels = get_new_bbox_and_bboxes(boxes_relation, bboxes, segms_list, labels)
        
        assert len(segms_list) == bboxes.shape[0] == labels.shape[0]
    if(is_server):
    # def draw_result_server(img, segms_list, labels, bboxes, scale_w, scale_h, border, token,scale=1, write_txt_path = None):

        img = draw_result_server(img, segms_list, labels, bboxes, scale_w, scale_h, border, token, write_txt_path)
    else:
        img = draw_result(img, segms_list, labels, bboxes, scale_w, scale_h, border, write_txt_path)
    return img


# def draw_result_server(img, segms_list, labels, bboxes, scale_w, scale_h, border, token, write_txt_path = None):

def draw_result(img, segms_list, labels, bboxes, scale_w, scale_h, border, write_txt_path = None):
    CLASSES_common = ['ptext','htext','pformula','hformula','ld','nld','oc','mixformula','graph','excel','p_formula_set','p_up_down','h_formula_set','h_up_down','p_matrix','h_matrix']

    # print(write_txt_path)
    # exit(0)
    scores = bboxes[:, 4]
    #o_txt = open(write_txt_path, "w")
    if(len(segms_list) != 0):
        # segms_list = npoly_2_4poly(segms_list, labels, img.shape)
        segms_list = npoly_2_4poly(segms_list, labels, img.shape, scale_w, scale_h, border)
    for i in range(labels.shape[0]):
        # if(labels[i] == 1):
        #     continue
        cv2.polylines(img,[segms_list[i]],True, colors[labels[i] + 1], 2)

        segms_list[i] = segms_list[i].reshape(-1)

        segms_list_int = segms_list.copy()

        segms_list[i] = [str(int(x)) for x in segms_list[i]]
        write_line = str(labels[i] + 1) + ',' + ",".join(segms_list[i]) + "," + str(scores[i])

        #o_txt.write(write_line + "\n")
    return img


def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(13, 1. / 3)))
colors = [_to_color(x, base) for x in range(21)]

def npoly_2_4poly(segms, labels, img_shape, scale_w, scale_h, border):

    for i in range(len(segms)):
        segms[i] = np.array(segms[i]).reshape(int(len(segms[i])/2), 2).astype(np.int32)
        
        # segms[i] = segms[i].reshape(-1)
        # segms[i] = segms[i] * scale
        segms[i][:, 0] = segms[i][:, 0] * scale_w
        segms[i][:, 1] = segms[i][:, 1] * scale_h
        
        bbox_offset = np.array([0, border],dtype=np.int32)
        segms[i] = segms[i] - bbox_offset
        segms[i][:, 0] = np.clip(segms[i][:, 0], 0, img_shape[1] - 1)
        segms[i][:, 1] = np.clip(segms[i][:, 1], 0, img_shape[0] - 1)

        # array_polygon = np.array(segms[i]).astype("int32").reshape(int(len(segms[i])/2),2)
        rect = cv2.minAreaRect(segms[i])
        box = cv2.boxPoints(rect).astype(np.int32)
        x_axis = box[:, 0]
        y_axis = box[:, 1]
        x_index = np.where(x_axis > img_shape[1] - 1)[0]
        x_axis[x_index] = img_shape[1] - 1
        y_index = np.where(y_axis > img_shape[0] - 1)[0]
        y_axis[y_index] = img_shape[0] - 1
        x_0_index = np.where(x_axis < 1)
        x_axis[x_0_index] = 1
        y_0_index = np.where(y_axis < 1)
        y_axis[y_0_index] = 1
        segms[i] = box
            
        

    return segms

        # img = draw_result_server(img, segms_list, labels, bboxes, scale_w, scale_h, border, token, write_txt_path)


def draw_result_server(img, segms_list, labels, bboxes, scale_w, scale_h, border, token, write_txt_path = None):
# def draw_result_server(scale, path, img_name, img, segms_list, labels, bboxes):
    scores = bboxes[:, 4]

    # o_txt = open(write_txt_path, "w")

    if(len(segms_list) != 0):
        # print("border:", border)
        # segms_list = npoly_2_4poly(segms_list, labels, img.shape)
        segms_list = npoly_2_4poly(segms_list, labels, img.shape, scale_w, scale_h, border)



    data_result = []
    data = {"token:": token, "result": data_result}
    
    for i in range(labels.shape[0]):
        cv2.polylines(img,[segms_list[i]],True, colors[labels[i] + 1], 3)
        # cv2.polylines(img,[segms_list[i]],True, colors[labels[i] + 1], 2)
        
        segms_list[i] = segms_list[i].reshape(-1)
        # segms_list[i] = segms_list[i] * scale
        segms_list_int = segms_list.copy()

        segms_list[i] = [str(int(x)) for x in segms_list[i]]
        write_line = str(labels[i] + 1) + ',' + ",".join(segms_list[i]) + "," + str(scores[i])
        
        segms_list_int[i] = segms_list_int[i].astype(int)
        result = segms_list_int[i].tolist()
        result.extend([float(scores[i])])
        result.extend([int(labels[i] + 1)])
        data_result.append(result)

        # o_txt.write(write_line + "\n")
        # pass
    # cv2.imwrite(write_txt_path.replace(".txt", ".jpg"), img)
    data["result"] = data_result
    return data
