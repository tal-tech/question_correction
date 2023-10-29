#i coding:utf-8`
import torch
import cv2
import numpy as np
import math, time, glob, time
import cv2, os, sys, shutil, time
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from numpy import *
import heapq
import json

class_num = 1
from threading import Thread


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label, img_index = zip(*batch)

    aa1 = 0
    bb1 = 0
    k = 0
    k1 = 0
    max_len = len(label[0]) + 1
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]
    if bb1 < 500:
        bb1 = 500
    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]
        img_mask_sub_s = torch.ones(1, img_size_h,
                                    img_size_w).type(torch.FloatTensor)
        img_mask_sub_s = img_mask_sub_s * 255.0
        img_mask_sub = torch.cat((ii, img_mask_sub_s), dim=0)
        padding_h = aa1 - img_size_h
        padding_w = bb1 - img_size_w
        m = torch.nn.ZeroPad2d((0, padding_w, 0, padding_h))
        img_mask_sub_padding = m(img_mask_sub)
        img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
        if k == 0:
            img_padding_mask = img_mask_sub_padding
        else:
            img_padding_mask = torch.cat(
                (img_padding_mask, img_mask_sub_padding), dim=0)
        k = k + 1

    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0, max_len - ii1_len, 0, 0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding, ii1_padding), dim=0)
        k1 = k1 + 1

    img_padding_mask = img_padding_mask / 255.0
    return img_padding_mask, label_padding, img_index


def load_decode_dict(dictFile):
    global class_num
    dict_decode = {}
    dict_line = open(dictFile).readlines()

    for line in dict_line:
        index, name = line.split()
        index = int(
            index.replace('\n', '').replace('\r', '').replace('\t', ''))
        name = name.replace('\n', '').replace('\r', '').replace('\t', '')
        if index not in dict_decode:
            dict_decode[index] = name
    class_num = len(dict_decode)
    return dict_decode


class custom_dset(data.Dataset):
    def __init__(self, train, train_label, train_idx):
        self.train = train
        self.train_label = train_label
        self.img_idx = train_idx

    def __getitem__(self, index):
        train_setting = torch.from_numpy(np.array(self.train[index]))
        label_setting = torch.from_numpy(np.array(
            self.train_label[index])).type(torch.LongTensor)
        idx_setting = self.img_idx[index]
        size = train_setting.size()

        train_setting = train_setting.view(1, size[2], size[3])
        label_setting = label_setting.view(-1)

        return train_setting, label_setting, idx_setting

    def __len__(self):
        return len(self.train)


def use_gpu(whichGpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = whichGpu  # gpu_id


class formula_ocr():
    def __init__(self, encoderPath, decoderPath, dict_path, max_batchszie,
                 PREDICT_GPU_ID):
        self.use_gpu = True
        gpu_use = PREDICT_GPU_ID
        worddicts_r = load_decode_dict(dict_path)
        num = len(gpu_use)
        gpuNum = gpu_use[0]
        for i in range(1, num):
            gpuNum = gpuNum + ',' + gpu_use[i]
        use_gpu(gpuNum)

        encoderPath = encoderPath
        decoderPath = decoderPath
        self.EOS_TOKEN = 1
        self.topwidth = 5
        self.hidden_size = 256
        self.maxlen = 148
        self.max_w = 800
        self.en_model = []
        self.de_model = []
        self.model_index = 0
        self.worddicts_r = worddicts_r
        self.max_batchsize = max_batchszie

        if encoderPath and decoderPath:
            print('loading pretrained formula models ......')
            self.encoder = torch.jit.load(encoderPath)
            self.attn_decoder1 = torch.jit.load(decoderPath)

        if torch.cuda.is_available() and use_gpu:
            print("set cuda model")
            self.encoder = self.encoder.cuda()
            self.attn_decoder1 = self.attn_decoder1.cuda()

        else:
            print("gpu is not avalible")
        self.encoder.eval()
        self.attn_decoder1.eval()

        self.en_model.append(self.encoder)
        self.de_model.append(self.attn_decoder1)

    def resize_64(self, img, out_h):

        resize_w = int(out_h * img.shape[1] / img.shape[0])
        if resize_w / out_h > 10.5:
            resize_w = int(out_h * 10.5)
        out_img = cv2.resize(img, (resize_w, out_h))
        mat = np.zeros([1, out_h, resize_w], dtype='uint8')
        mat[0, :, :] = out_img
        return mat

    def receive(self, imgs):
        feature_total = []
        label_total = []
        imgidx_total = []

        for idx_img, im in enumerate(imgs):
            if im.ndim == 3 and im.shape[-1] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            w_ratio = 1
            h_ratio = 1
            w_left = 0

            mat = self.resize_64(im, 64)
            ratio = [w_left, h_ratio, w_ratio]
            label = []
            feature_batch = []
            label_batch = []
            imgidx_batch = []
            feature_batch.append(mat)
            feature_total.append(feature_batch)
            label_batch.append(label)
            label_total.append(label_batch)
            imgidx_batch.append(idx_img)
            imgidx_total.append(imgidx_batch)

        return feature_total, label_total, imgidx_total, im, ratio

    def predict(self, test_fea, test_label, test_img_idx, img_ori, batch_size,
                ratio):
        batch_size_test = min(batch_size, self.max_batchsize)
        test_fea_ol = []

        for one in test_fea:
            test_fea_ol.append([one[0]])

        off_image_test = custom_dset(test_fea_ol, test_label, test_img_idx)
        encoder_model = self.encoder
        decoder_model = self.attn_decoder1

        test_loader = torch.utils.data.DataLoader(dataset=off_image_test,
                                                  batch_size=batch_size_test,
                                                  shuffle=False,
                                                  collate_fn=collate_fn)
        print("test_loader:", len(test_loader))
        formula_result_all = []
        prob_all = []
        point_result_all = []
        idx_all = []
        for step_t, (x_t, y_t, idx_img) in enumerate(test_loader):
            formula_result = []
            prob_batch = []
            point_result = []
            if len(x_t) < batch_size_test:
                batch_size_test = len(x_t)
            print("batch_size:#####", batch_size_test)

            h_mask_t = []
            w_mask_t = []
            for i in x_t:
                s_w_t = str(i[1][0])
                s_h_t = str(i[1][:, 1])
                w_t = s_w_t.count('1')
                h_t = s_h_t.count('1')
                h_comp_t = int(h_t / 16) + 1
                w_comp_t = int(w_t / 16) + 1
                h_mask_t.append(h_comp_t)
                w_mask_t.append(w_comp_t)

            x_t = x_t.cuda()
            y_t = y_t.cuda()

            output_highfeature_t = encoder_model(x_t)
            x_mean_t = torch.mean(output_highfeature_t)
            x_mean_t = float(x_mean_t)
            output_area_t1 = output_highfeature_t.size()
            output_area_t = output_area_t1[3]
            dense_input = output_area_t1[2]

            decoder_input_t = torch.LongTensor([1] * batch_size_test)
            decoder_input_t = decoder_input_t.cuda()
            decoder_hidden_t = torch.ones(batch_size_test, 1,
                                          self.hidden_size).cuda()
            decoder_hidden_t = decoder_hidden_t * x_mean_t
            decoder_hidden_t = torch.tanh(decoder_hidden_t)
            prediction = torch.zeros(batch_size_test, self.maxlen)
            topv_pre = torch.zeros(batch_size_test, self.maxlen)

            decoder_attention_t = torch.zeros(batch_size_test, 1, dense_input,
                                              output_area_t).cuda()
            attention_sum_t = torch.zeros(batch_size_test, 1, dense_input,
                                          output_area_t).cuda()

            m = torch.nn.ZeroPad2d((0, self.maxlen - y_t.size()[1], 0, 0))

            et_mask = torch.zeros(batch_size_test, dense_input,
                                  output_area_t).cuda()
            for i in range(batch_size_test):
                et_mask[i][:h_mask_t[i], :w_mask_t[i]] = 1

            for i in range(self.maxlen):

                decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t, ip1 = decoder_model(
                    decoder_input_t, decoder_hidden_t, output_highfeature_t,
                    attention_sum_t, decoder_attention_t, et_mask)
                topv, topi = torch.max(decoder_output, 2)
                if torch.sum(topi) == 0:
                    break
                decoder_input_t = topi
                decoder_input_t = decoder_input_t.view(batch_size_test)
                topv = topv.view(batch_size_test)
                topv_pre[:, i] = topv
                prediction[:, i] = decoder_input_t

            for i in range(batch_size_test):
                prediction_sub = []
                prediction_real = []
                for j in range(self.maxlen):
                    if int(prediction[i][j]) == 0:
                        break
                    else:
                        prediction_sub.append(int(prediction[i][j]))
                        prediction_real.append(self.worddicts_r[int(
                            prediction[i][j])])

                if len(prediction_sub) < self.maxlen:
                    prediction_sub.append(0)

                reg_result = ' '.join(prediction_real)
                formula_result.append(reg_result)
            sumtorch = torch.sum(topv_pre, 1)
            sum_numpy = sumtorch.detach().numpy()
            prob = np.e**(sum_numpy)
            prob = prob.tolist()
            prob_batch += prob

            for idx_list in idx_img:
                idx_all = idx_all + idx_list

            formula_result_all = formula_result_all + formula_result
            point_result_all = point_result_all + point_result
            prob_all = prob_all + prob_batch

        if len(idx_all) > 1:
            idx_result_pos = list(zip(idx_all, formula_result_all))
            idx_result_pos_prob = list(zip(idx_all, prob_all))
            idx_result_pos.sort()
            idx_result_pos_prob.sort()

            formula_result_all = [ocr for img_idx, ocr in idx_result_pos]
            prob_all = [ocr for img_idx, ocr in idx_result_pos_prob]

        return_list = [formula_result_all, point_result_all, prob_all]
        return formula_result_all, prob_all