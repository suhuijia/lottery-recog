#!/usr/bin/python
#!-*-coding:utf-8-*-
import os
import glob
import cv2
import time
import numpy as np
from char3755_recognition import *


def reversed_cmp(tp1, tp2):
    x = tp1[1]
    y = tp2[1]
    if x[0] < y[0]:
        return -1
    if x[0] > y[0]:
        return 1
    return 0


def getHorizontalHist(img, thr_value = 128):
    hist = np.sum((img > thr_value), axis=1).reshape((-1, 1))
    # h, w = img.shape
    # #print(h, w)
    # hist = [0 for z in range(0, h)]
    # for i in range(0, h):
    #     for j in range(0, w):
    #         if img[i, j] > 128:
    #             hist[i] += 1
    return hist


def getVerticalHist(img, thr_value = 128):
    hist = np.sum((img > thr_value), axis=0).reshape((-1, 1))
    # h, w = img.shape
    # #print(h, w)
    # hist = [0 for z in range(0, w)]
    # for i in range(0, w):
    #     for j in range(0, h):
    #         if img[j, i] > 128:
    #             hist[i] += 1
    return hist


def init_temp(path_lab, path_temp, offset=0):
    list_temp = []
    list_char = []
    filein = open(path_lab, 'r')
    id = offset
    for line in filein:
        x = line.replace('\r', '')
        labelname = x.replace('\n', '')
        #print(labelname)
        if offset > 0:
            labelname = ("\\" + labelname).decode("unicode_escape")
        list_char.append(labelname)
        paths = glob.glob(path_temp + str(id-offset) + "/*.*")
        for path in paths:
            path = path.replace('\\', '/')
            img = cv2.imread(path, 0)
            list_temp.append((id, img))
        id += 1
    return list_temp, list_char


def seg_lines(img):
    list_lines = []
    ret, bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    h, w = bin.shape
    hist = getHorizontalHist(bin)
    linebegin = -1
    for i in range(0, h):
        if hist[i] > 0 and linebegin < 0:
                linebegin = i
        elif hist[i] < 1 and linebegin >= 0:
            line_bin = bin[linebegin:i, :]
            hist_v = getVerticalHist(line_bin)
            l = -1
            r = w + 1
            for j in range(0, w, 1):
                if hist_v[j] > 0 and l == -1:
                    l = j
                if hist_v[w-j-1] > 0 and r == w+1:
                    r = w - j
                if l != -1 and r != w+1:
                    break

            list_lines.append([linebegin, i, l, r])
            linebegin = -1
    return list_lines
    
    
def seg_chars(img):
    list_chars = []
    ret, bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    h, w = bin.shape
    hist = getVerticalHist(bin)
    charbegin = -1
    list_buf = []
    for i in range(0, w):
        if hist[i] > 0 and charbegin < 0:
            charbegin = i
        elif hist[i] < 1 and charbegin >= 0:
            list_buf.append([charbegin, i])
            charbegin = -1
    if charbegin >= 0:
        list_buf.append([charbegin, w])
        charbegin = -1
    # print(len(list_buf))
    # print(list_buf)
    for ind in range(0, len(list_buf), 1):
        if ind == len(list_buf) - 1:
            list_chars.append(list_buf[ind])
            break
        if list_buf[ind][1] - list_buf[ind][0] > h * 3:
            continue
        if list_buf[ind][1] - list_buf[ind][0] > h * 3 / 4:
            list_chars.append(list_buf[ind])
            continue
        elif list_buf[ind+1][0] - list_buf[ind][1] < 5:
            list_buf[ind + 1][0] = list_buf[ind][0]
            continue
    return list_chars


def match(img_src, img_temp, thre=0.01):
    #list_loc = []
    res = cv2.matchTemplate(img_src, img_temp, cv2.TM_SQDIFF_NORMED)
    #print(res)
    #print(res.shape)
    return np.argwhere(res < thre)


def detection(img_path, list_temp, list_char, model, thre=0.01):
    list_res = []
    img = cv2.imread(img_path, 0)
    # print(img.shape[:])
    start_t = time.time()
    list_lines = seg_lines(img)
    print("seg_lines time:%.4fs"%(time.time()-start_t))
    # print(list_lines)
    match_time, recognize_time = 0.0, 0.0
    for id_l, line in enumerate(list_lines):
        if id_l == len(list_lines)-1:
            continue
        start_0 = time.time()
        # cv2.rectangle(img, (line[2], line[0]), (line[3], line[1]), (0, 255, 0), 2)
        # cv2.imshow('rect', img)
        # cv2.waitKey(1000)
        img_line = img[line[0]:line[1], line[2]:line[3]]
        list_linelab = []
        for t in list_temp:
            id = t[0]
            img_temp = t[1]
            h, w = img_temp.shape
            if h > line[1]-line[0]:
                continue
            res = match(img_line, img_temp, thre)
            if len(res) == 0:
                continue
            for r in res:
                lr = [r[1], r[1]+w]
                flag = True
                for ll in list_linelab:
                    box = ll[1]
                    if lr[0] == box[0] and lr[1] == box[2]:
                        flag = False
                        break
                    if (lr[0] < box[0] and lr[1] > box[0]) or (lr[0] < box[2] and lr[1] > box[2]):
                        flag = False
                        break
                if flag:
                    list_linelab.append((list_char[id], [lr[0], line[0], lr[1], line[1]]))
                    img_line[:, lr[0]:lr[1]+1] = 255
        start_1 = time.time()
        match_time += (start_1-start_0)

        ###### detect chn
        list_chars = seg_chars(img_line)
        # print(list_chars)
        for char_lr in list_chars:
            # print(char_lr)
            box = [char_lr[0], line[0], char_lr[1], line[1]]
            if char_lr[1] - char_lr[0] > 4*(line[1] - line[0]):
                continue
            char_img = img_line[:, char_lr[0]:char_lr[1]]
            ### recog box
            n_top = 1
            char_list, prob_list, img_nor = model.recognize(char_img, n_top)
            list_linelab.append((char_list[0], box))
            
        recognize_time += (time.time() - start_1)
        list_linelab = sorted(list_linelab, reversed_cmp)
        # print(list_linelab)
        list_res.append(list_linelab)

        # cv2.rectangle(img, (line[2], line[0]), (line[3], line[1]), (0, 255, 0), 2)
    # cv2.imwrite('result1.bmp', img)
    print("match time:%.4fs, \t recognize time:%.4fs"%(match_time, recognize_time))
    return list_lines, list_res


def lottery_recog(img_path, list_temp, list_char, model):
    """"""
    # list_temp, list_char = init_temp(path_lab, path_temp)
    list_lines, list_res = detection(img_path, list_temp, list_char, model)
    # print(list_lines)
    # print(len(list_res))
    # print(list_res)
    str_line_list = []
    for res_line in list_res:
        # print(len(res_line))
        # print(res_line)
        str_line = ''
        last = -1
        last_c = ''
        for r in res_line:
            # print(r)
            c = r[0]
            b = r[1]
            bias = 0
            if c == '1':
                bias += 1
            if last_c == '1':
                bias += 1
            if last < 0:
                last = b[2]
                last_w = b[2] - b[0]
                last_c = c
            elif len(last_c) > 1 and last_c[0] == 'O':
                str_line += ' '
                last = b[2]
                last_w = b[2] - b[0]
                last_c = c
            else:
                step = b[0] - last
                last = b[2]
                if step >= 12 + bias:
                    str_line += ' '
            str_line += r[0]
            # print(str_line)
        str_line_list.append(str_line)
        # print(str_line)
    # return list_lines, str_line_list, list_res
    item_list = []
    for i in range(len(list_lines)-1):

        if len(list_res[i]) == 0:
            continue

        item_dict = {}
        line_rect = list_lines[i]
        item_dict["itemstring"] = str_line_list[i]

        itemcoord_dict = {}
        itemcoord_dict['x'] = line_rect[2]
        itemcoord_dict['y'] = line_rect[0]
        itemcoord_dict['width'] = line_rect[3] - line_rect[2]
        itemcoord_dict['height'] = line_rect[1] - line_rect[0]
        # itemcoord = [itemcoord_dict]
        item_dict["itemcoord"] = [itemcoord_dict]

        words = []
        for res in list_res[i]:
            res_dict = {}
            res_dict['character'] = res[0]
            res_dict['x'] = res[1][0]
            res_dict['y'] = res[1][1]
            res_dict['width'] = res[1][2] - res[1][0]
            res_dict['height'] = res[1][3] - res[1][1]

            words.append(res_dict)

        item_dict["words"] = words

        item_list.append(item_dict)

    # print(item_list)
    return item_list



if __name__ == '__main__':
    path_lab = '../temp_all/map.txt'
    path_temp = '../temp_all/'
    list_temp, list_char = init_temp(path_lab, path_temp)
    path_chslab = './chs202_code.txt'
    path_chstemp = '../temp_chn/'
    list_chstemp, list_chschar = init_temp(path_chslab, path_chstemp, offset=84)
    #print(list_char)
    #print(len(list_temp))
    rootDir = '../test/'
    filename = 'sfc14.bmp'
    # filename = '4cjq.bmp'
    pb_file_path = "./graph_3755.pb"
    char_map_path = "./chs3755_code.txt"
    model = mnv2(pb_file_path, char_map_path)
    list_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    list_alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    list_op = ['*', '+', '-', '(', ')', '<', '>', '.']

    path = rootDir + filename
    list_lines, list_res = detection(path, list_temp + list_chstemp, list_char + list_chschar, model)
    for res_line in list_res:
        str_line = ''
        last = -1
        last_w = -1
        last_c = ''
        for r in res_line:
            c = r[0]
            b = r[1]
            h = b[3] - b[1]
            w = b[2] - b[0]
            bias = 0
            if c == '1':
                bias += 1
            if last_c == '1':
                bias += 1
            if last < 0:
                last = b[2]
                last_w = b[2] - b[0]
                last_c = c
            elif len(last_c) > 1 and last_c[0] == 'O':
                str_line += ' '
                last = b[2]
                last_w = b[2] - b[0]
                last_c = c
            else:
                step = b[0] - last
                last = b[2]
                last_w = b[2] - b[0]
                last_c = c
                #if c in list_num + list_alpha + list_op:
                #    h = h / 2
                if step >= 12 + bias:
                #print(h - (last_w + w) / 2)
                #if step >= h + (h - last_w - w) / 2:
                    str_line += ' '
            str_line += r[0]
        print(str_line)


    # result = lottery_recog(path, list_temp, list_char)
