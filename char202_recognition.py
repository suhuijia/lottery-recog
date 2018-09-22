# -*- coding: utf-8 -*-
import glob
import os
import tensorflow as tf
import time
import heapq
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # cpu


def im_resize(image, width, height, margin=4, inter=cv2.INTER_AREA):
    resized = cv2.resize(image, (width - 2 * margin, height - 2 * margin), interpolation=inter)
    if len(resized.shape) < 3:
        value = (255)
    else:
        b, g, r = 255, 255, 255
        value = (b, g, r)
    resized = cv2.copyMakeBorder(resized, margin, margin, margin, margin, cv2.BORDER_CONSTANT,
                                 value=value)
    return resized


def reader_data_test(data_path):
    char2id = {}
    char_map = {}
    with open(data_path, "r") as f:
        int_token = 0
        for line in f:
            sym = line.strip()
            char2id[sym] = int_token
            char_map[int_token] = sym
            int_token += 1

    return char_map, char2id


def normal_img(img):
    img_resized = im_resize(img, 32, 32, 3, cv2.INTER_CUBIC)
    return img_resized / 255.0


class mnv2(object):
    def __init__(self, pb_file_path, char_map_path):
        conprot = tf.ConfigProto()
        conprot.gpu_options.allow_growth = True
        conprot.allow_soft_placement = True
        self.char_map, self.char2id = reader_data_test(char_map_path)
        self.gf = tf.Graph().as_default()
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        self.sess = tf.Session(config=conprot)
        self.input_x = self.sess.graph.get_tensor_by_name("input:0")
        self.prob_tensor = self.sess.graph.get_tensor_by_name("mobilenetv2/prob:0")

    def recognize(self, img, n_top=10):
        img_nor = normal_img(img)
        x_out = [img_nor]
        feed_dict = {self.input_x: x_out}

        prob = self.sess.run(self.prob_tensor, feed_dict=feed_dict)[0]  # index 0 for batch_size
        topn = heapq.nlargest(n_top, range(len(prob)), prob.__getitem__)
        char_list, prob_list = [], []
        for i in topn:
            char = self.char_map.get(i)
            # x = ("\\" + char).encode('latin-1').decode('unicode_escape')  # python3
            x = ("\\" + char).decode("unicode_escape")  # python2
            char_list.append(x)
            prob_list.append(prob[i])
        return char_list, prob_list, img_nor


def main():
    # init
    pb_file_path = "./graph_202.pb"
    char_map_path = "./chs202_code.txt"
    # pb_file_path = "E:/models_dl/graph_202.pb"
    # char_map_path = "F:/data_common/chs202_code.txt"
    model = mnv2(pb_file_path, char_map_path)

    # recognize
    image_root = "./seg/*.jpg"
    # image_root = "E:/ziti_test/*.jpg"
    # image_root = "E:/seg/*.jpg"
    img_paths = glob.glob(image_root) + glob.glob(image_root.replace('.jpg', '.png')) + glob.glob(
        image_root.replace('.jpg', '.JPG')) + glob.glob(image_root.replace('.jpg', '.bmp'))
    total_time = 0
    num_img = 0
    for img_path in img_paths:
        assert img_path
        # img_path = "E:/seg/jc_177.jpg"
        num_img += 1
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        start = time.time()
        n_top = 1
        char_list, prob_list, img_nor = model.recognize(img, n_top)
        print('time: {}'.format(time.time() - start))
        for i in range(n_top):
            print("%d:%s, %.5f" % ((i + 1), char_list[i], prob_list[i]))
        # cv2.imshow("img_nor", img_nor)
        # cv2.waitKey(0)
        total_time += (time.time() - start)
    print("total num:%d, total time:%.5fs, every time:%.5fs" % (num_img, total_time, 1.0*total_time/num_img))


if __name__ == '__main__':
    main()
