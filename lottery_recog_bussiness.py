#!/usr/bin/python
#!-*-coding:utf-8-*-
import time, os
import base64
import cv2
import numpy as np
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.options import define, options
from tornado.escape import json_decode, json_encode
from tornado.httputil import HTTPHeaders
import json
import requests
from PIL import Image
from io import BytesIO
import cp_det
from char3755_recognition import mnv2

# global net, image_mean
path_lab = "../temp_all/map.txt"
path_temp = "../temp_all/"

list_temp, list_char = cp_det.init_temp(path_lab, path_temp)


port = 3608

err_str_6001 = "JSONDecodeError"
err_str_6002 = "Invalid multipart/form-data: no final boundary"
err_str_6003 = "400: Bad Request"
err_str_6004 = "The request data is incorrect!"
err_str_6005 = "The request data format is not supported!"
err_str_6006 = "The required parameters are incorrect"
err_str_6007 = "The required header is incorrect"
err_str_6008 = "Image data error!"

def save_image_str(img_str, img_save_name):
    img_str = base64.b64decode(img_str)
    img_data = np.frombuffer(img_str, dtype='uint8')
    decimg = cv2.imdecode(img_data, 1)
    cv2.imwrite(img_save_name, decimg)
    
    
def init_server():
    global g_model
    pb_file_path = "./graph_3755.pb"
    char_map_path = "./chs3755_code.txt"
    g_model = mnv2(pb_file_path, char_map_path)


define("port", default=port, help='run a test')
class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        """
        Returns:

        """
        # print "setting headers!!!"
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        # self.set_header("Content-Type", "application/json")

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self, *args, **kwargs):
        start_t = time.time()
        save_home = "../load_images"
        save_file_name = ""

        # print(self.request.headers)
        body_data = self.request.body
        head_cont_type = self.request.headers['Content-Type']
        print(head_cont_type)
        self.set_header("Content-Type", "application/json")

        # with open('data_header_body.txt', 'a+') as wf:
        #     wf.write(str(head_cont_type))
        #     wf.write('\r\n')
        #     wf.write(str(body_data))
        #     wf.write('\r\n')

        error_response = {}
        if head_cont_type.startswith("application/x-www-form-urlencoded"):
            print(self.request.arguments.keys())

            if "img_url" in self.request.arguments:
                print(self.request.arguments["img_url"][0])
                img_url = self.request.arguments["img_url"][0]
                save_file_name = time.strftime('%Y-%m-%d %H%M%S', time.localtime(time.time())) + ".bmp"
                response = requests.get(img_url)
                image = Image.open(BytesIO(response.content))
                image.save(save_home +  "/" + save_file_name)

            elif "img_str" in self.request.arguments:
                print("img_str")
                img_str_data = self.request.arguments["img_str"][0]
                # print(img_str_data)
                # print(type(img_str_data))
                try:
                    save_file_name = time.strftime('%Y-%m-%d %H%M%S', time.localtime(time.time())) + ".bmp"
                    save_image_str(img_str_data, save_home +  "/" + save_file_name)
                except Exception as e:
                    # err_str = "The request data is incorrect!"
                    print(e)
                    error_response["error_code"] = 6004
                    # error_response["error_msg"] = 
                    error_response["error_msg"] = e
                    self.finish(json.dumps(error_response))
                    return
            else:
                # err_str = "The required parameters are incorrect"
                error_response["error_code"] = 6006
                error_response["error_msg"] = err_str_6006
                self.finish(json.dumps(error_response))
                return

        elif head_cont_type.startswith("application/json"):
            # print(body_data)
            try:
                params_content = json_decode(body_data)
                print(params_content)
                print(type(params_content))
                if "img_str" in params_content:
                    img_content = params_content["img_str"]
                    # print(img_content)
                    # print(len(img_content))
                    try:
                        save_file_name = time.strftime('%Y-%m-%d %H%M%S', time.localtime(time.time())) + ".bmp"
                        save_image_str(img_content, save_home +  "/" + save_file_name)
                    except Exception as e:
                        print(e)
                        # err_str = "The request data is incorrect!"
                        error_response["error_code"] = 6004
                        error_response['error_msg'] = err_str_6004
                        self.finish(json.dumps(error_response))
                        return

                else:
                    # err_str = "The required parameters are incorrect"
                    error_response["error_code"] = 6006
                    error_response["error_msg"] = err_str_6006
                    self.finish(json.dumps(error_response))
                    return

            except Exception as e:
                # err_str = "JSONDecodeError"
                error_response["error_code"] = 6001
                error_response["error_msg"] = err_str_6001
                self.finish(json.dumps(error_response))
                return

        elif head_cont_type.startswith("multipart/form-data"):
            sub_str = b'\r\n'
            boundary = head_cont_type.split("boundary=")[1]
            boundary_byte = bytes(boundary)
            data_list = body_data.split(boundary_byte)
            if data_list[0].startswith(sub_str):
                check_one = data_list[0].split(sub_str)[1]
            else:
                check_one = data_list[0]
            check_two = data_list[-1].split(sub_str)[0]
            check_three = data_list[-2].rsplit(sub_str, 1)[-1]

            if len(check_one) < 2 or len(check_two) < 2 or len(check_three) != 2:
                # response = self.write("Invalid multipart/form-data: no f
                # inal boundary")
                # err_str = "Invalid multipart/form-data: no final boundary"
                error_response["error_code"] = 6002
                error_response["error_msg"] = err_str_6002
                self.finish(json.dumps(error_response))
                return 

            keys = list(self.request.files.keys())
            # print(keys)

            if len(keys) == 0:
                # raise tornado.web.HTTPError(400)
                # err_str = "400: Bad Request"
                error_response["error_code"] = 6003
                error_response["error_msg"] = err_str_6003
                self.finish(json.dumps(error_response))
                return 
            else:
                if "img_str" in keys:
                    imgfile = self.request.files.get('img_str')
                    img_content = imgfile[0]['body']


                    if len(img_content) == 0:
                        # response = self.write(u"请求数据为空�?)
                        # err_str = "The request data is incorrect!"
                        error_response["error_code"] = 6004
                        error_response['error_msg'] = err_str_6004
                        self.finish(json.dumps(error_response))
                        return

                    img_post = imgfile[0].filename.split(".")[-1]
                    img_post_lower = img_post.lower()

                    if img_post_lower not in ['jpg', 'jpeg', 'png', 'bmp']:
                        # response = self.write(u"请求数据格式不支持！")
                        # err_str = "The request data format is not supported!"
                        error_response["error_code"] = 6005
                        error_response['error_msg'] = err_str_6005
                        self.finish(json.dumps(error_response))
                        return

                    save_file_name = time.strftime('%Y-%m-%d %H%M%S', time.localtime(time.time())) + '.' + img_post
                    with open(save_home +  "/" + save_file_name, 'wb') as f:
                        f.write(imgfile[0]['body'])

                else:
                    # response = self.write(u"必选参数不正确�?)
                    # err_str = "The required parameters are incorrect"
                    error_response["error_code"] = 6006
                    error_response['error_msg'] = err_str_6006
                    self.finish(json.dumps(error_response))
                    return

        else:
            # err_str = "The required header is incorrect"
            error_response["error_code"] = 6007
            error_response['error_msg'] = err_str_6007
            self.finish(json.dumps(error_response))
            return

        print(save_file_name)
        response = {}
        img_path =save_home +  "/" + save_file_name
        if not os.path.exists(img_path):
            response["is_image"] = False
            response["msg"] = "no"
        else:
            try:
                print("lottery_recog")
                img = cv2.imread(img_path)
                # print(img.shape[:])
                result = cp_det.lottery_recog(img_path, list_temp, list_char, g_model)
                print("lottery_recog end")
                response["is_image"] = True
                response["msg"] = "ok"
                response["result"] = result

            except Exception as e:
                # err_str = "Image data error!"
                error_response['error_msg'] = e
                error_response["error_code"] = 6008
                # error_response['error_msg'] = err_str_6008
                self.finish(json.dumps(error_response))
                return
        print("total time:{:.3f}s".format(time.time() - start_t))
        # print("response: ", response)
        self.finish(json.dumps(response))


if __name__ == "__main__":
    init_server()
    print("init")
    application = tornado.web.Application([
        (r"/v1/lottery/recognition/", MainHandler),
    ])
    http_server = tornado.httpserver.HTTPServer(application)
    application.listen(port)
    # http_server.bind(port)
    # http_server.start(3)
    print("Srv started at %d." % port)
    tornado.ioloop.IOLoop.instance().start()
