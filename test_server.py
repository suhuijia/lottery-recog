import urllib.request
import urllib.parse
import base64
url = "http://116.85.30.140:3600/v1/lottery/recognition/"
img_path = '/home/dc2-user/workspace/cp_server/load_images/2018-08-13 102727.bmp'

def face_recog():

    f = open(img_path, 'rb')
    img = base64.b64encode(f.read())

    params = {
        # 'img_url': "http://img.my.csdn.net/uploads/201212/25/1356422284_1112.jpg",
        'img_str': img,
    }

    params = urllib.parse.urlencode(params).encode(encoding='utf-8')
    request = urllib.request.Request(url=url, data=params)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib.request.urlopen(request)
    content = response.read().decode()
    if content:
        print(content)
        print(type(content))

face_recog()
