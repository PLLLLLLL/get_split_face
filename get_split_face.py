# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import PIL.Image as Image
import numpy as np


# path = './result/'
def clean(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file,f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

def image_compose(k,image_names,IMAGES_PATH,IMAGE_SIZE,IMAGE_ROW,IMAGE_COLUMN,IMAGE_SAVE_PATH):
    # k = 'split函数里字典的key'  # 用来命名图片
    # image_names = dic[key]  # 字典里key键存储的图片名
    # IMAGE_SIZE = 160  # 每张小图片的大小
    # IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
    # IMAGE_COLUMN = 5  # 图片间隔，也就是合并成一张图后，一共有几列
    # IMAGE_SAVE_PATH = './cut/'  # 图片转换后的地址
    
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH + k + '.jpg')


#def split(path,IMAGES_PATH,IMAGE_SIZE,IMAGE_ROW,IMAGE_COLUMN,IMAGE_SAVE_PATH):
def split(IMAGES_PATH,IMAGE_SIZE,IMAGE_ROW,IMAGE_COLUMN,IMAGE_SAVE_PATH):
    # IMAGES_PATH = ''./result/tiny_face/''  # 图片集地址
    # #IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
    # IMAGE_SIZE = 160  # 每张小图片的大小
    # IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
    # IMAGE_COLUMN = 5  # 图片间隔，也就是合并成一张图后，一共有几列
    # IMAGE_SAVE_PATH = './cut/'  # 图片转换后的地址
    
    if not os.path.exists('./cut'):
        os.makedirs('./cut')
    else:
        clean('./cut')

    # 获取图片集地址下的所有图片名称
    #image_names = []
    dic = {}
    count = 1
    k = 0
    for name in os.listdir(IMAGES_PATH):
        if (os.path.splitext(name)[1] == '.jpg'):
            dic.setdefault(str(k), []).append(name)
            if (count % 5 == 0):
                k = k + 1
            count = count + 1


    for key in dic:
        len_val = len(dic[key])
        if len_val != 5 :
            # 使用Numpy创建一张A4(2105×1487)纸
            img = np.zeros((160,160,3), np.uint8)
            # 使用白色填充图片区域,默认为黑色
            img.fill(255)
            cv2.imwrite(IMAGES_PATH + 'white.jpg',img)
            for i in range(5-len_val):
                dic[key].append('white.jpg')
        image_names = dic[key]
        image_compose(key,image_names,IMAGES_PATH,IMAGE_SIZE,IMAGE_ROW,IMAGE_COLUMN,IMAGE_SAVE_PATH)

    print(dic)



def detect_face(image_path):
    #time_start=time.time()
    tiny_face_path = './result/tiny_face/'
    if not os.path.exists(tiny_face_path):
        os.makedirs(tiny_face_path)
    else:
        clean(tiny_face_path)
    
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
    print('detector',detector)
    img = cv2.imread(image_path)
    #print('img:',img)
    # run detector
    results = detector.detect_face(img)
    if results is not None:
        total_boxes = results[0]
        points = results[1]
        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 160, 0.37)

        for i, chip in enumerate(chips):
            cv2.imwrite(tiny_face_path + 'chip_'+str(i)+'.jpg', chip)

def detect_main(face_path): #face_path大图路径
    print('---')
    detect_face(face_path)
    split('./result/tiny_face/',160,1,5,'./cut/')
    cut_list = []

    for name in os.listdir('./cut'):
        cut_list.append('./cut/'+name)

    return cut_list


if __name__ == '__main__':
    detect_main('./test_pic/test.jpg')



