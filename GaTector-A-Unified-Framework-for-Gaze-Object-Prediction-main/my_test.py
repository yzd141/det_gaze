# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from lib.yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    id=8
    mode = "predict"
    img_path="./img/real_img/{}.jpg".format(id)

    image = Image.open(img_path)


    x,y = yolo.get_gaze(image,0)
    plot_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.circle(plot_img, (x, y), 70, [174, 19, 43], -1)
    cv2.imwrite("./img/real_img/process1/result_{}.jpg".format(id), plot_img)



