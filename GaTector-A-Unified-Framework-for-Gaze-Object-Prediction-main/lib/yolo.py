import colorsys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from PIL import ImageDraw, ImageFont
from torchvision import transforms
from lib.nets.gatector import GaTectorBody
# from lib.nets.gatector_my import GaTectorBody

from lib.utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from lib.utils.utils_bbox import DecodeBox
from facenet_pytorch import MTCNN

class YOLO(object):
    _defaults = {
        #Please modify it to the training mode
        "train_mode":0,
        #Please change the model path
        "model_path": os.path.abspath(os.path.join(os.getcwd(),
         '..')) + '/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction-main/model/ep013-loss30.587-val_loss79.297.pth',

        # "model_path": os.path.abspath(os.path.join(os.getcwd(),'..'))+'/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction-main/model/ep049-loss386.512-val_loss375.549.pth',
        # "model_path": os.path.abspath(os.path.join(os.getcwd(),
        #  '..')) + '/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction-main/model/ep003-loss3792.261-val_loss3458.602.pth',

        # "classes_path": os.path.abspath(os.path.join(os.getcwd(),
        # '..')) + '/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction-main/data/anchors/voc_classes.txt',

        "classes_path": os.path.abspath(os.path.join(os.getcwd(),'..'))+'/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction-main/txt/classes.txt',
        # ---------------------------------------------------------------------#
        # anchors_path represents the txt file corresponding to the a priori box, which is generally not modified.
        # anchors_mask is used to help the code find the corresponding a priori box, generally not modified.
        # ---------------------------------------------------------------------#
        "anchors_path": os.path.abspath(os.path.join(os.getcwd(),'..'))+'/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction-main/data/anchors/yolo_anchors.txt',
        "anchors_mask": [[0, 1, 2]],

        "anchors_path_3":os.path.abspath(os.path.join(os.getcwd(),'..'))+ '/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction-main/data/anchors/yolo_anchors_3.txt',
        "anchors_mask_3": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # ---------------------------------------------------------------------#
        #   The size of the input image must be a multiple of 32.
        # ---------------------------------------------------------------------#
        "input_shape": [224, 224],
        # ---------------------------------------------------------------------#
        #   Only prediction boxes with a score greater than the confidence level will be retained
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   The size of nms_iou used for non-maximum suppression
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   This variable is used to control whether to use letterbox_image to resize the input image without distortion
        # ---------------------------------------------------------------------#
        # "letterbox_image": True,
        "letterbox_image": False,

        # "cuda": True,
        "cuda": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   Initialize YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   Get the number of types and a priori boxes
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        if self.train_mode==0:
            self.anchors, self.num_anchors = get_anchors(self.anchors_path)
            self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                       self.anchors_mask)
        if self.train_mode == 1:
            self.anchors, self.num_anchors = get_anchors(self.anchors_path_3)
            self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                       self.anchors_mask_3)

        # ---------------------------------------------------#
        #   Set different colors for the picture frame
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    # ---------------------------------------------------#
    #   Generative model
    # ---------------------------------------------------#
    def generate(self):
        if self.train_mode==0:
            self.net = GaTectorBody(self.anchors_mask, self.num_classes, self.train_mode)
        if self.train_mode == 1:
            self.net = GaTectorBody(self.anchors_mask_3, self.num_classes, self.train_mode)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.net = nn.DataParallel(self.net)
        # self.net = self.net.cuda()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image,train_mode):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        image_data = torch.Tensor(image_data)
        image_data = self.transform(image_data)
        image_data = np.array(image_data, dtype=np.float32)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                face = images
                head = np.ones([1, 1, 224, 224])
                head = torch.from_numpy(head).type(torch.FloatTensor).cuda()
            outputs = self.net(images, head, face,train_mode)
            outputs = outputs[1:]
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   Set font and border thickness
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # ---------------------------------------------------------#
        #   Image drawing
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   Input images into the network for prediction
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   Stack the prediction boxes, and then perform non-maximum suppression
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   Input images into the network for prediction
                # ---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                # ---------------------------------------------------------#
                #   Stack the prediction boxes, and then perform non-maximum suppression
                # ---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, dataset,train_mode,image_id, image, class_names, map_out_path):
        if dataset==1:
            f = open(os.path.join(map_out_path, "detection-real-results/" + image_id + ".txt"), "w")
        if dataset==0:
            f = open(os.path.join(map_out_path, "detection-synth-results/" + image_id + ".txt"), "w")
        
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent the grayscale image from reporting errors during prediction.
        # The code only supports the prediction of RGB images, all other types of images will be converted into RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        # Add gray bars to the image to achieve undistorted resize
        # You can also directly resize for identification
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   Add the batch_size dimension
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        image_data = torch.Tensor(image_data)
        image_data = self.transform(image_data)
        image_data = np.array(image_data, dtype=np.float32)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                # For mAP, face and head are useless
                face = images
                head = np.ones([1, 1, 224, 224])
                head = torch.from_numpy(head).type(torch.FloatTensor).cuda()
            # ---------------------------------------------------------#
            #   Input images into the network for prediction
            # ---------------------------------------------------------#
            outputs = self.net(images, head, face,train_mode)
            outputs = outputs[1:]
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   Stack the prediction boxes, and then perform non-maximum suppression
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            if dataset == 1:
                f.write("%s %s %s %s %s %s\n" % (
                    predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
            if dataset == 0:
                f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        return

    def get_gaze(self,image, train_mode):
        # mtcnn = MTCNN()
        # face_img= mtcnn.detect(image)

        image_shape = np.array(np.shape(image)[0:2])

        image_data1 = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # image = resize_image(image, (self.input_shape[1], self.input_shape[0]), True)

        face,head=self.img_crop(image_data1)

        # face.save("./img/real_img/face_save.jpg")

        # image_data = cvtColor(image)
        face.save("./img/real_img/face_save.jpg")

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data1, dtype='float32')), (2, 0, 1)), 0)
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        image_data = torch.Tensor(image_data)
        images = self.transform(image_data)

        head=torch.from_numpy(head).type(torch.FloatTensor)

        # face = resize_image(face, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        face = resize_image(face, (self.input_shape[1], self.input_shape[0]), False)
        face = np.expand_dims(np.transpose(preprocess_input(np.array(face, dtype='float32')), (2, 0, 1)), 0)
        face = torch.Tensor(face)
        face = self.transform(face)



        self.net.eval()
        with torch.no_grad():
            outputs = self.net(images, head, face, train_mode)
            gaze_heatmap=outputs[0].squeeze()
            outputs_bbox = outputs[1:]
            outputs_bbox = self.bbox_util.decode_box(outputs_bbox)
            # ---------------------------------------------------------#
            #   Stack the prediction boxes, and then perform non-maximum suppression
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs_bbox, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            #计算gaze点
            x1,y1=self.heatmap2gaze(gaze_heatmap,image_shape)

            # plot_img = cv2.cvtColor(np.array(image_data1), cv2.COLOR_RGB2BGR)
            # cv2.circle(plot_img, (x1, y1), 2, [255,0,0], -1)
            # cv2.imwrite("./img/real_img/process1/gaze_resize1.jpg", plot_img)

            # image_data1






        #   Add the batch_size dimension
        # ---------------------------------------------------------#
        # print(1)
        return x1,y1

    def img_crop(self,img):
        width, height = img.size[:2]
        mtcnn = MTCNN()
        face_rect = mtcnn.detect(img)
        all_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        x1, y1, x2, y2 = tuple(map(int, face_rect[0][0]))
        head = np.zeros([1, 1, 224, 224])
        head[y1:y2, x1:x2]=1
        h = y2 - y1
        face_img = all_img[y1:y2, x1:x2]
        eye_point_x = (x1 + x2) / (2 * width)
        eye_point_y = (y1 + h / 3) / height
        face_img=self.preprocess_image(img,(eye_point_x,eye_point_y))
        return face_img,head

    def heatmap2gaze(self,heatmap, img_shape):
        ih,iw=img_shape[0],img_shape[1]
        scale1=self.input_shape[0]/heatmap.size()[0]

        max_index = torch.argmax(heatmap)
        # y = max_index // heatmap.shape[1]
        y=torch.div(max_index, heatmap.shape[1], rounding_mode='trunc')
        x = max_index % heatmap.shape[1]
        x=x*scale1
        y=y*scale1

        w=self.input_shape[0]
        h=self.input_shape[1]
        # gaze_x=int(x)
        # gaze_y = int(y)
        if self.letterbox_image:
            # scale = min(w / iw, h / ih)
            # h_fill=(h/scale-ih)/2
            # w_fill = (w / scale-iw) / 2
            #
            # gaze_x = int(x / scale-w_fill)
            # gaze_y = int(y / scale-h_fill)
            scale = min(w / iw, h / ih)
            h_fill=(h-ih*scale)/2
            w_fill = (w-scale*iw) / 2

            gaze_x = int((x-w_fill)/scale)
            gaze_y = int((y-h_fill)/scale)
        else:
            scale_x=iw / w
            scale_y=ih / h
            gaze_x = int(x * scale_x )
            gaze_y = int(y *scale_y)


        # if gaze_x < 0:
        #     gaze_x = 0
        # if gaze_y < 0:
        #     gaze_y = 0
        # if gaze_x > iw:
        #     gaze_x = iw
        # if gaze_y > ih:
        #     gaze_y = ih

        return gaze_x,gaze_y


    def preprocess_image(self,image, eye):

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # crop face
        x_c, y_c = eye
        x_0 = x_c - 0.15
        y_0 = y_c - 0.15
        x_1 = x_c + 0.15
        y_1 = y_c + 0.15
        if x_0 < 0:
            x_0 = 0
        if y_0 < 0:
            y_0 = 0
        if x_1 > 1:
            x_1 = 1
        if y_1 > 1:
            y_1 = 1

        h, w = image.shape[:2]
        face_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w), :]
        # process face_image for face net
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)


        return face_image




