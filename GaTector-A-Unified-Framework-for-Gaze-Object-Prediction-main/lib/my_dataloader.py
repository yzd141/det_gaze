from random import sample, shuffle
import pickle
import os
import sys
# print(os.path)
import cv2
import numpy as np
from PIL import Image,ImageDraw
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from lib.utils.utils import cvtColor, preprocess_input
from lib import gaze_imutils
import torch
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
import json

class GaTectorDataset(Dataset):
    def __init__(self, root_dir, mat_file, input_shape, num_classes, train_mode,train,letterbox_image=False):
        super(GaTectorDataset, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.train_mode=train_mode
        self.letterbox_image=letterbox_image

        # GOO pickle
        self.output_size = 64
        self.input_size = 224
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.data=[]
        with open(mat_file, 'r') as f:
            read_data= json.load(f)
            self.gt_process(read_data)
            self.image_num = len(self.data)
        #     self.data.append(data)
        # with open(mat_file, 'r') as file:
        #     for line in file:
        #         gt_dict=self.gt_process(line)
        #         self.data.append(gt_dict)
        #         self.image_num = len(self.data)
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

    def gt_process(self,read_data):
        read_data=read_data['data']
        for one_data in read_data:
            one_data=one_data[0]
            output_dict = {}
            image_path=one_data['image']
            output_dict['filename']=image_path
            # image_path = os.path.join(self.root_dir, image_path)
            # image_path = image_path.replace('\\', '/')
            # img = Image.open(image_path)
            # draw = ImageDraw.Draw(img)
            # width, height = img.size
            #bbox


            ann=one_data['annotations']
            for one_label in ann:
                class_label=one_label['label']
                coor_label=one_label['coordinates']
                if class_label=="gang":
                    bbox=[int(coor_label['x']-coor_label['width']/2),int(coor_label['y']-coor_label['height']/2)
                        ,int(coor_label['x']+coor_label['width']/2),int(coor_label['y']+coor_label['height']/2)]
                    ann_dict = {
                        'bboxes': bbox,
                        "labels": 0
                    }
                    output_dict["ann"] = ann_dict

                    # x1, y1 = (int(coor_label['x'] - coor_label['width'] / 2), int(coor_label['y']-coor_label['height']/2))  # 左上角坐标
                    # x2, y2 = (int(coor_label['x'] +coor_label['width'] / 2), int(coor_label['y']+coor_label['height']/2))
                    # draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=5)
                    # img.save('./output_image.jpg')

                if class_label=="eye":
                    eye_x=int(coor_label['x']-coor_label['width']/2)
                    eye_y=int(coor_label['y']-coor_label['height']/2)
                    output_dict["hx"] = eye_x
                    output_dict["hy"] = eye_y

                    # draw.ellipse((eye_x - 5,eye_y - 5, eye_x + 5, eye_y + 5), fill='blue')
                    # img.save('./img/output_image.jpg')

                if class_label == "gaze":
                    gaze_x = int(coor_label['x'] - coor_label['width'] / 2)
                    gaze_y = int(coor_label['y'] - coor_label['height'] / 2)
                    output_dict["gaze_cx"] = gaze_x
                    output_dict["gaze_cy"] = gaze_y
            self.data.append(output_dict)

    def __len__(self):
        return self.image_num

    def __getitem__(self, index):
        index = index % self.image_num


        # GOO pickle

        data = self.data[index]
        # data_split_list=data.split(" ")
        # image_path=data_split_list[0]
        image_path = data['filename']
        image_path = os.path.join(self.root_dir, image_path)
        image_path = image_path.replace('\\', '/')
        # gt_box_idx = data['gazeIdx']
        # Goo gt_box
        # box=[float(data_split_list[ii]) for ii in range(2,len(data_split_list))]
        if self.train_mode==0:
            gt_bboxes = np.copy( np.array(data['ann']['bboxes']).reshape((1,4)))
            gt_labels = np.copy( np.array(data['ann']['labels']).reshape((1,1)))
        if self.train_mode==1:
            gt_bboxes = np.copy(data['ann']['bboxes'])/ [640, 480, 640, 480] * [1920, 1080, 1920, 1080]
            gt_labels = np.copy(data['ann']['labels'])

        # gt_labels = gt_labels[..., np.newaxis]
        bbox = np.append(gt_bboxes, gt_labels, axis=1)
        # bbox = np.append(gt_bboxes, gt_labels)
        box = bbox.astype(np.int32)

        # gaze_gt_box = box[gt_box_idx]
        # gaze_gt_box = gaze_gt_box[np.newaxis, :]

        # GOO
        img = Image.open(image_path)
        img = img.convert('RGB')
        width, height = img.size

        if self.letterbox_image:
            scale = min(self.input_size /width, self.input_size /height)
            x_offset = (self.input_size- width*scale) / 2
            y_offset = (self.input_size- height*scale) / 2
            eye_px = data['hx']*scale+x_offset
            eye_py = data['hy']*scale+y_offset
            gaze_px = data['gaze_cx']*scale+x_offset
            gaze_py = data['gaze_cy']*scale+y_offset
        else:

            x_offset = 0
            y_offset=0




        eye = [float(data['hx']) / width, float(data['hy']) / height]
        gaze = [float(data['gaze_cx']) / width, float(data['gaze_cy']) / height]
        # eye = [float(data['hx']+x_offset) / width, float(data['hy']+y_offset) / height]
        # gaze = [float(data['gaze_cx']+x_offset) / width, float(data['gaze_cy']+y_offset) / height]

        gaze_x, gaze_y = gaze
        eye_x, eye_y = eye

        k = 0.1
        x_min = (eye_x - 0.15) * width
        y_min = (eye_y - 0.15) * height
        x_max = (eye_x + 0.15) * width
        y_max = (eye_y + 0.15) * height
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max < 0:
            x_max = 0
        if y_max < 0:
            y_max = 0
        # x_min -= k * abs(x_max - x_min)
        # y_min -= k * abs(y_max - y_min)
        # x_max += k * abs(x_max - x_min)
        # y_max += k * abs(y_max - y_min)
        # x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        # if self.train:
            # data augmentation
            # Jitter (expansion-only) bounding box size
            # if np.random.random_sample() <= 0.5:
            #     k = np.random.random_sample() * 0.2
            #     x_min -= k * abs(x_max - x_min)
            #     y_min -= k * abs(y_max - y_min)
            #     x_max += k * abs(x_max - x_min)
            #     y_max += k * abs(y_max - y_min)

            # Random Crop
            # if np.random.random_sample() <= 0.5:
            #     # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
            #     crop_x_min = np.min([gaze_x * width, x_min, x_max])
            #     crop_y_min = np.min([gaze_y * height, y_min, y_max])
            #     crop_x_max = np.max([gaze_x * width, x_min, x_max])
            #     crop_y_max = np.max([gaze_y * height, y_min, y_max])
            #
            #     # Randomly select a random top left corner
            #     if crop_x_min >= 0:
            #         crop_x_min = np.random.uniform(0, crop_x_min)
            #     if crop_y_min >= 0:
            #         crop_y_min = np.random.uniform(0, crop_y_min)
            #
            #     # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
            #     crop_width_min = crop_x_max - crop_x_min
            #     crop_height_min = crop_y_max - crop_y_min
            #     crop_width_max = width - crop_x_min
            #     crop_height_max = height - crop_y_min
            #     # Randomly select a width and a height
            #     crop_width = np.random.uniform(crop_width_min, crop_width_max)
            #     crop_height = np.random.uniform(crop_height_min, crop_height_max)
            #
            #     # Crop it
            #     img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
            #
            #     # Record the crop's (x, y) offset
            #     offset_x, offset_y = crop_x_min, crop_y_min
            #
            #     # convert coordinates into the cropped frame
            #     x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
            #     # if gaze_inside:
            #     gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
            #                      (gaze_y * height - offset_y) / float(crop_height)
            #
            #     width, height = crop_width, crop_height
            #
            #     box[:, [0, 2]] = box[:, [0, 2]] - crop_x_min
            #     box[:, [1, 3]] = box[:, [1, 3]] - crop_y_min
            #
            #     # operate gt_box
            #     gaze_gt_box[:, [0, 2]] = gaze_gt_box[:, [0, 2]] - crop_x_min
            #     gaze_gt_box[:, [1, 3]] = gaze_gt_box[:, [1, 3]] - crop_y_min

            # Random flip
            # if np.random.random_sample() <= 0.5:
            #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #     x_max_2 = width - x_min
            #     x_min_2 = width - x_max
            #     x_max = x_max_2
            #     x_min = x_min_2
            #     gaze_x = 1 - gaze_x
            #     box[:, [0, 2]] = width - box[:, [2, 0]]

            # # Random color change
            # if np.random.random_sample() <= 0.5:
            #     img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
            #     img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
            #     img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))
            #
            # # Random color change
            # if np.random.random_sample() <= 0.5:
            #     img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
            #     img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
            #     img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head_channel = gaze_imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                          resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        face.save("./face_img.jpg")
        face = face.resize((self.input_shape), Image.BICUBIC)
        face = np.transpose(preprocess_input(np.array(face, dtype=np.float32)), (2, 0, 1))
        face = torch.Tensor(face)
        face = self.transform(face)
        img= self.resize_image(img, (self.input_size, self.input_size))
        # draw = ImageDraw.Draw(img)
        img = img.resize((self.input_shape), Image.BICUBIC)
        img = np.transpose(preprocess_input(np.array(img, dtype=np.float32)), (2, 0, 1))
        img = torch.Tensor(img)
        img = self.transform(img)

        # Bbox deal
        if self.letterbox_image:
            box[:, [0, 2]] = box[:, [0, 2]]* scale+x_offset
            box[:, [1, 3]] = box[:, [1, 3]]* scale+y_offset
        else:
            box[:, [0, 2]] = box[:, [0, 2]] * self.input_size / width
            box[:, [1, 3]] = box[:, [1, 3]] * self.input_size / height

        # operate_gt_box
        # gaze_gt_box[:, [0, 2]] = gaze_gt_box[:, [0, 2]] * self.input_size / width
        # gaze_gt_box[:, [1, 3]] = gaze_gt_box[:, [1, 3]] * self.input_size / height
        # gaze_gt_box=np.copy(box)
        gaze_gt_box=np.copy(box)

        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > self.input_size] = self.input_size
        box[:, 3][box[:, 3] > self.input_size] = self.input_size
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]

        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

        if self.letterbox_image:
            eye_x =eye_px
            eye_y = eye_py
            gaze_x = float(gaze_px) / self.input_size
            gaze_y = float(gaze_py) / self.input_size
            eye=[float(eye_x)/self.input_size,float(eye_y)/self.input_size]
            gaze= [gaze_x, gaze_y]

        # generate the heatmap used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        gaze_heatmap = gaze_imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                   3,
                                                   type='Gaussian')


        # rec=(int(self.input_size*(box[0,0]-box[0,2]/2)),int(self.input_size*(box[0,1]-box[0,3]/2)),
        #      int(self.input_size*(box[0,0]+box[0,2]/2)),int(self.input_size*(box[0,1]+box[0,3]/2)))
        # draw.rectangle(rec, outline='red', width=3)
        # draw.ellipse((eye_x- 2, eye_y-2, eye_x+2, eye_y+2), fill='blue')
        # draw.ellipse((gaze_x - 2, gaze_y - 2, gaze_x + 2, gaze_y + 2), fill='yellow')
        # img.save('visi_img.jpg')

        face = np.array(face, dtype=np.float32)
        img = np.array(img, dtype=np.float32)
        head_channel = np.array(head_channel, dtype=np.float32)
        gaze_heatmap = np.array(gaze_heatmap, dtype=np.float32)

        return img, box, face, head_channel, gaze_heatmap, eye, gaze, gaze_gt_box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def resize_image(self,image, size):
        iw, ih = image.size
        w, h = size
        if self.letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)

        return new_image


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
        # face_img=self.preprocess_image(img,(eye_point_x,eye_point_y))
        # return face_img,head
        return eye_point_x,eye_point_y

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

        scale = min(w / iw, h / ih)
        h_fill=(h/scale-ih)/2
        w_fill = (w / scale-iw) / 2

        gaze_x = int(x / scale+w_fill)
        gaze_y = int(y / scale -h_fill)


        # gaze_x=int(x/scale)
        # gaze_y=int(y/scale)
        if gaze_x < 0:
            gaze_x = 0
        if gaze_y < 0:
            gaze_y = 0
        if gaze_x > iw:
            gaze_x = iw
        if gaze_y > ih:
            gaze_y = ih

        return gaze_x,gaze_y








# DataLoader_collate_fn
def gatector_dataset_collate(batch):
    images = []
    bboxes = []
    face = []
    head_channel = []
    gaze_heatmap = []
    eye = []
    gaze = []
    gt_boxes = []
    for img, box, face_, head, heatmap, eyes, gazes, gt_box in batch:
        images.append(img)
        bboxes.append(box)
        face.append(face_)
        head_channel.append(head)
        gaze_heatmap.append(heatmap)
        eye.append(eyes)
        gaze.append(gazes)
        gt_boxes.append(gt_box)
    images = np.array(images)
    face = np.array(face)
    head_channel = np.array(head_channel)
    gaze_heatmap = np.array(gaze_heatmap)
    eye = np.array(eye)
    gaze = np.array(gaze)
    gt_boxes = np.array(gt_boxes)
    return images, bboxes, face, head_channel, gaze_heatmap, eye, gaze, gt_boxes


