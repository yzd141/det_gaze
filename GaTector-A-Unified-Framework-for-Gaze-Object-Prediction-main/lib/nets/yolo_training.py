import torch
import torch.nn as nn
import math
import numpy as np


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothing=0):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.ignore_threshold = 0.5
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_ciou(self, b1, b2):
        """
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # ----------------------------------------------------#
        #   Find the upper left corner and lower right corner of the prediction box
        # ----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # ----------------------------------------------------#
        #   Find the upper left corner and lower right corner of the real frame
        # ----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # ----------------------------------------------------#
        #   Find all the iou of the true box and the predicted box
        # ----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)

        # ----------------------------------------------------#
        #   Gaps in computing centers
        # ----------------------------------------------------#
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

        # ----------------------------------------------------#
        #  Find the upper left and lower right corners of the smallest box that wraps the two boxes
        # ----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # ----------------------------------------------------#
        #   Calculate diagonal distance
        # ----------------------------------------------------#
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
            b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou

    # ---------------------------------------------------#
    #   Smooth label
    # ---------------------------------------------------#
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None):
        # --------------------------------#
        #   Get the number of pictures, the height and width of the feature layer
        # --------------------------------#
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()

        # -----------------------------------------------#
        #   Adjustment parameters for the center position of the a priori box
        # -----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # -----------------------------------------------#
        #   The width and height adjustment parameters of the a priori box
        # -----------------------------------------------#
        w = prediction[..., 2]
        h = prediction[..., 3]
        # -----------------------------------------------#
        #   Get confidence, whether there is an object
        # -----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        # -----------------------------------------------#
        #   Category confidence
        # -----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # -----------------------------------------------#
        #   Get the prediction results that the network should have
        # -----------------------------------------------#
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.cuda()
            noobj_mask = noobj_mask.cuda()
            box_loss_scale = box_loss_scale.cuda()

        box_loss_scale = 2 - box_loss_scale

        ciou = (1 - self.box_ciou(pred_boxes[y_true[..., 4] == 1], y_true[..., :4][y_true[..., 4] == 1])) * \
               box_loss_scale[y_true[..., 4] == 1]
        loss_loc = torch.sum(ciou)
        # -----------------------------------------------------------#
        #   Calculate the loss of confidence
        # -----------------------------------------------------------#
        loss_conf = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                    torch.sum(self.BCELoss(conf, y_true[..., 4]) * noobj_mask)

        loss_cls = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1],
                                          self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing,
                                                             self.num_classes)))

        loss = loss_loc + loss_conf + loss_cls
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        return loss, num_pos

    def calculate_iou(self, _box_a, _box_b):
        # -----------------------------------------------------------#
        #   Calculate the upper left and lower right corners of the real frame
        # -----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2

        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        # -----------------------------------------------------------#
        #   A is the number of real boxes, B is the number of a priori boxes
        # -----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        # -----------------------------------------------------------#
        #   Calculate the area of intersection
        # -----------------------------------------------------------#
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        # -----------------------------------------------------------#
        #   Calculate the respective areas of the predicted box and the real box
        # -----------------------------------------------------------#
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        x=inter / union
        return inter / union  # [A,B]

    def get_target(self, l, targets, anchors, in_h, in_w):
        # -----------------------------------------------------#
        #   Calculate how many pictures there are
        # -----------------------------------------------------#
        bs = len(targets)
        # -----------------------------------------------------#
        #   Used to select which a priori boxes do not contain objects
        # -----------------------------------------------------#
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        # -----------------------------------------------------#
        #   Let the network pay more attention to small goals
        # -----------------------------------------------------#
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])

            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))

            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))

            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                # ----------------------------------------#
                #  Determine which a priori box of the current feature point this a priori box is
                # ----------------------------------------#
                k = self.anchors_mask[l].index(best_n)
                # ----------------------------------------#
                #   Get which grid point the real frame belongs to
                # ----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                # ----------------------------------------#
                #   Type of real frame
                # ----------------------------------------#
                c = batch_target[t, 4].long()

                noobj_mask[b, k, j, i] = 0
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        # -----------------------------------------------------#
        #   Calculate how many pictures there are
        # -----------------------------------------------------#
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # -----------------------------------------------------#
        #   Generate grid, a priori box center, upper left corner of the grid
        # -----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # Generate the width and height of the a priori box
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # -------------------------------------------------------#
        #   Calculate the adjusted a priori box center and width and height
        # -------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)
        for b in range(bs):
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])

                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4]
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)
