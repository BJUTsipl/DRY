#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from ast import Import
from cv2 import threshold
from torch import float32
import torch
import torch.nn as nn

import numpy as np
import imageio

from .yolo_head import YOLOXHead
from .yolo_neck import YOLOXNeck
from .yolo_pafpn import YOLOPAFPN
from .darknet import CSPDarknet # backbone

from .GlobalDiscriminator import YOLOX_GA
from .MINE import MINELoss

class YOLOX_da(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, 
            backbone=None, 
            neck=None, 
            head=None, 
            dis_ga=None, 
            func_forward='default',
            args=None
            ):
        super().__init__()
        # if backbone is None:
        #     backbone = CSPDarknet(dep_mul=1.0, wid_mul=1.0, depthwise=False, act='silu')
        # if neck is None:
        #     neck = YOLOXNeck()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        if dis_ga is None:
            dis_ga = YOLOX_GA()

        self.backbone = backbone
        # self.neck = neck
        self.head = head
        self.discriminator_ga = dis_ga
        
        self.func_forward = func_forward
        if self.func_forward=='GDIFD_with_SpatialAttention' or self.func_forward=='DRL':
            from .GDIFD import GDIFD
            from .GlobalDiscriminator import YOLOX_GA
            self.GDIFD = GDIFD(depth=args['depth'], width=args['width'],num_convs=5)
            self.discriminator_Sdi = YOLOX_GA(depth=args['depth'], width=args['width'],grad_reverse_lambda=0.02)
            self.discriminator_Sds = YOLOX_GA(depth=args['depth'], width=args['width'],grad_reverse_lambda=0.02)
        
        if self.func_forward=='DRL':
            self.in_features=("P3", "P4", "P5")
            self.in_channels=[256, 512, 1024]
            for i in range(len(self.in_features)):
                self.add_module(
                    "MINELoss_{}".format(self.in_features[i]),
                    MINELoss(in_channels=int(self.in_channels[i] * args['width']))
                    )

        self.ga_dis_lambda = 0.1

    def forward(self, x, targets=None, with_DA=False):
        if self.func_forward=='DRL':
            return self.forward_DRL(x, targets, with_DA)
        if self.func_forward=='GDIFD_with_SpatialAttention':
            return self.forward_GDIFD_with_SpatialAttention(x, targets, with_DA)
        if self.func_forward=='ga_attention':
            return self.forward_ga_attention(x, targets, with_DA)
        else:
            return self.forward_ga(x, targets, with_DA)

    def forward_ga(self, x, targets=None, with_DA=False):
        # fpn output content features of [dark3, dark4, dark5]
        # backbone_outs = self.backbone(x)
        # fpn_outs = self.neck(backbone_outs)
        if with_DA == False:
            fpn_outs_s = self.backbone(x)
            if self.training:
                assert targets is not None
                det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs_s, targets, x
                )
                outputs = {
                    "total_loss": det_loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                outputs = self.head(fpn_outs_s)
        else:
            images_s, images_t = x
            fpn_outs_s = self.backbone(images_s)
            fpn_outs_t = self.backbone(images_t)
            
            if self.training:
                assert targets is not None
                det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs_s, targets, images_s
                )

                ga_loss, ga_loss_dict = self.discriminator_ga((fpn_outs_s, fpn_outs_t))
            
                total_loss = det_loss + self.ga_dis_lambda * ga_loss
                outputs = {
                    "total_loss": total_loss,
                    "ga_loss": ga_loss,
                    "det_loss": det_loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
                outputs.update(ga_loss_dict)
            else:
                outputs = self.head(fpn_outs_s)

        return outputs

    def forward_ga_attention(self, x, targets=None, with_DA=False):
        # fpn output content features of [dark3, dark4, dark5]
        # backbone_outs = self.backbone(x)
        # fpn_outs = self.neck(backbone_outs)
        if with_DA == False:
            fpn_outs_s = self.backbone(x)
            if self.training:
                assert targets is not None
                det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs_s, targets, x
                )
                outputs = {
                    "total_loss": det_loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                outputs = self.head(fpn_outs_s)
        else:
            images_s, images_t = x
            fpn_outs_s = self.backbone(images_s)
            fpn_outs_t = self.backbone(images_t)
            
            if self.training:
                assert targets is not None
                # det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                #     fpn_outs_s, targets, images_s
                # )
                (det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), head_da_outputs_s = self.head(
                    fpn_outs_s, targets, images_s, True
                )
                _, head_da_outputs_t = self.head(
                    fpn_outs_t, None, images_t, True
                )

                ga_loss, ga_loss_dict = self.discriminator_ga((fpn_outs_s, fpn_outs_t), head_features=(head_da_outputs_s,head_da_outputs_t))
            
                total_loss = det_loss + self.ga_dis_lambda * ga_loss
                outputs = {
                    "total_loss": total_loss,
                    "ga_loss": ga_loss,
                    "det_loss": det_loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
                outputs.update(ga_loss_dict)
            else:
                outputs = self.head(fpn_outs_s)
                
        return outputs

    def forward_GDIFD_with_SpatialAttention(self, x, targets=None, with_DA=False):
        # fpn output content features of [dark3, dark4, dark5]
        # backbone_outs = self.backbone(x)
        # fpn_outs = self.neck(backbone_outs)
        if with_DA == False:
            fpn_outs_s = self.backbone(x)
            if self.training:
                assert targets is not None
                det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs_s, targets, x
                )
                outputs = {
                    "total_loss": det_loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                outputs = self.head(fpn_outs_s)
        else:
            images_s, images_t = x
            fpn_outs_s = self.backbone(images_s)
            fpn_outs_t = self.backbone(images_t)

            loss_dict_gate = {}
            Sdi_s,Sds_s,loss_gate_s,loss_dict_gate_s = self.GDIFD(fpn_outs_s)
            Sdi_t,Sds_t,loss_gate_t,loss_dict_gate_t = self.GDIFD(fpn_outs_t)
            loss_gate = loss_gate_s + loss_gate_t
            for key in loss_dict_gate_s.keys():
                loss_dict_gate[key] = loss_dict_gate_s[key] + loss_dict_gate_t[key]
            
            features_Sdi_s = []
            features_Sds_s = []
            for i in range(len(Sdi_s)):
                features_Sdi_s.append((1+Sdi_s[i].expand_as(fpn_outs_s[i]))*fpn_outs_s[i])
                features_Sds_s.append(Sds_s[i].expand_as(fpn_outs_s[i])*fpn_outs_s[i])
            features_Sdi_t = []
            features_Sds_t = []
            for i in range(len(Sdi_t)):
                features_Sdi_t.append((1+Sdi_t[i].expand_as(fpn_outs_t[i]))*fpn_outs_t[i])
                features_Sds_t.append(Sds_t[i].expand_as(fpn_outs_t[i])*fpn_outs_t[i])
            
            if self.training:
                assert targets is not None
                # det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                #     fpn_outs_s, targets, images_s
                # )

                (det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), head_da_outputs_s = self.head(
                    features_Sdi_s, targets, images_s, True
                )
                _, head_da_outputs_t = self.head(
                    features_Sdi_t, None, images_t, True
                )

                loss_AMFA, loss_dict_AMFA = self.discriminator_ga(
                    (features_Sdi_s, features_Sdi_t), 
                    head_features=(head_da_outputs_s,head_da_outputs_t),
                    loss_dict_prefix="loss_AMFA"
                    )
                loss_Sdi, loss_dict_Sdi = self.discriminator_Sdi(
                    (features_Sdi_s, features_Sdi_t),
                    loss_dict_prefix="loss_Sdi"
                    )
                loss_Sds, loss_dict_Sds = self.discriminator_Sds(
                    (features_Sds_s, features_Sds_t),
                    loss_dict_prefix="loss_Sds"
                    )
            
                total_loss = det_loss + self.ga_dis_lambda * loss_AMFA + 0.1 * loss_Sdi + 0.1 * loss_gate + 0.1 * loss_Sds
                outputs = {
                    "total_loss": total_loss,
                    "AMFA_loss": loss_AMFA,
                    "Sdi_loss": loss_Sdi,
                    "Sds_loss": loss_Sds,
                    "gate_loss": loss_gate,
                    "det_loss": det_loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
                outputs.update(loss_dict_AMFA)
                outputs.update(loss_dict_gate)
                outputs.update(loss_dict_Sdi)
                outputs.update(loss_dict_Sds)
            else:
                outputs = self.head(fpn_outs_s)
                
        return outputs

    def calculate_MI_loss(self, inputs_di, inputs_ds):
        # MI_loss = (self.MINELoss_P3(inputs[0]),self.MINELoss_P4(inputs[1]),self.MINELoss_P5(inputs[2]))
        # loss_dict_MI = {}
        # for i in range(len(MI_loss)):
        #     loss_dict_MI['loss_gate_{}'.format(self.in_features[i])] = MI_loss[i]
        # return loss_dict_MI
        return [
            -1.0 * self.MINELoss_P3(inputs_di[0][:,:,0,0],inputs_ds[0][:,:,0,0]),
            -1.0 * self.MINELoss_P4(inputs_di[1][:,:,0,0],inputs_ds[1][:,:,0,0]),
            -1.0 * self.MINELoss_P5(inputs_di[2][:,:,0,0],inputs_ds[2][:,:,0,0])
            ]

    def forward_DRL(self, x, targets=None, with_DA=False):
        # fpn output content features of [dark3, dark4, dark5]
        # backbone_outs = self.backbone(x)
        # fpn_outs = self.neck(backbone_outs)
        if with_DA == False:
            fpn_outs_s = self.backbone(x)
            if self.training:
                assert targets is not None
                det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs_s, targets, x
                )
                outputs = {
                    "total_loss": det_loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            # else:
            #     outputs = self.head(fpn_outs_s)
            # for att map vis
            else:
                # outputs = self.head(fpn_outs_s)
                outputs, head_da_outputs_t = self.head(
                    fpn_outs_s, None, x, True
                )
                vis_path='./vis_output/'
                out_filename = 'input_img.pt'
                torch.save(x.cpu(), vis_path+out_filename)
                # input_img = x.cpu().numpy()
                # out_filename = 'input_img.bin'
                # input_img.tofile(vis_path+out_filename)
                out_filename = 'fpn_outs_s.pt'
                torch.save(fpn_outs_s, vis_path+out_filename)
                # for i, arr in enumerate(fpn_outs_s):
                #     out = arr.cpu().numpy()
                #     out_filename = 'fpn_outs_scale{}.bin'.format(i)
                #     out.tofile(vis_path+out_filename)
                out_filename = 'head_da_outputs_t.pt'
                torch.save(head_da_outputs_t, vis_path+out_filename)
                # for i, arr in enumerate(head_da_outputs_t):
                #     out = arr.cpu().numpy()
                #     out_filename = 'predict_results_scale{}.bin'.format(i)
                #     out.tofile(vis_path+out_filename)

        else:
            images_s, images_t = x
            fpn_outs_s = self.backbone(images_s)
            fpn_outs_t = self.backbone(images_t)

            loss_dict_gate = {}
            Sdi_s,Sds_s,loss_gate_s,loss_dict_gate_s = self.GDIFD(fpn_outs_s)
            Sdi_t,Sds_t,loss_gate_t,loss_dict_gate_t = self.GDIFD(fpn_outs_t)
            loss_gate = loss_gate_s + loss_gate_t
            for key in loss_dict_gate_s.keys():
                loss_dict_gate[key] = loss_dict_gate_s[key] + loss_dict_gate_t[key]
            
            features_Sdi_s = []
            features_Sds_s = []
            for i in range(len(Sdi_s)):
                features_Sdi_s.append((1+Sdi_s[i].expand_as(fpn_outs_s[i]))*fpn_outs_s[i])
                features_Sds_s.append(Sds_s[i].expand_as(fpn_outs_s[i])*fpn_outs_s[i])
            features_Sdi_t = []
            features_Sds_t = []
            for i in range(len(Sdi_t)):
                features_Sdi_t.append((1+Sdi_t[i].expand_as(fpn_outs_t[i]))*fpn_outs_t[i])
                features_Sds_t.append(Sds_t[i].expand_as(fpn_outs_t[i])*fpn_outs_t[i])
            
            #Mutual Information
            MI_losss_s = self.calculate_MI_loss(Sdi_s, Sds_s)
            MI_losss_t = self.calculate_MI_loss(Sdi_t, Sds_t)

            loss_dict_MI = {}
            for i in range(len(MI_losss_s)):
                loss_dict_MI['loss_MI_{}'.format(self.in_features[i])] = MI_losss_s[i] + MI_losss_t[i]

            loss_MI = sum(loss for loss in loss_dict_MI.values())

            if self.training:
                assert targets is not None
                # det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                #     fpn_outs_s, targets, images_s
                # )
                (det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), head_da_outputs_s = self.head(
                    features_Sdi_s, targets, images_s, True
                )
                _, head_da_outputs_t = self.head(
                    features_Sdi_t, None, images_t, True
                )

                loss_AMFA, loss_dict_AMFA = self.discriminator_ga(
                    (features_Sdi_s, features_Sdi_t), 
                    head_features=(head_da_outputs_s,head_da_outputs_t),
                    loss_dict_prefix="loss_AMFA"
                    )
                loss_Sdi, loss_dict_Sdi = self.discriminator_Sdi(
                    (features_Sdi_s, features_Sdi_t),
                    loss_dict_prefix="loss_Sdi"
                    )
                loss_Sds, loss_dict_Sds = self.discriminator_Sds(
                    (features_Sds_s, features_Sds_t),
                    loss_dict_prefix="loss_Sds"
                    )

                total_loss = det_loss + loss_AMFA + self.ga_dis_lambda * loss_Sdi + self.ga_dis_lambda *  loss_Sds + self.ga_dis_lambda *  loss_MI + loss_gate
                outputs = {
                    "total_loss": total_loss,
                    "AMFA_loss": loss_AMFA,
                    "Sdi_loss": loss_Sdi,
                    "Sds_loss": loss_Sds,
                    "gate_loss": loss_gate,
                    "MI_loss": loss_MI,
                    "det_loss": det_loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
                outputs.update(loss_dict_AMFA)
                # outputs.update(loss_dict_gate)
                outputs.update(loss_dict_Sdi)
                outputs.update(loss_dict_Sds)
            else:
                outputs = self.head(fpn_outs_s)
                
        return outputs

