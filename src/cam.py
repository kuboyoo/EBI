from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="../datasets/test_images_1", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_200.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    prev_time = time.time()

    # cap=cv2.VideoCapture(1)
    URL='http://vrl-shrimp.cv:5000/video_feed'
    cap = cv2.VideoCapture(URL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:continue
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            cv2.imwrite('./save_imgs/{}.jpg'.format(len(os.listdir('./save_imgs'))),frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height,width,_=frame.shape
        x = torch.from_numpy(frame.transpose(2, 0, 1))
        x = x.unsqueeze(0).float()  # x = (1, 3, H, W)

        # Apply letterbox resize
        _, _, h, w = x.size()
        ih, iw = (416, 416)
        dim_diff = np.abs(h - w)
        pad1, pad2 = int(dim_diff // 2), int(dim_diff - dim_diff // 2)
        pad = (pad1, pad2, 0, 0) if w <= h else (0, 0, pad1, pad2)
        x = F.pad(x, pad=pad, mode='constant', value=127.5) / 255.0
        x = F.upsample(x, size=(ih, iw), mode='bilinear') # x = (1, 3, 416, 416)
        x=x.to(device)
        with torch.no_grad():
            detections = model(x)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        detections=detections[0]
        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size,[height,width])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                x1, y1, x2, y2=map(int,[x1, y1, x2, y2])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        
        frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
