import torch
import numpy as np
import random
import yolov5


class PlateYolov5Detector():
    def __init__(self):
        super(PlateYolov5Detector, self).__init__()

        self.model = yolov5.load('weights/plate_detection/best.pt', device='cpu')
        
        # set model parameters
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image


    def detect(self, img):
        
        results = self.model(img, size=600)
        predictions = results.pred[0]
        # boxes = predictions[:, :4] # x1, y1, x2, y2
        # scores = predictions[:, 4]
        # categories = predictions[:, 5]
        box=[]
        preds=[]
        s_max=0
        for p in predictions:
            bbox=p[:4]
            xyxy=bbox.view(1, 4).clone().view(-1).tolist()   # box with xyxy format, (N, 4)
            preds.append(xyxy)
            s_=abs(xyxy[0]-xyxy[2])*abs(xyxy[1]-xyxy[3])
            if s_>s_max:
                s_max=s_
                box=xyxy
    
        return preds
    
    

    
    
    