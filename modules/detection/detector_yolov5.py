import torch
import numpy as np
import random
from modules.detection.base import DetectorBase
from modules.detection.yolov5.utils.augmentations import letterbox
from modules.detection.yolov5.utils.general import (
     non_max_suppression, scale_boxes, xyxy2xywh
)
import onnxruntime

class CarDetector(DetectorBase):
    def __init__(self, cfg):
        super(CarDetector, self).__init__()
        self._cfg = cfg
        providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(cfg["model_path"], providers=providers)
        meta = self.session.get_modelmeta().custom_metadata_map  # metadata
        if 'stride' in meta:
            self._stride, self._names = int(meta['stride']), eval(meta['names'])
        # self._colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self._names))]
        self._conf_thr = cfg["conf_thr"]
        self._iou_thr = cfg["iou_thr"]
        self._input_size = cfg["input_size"]

    def _preprocess(self, img):
        # Padded resize
        img = letterbox(img,new_shape=self._input_size, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.cpu().numpy() 
        return img

    def detect(self, img_src):
        img = self._preprocess(img_src)
        pred = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})[0]
        pred = torch.tensor(pred) if isinstance(pred, np.ndarray) else pred

        box = self._postprocess(pred, img.shape, img_src.shape)
        
        return box
    
    
    def _postprocess(self, pred, img_shape, img0_shape):
        pred = non_max_suppression(pred, self._conf_thr, self._iou_thr, agnostic=False,multi_label=True, max_det=4)
        dets = []
        s_max=0
        
        max_conf=0
        
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img_shape[2:], det[:, :4], img0_shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    # print(xyxy)
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    # if c in [2,5,7]:
                    if c in [0,2,3]:
                        xyxy=torch.tensor(xyxy).view(1, 4).clone().view(-1).tolist()
                        
                        # dets.append(xyxy)
                        
                                            
                        s_=abs(xyxy[0]-xyxy[2])*abs(xyxy[1]-xyxy[3])
                        if s_>s_max:
                            s_max=s_
                            dets=xyxy
                        
                        
                        # if conf > max_conf:
                        #     max_conf=conf
                        #     dets=xyxy
                        
        return dets


    
    
    