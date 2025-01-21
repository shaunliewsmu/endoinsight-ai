# src/yolo_detector.py
import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import yaml

# Add YOLOv5 directory to path
YOLOV5_DIR = Path(__file__).parent.parent / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.append(str(YOLOV5_DIR))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class YOLODetector:
    def __init__(self, weights_path, data_path, device='0'):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device, data=data_path)
        self.stride = self.model.stride
        self.names = ['glottis']
        self.model.names = self.names
        self.imgsz = check_img_size((640, 640), s=self.stride)
        self.model.eval()

    def process_image(self, image_array, conf_thres=0.25, iou_thres=0.45, max_det=1000):
        # Padded resize
        im = letterbox(image_array, self.imgsz, stride=self.stride)[0]
        
        # Convert
        im = im.transpose((2, 0, 1))[::-1].copy()
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        # Process predictions
        detections = []
        annotator = Annotator(image_array.copy(), line_width=3)
        
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image_array.shape).round()

                # Convert detections to list of dicts
                for *xyxy, conf, cls in reversed(det):
                    detection = {
                        "confidence": float(conf),
                        "bbox": [int(x) for x in xyxy],  # Convert to int for JSON serialization
                        "class": self.names[int(cls)]
                    }
                    detections.append(detection)
                    
                    # Add box to image
                    label = f'glottis {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(0, True))

        return {
            "detections": detections,
            "annotated_image": annotator.result()
        }