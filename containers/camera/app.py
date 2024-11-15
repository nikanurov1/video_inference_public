from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from omegaconf import OmegaConf
from src.leaf_pipeline_core import LeafsDiseases
from src.utils.base_classes import DSSequence, DSFrame
import io
from typing import Dict, List

app = FastAPI()

class CameraService:
    def __init__(self, config_path='./configs/leaf_detector.yaml'):
        self.config = OmegaConf.load(config_path)
        self.leaf_diseases = LeafsDiseases(self.config)
        # self.camera = cv2.VideoCapture(0)
        for camera_index in [0, 1, 2]:
            self.camera = cv2.VideoCapture(camera_index)
            if self.camera.isOpened():
                break
        
        if not self.camera.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    def capture_and_process(self):
        ret, frame = self.camera.read()
        if not ret:
            return None, None
        
        ds_frame = DSFrame(frame)
        sequence = DSSequence([ds_frame])
        processed_sequence = self.leaf_diseases(sequence)
        processed_frame = processed_sequence[0]
        plotted_image = processed_frame.get_plot()
        
        class_boxes: Dict[str, List] = {}
        
        if hasattr(processed_frame, 'boxes') and processed_frame.boxes is not None:
            for box in processed_frame.boxes:
                leaf_diseases_tag = next(
                    (tag for tag in box.tags if tag.model_name == 'leaf_diseases'), 
                    None
                )
                
                if leaf_diseases_tag:
                    class_name = str(leaf_diseases_tag.class_id)
                    if class_name not in class_boxes:
                        class_boxes[class_name] = []
                    
                    box_info = {
                        'bbox': [
                            int((box.x - box.w/2) * frame.shape[1]),  # x1
                            int((box.y - box.h/2) * frame.shape[0]),  # y1
                            int((box.x + box.w/2) * frame.shape[1]),  # x2
                            int((box.y + box.h/2) * frame.shape[0])   # y2
                        ],
                        'confidence': float(leaf_diseases_tag.conf),
                        'image': frame[
                            int((box.y - box.h/2) * frame.shape[0]):int((box.y + box.h/2) * frame.shape[0]),
                            int((box.x - box.w/2) * frame.shape[1]):int((box.x + box.w/2) * frame.shape[1])
                        ]
                    }
                    class_boxes[class_name].append(box_info)
            
            for class_name in class_boxes:
                class_boxes[class_name].sort(key=lambda x: x['confidence'], reverse=True)
                class_boxes[class_name] = class_boxes[class_name][:10]
        
        return plotted_image, class_boxes

@app.post("/capture")
async def capture_image():
    camera_service = CameraService()
    image, class_boxes = camera_service.capture_and_process()
    
    if image is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to capture image"}
        )
    
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to encode image"}
        )
    
    response_data = {
        "main_image": buffer.tobytes().hex(),
        "classes": {}
    }
    
    for class_name, boxes in class_boxes.items():
        class_data = {
            "boxes": [],
            "confidences": [],
            "images": []
        }
        
        for box in boxes:
            is_success, box_buffer = cv2.imencode(".jpg", box['image'])
            if is_success:
                class_data["boxes"].append(box['bbox'])
                class_data["confidences"].append(box['confidence'])
                class_data["images"].append(box_buffer.tobytes().hex())
        
        response_data["classes"][class_name] = class_data
    
    return JSONResponse(content=response_data)