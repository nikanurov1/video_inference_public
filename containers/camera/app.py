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
        # Пробуем разные индексы камеры
        for camera_index in [0, 1, 2]:
            self.camera = cv2.VideoCapture(camera_index)
            if self.camera.isOpened():
                break
        
        if not self.camera.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        # Установка разрешения 4K
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    def capture_and_process(self):
        ret, frame = self.camera.read()
        if not ret:
            return None, None
        print(frame)
        
        # Создаем DSFrame и DSSequence
        ds_frame = DSFrame(frame)
        sequence = DSSequence([ds_frame])
        
        # Обработка изображения
        processed_sequence = self.leaf_diseases(sequence)
        processed_frame = processed_sequence[0]
        
        # Получаем изображение с отрисованными боксами
        plotted_image = processed_frame.get_plot()
        
        # Группируем боксы по классам и сортируем по уверенности
        class_boxes: Dict[str, List] = {}
        
        if hasattr(processed_frame, 'boxes') and processed_frame.boxes is not None:
            for box in processed_frame.boxes:
                if box.class_name not in class_boxes:
                    class_boxes[box.class_name] = []
                
                box_info = {
                    'bbox': box.bbox.tolist(),
                    'confidence': float(box.confidence),
                    'image': frame[int(box.bbox[1]):int(box.bbox[3]), 
                                 int(box.bbox[0]):int(box.bbox[2])]
                }
                class_boxes[box.class_name].append(box_info)
            
            # Сортируем и берем топ-10 для каждого класса
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
    
    # Конвертируем основное изображение в bytes
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to encode image"}
        )
    
    # Подготавливаем ответ
    response_data = {
        "main_image": buffer.tobytes().hex(),
        "classes": {}
    }
    
    # Добавляем информацию о боксах каждого класса
    for class_name, boxes in class_boxes.items():
        class_data = {
            "boxes": [],
            "confidences": [],
            "images": []
        }
        
        for box in boxes:
            # Конвертируем вырезанное изображение бокса
            is_success, box_buffer = cv2.imencode(".jpg", box['image'])
            if is_success:
                class_data["boxes"].append(box['bbox'])
                class_data["confidences"].append(box['confidence'])
                class_data["images"].append(box_buffer.tobytes().hex())
        
        response_data["classes"][class_name] = class_data
    
    return JSONResponse(content=response_data)