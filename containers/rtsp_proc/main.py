import cv2
import numpy as np
from omegaconf import OmegaConf
from src.leaf_pipeline_core import LeafsDiseases
from src.utils.base_classes import DSSequence, DSFrame
import time
import os

class RTSPProcessor:
    def __init__(self, rtsp_url, config_path='./configs/leaf_detector.yaml'):
        self.config = OmegaConf.load(config_path)
        self.leaf_diseases = LeafsDiseases(self.config)
        self.rtsp_url = rtsp_url
        
    def process_stream(self):
        cap = cv2.VideoCapture(1)
        
        # Создаем RTSP сервер для выходного потока
        gstreamer_out = (
            'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=2000 ! '
            'rtph264pay ! udpsink host=localhost port=8000'
        )
        output_stream = cv2.VideoWriter(
            gstreamer_out,
            cv2.CAP_GSTREAMER,
            0, 30, (1280, 720)
        )
        
        if not cap.isOpened():
            print(f"Ошибка при открытии RTSP потока: {self.rtsp_url}")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Обработка кадра
                ds_frame = DSFrame(frame)
                sequence = DSSequence([ds_frame])
                processed = self.leaf_diseases(sequence)
                
                # Получаем кадр с отрисованными боксами
                plotted_img = processed[0].get_plot()
                
                # Отправляем в выходной поток
                output_stream.write(plotted_img)
                
                time.sleep(0.033)  # ~30 FPS
                
        finally:
            cap.release()
            output_stream.release()

if __name__ == "__main__":
    rtsp_url = os.getenv("RTSP_URL", "rtsp://192.168.1.100:8554/stream")
    processor = RTSPProcessor(rtsp_url)
    processor.process_stream()