from ultralytics import YOLO
from src.utils.base_classes import DSBox, DSSequence
import torchvision.ops as ops
import torch
import time


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


class YoloDetector(YOLO):
    def __init__(
        self,
        config,
    ):
        super().__init__(model=config.weights_path, task=config.task, verbose=True)
        self.config = config
        self.imgsz = config.imgsz
        self.half = config.half
        self.threshold = config.threshold
        self.iou = config.iou
        self.classes = config.classes
        self.__device = get_device()

        self.model.to( self.__device )

    def get_leafs(self, frames: DSSequence) -> DSSequence:
        start_time = time.time()


        results = self.predict(
            [frame.image for frame in frames],
            imgsz=self.imgsz,
            conf=self.threshold,
            iou=self.iou,
            half=self.half, 
            device = self.__device
        )

        class_id = self.classes.get('leaf')


        for result, frame in zip(results, frames):
            boxes = result.boxes.xywh  # Координаты боксов
            scores = result.boxes.conf # Оценки уверенности
            labels = result.boxes.cls # Классы
            boxes_nms = result.boxes.xyxy # Боксы для NMS

            # Применяем Non-Maximum Suppression (NMS) для устранения перекрытий
            keep = ops.nms(boxes_nms, scores, iou_threshold=self.iou)


            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            h_size,w_size,_ = frame.image.shape

            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if label == class_id:
                    x, y, w, h = map(int, box)  # Преобразуем координаты в целые
                    frame.boxes.append(
                        DSBox(
                            x=x/w_size,
                            y=y/h_size,
                            w=w/w_size,
                            h=h/h_size,
                            class_id=label,
                            conf=score,
                            model_name="leaf detector",
                        )
                    )

        print(f"TAKE TIME INFERENCE {time.time() - start_time}")

        return frames
    
    def launch_node(self, frames: DSSequence) -> DSSequence:
        frames = self.get_leafs(frames)
        return frames
    
    