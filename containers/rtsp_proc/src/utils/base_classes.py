from typing import List, Optional, Tuple
import numpy as np
import cv2


class DSTag:
    def __init__(self, 
                 class_id: int, # class_id_name: Optional[str],  name: Optional[str] = None
                 conf: float, 
                 model_name: str,
                 ):
        
        self.class_id: int = class_id # for plate  1
        self.conf: float = conf # общий plate 
        self.model_name = model_name 

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class DSBox:
    def __init__(self,
                 x: float, # x центр бокса,   относительные кооридинаты (normolized)
                 y: float, # y центр бокса,   относительные кооридинаты (normolized)
                 w: float, # ширина бокса,    относительные кооридинаты (normolized)
                 h: float, # высота бокса,    относительные кооридинаты (normolized)
                 class_id: int,
                 conf: float,
                 model_name: str,
                 ) -> None:

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id
        self.conf = conf
        self.model_name = model_name
        self.tags: List[DSTag] = []
        self.np_boxCrop = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        print()
        return f"model_name={self.model_name}\nx={self.x}\ny={self.y}\nw={self.w}\nh={self.h}\nclass_id={self.class_id}\nconf={self.conf}\n\n"

    
class DSFrame:
    """ attribute image it is np.ndarray BRG """
    def __init__(self,
        image: np.ndarray,
        request: dict = {} ) -> None:
        
        self.image: np.ndarray = image
        self.boxes: List[DSBox] = []
        self.plotted_img = None
        self.request = request

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def getList_boxCrop(self) -> List[np.array]:
        list_boxCrop =[]
        for box in self.boxes:
            # TODO cheking necessary class from detector
            # if box.class_id==class_id:
            # print(f"box.np_boxCrop {box.np_boxCrop}") 
            # if isinstance(box.np_boxCrop, np.ndarray ):
            #     print(f"box.np_boxCrop SHAPE {box.np_boxCrop.shape}") 

            if isinstance(box.np_boxCrop, np.ndarray ):
                pass
            elif box.np_boxCrop==None:
                np_boxCrop = self._box_crop(box)
                box.np_boxCrop = np_boxCrop

            list_boxCrop.append(box.np_boxCrop)
        return list_boxCrop

    def get_plot(self) -> np.ndarray:
        if self.plotted_img is None:
            self._draw_boxes()
        return self.plotted_img

    def _box_crop(self, box):
        h_size,w_size,_ = self.image.shape
        xstrt=int((box.x-box.w/2)*w_size)
        xend=int((box.x+box.w/2)*w_size)
        ystrt=int((box.y-box.h/2)*h_size)
        yend=int((box.y+box.h/2)*h_size)
        np_boxCrop=self.image[ystrt:yend, xstrt:xend]
        return np_boxCrop
    

    def _draw_boxes(self) -> None:
        self.plotted_img = self.image.copy()
        for box in self.boxes:
            x1, y1 = int((box.x - box.w/2) * self.image.shape[1]), int((box.y - box.h/2) * self.image.shape[0])
            x2, y2 = int((box.x + box.w/2) * self.image.shape[1]), int((box.y + box.h/2) * self.image.shape[0])
            
            color = (255, 255, 255) 
            
            leaf_diseases_tag = next((tag for tag in box.tags if tag.model_name == 'leaf_diseases'), None)
            if leaf_diseases_tag and leaf_diseases_tag.class_id != 0:
                color = (0, 0, 255) 
            
            cv2.rectangle(self.plotted_img, (x1, y1), (x2, y2), color, 2)
            
            leaf_quality_tag = next((tag for tag in box.tags if tag.model_name == 'leaf_quality'), None)
            if leaf_quality_tag and leaf_quality_tag.class_id in [0, 1]:
                cv2.rectangle(self.plotted_img, (x1, y1), (x1+20, y1+20), (0, 165, 255), -1) 

        self.plotted_img = self.plotted_img




class DSSequence:
    def __init__(self, frames: List[DSFrame] ):
        self.frames = frames

    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

    def extend(self, other_sequence):
        self.frames.extend(other_sequence.frames)

    def append(self, frame):
        self.frames.append(frame)

    def __iter__(self):
        return iter(self.frames)






