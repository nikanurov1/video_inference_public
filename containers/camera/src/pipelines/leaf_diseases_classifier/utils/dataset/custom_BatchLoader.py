from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from typing import List, Tuple
from PIL import Image
import cv2 as cv


from leaf_diseases_classifier.utils.tools.dirs_handler import dir_all_images



class Batches:
    def __init__(self,batch_size):
        self.batch_size=batch_size
        self.list_bathes = []

    def add_img2batches(self, img_tnsr):
        img_tnsr=img_tnsr.unsqueeze(0)
        if not self.list_bathes: # cheking if list of batches Empty  
            self.list_bathes.append(img_tnsr)
        
        elif self.list_bathes[-1].shape[0]==self.batch_size: # Если последний батч в list ЗАПОЛНЕН до размера batch_size
            self.list_bathes.append(img_tnsr)

        elif self.list_bathes[-1].shape[0]<self.batch_size: # Если последний батч в list НЕ заполнен до размера batch_size
            self.list_bathes[-1] = torch.cat((self.list_bathes[-1], img_tnsr), 0)
        
        else:
            raise Exception('Непредвиденный размер батча')
            
        return None
            
        

class Custom_bathloader(Dataset):
    def __init__(self,
                size: int, # image size
                batch:int,
                use_albu: bool = False,
                use_norm: bool = False,
                ):
        super().__init__()
        
        self.size = size
        self.use_norm = use_norm
        self.use_albu = use_albu
        self.batch_size=batch

        if type(size)==int:
            self.size = (size,size)
        elif isinstance(size, (list, tuple)):
            self.size = size
        else:
            raise TypeError("Variable size should be int or tuple!")


    def imgPath_to_Tensor(self,img_path) -> Tensor:
        img_tensor = Image.open(img_path).convert('RGB')
        img_tensor = transforms.Resize(self.size)(img_tensor)
        img_tensor = transforms.ToTensor()(img_tensor)
        return img_tensor
    
    def nparray_to_Tensor(self,np_array) -> Tensor:
        np_array_rgb = cv.cvtColor(np_array, cv.COLOR_BGR2RGB)
        img_tensor=Image.fromarray(np_array_rgb).convert('RGB')
        # img_tensor = torch.from_numpy(np_array_rgb)
        img_tensor = transforms.Resize(self.size)(img_tensor)
        img_tensor = transforms.ToTensor()(img_tensor)
        return img_tensor

        
    def Batches_fromPathImages(self, path_images ) -> List[Tensor]:
        """
        path_images - path in dir, where located many images
        """

        list_pathimgs:List[str] = dir_all_images(path_images)

        ## Init Batches class
        batch_class = Batches(self.batch_size)

        for idx, img_path in enumerate(list_pathimgs):
            img_tensor = self.imgPath_to_Tensor(img_path)
            batch_class.add_img2batches(img_tensor)

        return batch_class.list_bathes


    def Batches_fromListNumpy(self, list_nparray ) -> List[Tensor]:
        ## Init Batches class
        batch_class = Batches(self.batch_size)

        for idx, img_path in enumerate(list_nparray):
            img_tensor = self.nparray_to_Tensor(img_path)
            batch_class.add_img2batches(img_tensor)
        
        return batch_class.list_bathes




