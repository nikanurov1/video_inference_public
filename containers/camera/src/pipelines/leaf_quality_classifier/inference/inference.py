from leaf_quality_classifier.utils.tools.get_device import get_device
from leaf_quality_classifier.utils.tools.get_yaml_config import get_yaml_config

from leaf_quality_classifier.utils.dataset.tools_dataset import dir_all_images
from leaf_quality_classifier.utils.models.create_models import Model_creator
import torch

from leaf_quality_classifier.utils.dataset.custom_BatchLoader import Custom_bathloader
from leaf_quality_classifier.utils.tools.toch import get_clas_conf
import omegaconf
import numpy as np

from src.utils.base_classes import DSBox, DSSequence, DSTag




class Inference_classifer:  
    def __init__(self, pth_yaml_config):
        print(f"pth_yaml_config {pth_yaml_config}")
        print(f"pth_yaml_config TYPE {type(pth_yaml_config)}")

        if isinstance( pth_yaml_config, str):
            yaml_config=get_yaml_config(pth_yaml_config)
        elif isinstance( pth_yaml_config, omegaconf.dictconfig.DictConfig):
            yaml_config = dict(pth_yaml_config)
        elif isinstance( pth_yaml_config, dict):
            yaml_config = pth_yaml_config

        ## Initialize model
        self.device=get_device(yaml_config)
        backend_model = yaml_config["backend_model"]
        architecture_model = yaml_config["architecture_model"]
        classes=yaml_config["classes"]
        num_classes = len(classes)
        checkpoint = yaml_config["checkpoint"]

        print(f"self.device {self.device}")
        print(f"backend_model {backend_model}")
        print(f"architecture_model {architecture_model}")
        print(f"classes {classes}")
        print(f"num_classes {num_classes}")

        self.model=Model_creator(backend_model, architecture_model, num_classes)
 
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        ## Initialize config parametrs
        self.size = yaml_config["size"]
        self.batch = yaml_config["batch"]
        self.use_albu = yaml_config["use_albu"]
        self.use_norm = yaml_config["use_norm"]

        ## NN constants
        self.softmax_batch=torch.nn.Softmax(dim=1)

    def inference_Items(self, items):
        custom_bathloader = Custom_bathloader(self.size, self.batch, self.use_albu, self.use_norm)

        if isinstance(items, str): # PathImages if string => items==Directory Path => where located images
            list_bathes = custom_bathloader.Batches_fromPathImages(items)
        elif isinstance(items, list): #and isinstance(items[0], np.ndarray): # from ListNumpy if List[np.array]
            list_bathes = custom_bathloader.Batches_fromListNumpy(items)
        else:
            raise Exception("Unsupported type Items")

        # Inference
        list_answer = self.inference_batches(list_bathes)
        return list_answer



    def inference_PathImages(self, path_imgs):

        custom_bathloader = Custom_bathloader(self.size, self.batch, self.use_albu, self.use_norm)
        list_bathes = custom_bathloader.Batches_fromPathImages(path_imgs)
        list_answer = self.inference_batches(list_bathes)
        print(list_answer)

    
    def inference_batches(self, list_bathes):
        list_answer = []
        with torch.no_grad():
            for batch in list_bathes:
                batch = batch.to(self.device)
                out = self.model(batch)
                out_softmax = self.softmax_batch(out)
                out_argmax=torch.argmax(out_softmax, dim=1)
                list_out=get_clas_conf(out_softmax, out_argmax)
                list_answer.extend(list_out)
        return list_answer
    

    def launch_node(self, frames):

        list_all_BoxCrop = []
        for DSframe in frames:
            list_boxCropFrame=DSframe.getList_boxCrop()
            list_all_BoxCrop.extend(list_boxCropFrame)

        # print(f"list_all_BoxCrop {list_all_BoxCrop}")
        # print(f"list_all_BoxCrop TYPE {type(list_all_BoxCrop)}")

        list_answer=self.inference_Items(list_all_BoxCrop)

        iter_answer = iter(list_answer)

        for DSframe in frames:
            for box in DSframe.boxes:
                
                out_model=next(iter_answer)
                # out_model["class"],out_model["conf"]
                box.tags.append(
                                DSTag(
                                    class_id = out_model["class"],
                                    conf = out_model["conf"],
                                    model_name = "leaf_quality"
                                    )
                                )
        return frames



