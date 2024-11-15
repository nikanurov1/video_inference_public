import numpy as np

from src.utils.init_functions import get_instance
from src.utils.base_classes import DSFrame, DSSequence


class LeafsDiseases:
    def __init__(self, config: dict):
        """
        Initialization function

        Args:
            config_path (dict): configuration
        """
        # TODO: make path to config or OmegaConf config as input
        self.config = config
        self._load_pipelines()
        self.batch_size = config.get('batch_size', 4)
    
    def _warmup_models(self):
        # TODO: do it
        pass
    
    def _load_pipelines(self):
        self.pipelines = []
        for pipeline_name in self.config.pipelines_order:
            self.pipelines.append({pipeline_name: get_instance(self.config[pipeline_name])})

    def __call__(self, frames: DSSequence, *args, **kwds) -> DSSequence:
        """Main function, process images trought all pipelines

        Args:
            images (list[np.ndarray]): list of images in BGR (cv2) format 

        Returns:
            list[dict]: list of responses for each image 
        """
        
        # for pipeline_dict in self.pipelines:
        #     pipeline_name, pipeline = list(pipeline_dict.items())[0]
        #     frames = pipeline.launch_node(frames)
        
        # return frames
    
        processed_frames = DSSequence([])

        for i in range(0, len(frames), self.batch_size):
            batch = DSSequence(frames[i:i+self.batch_size])
            
            for pipeline_dict in self.pipelines:
                pipeline_name, pipeline = list(pipeline_dict.items())[0]
                batch = pipeline.launch_node(batch)
            
            processed_frames.extend(batch)
        
        return processed_frames
    