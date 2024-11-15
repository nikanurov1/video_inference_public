from importlib import import_module
import os 
import sys


def get_instance(config: dict):
    """
    create an instance of class from config

    Args:
        config (dict): config of class

    Returns:
        _type_: _description_
    """
    assert "__self__" in config.keys()
    # print(f"config {config}")

    class_name = config['__self__'].split('.')[-1]
    module_name = '.'.join(config['__self__'].split('.')[:-1])
    # print(f"module_name {module_name}")
    # print(f"class_name {class_name}")

    # Add sys path "src/pipelines" as source
    sys_path_pipline = "src/pipelines"
    root_dir = os.getcwd()
    path_lib = os.path.join(root_dir, sys_path_pipline)
    sys.path.append(path_lib)

    
    try:
        print(module_name)
        class_obj = getattr(import_module(module_name), class_name)
        instance = class_obj(config)
        return instance
    
    except ImportError:
        print("No module")
        raise
        # return None
        
    except AttributeError:
        print("No class")
        raise
        # return None