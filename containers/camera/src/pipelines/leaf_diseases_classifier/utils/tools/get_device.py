import torch


def get_device(yaml_config):

    if "device" in yaml_config : # (not yaml_config["device"] is None)
        device = yaml_config["device"]
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device