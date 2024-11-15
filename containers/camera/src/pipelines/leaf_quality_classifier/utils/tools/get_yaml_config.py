import yaml


def get_yaml_config(pth):
    with open(pth, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config