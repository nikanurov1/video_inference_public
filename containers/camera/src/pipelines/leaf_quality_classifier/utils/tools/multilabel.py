import yaml


def get_names_classes(c):
    list_cl_names = []
    for i in c:
        list_cl_names.append(i[0])
    return list_cl_names


def load_yaml(data_yaml):
    with open(data_yaml, "r") as stream:
        try:
            data_yaml = yaml.safe_load(stream)
            data_yaml=tuple(data_yaml.items())
            return data_yaml
        except yaml.YAMLError as exc:
            print(exc)


def cheking_yaml(data_yaml, classes):
    wrong_labels =[]
    data=load_yaml(data_yaml)
    for i in data:
        keys_labels = list(i[1].keys())
        if keys_labels != classes:
            wrong_labels.append([i[0],list(i[1].keys())])
        
    return wrong_labels