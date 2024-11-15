import timm
import torch
from torch import nn
import transformers


class Model_creator(nn.Module):
    def __init__(self, backed_model, architecture_model, num_classes):
        super().__init__()

        # Выбираю backend
        if backed_model=="timm":
            print(f"Creator model {architecture_model}")
            model_name=architecture_model
            print(f"model_name {model_name}")
            # self.fe = timm.create_model(architecture_model, num_classes=num_classes, pretrained=True) # architecture_model = "levit_128s"
            self.fe = transformers.LevitForImageClassification.from_pretrained(
                model_name, 
                num_labels=num_classes, 
                ignore_mismatched_sizes=True
            )

    def forward(self, x):
        out = self.fe(x)
        return out




