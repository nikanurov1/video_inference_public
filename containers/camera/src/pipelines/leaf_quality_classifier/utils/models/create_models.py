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
            self.fe = timm.create_model(architecture_model, num_classes=num_classes, pretrained=True) # architecture_model = "levit_128s"
            

    def forward(self, x):
        out = self.fe(x)
        return out




