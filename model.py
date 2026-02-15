# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet(num_classes=2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_densenet(num_classes=2):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
