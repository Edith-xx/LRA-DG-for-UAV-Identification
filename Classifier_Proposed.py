import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self, num_devices=18, num_classes=6):
        super(Classifier, self).__init__()
        self.classifier1 = nn.Linear(48, 18)
        self.classifier2 = nn.Linear(48, 6)
        #self.classifier2 = nn.Sequential(nn.Linear(1280, 512), nn.Linear(512, num_classes))
    def forward(self, x):
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2