import torch
import torch.nn as nn
from torchvision.models import convnext_base


class EchoPrime(nn.Module):
    def __init__(self, weights: str):
        super().__init__()
        self.view_classifier = convnext_base()
        self.view_classifier.classifier[-1] = torch.nn.Linear(self.view_classifier.classifier[-1].in_features, 11)
        self.view_classifier.load_state_dict(torch.load(weights))

    def forward(self, x):
        return self.view_classifier(x)
