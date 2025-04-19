import torch.nn as nn
from torchvision import models


class EmotionEfficientNetB0(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionEfficientNetB0, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)

        for param in self.base_model.features.parameters():
            param.requires_grad = False

        self.base_model.classifier = nn.Identity()

        self.custom_classifier = nn.Sequential(
            nn.Linear(1280, 128),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.custom_classifier(x)
        return x