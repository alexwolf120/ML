import copy
import torch.nn as nn


class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.features(x)


class Head(nn.Module):
    def __init__(self, in_features=64 * 7 * 7, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class MultiHeadCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = ConvFeatureExtractor()
        self.heads = nn.ModuleDict()
        self.num_classes = num_classes

    def add_head(self, name):
        if name not in self.heads:
            self.heads[name] = Head(num_classes=self.num_classes)

    def forward(self, x, head='mnist'):
        feats = self.backbone(x)
        return self.heads[head](feats)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_first_layers(self, n_layers):
        for i, module in enumerate(self.backbone.features):
            if i < n_layers:
                for p in module.parameters():
                    p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_state(self):
        return copy.deepcopy(self.state_dict())