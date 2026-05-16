import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2SSD(nn.Module):
    def __init__(self, num_classes=21):
        super(MobileNetV2SSD, self).__init__()
        self.num_classes = num_classes

        mobilenet = models.mobilenet_v2(pretrained=True)
        features = mobilenet.features

        self.feature_extractor1 = features[:14]   # stride 16, 96 channels
        self.feature_extractor2 = features[14:]   # stride 32, 1280 channels

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
        ])

        # Anchors per location: [4, 6, 6, 6] across 4 feature maps
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(96,   4 * 4,  kernel_size=3, padding=1),
            nn.Conv2d(1280, 6 * 4,  kernel_size=3, padding=1),
            nn.Conv2d(512,  6 * 4,  kernel_size=3, padding=1),
            nn.Conv2d(256,  6 * 4,  kernel_size=3, padding=1),
        ])
        self.cls_layers = nn.ModuleList([
            nn.Conv2d(96,   4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1280, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512,  6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256,  6 * num_classes, kernel_size=3, padding=1),
        ])

    def forward(self, x):
        sources = []
        loc, cls = [], []

        x = self.feature_extractor1(x)
        sources.append(x)

        x = self.feature_extractor2(x)
        sources.append(x)

        for layer in self.extras:
            x = layer(x)
            sources.append(x)

        for (src, loc_layer, cls_layer) in zip(sources, self.loc_layers, self.cls_layers):
            loc.append(loc_layer(src).permute(0, 2, 3, 1).contiguous())
            cls.append(cls_layer(src).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        cls = torch.cat([o.view(o.size(0), -1) for o in cls], 1)

        return loc.view(loc.size(0), -1, 4), cls.view(cls.size(0), -1, self.num_classes)


def build_mobilenet_ssd(num_classes=21):
    return MobileNetV2SSD(num_classes=num_classes)
