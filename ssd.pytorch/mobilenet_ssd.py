import torch
import torch.nn as nn
import torchvision.models as models
from layers.functions.prior_box import PriorBox
from layers.functions.detection import Detect
from data.config import mobilenet_voc


class MobileNetV2SSD(nn.Module):
    """SSD with MobileNetV2 backbone instead of VGG16.

    Feature maps (300×300 input):
        features[:14]  → 19×19, 96ch   (stride 16) — 4 anchors/location
        features[14:]  → 10×10, 1280ch (stride 32) — 6 anchors/location
        extra[0]       →  5×5,  512ch  (stride 64) — 6 anchors/location
        extra[1]       →  3×3,  256ch  (stride 100)— 6 anchors/location
    Total anchors: 19²×4 + 10²×6 + 5²×6 + 3²×6 = 2248
    """

    def __init__(self, phase, num_classes=21):
        super(MobileNetV2SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = 300
        self.cfg = mobilenet_voc

        # Build and cache prior boxes using the MobileNet-specific config
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.register_buffer('priors', self.priorbox.forward())

        from torchvision.models import MobileNet_V2_Weights
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        features = mobilenet.features
        self.feature_extractor1 = features[:14]   # stride 16 → 19×19, 96ch
        self.feature_extractor2 = features[14:]   # stride 32 → 10×10, 1280ch

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),  # 10×10 → 5×5, 512ch
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),  # 5×5 → 3×3, 256ch
        ])

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

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

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

        if self.phase == 'test':
            return self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(cls.view(cls.size(0), -1, self.num_classes)),
                self.priors.to(loc.device)
            )
        else:
            return (
                loc.view(loc.size(0), -1, 4),
                cls.view(cls.size(0), -1, self.num_classes),
                self.priors.to(loc.device)
            )


def build_mobilenet_ssd(phase, num_classes=21):
    return MobileNetV2SSD(phase, num_classes)
