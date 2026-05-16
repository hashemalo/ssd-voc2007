# SSD Object Detection Project — Claude Code Setup

## Project Goal
Replicate SSD (Single Shot MultiBox Detector) on Pascal VOC2007, then swap the VGG16 backbone for MobileNetV2 and compare accuracy vs. speed.

---

## Step 1: Clone the Base Repo

```bash
git clone https://github.com/amdegroot/ssd.pytorch.git
cd ssd.pytorch
```

---

## Step 2: Install Dependencies

```bash
pip install torch torchvision opencv-python matplotlib numpy tqdm
```

---

## Step 3: Download Pascal VOC2007

```bash
mkdir -p data/VOCdevkit
cd data
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
cd ..
```

---

## Step 4: Download Pretrained VGG16 Weights (Baseline)

```bash
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
cd ..
```

---

## Step 5: Create MobileNetV2 Backbone

Create a new file `mobilenet_ssd.py` in the project root. This replaces VGG16 with MobileNetV2 as the feature extractor for SSD.

```python
# mobilenet_ssd.py
import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2SSD(nn.Module):
    """
    SSD with MobileNetV2 backbone instead of VGG16.
    Feature maps extracted from layer 14 and layer 18 of MobileNetV2.
    """
    def __init__(self, num_classes=21):
        super(MobileNetV2SSD, self).__init__()
        self.num_classes = num_classes

        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=True)
        features = mobilenet.features

        # Split into two feature extractors
        self.feature_extractor1 = features[:14]   # stride 16, 96 channels
        self.feature_extractor2 = features[14:]   # stride 32, 1280 channels

        # Extra layers for additional feature maps (matches SSD convention)
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

        # Classification and localization heads
        # Anchors per location: [4, 6, 6, 6, 4, 4] — same as SSD300
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(96,   4 * 4,            kernel_size=3, padding=1),
            nn.Conv2d(1280, 6 * 4,            kernel_size=3, padding=1),
            nn.Conv2d(512,  6 * 4,            kernel_size=3, padding=1),
            nn.Conv2d(256,  6 * 4,            kernel_size=3, padding=1),
        ])
        self.cls_layers = nn.ModuleList([
            nn.Conv2d(96,   4 * num_classes,  kernel_size=3, padding=1),
            nn.Conv2d(1280, 6 * num_classes,  kernel_size=3, padding=1),
            nn.Conv2d(512,  6 * num_classes,  kernel_size=3, padding=1),
            nn.Conv2d(256,  6 * num_classes,  kernel_size=3, padding=1),
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
```

---

## Step 6: Training Script

Create `train_mobilenet.py`:

```python
# train_mobilenet.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import VOCDetection, VOCAnnotationTransform, detection_collate, BaseTransform
from mobilenet_ssd import build_mobilenet_ssd
from multibox_loss import MultiBoxLoss
from utils.augmentations import SSDAugmentation
import time

# Config
NUM_CLASSES = 21
BATCH_SIZE = 16       # Reduce to 8 if OOM
LR = 1e-3
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOC_ROOT = './data/VOCdevkit'

# Dataset
dataset = VOCDetection(
    root=VOC_ROOT,
    transform=SSDAugmentation(300, (104, 117, 123))
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=detection_collate,
    num_workers=2
)

# Model
net = build_mobilenet_ssd(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
criterion = MultiBoxLoss(NUM_CLASSES, 0.5, True, 0, True, 3, 0.5, False, DEVICE == 'cuda')

# Training loop
net.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    t0 = time.time()
    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        targets = [t.to(DEVICE) for t in targets]

        out = net(imgs)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(loader):.4f} | Time: {time.time()-t0:.1f}s")
    if (epoch + 1) % 10 == 0:
        torch.save(net.state_dict(), f'weights/mobilenet_ssd_epoch{epoch+1}.pth')

print("Done. Weights saved to weights/")
```

---

## Step 7: Evaluation Script

Create `eval_model.py` to compute mAP and inference speed for both models:

```python
# eval_model.py
import torch
import time
from data import VOCDetection, VOCAnnotationTransform, BaseTransform
from ssd import build_ssd           # original VGG16 SSD
from mobilenet_ssd import build_mobilenet_ssd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOC_ROOT = './data/VOCdevkit'

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6  # millions

def benchmark_speed(model, n=100, input_size=(1, 3, 300, 300)):
    model.eval()
    dummy = torch.randn(*input_size).to(DEVICE)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy)
        t0 = time.time()
        for _ in range(n):
            _ = model(dummy)
    return (time.time() - t0) / n * 1000  # ms per image

# Load VGG16 SSD
vgg_net = build_ssd('test', 300, 21).to(DEVICE)
vgg_net.load_weights('weights/ssd300_mAP_77.43_v2.pth')

# Load MobileNetV2 SSD
mob_net = build_mobilenet_ssd(21).to(DEVICE)
mob_net.load_state_dict(torch.load('weights/mobilenet_ssd_epoch50.pth'))

print("=== Model Comparison ===")
print(f"VGG16-SSD    | Params: {count_params(vgg_net):.1f}M | Latency: {benchmark_speed(vgg_net):.2f}ms")
print(f"MobileNet-SSD| Params: {count_params(mob_net):.1f}M | Latency: {benchmark_speed(mob_net):.2f}ms")
# mAP eval: run voc_eval.py from the repo separately
```

---

## Step 8: Run mAP Evaluation

Use the existing eval script from the repo:

```bash
# Evaluate VGG16 baseline
python eval.py --trained_model weights/ssd300_mAP_77.43_v2.pth --voc_root ./data/VOCdevkit

# Evaluate MobileNetV2 model
python eval.py --trained_model weights/mobilenet_ssd_epoch50.pth --voc_root ./data/VOCdevkit
```

---

## Expected Results to Report

| Model | mAP (VOC2007) | Params | Latency (ms) |
|---|---|---|---|
| VGG16-SSD (paper) | 77.2 | ~26M | ~baseline~ |
| VGG16-SSD (ours) | ~77 | ~26M | measure |
| MobileNetV2-SSD (ours) | ~65–70 | ~4–5M | measure |

---

## Notes

- If Colab GPU runs out of memory, drop `BATCH_SIZE` to 8 or even 4
- Use only VOC2007 (not 2007+2012) to keep training under ~2 hours on a T4
- Save checkpoints every 10 epochs in case Colab disconnects
- For the paper: focus the analysis on **where** MobileNetV2 degrades — small objects are a good hypothesis to test visually

---

## File Structure When Done

```
ssd.pytorch/
├── data/VOCdevkit/VOC2007/
├── weights/
│   ├── vgg16_reducedfc.pth
│   ├── ssd300_mAP_77.43_v2.pth
│   └── mobilenet_ssd_epoch50.pth
├── mobilenet_ssd.py       ← new
├── train_mobilenet.py     ← new
├── eval_model.py          ← new
└── ... (original repo files)
```
