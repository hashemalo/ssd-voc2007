import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import VOCDetection, detection_collate
from mobilenet_ssd import build_mobilenet_ssd
from layers.modules import MultiBoxLoss
from utils.augmentations import SSDAugmentation
import time
import os

NUM_CLASSES = 21
BATCH_SIZE = 16       # Reduce to 8 if OOM
LR = 1e-3
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOC_ROOT = './data/VOCdevkit'

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

net = build_mobilenet_ssd('train', num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
criterion = MultiBoxLoss(NUM_CLASSES, 0.5, True, 0, True, 3, 0.5, False, DEVICE == 'cuda')

os.makedirs('weights', exist_ok=True)

net.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    t0 = time.time()
    for i, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        targets = [t.to(DEVICE) for t in targets]

        out = net(imgs)           # returns (loc, cls, priors)
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
