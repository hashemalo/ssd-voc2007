import torch
import time
from ssd import build_ssd
from mobilenet_ssd import build_mobilenet_ssd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def benchmark_speed(model, n=100, input_size=(1, 3, 300, 300)):
    model.eval()
    dummy = torch.randn(*input_size).to(DEVICE)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        t0 = time.time()
        for _ in range(n):
            _ = model(dummy)
    return (time.time() - t0) / n * 1000  # ms per image


vgg_net = build_ssd('test', 300, 21).to(DEVICE)
vgg_net.load_weights('weights/ssd300_mAP_77.43_v2.pth')

mob_net = build_mobilenet_ssd(21).to(DEVICE)
mob_net.load_state_dict(torch.load('weights/mobilenet_ssd_epoch50.pth', map_location=DEVICE))

print("=== Model Comparison ===")
print(f"VGG16-SSD    | Params: {count_params(vgg_net):.1f}M | Latency: {benchmark_speed(vgg_net):.2f}ms")
print(f"MobileNet-SSD| Params: {count_params(mob_net):.1f}M | Latency: {benchmark_speed(mob_net):.2f}ms")
print("\nRun voc_eval via eval.py for mAP numbers.")
