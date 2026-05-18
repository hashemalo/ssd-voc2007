# SSD Backbone Swap: Session Summary

## Experiment Goal

Compare VGG16-SSD (paper baseline) against a MobileNetV2-SSD (efficiency experiment) on Pascal VOC2007. Metrics: mAP, parameter count, inference latency. The experiment reuses the amdegroot SSD codebase with a backbone swap — all non-backbone code (loss, anchors, NMS, eval) stays identical.

---

## Architecture Overview

### VGG16-SSD (baseline)
- Backbone: VGG16, ~26M params
- Feature maps: 6 scales — 38×38, 19×19, 10×10, 5×5, 3×3, 1×1
- Anchors: 8,732 total
- VOC2007 mAP: 77.2% (paper), 77.49% (our run)

### MobileNetV2-SSD (experiment)
- Backbone: MobileNetV2, ~6.9M params total
- Feature maps: 4 scales — 19×19, 10×10, 5×5, 3×3
- Anchors: 2,248 total
- `features[:14]` → 19×19 @ 96 channels (stride 16)
- `features[14:]` → 10×10 @ 1280 channels (stride 32)
- Two extra conv layers added to produce 5×5 and 3×3 maps
- VOC2007 mAP: 54.08% (our run, epoch 50)

**Key tradeoff:** MobileNetV2 is missing the 38×38 feature map (finest scale). This degrades small-object detection. Everything else — loss function, data pipeline, eval protocol — is unchanged between the two models.

---

## Final Measured Results

### mAP (VOC2007 test, 11-point metric)
| Model | mAP |
|-------|-----|
| VGG16-SSD | **77.49%** |
| MobileNetV2-SSD | **54.08%** |

### Speed & Parameters
| Model | Params | GPU latency | CPU latency |
|-------|--------|-------------|-------------|
| VGG16-SSD | ~26M | **6.44 ms** | ~slower |
| MobileNetV2-SSD | ~6.9M | ~54.21 ms | **faster** |

**Counterintuitive GPU result:** MobileNetV2 is *slower* on GPU. Depthwise separable convolutions have poor GPU parallelism — the GPU sits underutilized processing them sequentially. VGG16's large standard convolutions saturate GPU compute efficiently. MobileNetV2 wins on CPU/edge devices, which is its intended deployment target.

### Size-Stratified AP Analysis (COCO-style area thresholds)

| Size bucket | GT boxes | VGG16 mAP | MobileNet mAP | Absolute deficit | Relative deficit |
|-------------|----------|-----------|---------------|-----------------|-----------------|
| Small (<32²) | 566 | 2.06% | 0.03% | -2.03% | **-98.5%** |
| Medium (32²–96²) | 3,823 | 44.82% | 11.78% | -33.05% | **-73.7%** |
| Large (≥96²) | 7,643 | 81.52% | 63.48% | -18.04% | **-22.1%** |
| Overall | 12,032 | 77.49% | 54.08% | -23.41% | **-30.2%** |

**Hypothesis confirmed:** MobileNet's relative degradation is worst on small objects (retains only 1.5% of VGG's performance) and best on large objects (retains 77.9%). The missing 38×38 feature map is the direct cause — smaller feature maps cannot localise objects that occupy only a few pixels.

Note: Use relative deficit (not absolute) when comparing across buckets. The absolute gap is small for Small objects simply because both models score near zero there — relative deficit reveals the true proportional degradation.

---

## Files Created or Modified

### New Files
| File | Purpose |
|------|---------|
| `ssd.pytorch/mobilenet_ssd.py` | Complete MobileNetV2-SSD model with correct anchor config, prior box generation, and phase-aware forward() |
| `ssd.pytorch/eval_mobilenet.py` | VOC2007 mAP eval script for MobileNetV2-SSD |
| `ssd.pytorch/eval_size_analysis.py` | Size-stratified AP analysis — re-runs inference for both models, buckets GT boxes by area, prints per-bucket mAP table |
| `ssd_mobilenet_colab.ipynb` | End-to-end Colab notebook: clone → data → weights → train → eval → size analysis |
| `ssd_mobilenet_gcp.ipynb` | GCP custom runtime variant (persistent disk paths, skip-if-exists logic, no force-reinstall cell) |

### Modified Files
| File | Change |
|------|--------|
| `ssd.pytorch/data/config.py` | Added `mobilenet_voc` anchor config dict (4 feature maps) |
| `ssd.pytorch/layers/functions/detection.py` | Rewrote `Detect` from `autograd.Function` to `nn.Module` |
| `ssd.pytorch/layers/modules/multibox_loss.py` | Removed `Variable`, fixed `size_average` → `reduction='sum'`, fixed integer division, fixed view/mask ordering for hard negative mining |
| `ssd.pytorch/ssd.py` | Removed `Variable`/`volatile`, used `torch.no_grad()`, changed `.type(type(x.data))` → `.to(x.device)` |
| `ssd.pytorch/layers/box_utils.py` | Removed deprecated `out=` parameter from all `torch.index_select` calls |
| `ssd.pytorch/train.py` | Removed `Variable` wrappers, `.data[0]` → `.item()` |
| `ssd.pytorch/eval.py` | Removed `Variable`, `np.bool` → `bool`, fixed `dets == []` → `len(dets) == 0` |
| `ssd.pytorch/eval_mobilenet.py` | Fixed `dets == []` → `len(dets) == 0` |
| `ssd.pytorch/data/coco.py` | Fixed default argument crash: `COCOAnnotationTransform()` → lazy `None` instantiation |
| `ssd.pytorch/utils/augmentations.py` | Fixed NumPy 1.24+ crash: `random.choice()` → `random.randint()` index |

---

## Architectural Bug Fixes

### Bug 1: Wrong Anchor Config

**Problem:** `MobileNetV2SSD` had no anchor configuration — it fell through to using VGG16's `voc` config, which describes 6 feature maps and generates 8,732 anchor boxes. MobileNetV2 only produces 4 feature maps with 2,248 positions. The mismatch caused a shape error when `MultiBoxLoss` tried to match predictions to ground truth.

**Fix:** Added `mobilenet_voc` to `data/config.py`:

```python
mobilenet_voc = {
    'num_classes': 21,
    'feature_maps': [19, 10, 5, 3],
    'min_dim': 300,
    'steps': [16, 30, 60, 100],
    'min_sizes': [60, 105, 150, 195],
    'max_sizes': [105, 150, 195, 240],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'mobilenet_VOC',
}
```

---

### Bug 2: Priors Not Passed Through Forward

**Problem:** The original `MobileNetV2SSD.forward()` returned `(loc, cls)` — a 2-tuple. `MultiBoxLoss.forward()` unpacks `(loc_data, conf_data, priors)` — a 3-tuple. Immediate unpack error at first training step.

**Fix:** `forward()` returns `(loc, cls, self.priors.to(loc.device))` during training, `self.detect(...)` during test.

---

### Bug 3: Priors on Wrong Device (CPU vs CUDA)

**Problem:** `MobileNetV2SSD` stored priors as a plain tensor attribute (`self.priors = self.priorbox.forward()`). When the model was moved to GPU with `.cuda()`, plain attributes do not move — only `nn.Parameter` and registered buffers do. At inference, priors stayed on CPU while `loc` was on CUDA, causing:
```
RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu
```

**Fix (mobilenet_ssd.py):** Changed to `register_buffer`:
```python
self.register_buffer('priors', self.priorbox.forward())
```
Registered buffers move automatically with `.cuda()` / `.to(device)`.

**VGG16 (ssd.py) kept as plain attribute** — the pretrained checkpoint `ssd300_mAP_77.43_v2.pth` does not contain `priors` in its state_dict. Using `register_buffer` there caused `Missing key: priors` on load. Instead, `.to(x.device)` is called in `forward()` at inference time.

---

### Bug 4: IndexError in Hard Negative Mining

**Problem:** In `multibox_loss.py`, the hard negative mining section did:
```python
loss_c[pos] = 0
loss_c = loss_c.view(num, -1)
```
`pos` was a mask of shape `[num, num_priors]`, but `loss_c` at that point was still shape `[num*num_priors, 1]`. Masking a 1D tensor with a 2D mask raised `IndexError`.

**Fix:** Swap the two lines — view first, then mask:
```python
loss_c = loss_c.view(num, -1)
loss_c[pos] = 0
```

---

## PyTorch Compatibility Fixes

The codebase was written for PyTorch ~0.3. Modern PyTorch (1.x+) removed several APIs:

| Old API | New API | File |
|---------|---------|------|
| `Variable(x, volatile=True)` | `with torch.no_grad(): x` | `ssd.py`, `train.py`, `eval.py` |
| `Variable(x, requires_grad=False)` | `x.requires_grad_(False)` | `multibox_loss.py` |
| `loss.data[0]` | `loss.item()` | `train.py` |
| `F.smooth_l1_loss(..., size_average=False)` | `reduction='sum'` | `multibox_loss.py` |
| `F.cross_entropy(..., size_average=False)` | `reduction='sum'` | `multibox_loss.py` |
| `class Detect(Function)` | `class Detect(nn.Module)` | `detection.py` |
| `.type(type(x.data))` | `.to(x.device)` | `ssd.py`, `mobilenet_ssd.py` |
| `torch.index_select(..., out=buf)` | `buf = torch.index_select(...)` | `box_utils.py` |
| `pretrained=True` in torchvision | `weights=MobileNet_V2_Weights.DEFAULT` | `mobilenet_ssd.py` |
| `dets == []` | `len(dets) == 0` | `eval.py`, `eval_mobilenet.py` |

---

## Runtime Debugging Sessions

### 1. `FileNotFoundError: /root/data/coco/coco_labels.txt`

**When:** Triggered on `from data import VOCDetection` — before any training code ran.

**Root cause:** `data/__init__.py` imports `coco.py`. `COCODetection.__init__` had `target_transform=COCOAnnotationTransform()` as a default argument. Python evaluates default arguments once at class definition time — so `COCOAnnotationTransform.__init__` tried to open `coco_labels.txt` on every import, even though COCO was never being used.

**Fix:** Changed default to `None`, instantiate lazily inside the function body.

---

### 2. `FileNotFoundError: VOCdevkit/VOC2007/ImageSets/Main/trainval.txt`

**Root cause:** Download cell timed out silently; files were never extracted.

**Fix:** Rewrote download cell with explicit `-O` and `-C` flags. Added a verification cell that checks all required paths before proceeding.

---

### 3. `FileNotFoundError: VOCdevkit/VOC2012/ImageSets/Main/trainval.txt`

**Root cause:** `VOCDetection.__init__` defaults to loading both 2007 and 2012. Only 2007 was downloaded.

**Fix:** Explicit `image_sets=[('2007', 'trainval')]` in the training cell.

---

### 4. `ValueError: numpy.random.choice on inhomogeneous sequence`

**When:** First training batch — inside `RandomSampleCrop.__call__` in `utils/augmentations.py`.

**Root cause:** `self.sample_options` contains `None` and tuples. NumPy 1.24+ refuses to build an array from a mixed-type sequence.

**Fix:**
```python
mode = self.sample_options[random.randint(len(self.sample_options))]
```

---

### 5. Stale `.pyc` Bytecode Surviving File Edits

**When:** After patching `augmentations.py`, the same `ValueError` recurred.

**Root cause:** Python runs cached `.pyc` bytecode, not the current `.py`. DataLoader worker processes also pick up the stale bytecode.

**Fix (baked into training cell permanently):**
```python
!find /content/ssd-voc2007/ssd.pytorch -name "*.pyc" -delete
# then fresh imports with sys.modules purge
```

---

### 6. `TypeError: super(type, obj): obj must be an instance or subtype of type`

**When:** Re-running the training cell after editing source files (not after a full restart).

**Root cause:** `importlib.reload()` re-executes the module body but does not update already-imported names in other modules. If module A was reloaded but module B still holds a reference to A's old class, `super()` in A's new class finds an incompatible MRO.

**Fix:** Replace `importlib.reload` with full `sys.modules` deletion using prefix matching:
```python
_ours = ['mobilenet_ssd', 'layers', 'utils', 'data']
for _mod in list(sys.modules.keys()):
    if any(_mod == x or _mod.startswith(x + '.') for x in _ours):
        del sys.modules[_mod]
```
**Critical:** Use `_mod == x or _mod.startswith(x + '.')` — NOT `x in _mod`. The substring form matches `torch.utils`, `torch.utils.data`, etc., corrupting PyTorch internals silently.

---

### 7. `AssertionError` in `torch/_ops.py` / `ImportError: InlinedCodeCache` on A100

**When:** Switched from T4 to A100 runtime. Training had been working fine on T4.

**Root cause:** Colab A100/H100 VM images ship with a mismatched PyTorch installation — internal files like `_dynamo/symbolic_convert.py` and `_guards.py` are from different torch versions, causing assertion errors and import failures at startup.

**Fix:** Force-reinstall torch + torchvision + pillow to a consistent version, then restart runtime:
```python
!pip install --force-reinstall -q torch torchvision pillow
```
This is now Cell 1 in the notebook (run first on A100/H100 runtimes). After the reinstall, restart the runtime once and run from Section 1 normally.

---

### 8. `ImportError: cannot import name '_Ink' from 'PIL'`

**When:** After the torch force-reinstall, PIL failed to import.

**Root cause:** `pip install --force-reinstall torch torchvision` caused Pillow to be partially upgraded — mismatched internal symbols.

**Fix:** Added `pillow` explicitly to the force-reinstall command:
```python
!pip install --force-reinstall -q torch torchvision pillow
```

---

### 9. `git pull` Conflict When Pulling Updates Into Running Colab Session

**When:** Tried to pull updated repo files without restarting.

**Root cause:** Colab's in-memory notebook state modified files that were also changed on GitHub.

**Fix:** Use `git checkout origin/main -- <specific_file>` to pull only the new file needed, leaving everything else untouched:
```python
!git -C /content/ssd-voc2007 fetch origin
!git -C /content/ssd-voc2007 checkout origin/main -- ssd.pytorch/eval_size_analysis.py
```

---

### 10. `UserWarning: out= keyword argument is deprecated for torch.index_select`

**When:** Running NMS during eval (every image).

**Root cause:** `box_utils.py` used the deprecated `out=` argument pattern inherited from the old PyTorch API:
```python
torch.index_select(x1, 0, idx, out=xx1)
```

**Fix:** Assign return values directly:
```python
xx1 = torch.index_select(x1, 0, idx)
```

---

### 11. Detection File Conflict Between eval.py and eval_mobilenet.py

**Problem:** Both `eval.py` and `eval_mobilenet.py` write per-class detection `.txt` files to the same path:
```
{voc_root}VOC2007/results/det_test_{cls}.txt
```
When MobileNet eval runs after VGG16 eval, it silently overwrites VGG16's detection files. Running eval in the opposite order would overwrite MobileNet's files.

**Impact:** If you try to analyse VGG16 detections after running both evals, the files are gone.

**Fix (in eval_size_analysis.py):** The size analysis script re-runs inference for both models independently, writing to separate directories (`size_analysis/vgg/` and `size_analysis/mob/`), so it is not affected by whichever eval ran last.

---

### 12. `RuntimeError: Missing key(s) in state_dict: "priors"`

**When:** Loading VGG16 pretrained checkpoint after changing `ssd.py` to use `register_buffer('priors', ...)`.

**Root cause:** The pretrained `ssd300_mAP_77.43_v2.pth` was saved when `priors` was a plain attribute, not a buffer. Buffers are included in `state_dict` — so `load_state_dict` expected a `priors` key that wasn't in the checkpoint.

**Fix:** Reverted `ssd.py` to keep `self.priors` as a plain attribute (not a buffer). Device handling is done via `.to(x.device)` in `forward()` instead.

---

## Notebook Structure (`ssd_mobilenet_colab.ipynb`)

| Section | What It Does |
|---------|-------------|
| Cell 1 (A100 fix) | `pip install --force-reinstall torch torchvision pillow` — run first on A100/H100 only |
| 1. Clone & Install | `git clone hashemalo/ssd-voc2007`, `pip install` dependencies |
| 2. Download VOC2007 | wget trainval + test tarballs, extract, verify paths |
| 3. Download Weights | `vgg16_reducedfc.pth` (VGG16 train init), `ssd300_mAP_77.43_v2.pth` (VGG16 eval checkpoint) |
| 4. Train MobileNetV2-SSD | 50 epochs, SGD lr=1e-3, milestones [30,40], saves every 10 epochs to `weights/` |
| 5. Speed & Params | `count_params()` + GPU + CPU latency benchmarks for both models |
| 6. mAP Eval | `eval.py` for VGG16, `eval_mobilenet.py` for MobileNetV2 |
| 7. Results Table | mAP / Params / Latency comparison (fill in measured values) |
| 8. Size-Stratified AP | `eval_size_analysis.py` — re-runs inference for both, prints per-bucket mAP table |

---

## Saving Work Before Colab Session Ends

**Only file that cannot be recovered:** `weights/mobilenet_ssd_epoch50.pth` (trained weights, took hours to produce).

Everything else (dataset, pretrained weights, code, eval results) can be re-downloaded or regenerated.

**Save to Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy(
    '/content/ssd-voc2007/ssd.pytorch/weights/mobilenet_ssd_epoch50.pth',
    '/content/drive/MyDrive/mobilenet_ssd_epoch50.pth'
)
```

**Restore in a new session** (after running Sections 1–3):
```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy(
    '/content/drive/MyDrive/mobilenet_ssd_epoch50.pth',
    '/content/ssd-voc2007/ssd.pytorch/weights/mobilenet_ssd_epoch50.pth'
)
# Then skip Section 4 (training) and run from Section 5 onwards
```

---

## Key Design Decisions & Gotchas

| Decision | Reason |
|----------|--------|
| `register_buffer` for priors in MobileNet, plain attr in VGG16 | VGG16 pretrained checkpoint doesn't include priors; buffers would cause load failure |
| Prefix matching in sys.modules purge (`startswith`) | Substring matching hits `torch.utils`, `torch.utils.data` and silently corrupts PyTorch |
| Size analysis re-runs inference for both models | Both eval scripts write to the same results dir — can't rely on either set of files being present |
| Relative deficit, not absolute, for size analysis | Absolute deficit is misleading when both models score near zero (small objects) |
| MobileNet slower on GPU than VGG16 | Depthwise separable convolutions have poor GPU parallelism; advantage is on CPU/edge |
