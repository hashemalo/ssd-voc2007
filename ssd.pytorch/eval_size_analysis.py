"""Size-stratified AP analysis: VGG16-SSD vs MobileNetV2-SSD on VOC 2007.

Runs inference for both models then computes mAP bucketed by GT bounding-box
area using COCO-style thresholds:
    Small  : area < 32²  = 1024 px²
    Medium : 1024 ≤ area < 96² = 9216 px²
    Large  : area ≥ 9216 px²

Usage:
    python eval_size_analysis.py \\
        --vgg_model weights/ssd300_mAP_77.43_v2.pth \\
        --mob_model weights/mobilenet_ssd_epoch50.pth \\
        --voc_root  ./data/VOCdevkit \\
        --cuda True
"""
from __future__ import print_function
import os
import time
import argparse
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import xml.etree.ElementTree as ET

from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
from ssd import build_ssd
from mobilenet_ssd import build_mobilenet_ssd


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(description='Size-Stratified AP Analysis')
parser.add_argument('--vgg_model', default='weights/ssd300_mAP_77.43_v2.pth', type=str)
parser.add_argument('--mob_model', default='weights/mobilenet_ssd_epoch50.pth', type=str)
parser.add_argument('--voc_root', default=VOC_ROOT)
parser.add_argument('--cuda', default=True, type=str2bool)
parser.add_argument('--out_dir', default='size_analysis', type=str)
args = parser.parse_args()

DATASET_MEAN = (104, 117, 123)
SET_TYPE = 'test'
YEAR = '2007'
NUM_CLASSES = len(labelmap) + 1

# Same path convention as eval.py (string concatenation, not os.path.join)
annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
devkit_path = args.voc_root + 'VOC' + YEAR
cachefile = os.path.join(devkit_path, 'annotations_cache', 'annots.pkl')

# (bucket_label, area_lo_inclusive, area_hi_exclusive)
BUCKETS = [
    ('Small  (<32²)',     0,    1024),
    ('Medium (32²–96²)', 1024, 9216),
    ('Large  (≥96²)',    9216, float('inf')),
    ('Overall',          0,    float('inf')),
]


# ── Annotation helpers ────────────────────────────────────────────────────────

def parse_rec(filename):
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        s = {}
        s['name'] = obj.find('name').text
        s['pose'] = obj.find('pose').text
        s['truncated'] = int(obj.find('truncated').text)
        s['difficult'] = int(obj.find('difficult').text)
        bb = obj.find('bndbox')
        s['bbox'] = [int(bb.find('xmin').text) - 1,
                     int(bb.find('ymin').text) - 1,
                     int(bb.find('xmax').text) - 1,
                     int(bb.find('ymax').text) - 1]
        objects.append(s)
    return objects


def load_annotations(imagesetfile):
    """Load (or build) the annotation cache. Returns (imagenames, recs)."""
    with open(imagesetfile, 'r') as f:
        imagenames = [x.strip() for x in f.readlines()]

    if not os.path.isfile(cachefile):
        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        recs = {}
        for i, name in enumerate(imagenames):
            recs[name] = parse_rec(annopath % name)
            if i % 100 == 0:
                print('Reading annotation {:d}/{:d}'.format(i + 1, len(imagenames)))
        print('Saving annotation cache to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
        print('Loaded annotation cache from {:s}'.format(cachefile))

    return imagenames, recs


# ── VOC AP metric ─────────────────────────────────────────────────────────────

def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0.
            ap += p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# ── Size-aware per-class AP ───────────────────────────────────────────────────

def class_ap_by_size(detpath, imagenames, recs, classname,
                     size_lo, size_hi, ovthresh=0.5):
    """voc_eval for one class, restricted to GT boxes in [size_lo, size_hi).

    Returns (ap, npos) where ap == -1 means no GT boxes in this bucket.
    """
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(bool)
        # Mask GT boxes that fall outside this size bucket
        for k, b in enumerate(bbox):
            area = (b[2] - b[0]) * (b[3] - b[1])
            if not (size_lo <= area < size_hi):
                difficult[k] = True
        det = [False] * len(R)
        npos += int(sum(~difficult))
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    if npos == 0:
        return -1., 0

    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    if not lines:
        return -1., npos

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[i] for i in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return ap, npos


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(net, dataset, use_cuda, model_tag):
    """Run net on the full dataset; return path template for per-class det files."""
    det_dir = os.path.join(args.out_dir, model_tag)
    os.makedirs(det_dir, exist_ok=True)

    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(NUM_CLASSES)]

    print('\n--- Inference: {} ({} images) ---'.format(model_tag, num_images))
    t0 = time.time()
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        x = im.unsqueeze(0)
        if use_cuda:
            x = x.cuda()
        with torch.no_grad():
            detections = net(x).data

        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:].clone()
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack(
                (boxes.cpu().numpy(), scores[:, np.newaxis])
            ).astype(np.float32)
            all_boxes[j][i] = cls_dets

        if (i + 1) % 500 == 0 or (i + 1) == num_images:
            elapsed = time.time() - t0
            print('  {:d}/{:d}  {:.1f}s'.format(i + 1, num_images, elapsed))

    for cls_ind, cls in enumerate(labelmap):
        detfile = os.path.join(det_dir, 'det_test_{:s}.txt'.format(cls))
        with open(detfile, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if len(dets) == 0:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        index[1], dets[k, -1],
                        dets[k, 0] + 1, dets[k, 1] + 1,
                        dets[k, 2] + 1, dets[k, 3] + 1))

    print('Detection files written → {}/'.format(det_dir))
    return os.path.join(det_dir, 'det_test_{:s}.txt')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        cudnn.benchmark = True

    dataset = VOCDetection(
        args.voc_root, [('2007', SET_TYPE)],
        BaseTransform(300, DATASET_MEAN),
        VOCAnnotationTransform()
    )

    os.makedirs(args.out_dir, exist_ok=True)
    imagenames, recs = load_annotations(imgsetpath.format(SET_TYPE))

    # VGG16-SSD
    print('\nLoading VGG16-SSD from', args.vgg_model)
    net_vgg = build_ssd('test', 300, NUM_CLASSES)
    net_vgg.load_state_dict(torch.load(args.vgg_model, map_location='cpu'))
    net_vgg.eval()
    if use_cuda:
        net_vgg = net_vgg.cuda()
    vgg_detpath = run_inference(net_vgg, dataset, use_cuda, 'vgg')
    del net_vgg
    if use_cuda:
        torch.cuda.empty_cache()

    # MobileNetV2-SSD
    print('\nLoading MobileNetV2-SSD from', args.mob_model)
    net_mob = build_mobilenet_ssd('test', NUM_CLASSES)
    net_mob.load_state_dict(torch.load(args.mob_model, map_location='cpu'))
    net_mob.eval()
    if use_cuda:
        net_mob = net_mob.cuda()
    mob_detpath = run_inference(net_mob, dataset, use_cuda, 'mob')
    del net_mob
    if use_cuda:
        torch.cuda.empty_cache()

    # Size-stratified AP table
    W = 72
    print('\n' + '=' * W)
    print('  Size-Stratified mAP: VGG16-SSD vs MobileNetV2-SSD')
    print('=' * W)
    print('{:<22s} {:>8s} {:>12s} {:>14s} {:>14s}'.format(
        'Size bucket', 'GT boxes', 'VGG16 mAP', 'MobileNet mAP', 'MOB deficit'))
    print('-' * W)

    for bucket_name, size_lo, size_hi in BUCKETS:
        vgg_aps, mob_aps = [], []
        total_gt = 0
        per_class_results = []

        for cls in labelmap:
            vgg_ap, npos = class_ap_by_size(
                vgg_detpath, imagenames, recs, cls, size_lo, size_hi)
            mob_ap, _ = class_ap_by_size(
                mob_detpath, imagenames, recs, cls, size_lo, size_hi)
            total_gt += npos
            if vgg_ap >= 0:
                vgg_aps.append(vgg_ap)
            if mob_ap >= 0:
                mob_aps.append(mob_ap)
            per_class_results.append((cls, vgg_ap, mob_ap, npos))

        vgg_map = np.mean(vgg_aps) * 100 if vgg_aps else float('nan')
        mob_map = np.mean(mob_aps) * 100 if mob_aps else float('nan')
        deficit = mob_map - vgg_map

        print('{:<22s} {:>8d} {:>11.2f}% {:>13.2f}% {:>+13.2f}%'.format(
            bucket_name, total_gt, vgg_map, mob_map, deficit))

        if bucket_name != 'Overall':
            for cls, vap, map_, npos in per_class_results:
                if npos == 0:
                    continue
                vap_s = '{:6.2f}%'.format(vap * 100) if vap >= 0 else '   N/A '
                map_s = '{:6.2f}%'.format(map_ * 100) if map_ >= 0 else '   N/A '
                print('    {:12s}  VGG={:s}  MOB={:s}  GT={:d}'.format(
                    cls, vap_s, map_s, npos))

    print('=' * W)
    print('\nHypothesis: MobileNet should show the largest deficit in the Small bucket')
    print('(no 38×38 feature map → less sensitivity to small objects)\n')


if __name__ == '__main__':
    main()
