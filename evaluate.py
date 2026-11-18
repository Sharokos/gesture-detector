"""
Compare detected gestures with manual ELAN annotations, over ALL videos at once.

Tuning on a single video overfits; run this after every config change to see the effect on every
annotated video.

Ground truth: an ELAN file with the same name as the video, next to the video
    INPUT_VIDEOS/<sub folders>/<video>.mp4
    INPUT_VIDEOS/<sub folders>/<video>.eaf      <- manual annotations
Predictions: OUTPUT_INPUT_VIDEOS/<sub folders>/<video>/results/gestures.json (written by main.py)

Usage:
    python evaluate.py                      # uses INPUT_DIR from .env, tier "Gesture_Unit"
    python evaluate.py --tier Gesture --tolerance 0.3
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from config import INPUT_DIR
from data_convertor import parse_time_slots
from main import find_videos

RESOLUTION_S = 0.01  # frame-level metrics on a 10 ms grid


def read_eaf_tier(eaf_path, tier_id):
    root = ET.parse(eaf_path).getroot()
    slots = parse_time_slots(root)
    tiers = {t.attrib.get("TIER_ID"): t for t in root.findall("TIER")}
    if tier_id not in tiers:
        raise KeyError(f"tier '{tier_id}' not in {eaf_path}; available tiers: {list(tiers)}")
    spans = []
    for ann in tiers[tier_id].findall(".//ALIGNABLE_ANNOTATION"):
        start = slots.get(ann.attrib["TIME_SLOT_REF1"])
        end = slots.get(ann.attrib["TIME_SLOT_REF2"])
        if start is not None and end is not None and end > start:
            spans.append((start / 1000.0, end / 1000.0))
    return sorted(spans)


def to_mask(spans, duration):
    mask = np.zeros(int(np.ceil(duration / RESOLUTION_S)) + 1, dtype=bool)
    for s, e in spans:
        mask[int(s / RESOLUTION_S):int(e / RESOLUTION_S)] = True
    return mask


def overlaps(a, b, tolerance):
    return a[0] < b[1] + tolerance and b[0] < a[1] + tolerance


def evaluate(gt, pred, tolerance):
    duration = max([e for _, e in gt + pred] + [0])
    g, p = to_mask(gt, duration), to_mask(pred, duration)
    tp = np.count_nonzero(g & p)
    frame_precision = tp / max(np.count_nonzero(p), 1)
    frame_recall = tp / max(np.count_nonzero(g), 1)

    gt_hit = [any(overlaps(x, y, tolerance) for y in pred) for x in gt]
    pred_hit = [any(overlaps(y, x, tolerance) for x in gt) for y in pred]
    onset_err = [min(abs(x[0] - y[0]) for y in pred if overlaps(x, y, tolerance))
                 for x, hit in zip(gt, gt_hit) if hit]
    return {
        "gt": len(gt),
        "pred": len(pred),
        "frame_P": frame_precision,
        "frame_R": frame_recall,
        "frame_F1": 2 * frame_precision * frame_recall / max(frame_precision + frame_recall, 1e-9),
        "event_R": float(np.mean(gt_hit)) if gt else float("nan"),
        "event_P": float(np.mean(pred_hit)) if pred else float("nan"),
        "onset_err_s": float(np.median(onset_err)) if onset_err else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT_DIR, help="folder with videos + manual .eaf files")
    parser.add_argument("--output", default=None, help="OUTPUT_ folder (default: derived from --input)")
    parser.add_argument("--tier", default="Gesture_Unit", help="ELAN tier with the manual gestures")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="seconds of slack when deciding whether two gestures overlap")
    args = parser.parse_args()

    input_root = os.path.normpath(args.input)
    output_root = args.output or os.path.join(
        os.path.dirname(input_root), "OUTPUT_" + os.path.basename(input_root)
    )

    rows = []
    for rel in find_videos(input_root):
        rel_dir, file = os.path.split(rel)
        name = os.path.splitext(file)[0]
        gt_path = os.path.join(input_root, rel_dir, name + ".eaf")
        pred_path = os.path.join(output_root, rel_dir, name, "results", "gestures.json")
        if not os.path.exists(gt_path):
            print(f"[skip] no annotation {gt_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"[skip] no detections {pred_path} (run main.py first)")
            continue
        try:
            gt = read_eaf_tier(gt_path, args.tier)
        except KeyError as e:
            print(f"[skip] {e}")
            continue
        with open(pred_path) as f:
            pred = sorted((g["start"], g["end"]) for g in json.load(f))
        rows.append((name, evaluate(gt, pred, args.tolerance)))

    if not rows:
        print("Nothing to evaluate.")
        return

    keys = ["gt", "pred", "frame_P", "frame_R", "frame_F1", "event_P", "event_R", "onset_err_s"]
    width = max(len(n) for n, _ in rows + [("MEAN", None)])
    print("\n" + "video".ljust(width) + "".join(k.rjust(12) for k in keys))
    for name, m in rows:
        print(name.ljust(width) + "".join(
            f"{m[k]:12d}" if isinstance(m[k], int) else f"{m[k]:12.3f}" for k in keys))
    means = {k: float(np.nanmean([m[k] for _, m in rows])) for k in keys}
    print("MEAN".ljust(width) + "".join(f"{means[k]:12.3f}" for k in keys))


if __name__ == "__main__":
    main()
