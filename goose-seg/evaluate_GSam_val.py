"""
evaluate_GSam_val.py

Compare GSam-only predicted masks against GOOSE val GT labels for the
target (low-mIoU) classes.  Prints a per-class IoU table and saves a
Markdown report alongside the script.

Usage:
    conda run -n mit-ground-sam python goose-seg/evaluate_GSam_val.py \
        --dataset_root ~/goose-semseg/goose-dataset \
        --split val \
        --output_report goose-seg/gsam_eval_report.md
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Baseline mIoU from the closed-set model (for delta reporting) ─────────────
BASELINE_MIOU = {
    7:  0.00,   # bikeway
    9:  0.00,   # pedestrian_crossing
    18: 1.70,   # moss
    25: 30.67,  # boom_barrier
    30: 20.21,  # crops
    34: 22.02,  # truck
    39: 32.25,  # wall
    43: 39.01,  # bridge
    45: 33.22,  # pole
    48: 36.38,  # barrier_tape
    49: 30.04,  # kick_scooter
    55: 32.12,  # wire
    62: 9.17,   # tree_root
}

CLASS_NAMES = {
    7:  "bikeway",
    9:  "pedestrian_crossing",
    18: "moss",
    25: "boom_barrier",
    30: "crops",
    34: "truck",
    39: "wall",
    43: "bridge",
    45: "pole",
    48: "barrier_tape",
    49: "kick_scooter",
    55: "wire",
    62: "tree_root",
}

TARGET_IDS = sorted(CLASS_NAMES.keys())


def iter_paired(gt_root: Path, gsam_root: Path):
    """
    Yield (gt_path, gsam_path) for every GT label file.
    If a GSam label doesn't exist for an image it is treated as all-zero
    (no prediction → all FN for that image).
    """
    for gt_path in sorted(gt_root.rglob("*_labelids.png")):
        rel = gt_path.relative_to(gt_root)
        gsam_path = gsam_root / rel
        yield gt_path, gsam_path if gsam_path.exists() else None


def evaluate(gt_root: Path, gsam_root: Path):
    """
    Returns per-class dict of {class_id: {"tp": int, "fp": int, "fn": int}}.
    """
    stats = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in TARGET_IDS}

    pairs = list(iter_paired(gt_root, gsam_root))
    for gt_path, gsam_path in tqdm(pairs, desc="Evaluating"):
        gt = np.array(Image.open(gt_path))
        gsam = (np.array(Image.open(gsam_path))
                if gsam_path is not None
                else np.zeros_like(gt))

        for cid in TARGET_IDS:
            gt_mask = gt == cid
            pred_mask = gsam == cid
            stats[cid]["tp"] += int(np.logical_and(gt_mask, pred_mask).sum())
            stats[cid]["fp"] += int(np.logical_and(~gt_mask, pred_mask).sum())
            stats[cid]["fn"] += int(np.logical_and(gt_mask, ~pred_mask).sum())

    return stats


def compute_iou(stats: dict) -> dict[int, float | None]:
    ious = {}
    for cid, s in stats.items():
        denom = s["tp"] + s["fp"] + s["fn"]
        ious[cid] = s["tp"] / denom if denom > 0 else None
    return ious


def print_and_save(ious: dict, output_path: Path | None) -> None:
    header = f"{'ID':>3}  {'Class':<25} {'GSam mIoU':>10} {'Baseline':>10} {'Delta':>8}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    lines = [
        "# GSam-only Evaluation vs Baseline (GOOSE Val)",
        "",
        "| ID | Class | GSam mIoU (%) | Baseline mIoU (%) | Delta (%) |",
        "|----|-------|:-------------:|:-----------------:|:---------:|",
    ]

    gsam_valid, baseline_valid = [], []
    for cid in TARGET_IDS:
        name = CLASS_NAMES[cid]
        baseline = BASELINE_MIOU.get(cid, float("nan"))
        iou = ious[cid]
        if iou is None:
            gsam_str = "N/A"
            delta_str = "N/A"
            row = f"{cid:>3}  {name:<25} {'N/A':>10} {baseline*100:>9.2f}%  {'N/A':>7}"
        else:
            gsam_pct = iou * 100
            delta = gsam_pct - baseline
            gsam_str = f"{gsam_pct:.2f}"
            delta_str = f"{delta:+.2f}"
            row = f"{cid:>3}  {name:<25} {gsam_pct:>9.2f}% {baseline:>9.2f}%  {delta:>+7.2f}%"
            gsam_valid.append(gsam_pct)
            baseline_valid.append(baseline)
        print(row)
        lines.append(f"| {cid} | {name} | {gsam_str} | {baseline:.2f} | {delta_str} |")

    print(separator)
    if gsam_valid:
        avg_gsam = np.mean(gsam_valid)
        avg_base = np.mean(baseline_valid)
        delta = avg_gsam - avg_base
        print(f"{'':>3}  {'Mean (valid classes)':<25} {avg_gsam:>9.2f}% {avg_base:>9.2f}%  {delta:>+7.2f}%")
        lines += [
            "",
            f"**Mean (valid classes):** GSam {avg_gsam:.2f}% | Baseline {avg_base:.2f}% | Delta {delta:+.2f}%",
        ]

    if output_path is not None:
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nReport saved → {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path,
                   default=Path("~/goose-semseg/goose-dataset").expanduser())
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--output_report", type=Path,
                   default=Path(__file__).parent / "gsam_eval_report.md")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    gt_root = dataset_root / "labels" / args.split
    gsam_root = dataset_root / "GSam_labels" / args.split

    print(f"GT labels  : {gt_root}")
    print(f"GSam masks : {gsam_root}")

    if not gsam_root.exists():
        print(f"\n[ERROR] GSam_labels directory not found: {gsam_root}")
        print("Run grounded_sam_goose.py first to generate predictions.")
        return

    stats = evaluate(gt_root, gsam_root)
    ious = compute_iou(stats)
    print_and_save(ious, args.output_report.resolve())


if __name__ == "__main__":
    main()
