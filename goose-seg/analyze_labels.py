"""
analyze_labels.py

Scans the GOOSE dataset validation label files and reports which images
contain each of the target low-mIoU classes.  Saves a Markdown report.

Usage:
    python analyze_labels.py \
        --labels_dir ~/goose-semseg/goose-dataset/labels/val \
        --output_report label_analysis_report.md
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Target classes (low / zero mIoU from evaluation) ──────────────────────────
TARGET_CLASSES = {
    7:  ("bikeway",             0.00),
    9:  ("pedestrian_crossing", 0.00),
    18: ("moss",                1.70),
    25: ("boom_barrier",       30.67),
    30: ("crops",              20.21),
    34: ("truck",              22.02),
    35: ("on_rails",            0.00),
    39: ("wall",               32.25),
    43: ("bridge",             39.01),
    44: ("tunnel",              0.00),
    45: ("pole",               33.22),
    48: ("barrier_tape",       36.38),
    49: ("kick_scooter",       30.04),
    55: ("wire",               32.12),
    56: ("outlier",             0.00),
    61: ("pipe",                0.00),
    62: ("tree_root",           9.17),
    63: ("military_vehicle",    0.00),
}


def scan_label_files(labels_dir: Path) -> dict:
    """
    Walk labels_dir recursively, find all *_labelids.png files,
    and record which target classes appear in each image.

    Returns:
        {
          class_id: {
              "name": str,
              "miou": float,
              "images": [
                  {"scenario": str, "file": str, "pixel_count": int}
              ]
          }
        }
    """
    results = {
        cid: {"name": name, "miou": miou, "images": []}
        for cid, (name, miou) in TARGET_CLASSES.items()
    }

    label_files = sorted(labels_dir.rglob("*_labelids.png"))
    print(f"Found {len(label_files)} label files under {labels_dir}")

    for lf in tqdm(label_files, desc="Scanning labels"):
        # Derive scenario name from the immediate parent directory
        scenario = lf.parent.name

        img = np.array(Image.open(lf))
        unique, counts = np.unique(img, return_counts=True)
        present = dict(zip(unique.tolist(), counts.tolist()))

        for cid in TARGET_CLASSES:
            if cid in present:
                results[cid]["images"].append({
                    "scenario": scenario,
                    "file": lf.name,
                    "pixel_count": present[cid],
                })

    return results


def write_markdown_report(results: dict, output_path: Path) -> None:
    lines = []
    lines.append("# GOOSE Val — Target Class Presence Report\n")
    lines.append(
        "Classes selected: those with **mIoU ≤ ~39 %** in the current evaluation.\n"
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append("## Summary\n")
    lines.append("| Class ID | Class Name | Current mIoU (%) | # Images | # Scenarios |")
    lines.append("|----------|------------|-----------------|----------|-------------|")

    for cid in sorted(results):
        info = results[cid]
        n_images = len(info["images"])
        scenarios = {e["scenario"] for e in info["images"]}
        lines.append(
            f"| {cid} | {info['name']} | {info['miou']:.2f} | {n_images} | {len(scenarios)} |"
        )

    lines.append("")

    # ── Per-class detail ───────────────────────────────────────────────────────
    lines.append("## Per-Class Details\n")

    for cid in sorted(results):
        info = results[cid]
        lines.append(f"### {cid} — `{info['name']}` (mIoU {info['miou']:.2f} %)\n")

        if not info["images"]:
            lines.append("**No images found in the validation set.**\n")
            continue

        # Group by scenario
        by_scenario = defaultdict(list)
        for entry in info["images"]:
            by_scenario[entry["scenario"]].append(entry)

        lines.append(f"- **Total images**: {len(info['images'])}")
        lines.append(f"- **Scenarios**: {', '.join(sorted(by_scenario.keys()))}\n")

        lines.append("| Scenario | File | Pixel Count |")
        lines.append("|----------|------|-------------|")
        for entry in sorted(info["images"], key=lambda x: (x["scenario"], x["file"])):
            lines.append(
                f"| {entry['scenario']} | {entry['file']} | {entry['pixel_count']:,} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GOOSE val labels for low-mIoU target classes."
    )
    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=Path("~/goose-semseg/goose-dataset/labels/val").expanduser(),
        help="Path to the validation labels directory",
    )
    parser.add_argument(
        "--output_report",
        type=Path,
        default=Path("label_analysis_report.md"),
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    labels_dir = args.labels_dir.expanduser().resolve()
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    results = scan_label_files(labels_dir)
    write_markdown_report(results, args.output_report.resolve())


if __name__ == "__main__":
    main()
