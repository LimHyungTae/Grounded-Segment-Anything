# GOOSE-Seg × Grounded-SAM — Long-Tail Hybrid Pipeline

## Motivation

The closed-set segmentation model (ConvNeXt + Mask2Former, ~51% composite mIoU) hits a performance ceiling on **long-tail / rare classes**: network changes and loss tuning alone cannot recover False Negatives for classes with very few training examples.

The idea here is a **Hybrid pipeline**:

1. Run the closed-set model as usual.
2. For each rare class, issue an explicit text query to **Grounding DINO** (open-set detector) → get bounding boxes.
3. Refine boxes to pixel-wise masks using **SAM-HQ** (ViT-H).
4. **Late-fuse** the open-set masks into the closed-set prediction to patch False Negatives.

This directory contains the scripts to execute and evaluate Step 1-3, and will grow to include the fusion step.

---

## Repository layout

```
goose-seg/
├── grounded_sam_goose.py     # Main pipeline: DINO + SAM-HQ → GSam_labels/
├── evaluate_GSam_val.py      # GSam-only IoU vs GT (standalone quality check)
├── analyze_labels.py         # Scan val labels for target-class presence
├── label_analysis_report.md  # Output of analyze_labels.py
├── gsam_eval_report.md       # Output of evaluate_GSam_val.py (this run)
└── README.md                 # This file
```

---

## Target classes

Classes selected for open-set recovery (low mIoU from closed-set model, with at least 1 GT image in val):

| Class ID | Name | Val images | Baseline mIoU |
|----------|------|:----------:|:-------------:|
| 7  | bikeway             |   3   |  0.00% |
| 9  | pedestrian_crossing |   2   |  0.00% |
| 18 | moss                |  20   |  1.70% |
| 25 | boom_barrier        |  21   | 30.67% |
| 30 | crops               | 109   | 20.21% |
| 34 | truck               | 120   | 22.02% |
| 39 | wall                | 142   | 32.25% |
| 43 | bridge              |  49   | 39.01% |
| 45 | pole                | 731   | 33.22% |
| 48 | barrier_tape        |  16   | 36.38% |
| 49 | kick_scooter        |   5   | 30.04% |
| 55 | wire                | 335   | 32.12% |
| 62 | tree_root           |   3   |  9.17% |

> Classes with 0 GT images in val (`on_rails`, `tunnel`, `outlier`, `pipe`, `military_vehicle`) are excluded — their 0% mIoU is trivially explained by absence in the val split.

---

## Setup

```bash
# 1. Compile GroundingDINO CUDA ops (required once)
cd GroundingDINO
conda run -n mit-ground-sam python setup.py build_ext --inplace
cd ..

# 2. Checkpoints (already present in repo root after running make run)
#    groundingdino_swint_ogc.pth  — Grounding DINO SwinT-OGC
#    sam_hq_vit_h.pth             — SAM-HQ ViT-H
#    (download SAM-HQ from https://github.com/SysCV/sam-hq#model-checkpoints)
```

---

## Running the pipeline

### Step 1 — Generate GSam label masks

```bash
conda run -n mit-ground-sam python goose-seg/grounded_sam_goose.py \
    --dataset_root ~/goose-semseg/goose-dataset \
    --split val \
    --device cuda
```

Outputs `uint8` PNG masks (pixel value = GOOSE class ID, 0 = background/no detection) to:
```
~/goose-semseg/goose-dataset/GSam_labels/val/<scenario>/<stem>_labelids.png
```

Runtime: ~10 min on a single GPU for 1369 val images.
Result: **1304 masks saved, 65 skipped** (DINO found no target-class detections).

### Step 2 — Evaluate GSam-only quality

```bash
conda run -n mit-ground-sam python goose-seg/evaluate_GSam_val.py \
    --dataset_root ~/goose-semseg/goose-dataset \
    --output_report goose-seg/gsam_eval_report.md
```

---

## Results — GSam standalone vs Closed-set baseline

| ID | Class | GSam mIoU (%) | Baseline mIoU (%) | Delta (%) |
|----|-------|:-------------:|:-----------------:|:---------:|
|  7 | bikeway             |  0.00 |  0.00 |  +0.00 |
|  9 | pedestrian_crossing |  0.00 |  0.00 |  +0.00 |
| 18 | moss                |  0.00 |  1.70 |  -1.70 |
| 25 | boom_barrier        |  0.01 | 30.67 | -30.66 |
| 30 | crops               |  6.95 | 20.21 | -13.26 |
| 34 | truck               | 12.65 | 22.02 |  -9.37 |
| 39 | wall                |  5.72 | 32.25 | -26.53 |
| 43 | bridge              | 12.29 | 39.01 | -26.72 |
| 45 | pole                | 13.44 | 33.22 | -19.78 |
| 48 | barrier_tape        |  0.00 | 36.38 | -36.38 |
| 49 | kick_scooter        |  0.01 | 30.04 | -30.03 |
| 55 | wire                |  4.89 | 32.12 | -27.23 |
| 62 | tree_root           |  0.00 |  9.17 |  -9.17 |
| **—** | **Mean** | **4.30** | **22.06** | **-17.76** |

### Analysis

**GSam standalone scores are low by design** — this is expected and not the goal.

The standalone evaluation measures IoU against the full GT label map. GSam acts as a dense open-set detector: it fires on all visually plausible regions matching the text prompt, resulting in many **False Positive** pixels outside the true GT boundaries. IoU = TP / (TP + FP + FN), so high FP devastates the score even when detections are structurally correct.

Concretely:

| Issue | Example |
|-------|---------|
| **Over-detection (FP)** | "wall" fires on building facades, fences, large flat surfaces |
| **Semantic mismatch** | "wire" prompt matches thin cables and guard rails alike |
| **Rare class starvation** | bikeway/barrier\_tape/tree\_root have ≤3–16 GT images; any FP overwhelms TP |
| **Prompt under-specification** | GOOSE class definitions are narrower than common-language text prompts |

The useful signal here: classes with **non-zero GSam IoU** (`truck` 12.6%, `pole` 13.4%, `bridge` 12.3%) confirm that Grounding DINO **does** localise these objects. The detections are structurally present but too noisy to win in isolation.

---

## v2 — Per-class query + threshold tuning (in progress)

### Root cause analysis of v1

Running per-image precision/recall breakdown revealed three distinct failure modes:

| Group | Classes | Recall | Pred/GT ratio | Root cause |
|-------|---------|:------:|:-------------:|------------|
| **A** | crops, truck, bridge | 45–80% | 3–10× | High recall but massive FP from over-detection |
| **B** | bikeway, pedestrian_crossing, barrier_tape, tree_root | 0% | 33–47,000× | Prompt fires on wrong semantic regions entirely |
| **C** | pole, wire | 9–19% | 0.6–0.8× | Under-detecting; precision ok but recall too low |

- `truck` had **79.8% recall** — DINO found trucks, but with 6× too many FP pixels → IoU collapses  
- `bikeway` had **47,130× over-prediction** at 0% recall — "bikeway" prompt fires on road surfaces universally

### v2 changes

1. **Per-class independent DINO query** (not one combined prompt): eliminates cross-class interference
2. **Per-class box/text thresholds**: Group A gets high thresholds (0.50) to suppress FP; Group C gets lower (0.35) to recover recall
3. **Per-class `max_area_ratio` filter**: detections covering more than X% of the image are discarded as whole-scene false positives
4. **NMS per class** (IoU=0.50): removes duplicate detections before passing to SAM
5. **More specific prompts**: e.g. `"bicycle lane road marking . bike path on road"` instead of `"bikeway"`

---

## Next step — Late fusion

The planned fusion strategy: run GSam only as a **False Negative corrector**, not as a standalone predictor.

```
fused[pixel] = closed_set[pixel]                       # default: trust closed-set
             | gsam[pixel]  if closed_set[pixel] == 0  # fill in where closed-set predicts nothing
```

This suppresses GSam FP (regions already covered by the closed-set model are unchanged) while recovering FN that the closed-set model silently drops. The fusion script (`fuse_and_evaluate.py`) is the next deliverable.
