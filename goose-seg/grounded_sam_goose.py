"""
grounded_sam_goose.py

Run Grounding DINO + SAM-HQ on GOOSE val images for low-mIoU target classes
and save pixel-label masks (uint8 PNG, pixel value = GOOSE class ID) to:
    <dataset_root>/GSam_labels/val/<scenario>/<stem>_labelids.png

Usage:
    conda run -n mit-ground-sam python goose-seg/grounded_sam_goose.py \
        --dataset_root ~/goose-semseg/goose-dataset \
        --split val \
        --grounded_checkpoint groundingdino_swint_ogc.pth \
        --sam_hq_checkpoint sam_hq_vit_h.pth \
        --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "GroundingDINO"))
sys.path.insert(0, str(REPO_ROOT / "segment_anything"))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from segment_anything import SamPredictor, sam_hq_model_registry

# ── Target classes (class_id → text query) ────────────────────────────────────
# Only classes that actually appear in the val split (from label_analysis_report)
TARGET_QUERIES = {
    7:  "bikeway",
    9:  "pedestrian crossing",
    18: "moss",
    25: "boom barrier",
    30: "crops",
    34: "truck",
    39: "wall",
    43: "bridge",
    45: "pole",
    48: "barrier tape",
    49: "kick scooter",
    55: "wire",
    62: "tree root",
}

# Phrase → class_id lookup (longest key matched first to avoid prefix collisions)
_PHRASE_TO_CLASS: dict[str, int] = {}
for _cid, _q in TARGET_QUERIES.items():
    _PHRASE_TO_CLASS[_q] = _cid
    _PHRASE_TO_CLASS[_q.replace(" ", "_")] = _cid
# Extra aliases
_PHRASE_TO_CLASS.update({
    "crop": 30,
    "scooter": 49,
    "electric scooter": 49,
    "power line": 55,
    "cable": 55,
    "overhead wire": 55,
    "utility pole": 45,
    "lamp pole": 45,
    "street pole": 45,
    "traffic pole": 45,
})
# Sort keys longest-first so substring search is greedy
_SORTED_KEYS = sorted(_PHRASE_TO_CLASS.keys(), key=len, reverse=True)

# Combined prompt sent to Grounding DINO for all classes at once
COMBINED_PROMPT = " . ".join(TARGET_QUERIES.values()) + " ."


def phrase_to_class_id(phrase: str) -> int | None:
    """Map a DINO-returned phrase to a GOOSE class ID. Returns None if unknown."""
    phrase_lower = phrase.lower().strip()
    for key in _SORTED_KEYS:
        if key in phrase_lower:
            return _PHRASE_TO_CLASS[key]
    return None


# ── Model loading ──────────────────────────────────────────────────────────────

def load_dino(config_path: str, checkpoint_path: str, device: str):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    return model.to(device)


def load_sam_hq(checkpoint_path: str, device: str) -> SamPredictor:
    sam = sam_hq_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    return SamPredictor(sam)


# ── Inference helpers ──────────────────────────────────────────────────────────

_DINO_TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess_image(image_pil: Image.Image):
    image_tensor, _ = _DINO_TRANSFORM(image_pil, None)
    return image_tensor  # (3, H, W)


@torch.no_grad()
def run_dino(model, image_tensor: torch.Tensor, caption: str,
             box_thresh: float, text_thresh: float,
             device: str) -> tuple[torch.Tensor, list[str]]:
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]              # (nq, 4) cx,cy,w,h normed

    mask = logits.max(dim=1)[0] > box_thresh
    logits_filt = logits[mask]
    boxes_filt = boxes[mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = []
    for logit in logits_filt:
        phrase = get_phrases_from_posmap(logit > text_thresh, tokenized, tokenizer)
        phrases.append(phrase)

    return boxes_filt, phrases


def build_label_map(
    image_bgr: np.ndarray,
    boxes_xyxy: torch.Tensor,
    phrases: list[str],
    logit_scores: list[float],
    predictor: SamPredictor,
    device: str,
) -> np.ndarray:
    """
    Returns uint8 HxW label map with pixel values = GOOSE class ID (0 = background).
    When masks overlap, higher DINO logit score wins.
    """
    H, W = image_bgr.shape[:2]
    label_map = np.zeros((H, W), dtype=np.uint8)
    confidence_map = np.full((H, W), -1.0, dtype=np.float32)

    if boxes_xyxy.shape[0] == 0:
        return label_map

    # Run SAM-HQ once for all boxes in this image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_bgr.shape[:2]
    ).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
        hq_token_only=False,
    )
    # masks: (N, 1, H, W) bool

    for mask, phrase, score in zip(masks, phrases, logit_scores):
        class_id = phrase_to_class_id(phrase)
        if class_id is None:
            continue
        binary = mask[0].cpu().numpy()  # (H, W) bool
        update = binary & (score > confidence_map)
        label_map[update] = class_id
        confidence_map[update] = score

    return label_map


# ── Dataset helpers ────────────────────────────────────────────────────────────

def iter_val_images(images_val_dir: Path, labels_val_dir: Path):
    """
    Yield (scenario, stem, image_path) for every GT label file that has a
    matching image, regardless of camera suffix (_windshield_vis, _camera_left,
    _realsense, _front, …).  Driving iteration from GT labels ensures 1-to-1
    correspondence and handles all camera naming conventions.
    """
    for gt_path in sorted(labels_val_dir.rglob("*_labelids.png")):
        scenario = gt_path.parent.name
        stem = gt_path.name.replace("_labelids.png", "")
        img_dir = images_val_dir / scenario
        matches = sorted(img_dir.glob(f"{stem}_*.png"))
        if not matches:
            continue
        yield scenario, stem, matches[0]


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path,
                   default=Path("~/goose-semseg/goose-dataset").expanduser())
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--config", type=str,
                   default=str(REPO_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    p.add_argument("--grounded_checkpoint", type=str,
                   default=str(REPO_ROOT / "groundingdino_swint_ogc.pth"))
    p.add_argument("--sam_hq_checkpoint", type=str,
                   default=str(REPO_ROOT / "sam_hq_vit_h.pth"))
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bert_base_uncased_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    images_dir = dataset_root / "images" / args.split
    labels_dir = dataset_root / "labels" / args.split
    output_root = dataset_root / "GSam_labels" / args.split

    print(f"Images : {images_dir}")
    print(f"Labels : {labels_dir}")
    print(f"Output : {output_root}")
    print(f"Prompt : {COMBINED_PROMPT}")
    print(f"Device : {args.device}")

    # Load models
    print("\nLoading Grounding DINO...")
    if args.bert_base_uncased_path:
        # inject bert path into SLConfig before build
        cfg = SLConfig.fromfile(args.config)
        cfg.bert_base_uncased_path = args.bert_base_uncased_path
        cfg.device = args.device
        from GroundingDINO.groundingdino.models import build_model as _build
        dino = _build(cfg)
        ckpt = torch.load(args.grounded_checkpoint, map_location="cpu")
        dino.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        dino.eval().to(args.device)
    else:
        dino = load_dino(args.config, args.grounded_checkpoint, args.device)

    print("Loading SAM-HQ vit_h...")
    sam_predictor = load_sam_hq(args.sam_hq_checkpoint, args.device)

    # Iterate images
    image_list = list(iter_val_images(images_dir, labels_dir))
    skipped, saved = 0, 0

    for scenario, stem, img_path in tqdm(image_list, desc="Processing"):
        # Load & preprocess
        image_pil = Image.open(img_path).convert("RGB")
        image_tensor = preprocess_image(image_pil)

        # DINO detection
        boxes_filt, phrases = run_dino(
            dino, image_tensor, COMBINED_PROMPT,
            args.box_threshold, args.text_threshold, args.device,
        )

        if boxes_filt.shape[0] == 0:
            skipped += 1
            continue

        # Extract logit scores from phrase strings (DINO appends "(score)")
        logit_scores = []
        clean_phrases = []
        for ph in phrases:
            if "(" in ph and ph.endswith(")"):
                name, score_str = ph.rsplit("(", 1)
                try:
                    logit_scores.append(float(score_str[:-1]))
                except ValueError:
                    logit_scores.append(0.0)
                clean_phrases.append(name.strip())
            else:
                logit_scores.append(0.5)
                clean_phrases.append(ph.strip())

        # Filter to only known target classes (discard unrecognised phrases)
        known_mask = [phrase_to_class_id(p) is not None for p in clean_phrases]
        if not any(known_mask):
            skipped += 1
            continue

        boxes_filt = boxes_filt[known_mask]
        clean_phrases = [p for p, k in zip(clean_phrases, known_mask) if k]
        logit_scores = [s for s, k in zip(logit_scores, known_mask) if k]

        # Convert DINO boxes (cx,cy,w,h normalised) → xyxy pixel coords
        W_img, H_img = image_pil.size
        boxes_xyxy = boxes_filt.clone()
        boxes_xyxy *= torch.tensor([W_img, H_img, W_img, H_img])
        boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

        # SAM-HQ masks → label map
        image_bgr = cv2.imread(str(img_path))
        label_map = build_label_map(
            image_bgr, boxes_xyxy, clean_phrases, logit_scores,
            sam_predictor, args.device,
        )

        # Save
        out_dir = output_root / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_labelids.png"
        Image.fromarray(label_map, mode="L").save(out_path)
        saved += 1

    print(f"\nDone. Saved {saved} masks, skipped {skipped} (no target detected).")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
