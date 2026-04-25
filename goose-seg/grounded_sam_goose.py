"""
grounded_sam_goose.py

Run Grounding DINO + SAM-HQ on GOOSE val images for low-mIoU target classes
and save pixel-label masks (uint8 PNG, pixel value = GOOSE class ID) to:
    <dataset_root>/GSam_labels/val/<scenario>/<stem>_labelids.png

Design (v3):
  - Combined prompt (all classes in one DINO call) — more aligned with DINO's
    training distribution than per-class calls; preserves scene context.
  - Moderate global threshold (box=0.40, text=0.30) to reduce FP vs v1 (0.30/0.25)
    while keeping recall for Group-A classes (truck 80%, crops 72%).
  - NMS per image (IoU=0.50) before SAM to remove duplicate boxes.
  - Per-class max_area_ratio filter: discard SAM masks that cover an
    unrealistically large fraction of the image (whole-scene FP guard).
  - phrase→class matching still uses substring on DINO-returned phrases.

Usage:
    conda run -n mit-ground-sam python goose-seg/grounded_sam_goose.py \
        --dataset_root ~/goose-semseg/goose-dataset \
        --split val \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
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

# ── Target class text queries ──────────────────────────────────────────────────
# Prompts are more specific than v1 to reduce wrong-location FP for Group-B
# classes (bikeway, pedestrian_crossing, barrier_tape, tree_root).
TARGET_QUERIES: dict[int, str] = {
    7:  "bicycle lane road marking",
    9:  "zebra crossing white stripes",
    18: "green moss on rock",
    25: "red white boom barrier gate",
    30: "agricultural crops field",
    34: "large freight truck",
    39: "retaining wall stone wall",
    43: "road bridge overpass",
    45: "street pole utility pole",
    48: "red white warning tape",
    49: "electric kick scooter",
    55: "overhead power line wire",
    62: "exposed tree root on ground",
}

# Combined prompt for one-shot DINO call per image
COMBINED_PROMPT = " . ".join(TARGET_QUERIES.values()) + " ."

# Per-class max fraction of image area a single SAM mask may cover.
# Keeps small/medium objects from generating scene-sized FP masks.
MAX_AREA_RATIO: dict[int, float] = {
    7:  0.15,   # bikeway — road lane, not entire scene
    9:  0.20,   # pedestrian crossing — one crossing
    18: 0.30,   # moss — can cover large rock faces
    25: 0.05,   # boom_barrier — small gate arm
    30: 0.80,   # crops — can dominate a frame
    34: 0.30,   # truck — large vehicle but not whole frame
    39: 0.35,   # wall — can span wide
    43: 0.50,   # bridge — large structure
    45: 0.08,   # pole — single thin object
    48: 0.08,   # barrier_tape — strip, not large area
    49: 0.05,   # kick_scooter — small vehicle
    55: 0.20,   # wire — thin line, moderate area with SAM
    62: 0.10,   # tree_root — ground-level feature
}

# Phrase → class_id (longest-key-first to avoid prefix collisions)
_PHRASE_TO_CLASS: dict[str, int] = {}
for _cid, _q in TARGET_QUERIES.items():
    for _tok in _q.split():
        pass  # build below
    _PHRASE_TO_CLASS[_q] = _cid
    _PHRASE_TO_CLASS[_q.replace(" ", "_")] = _cid

_PHRASE_TO_CLASS.update({
    # aliases for common DINO return variations
    "bicycle lane": 7,
    "bike lane": 7,
    "bike path": 7,
    "zebra crossing": 9,
    "crosswalk": 9,
    "pedestrian crossing": 9,
    "moss": 18,
    "boom barrier": 25,
    "parking gate": 25,
    "barrier gate": 25,
    "crops": 30,
    "crop field": 30,
    "truck": 34,
    "lorry": 34,
    "retaining wall": 39,
    "stone wall": 39,
    "concrete wall": 39,
    "wall": 39,
    "bridge": 43,
    "overpass": 43,
    "pole": 45,
    "utility pole": 45,
    "street pole": 45,
    "warning tape": 48,
    "caution tape": 48,
    "barrier tape": 48,
    "kick scooter": 49,
    "scooter": 49,
    "power line": 55,
    "wire": 55,
    "overhead wire": 55,
    "tree root": 62,
    "tree roots": 62,
})
_SORTED_KEYS = sorted(_PHRASE_TO_CLASS.keys(), key=len, reverse=True)


def phrase_to_class_id(phrase: str) -> int | None:
    p = phrase.lower().strip()
    for key in _SORTED_KEYS:
        if key in p:
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


# ── Inference ─────────────────────────────────────────────────────────────────

_DINO_TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess_image(image_pil: Image.Image) -> torch.Tensor:
    tensor, _ = _DINO_TRANSFORM(image_pil, None)
    return tensor


@torch.no_grad()
def run_dino(model, image_tensor: torch.Tensor, caption: str,
             box_thresh: float, text_thresh: float,
             device: str) -> tuple[torch.Tensor, list[str], list[float]]:
    """Returns (boxes_norm, phrases, logit_scores)."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes  = outputs["pred_boxes"].cpu()[0]             # (nq, 4)

    filt = logits.max(dim=1)[0] > box_thresh
    logits_f = logits[filt]
    boxes_f  = boxes[filt]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases, scores = [], []
    for logit in logits_f:
        phrase = get_phrases_from_posmap(logit > text_thresh, tokenized, tokenizer)
        phrases.append(phrase.strip())
        scores.append(float(logit.max().item()))

    return boxes_f, phrases, scores


def build_label_map(
    image_bgr: np.ndarray,
    image_tensor: torch.Tensor,
    predictor: SamPredictor,
    dino,
    box_thresh: float,
    text_thresh: float,
    device: str,
) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    total_px = H * W
    label_map      = np.zeros((H, W), dtype=np.uint8)
    confidence_map = np.full((H, W), -1.0, dtype=np.float32)

    boxes_norm, phrases, scores = run_dino(
        dino, image_tensor, COMBINED_PROMPT, box_thresh, text_thresh, device)

    if boxes_norm.shape[0] == 0:
        return label_map

    # Map each box to a class_id; drop unknowns
    class_ids = [phrase_to_class_id(p) for p in phrases]
    keep_idx  = [i for i, c in enumerate(class_ids) if c is not None]
    if not keep_idx:
        return label_map

    boxes_norm = boxes_norm[keep_idx]
    class_ids  = [class_ids[i] for i in keep_idx]
    scores     = [scores[i]    for i in keep_idx]

    # cx,cy,w,h → x1,y1,x2,y2 pixel coords
    boxes_xyxy = boxes_norm.clone()
    boxes_xyxy *= torch.tensor([W, H, W, H], dtype=torch.float32)
    boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
    boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

    # NMS across all classes together (removes duplicate / heavily overlapping boxes)
    score_t = torch.tensor(scores, dtype=torch.float32)
    keep    = torchvision.ops.nms(boxes_xyxy.float(), score_t, iou_threshold=0.50)
    boxes_xyxy = boxes_xyxy[keep]
    class_ids  = [class_ids[i] for i in keep.tolist()]
    scores     = [scores[i]    for i in keep.tolist()]

    if boxes_xyxy.shape[0] == 0:
        return label_map

    # SAM-HQ: encode image once, predict masks for all surviving boxes
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    transformed = predictor.transform.apply_boxes_torch(
        boxes_xyxy, (H, W)).to(device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=transformed, multimask_output=False, hq_token_only=False)
    # masks: (N, 1, H, W) bool

    for mask, cid, score in zip(masks, class_ids, scores):
        binary = mask[0].cpu().numpy()
        # Area filter: skip masks that are unrealistically large
        if binary.sum() / total_px > MAX_AREA_RATIO.get(cid, 0.40):
            continue
        # Higher-confidence detection wins on overlap
        update = binary & (score > confidence_map)
        label_map[update] = cid
        confidence_map[update] = score

    return label_map


# ── Dataset ────────────────────────────────────────────────────────────────────

def iter_val_images(images_val_dir: Path, labels_val_dir: Path):
    for gt_path in sorted(labels_val_dir.rglob("*_labelids.png")):
        scenario = gt_path.parent.name
        stem     = gt_path.name.replace("_labelids.png", "")
        matches  = sorted((images_val_dir / scenario).glob(f"{stem}_*.png"))
        if matches:
            yield scenario, stem, matches[0]


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path,
                   default=Path("~/goose-semseg/goose-dataset").expanduser())
    p.add_argument("--split",  type=str,   default="val")
    p.add_argument("--config", type=str,
                   default=str(REPO_ROOT / "GroundingDINO/groundingdino/config"
                               "/GroundingDINO_SwinT_OGC.py"))
    p.add_argument("--grounded_checkpoint", type=str,
                   default=str(REPO_ROOT / "groundingdino_swint_ogc.pth"))
    p.add_argument("--sam_hq_checkpoint",   type=str,
                   default=str(REPO_ROOT / "sam_hq_vit_h.pth"))
    p.add_argument("--box_threshold",  type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    images_dir   = dataset_root / "images" / args.split
    labels_dir   = dataset_root / "labels" / args.split
    output_root  = dataset_root / "GSam_labels" / args.split

    print(f"Prompt       : {COMBINED_PROMPT}")
    print(f"box/text thr : {args.box_threshold} / {args.text_threshold}")
    print(f"Output       : {output_root}")

    dino          = load_dino(args.config, args.grounded_checkpoint, args.device)
    sam_predictor = load_sam_hq(args.sam_hq_checkpoint, args.device)

    image_list = list(iter_val_images(images_dir, labels_dir))
    for scenario, stem, img_path in tqdm(image_list, desc="Processing"):
        image_pil    = Image.open(img_path).convert("RGB")
        image_tensor = preprocess_image(image_pil)
        image_bgr    = cv2.imread(str(img_path))

        label_map = build_label_map(
            image_bgr, image_tensor, sam_predictor, dino,
            args.box_threshold, args.text_threshold, args.device)

        out_dir = output_root / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_map).save(out_dir / f"{stem}_labelids.png")

    print(f"\nDone. {len(image_list)} images → {output_root}")


if __name__ == "__main__":
    main()
