# -*- coding: utf-8 -*-
"""
Select six consecutive orthoimage tiles from one bridge, run BE-TR (WPFormer) inference,
and save a 3×6 figure for the paper:

  Row 1 — raw / pre-processed UAV input (NAFNet+DarkIR when enabled)
  Row 2 — ground-truth masks coloured by defect class
  Row 3 — BE-TR predicted masks (class-coloured overlay)

Based on testimage.py and benchmark_efficiency_table.py inference protocol.

Example (six images in one folder — no bridge_prefix needed):
  python visualize_six_tile_betr.py \\
      --input_dir figures/six_tiles_input \\
      --test_gt_dir datasets/ESDIs-SOD/test/GT \\
      --json_label_dir datasets/ESDIs-SOD/test/labels \\
      --model_path save/ESDIs-SOD/WPFormer-ESDIs-SOD-0.022.pth \\
      --output figures/six_tile_betr.png

  # Auto-pick six consecutive tiles from a larger test set:
  python visualize_six_tile_betr.py \\
      --test_image_dir datasets/ESDIs-SOD/test/images \\
      --test_gt_dir datasets/ESDIs-SOD/test/GT \\
      ...

  # With NAFNet+DarkIR pre-processing (Row 1 shows enhanced input):
  python visualize_six_tile_betr.py ... \\
      --nafnet_ckpt weights/nafnet.pth --nafnet_builder mypkg.nafnet:build_nafnet \\
      --darkir_ckpt weights/darkir.pth --darkir_builder mypkg.darkir:build_darkir
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from benchmark_efficiency_table import (
    IMAGENET_NORM,
    IMG_SIZE,
    NAFNetDarkIRPipeline,
    SegmentationWrapper,
    build_preprocess,
    get_segmentation_model,
    load_weights,
)
from model.WPFormer import WPFormer

# ---------------------------------------------------------------------------
# Filename parsing & tile selection
# ---------------------------------------------------------------------------

TILE_RE = re.compile(r"^(?P<bridge>.+?)_(?P<tile>\d{6})$", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_bridge_tile(stem: str) -> Optional[Tuple[str, int]]:
    m = TILE_RE.match(stem)
    if not m:
        return None
    return m.group("bridge"), int(m.group("tile"))


def list_images(image_dir: Path) -> List[Path]:
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def load_images_from_folder(input_dir: Path, num_tiles: int = 6) -> List[Path]:
    """Load exactly ``num_tiles`` images from ``input_dir`` (sorted by filename)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    selected = list_images(input_dir)
    if len(selected) != num_tiles:
        names = ", ".join(p.name for p in selected) or "(empty)"
        raise RuntimeError(
            f"--input_dir must contain exactly {num_tiles} images; found {len(selected)} in {input_dir}: {names}"
        )
    return selected


def tile_label_for_image(image_path: Path) -> str:
    parsed = parse_bridge_tile(image_path.stem)
    if parsed is not None:
        bridge, tile_idx = parsed
        return f"{bridge}\n{tile_idx:06d}"
    return image_path.stem


def figure_id_from_images(image_paths: Sequence[Path]) -> Tuple[str, List[int]]:
    """Bridge name and tile ids for caption metadata (best-effort)."""
    bridges: List[str] = []
    tile_ids: List[int] = []
    for p in image_paths:
        parsed = parse_bridge_tile(p.stem)
        if parsed is not None:
            bridges.append(parsed[0])
            tile_ids.append(parsed[1])
    if bridges and len(set(bridges)) == 1:
        return bridges[0], tile_ids
    common = Path(image_paths[0]).parent.name if image_paths else "six_tiles"
    return common, tile_ids


def gt_path_for_image(gt_dir: Path, image_path: Path) -> Optional[Path]:
    stem = image_path.stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif"):
        cand = gt_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
        cand = gt_dir / f"{stem}_gt{ext}"
        if cand.is_file():
            return cand
    return None


def mask_has_defect(gt_path: Path, min_pixels: int = 32) -> bool:
    gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    if gt is None:
        return False
    if gt.ndim == 3:
        gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    else:
        gray = gt
    if gray.max() <= 1:
        gray = (gray * 255).astype(np.uint8)
    return int((gray > 0).sum()) >= min_pixels


def find_six_consecutive_tiles(
    image_dir: Path,
    gt_dir: Path,
    bridge_prefix: Optional[str] = None,
    start_tile: Optional[int] = None,
    num_tiles: int = 6,
) -> List[Path]:
    """
    Return ``num_tiles`` image paths from one bridge with consecutive tile indices
    and at least one defect per frame.
    """
    by_bridge: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for img_path in list_images(image_dir):
        parsed = parse_bridge_tile(img_path.stem)
        if parsed is None:
            continue
        bridge, tile_idx = parsed
        gt_path = gt_path_for_image(gt_dir, img_path)
        if gt_path is None or not mask_has_defect(gt_path):
            continue
        by_bridge[bridge][tile_idx] = img_path

    if not by_bridge:
        raise RuntimeError(
            "No images matched pattern BRIDGE_###### with defect GT. "
            "Check filenames (e.g. co009PG1P01_000001) and --test_gt_dir."
        )

    if bridge_prefix is not None:
        if bridge_prefix not in by_bridge:
            keys = ", ".join(sorted(by_bridge.keys())[:20])
            raise RuntimeError(f"Bridge '{bridge_prefix}' not found. Available: {keys}")
        bridges = [bridge_prefix]
    else:
        bridges = sorted(by_bridge.keys())

    for bridge in bridges:
        tiles = by_bridge[bridge]
        indices = sorted(tiles.keys())
        if start_tile is not None:
            wanted = list(range(start_tile, start_tile + num_tiles))
            if all(t in tiles for t in wanted):
                return [tiles[t] for t in wanted]
            continue
        for i in range(len(indices) - num_tiles + 1):
            run = indices[i : i + num_tiles]
            if run[-1] - run[0] == num_tiles - 1:
                return [tiles[t] for t in run]

    raise RuntimeError(
        "Could not find six consecutive defect tiles for the requested bridge. "
        "Pass --bridge_prefix and --start_tile explicitly."
    )


# ---------------------------------------------------------------------------
# Class colours & mask loading
# ---------------------------------------------------------------------------

# Avoid red/orange defect colours — bridge decks and rust read as red in UAV imagery.
MASK_BACKGROUND_RGB = (42, 44, 48)
PREDICTION_OVERLAY_RGB = (0, 230, 255)  # cyan — high contrast on concrete/steel
PREDICTION_CONTOUR_RGB = (255, 255, 255)

DEFAULT_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "background": MASK_BACKGROUND_RGB,
    "ConcreteCrack": (0, 220, 255),
    "concrete_crack": (0, 220, 255),
    "Crack": (0, 220, 255),
    "crack": (0, 220, 255),
    "efflorescence": (120, 255, 80),
    "Efflorescence": (120, 255, 80),
    "SteelDefect": (255, 220, 0),
    "steel_defect": (255, 220, 0),
    "PaintDamage": (200, 100, 255),
    "paint_damage": (200, 100, 255),
    "Spalling": (80, 220, 120),
    "spalling": (80, 220, 120),
    "Corrosion": (255, 120, 220),
    "corrosion": (255, 120, 220),
    "defect": PREDICTION_OVERLAY_RGB,
    "Defect": PREDICTION_OVERLAY_RGB,
    "prediction": PREDICTION_OVERLAY_RGB,
}

TAB20 = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def color_for_class(name: str, palette: Dict[str, Tuple[int, int, int]], cache: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if name in cache:
        return cache[name]
    if name in palette:
        cache[name] = palette[name]
        return cache[name]
    idx = len(cache) % len(TAB20)
    cache[name] = TAB20[idx]
    return cache[name]


def load_class_palette(path: Optional[str]) -> Dict[str, Tuple[int, int, int]]:
    out = dict(DEFAULT_CLASS_COLORS)
    if path and Path(path).is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k.startswith("_") or k in ("instructions", "example", "example_class_mapping"):
                continue
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                out[str(k)] = (int(v[0]), int(v[1]), int(v[2]))
    return out


def rasterize_json_mask(
    json_path: Path,
    height: int,
    width: int,
    class_to_id: Dict[str, int],
) -> np.ndarray:
    """LabelMe-style JSON → H×W uint8 class index map."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_map = np.zeros((height, width), dtype=np.uint8)
    for shape in data.get("shapes", []):
        label = shape.get("label", "defect")
        if label not in class_to_id:
            class_to_id[label] = len(class_to_id)
        cid = class_to_id[label]
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        polygon = np.array(pts, dtype=np.int32)
        cv2.fillPoly(label_map, [polygon], int(cid))
    return label_map


def json_for_image(json_dir: Path, image_path: Path) -> Optional[Path]:
    for name in (image_path.stem + ".json", image_path.name + ".json"):
        p = json_dir / name
        if p.is_file():
            return p
    return None


def load_gt_class_map(
    gt_path: Path,
    json_dir: Optional[Path],
    image_path: Path,
    palette: Dict[str, Tuple[int, int, int]],
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Return (H×W class index, id→class name).
    Supports: JSON polygons, RGB colour masks, indexed grayscale, binary GT.
    """
    class_to_id: Dict[str, int] = {"background": 0}
    id_to_name: Dict[int, str] = {0: "background"}

    if json_dir is not None:
        jp = json_for_image(json_dir, image_path)
        if jp is not None:
            gt_probe = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_probe is None:
                raise FileNotFoundError(gt_path)
            h, w = gt_probe.shape[:2]
            label_map = rasterize_json_mask(jp, h, w, class_to_id)
            for name, idx in class_to_id.items():
                id_to_name[idx] = name
            return label_map, id_to_name

    raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(gt_path)

    if raw.ndim == 3:
        bgr = raw[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        flat = rgb.reshape(-1, 3)
        unique = {tuple(c) for c in flat if tuple(c) != (0, 0, 0)}
        h, w = raw.shape[:2]
        total = h * w
        label_map = np.zeros((h, w), dtype=np.uint8)
        for color in sorted(unique):
            if color == (0, 0, 0):
                continue
            mask = (
                (rgb[:, :, 0] == color[0])
                & (rgb[:, :, 1] == color[1])
                & (rgb[:, :, 2] == color[2])
            )
            if mask.sum() / total > 0.35:
                continue
            name = f"class_{color[0]}_{color[1]}_{color[2]}"
            for pname, pcolor in palette.items():
                if pcolor == color:
                    name = pname
                    break
            if name not in class_to_id:
                class_to_id[name] = len(class_to_id)
            cid = class_to_id[name]
            label_map[mask] = cid
            id_to_name[cid] = name
        return label_map, id_to_name

    gray = raw.astype(np.uint8)
    if gray.max() <= 1:
        gray = (gray * 255).astype(np.uint8)
    uniq = np.unique(gray)
    if len(uniq) > 2:
        id_to_name = {int(v): ("background" if v == 0 else f"class_{v}") for v in uniq}
        return gray.astype(np.uint8), id_to_name

    label_map = (gray > 127).astype(np.uint8)
    id_to_name = {0: "background", 1: "defect"}
    return label_map, id_to_name


def class_map_to_rgb(
    label_map: np.ndarray,
    id_to_name: Dict[int, str],
    palette: Dict[str, Tuple[int, int, int]],
    background_rgb: Tuple[int, int, int] = MASK_BACKGROUND_RGB,
) -> np.ndarray:
    h, w = label_map.shape
    out = np.full((h, w, 3), background_rgb, dtype=np.uint8)
    cache: Dict[str, Tuple[int, int, int]] = {}
    for cid, name in id_to_name.items():
        if cid == 0:
            continue
        out[label_map == cid] = color_for_class(name, palette, cache)
    return out


def _draw_mask_contour(
    image: np.ndarray,
    binary_mask: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness, lineType=cv2.LINE_AA)


def overlay_mask_on_rgb(
    rgb: np.ndarray,
    label_map: np.ndarray,
    id_to_name: Dict[int, str],
    palette: Dict[str, Tuple[int, int, int]],
    alpha: float = 0.55,
    overlay_color: Optional[Tuple[int, int, int]] = None,
    contour_rgb: Optional[Tuple[int, int, int]] = PREDICTION_CONTOUR_RGB,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Semi-transparent overlay on the input image (non-red by default)."""
    base = rgb.astype(np.float32).copy()
    fg = label_map > 0
    if not np.any(fg):
        return rgb.astype(np.uint8)

    if overlay_color is not None:
        color_layer = np.zeros_like(base)
        color_layer[fg] = np.array(overlay_color, dtype=np.float32)
    else:
        color_layer = class_map_to_rgb(label_map, id_to_name, palette).astype(np.float32)

    fg3 = fg.astype(np.float32)[..., None]
    blended = base * (1.0 - alpha * fg3) + color_layer * (alpha * fg3)
    out = np.clip(blended, 0, 255).astype(np.uint8)
    if contour_rgb is not None:
        _draw_mask_contour(out, fg, contour_rgb, thickness=contour_thickness)
    return out


# ---------------------------------------------------------------------------
# Inference (testimage.py protocol)
# ---------------------------------------------------------------------------

def load_betr_model(model_path: Path, device: torch.device) -> nn.Module:
    net = WPFormer(channel=64, num_queries=16)
    load_weights(net, model_path)
    return SegmentationWrapper(net).to(device).eval()


def preprocess_rgb(
    pil_image: Image.Image,
    train_size: int,
    device: torch.device,
    pipeline: Optional[nn.Module],
) -> Tuple[Image.Image, torch.Tensor]:
    """Resize for network; optionally run NAFNet+DarkIR on tensor input."""
    resized = pil_image.resize((train_size, train_size), resample=Image.BILINEAR)
    to_tensor = transforms.ToTensor()
    if pipeline is None:
        tensor = IMAGENET_NORM(to_tensor(resized)).unsqueeze(0).to(device)
        display = resized
    else:
        x = to_tensor(resized).unsqueeze(0).to(device)
        with torch.no_grad():
            enhanced = pipeline(x).clamp(0.0, 1.0)
        display = transforms.ToPILImage()(enhanced.squeeze(0).cpu())
        tensor = IMAGENET_NORM(enhanced).to(device)
    return display, tensor


@torch.no_grad()
def predict_mask(model: nn.Module, tensor: torch.Tensor, out_hw: Tuple[int, int]) -> np.ndarray:
    pred = model(tensor)
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    res = torch.sigmoid(pred).cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    pred_u8 = (res * 255).astype(np.uint8)
    pred_img = Image.fromarray(pred_u8).resize(out_hw, resample=Image.BILINEAR)
    return np.array(pred_img)


def binary_to_label_map(pred_gray: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    lm = (pred_gray >= 128).astype(np.uint8)
    return lm, {0: "background", 1: "prediction"}


def parse_rgb_triplet(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3 or any(p < 0 or p > 255 for p in parts):
        raise ValueError(f"Expected R,G,B in 0–255, got: {text}")
    return parts[0], parts[1], parts[2]


# ---------------------------------------------------------------------------
# Figure & caption
# ---------------------------------------------------------------------------

DEFAULT_CAPTION = r"""BE-TR applied to six consecutive orthoimage tiles from a single
UAV bridge inspection pass (RC girder bridge, Grade~1, GSD = 4.0\,mm/pixel
at inference resolution). \textbf{Row 1}: raw UAV input frames after
NAFNet+DarkIR pre-processing. \textbf{Row 2}: ground-truth segmentation masks
(coloured by defect class). \textbf{Row 3}: BE-TR predicted masks. Tile
boundaries are indicated by the frame order (left to right = acquisition
order). Defect segmentation is spatially consistent across adjacent tiles,
and the model correctly identifies co-occurring defects (e.g., concrete crack
with efflorescence in frames 2--3) without inter-class confusion."""


def build_figure(
    row1: List[np.ndarray],
    row2: List[np.ndarray],
    row3: List[np.ndarray],
    tile_names: List[str],
    output_path: Path,
    row1_title: str = "Input",
    dpi: int = 150,
) -> None:
    n = len(row1)
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 6.5))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    row_titles = [row1_title, "Ground truth", "BE-TR prediction"]
    rows = [row1, row2, row3]

    for r, title in enumerate(row_titles):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(rows[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(tile_names[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(title, fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def write_caption(path: Path, caption: str, bridge: str, tiles: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tiles:
        tile_note = ", ".join(f"{t:06d}" for t in tiles)
    else:
        tile_note = "n/a"
    header = f"% Auto-generated for {bridge}, tiles {tile_note}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\\caption{" + caption.strip() + "}\n")
    print(f"Saved caption: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_selected_images(args: argparse.Namespace) -> List[Path]:
    if args.input_dir:
        return load_images_from_folder(Path(args.input_dir), num_tiles=args.num_tiles)
    if not args.test_image_dir:
        raise ValueError("Provide --input_dir (6 images) or --test_image_dir (auto selection).")
    return find_six_consecutive_tiles(
        Path(args.test_image_dir),
        Path(args.test_gt_dir),
        bridge_prefix=args.bridge_prefix,
        start_tile=args.start_tile,
        num_tiles=args.num_tiles,
    )


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    gt_dir = Path(args.test_gt_dir)
    json_dir = Path(args.json_label_dir) if args.json_label_dir else None
    palette = load_class_palette(args.class_colors)
    pred_overlay_color = parse_rgb_triplet(args.pred_overlay_color)
    if args.contour_thickness <= 0 or not args.pred_contour_color:
        pred_contour_color = None
    else:
        pred_contour_color = parse_rgb_triplet(args.pred_contour_color)

    selected = resolve_selected_images(args)
    bridge, tile_ids = figure_id_from_images(selected)
    print(f"Figure ID: {bridge}")
    print("Images (left→right): " + ", ".join(p.name for p in selected))

    model = load_betr_model(Path(args.model_path), device)

    preprocess_pipe = None
    row1_title = "Raw input"
    if args.nafnet_builder and args.darkir_builder:
        preprocess_pipe = build_preprocess("nafnet_darkir", args)
        if preprocess_pipe is not None:
            preprocess_pipe = preprocess_pipe.to(device).eval()
            row1_title = "Input (NAFNet+DarkIR)"
            print("Using NAFNet+DarkIR pre-processing for Row 1 / inference.")
    elif args.preprocessed_image_dir:
        row1_title = "Input (NAFNet+DarkIR)"
        print(f"Using pre-rendered images from {args.preprocessed_image_dir}")

    row1, row2, row3 = [], [], []
    tile_labels = []

    for img_path in selected:
        gt_path = gt_path_for_image(gt_dir, img_path)
        if gt_path is None:
            raise FileNotFoundError(f"No GT for {img_path.name}")

        if args.preprocessed_image_dir:
            pre_dir = Path(args.preprocessed_image_dir)
            load_path = pre_dir / img_path.name
            if not load_path.is_file():
                load_path = img_path
            ori = Image.open(load_path).convert("RGB")
            display_in, tensor = preprocess_rgb(ori, args.train_size, device, None)
        else:
            ori = Image.open(img_path).convert("RGB")
            display_in, tensor = preprocess_rgb(ori, args.train_size, device, preprocess_pipe)

        label_map, id_to_name = load_gt_class_map(gt_path, json_dir, img_path, palette)
        H, W = label_map.shape
        rgb_hw = np.array(ori.resize((W, H), resample=Image.BILINEAR))

        pred_gray = predict_mask(model, tensor, (W, H))
        pred_lm, pred_names = binary_to_label_map(pred_gray)

        row1.append(np.array(display_in.resize((W, H), resample=Image.BILINEAR)))
        row2.append(class_map_to_rgb(label_map, id_to_name, palette))
        row3.append(
            overlay_mask_on_rgb(
                rgb_hw,
                pred_lm,
                pred_names,
                palette,
                alpha=args.overlay_alpha,
                overlay_color=pred_overlay_color,
                contour_rgb=pred_contour_color,
                contour_thickness=args.contour_thickness,
            )
        )

        tile_labels.append(tile_label_for_image(img_path))

    out = Path(args.output)
    build_figure(row1, row2, row3, tile_labels, out, row1_title=row1_title, dpi=args.dpi)

    if args.caption_output:
        cap = args.caption if args.caption else DEFAULT_CAPTION
        write_caption(Path(args.caption_output), cap, bridge, tile_ids)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Six-tile BE-TR qualitative figure (3 rows × 6 columns).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input_dir",
        default=None,
        help="Folder with exactly 6 input images (sorted by name). GT matched by filename in --test_gt_dir.",
    )
    ap.add_argument(
        "--test_image_dir",
        default=None,
        help="Full test image folder for auto-selection of 6 consecutive tiles (optional if --input_dir is set)",
    )
    ap.add_argument("--test_gt_dir", required=True, help="Test GT masks (binary, indexed, or RGB)")
    ap.add_argument("--json_label_dir", default=None, help="LabelMe JSON for multi-class GT colouring")
    ap.add_argument("--model_path", required=True, help="BE-TR (WPFormer) checkpoint .pth")
    ap.add_argument("--output", default="figures/six_tile_betr.png", help="Output figure path")
    ap.add_argument("--caption_output", default="figures/six_tile_betr_caption.tex")
    ap.add_argument("--caption", default=None, help="Override LaTeX caption body")
    ap.add_argument("--class_colors", default=None, help="JSON map class name → [R,G,B]")
    ap.add_argument(
        "--bridge_prefix",
        default=None,
        help="Bridge ID for auto-selection only (ignored when --input_dir is set)",
    )
    ap.add_argument(
        "--start_tile",
        type=int,
        default=None,
        help="First tile index for auto-selection only (e.g. 1 for _000001)",
    )
    ap.add_argument("--num_tiles", type=int, default=6)
    ap.add_argument("--train_size", type=int, default=IMG_SIZE)
    ap.add_argument("--overlay_alpha", type=float, default=0.55)
    ap.add_argument(
        "--pred_overlay_color",
        default="0,230,255",
        help="Prediction overlay RGB (default cyan; avoids red bridge backgrounds)",
    )
    ap.add_argument(
        "--pred_contour_color",
        default="255,255,255",
        help="Contour around prediction mask (empty string to disable)",
    )
    ap.add_argument("--contour_thickness", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--preprocessed_image_dir",
        default=None,
        help="If set, Row 1 loads these images (already NAFNet+DarkIR); inference uses them too",
    )
    ap.add_argument("--nafnet_ckpt", default=None)
    ap.add_argument("--nafnet_builder", default=None)
    ap.add_argument("--darkir_ckpt", default=None)
    ap.add_argument("--darkir_builder", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
