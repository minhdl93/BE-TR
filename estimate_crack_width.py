# -*- coding: utf-8 -*-
"""
Physical crack width estimates from GT and BE-TR predicted masks (Table 12 protocol).

Method (paper):
  - Inference resolution 384×384; GSD_eff = 4.0 mm/pixel
  - Binarize crack masks, skeletonize, sample EDT on skeleton
  - Mean width (px) = 2 × mean(EDT on skeleton); convert to mm with GSD_eff
  - Error (mm) = |GT width − Predicted width|

Example (masks already saved):
  python estimate_crack_width.py \\
      --image figures/crack_samples/sample1.jpg \\
      --gt figures/crack_samples/sample1_gt.png \\
      --pred figures/crack_samples/sample1_pred.png \\
      --sample_name "Sample 1 (hairline)"

Example (run BE-TR on image, GT from file):
  python estimate_crack_width.py \\
      --image figures/crack_samples/sample1.jpg \\
      --gt figures/crack_samples/sample1_gt.png \\
      --model_path save/ESDIs-SOD/WPFormer-ESDIs-SOD-0.022.pth

Batch (CSV manifest: sample,image,gt,pred — pred optional if --model_path):
  python estimate_crack_width.py --manifest crack_width_manifest.csv --output results/table12.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from torchvision import transforms

from benchmark_efficiency_table import (
    IMAGENET_NORM,
    IMG_SIZE,
    SegmentationWrapper,
    load_weights,
)
from model.WPFormer import WPFormer

# Table 12 defaults
GSD_EFF_MM_PER_PX = 4.0
INFERENCE_SIZE = IMG_SIZE  # 384

@dataclass
class CrackWidthResult:
    sample: str
    gt_width_mm: float
    predicted_width_mm: float
    error_mm: float

    def as_row(self) -> Dict[str, str]:
        return {
            "sample": self.sample,
            "gt_width_mm": f"{self.gt_width_mm:.3f}",
            "predicted_width_mm": f"{self.predicted_width_mm:.3f}",
            "error_mm": f"{self.error_mm:.3f}",
        }


def _resize_mask_nearest(mask: np.ndarray, size: int = INFERENCE_SIZE) -> np.ndarray:
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def _to_binary_crack(mask: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Return H×W bool crack foreground."""
    if mask.dtype == bool:
        return mask
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.shape[2] == 3 else mask[:, :, 0]
    else:
        gray = mask.astype(np.uint8)
    if gray.max() <= 1:
        gray = (gray * 255).astype(np.uint8)
    uniq = np.unique(gray)
    if len(uniq) > 2:
        # Indexed / multi-class: keep crack-like class ids (>0) or named crack ids
        return gray > 0
    return gray >= threshold


def mean_width_mm_from_mask(
    mask: np.ndarray,
    gsd_mm_per_px: float = GSD_EFF_MM_PER_PX,
    inference_size: int = INFERENCE_SIZE,
    threshold: int = 128,
) -> float:
    """
    Estimate mean physical crack width (mm) from a binary or grayscale mask.

    Width in pixels = 2 × mean(EDT on skeleton); mm = width_px × GSD_eff.
    """
    binary = _to_binary_crack(mask, threshold=threshold)
    binary = _resize_mask_nearest(binary.astype(np.uint8), inference_size).astype(bool)
    if not binary.any():
        return float("nan")

    skel = skeletonize(binary)
    if not skel.any():
        return float("nan")

    dist = distance_transform_edt(binary)
    radii = dist[skel]
    if radii.size == 0:
        return float("nan")

    width_px = 2.0 * float(np.mean(radii))
    return width_px * gsd_mm_per_px


def estimate_crack_width(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gsd_mm_per_px: float = GSD_EFF_MM_PER_PX,
    inference_size: int = INFERENCE_SIZE,
    gt_threshold: int = 128,
    pred_threshold: int = 128,
) -> Tuple[float, float, float]:
    """
    Return (gt_width_mm, predicted_width_mm, error_mm) for one sample.
    """
    gt_mm = mean_width_mm_from_mask(
        gt_mask, gsd_mm_per_px=gsd_mm_per_px, inference_size=inference_size, threshold=gt_threshold
    )
    pred_mm = mean_width_mm_from_mask(
        pred_mask,
        gsd_mm_per_px=gsd_mm_per_px,
        inference_size=inference_size,
        threshold=pred_threshold,
    )
    if np.isnan(gt_mm) or np.isnan(pred_mm):
        error_mm = float("nan")
    else:
        error_mm = abs(gt_mm - pred_mm)
    return gt_mm, pred_mm, error_mm


def load_mask_array(path: Union[str, Path]) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return arr


def build_betr_model(model_path: Path, device: torch.device) -> nn.Module:
    net = WPFormer(channel=64, num_queries=16)
    load_weights(net, model_path)
    return SegmentationWrapper(net).to(device).eval()


@torch.no_grad()
def predict_betr_mask(
    image_path: Union[str, Path],
    model: nn.Module,
    device: torch.device,
    inference_size: int = INFERENCE_SIZE,
) -> np.ndarray:
    """BE-TR probability map (0–255 uint8) at original image resolution."""
    pil = Image.open(image_path).convert("RGB")
    h, w = pil.size[1], pil.size[0]
    resized = pil.resize((inference_size, inference_size), resample=Image.BILINEAR)
    tensor = IMAGENET_NORM(transforms.ToTensor()(resized)).unsqueeze(0).to(device)
    pred = model(tensor)
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    res = torch.sigmoid(pred).cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    pred_u8 = (res * 255).astype(np.uint8)
    return np.array(Image.fromarray(pred_u8).resize((w, h), resample=Image.BILINEAR))


def evaluate_sample(
    sample_name: str,
    gt_path: Path,
    pred_mask: np.ndarray,
    gsd_mm_per_px: float = GSD_EFF_MM_PER_PX,
    inference_size: int = INFERENCE_SIZE,
    gt_threshold: int = 128,
    pred_threshold: int = 128,
) -> CrackWidthResult:
    gt_mask = load_mask_array(gt_path)
    gt_mm, pred_mm, err_mm = estimate_crack_width(
        gt_mask,
        pred_mask,
        gsd_mm_per_px=gsd_mm_per_px,
        inference_size=inference_size,
        gt_threshold=gt_threshold,
        pred_threshold=pred_threshold,
    )
    return CrackWidthResult(
        sample=sample_name,
        gt_width_mm=gt_mm,
        predicted_width_mm=pred_mm,
        error_mm=err_mm,
    )


def mean_error(results: Sequence[CrackWidthResult]) -> float:
    errs = [r.error_mm for r in results if not np.isnan(r.error_mm)]
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def print_table(results: Sequence[CrackWidthResult]) -> None:
    headers = ("Sample", "GT width (mm)", "Predicted width (mm)", "Error (mm)")
    rows = [headers]
    for r in results:
        rows.append(
            (
                r.sample,
                f"{r.gt_width_mm:.3f}" if not np.isnan(r.gt_width_mm) else "—",
                f"{r.predicted_width_mm:.3f}" if not np.isnan(r.predicted_width_mm) else "—",
                f"{r.error_mm:.3f}" if not np.isnan(r.error_mm) else "—",
            )
        )
    mae = mean_error(results)
    rows.append(("Mean error", "", "", f"{mae:.3f}" if not np.isnan(mae) else "—"))

    col_w = [max(len(str(row[i])) for row in rows) for i in range(4)]
    for i, row in enumerate(rows):
        line = "  ".join(str(cell).ljust(col_w[j]) for j, cell in enumerate(row))
        print(line)
        if i == 0:
            print("  ".join("-" * col_w[j] for j in range(4)))


def write_csv(results: Sequence[CrackWidthResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample", "gt_width_mm", "predicted_width_mm", "error_mm"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.as_row())
        w.writerow(
            {
                "sample": "Mean error",
                "gt_width_mm": "",
                "predicted_width_mm": "",
                "error_mm": f"{mean_error(results):.3f}",
            }
        )


def load_manifest(path: Path) -> List[Tuple[str, Path, Path, Optional[Path]]]:
    """
    CSV columns: sample,image,gt[,pred]
    Header row optional (auto-detected if first cell is 'sample').
    """
    rows: List[Tuple[str, Path, Path, Optional[Path]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or all(not c.strip() for c in row):
                continue
            if i == 0 and row[0].strip().lower() in ("sample", "name", "id"):
                continue
            if len(row) < 3:
                raise ValueError(f"Manifest row needs sample,image,gt[,pred]: {row}")
            sample = row[0].strip()
            image = Path(row[1].strip())
            gt = Path(row[2].strip())
            pred = Path(row[3].strip()) if len(row) > 3 and row[3].strip() else None
            rows.append((sample, image, gt, pred))
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Physical crack width from GT and BE-TR masks (Table 12).")
    ap.add_argument("--image", type=Path, help="Input RGB image (for BE-TR inference if --pred omitted)")
    ap.add_argument("--gt", type=Path, help="Ground-truth crack mask")
    ap.add_argument("--pred", type=Path, help="BE-TR predicted mask (uint8 prob or binary)")
    ap.add_argument("--model_path", type=Path, help="WPFormer checkpoint; used when --pred is omitted")
    ap.add_argument("--manifest", type=Path, help="CSV: sample,image,gt[,pred]")
    ap.add_argument("--sample_name", type=str, default="Sample", help="Label for single-sample mode")
    ap.add_argument("--output", type=Path, help="Write results CSV")
    ap.add_argument("--gsd", type=float, default=GSD_EFF_MM_PER_PX, help="GSD_eff in mm/pixel (default 4.0)")
    ap.add_argument("--size", type=int, default=INFERENCE_SIZE, help="Mask resize for width (default 384)")
    ap.add_argument("--gt_threshold", type=int, default=128)
    ap.add_argument("--pred_threshold", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model: Optional[nn.Module] = None
    if args.model_path is not None:
        model = build_betr_model(args.model_path, device)

    results: List[CrackWidthResult] = []

    if args.manifest is not None:
        entries = load_manifest(args.manifest)
        for sample, image_p, gt_p, pred_p in entries:
            if pred_p is not None and pred_p.is_file():
                pred_mask = load_mask_array(pred_p)
            elif model is not None:
                pred_mask = predict_betr_mask(image_p, model, device, args.size)
            else:
                raise ValueError(
                    f"Sample {sample}: provide pred mask in manifest or --model_path for inference."
                )
            results.append(
                evaluate_sample(
                    sample,
                    gt_p,
                    pred_mask,
                    gsd_mm_per_px=args.gsd,
                    inference_size=args.size,
                    gt_threshold=args.gt_threshold,
                    pred_threshold=args.pred_threshold,
                )
            )
    elif args.gt is not None:
        image_p = args.image
        if args.pred is not None:
            pred_mask = load_mask_array(args.pred)
        elif model is not None and image_p is not None:
            pred_mask = predict_betr_mask(image_p, model, device, args.size)
        else:
            raise SystemExit("Provide --pred mask and/or --image with --model_path for BE-TR inference.")
        results.append(
            evaluate_sample(
                args.sample_name,
                args.gt,
                pred_mask,
                gsd_mm_per_px=args.gsd,
                inference_size=args.size,
                gt_threshold=args.gt_threshold,
                pred_threshold=args.pred_threshold,
            )
        )
    else:
        raise SystemExit("Use --gt (single sample) or --manifest (batch).")

    print_table(results)
    if args.output is not None:
        write_csv(results, args.output)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
