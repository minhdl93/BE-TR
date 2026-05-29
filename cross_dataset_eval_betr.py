# -*- coding: utf-8 -*-
"""
BE-TR segmentation on test sets (ESDIs-SOD, CrackSeg9k) + optional cross-dataset metrics.

Saves per-dataset visualizations under --viz_dir:
  {dataset}/panel/     — side-by-side [RGB | GT | Prediction]
  {dataset}/overlay/   — BE-TR mask overlaid on the input image
  {dataset}/pred/      — predicted saliency maps (0–255)
  {dataset}/gt/        — ground-truth copies
  {dataset}/image/     — input RGB (resized to GT size)

Example — show segmentation on both test folders (in-domain checkpoints):
  python cross_dataset_eval_betr.py \\
      --data_root /data1/cvpr/bridge/UPFormer/datasets \\
      --save_root /data1/cvpr/bridge/UPFormer/save \\
      --viz_dir results/betr_segmentation_viz

Single checkpoint on both datasets:
  python cross_dataset_eval_betr.py \\
      --model_path save/ESDIs-SOD/WPFormer-ESDIs-SOD-0.022.pth \\
      --test_datasets ESDIs-SOD CrackSeg9k \\
      --viz_dir results/betr_segmentation_viz

Cross-dataset metrics table only (no images):
  python cross_dataset_eval_betr.py --metrics --no_viz \\
      --train_datasets ESDIs-SOD CrackSeg9k \\
      --test_datasets ESDIs-SOD CrackSeg9k minh
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from benchmark_efficiency_table import (
    IMAGENET_NORM,
    IMG_SIZE,
    SegmentationWrapper,
    load_weights,
)
from model.WPFormer import WPFormer
from sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GT_DIR_CANDIDATES = ("gt", "GT", "mask", "masks", "label", "labels")
IMAGE_DIR_CANDIDATES = ("images", "image", "imgs", "Imgs")

DEFAULT_DATA_ROOT = Path("/data1/cvpr/bridge/UPFormer/datasets")
DEFAULT_SAVE_ROOT = Path("/data1/cvpr/bridge/UPFormer/save")
DEFAULT_VIZ_DATASETS = ("ESDIs-SOD", "CrackSeg9k")
OVERLAY_COLOR_BGR = (255, 200, 0)  # cyan-ish on BGR for crack/defect highlight


@dataclass
class EvalResult:
    train_dataset: str
    test_dataset: str
    checkpoint: str
    num_samples: int
    mae: float
    wfm: float
    sm: float
    mean_fm: float
    mean_em: float

    def as_dict(self) -> Dict:
        d = asdict(self)
        for k in ("mae", "wfm", "sm", "mean_fm", "mean_em"):
            d[k] = round(d[k], 4)
        return d


def gt_path_for_image(gt_dir: Path, image_path: Path) -> Optional[Path]:
    stem = image_path.stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        for name in (f"{stem}{ext}", f"{stem}_gt{ext}"):
            cand = gt_dir / name
            if cand.is_file():
                return cand
    return None


def resolve_subdir(parent: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = parent / name
        if p.is_dir():
            return p
    return None


def resolve_test_split(dataset_root: Path) -> Tuple[Path, Path]:
    test_root = dataset_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test split: {test_root}")

    image_dir = resolve_subdir(test_root, IMAGE_DIR_CANDIDATES)
    if image_dir is None:
        raise FileNotFoundError(
            f"No image folder under {test_root} (tried {IMAGE_DIR_CANDIDATES})"
        )

    gt_dir = resolve_subdir(test_root, GT_DIR_CANDIDATES)
    if gt_dir is None:
        raise FileNotFoundError(
            f"No GT folder under {test_root} (tried {GT_DIR_CANDIDATES})"
        )
    return image_dir, gt_dir


def list_image_gt_pairs(image_dir: Path, gt_dir: Path) -> List[Tuple[str, Path, Path]]:
    images = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    pairs: List[Tuple[str, Path, Path]] = []
    missing: List[str] = []
    for img in images:
        gt = gt_path_for_image(gt_dir, img)
        if gt is None:
            missing.append(img.name)
            continue
        pairs.append((img.stem, img, gt))
    if not pairs:
        raise RuntimeError(f"No image/GT pairs under {image_dir} and {gt_dir}")
    if missing:
        print(f"  Warning: skipped {len(missing)} images with no GT (e.g. {missing[0]})")
    return pairs


def build_betr_model(model_path: Path, device: torch.device) -> nn.Module:
    net = WPFormer(channel=64, num_queries=16)
    load_weights(net, model_path)
    return SegmentationWrapper(net).to(device).eval()


@torch.no_grad()
def predict_saliency(
    image_path: Path,
    model: nn.Module,
    device: torch.device,
    inference_size: int = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pred uint8 H×W, rgb uint8 H×W at GT resolution)."""
    pil = Image.open(image_path).convert("RGB")
    h, w = pil.size[1], pil.size[0]
    resized = pil.resize((inference_size, inference_size), resample=Image.BILINEAR)
    to_tensor = transforms.ToTensor()
    tensor = IMAGENET_NORM(to_tensor(resized)).unsqueeze(0).to(device)
    pred = model(tensor)
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    res = torch.sigmoid(pred).cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    pred_u8 = (res * 255).astype(np.uint8)
    pred_full = np.array(Image.fromarray(pred_u8).resize((w, h), resample=Image.BILINEAR))
    rgb_full = np.array(pil.resize((w, h), resample=Image.BILINEAR))
    return pred_full, rgb_full


def make_panel(rgb: np.ndarray, gt_gray: np.ndarray, pred_gray: np.ndarray) -> np.ndarray:
    """3-column panel: [RGB | GT | Prediction]."""
    gt3 = np.stack([gt_gray] * 3, axis=-1)
    pred3 = np.stack([pred_gray] * 3, axis=-1)
    return np.hstack([rgb, gt3, pred3])


def overlay_pred_on_rgb(
    rgb: np.ndarray,
    pred_gray: np.ndarray,
    alpha: float = 0.55,
    threshold: int = 128,
) -> np.ndarray:
    """Semi-transparent defect overlay on the input image."""
    fg = pred_gray >= threshold
    if not np.any(fg):
        return rgb.copy()
    base = rgb.astype(np.float32)
    color = np.array(OVERLAY_COLOR_BGR[::-1], dtype=np.float32)  # RGB overlay tint
    fg3 = fg.astype(np.float32)[..., None]
    blended = base * (1.0 - alpha * fg3) + color * (alpha * fg3)
    out = np.clip(blended, 0, 255).astype(np.uint8)
    mask_u8 = (fg.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color.astype(int).tolist(), 2, lineType=cv2.LINE_AA)
    return out


def ensure_viz_dirs(viz_root: Path, dataset_name: str) -> Dict[str, Path]:
    base = viz_root / dataset_name
    subdirs = ("image", "gt", "pred", "pred_thr", "panel", "overlay")
    paths: Dict[str, Path] = {}
    for name in subdirs:
        p = base / name
        p.mkdir(parents=True, exist_ok=True)
        paths[name] = p
    return paths


def save_sample_visuals(
    stem: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out: Dict[str, Path],
    save_thresholded: bool = True,
) -> None:
    cv2.imwrite(str(out["pred"] / f"{stem}_pred.png"), pred)
    cv2.imwrite(str(out["gt"] / f"{stem}_gt.png"), gt)
    cv2.imwrite(
        str(out["image"] / f"{stem}_rgb.png"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    if save_thresholded:
        thr = ((pred >= 128).astype(np.uint8) * 255)
        cv2.imwrite(str(out["pred_thr"] / f"{stem}_pred_thr.png"), thr)

    panel = make_panel(rgb, gt, pred)
    cv2.imwrite(
        str(out["panel"] / f"{stem}_panel.png"),
        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
    )

    overlay = overlay_pred_on_rgb(rgb, pred)
    cv2.imwrite(
        str(out["overlay"] / f"{stem}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


def evaluate_and_visualize_dataset(
    pairs: List[Tuple[str, Path, Path]],
    model: nn.Module,
    device: torch.device,
    dataset_name: str,
    viz_dir: Optional[Path] = None,
    inference_size: int = IMG_SIZE,
    max_samples: Optional[int] = None,
    compute_metrics: bool = True,
) -> Optional[Dict[str, float]]:
    if max_samples is not None:
        pairs = pairs[:max_samples]

    out_dirs = ensure_viz_dirs(viz_dir, dataset_name) if viz_dir else None

    FM = Fmeasure() if compute_metrics else None
    WFM = WeightedFmeasure() if compute_metrics else None
    SM = Smeasure() if compute_metrics else None
    EM = Emeasure() if compute_metrics else None
    M = MAE() if compute_metrics else None

    for i, (stem, img_path, gt_path) in enumerate(pairs):
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Could not read GT: {gt_path}")

        pred, rgb = predict_saliency(img_path, model, device, inference_size)

        if out_dirs is not None:
            save_sample_visuals(stem, rgb, gt, pred, out_dirs)
            if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
                print(f"     saved {i + 1}/{len(pairs)} visualizations")

        if compute_metrics:
            assert FM is not None and WFM is not None and SM is not None
            assert EM is not None and M is not None
            FM.step(pred=pred, gt=gt)
            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            EM.step(pred=pred, gt=gt)
            M.step(pred=pred, gt=gt)

    if not compute_metrics:
        return None

    fm = FM.get_results()["fm"]
    em = EM.get_results()["em"]
    return {
        "mae": float(M.get_results()["mae"]),
        "wfm": float(WFM.get_results()["wfm"]),
        "sm": float(SM.get_results()["sm"]),
        "mean_fm": float(fm["curve"].mean()),
        "mean_em": float(em["curve"].mean()),
    }


def _score_from_checkpoint_name(path: Path) -> float:
    m = re.search(r"([\d.]+)\s*\.pth$", path.name, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return -1.0


def find_checkpoint(save_root: Path, train_dataset: str) -> Path:
    ds_dir = save_root / train_dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ds_dir}")

    patterns = (
        f"WPFormer-{train_dataset}-*.pth",
        f"WPFormer_{train_dataset}*.pth",
        f"*WPFormer*{train_dataset}*.pth",
        "WPFormer*.pth",
        "*.pth",
    )
    candidates: List[Path] = []
    for pat in patterns:
        found = sorted(ds_dir.glob(pat))
        if found:
            candidates = found
            break
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints in {ds_dir}")
    return max(candidates, key=_score_from_checkpoint_name)


def load_checkpoint_map(path: Path) -> Dict[str, Path]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) for k, v in raw.items()}


def format_markdown_table(results: List[EvalResult]) -> str:
    headers = ["Train", "Test", "N", "MAE↓", "wF↑", "S↑", "F↑", "E↑"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        lines.append(
            f"| {r.train_dataset} | {r.test_dataset} | {r.num_samples} "
            f"| {r.mae:.4f} | {r.wfm:.4f} | {r.sm:.4f} "
            f"| {r.mean_fm:.4f} | {r.mean_em:.4f} |"
        )
    return "\n".join(lines)


def format_latex_table(results: List[EvalResult]) -> str:
    lines = [
        r"\begin{tabular}{llrccccc}",
        r"\toprule",
        r"Train & Test & $N$ & MAE$\downarrow$ & $F_\beta^w$$\uparrow$ "
        r"& $S_\alpha$$\uparrow$ & $\bar{F}$$\uparrow$ & $\bar{E}$$\uparrow$ \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(
            f"{r.train_dataset} & {r.test_dataset} & {r.num_samples} "
            f"& {r.mae:.4f} & {r.wfm:.4f} & {r.sm:.4f} "
            f"& {r.mean_fm:.4f} & {r.mean_em:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def run_test_visualization(
    test_datasets: Sequence[str],
    data_root: Path,
    checkpoint_map: Dict[str, Path],
    device: torch.device,
    viz_dir: Path,
    inference_size: int = IMG_SIZE,
    max_samples: Optional[int] = None,
    compute_metrics: bool = True,
) -> List[EvalResult]:
    """Run BE-TR on each test set and save segmentation visualizations."""
    results: List[EvalResult] = []
    viz_dir.mkdir(parents=True, exist_ok=True)

    for test_ds in test_datasets:
        if test_ds not in checkpoint_map:
            raise KeyError(
                f"No checkpoint for {test_ds}. Pass --model_path, --checkpoint_map, "
                f"or ensure save/{test_ds}/ has a WPFormer checkpoint."
            )
        ckpt = checkpoint_map[test_ds]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint for {test_ds}: {ckpt}")

        dataset_root = data_root / test_ds
        print(f"\n=== {test_ds} test set ===")
        print(f"    checkpoint: {ckpt}")
        image_dir, gt_dir = resolve_test_split(dataset_root)
        pairs = list_image_gt_pairs(image_dir, gt_dir)
        print(f"    {len(pairs)} samples | images: {image_dir} | gt: {gt_dir}")
        print(f"    saving visuals -> {viz_dir / test_ds}")

        model = build_betr_model(ckpt, device)
        metrics = evaluate_and_visualize_dataset(
            pairs,
            model,
            device,
            dataset_name=test_ds,
            viz_dir=viz_dir,
            inference_size=inference_size,
            max_samples=max_samples,
            compute_metrics=compute_metrics,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if metrics is not None:
            n_done = len(pairs) if max_samples is None else min(len(pairs), max_samples)
            row = EvalResult(
                train_dataset=test_ds,
                test_dataset=test_ds,
                checkpoint=str(ckpt),
                num_samples=n_done,
                **metrics,
            )
            results.append(row)
            print(
                f"    MAE={row.mae:.4f} wF={row.wfm:.4f} S={row.sm:.4f} "
                f"F={row.mean_fm:.4f} E={row.mean_em:.4f}"
            )

    return results


def run_cross_dataset_eval(
    train_datasets: Sequence[str],
    test_datasets: Sequence[str],
    data_root: Path,
    checkpoint_map: Dict[str, Path],
    device: torch.device,
    inference_size: int = IMG_SIZE,
) -> List[EvalResult]:
    results: List[EvalResult] = []

    for train_ds in train_datasets:
        ckpt = checkpoint_map[train_ds]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint for {train_ds}: {ckpt}")
        print(f"\n=== BE-TR trained on {train_ds} ===")
        print(f"    checkpoint: {ckpt}")
        model = build_betr_model(ckpt, device)

        for test_ds in test_datasets:
            dataset_root = data_root / test_ds
            print(f"\n  -> test on {test_ds}")
            image_dir, gt_dir = resolve_test_split(dataset_root)
            pairs = list_image_gt_pairs(image_dir, gt_dir)

            metrics = evaluate_and_visualize_dataset(
                pairs,
                model,
                device,
                dataset_name=test_ds,
                viz_dir=None,
                inference_size=inference_size,
                compute_metrics=True,
            )
            assert metrics is not None
            results.append(
                EvalResult(
                    train_dataset=train_ds,
                    test_dataset=test_ds,
                    checkpoint=str(ckpt),
                    num_samples=len(pairs),
                    **metrics,
                )
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def save_results(results: List[EvalResult], output_dir: Path) -> None:
    if not results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "cross_dataset_betr.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.as_dict() for r in results], f, indent=2)

    csv_path = output_dir / "cross_dataset_betr.csv"
    fieldnames = list(results[0].as_dict().keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_dict())

    md_path = output_dir / "cross_dataset_betr.md"
    md_path.write_text(format_markdown_table(results), encoding="utf-8")

    tex_path = output_dir / "cross_dataset_betr.tex"
    tex_path.write_text(format_latex_table(results), encoding="utf-8")

    print(f"\nMetrics saved under {output_dir}")


def build_in_domain_checkpoint_map(
    datasets: Sequence[str],
    save_root: Path,
    model_path: Optional[Path],
) -> Dict[str, Path]:
    if model_path is not None:
        return {ds: model_path for ds in datasets}
    return {ds: find_checkpoint(save_root, ds) for ds in datasets}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="BE-TR segmentation visualizations and optional cross-dataset metrics."
    )
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT)
    ap.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="One checkpoint used for every --test_datasets entry",
    )
    ap.add_argument("--checkpoint_map", type=Path, default=None)
    ap.add_argument(
        "--test_datasets",
        nargs="+",
        default=list(DEFAULT_VIZ_DATASETS),
        help="Test splits to run (default: ESDIs-SOD CrackSeg9k)",
    )
    ap.add_argument(
        "--viz_dir",
        type=Path,
        default=Path("results/betr_segmentation_viz"),
        help="Save segmentation panels/overlays here (set --no_viz to disable)",
    )
    ap.add_argument("--no_viz", action="store_true", help="Skip saving visualization images")
    ap.add_argument(
        "--max_viz",
        type=int,
        default=None,
        help="Max test images per dataset (default: all)",
    )
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="Also run full train×test cross-dataset metric matrix",
    )
    ap.add_argument(
        "--train_datasets",
        nargs="+",
        default=None,
        help="Training sets for --metrics cross matrix",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/cross_dataset_betr"),
        help="Metrics table output when --metrics is set",
    )
    ap.add_argument("--inference_size", type=int, default=IMG_SIZE)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    test_datasets = list(args.test_datasets)
    missing = [ds for ds in test_datasets if not (args.data_root / ds).is_dir()]
    if missing:
        raise SystemExit(f"Dataset folders not found under {args.data_root}: {missing}")

    if args.checkpoint_map is not None:
        checkpoint_map = load_checkpoint_map(args.checkpoint_map)
    else:
        checkpoint_map = build_in_domain_checkpoint_map(
            test_datasets, args.save_root, args.model_path
        )

    all_results: List[EvalResult] = []

    if not args.no_viz:
        viz_results = run_test_visualization(
            test_datasets=test_datasets,
            data_root=args.data_root,
            checkpoint_map=checkpoint_map,
            device=device,
            viz_dir=args.viz_dir,
            inference_size=args.inference_size,
            max_samples=args.max_viz,
            compute_metrics=True,
        )
        all_results.extend(viz_results)
        print(f"\nSegmentation results saved under: {args.viz_dir.resolve()}")
        for ds in test_datasets:
            print(f"  {ds}/panel/    — RGB | GT | BE-TR prediction")
            print(f"  {ds}/overlay/  — BE-TR mask on input image")

    if args.metrics:
        train_datasets = args.train_datasets or test_datasets
        if args.checkpoint_map is not None:
            metric_ckpt_map = load_checkpoint_map(args.checkpoint_map)
        elif args.model_path is not None:
            metric_ckpt_map = {ds: args.model_path for ds in train_datasets}
        else:
            metric_ckpt_map = {
                ds: find_checkpoint(args.save_root, ds) for ds in train_datasets
            }
        cross_results = run_cross_dataset_eval(
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            data_root=args.data_root,
            checkpoint_map=metric_ckpt_map,
            device=device,
            inference_size=args.inference_size,
        )
        all_results = cross_results
        print("\n" + format_markdown_table(cross_results))
        save_results(cross_results, args.output_dir)
    elif all_results:
        print("\n" + format_markdown_table(all_results))
        save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
