import argparse
import os
from pathlib import Path
import time
import json
import sys

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from PIL import Image
import cv2

# local modules (make sure these are importable)
from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from model.WPFormer import WPFormer
from ESDI_dataloader import get_loader

IMG_SIZE = 384

def eval_metrics(test_image_root: Path, test_gt_root: Path, model: torch.nn.Module):
    FM = Fmeasure(); WFM = WeightedFmeasure(); SM = Smeasure(); EM = Emeasure(); M = MAE()

    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    images = sorted([p for p in test_image_root.iterdir() if p.is_file()])
    gts    = sorted([p for p in test_gt_root.iterdir() if p.is_file()])

    model.eval()
    assert len(images) == len(gts), f"#images ({len(images)}) != #gts ({len(gts)})"

    for img_path, gt_path in zip(images, gts):
        ori_image = Image.open(img_path).convert("RGB")
        image = img_transform(ori_image).unsqueeze(0).cuda() if torch.cuda.is_available() else img_transform(ori_image).unsqueeze(0)

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape

        with torch.no_grad():
            preds = model(image)
            res = preds[-1]

        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = Image.fromarray((pred * 255).astype(np.uint8)).convert("L").resize((W, H), resample=Image.BILINEAR)
        pred = np.array(pred)

        FM.step(pred=pred, gt=gt); WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt); EM.step(pred=pred, gt=gt); M.step(pred=pred, gt=gt)

    fm  = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm  = SM.get_results()["sm"]
    em  = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    return {"mae": mae, "wfm": wfm, "sm": sm, "em_mean": em["curve"].mean(), "fm": fm}

def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce = nn.BCELoss()(pred, mask)
    inter = (pred * mask).sum(dim=(2, 3))
    # soft union = sum(pred) + sum(mask) - inter
    union = pred.sum(dim=(2,3)) + mask.sum(dim=(2,3)) - inter + 1e-8
    iou_loss = 1.0 - (inter / union).mean()
    return bce + iou_loss

def train(model_name: str, dataset_name: str, data_root: Path, save_root: Path):
    # epochs per dataset
    if dataset_name == "CrackSeg9k":
        epoch_num, epoch_val = 60, 50
    elif dataset_name == "ZJU-Leaper":
        epoch_num, epoch_val = 24, 20
    elif dataset_name == "ESDIs-SOD":
        epoch_num, epoch_val = 150, 100
    else:
        epoch_num, epoch_val = 24, 17

    net = WPFormer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): net = net.cuda()

    file_dir = data_root

    
    train_image_root = file_dir / dataset_name / "train" / "images"
    train_gt_root    = file_dir / dataset_name / "train" / "gt"
    test_image_root  = file_dir / dataset_name / "test" / "images"
    test_gt_root     = file_dir / dataset_name / "test" / "gt"
    #print(train_image_root)
    for p in [train_image_root, train_gt_root, test_image_root, test_gt_root]:
        if not p.exists():
            raise FileNotFoundError(f"Missing path: {p}")
    train_image_root = str(train_image_root) + os.sep
    train_gt_root    = str(train_gt_root) + os.sep
    #test_image_root = str(test_image_root) + os.sep
    #test_gt_root     = str(test_gt_root) + os.sep

    train_loader = get_loader(str(train_image_root), str(train_gt_root),
                              batchsize=72, trainsize=IMG_SIZE, is_train=True)

    optimizer = optim.Adam(net.parameters(), lr=8e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)

    save_dir = save_root / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_wfm = 0.0
    print("--- start training ---")
    for epoch in range(epoch_num):
        net.train()
        t0 = time.time()
        running_loss = 0.0

        for batch in train_loader:
            inputs, labels = batch['image'], batch['label']
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            preds = net(inputs)

            loss = 0.0
            for p in preds:
                loss = loss + total_loss(p, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        lr_scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{epoch_num} | loss {running_loss:.4f} | time {dt:.1f}s")

        if (epoch + 1) >= epoch_val:
            metrics = eval_metrics(test_image_root, test_gt_root, net)
            wfm = metrics["wfm"]
            if wfm > best_wfm:
                best_wfm = wfm
                ckpt = save_dir / f"{model_name}-{dataset_name}-{best_wfm:.4f}.pth"
                torch.save(net.state_dict(), ckpt)
                print(f"[BEST] wFmeasure={best_wfm:.4f} -> saved {ckpt}")
            print(f"[VAL] mae={metrics['mae']:.4f} best_wfm={best_wfm:.4f} wfm={wfm:.4f}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ESDIs-SOD")
    ap.add_argument("--data_root", default="datasets")
    ap.add_argument("--save_root", default="save")
    ap.add_argument("--model_name", default="WPFormer")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model_name,
        dataset_name=args.dataset,
        data_root=Path(args.data_root),
        save_root=Path(args.save_root),
    )
