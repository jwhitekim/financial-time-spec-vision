"""
train.py
========
논문 ViT-num-spec 시계열 예측 학습 스크립트.

사용 예시:
  python train.py --dataset synthetic
  python train.py --dataset financial
  python train.py --dataset temperature
  python train.py --dataset synthetic --freeze_backbone   # head 만 학습

논문 하이퍼파라미터:
  optimizer : AdamW  (weight_decay=0.05)
  batch_size: 128
  max_epochs: 200
  early_stop: patience=10
  loss      : MSE
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 프로젝트 루트 → sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.dataset import (
    make_synthetic_dataset,
    make_temperature_dataset,
    make_financial_dataset,
)
from src.metrics import MetricAccumulator
from src.model import VisionTransformerForecaster, ViTNumForecaster

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── 재현성 ───────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 데이터 로더 ───────────────────────────────────────────────────────────────
def build_loaders(args):
    use_num         = (args.model_type == "vit_num")
    image_type      = "spec" if use_num else args.model_type.replace("vit_", "")
    # vit_spec → "spec" / vit_lineplot → "lineplot" / vit_intensity → "intensity"
    if image_type not in ("spec", "lineplot", "intensity"):
        image_type = "spec"
    kw_ds = dict(image_type=image_type,
                 scale_max_ratio=args.scale_max_ratio,
                 use_num=use_num)

    if args.dataset == "synthetic":
        train_ds = make_synthetic_dataset("train", **kw_ds)
        val_ds   = make_synthetic_dataset("val",   **kw_ds)
        test_ds  = make_synthetic_dataset("test",  **kw_ds)
    elif args.dataset == "temperature":
        # 논문 설정: 2015→train, 2016→val, 2017→test, 각 4,220개 (422 관측소 × 10 윈도우)
        train_ds = make_temperature_dataset(split="train", **kw_ds)
        val_ds   = make_temperature_dataset(split="val",   **kw_ds)
        test_ds  = make_temperature_dataset(split="test",  **kw_ds)
    elif args.dataset == "financial":
        train_ds = make_financial_dataset(split="train", **kw_ds)
        test_ds  = make_financial_dataset(split="test",  **kw_ds)
        val_size   = max(1, int(len(train_ds) * 0.2))
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(
            train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        raise ValueError(f"알 수 없는 dataset: {args.dataset}")

    kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ── 학습 루프 ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"  train epoch {epoch}", unit="batch", leave=False)
    for imgs, targets, _, __ in pbar:
        imgs    = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss = model(imgs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


# ── 검증 루프 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    acc = MetricAccumulator()
    total_loss = 0.0
    for imgs, targets, last_vals, norm_params in loader:
        imgs    = imgs.to(device)
        targets = targets.to(device)
        preds   = model(imgs)
        loss    = nn.functional.mse_loss(preds, targets)
        total_loss += loss.item() * imgs.size(0)

        # 역정규화: 원래 스케일에서 SMAPE/MASE 계산 (논문 5.3절)
        nm = norm_params[:, 0:1]          # (B, 1) norm_min
        nr = norm_params[:, 1:2]          # (B, 1) norm_range
        preds_inv   = preds.cpu()   * nr + nm
        targets_inv = targets.cpu() * nr + nm
        last_inv    = last_vals * nr.squeeze(1) + nm.squeeze(1)
        acc.update(preds_inv, targets_inv, last_inv)

    results = acc.compute()
    results["mse"] = total_loss / len(loader.dataset)
    return results


# ── 모델 저장 ─────────────────────────────────────────────────────────────────
def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"체크포인트 저장: {path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ViT 시계열 예측 학습")

    # 데이터
    parser.add_argument("--dataset", choices=["synthetic", "temperature", "financial"],
                        default="synthetic")
    parser.add_argument("--num_workers", type=int, default=4)

    # 모델 타입
    parser.add_argument(
        "--model_type",
        choices=["vit_spec", "vit_num", "vit_lineplot", "vit_intensity"],
        default="vit_spec",
        help=(
            "vit_spec      : CWT 스펙트로그램 이미지 입력 (논문 제안)\n"
            "vit_num       : 수치 시계열 직접 입력 (ViT-num 베이스라인)\n"
            "vit_lineplot  : 꺾은선 그래프 이미지 입력 (ViT-lineplot 베이스라인)\n"
            "vit_intensity : intensity-only 이미지 입력 (어블레이션)"
        ),
    )
    parser.add_argument(
        "--scale_max_ratio",
        type=float,
        default=0.5,
        help="CWT 최대 스케일 = input_len × ratio (기본 0.5). "
             "vit_spec 에서만 유효. 추천: 0.25, 0.5, 1.0, 2.0",
    )
    parser.add_argument(
        "--patch_size_1d",
        type=int,
        default=10,
        help="ViT-num 전용: 1D 패치 크기 (input_len 의 약수여야 함, 기본 10)",
    )

    # 모델 하이퍼파라미터 (논문: embed_dim=128, patch=16, num_heads=4)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=6)
    parser.add_argument("--num_heads",  type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.1)

    # 학습
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience",     type=int,   default=10,
                        help="Early stopping patience")
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "plateau"],
        default="cosine",
        help=(
            "plateau : ReduceLROnPlateau (val_mse 기준, adaptive — 권장)\n"
            "cosine  : CosineAnnealingLR T_max=epochs (원래 설정, 비교용)"
        ),
    )
    parser.add_argument("--seed",         type=int,   default=42)

    # 출력
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--name",       default=None,
                        help="실행 이름 (None → {dataset}_{model_type} 사용)")

    args = parser.parse_args()

    if args.name is None:
        tag = args.model_type
        if args.model_type == "vit_spec" and args.scale_max_ratio != 0.5:
            tag += f"_s{args.scale_max_ratio}"
        args.name = f"{args.dataset}_{tag}"

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"장치: {device}  모델: {args.model_type}  "
                f"데이터: {args.dataset}")

    forecast_len = 10 if args.dataset == "temperature" else 20
    input_len    = 50 if args.dataset == "temperature" else 80

    # ── 데이터 로더 ──────────────────────────────────────────────────────────
    logger.info(f"데이터셋 준비: {args.dataset}")
    train_loader, val_loader, test_loader = build_loaders(args)
    logger.info(
        f"  train={len(train_loader.dataset):,}  "
        f"val={len(val_loader.dataset):,}  "
        f"test={len(test_loader.dataset):,}"
    )

    # ── 모델 ─────────────────────────────────────────────────────────────────
    logger.info(f"모델 초기화: embed_dim={args.embed_dim}, "
                f"layers={args.num_layers}, heads={args.num_heads}")

    if args.model_type == "vit_num":
        model = ViTNumForecaster(
            input_len=input_len,
            patch_size_1d=args.patch_size_1d,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            forecast_len=forecast_len,
        ).to(device)
    else:
        # vit_spec / vit_lineplot / vit_intensity — 모두 이미지 입력
        model = VisionTransformerForecaster(
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            forecast_len=forecast_len,
        ).to(device)

    logger.info(f"학습 파라미터 수: {model.num_parameters() / 1e6:.1f}M")

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(   # 원래 설정
    #     optimizer, T_max=args.epochs, eta_min=1e-6,            # early stopping 43 epoch 시
    # )                                                           # LR이 초기값의 89% — fine-tuning 미진입
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

    # ── 학습 ─────────────────────────────────────────────────────────────────
    best_val_mse = float("inf")
    patience_cnt = 0
    ckpt_path    = os.path.join(args.output_dir, f"{args.name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics["mse"])

        logger.info(
            f"[{epoch:3d}/{args.epochs}]  "
            f"train_mse={train_loss:.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  "
            f"val_smape={val_metrics['smape']:.4f}  "
            f"val_sign_acc={val_metrics['sign_accuracy']:.4f}  "
            f"val_sign_acc_thr={val_metrics['sign_accuracy_threshold']:.4f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            patience_cnt = 0
            save_checkpoint(model, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── 테스트 평가 ───────────────────────────────────────────────────────────
    logger.info("=== Test 평가 ===")
    if not os.path.exists(ckpt_path):
        logger.warning("저장된 체크포인트가 없습니다. 현재 모델로 평가합니다.")
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    logger.info(
        f"  test_mse={test_metrics['mse']:.4f}  "
        f"test_smape={test_metrics['smape']:.4f}  "
        f"test_sign_acc={test_metrics['sign_accuracy']:.4f}  "
        f"test_sign_acc_thr={test_metrics['sign_accuracy_threshold']:.4f}"
    )

    # ── 결과 파일 저장 ────────────────────────────────────────────────────────
    results = {
        "dataset":    args.dataset,
        "name":       args.name,
        "test_mse":               round(test_metrics["mse"],                      6),
        "test_smape":             round(test_metrics["smape"],                    6),
        "test_sign_acc":          round(test_metrics["sign_accuracy"],            6),
        "test_sign_acc_threshold":round(test_metrics["sign_accuracy_threshold"],  6),
    }
    if "mase" in test_metrics:
        results["test_mase"] = round(test_metrics["mase"], 6)

    results_path = os.path.join(args.output_dir, f"{args.name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"결과 저장: {results_path}")


if __name__ == "__main__":
    main()
