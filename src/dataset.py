"""
dataset.py
==========
전처리가 완료된 .npz 파일을 로드하는 Dataset 클래스.

전처리는 preprocess.py 에서 별도로 실행한다.
  python preprocess.py --dataset synthetic [--image_type spec|lineplot|intensity]

주요 클래스
-----------
NpzDataset  : 이미지 기반 데이터셋 (ViT-spec / ViT-lineplot / ViT-intensity)
NumDataset  : 수치 시계열 데이터셋 (ViT-num)

image_type 이 lineplot / intensity 인 경우 npz 안의 raw windows 를 읽어
on-the-fly 로 이미지를 생성한다 (preprocess.py 로 미리 만든 npz 도 사용 가능).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


# ── 기본 transform (ViT 입력용) ──────────────────────────────────────────────
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                         # uint8 [0,255] → float [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),     # [0,1] → [-1,1]
])

IMAGE_TYPES = ("spec", "lineplot", "intensity")


# ── NpzDataset ────────────────────────────────────────────────────────────────
class NpzDataset(Dataset):
    """
    preprocess.py 가 저장한 .npz 파일에서 (이미지, 타깃) 쌍을 로드.

    image_type
    ----------
    "spec"      : npz 안의 pre-computed 스펙트로그램 이미지 사용 (기본, 빠름)
    "lineplot"  : npz 안의 raw windows 로 on-the-fly 꺾은선 그래프 생성
    "intensity" : npz 안의 raw windows 로 on-the-fly intensity-only 이미지 생성

    lineplot / intensity 는 windows 키가 npz 에 있어야 한다.
    (preprocess.py ≥ 현재 버전으로 생성한 npz 에는 항상 포함)
    """

    def __init__(self, npz_path: str, transform=None, image_type: str = "spec"):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"전처리 파일 없음: {npz_path}\n"
                f"먼저 실행하세요: python preprocess.py --dataset <name>"
            )
        data = np.load(npz_path)

        self.image_type = image_type

        if image_type == "spec":
            self._images = data["images"]      # (N, 128, 128, 3) uint8
        else:
            # on-the-fly 생성: windows 필요
            if "windows" not in data:
                raise KeyError(
                    f"'{npz_path}' 에 'windows' 키가 없습니다.\n"
                    "최신 preprocess.py 로 재전처리 하세요."
                )
            from src.spectrogram import build_image as _build_image
            self._build_image = _build_image
            self._raw_windows = data["windows"]   # (N, input_len) float32

        self._targets    = data["targets"]     # (N, forecast_len) float32
        self._last_vals  = data["last_vals"]   # (N,) float32
        self._norm_min   = data["norm_min"]    # (N,) float32
        self._norm_range = data["norm_range"]  # (N,) float32
        self.transform   = transform if transform is not None else DEFAULT_TRANSFORM

    def __len__(self) -> int:
        if self.image_type == "spec":
            return len(self._images)
        return len(self._raw_windows)

    def __getitem__(self, idx: int):
        if self.image_type == "spec":
            img_np = self._images[idx]
        else:
            img_np = self._build_image(self._raw_windows[idx],
                                       image_type=self.image_type)

        if self.transform is DEFAULT_TRANSFORM:
            img = (torch.from_numpy(img_np.copy())
                   .permute(2, 0, 1).float().div_(255.0).sub_(0.5).div_(0.5))
        else:
            from PIL import Image
            img = self.transform(Image.fromarray(img_np))

        target      = torch.from_numpy(self._targets[idx])
        last_val    = torch.tensor(self._last_vals[idx],  dtype=torch.float32)
        norm_params = torch.tensor([self._norm_min[idx], self._norm_range[idx]],
                                   dtype=torch.float32)
        return img, target, last_val, norm_params


# ── NumDataset: ViT-num 용 수치 입력 ─────────────────────────────────────────
class NumDataset(Dataset):
    """
    ViT-num 용: 이미지 대신 정규화된 1D 시계열 창을 반환.

    반환 튜플: (window, target, last_val, norm_params)
      window : float32 (input_len,)  — 정규화된 시계열 입력
    """

    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"전처리 파일 없음: {npz_path}\n"
                f"먼저 실행하세요: python preprocess.py --dataset <name>"
            )
        data = np.load(npz_path)
        if "windows" not in data:
            raise KeyError(
                f"'{npz_path}' 에 'windows' 키가 없습니다.\n"
                "최신 preprocess.py 로 재전처리 하세요."
            )
        self._windows    = data["windows"]     # (N, input_len) float32
        self._targets    = data["targets"]
        self._last_vals  = data["last_vals"]
        self._norm_min   = data["norm_min"]
        self._norm_range = data["norm_range"]

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        window      = torch.from_numpy(self._windows[idx])   # (input_len,)
        target      = torch.from_numpy(self._targets[idx])
        last_val    = torch.tensor(self._last_vals[idx],  dtype=torch.float32)
        norm_params = torch.tensor([self._norm_min[idx], self._norm_range[idx]],
                                   dtype=torch.float32)
        return window, target, last_val, norm_params


# ── 팩토리 함수 ───────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _npz_name(base: str, image_type: str, scale_max_ratio: float) -> str:
    """
    npz 파일 이름 결정.

    - spec + scale=0.5  → <base>.npz          (기본)
    - spec + scale≠0.5  → <base>_s{ratio}.npz  (scale 어블레이션)
    - lineplot / intensity → <base>.npz         (base 파일에서 on-the-fly 생성)
    - vit_num             → <base>.npz          (base 파일에서 windows 로드)
    """
    if image_type in ("spec", "lineplot", "intensity"):
        suffix = "" if scale_max_ratio == 0.5 else f"_s{scale_max_ratio}"
    else:
        suffix = ""
    return f"{base}{suffix}.npz"


def _load_dataset(
    base_path:       str,
    transform:       object = None,
    image_type:      str    = "spec",
    scale_max_ratio: float  = 0.5,
    use_num:         bool   = False,
) -> Dataset:
    """
    단일 .npz 또는 청크 분할 파일(*_chunk???.npz) 자동 감지 후 로드.

    use_num=True → NumDataset (ViT-num)
    """
    # 청크 파일 우선 탐색
    chunks = sorted(glob.glob(base_path.replace(".npz", "_chunk*.npz")))
    paths  = chunks if chunks else [base_path]

    if use_num:
        datasets = [NumDataset(p) for p in paths]
    else:
        datasets = [NpzDataset(p, transform, image_type=image_type) for p in paths]

    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


def make_synthetic_dataset(
    split:           str   = "train",
    transform:       object = None,
    image_type:      str   = "spec",
    scale_max_ratio: float = 0.5,
    use_num:         bool  = False,
) -> Dataset:
    name = _npz_name(f"synthetic_{split}", image_type, scale_max_ratio)
    path = os.path.join(PROCESSED_DIR, name)
    return _load_dataset(path, transform, image_type, scale_max_ratio, use_num)


def make_temperature_dataset(
    split:           str   = "train",
    transform:       object = None,
    image_type:      str   = "spec",
    scale_max_ratio: float = 0.5,
    use_num:         bool  = False,
) -> Dataset:
    name = _npz_name(f"temperature_{split}", image_type, scale_max_ratio)
    path = os.path.join(PROCESSED_DIR, name)
    return _load_dataset(path, transform, image_type, scale_max_ratio, use_num)


def make_financial_dataset(
    split:           str   = "train",
    transform:       object = None,
    image_type:      str   = "spec",
    scale_max_ratio: float = 0.5,
    use_num:         bool  = False,
) -> Dataset:
    name = _npz_name(f"financial_{split}", image_type, scale_max_ratio)
    path = os.path.join(PROCESSED_DIR, name)
    return _load_dataset(path, transform, image_type, scale_max_ratio, use_num)


# ── 윈도우 생성 헬퍼 (preprocess.py 에서 사용) ───────────────────────────────
def forward_fill(series: np.ndarray) -> np.ndarray:
    series = np.array(series, dtype=np.float64)
    mask = np.isnan(series)
    if not mask.any():
        return series
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, out=idx)
    series[mask] = series[idx[mask]]
    return np.where(np.isnan(series), 0.0, series)


def make_windows(
    series_list: list[np.ndarray],
    input_len: int,
    forecast_len: int,
    stride: int = 1,
    max_windows_per_series: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """시계열 리스트 → 정규화된 (windows, targets, last_vals) 리스트.

    last_vals: 각 윈도우의 마지막 입력값 (정규화 스케일).
               sign_accuracy 계산 시 방향성 기준점으로 사용.
    max_windows_per_series: 시리즈당 최대 윈도우 수 (None이면 무제한).
    """
    windows, targets, last_vals, norm_mins, norm_ranges = [], [], [], [], []
    needed = input_len + forecast_len
    for raw in series_list:
        series = forward_fill(np.asarray(raw, dtype=np.float64))
        if len(series) < needed:
            continue
        count = 0
        for start in range(0, len(series) - needed + 1, stride):
            if max_windows_per_series is not None and count >= max_windows_per_series:
                break
            w = series[start : start + input_len]
            t = series[start + input_len : start + input_len + forecast_len]
            denom = (w.max() - w.min()) or 1.0
            w_min  = np.float32(w.min())
            w_norm = ((w - w_min) / denom).astype(np.float32)
            t_norm = ((t - w_min) / denom).astype(np.float32)
            windows.append(w_norm)
            targets.append(t_norm)
            last_vals.append(np.float32(w_norm[-1]))
            norm_mins.append(w_min)
            norm_ranges.append(np.float32(denom))
            count += 1
    return windows, targets, last_vals, norm_mins, norm_ranges
