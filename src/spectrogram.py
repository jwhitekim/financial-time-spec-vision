"""
spectrogram.py
==============
논문 "From Pixels to Predictions: Spectrogram and Vision Transformer
for Better Time Series Forecasting" (ICAIF 2023) 의 핵심 이미지 변환 모듈.

지원하는 이미지 타입:
  spec      : (기본) Morlet CWT 스펙트로그램 + intensity stripe — 논문 제안
  lineplot  : 단순 꺾은선 그래프 이미지                         — ViT-lineplot 베이스라인
  intensity : intensity stripe 만 128×128 로 확장               — 어블레이션

스펙트로그램 이미지 구조 (spec 모드):
  rows   0-15  : intensity stripe  (원본 시계열 값, 부호 보존, [0-255])
  rows  16-127 : Morlet CWT 스펙트로그램 magnitude (고주파 위, 저주파 아래)

Scale 파라미터 (논문 Section 3):
  scale_max_ratio : CWT 최대 스케일 = input_len × scale_max_ratio
                    추천 탐색 범위 : 0.25, 0.5 (기본), 1.0, 2.0
"""

import numpy as np
import pywt


# ── 상수 ────────────────────────────────────────────────────────────────────
IMG_SIZE       = 128
STRIPE_ROWS    = 16
SPEC_ROWS      = IMG_SIZE - STRIPE_ROWS   # 112
MORLET_WAVELET = "cmor1.5-1.0"

IMAGE_TYPES    = ("spec", "lineplot", "intensity")


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────
def _minmax_norm(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a_min, a_max = arr.min(), arr.max()
    denom = a_max - a_min
    return (arr - a_min) / (denom if denom > eps else 1.0)


def _resize_cols(arr2d: np.ndarray, target_width: int) -> np.ndarray:
    T = arr2d.shape[1]
    if T == target_width:
        return arr2d
    x_old = np.arange(T)
    x_new = np.linspace(0, T - 1, target_width)
    return np.array([np.interp(x_new, x_old, row) for row in arr2d])


def _normalize_series(series: np.ndarray) -> np.ndarray:
    """시계열 → [-1, 1] 정규화."""
    s_min, s_max = series.min(), series.max()
    denom = (s_max - s_min) if (s_max - s_min) > 1e-8 else 1.0
    return 2.0 * (series - s_min) / denom - 1.0


# ── CWT ──────────────────────────────────────────────────────────────────────
def morlet_cwt(
    series: np.ndarray,
    n_scales: int = SPEC_ROWS,
    scale_max_ratio: float = 0.5,
) -> np.ndarray:
    """
    Morlet CWT magnitude 계산.

    Parameters
    ----------
    series          : 1D numpy array (정규화 권장)
    n_scales        : 주파수 스케일 수 (기본 112)
    scale_max_ratio : 최대 스케일 = T × scale_max_ratio  (기본 0.5 = T/2)
                      논문 어블레이션 후보: 0.25, 0.5, 1.0, 2.0

    Returns
    -------
    magnitude : (n_scales, T)  인덱스 0 = 고주파(작은 스케일)
    """
    T      = len(series)
    series = np.asarray(series, dtype=np.float64)
    max_sc = max(T * scale_max_ratio, 2.0)
    scales = np.geomspace(1, max_sc, n_scales)
    coef, _ = pywt.cwt(series, scales, MORLET_WAVELET)
    return np.abs(coef)   # (n_scales, T)


# ── 이미지 빌더 ───────────────────────────────────────────────────────────────
def build_spectrogram_image(
    series: np.ndarray,
    img_size: int = IMG_SIZE,
    scale_max_ratio: float = 0.5,
) -> np.ndarray:
    """
    1D 시계열 → 128×128 RGB 스펙트로그램 이미지.

    rows  0-15  : intensity stripe
    rows 16-127 : Morlet CWT spectrogram (고주파 상단)
    """
    stripe_rows = STRIPE_ROWS
    spec_rows   = img_size - stripe_rows

    series_norm = _normalize_series(series)   # [-1, 1]

    # Intensity stripe
    stripe_1d = ((series_norm + 1.0) / 2.0 * 255.0).clip(0, 255)
    stripe_2d = np.tile(stripe_1d, (stripe_rows, 1))
    stripe_2d = _resize_cols(stripe_2d, img_size).clip(0, 255).astype(np.uint8)

    # Morlet CWT spectrogram
    magnitude = morlet_cwt(series_norm, n_scales=spec_rows,
                           scale_max_ratio=scale_max_ratio)
    mag_norm  = _minmax_norm(magnitude) * 255.0
    mag_norm  = _resize_cols(mag_norm, img_size).clip(0, 255).astype(np.uint8)
    mag_norm  = mag_norm[::-1].copy()   # 고주파 → 상단

    gray = np.vstack([stripe_2d, mag_norm])         # (128, 128)
    return np.stack([gray, gray, gray], axis=-1)    # (128, 128, 3) uint8


def build_lineplot_image(
    series: np.ndarray,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    1D 시계열 → 128×128 RGB 꺾은선 그래프 이미지 (ViT-lineplot 베이스라인).

    흰 배경에 검은 선: matplotlib 의존 없이 픽셀 단위로 직접 그린다.
    """
    s_min, s_max = series.min(), series.max()
    denom = (s_max - s_min) if (s_max - s_min) > 1e-8 else 1.0
    s_norm = (series - s_min) / denom   # [0, 1]

    # img_size 열로 리사이즈
    T      = len(series)
    x_old  = np.arange(T)
    x_new  = np.linspace(0, T - 1, img_size)
    s_col  = np.interp(x_new, x_old, s_norm)   # (img_size,)

    img = np.full((img_size, img_size), 255, dtype=np.uint8)   # 흰 배경

    rows = ((1.0 - s_col) * (img_size - 1)).clip(0, img_size - 1).astype(int)

    for col in range(img_size):
        img[rows[col], col] = 0                 # 현재 점
        if col > 0:
            r0, r1 = rows[col - 1], rows[col]
            r_lo, r_hi = min(r0, r1), max(r0, r1)
            img[r_lo : r_hi + 1, col] = 0      # 수직 선분 연결

    return np.stack([img, img, img], axis=-1)   # (128, 128, 3) uint8


def build_intensity_only_image(
    series: np.ndarray,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Intensity stripe 만 128×128 로 확장한 이미지 (스펙트로그램 제거 어블레이션).

    시계열 진폭 정보만 포함 (CWT 주파수 정보 없음).
    """
    series_norm = _normalize_series(series)   # [-1, 1]
    stripe_1d   = ((series_norm + 1.0) / 2.0 * 255.0).clip(0, 255)
    stripe_2d   = np.tile(stripe_1d, (img_size, 1))        # (img_size, T)
    stripe_2d   = _resize_cols(stripe_2d, img_size).clip(0, 255).astype(np.uint8)
    return np.stack([stripe_2d, stripe_2d, stripe_2d], axis=-1)


def build_image(
    series: np.ndarray,
    image_type: str = "spec",
    img_size: int = IMG_SIZE,
    scale_max_ratio: float = 0.5,
) -> np.ndarray:
    """
    image_type 에 따라 적절한 128×128 RGB 이미지를 반환하는 디스패처.

    image_type
    ----------
    "spec"      : Morlet CWT 스펙트로그램 + intensity stripe (논문 제안)
    "lineplot"  : 꺾은선 그래프 (ViT-lineplot 베이스라인)
    "intensity" : intensity stripe 만 확장 (어블레이션)
    """
    if image_type == "spec":
        return build_spectrogram_image(series, img_size, scale_max_ratio)
    elif image_type == "lineplot":
        return build_lineplot_image(series, img_size)
    elif image_type == "intensity":
        return build_intensity_only_image(series, img_size)
    else:
        raise ValueError(f"알 수 없는 image_type: {image_type!r}. "
                         f"선택: {IMAGE_TYPES}")


# ── 디버그 / 시각화 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.linspace(0, 1, 80)
    test_series = (
        np.sin(2 * np.pi * 5 * t)
        + 0.5 * np.sin(2 * np.pi * 20 * t)
        + 0.1 * np.random.randn(80)
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        ("spec (ratio=0.25)", {"scale_max_ratio": 0.25}),
        ("spec (ratio=0.5)",  {"scale_max_ratio": 0.5}),
        ("spec (ratio=1.0)",  {"scale_max_ratio": 1.0}),
        ("spec (ratio=2.0)",  {"scale_max_ratio": 2.0}),
    ]
    for ax, (title, kw) in zip(axes, titles):
        img = build_spectrogram_image(test_series, **kw)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("scale_ablation.png", dpi=100)
    print("저장: scale_ablation.png")

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, itype in zip(axes2, IMAGE_TYPES):
        img = build_image(test_series, image_type=itype)
        ax.imshow(img)
        ax.set_title(itype)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("image_type_comparison.png", dpi=100)
    print("저장: image_type_comparison.png")
