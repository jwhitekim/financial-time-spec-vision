"""
metrics.py
==========
논문에서 사용하는 평가 지표 구현.

  - SMAPE : Symmetric Mean Absolute Percentage Error  [0, 2], 낮을수록 좋음
  - MASE  : Mean Absolute Scaled Error                낮을수록 좋음
  - SignAccuracy : 3-class (up / flat / down) 방향 정확도  [0, 1], 높을수록 좋음
"""

from __future__ import annotations

import numpy as np
import torch


# ── SMAPE ────────────────────────────────────────────────────────────────────
def smape(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    SMAPE = mean( |y - ŷ| / ((|y| + |ŷ|) / 2 + ε) )

    범위: [0, 2]  (0 = 완벽, 2 = 최악)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    num   = np.abs(y_true - y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / denom))


# ── MASE ─────────────────────────────────────────────────────────────────────
def mase(
    y_true:  np.ndarray | torch.Tensor,
    y_pred:  np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE(y, ŷ) / MAE_naive(y_train)
    MAE_naive = mean(|y_train[t] - y_train[t-1]|)  (naive 1-step 예측 오차)

    값 < 1 : naive 예측보다 좋음
    값 > 1 : naive 예측보다 나쁨
    """
    y_true  = _to_numpy(y_true).ravel()
    y_pred  = _to_numpy(y_pred).ravel()
    y_train = _to_numpy(y_train).ravel()

    mae         = np.mean(np.abs(y_true - y_pred))
    naive_scale = np.mean(np.abs(np.diff(y_train))) + eps
    return float(mae / naive_scale)


# ── Sign Accuracy ─────────────────────────────────────────────────────────────
def sign_accuracy(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    y_last: np.ndarray | torch.Tensor,
) -> float:
    """
    2-class 방향 예측 정확도: up(+1) / down(-1).

    threshold 없이 단순 부호 비교.
    sign(forecast - last_input) vs sign(true - last_input)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    y_last = _to_numpy(y_last).reshape(-1, 1)

    delta_true = y_true - y_last
    delta_pred = y_pred - y_last

    return float(np.mean(np.sign(delta_true) == np.sign(delta_pred)))


def sign_accuracy_thresholded(
    y_true:     np.ndarray | torch.Tensor,
    y_pred:     np.ndarray | torch.Tensor,
    y_last:     np.ndarray | torch.Tensor,
    flat_ratio: float = 0.2,
) -> float:
    """
    3-class 방향 예측 정확도: up(+1) / flat(0) / down(-1).

    논문 정의: 예측/실제값에서 입력 윈도우 마지막 관측값(y_last)을 뺀
    변화량의 부호를 비교.  sign(forecast[t] - last_input)

    Parameters
    ----------
    y_true     : (N, forecast_len)  실제 예측 타깃
    y_pred     : (N, forecast_len)  모델 예측값
    y_last     : (N,)               입력 윈도우 마지막 값
    flat_ratio : flat 임계값 = flat_ratio × std(delta_true)  (논문: 0.2)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    y_last = _to_numpy(y_last).reshape(-1, 1)

    delta_true = y_true - y_last
    delta_pred = y_pred - y_last

    threshold = flat_ratio * (np.std(delta_true) + 1e-8)

    def classify(arr: np.ndarray) -> np.ndarray:
        return np.where(arr > threshold, 1,
               np.where(arr < -threshold, -1, 0))

    return float(np.mean(classify(delta_true) == classify(delta_pred)))


# ── 배치 단위 누적 집계 헬퍼 ──────────────────────────────────────────────────
class MetricAccumulator:
    """
    미니배치마다 예측/실제값을 누적한 뒤 에폭 끝에 한꺼번에 계산.

    Usage
    -----
    acc = MetricAccumulator()
    for batch in loader:
        preds, targets = model(batch)
        acc.update(preds, targets)
    results = acc.compute(y_train=train_series)
    """

    def __init__(self):
        self._preds:     list[np.ndarray] = []
        self._truths:    list[np.ndarray] = []
        self._last_vals: list[np.ndarray] = []

    def update(
        self,
        preds:     np.ndarray | torch.Tensor,
        targets:   np.ndarray | torch.Tensor,
        last_vals: np.ndarray | torch.Tensor,
    ):
        self._preds.append(_to_numpy(preds))
        self._truths.append(_to_numpy(targets))
        self._last_vals.append(_to_numpy(last_vals).ravel())

    def reset(self):
        self._preds.clear()
        self._truths.clear()
        self._last_vals.clear()

    def compute(
        self,
        y_train: np.ndarray | None = None,
    ) -> dict[str, float]:
        all_preds     = np.concatenate(self._preds,     axis=0)
        all_truths    = np.concatenate(self._truths,    axis=0)
        all_last_vals = np.concatenate(self._last_vals, axis=0)

        results: dict[str, float] = {
            "smape":                   smape(all_truths, all_preds),
            "sign_accuracy":           sign_accuracy(all_truths, all_preds, all_last_vals),
            "sign_accuracy_threshold": sign_accuracy_thresholded(all_truths, all_preds, all_last_vals),
        }
        if y_train is not None:
            results["mase"] = mase(all_truths, all_preds, y_train)

        return results


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────
def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# ── 빠른 확인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true  = rng.standard_normal(100)
    y_pred  = y_true + rng.standard_normal(100) * 0.1
    y_train = rng.standard_normal(80)

    print(f"SMAPE         : {smape(y_true, y_pred):.4f}")
    print(f"MASE          : {mase(y_true, y_pred, y_train):.4f}")
    print(f"Sign Accuracy : {sign_accuracy(y_true, y_pred):.4f}")

    # 누적 집계 테스트
    acc = MetricAccumulator()
    for i in range(5):
        last = y_train[i*20:(i+1)*20]
        acc.update(y_pred[i*20:(i+1)*20], y_true[i*20:(i+1)*20], last)
    print(f"누적 집계: {acc.compute(y_train=y_train)}")
