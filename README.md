# Financial Time Series Spectrogram Vision

논문 **"From Pixels to Predictions: Spectrogram and Vision Transformer for Better Time Series Forecasting"** (ICAIF 2023) 구현체.

시계열 데이터를 Morlet CWT 스펙트로그램 이미지로 변환한 뒤 Vision Transformer로 예측하는 **ViT-num-spec** 모델.

---

## 구조

```
financial_time_spec_vision/
├── data/
│   ├── raw/
│   │   ├── sp500_close.csv
│   │   └── temperature_rain_dataset_without_missing_values.tsf
│   ├── fetch_financial.py       # S&P 500 주가 다운로드
│   ├── fetch_temperature.py     # 기온 데이터 파싱
│   └── generate_synthetic.py   # 합성 시계열 생성
├── src/
│   ├── spectrogram.py           # Morlet CWT → 128×128 이미지
│   ├── dataset.py               # TimeSeriesDataset (슬라이딩 윈도우)
│   ├── model.py                 # VisionTransformerForecaster
│   └── metrics.py               # SMAPE / MASE / Sign Accuracy
├── train.py                     # 학습 진입점
├── requirements.txt
└── README.md
```

---

## 이미지 구조 (128×128)

```
rows  0-15  : Intensity Stripe  — 원본 시계열 값 (부호 보존, [0-255])
rows 16-127 : Morlet CWT        — 112 스케일, 고주파 위 / 저주파 아래
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 데이터 준비

```bash
# S&P 500 (인터넷 필요, 시간 소요)
python data/fetch_financial.py

# Synthetic (로컬 생성)
python data/generate_synthetic.py
```

---

## 학습

```bash
# Synthetic
python train.py --dataset synthetic

# Financial (로컬 pytorch_model.bin 사용)
python train.py --dataset financial --hf_model pytorch_model.bin

# Temperature
python train.py --dataset temperature --epochs 200 --batch_size 128
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `synthetic` | `synthetic` / `financial` / `temperature` |
| `--hf_model` | `google/vit-base-patch16-224` | HuggingFace ID 또는 `.bin` 경로 |
| `--epochs` | `200` | 최대 에폭 수 |
| `--batch_size` | `128` | 배치 크기 |
| `--lr` | `1e-4` | 학습률 |
| `--patience` | `10` | Early stopping patience |
| `--freeze_backbone` | `False` | backbone 동결 (head 만 학습) |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| SMAPE | Symmetric Mean Absolute Percentage Error (낮을수록 좋음) |
| MASE | Mean Absolute Scaled Error (낮을수록 좋음) |
| Sign Accuracy | 상승/보합/하락 3-class 방향 정확도 (높을수록 좋음) |

---

## 논문

> Medina et al., *From Pixels to Predictions: Spectrogram and Vision Transformer for Better Time Series Forecasting*, ICAIF 2023.
> https://dl.acm.org/doi/10.1145/3604237.3626905
