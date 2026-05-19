"""
model.py
========
논문 ViT-num-spec scratch 구현.

논문 명시 파라미터:
  embed_dim  : 128
  patch_size : 16×16
  img_size   : 128  → num_patches = (128/16)² = 64
  num_heads  : 4    (head_dim = 32)
  num_layers : 6    (논문 미명시, 소형 ViT 표준)
  mlp_ratio  : 4.0
"""

import torch
import torch.nn as nn


# ── Patch Embedding ───────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2
        self.proj      = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x   = self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                 # (B, N+1, D)
        return x + self.pos_embed


# ── Transformer Block ─────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ── Vision Transformer Forecaster ────────────────────────────────────────────
class VisionTransformerForecaster(nn.Module):
    """
    128×128 스펙트로그램 이미지 → 다단계 시계열 예측 (scratch 학습).

    Parameters
    ----------
    embed_dim    : Transformer 잠재 임베딩 차원 (논문: 128)
    num_layers   : Transformer 블록 수          (논문 미명시, 기본 6)
    num_heads    : Multi-head attention 헤드 수 (논문: 4)
    forecast_len : 예측 horizon
    """

    def __init__(
        self,
        img_size:     int   = 128,
        patch_size:   int   = 16,
        embed_dim:    int   = 128,
        num_layers:   int   = 6,
        num_heads:    int   = 4,
        mlp_ratio:    float = 4.0,
        dropout:      float = 0.1,
        forecast_len: int   = 20,
    ):
        super().__init__()
        self.forecast_len = forecast_len

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.blocks      = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ── 회귀 헤드: CLS 토큰 → forecast_len ──────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, forecast_len),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:       torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x     = self.patch_embed(x)
        x     = self.blocks(x)
        cls   = self.norm(x[:, 0])    # CLS 토큰  [B, embed_dim]
        preds = self.head(cls)         # [B, forecast_len]

        if targets is not None:
            return nn.functional.mse_loss(preds, targets)
        return preds

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )


# ── ViT-num: 수치 시계열 직접 입력 ViT ───────────────────────────────────────
class Num1DEmbedding(nn.Module):
    """
    1D 시계열 → 1D 패치 임베딩 (ViT-1D).

    input_len 을 patch_size_1d 크기로 나눠 패치를 만들고,
    각 패치를 embed_dim 으로 선형 투영.
    """
    def __init__(self, input_len: int, patch_size_1d: int, embed_dim: int):
        super().__init__()
        assert input_len % patch_size_1d == 0, \
            f"input_len({input_len}) must be divisible by patch_size_1d({patch_size_1d})"
        n_patches = input_len // patch_size_1d
        self.patch_size_1d = patch_size_1d
        self.proj      = nn.Linear(patch_size_1d, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_len)
        B  = x.shape[0]
        p  = self.patch_size_1d
        x  = x.reshape(B, -1, p)              # (B, n_patches, p)
        x  = self.proj(x)                      # (B, n_patches, D)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)       # (B, n_patches+1, D)
        return x + self.pos_embed


class ViTNumForecaster(nn.Module):
    """
    원시 수치 시계열 → 다단계 예측 (ViT-num 베이스라인).

    이미지 변환 없이 1D 시계열 패치를 직접 Transformer 에 입력.
    논문에서 스펙트로그램 대비 성능 비교 베이스라인으로 사용.

    Parameters
    ----------
    input_len     : 입력 시계열 길이 (financial/synthetic: 80, temperature: 50)
    patch_size_1d : 1D 패치 크기  (input_len 의 약수여야 함)
                    예) input_len=80, patch_size_1d=10 → 8패치
    embed_dim     : 논문 128
    """

    def __init__(
        self,
        input_len:     int   = 80,
        patch_size_1d: int   = 10,
        embed_dim:     int   = 128,
        num_layers:    int   = 6,
        num_heads:     int   = 4,
        mlp_ratio:     float = 4.0,
        dropout:       float = 0.1,
        forecast_len:  int   = 20,
    ):
        super().__init__()
        self.forecast_len = forecast_len

        self.num_embed = Num1DEmbedding(input_len, patch_size_1d, embed_dim)
        self.blocks    = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, forecast_len),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:       torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, input_len)
        x     = self.num_embed(x)
        x     = self.blocks(x)
        cls   = self.norm(x[:, 0])
        preds = self.head(cls)

        if targets is not None:
            return nn.functional.mse_loss(preds, targets)
        return preds

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )


# ── 빠른 확인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ViT-spec (이미지 입력)
    vit_spec = VisionTransformerForecaster(embed_dim=128, num_layers=6, forecast_len=20)
    print(f"[ViT-spec] 파라미터: {vit_spec.num_parameters() / 1e6:.2f}M")
    dummy_img = torch.randn(4, 3, 128, 128)
    print(f"[ViT-spec] 출력 shape: {vit_spec(dummy_img).shape}")

    # ViT-num (수치 입력)
    vit_num = ViTNumForecaster(input_len=80, patch_size_1d=10,
                               embed_dim=128, num_layers=6, forecast_len=20)
    print(f"[ViT-num ] 파라미터: {vit_num.num_parameters() / 1e6:.2f}M")
    dummy_ts = torch.randn(4, 80)
    print(f"[ViT-num ] 출력 shape: {vit_num(dummy_ts).shape}")

    tgt  = torch.randn(4, 20)
    loss = vit_num(dummy_ts, tgt)
    print(f"[ViT-num ] loss: {loss.item():.4f}")
