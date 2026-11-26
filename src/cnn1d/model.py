"""D — Kiến trúc 1D-CNN có dùng mask và kiểm tra bất biến với padding.

Conv1d không tự biết đâu là padding. Mô hình xóa padding trước mỗi lớp conv và
chỉ lấy trung bình trên packet thật, nên giá trị padding không ảnh hưởng output.
"""
from __future__ import annotations

import torch
from torch import nn


class MaskedConv1D(nn.Module):
    def __init__(self, input_channels=3, conv_channels=(32, 64), kernel_size=3,
                 padding="same", pooling="masked_mean", dropout=0.2, output_logits=1):
        super().__init__()
        # Chỉ hỗ trợ đúng baseline trong configs/cnn1d.yaml
        assert padding == "same" and pooling == "masked_mean" and output_logits == 1
        convs, c = [], input_channels
        for out in conv_channels:
            convs.append(nn.Conv1d(c, out, kernel_size, padding="same"))
            c = out
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (N, C, L), mask: (N, L) True = packet thật. Trả về logits (N,)."""
        m = mask.unsqueeze(1).to(x.dtype)  # (N, 1, L)
        h = x * m  # xóa giá trị padding trước conv đầu
        for conv in self.convs:
            # Zero lại padding sau mỗi lớp: bias/ReLU làm padding khác 0, nếu để nguyên
            # thì kernel của packet thật ở rìa sẽ đọc phải
            h = torch.relu(conv(h)) * m
        pooled = h.sum(dim=2) / m.sum(dim=2).clamp_min(1.0)  # masked mean pooling
        return self.head(self.dropout(pooled)).squeeze(1)


def build_model(model_cfg: dict) -> MaskedConv1D:
    return MaskedConv1D(**model_cfg)


@torch.no_grad()
def check_mask_invariance(model, X, mask, atol=1e-6, rtol=1e-5, seed=42) -> dict:
    """Thay padding bằng nhiễu lớn, giữ packet thật + mask: score ở eval phải không đổi."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    noise = (torch.randn(X.shape, generator=gen) * 100).to(X.device)
    pad = ~mask.unsqueeze(1).expand_as(X)
    X_noisy = torch.where(pad, noise, X)
    a = torch.sigmoid(model(X, mask))
    b = torch.sigmoid(model(X_noisy, mask))
    return {
        "passed": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "n_samples": int(len(X)),
        "n_samples_with_padding": int((~mask).any(dim=1).sum()),
        "atol": atol,
        "rtol": rtol,
        "max_abs_score_difference": float((a - b).abs().max()),
        "mode": "eval",
        "perturbation": "padding replaced by N(0, 100^2) noise",
    }
