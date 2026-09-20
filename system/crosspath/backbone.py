"""冻结的视觉—文本主干，以及在其之上的两个基础模型（端点）。

第四章把每个基础模型 E_i 拆成查询组合器 C_i 和图库编码器 V_i，并且全程冻结
基础模型参数。系统这一层只负责把这两件事做出来：

* 主干：CLIP ViT-B/32，权重下载后全程 ``requires_grad_(False)``。
* 端点：E0 与 E1 共享主干，各自带一组轻量头（一个图库投影 + 一个查询组合器）。
  两组头用不同随机种子（20260718 / 20260722）独立训练，对应论文 FashionGen
  设置里「结构相同、独立训练的两个基础模型」。

两个端点必须住在同一个 512 维空间里，否则 q_0·g_1 这样的交叉路径根本没法算。
共享主干 + 恒等初始化的头保证了这一点：训练只在恒等映射附近移动，两个端点的
坐标系仍然对齐。这正是第四章交叉路径成立的前提。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
EMBED_DIM = 512
SEEDS = (20260718, 20260722)


class Endpoint(nn.Module):
    """一个基础模型 E_i = (查询组合器 C_i, 图库编码器 V_i)。"""

    def __init__(self, dim: int = EMBED_DIM, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.gallery = nn.Linear(dim, dim, bias=False)
        self.compose = nn.Linear(dim, dim, bias=False)
        for layer in (self.gallery, self.compose):
            noise = torch.randn(dim, dim, generator=generator) * 0.01
            layer.weight.data = torch.eye(dim) + noise
        # 图像与文本两路的融合权重，softmax 之后相加，初值给文本略高一点。
        self.mix = nn.Parameter(torch.tensor([0.0, 0.4]))

    def encode_gallery(self, image_features: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.gallery(image_features), dim=-1)

    def encode_query(self, image_features: torch.Tensor,
                     text_features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.mix, dim=0)
        fused = weights[0] * image_features + weights[1] * text_features
        return nn.functional.normalize(self.compose(fused), dim=-1)


class Backbone:
    """CLIP 主干的惰性加载封装，只在第一次真的要编码时才载入权重。"""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED)
            model.eval().requires_grad_(False).to(self.device)
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        return self._model

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 64) -> np.ndarray:
        model = self._load()
        out = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            batch = torch.stack([self._preprocess(im) for im in chunk]).to(self.device)
            feats = model.encode_image(batch).float()
            out.append(nn.functional.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, EMBED_DIM), np.float32)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size: int = 128) -> np.ndarray:
        model = self._load()
        out = []
        for start in range(0, len(texts), batch_size):
            tokens = self._tokenizer(texts[start:start + batch_size]).to(self.device)
            feats = model.encode_text(tokens).float()
            out.append(nn.functional.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, EMBED_DIM), np.float32)


def checkpoint_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
