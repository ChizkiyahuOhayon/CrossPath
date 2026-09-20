"""Initialise stage-1 pre-training from the released ALBEF checkpoint.

FashionSAP, the baseline this chapter reproduces, starts its FashionGen domain
pre-training from ``ALBEF.pth`` (``fashion_pretrain.py``, ``--pre_point``).
Starting instead from ImageNet ViT + bert-base would compare a model that never
saw image-text pretraining against published numbers that did, so this module
maps ALBEF's parameter names onto :class:`~ch3_mlaff.model_ch3.Ch3Model`.

ALBEF's ViT is already at 256x256 (``pos_embed`` is ``[1, 257, 768]``), so no
position-embedding interpolation is needed.  The momentum copies and the queues
in the checkpoint are skipped: the momentum branch is re-seeded from the loaded
weights and the queues start empty.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

#: Sub-paths inside one BERT layer, longest first so that "attention.output"
#: is matched before the bare "output".
_LAYER_MAP: Tuple[Tuple[str, str], ...] = (
    ("attention.self.query", "attention.query"),
    ("attention.self.key", "attention.key"),
    ("attention.self.value", "attention.value"),
    ("attention.output.dense", "attention_output.dense"),
    ("attention.output.LayerNorm", "attention_output.LayerNorm"),
    ("crossattention.self.query", "crossattention.query"),
    ("crossattention.self.key", "crossattention.key"),
    ("crossattention.self.value", "crossattention.value"),
    ("crossattention.output.dense", "crossattention_output.dense"),
    ("crossattention.output.LayerNorm", "crossattention_output.LayerNorm"),
    ("intermediate.dense", "intermediate"),
    ("output.dense", "output.dense"),
    ("output.LayerNorm", "output.LayerNorm"),
)

_EMBEDDING_MAP = {
    "word_embeddings.weight": "word_embeddings.weight",
    "position_embeddings.weight": "position_embeddings.weight",
    "token_type_embeddings.weight": "token_type_embeddings.weight",
    "LayerNorm.weight": "emb_ln.weight",
    "LayerNorm.bias": "emb_ln.bias",
}

_MLM_MAP = {
    "transform.dense.weight": "mlm_head.dense.weight",
    "transform.dense.bias": "mlm_head.dense.bias",
    "transform.LayerNorm.weight": "mlm_head.LayerNorm.weight",
    "transform.LayerNorm.bias": "mlm_head.LayerNorm.bias",
    "decoder.weight": "mlm_head.decoder.weight",
    "decoder.bias": "mlm_head.decoder.bias",
}

_TOP_LEVEL = ("vision_proj.weight", "vision_proj.bias",
              "text_proj.weight", "text_proj.bias",
              "itm_head.weight", "itm_head.bias", "temp")


def _map_layer(index: int, suffix: str, text_layers: int) -> str | None:
    """``encoder.layer.<index>.<suffix>`` -> the SplitBert parameter name."""
    block = (f"text_blocks.{index}" if index < text_layers
             else f"fusion_blocks.{index - text_layers}")
    for src, dst in sorted(_LAYER_MAP, key=lambda p: -len(p[0])):
        if suffix.startswith(src + "."):
            return f"text_encoder.{block}.{dst}{suffix[len(src):]}"
    return None


def remap_albef(state: Dict[str, torch.Tensor], text_layers: int = 6) -> Dict[str, torch.Tensor]:
    """Translate an ALBEF ``model`` state dict into Ch3Model / MLM-head names."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith("_m") or ".visual_encoder_m." in f".{key}" or key.startswith(
                ("visual_encoder_m.", "text_encoder_m.", "vision_proj_m.", "text_proj_m.")):
            continue
        if key in ("image_queue", "text_queue", "queue_ptr"):
            continue

        if key.startswith("visual_encoder."):
            out["visual_encoder.vit." + key[len("visual_encoder."):]] = value
        elif key.startswith("text_encoder.bert.embeddings."):
            suffix = key[len("text_encoder.bert.embeddings."):]
            if suffix in _EMBEDDING_MAP:
                out["text_encoder." + _EMBEDDING_MAP[suffix]] = value
        elif key.startswith("text_encoder.bert.encoder.layer."):
            rest = key[len("text_encoder.bert.encoder.layer."):]
            index, suffix = rest.split(".", 1)
            mapped = _map_layer(int(index), suffix, text_layers)
            if mapped:
                out[mapped] = value
        elif key.startswith("text_encoder.cls.predictions."):
            suffix = key[len("text_encoder.cls.predictions."):]
            if suffix in _MLM_MAP:
                out[_MLM_MAP[suffix]] = value
        elif key in _TOP_LEVEL:
            out[key] = value
    return out


def target_names(mapped: Dict[str, torch.Tensor], has_mlm_head: bool) -> Dict[str, torch.Tensor]:
    """Re-key for the module being loaded.

    ``Ch3PretrainModel`` wraps the retrieval model as ``base`` and owns the MLM
    head itself; a plain ``Ch3Model`` has no MLM head, so those tensors drop.
    """
    if not has_mlm_head:
        return {k: v for k, v in mapped.items() if not k.startswith("mlm_head.")}
    return {(k if k.startswith("mlm_head.") else "base." + k): v for k, v in mapped.items()}


_MOMENTUM_PREFIXES = ("visual_encoder_m.", "text_encoder_m.", "vision_proj_m.", "text_proj_m.",
                      "cross_blocks_m.", "gate_m.", "region_proj_m.")
_BUFFERS = ("image_queue", "text_queue", "region_queue", "idx_queue", "queue_ptr")


def _is_reseeded(name: str) -> bool:
    """True for parameters the model fills in itself after loading."""
    bare = name[len("base."):] if name.startswith("base.") else name
    return bare.startswith(_MOMENTUM_PREFIXES) or bare in _BUFFERS


def load_albef(model, path: str, text_layers: int = 6) -> dict:
    """Load ALBEF weights into a ``Ch3Model`` or ``Ch3PretrainModel``.

    Returns a report naming every parameter that stayed randomly initialised,
    so a run manifest records exactly what ALBEF did and did not provide.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    mapped = remap_albef(blob["model"] if "model" in blob else blob, text_layers)
    base = getattr(model, "base", model)

    state = target_names(mapped, has_mlm_head=hasattr(model, "mlm_head"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if base.use_momentum:
        base.copy_params()

    return {
        "checkpoint": path,
        "albef_epoch": blob.get("epoch"),
        "loaded_tensors": len(state),
        "randomly_initialised": sorted(n for n in missing if not _is_reseeded(n)),
        "ignored_from_checkpoint": sorted(unexpected),
    }
