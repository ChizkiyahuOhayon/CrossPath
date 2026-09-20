"""The model configurations behind tables 3.5 / 3.6 and figure 3.6."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from .model_ch3 import Ch3Config

#: Table 3.5 ablation ladder.  ``m1`` is the ``Baseline+`` row of table 3.2 and
#: ``m5`` (aliased ``full``) is the chapter's complete model.
ABLATION_CONFIGS: Dict[str, Ch3Config] = {
    "m1": Ch3Config(vis_layers=(12,), use_cross_attn=False, fusion="none", use_rea=False),
    "m2": Ch3Config(vis_layers=(12,), use_cross_attn=True, fusion="none", use_rea=False),
    "m3": Ch3Config(vis_layers=(4, 8, 12), use_cross_attn=True, fusion="direct", use_rea=False),
    "m4": Ch3Config(vis_layers=(4, 8, 12), use_cross_attn=True, fusion="gated", use_rea=False),
    "m5": Ch3Config(vis_layers=(4, 8, 12), use_cross_attn=True, fusion="gated", use_rea=True),
}

#: Table 3.6 visual-layer combinations.  ``L4_L8_L12`` is the default, i.e. m5.
LAYER_CONFIGS: Dict[str, Ch3Config] = {
    "L12": replace(ABLATION_CONFIGS["m5"], vis_layers=(12,), fusion="none"),
    "L8": replace(ABLATION_CONFIGS["m5"], vis_layers=(8,), fusion="none"),
    "L4_L8": replace(ABLATION_CONFIGS["m5"], vis_layers=(4, 8)),
    "L8_L12": replace(ABLATION_CONFIGS["m5"], vis_layers=(8, 12)),
}

#: Figure 3.6 sweep over the region-enhanced loss weight lambda; 0.5 is m5.
LAMBDA_CONFIGS: Dict[str, Ch3Config] = {
    f"lam{w}".replace(".", "_"): replace(ABLATION_CONFIGS["m5"], rea_weight=w, use_rea=w > 0)
    for w in (0.0, 0.1, 0.3, 1.0)
}

CONFIGS: Dict[str, Ch3Config] = {**ABLATION_CONFIGS, **LAYER_CONFIGS, **LAMBDA_CONFIGS}
CONFIGS["baseline"] = CONFIGS["m1"]
CONFIGS["full"] = CONFIGS["L4_L8_L12"] = CONFIGS["lam0_5"] = CONFIGS["m5"]
