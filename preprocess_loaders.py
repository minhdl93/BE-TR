"""
Load NAFNet (stage 1) and DarkIR (stage 2) preprocessing models for efficiency benchmarks.

These restorers are not vendored in this repo. Provide checkpoints and optional
builder callables so benchmark_efficiency_table.py can time the real pipeline.

Example (external repo on PYTHONPATH):

    # preprocess/nafnet_build.py
    def build_nafnet():
        from basicsr.archs.nafnet_arch import NAFNet
        net = NAFNet(img_channel=3, width=32, ...)
        return net

    python benchmark_efficiency_table.py \\
        --nafnet_ckpt path/to/nafnet.pth \\
        --nafnet_builder preprocess.nafnet_build:build_nafnet \\
        --darkir_ckpt path/to/darkir.pth \\
        --darkir_builder preprocess.darkir_build:build_darkir
"""

from __future__ import annotations

import importlib
from typing import Callable, Optional

import torch
import torch.nn as nn


def _import_builder(spec: str) -> Callable[[], nn.Module]:
    """Import ``module.path:callable`` that returns an nn.Module (no args)."""
    if ":" not in spec:
        raise ValueError(f"Builder must be 'module.path:function', got: {spec}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def _load_state(model: nn.Module, ckpt_path: str) -> nn.Module:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "params", "model", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "", 1) if k.startswith("module.") else k
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)
    return model


def load_preprocess_model(
    name: str,
    checkpoint: Optional[str] = None,
    builder_spec: Optional[str] = None,
) -> nn.Module:
    """
    Load NAFNet or DarkIR.

    Args:
        name: ``nafnet`` or ``darkir``
        checkpoint: path to .pth weights (optional for timing architecture only)
        builder_spec: ``module:function`` returning the model
    """
    if builder_spec:
        model = _import_builder(builder_spec)()
    else:
        raise FileNotFoundError(
            f"No builder for {name}. Pass --{name}_builder module:function "
            f"that constructs the network, and --{name}_ckpt for weights."
        )
    if checkpoint:
        _load_state(model, checkpoint)
    return model


class NAFNetStage(nn.Module):
    """Stage-1 preprocessor (expects RGB in [0, 1], shape B×3×H×W)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DarkIRStage(nn.Module):
    """Stage-2 preprocessor (expects RGB in [0, 1])."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NAFNetDarkIRPipeline(nn.Module):
    """Sequential NAFNet ? DarkIR."""

    def __init__(self, nafnet: nn.Module, darkir: nn.Module):
        super().__init__()
        self.nafnet = NAFNetStage(nafnet)
        self.darkir = DarkIRStage(darkir)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.darkir(self.nafnet(x))
