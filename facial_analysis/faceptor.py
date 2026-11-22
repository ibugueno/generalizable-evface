"""Adapter boundary for the official Faceptor implementation.

Faceptor's public repository is a training/testing framework rather than a
small pip package. Its model entry exposes a shared backbone, and the official
backbone returns a list of intermediate feature maps. This module keeps the
reconstruction project independent from Faceptor's experiment configuration:
the server provides a tiny factory that constructs the already-trained
Faceptor model or backbone, and this adapter extracts a spatial feature tensor.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


def import_object(spec: str):
    if ":" not in spec:
        raise ValueError(f"factory must use module:object syntax, got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _call_factory(factory, checkpoint: str | None):
    if checkpoint:
        try:
            return factory(checkpoint)
        except TypeError:
            return factory(checkpoint_path=checkpoint)
    return factory()


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("backbone_output", "features", "feature", "embedding", "x"):
            if key in output:
                try:
                    return _extract_tensor(output[key])
                except TypeError:
                    pass
        for value in output.values():
            try:
                return _extract_tensor(value)
            except TypeError:
                continue
    if isinstance(output, (list, tuple)):
        # Faceptor's backbone returns feature maps from multiple layers. The
        # final spatial map is the least surprising default for reconstruction.
        for value in reversed(output):
            try:
                return _extract_tensor(value)
            except TypeError:
                continue
    raise TypeError(f"could not extract a tensor from Faceptor output type {type(output).__name__}")


def _to_spatial(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return features
    if features.ndim == 3:
        # Accept [B, tokens, channels] and remove a CLS token when present.
        batch, tokens, channels = features.shape
        side = int(tokens**0.5)
        if side * side != tokens:
            if int((tokens - 1) ** 0.5) ** 2 == tokens - 1:
                features = features[:, 1:]
                tokens -= 1
                side = int(tokens**0.5)
            else:
                raise ValueError(f"Faceptor returned non-spatial token shape {tuple(features.shape)}")
        return features.transpose(1, 2).reshape(batch, channels, side, side)
    if features.ndim == 2:
        return features[:, :, None, None]
    raise ValueError(f"Faceptor feature tensor must be 2D, 3D or 4D, got {tuple(features.shape)}")


class FaceptorFeatureTeacher(nn.Module):
    """Frozen feature extractor around a user-supplied Faceptor factory."""

    def __init__(self, factory_spec: str, checkpoint: str | None = None):
        super().__init__()
        factory = import_object(factory_spec)
        model = _call_factory(factory, checkpoint)
        if not isinstance(model, nn.Module):
            raise TypeError("Faceptor factory must return torch.nn.Module")
        self.model = model
        self.checkpoint = str(Path(checkpoint).resolve()) if checkpoint else None
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        # A teacher remains in evaluation mode even when the reconstruction
        # model is switched to train mode.
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "backbone_module"):
            output = self.model.backbone_module(images)
        elif hasattr(self.model, "forward_features"):
            output = self.model.forward_features(images)
        else:
            output = self.model(images)
        return _to_spatial(_extract_tensor(output))
