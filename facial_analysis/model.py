"""RGB-teacher to event-representation reconstruction model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .faceptor import FaceptorFeatureTeacher


class TinyRGBTeacher(nn.Module):
    """Small deterministic fallback used only for local smoke tests."""

    output_channels = 128

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, self.output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(16, self.output_channels),
            nn.SiLU(),
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class ResNet18Teacher(nn.Module):
    """Frozen torchvision ResNet-18 feature map for controlled smoke runs."""

    output_channels = 512

    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            state = state.get("state_dict", state.get("model", state))
            state = {key.removeprefix("module."): value for key, value in state.items()}
            backbone.load_state_dict(state, strict=False)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


def build_rgb_teacher(
    teacher_type: str,
    *,
    checkpoint: str | None = None,
    faceptor_factory: str | None = None,
) -> nn.Module:
    teacher_type = teacher_type.lower()
    if teacher_type == "faceptor":
        if not faceptor_factory:
            raise ValueError("--faceptor-factory is required when --teacher-type=faceptor")
        return FaceptorFeatureTeacher(faceptor_factory, checkpoint)
    if teacher_type == "resnet18":
        return ResNet18Teacher(checkpoint)
    if teacher_type == "tiny":
        return TinyRGBTeacher()
    raise ValueError("teacher type must be one of: faceptor, resnet18, tiny")


class TemporalFeatureAggregator(nn.Module):
    """Aggregate per-frame teacher maps while retaining spatial structure."""

    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, min(16, channels))
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # [B, T, C, H, W] -> [B, C, T, H, W]
        features = features.transpose(1, 2)
        return self.block(features).mean(dim=2)


class ReconstructionDecoder(nn.Module):
    def __init__(self, channels: int, output_size: int):
        super().__init__()
        hidden = min(256, max(32, channels // 2))
        groups = max(1, min(16, hidden))
        while hidden % groups:
            groups -= 1
        self.output_size = output_size
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.body(features)
        features = F.interpolate(features, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return torch.tanh(self.head(features))


class RGBToEventReconstruction(nn.Module):
    """Frozen RGB teacher + trainable temporal aggregator and decoder."""

    def __init__(self, teacher: nn.Module, teacher_channels: int, output_size: int):
        super().__init__()
        self.teacher = teacher
        self.temporal = TemporalFeatureAggregator(teacher_channels)
        self.decoder = ReconstructionDecoder(teacher_channels, output_size)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, rgb_sequence: torch.Tensor) -> torch.Tensor:
        if rgb_sequence.ndim != 5:
            raise ValueError(f"expected RGB sequence [B,T,3,H,W], got {tuple(rgb_sequence.shape)}")
        frame_features = []
        with torch.no_grad():
            for frame_index in range(rgb_sequence.shape[1]):
                frame_features.append(self.teacher(rgb_sequence[:, frame_index]))
        features = torch.stack(frame_features, dim=1)
        return self.decoder(self.temporal(features))


def infer_teacher_channels(teacher: nn.Module, image_size: int = 128) -> int:
    try:
        device = next(teacher.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.no_grad():
        output = teacher(torch.zeros(1, 3, image_size, image_size, device=device))
    if output.ndim != 4:
        raise ValueError(f"RGB teacher must produce a feature map [B,C,H,W], got {tuple(output.shape)}")
    return int(output.shape[1])
