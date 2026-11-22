"""Construct the official Faceptor backbone from its native config/checkpoint.

The Faceptor source tree is mounted at ``/app/faceptor`` by ``run_docker.sh``.
Set ``FACEPTOR_CONFIG`` to the official Faceptor YAML and pass the downloaded
checkpoint through ``--teacher-checkpoint``. The factory only builds the
backbone, so reconstruction never invokes Faceptor's task heads.
"""

from __future__ import annotations

import os
from argparse import Namespace

import torch


def make_faceptor_backbone(checkpoint_path: str | None = None):
    config_path = os.environ.get("FACEPTOR_CONFIG")
    if not config_path:
        raise RuntimeError("FACEPTOR_CONFIG must point to an official Faceptor YAML file")
    try:
        from core.config import Config
        from core.model.backbone import backbone_entry
    except ImportError as exc:
        raise RuntimeError(
            "Faceptor source is not importable. Mount its faceptor_project directory "
            "at /app/faceptor or set PYTHONPATH accordingly."
        ) from exc

    args = Namespace(
        config=config_path,
        expname=None,
        start_time="20230000_000000",
        now="20230000_000000",
        out_dir=None,
        load_path="",
        load_iter="newest",
        load_ignore=[],
    )
    configuration = Config(args, rank=0, local_rank=0, world_size=1)
    backbone = backbone_entry(configuration.common.backbone)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        prefixes = ("module.backbone_module.", "backbone_module.", "module.")
        cleaned = {}
        for key, value in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            cleaned[key] = value
        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"Faceptor backbone missing {len(missing)} checkpoint keys")
        if unexpected:
            print(f"Faceptor checkpoint has {len(unexpected)} unused keys")
    return backbone
