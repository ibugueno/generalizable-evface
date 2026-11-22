#!/usr/bin/env python3
"""Selectively and resumably download FacEvent files from Hugging Face.

The full FacEvent repository is hundreds of GB. This script lists repository
metadata, selects complete clip file groups, and downloads only the requested
files into the persistent directory while preserving the dataset layout.

Examples:

  # Inspect the first two clips without downloading anything.
  python scripts/download_facevent.py --output-dir /app/persistent/FacEvent \
      --batch 0 --max-clips 2 --dry-run

  # Download RGB videos, events and landmarks for two clips.
  python scripts/download_facevent.py --output-dir /app/persistent/FacEvent \
      --batch 0 --max-clips 2 --include-landmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ID = "GabrieleMagrini/FacEvent"
REPO_TYPE = "dataset"


def _clip_key(path: str) -> str | None:
    name = Path(path).name
    suffixes = ("_face640.mp4", "_face640_events.h5", "_face640_landmarks.npz")
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _select_files(files: list[str], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    prefix = f"naturalistic/{args.split}/"
    if args.batch is not None:
        prefix += f"{args.batch:04d}/"
    candidates = [path for path in files if path.startswith(prefix)]
    clip_names = sorted({key for path in candidates if (key := _clip_key(path)) is not None})
    if args.clip_id:
        requested = set(args.clip_id)
        clip_names = [name for name in clip_names if name in requested or Path(name).name in requested]
    if args.max_clips >= 0:
        clip_names = clip_names[: args.max_clips]

    allowed_suffixes = {"_face640.mp4", "_face640_events.h5"}
    if args.include_landmarks:
        allowed_suffixes.add("_face640_landmarks.npz")
    selected = []
    missing = []
    for clip in clip_names:
        clip_files = [path for path in candidates if _clip_key(path) == clip]
        for suffix in sorted(allowed_suffixes):
            matches = [path for path in clip_files if path.endswith(suffix)]
            if matches:
                selected.extend(matches)
            elif suffix in {"_face640.mp4", "_face640_events.h5"}:
                missing.append(f"{clip}{suffix}")
    return sorted(selected), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch", type=int, help="FacEvent batch number, for example 0")
    parser.add_argument("--clip-id", action="append", help="exact clip id; repeat for multiple clips")
    parser.add_argument("--max-clips", type=int, default=-1)
    parser.add_argument("--include-landmarks", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--token", help="Hugging Face token; otherwise HF_TOKEN is used if available")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub in the container before downloading FacEvent") from exc

    api = HfApi(token=args.token)
    print(f"Listing files for {args.repo_id}...")
    files = list(api.list_repo_files(repo_id=args.repo_id, repo_type=REPO_TYPE, token=args.token))
    selected, missing = _select_files(files, args)
    if missing:
        raise SystemExit("Required clip files are missing:\n" + "\n".join(missing[:20]))
    if not selected:
        raise SystemExit("No matching FacEvent clips/files were selected")

    print(f"Selected {len(selected)} files from {len({ _clip_key(path) for path in selected })} clips:")
    for path in selected:
        print(f"  {path}")
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=args.repo_id,
        repo_type=REPO_TYPE,
        token=args.token,
        local_dir=str(args.output_dir),
        allow_patterns=selected,
        max_workers=max(1, args.max_workers),
        resume_download=True,
    )
    metadata = {
        "repo_id": args.repo_id,
        "repo_type": REPO_TYPE,
        "selected_files": selected,
        "local_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "download_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Downloaded selected files under {downloaded}")


if __name__ == "__main__":
    main()
