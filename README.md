# Generalizable Event-Based Facial Perception — first implementation

This directory contains only the first research task: RGB-to-event
representation reconstruction. It is separate from eBIRD's reconstruction
implementation, but follows its practical server conventions: CUDA 12.1,
PyTorch 2.1.2, one visible GPU, read-only source/data mounts, and writable
persistent outputs.

The learning path is:

```text
FacEvent RGB frame sequence
          |
          v
Frozen Faceptor RGB backbone
          |
          v
Trainable temporal feature aggregation
          |
          v
Trainable grayscale reconstruction decoder
          |
          v
ACC / SAE / TBR / LADS target generated from FacEvent events
```

Only the temporal aggregator and decoder are optimized. Faceptor is used as a
frozen RGB teacher. The `tiny` and `resnet18` teachers exist only for local
smoke tests; they are not the research configuration.

## What is implemented

- `facial_analysis.faceptor_factory`: builds the official Faceptor backbone
  from its native YAML and checkpoint when the Faceptor source is mounted.
- `FacEventReconstructionDataset`: consumes synchronized RGB videos and event
  streams from a lightweight CSV manifest.
- `scripts/build_facevent_manifest.py`: creates clip-window rows from the
  FacEvent layout without loading the 472 GB event corpus into memory.
- `scripts/generate_representations.py`: generates common single-channel
  `float32` representations for event files.
- `scripts/train_reconstruction.py`: single-GPU reconstruction training with
  checkpoints and train-loss logging.
- `scripts/predict_reconstruction.py`: writes reconstructed `.npy` files and
  grayscale previews.

This first version intentionally does not automatically download, validate,
split, or copy the complete FacEvent database. The server controls which
subset is downloaded and where it is stored.

## Download a bounded FacEvent subset

The image includes `huggingface_hub`, but the downloader is selective by
design. It lists the repository files, selects complete clip groups, resumes
through the Hugging Face cache, and preserves the original layout. It does not
download the whole dataset unless you explicitly select all clips.

First inspect a small selection:

```bash
python scripts/download_facevent.py \
  --output-dir /app/persistent/datasets/FacEvent \
  --split train --batch 0 --max-clips 2 --dry-run
```

Then download the RGB video and event HDF5 files:

```bash
python scripts/download_facevent.py \
  --output-dir /app/persistent/datasets/FacEvent \
  --split train --batch 0 --max-clips 2 \
  --max-workers 2
```

Add `--include-landmarks` when the landmark files are needed. Use repeated
`--clip-id` arguments instead of `--max-clips` for an explicit clip list. A
Hugging Face token can be supplied with `--token` or `HF_TOKEN`; no token is
hard-coded.

## Container setup

Build from this directory:

```bash
./build_docker.sh
```

Start the container with GPU 0 only:

```bash
GPU_ID=0 \
FACIAL_ANALYSIS_PERSISTENT_DIR=/server/persistent/facial-analysis \
FACIAL_ANALYSIS_DATA_DIR=/server/datasets/FacEvent \
FACIAL_ANALYSIS_FACEPTOR_DIR=/server/models/Faceptor \
FACEPTOR_CONFIG=/app/faceptor/faceptor_project/expreiments/faceptor/stage_1/configs/your_config.yaml \
./run_docker.sh
```

`run_docker.sh` maps the host GPU selected by `GPU_ID` to container device 0
and sets `CUDA_VISIBLE_DEVICES=0`. Keep `GPU_ID=0` for the requested protocol.
The code and data are read-only; `/app/persistent` is writable for manifests,
checkpoints, logs, and generated outputs.

The Faceptor mount must contain the official repository's
`faceptor_project/core/...` tree. The downloaded Faceptor checkpoint is not
committed and should remain in the server's model/persistent storage.

## FacEvent manifest

FacEvent stores each clip as a 640x640 MP4 beside an HDF5 event file with
datasets `events/{t,x,y,p}` and timestamps in microseconds. Build a lightweight
window manifest inside the container:

```bash
python scripts/build_facevent_manifest.py \
  --dataset-root /app/data/naturalistic/train \
  --output /app/persistent/manifests/facevent_train.csv \
  --sequence-length 8 \
  --stride 8
```

For an initial smoke run, use `--max-clips 1 --max-windows 16`. The script only
indexes video metadata and filenames; it does not read all event arrays.

The generated CSV has the following important fields:

```text
sample_id,rgb_path,event_path,frame_start,frame_count,
event_start,event_end,event_height,event_width,timestamp_unit
```

The dataset converts FacEvent microseconds to seconds internally before
building the target representation. Event windows correspond to the selected
RGB frame interval.

## FPS-aligned preprocessing and pairing

For every video frame, `scripts/preprocess_facevent.py` uses the matching
window `[i/fps, (i+1)/fps)` from the event HDF5 file. It writes:

- `pairs_<representation>.csv`: one RGB/grayscale/event row per frame;
- `reconstruction_manifest_<representation>.csv`: the same targets with a
  temporal RGB context window controlled by `--sequence-length`;
- `preprocessing.json`: the exact FPS/window and export settings.

The default mode writes manifests only: RGB remains in the source MP4, and
event targets are generated on demand during training. This avoids duplicating
the 640×640 RGB corpus or materializing millions of event arrays.

```bash
python scripts/preprocess_facevent.py \
  --dataset-root /app/persistent/datasets/FacEvent/naturalistic/train \
  --output-root /app/persistent/preprocessed/facevent_train \
  --representations acc sae tbr lads \
  --sequence-length 8
```

For a small visual inspection export, add:

```bash
  --max-clips 1 --max-frames 128 \
  --write-event-representations --write-event-png \
  --export-rgb-frames --export-gray-frames
```

When event arrays are materialized, train from the representation-specific
manifest, for example:

```bash
python scripts/train_reconstruction.py \
  --manifest /app/persistent/preprocessed/facevent_train/reconstruction_manifest_acc.csv \
  --output-dir /app/persistent/reconstruction/acc \
  --config configs/reconstruction.yaml
```

## Generate representations

Representations can be generated ahead of training, or generated on demand by
the reconstruction dataset. For bounded NPZ/NPY/text files, the command below
works directly. For FacEvent HDF5 files, always provide a time window; the
script refuses an unbounded HDF5 request so it cannot accidentally materialize
an entire multi-terabyte event stream.

```bash
python scripts/generate_representations.py \
  --input-root /app/data/events \
  --output-root /app/persistent/representations \
  --height 640 --width 640 \
  --timestamp-unit us \
  --time-start 0 --time-end 33333 \
  --representations acc sae tbr lads \
  --tbr-bins 8 --save-png --recursive
```

The supported input formats are `.npz`, `.npy`, `.csv`, `.txt`, `.h5`, and
`.hdf5`. Matrix files default to `[t, x, y, polarity]`; use `--event-layout`
for a different four-column order. Outputs are `H x W x 1` float32 `.npy`
files in `[-1, 1]`, plus optional visualization PNGs.

Representation conventions in this baseline are:

- ACC: signed polarity accumulation with robust normalization.
- SAE: signed recency of the most recent event at each pixel.
- TBR: temporal activity bits compacted into one scalar channel.
- LADS: event-rate adaptive decay surface. This is the event-rate baseline and
  is isolated in `locally_adaptive_decay_surface` so other LADS strategies can
  be added later.

## Train with Faceptor

The native Faceptor config/checkpoint loading is exposed through the default
factory:

```bash
python scripts/train_reconstruction.py \
  --manifest /app/persistent/manifests/facevent_train.csv \
  --output-dir /app/persistent/reconstruction/acc \
  --representation acc \
  --teacher-type faceptor \
  --teacher-factory facial_analysis.faceptor_factory:make_faceptor_backbone \
  --teacher-checkpoint /app/persistent/models/faceptor/checkpoint_rank0_iter_50000.pth.tar \
  --image-size 224 \
  --sequence-length 8 \
  --batch-size 1 \
  --num-workers 4 \
  --epochs 10 \
  --device cuda:0
```

`FACEPTOR_CONFIG` must be set in the container environment to the native
Faceptor YAML. The factory constructs only `common.backbone`, loads the
checkpoint keys belonging to `backbone_module`, and leaves the Faceptor task
heads out of reconstruction.

The trainer has no database-validation stage and no local train/validation
split requirement. It consumes the manifest supplied by the server and writes
`last.pt`, `best.pt`, `config.json`, and `train_metrics.csv` to the persistent
output directory. `best.pt` is selected by training loss in this first version;
it is not a held-out validation result.

For a code-only smoke test without Faceptor weights:

```bash
python scripts/train_reconstruction.py \
  --manifest /path/to/small_manifest.csv \
  --output-dir /app/persistent/reconstruction/smoke \
  --teacher-type tiny --device cpu --epochs 1 --num-workers 0
```

## Reconstruct outputs

```bash
python scripts/predict_reconstruction.py \
  --checkpoint /app/persistent/reconstruction/acc/best.pt \
  --manifest /app/persistent/manifests/facevent_train.csv \
  --output-dir /app/persistent/reconstruction/acc/predictions \
  --device cuda:0
```

This produces predicted `H x W x 1` arrays and grayscale preview images. No
RGB or Faceptor input is required after a standalone downstream student is
implemented; that later task is intentionally not included here.
