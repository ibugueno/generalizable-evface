# The runtime follows ebird-diffusion/event_reconstruction_baselines: one
# visible GPU, CUDA 12.1, and PyTorch 2.1.2.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Keep NumPy below 2.x for compatibility with the PyTorch 2.1 image and use
# headless OpenCV because this project only reads/writes images on the server.
RUN pip install --no-cache-dir \
        "numpy==1.26.4" \
        "opencv-python-headless==4.8.1.78" \
        "h5py==3.10.0" \
        "Pillow==10.2.0" \
        "PyYAML==6.0.1" \
        "tqdm==4.66.1" \
        "easydict==1.13" \
        "tensorboard==2.15.1" \
        "timm==0.9.12" \
        "huggingface_hub==0.20.3"

WORKDIR /app/facial-analysis
COPY . /app/facial-analysis

RUN mkdir -p /app/persistent /app/data

ENV PYTHONPATH=/app/facial-analysis \
    DATA_ROOT=/app/data \
    PERSISTENT_ROOT=/app/persistent \
    HF_HOME=/app/persistent/hf_cache

CMD ["bash"]
