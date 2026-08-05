# syntax=docker/dockerfile:1

# MotionFlow-MultiView reproducible container
# Base: CUDA 12.1 development image on Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /workspace/motionflow-multiview

# Install PyTorch for CUDA 12.1 first to satisfy the project pin
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
        torch==2.4.0+cu121 \
        torchvision==0.19.0+cu121 \
        torchaudio==2.4.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Install common HPE utilities used by experiments but not listed in requirements.txt
RUN python3 -m pip install opencv-contrib-python-headless matplotlib scipy

# Default entrypoint: drop into a bash shell
CMD ["/bin/bash"]
