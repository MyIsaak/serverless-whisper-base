# Must use a Cuda version 11+ with cuBLAS 11.x and cuDNN 8.x as per the CTranslate2 docs
# See: https://opennmt.net/CTranslate2/installation.html
# It's highly recommended to use base images from Nvidia as they are optimized for these libraries
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

USER root
WORKDIR /root

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV PATH="$HOME/.poetry/bin:$PATH"
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    rm -rf ~/.cache/pip

# Install cuBLAS 11.x
RUN pip install nvidia-cublas-cu11 && \
    rm -rf ~/.cache/pip

# Install Git and FFmpeg for Whisper
RUN apt-get update && apt-get install -y \
    git ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python packages
ADD pyproject.toml .
ADD poetry.lock .
RUN poetry install

# Banana boilerplate
ADD server.py .

# Model weight files
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

# Custom app code, init() and inference()
ADD app.py .
ADD util.py .

EXPOSE 8000

CMD python3 -u server.py
