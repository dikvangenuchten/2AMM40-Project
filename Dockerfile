FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

# Install tools for visualization
RUN apt update && apt install graphviz ffmpeg libsm6 libxext6 -y --no-install-recommends

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt