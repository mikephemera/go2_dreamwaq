FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="Jongkuk Lim <limjk@jmarple.ai>"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV TERM xterm-256color
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DEFAULT_TIMEOUT=1000

ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o user && useradd -m -u $UID -g $GID -o -s /bin/bash user

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 替换 apt 源为清华 tuna 镜像
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y sudo dialog apt-utils tzdata
RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && echo "user:user" | chpasswd && adduser user sudo

WORKDIR /home/user
USER user

RUN sudo apt-get update && sudo apt-get install -y libgl1-mesa-dev && sudo apt-get -y install jq

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN sudo apt-get update && sudo apt-get -y install wget curl git
COPY --chown=user:user docker/run_fixed.sh /home/user/run_fixed.sh
RUN chmod +x /home/user/run_fixed.sh && /home/user/run_fixed.sh

RUN sudo apt-get update && sudo apt-get install -y zsh && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
RUN echo "\n# Custom settings" >> /home/user/.zshrc && \
    echo "export PATH=/home/user/.local/bin:$PATH" >> /home/user/.zshrc && \
    echo "export LC_ALL=C.UTF-8 && export LANG=C.UTF-8" >> /home/user/.zshrc && \
    sed '11 c\ZSH_THEME=powerlevel10k/powerlevel10k' ~/.zshrc  > tmp.txt && mv tmp.txt ~/.zshrc && \
    echo 'POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true' >> ~/.zshrc
RUN sudo chsh -s $(which zsh) $(whoami)
# Place your environment here

RUN sudo apt-get install -y python3-pip && pip3 install --upgrade pip
# change python to be called as python, not python3
RUN sudo apt-get install -y python-is-python3

# Pre-download wheels to cache
RUN mkdir -p /tmp/wheels && \
    python3 -m pip download --default-timeout=2000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cuda-cupti-cu12

# Download NVIDIA CUDA libraries (very large, especially cudnn)
RUN cd /tmp/wheels && \
    python3 -m pip download --default-timeout=2000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links /tmp/wheels \
        nvidia-cudnn-cu12 \
        nvidia-cublas-cu12

# Download NVIDIA CUDA libraries
RUN cd /tmp/wheels && \
    python3 -m pip download --default-timeout=2000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links /tmp/wheels \
        nvidia-cufft-cu12 \
        nvidia-curand-cu12 \
        nvidia-cusolver-cu12 \
        nvidia-cusparse-cu12 \
        nvidia-nccl-cu12 \
        nvidia-nvtx-cu12 \
        triton \
        nvidia-nvjitlink-cu12

RUN cd /tmp/wheels && \
    python3 -m pip download --default-timeout=2000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links /tmp/wheels \
        tensorboard \
        wandb \
        opencv-python \
        imageio \
        ninja \
        pillow \
        scipy
        
RUN cd /tmp/wheels && \
    python3 -m pip download --default-timeout=2000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links /tmp/wheels \
        nvidia-cudnn-cu12==9.1.0.70 \
        nvidia-cublas-cu12==12.1.3.1

RUN cd /tmp/wheels && \
    python3 -m pip download --default-timeout=2000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links /tmp/wheels \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nvtx-cu12==12.1.105 \
        nvidia-nccl-cu12==2.20.5 \
        triton==3.0.0

RUN cd /tmp/wheels && \
    python3 -m pip download --default-timeout=2000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links /tmp/wheels \
        torch

RUN cd /tmp/wheels && \
    python3 -m pip download --default-timeout=2000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links /tmp/wheels \
        torchvision \
        filelock \
        sympy \
        networkx \
        jinja2 \
        fsspec \
        pyyaml \
        numpy \
        protobuf
        
RUN python3 -m pip install --default-timeout=1000 --retries=10 \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
    --find-links /tmp/wheels tensorboard wandb opencv-python

# add gcc9
RUN sudo apt-get -y install mesa-common-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev vulkan-utils mesa-vulkan-drivers libegl1 gcc-8 g++-8 gcc-9 g++-9 pigz
RUN sudo rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json
# set gcc-9 as default
RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8 && \
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 && \
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9 && \
    sudo update-alternatives --set gcc /usr/bin/gcc-9 && \
    sudo update-alternatives --set g++ /usr/bin/g++-9

COPY asset/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY asset/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY asset/IsaacGym_Preview_4_Package.tar.gz /home/user/
WORKDIR /home/user
RUN tar -xzf ~/IsaacGym_Preview_4_Package.tar.gz && rm ~/IsaacGym_Preview_4_Package.tar.gz
RUN cd isaacgym/python && \
    python3 -m pip install --default-timeout=1000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --find-links /tmp/wheels -e .

COPY --chown=user:user rsl_rl /home/user/rsl_rl
WORKDIR /home/user
RUN cd rsl_rl && \
    python3 -m pip install --default-timeout=1000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --find-links /tmp/wheels -e .

COPY --chown=user:user legged_gym /home/user/legged_gym
WORKDIR /home/user
RUN cd legged_gym && \
    python3 -m pip install --default-timeout=1000 --retries=10 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
        --find-links /tmp/wheels -e .

RUN sudo apt-get -y install python3-actionlib
COPY requirements.txt /home/user/
WORKDIR /home/user
# RUN python3 -m pip install -r requirements.txt && rm requirements.txt

# Clean up downloaded wheels
RUN rm -rf /tmp/wheels

