FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Combine all apt-get install commands into one RUN command
RUN apt-get update && apt-get install -qq -y \
    software-properties-common \
    wget \
    gcc-11 \
    g++-11 \
    git \
    build-essential \
    sudo \
    libeigen3-dev \
    libboost-all-dev \
    libjsoncpp-dev \
    libopengl-dev \
    mesa-utils \
    libglfw3-dev \
    libglm-dev \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    curl \
    zip \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libswresample-dev \
    libssl-dev && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && apt-get install -qq -y gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    rm -rf /var/lib/apt/lists/*

# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-Linux-x86_64.sh -O /cmake-3.22.1.sh && \
    chmod +x /cmake-3.22.1.sh && \
    /cmake-3.22.1.sh --skip-license --prefix=/usr/local && \
    rm /cmake-3.22.1.sh

# opencv
RUN mkdir /opencv && cd /opencv && \
    wget https://github.com/opencv/opencv/archive/refs/tags/4.8.0.zip -O opencv-4.8.0.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.0.zip -O opencv_contrib-4.8.0.zip && \
    unzip -q opencv-4.8.0.zip && \
    unzip -q opencv_contrib-4.8.0.zip && \
    rm opencv-4.8.0.zip && rm opencv_contrib-4.8.0.zip && \
    mkdir /opencv/opencv-4.8.0/build && cd /opencv/opencv-4.8.0/build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE \
          -DWITH_CUDA=ON \
          -DWITH_CUDNN=ON \
          -DOPENCV_DNN_CUDA=ON \
          -DWITH_NVCUVID=ON \
          -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
          -DOPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib-4.8.0/modules \
          -DBUILD_TIFF=ON \
          -DBUILD_ZLIB=ON \
          -DBUILD_JASPER=ON \
          -DBUILD_JPEG=ON \
          -DWITH_FFMPEG=ON \
          .. && \
    make -s -j$(nproc) && \
    make -s install && \
    ldconfig && \
    cd / && rm -rf /opencv

# libtorch
RUN cd / && \
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip -O libtorch-cu118.zip && \
    unzip -q libtorch-cu118.zip && rm libtorch-cu118.zip

ENV Torch_DIR /libtorch/share/cmake/Torch

# repo
RUN git clone https://github.com/gloryhry/Photo-SLAM.git /Photo-SLAM

WORKDIR /Photo-SLAM
