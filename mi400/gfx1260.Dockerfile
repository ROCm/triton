# This is primarily meant as a docker image for gfx1260 CI.
# This docker image tries to install necessary packages to be hermetic.
# Though in order to make updating key components easier, it expects volume
# binding to the following directories inside docker:
#
# - /code/: Github Action Runner's Triton work directory
# - /llvm/: LLVM pre-built package
# - /ffm/: Directory containing FFM pre-built package
# - /root/.triton/: Triton cache directory
# - /root/.ccache/: ccache directory

# Build docker with public PyTorch:
# docker build . -f /path/to/triton/mi400/gfx1260.Dockerfile -t ci/gfx1260-env

# Build docker with NPI PyTorch:
# docker build . -f /path/to/triton/mi400/gfx1260.Dockerfile -t ci/gfx1260-pytorch-env \
#   --build-arg USE_NPI_TORCH=TRUE

# Build docker with NPI ROCm + roccap:
# docker build . -f /path/to/triton/mi400/gfx1260.Dockerfile -t ci/gfx1260-roccap \
#   --build-arg USE_ROCCAP=TRUE
FROM ubuntu:24.04

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

# Basic development environment
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
  git clang lld ccache \
  python3 python3-dev python3-pip \
  sudo numactl libelf1 libzstd-dev curl wget rsync && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# gfx1260 development environment
# We are in a docker so it's fine to break system packages
RUN pip config set global.break-system-packages true
RUN pip install --no-cache-dir --ignore-installed --upgrade pip PyYAML

# Install pip packages needed for Triton development
RUN pip install --no-cache-dir --ignore-installed --upgrade "setuptools>=40.8.0" wheel
RUN pip install --no-cache-dir --upgrade "cmake>=3.20,<4.0" "ninja>=1.11.1" "pybind11>=2.13.1" nanobind \
  numpy scipy pandas matplotlib einops \
  pytest pytest-xdist pytest-repeat pytest-forked lit \
  expecttest pylama pre-commit clang-format

# Switch to choose either NPI or regular torch distribution
ARG USE_NPI_TORCH=FALSE

# TODO: Use pip once it's ready
COPY whls-gfx1260/ /whls/

RUN set -eux; \
  if [ "${USE_NPI_TORCH}" = "TRUE" ]; then \
    pip install /whls/* && \
    # hip-python not required but needed to support current gfx1260 workarounds
    pip install --no-cache-dir --upgrade hip-python -i https://test.pypi.org/simple/ && \
    pip uninstall -y triton pytorch-triton pytorch-triton-rocm && \
    rm -rf /whls; \
  else \
    pip install --no-cache-dir --upgrade hip-python -i https://test.pypi.org/simple/ && \
    pip install --no-cache-dir torch -i https://download.pytorch.org/whl/nightly/rocm6.4 && \
    pip uninstall -y triton pytorch-triton pytorch-triton-rocm && \
    rm -rf $(pip show torch | grep ^Location: | cut -d' ' -f2-)/torch/lib/libamdhip64.so; \
  fi

# Copy a local FFM Lite package for hermetic environment
# In CI we may want to rebind it when invoking docker for easy upgrade.
COPY ffmlite/ /ffm

# Ccache settings
RUN ccache --max-size=15G

ARG USE_ROCCAP=FALSE
ARG ROCPLAYCAP_VERSION="4.5.1"

RUN set -eux; \
  if [ "${USE_ROCCAP}" = "TRUE" ]; then \
    wget https://atlartifactory.amd.com/artifactory/HW-RocPlayCap-REL/releases/rocplaycap-${ROCPLAYCAP_VERSION}/rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz && \
    tar -xf ./rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz && \
    cd ./rocplaycap-src-${ROCPLAYCAP_VERSION} && \
    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/.local -DCMAKE_PREFIX_PATH=/opt/rocm/ -DHSA_ROOT_DIR:PATH=/opt/rocm/hsa/ && \
    cmake --build build --target install && \
    cd .. && rm -rf ./rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz ./rocplaycap-src-${ROCPLAYCAP_VERSION}; \
  fi

RUN mkdir $HOME/.ssh && echo -e "Host github.com\n\tHostname ssh.github.com\n\tPort 443" >> $HOME/.ssh/config
ENV CCACHE_DIR=/root/.ccache
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /code
ENTRYPOINT /usr/bin/bash
