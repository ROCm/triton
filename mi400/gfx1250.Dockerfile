# This is primarily meant as a docker image for gfx1250 CI.
# But it tries to be cooperative with mirror accounts inside the docker to
# avoid touching all files with root ownership; so you can build on top it
# for personal development environment.
#
# This docker image tries to install necessary packages to be hermetic.
# Though in order to make updating key components easier, it expects volume
# binding to the following directories inside docker:
#
# - /code/: Github Action Runner's Triton work directory
# - /llvm/: LLVM pre-built package
# - /ffm/: Directory containing FFM pre-built package
# - /home/mirror/.triton/: Triton cache directory
# - /home/mirror/.ccache/: ccache directory
#
# Build with:
# sudo docker build /path/to/ffm -f /path/to/this/dockerfile -t ci/gfx1250-env \
#   --build-arg DOCKER_USERID=$(id -u) --build-arg DOCKER_GROUPID=$(id -g) --build-arg DOCKER_RENDERID=$(getent group render | cut -d: -f3)
# For NPI PyTorch add the following build args:
#   --build-arg USE_NPI_TORCH=TRUE --build-arg ROCM_BUILD_NUMBER=710
FROM ubuntu:24.04

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

# Basic development environment
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
  git clang lld ccache \
  python3 python3-dev python3-pip \
  sudo numactl libelf1 libzstd-dev curl wget rsync
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# gfx1250 development environment
# We are in a docker so it's fine to break system packages
RUN pip config set global.break-system-packages true
RUN pip install --ignore-installed --upgrade pip PyYAML

# Install pip packages needed for Triton development
RUN pip install --ignore-installed --upgrade "setuptools>=40.8.0" wheel
RUN pip install --upgrade "cmake>=3.20,<4.0" "ninja>=1.11.1" "pybind11>=2.13.1" nanobind \
  numpy scipy pandas matplotlib einops \
  pytest pytest-xdist pytest-repeat lit expecttest \
  pylama pre-commit clang-format

# Switch to choose either NPI or regular torch distribution
ARG USE_NPI_TORCH=FALSE

# Pick ROCm build number and it must match PyTorch NPI build
# To get the build numbers, go to: http://rocm-ci.amd.com/view/mi450/job/compute-rocm-npi-mi450/
# and pick the desired build
ARG ROCM_BUILD_NUMBER=710
# We try to automatically get AMDGPU_BUILD_NUMBER in the following with curl.
# If running into errors, you can manually specify AMDGPU_BUILD_NUMBER by
# getting it with your browser.

RUN \
  if [ "${USE_NPI_TORCH}" = "TRUE" ]; then \
    AMDGPU_BUILD_NUMBER=$(curl -s http://rocm-ci.amd.com/view/mi450/job/compute-rocm-npi-mi450/${ROCM_BUILD_NUMBER}/ | grep -oP 'Mesa UMD Build Number:\K\d+') && \
    echo "Using ROCm Build Number: ${ROCM_BUILD_NUMBER}" && \
    echo "Using AMDGPU Build Number: ${AMDGPU_BUILD_NUMBER}" && \
    wget https://artifactory-cdn.amd.com/artifactory/list/amdgpu-deb/amdgpu-install-internal_7.2-24.04-1_all.deb && \
    sudo apt-get install ./amdgpu-install-internal_7.2-24.04-1_all.deb && \
    amdgpu-repo --amdgpu-build=${AMDGPU_BUILD_NUMBER} --rocm-build=compute-rocm-npi-mi450/${ROCM_BUILD_NUMBER} && \
    amdgpu-install -y --usecase=rocm; \
  fi

RUN \
  if [ "${USE_NPI_TORCH}" = "TRUE" ]; then \
    URL_BASE=https://compute-artifactory.amd.com/artifactory/compute-pytorch-rocm/compute-rocm-npi-mi450/${ROCM_BUILD_NUMBER}/mi450 && \
    echo "Fetching wheels from ${URL_BASE}" && \
    TORCH_WHL=$(curl -s ${URL_BASE}/ | grep -oP 'torch-[^"]*'\\.whl | head -n1) && \
    TORCHVISION_WHL=$(curl -s ${URL_BASE}/ | grep -oP 'torchvision-[^"]*'\\.whl | head -n1) && \
    TRITON_WHL=$(curl -s ${URL_BASE}/ | grep -oP 'triton-[^"]*'\\.whl | head -n1) && \
    pip install \
      ${URL_BASE}/${TORCH_WHL} \
      ${URL_BASE}/${TORCHVISION_WHL} \
      ${URL_BASE}/${TRITON_WHL} && \
    # hip-python not required but needed to support current gfx1250 workarounds
    pip install --upgrade hip-python -i https://test.pypi.org/simple/ && \
    pip uninstall -y triton pytorch-triton pytorch-triton-rocm; \
  else \
    pip install --upgrade hip-python -i https://test.pypi.org/simple/ && \
    pip install torch -i https://download.pytorch.org/whl/nightly/rocm6.4 && \
    pip uninstall -y triton pytorch-triton pytorch-triton-rocm && \
    rm -rf $(pip show torch | grep ^Location: | cut -d' ' -f2-)/torch/lib/libamdhip64.so; \
  fi

# Copy a local FFM Lite package for hermetic environment
# In CI we may want to rebind it when invoking docker for easy upgrade.
COPY rocm-ffmlite-mi450-oai-7ac1dbc-rel-20251031/ /ffm

# Create non-root user account to mirror host user account
ARG DOCKER_USERID=0
ARG DOCKER_GROUPID=0
ARG DOCKER_USERNAME=mirror
ARG DOCKER_GROUPNAME=mirror

RUN if [ ${DOCKER_USERID} -ne 0 ] && [ ${DOCKER_GROUPID} -ne 0 ]; then \
    groupadd --gid ${DOCKER_GROUPID} ${DOCKER_GROUPNAME} && \
    useradd --no-log-init --create-home \
      --uid ${DOCKER_USERID} --gid ${DOCKER_GROUPID} \
      --shell /usr/bin/zsh ${DOCKER_USERNAME}; \
fi

# Create render group needed for AMD GPU access
ARG DOCKER_RENDERID=0

RUN if [ ${DOCKER_USERID} -ne 0 ] && [ ${DOCKER_RENDERID} -ne 0 ]; then \
    groupadd --force --gid ${DOCKER_RENDERID} render && \
    usermod -aG render ${DOCKER_USERNAME} && \
    usermod -aG video ${DOCKER_USERNAME}; \
fi

# Set up sudo access
RUN if [ ${DOCKER_USERID} -ne 0 ] && [ ${DOCKER_RENDERID} -ne 0 ]; then \
    echo "${DOCKER_USERNAME}:${DOCKER_USERNAME}" | chpasswd && \
    usermod -aG sudo ${DOCKER_USERNAME} && \
    echo "username ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/username; \
fi

# Ccache settings
RUN ccache --max-size=15G

# Create mapping directories and chown before switching user
RUN mkdir -p /code && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /code && \
  mkdir -p /llvm && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /llvm && \
  mkdir -p /ffm && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /ffm && \
  mkdir -p /home/${DOCKER_USERNAME}/.triton && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /home/${DOCKER_USERNAME}/.triton && \
  mkdir -p /home/${DOCKER_USERNAME}/.ccache && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /home/${DOCKER_USERNAME}/.ccache

# Now switch to the mirror user and setup configurations
USER ${DOCKER_USERNAME}
WORKDIR /home/${DOCKER_USERNAME}

ARG ROCPLAYCAP_VERSION="4.5.1"

RUN \
  if [ "${USE_NPI_TORCH}" = "TRUE" ]; then \
    wget https://atlartifactory.amd.com/artifactory/HW-RocPlayCap-REL/releases/rocplaycap-${ROCPLAYCAP_VERSION}/rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz && \
    tar -xf ./rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz && \
    cd ./rocplaycap-src-${ROCPLAYCAP_VERSION} && \
    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/.local -DCMAKE_PREFIX_PATH=/opt/rocm/ -DHSA_ROOT_DIR:PATH=/opt/rocm/hsa/ && \
    cmake --build build --target install && \
    cd .. && rm -rf ./rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz ./rocplaycap-src-${ROCPLAYCAP_VERSION}; \
  fi

RUN pip config set global.break-system-packages true
RUN mkdir $HOME/.ssh && echo -e "Host github.com\n\tHostname ssh.github.com\n\tPort 443" >> $HOME/.ssh/config
ENV CCACHE_DIR=/home/${DOCKER_USERNAME}/.ccache
ENV PATH="/home/${DOCKER_USERNAME}/.local/bin:${PATH}"

WORKDIR /code
ENTRYPOINT /usr/bin/bash
