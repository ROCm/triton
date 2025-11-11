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
FROM ubuntu:24.04

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

# Basic development environment
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
  git clang lld ccache \
  python3 python3-dev python3-pip \
  sudo numactl libelf1 libzstd-dev
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
RUN pip install --upgrade hip-python -i https://test.pypi.org/simple/
RUN pip install torch -i https://download.pytorch.org/whl/rocm6.4 && \
  pip uninstall -y triton pytorch-triton pytorch-triton-rocm
# We need to remove the PyTorch bundled libamdhip64.so to avoid interfering!
RUN rm -rf $(pip show torch | grep ^Location: | cut -d' ' -f2-)/torch/lib/libamdhip64.so

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
    groupadd --gid ${DOCKER_RENDERID} render && \
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

RUN pip config set global.break-system-packages true
RUN mkdir $HOME/.ssh && echo -e "Host github.com\n\tHostname ssh.github.com\n\tPort 443" >> $HOME/.ssh/config
ENV CCACHE_DIR=/home/${DOCKER_USERNAME}/.ccache

WORKDIR /code
ENTRYPOINT /usr/bin/bash
