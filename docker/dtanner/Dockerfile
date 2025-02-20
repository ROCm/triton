# https://confluence.amd.com/display/MLSE/Docker+Setup
ARG MUID=1000
ARG MGID=1000
ARG USERNAME=developer

# https://hub.docker.com/r/rocm/pytorch/tags
# FROM rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2
FROM rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0
#FROM rocm/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN userdel jenkins
RUN apt-get update
RUN apt-get install -y netcat-openbsd

# Here is all personal preference, with a few notes on software you may find useful
RUN apt-get update && \
apt-get install -y vim zsh zsh-doc tmux git less man man-db mlocate graphviz cmake-curses-gui ccache bash-completion && \
apt-get install -y silversearcher-ag keychain htop rcm && \
apt-get install -y libbz2-dev liblzma-dev libssl-dev libreadline-dev &&\
apt-get install -y clang libclang-14-dev
#clang and libclang were needed to build ccls when upgrading to rocm623

# Needed for Tensile
RUN apt-get install -y libmsgpack-dev libboost-program-options-dev python3-joblib python3-msgpack

# Remove pre-build triton version
RUN pip uninstall -y triton

# These were needed to build Triton; experimenting with adding them to Dockerfile
RUN pip install --upgrade pip
RUN pip install ninja cmake wheel pybind11 scipy numba numpy torch pytest lit pandas matplotlib contourpy websockets librosa pre-commit
# put all dependencies on same line, so pip can resolve interdependencies.

# Build ccls if you want to
RUN cd /opt && git clone --depth=1 --recursive https://github.com/MaskRay/ccls && \
mkdir -p /opt/ccls/build && cd /opt/ccls/build && \
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-${LLVM_VER} \
    -DLLVM_INCLUDE_DIR=/usr/lib/llvm-${LLVM_VER}/include && \
make -j8 && \
make install && \
cd /opt && rm -rf /opt/ccls

# Install Neovim if you want to
RUN apt-get install -yqq ninja-build gettext cmake unzip curl build-essential && \
wget --progress=bar:force:noscroll https://github.com/neovim/neovim/archive/refs/tags/v0.9.5.tar.gz && \
tar -xvf ./v0.9.5.tar.gz && cd neovim-0.9.5 \
make CMAKE_BUILD_TYPE=RelWithDebInfo && make install && \
cd .. && rm v0.9.5.tar.gz && rm -rf neovim-0.9.5

# Arcanist is needed for upstream, rocm-debug-agent is useful when debugging
RUN apt-get -y install arcanist rocm-debug-agent && updatedb

ARG MUID
ARG MGID
ARG USERNAME

RUN addgroup --gid ${MGID} ${USERNAME}
RUN useradd -d /home/${USERNAME} -g ${MGID} --no-create-home -u ${MUID} --shell /usr/bin/bash ${USERNAME}
RUN adduser ${USERNAME} sudo
RUN adduser ${USERNAME} video
RUN adduser ${USERNAME} render
# The /var is only needed on lockhart
RUN mkdir /home/${USERNAME} /var/${USERNAME}
RUN chown ${USERNAME}:${USERNAME} /home/${USERNAME} /var/${USERNAME}
RUN chown -R ${USERNAME}:${USERNAME} /opt/conda

# Make sudo work without a password
RUN sed -i~ -e 's/%sudo\tALL=(ALL:ALL) ALL/%sudo\tALL=(ALL:ALL) NOPASSWD:ALL/g' /etc/sudoers

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV HOME=/home/${USERNAME}

# Keep container alive
CMD tmux new-session -d && tail -f /dev/null
