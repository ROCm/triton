

# Setup Multithreaded MI450 FFM with Tritonn

**Workflow**: Linux Host Machine → MI450 FFM Docker Container → Triton through FFM Driver override

> ⚠️ **This guide is written and tested for a Linux host machine and could work with a Windows host with some modifications (untested).**

---

## 🔐 Setup SSH-Key Forwarding

Paste the below snippet to the end of `~/.bashrc` on the host machine that will be launching the container. This will allow the container to be authenticated via SSH with no need for SSH key copying and ownership transferring.


```bash
export SSH_ENV="$HOME/.ssh/agent-environment"
start_agent() {

    echo "Starting new ssh-agent..."
    (umask 066; ssh-agent -a "$HOME/.ssh/ssh-agent.sock" > "$SSH_ENV")
    source "$SSH_ENV"
    # Load *all* private keys that don't already live in the agent
    for key in ~/.ssh/id_*; do
        [[ -f "$key" && "$key" != *.pub ]] && ssh-add -l | grep -q "$(ssh-keygen -lf "$key" | awk '{print $2}')" || ssh-add "$key" 2>/dev/null
    done
    export SSH_AUTH_SOCK_DIR="$(dirname "$SSH_AUTH_SOCK")"
}

if [[ -f "$SSH_ENV" ]]; then
    source "$SSH_ENV" > /dev/null
    [[ "$SSH_AUTH_SOCK" != "$HOME/.ssh/ssh-agent.sock" || ! -S "$HOME/.ssh/ssh-agent.sock" ]] && start_agent
    # Agent dead?  Restart.
    kill -0 "$SSH_AGENT_PID" 2>/dev/null || start_agent
else
    start_agent
fi

# Always export the directory helper so VS Code can see it
export SSH_AUTH_SOCK_DIR="$(dirname "$SSH_AUTH_SOCK")"
```

After saving the modified `~/.bashrc`, file run `source ~/.bashrc` in the terminal. If there are any errors returned (such as Permission denied (public key)), it could mean that the file ownership and permissions are invalid on the host machine. To reset the `~/.ssh` directory and contents to the correct permissions, run the following commands:


```bash

sudo chown -R <system_username>:<system_username> ~/.ssh
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_private_key_A ~/.ssh/id_private_key_B ... (continue for all keys)
chmod 644 ~/.ssh/id_public_key_A.pub ~/.ssh/id_public_key_B.pub ... (continue for all keys)
chmod 644 ~/.ssh/config ~/.ssh/known_hosts

```

Example commands for a host user atgdebug with SSH keys `id_ed25519` and `id_rsa` would be:

```
sudo chown -R atgdebug:atgdebug ~/.ssh
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub
chmod 644 ~/.ssh/config ~/.ssh/known_hosts
```

Then run ```source ~/.bashrc``` again, and it should return without errors.

## Setup Multithreaded FFM Container (on a Linux machine)

Steps adopted from the [FFM Jitcu Package Docker WSL](https://amd.atlassian.net/wiki/spaces/GFXAM/pages/894737712/FFM+Jitcu+Package+Docker+WSL#%F0%9F%93%81-Step-2.-Create-Project-Structure) guide.

### Step 1: Install Docker
(You can skip this step if you're using conductor)

- Follow the [install docker](https://github.com/ROCm/ROCm-docker/blob/master/quick-start.md#step-2-install-docker) step from the ROCm-docker quick start guide, if docker is not already installed on the host machine.

### Step 2: Create project structure

Create a folder on your home directory:
```bash
mkdir jitcu_docker
cd jitcu_docker
```
Inside it, create a `.devcontainer/` folder containing two files:
- `devcontainer.json`
- `Dockerfile`


### Step 3: Copy contents to Dockerfile and .devcontainer.json



#### Updated `Dockerfile`

```Dockerfile

# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Set environment variables to avoid interaction during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    clang-19 \
    ninja-build \
    cmake \
    git \
    curl \
    libcurl4-openssl-dev \
    rsync \
    ssh \
    wget \
    ca-certificates \
    python3-pip \
    less \
    vim \
    python3.12-dev \
    python3.12-venv \
    libzstd-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# get the latest build from http://rocm-ci.amd.com/job/compute-rocm-npi-mi450/lastSuccessfulBuild/
RUN wget https://artifactory-cdn.amd.com/artifactory/list/amdgpu-deb/amdgpu-install-internal_7.1-24.04-1_all.deb
RUN apt-get install ./amdgpu-install-internal_7.1-24.04-1_all.deb
RUN amdgpu-repo --amdgpu-build=2219470 --rocm-build=compute-rocm-npi-mi450/605
RUN apt -y install rocm-llvm rocm-llvm-dev rocm-device-libs rocprofiler-register comgr rocm-hip-runtime-dev rocblas rocblas-dev hipblas-common-dev hipblaslt hipblaslt-dev hipblas

RUN wget --no-check-certificate https://confluence.amd.com/download/attachments/1035169166/AMD_CA.crt

RUN pip config set global.break-system-packages true

RUN apt-get remove -y python3-distro && pip install conan einops

RUN echo "export PATH=~/.local/bin:/opt/rocm/bin:$PATH" >> ~/.bashrc
RUN echo "export NODE_EXTRA_CA_CERTS=~/AMD_CA.crt" >> ~/.bashrc

# installs nodejs and npm for compiler explorer
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs
```
- You can get the latest build from http://rocm-ci.amd.com/job/compute-rocm-npi-mi450/lastSuccessfulBuild/ and update the `RUN amdgpu-repo ...` line. 544 contains the gfx1250 enabled hipBLASLt library.

#### Updated `devcontainer.json`

> ℹ️ Swap `atgdebug` with whatever username is on the host machine.

> ℹ️ In `devcontainer.json`, make sure the paths in the “mount” align with your expectations. See below an ATG example.

```json

// For format details, see https://aka.ms/devcontainer.json. For config options, see the
// README at: https://github.com/devcontainers/templates/tree/main/src/docker-existing-dockerfile
{
	"name": "jitcu_docker",
	"build": {
		"dockerfile": "Dockerfile",
	},
	"mounts": [
        // Required for ssh key forwarding
		"type=bind,source=${localEnv:SSH_AUTH_SOCK},target=/ssh-agent,type=bind",

		// Optional, mount some dev directory
    	// "type=bind,source=C:\\dev,target=/mnt/c/dev,consistency=cached"
		"type=bind,source=/home/atgdebug/dev/,target=/mnt/c/dev,consistency=cached"
	],
	// Features to add to the dev container. More info: https://containers.dev/features.
	// "features": {},
	// Use 'forwardPorts' to make a list of ports inside the container available locally.
	// "forwardPorts": [],
	// Uncomment the next line to run commands after the container is created.
	// "postCreateCommand": "cat /etc/os-release",
	// Configure tool-specific properties.
	// "customizations": {},
	// Uncomment to connect as an existing user other than the container default. More info: https://aka.ms/dev-containers-non-root.
	"capAdd": [ "SYS_PTRACE" ],
	"securityOpt": [ "seccomp=unconfined" ],
	"remoteUser": "root",
	"privileged": true
}

```

## FFM Setup Script

Steps also adopted from the [FFM Jitcu Package Docker WSL](https://amd.atlassian.net/wiki/spaces/GFXAM/pages/894737712/FFM+Jitcu+Package+Docker+WSL#%F0%9F%A7%A9-Step-5.-Open-in-VS-Code) guide.

- Open jitcu_docker folder in VS Code.
- Install the Dev Containers extension.
- Open command palette with Ctrl + Shift + P, then search for `Dev Containers: Reopen in Container` and enter.
- Wait for Docker build to complete. (This will take a while, so you can go for a 🍵 or a 🏃‍♂️‍➡️)
- In VS Code, go to File → Add Folder to Workspace..., and select the folder mounted as `/workspaces/jitcu_docker` inside the container
- Wait for VS Code to reload.

Then, inside the `/workspaces/jitcu_docker` directory, run the below commands to install FFM:

```bash

read -p "Enter AMD username: " USERNAME
read -s -p "Enter AMD password: " PASSWORD
echo

export USERNAME
export PASSWORD

mkdir -p /workspaces/mi450_5
cd /workspaces/mi450_5
wget --user="$USERNAME" --password="$PASSWORD" https://logviewer-atl.amd.com/proj/gfxip_sde_users10/jordans/data/jitcu_package_2/mi450/5/jitcu.tar.gz
tar xzvf jitcu.tar.gz
rm -rf jitcu.tar.gz
rm -rf ffm

# Multithreading has now been merged to main branch
git clone git@github.amd.com:GFX-Modeling/shader_complex_ffm.git ffm

# Fix a known compilation failure
sed -i 's|^#include[[:space:]]*<sq_uc/sp3_inst_info.h>|#include <sq_uc/sp3_mi400_inst_info.h>|' ./ffm/libs/funclib/shader/jitcu_base/src/jitcu_analyze.h
sed -i 's|clang-llvm|llvm-core|' ./conanfile.py

conan remote add gfxip_conan_local https://atlartifactory.amd.com/artifactory/api/conan/gfxip_conan_local --force
conan remote login gfxip_conan_local $USERNAME -p $PASSWORD

# Ran conan profile detect from within the jitcu_docker/mi450_5 directory
# Add --exist-ok if you want to rerun the script
conan profile detect

# Release build
conan build . -pr:a $HOME/.conan2/profiles/default -s build_type=Release -of . -o rocr_bridge=ON --build=missing

```

## Set Environment Variables

```bash
export FFM_PATH=/workspaces/mi450_5
export HSA_MODEL_LIB=$FFM_PATH/_builds/Release/lib/libhsakmtmodel.so
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HSA_MODEL_TOPOLOGY=$FFM_PATH/topology
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib
export TARGET_ARCH=gfx1250
export HSA_MODEL_NUM_THREADS=$(nproc)
```
- Ensure `HSA_MODEL_NUM_THREADS` has a value for multithreading to work.

## Sanity Check: Run simple.cpp test

Copied from [here](https://amd.atlassian.net/wiki/spaces/GFXAM/pages/721063271/FFM+Jitcu+Package+-+ROCR+in+WSL) (Build and Run Simple.cpp section)

simple.cpp: (you can also get the latest from [here](https://amd.atlassian.net/wiki/download/attachments/721063271/simple.cpp?version=1&modificationDate=1738883082143&cacheVersion=1&api=v2)).
```cpp
#define __HIP_PLATFORM_AMD__
#define __HIP_CLANG_ONLY__
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// GPU Kernel
__global__ void kernel_add(int* a, int b) {
  int idx = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
  a[idx] += b;
}

int main() {
  constexpr size_t size = 100;
  int* ptr;
  hipMalloc(&ptr, sizeof(int) * size);
  hipMemset(ptr, 0, sizeof(int) * size);
  std::vector<int> input(size, 0);
  size_t i = 100;
  std::for_each(input.begin(), input.end(), [&](int& a) { a = i; });
  // Print input vector
  std::for_each(input.begin(), input.end(), [&](int a) { std::cout << a << " " ;});
  std::cout << std::endl;
  hipMemcpy(ptr, input.data(), sizeof(int) * size, hipMemcpyHostToDevice);
  // Run kernel
  kernel_add<<<1, size>>>(ptr, 10);
  std::vector<int> output = input;
  hipMemcpy(output.data(), ptr, sizeof(int) * size, hipMemcpyDeviceToHost);
  // Print output vector
  std::for_each(output.begin(), output.end(), [&](int a) { std::cout << a << " " ;});
  std::cout << std::endl;
  // Verify the output vector against the expected one.
  std::cout << ((std::all_of(output.begin(), output.end(), [&](int a) { return a == (i + 10); }))
                    ? "passed"
                    : "failed")
            << std::endl;
  hipFree(ptr);
}
```

```bash
cd /workspaces/jitcu_docker

mkdir simple_hip
cd simple_hip
touch simple.cpp # copy & paste the content here

# Copy simple.cpp into this folder.  Link to the source file for simple.cpp above
amdclang -g -O1 -x hip --offload-arch=$TARGET_ARCH --rocm-path=$ROCM_PATH -rpath $ROCM_PATH/lib -lstdc++ simple.cpp -I$ROCM_PATH/include

#Run
./a.out
```
If the tests works, you should see a line of "100"'s, and then a second line with "110"'s, followed by the word "passed"

If it doesn't work, check:
- $TARGET_ARCH is gfx1200 for navi44, gfx1250 for mi450
- Check your $HSA_MODEL_TOPOLOGY - it should point to the folder that contains the generation_id file.

## Using Triton to Write & Test Kernels for MI450 in FFM

This may not be the correct place for these instructions but putting them here for now. Adopted from Giuseppe's page found [here](https://amd.atlassian.net/wiki/spaces/MI400AGSSW/pages/588940366/Triton+on+FFM).

### Setup Environment

The environment does not have to be inside the same FFM container that is setup above, can be done natively on host machine as well.

Compiling triton kernels for MI450 requires access to:
- https://github.amd.com/GFX-IP-Arch/triton
- https://github.com/AMD-Lightning-Internal/llvm-project

First clone the triton repository:
```bash
git clone https://github.amd.com/GFX-IP-Arch/triton.git
git checkout mi400_triton_dev
```

Then, clone the llvm repository and build:
```bash
git clone https://github.com/AMD-Lightning-Internal/llvm-project.git
cd llvm-project
git checkout `cat path_to_triton_repo/cmake/llvm-hash.txt`
mkdir build && cd build
cmake -G Ninja ../llvm -DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
-DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"

# build
ninja
```

Once LLVM is built, `cd` into the triton repo root directory. build/install the triton python package with:
```bash
# [Optional]If you run into any "Command not found" error, you may want to install these dependencies
apt-get install -y cmake
pip install ninja cmake wheel pybind11
pip install numpy==1.26.4

# Build and install Triton with custom llvm
export LLVM_BUILD_DIR=/path/to/llvm-project/build

 LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  pip3 install --no-build-isolation --verbose .
```
- If there is a CMAKE not found error, and CMAKE is installed, you might need to prepend `PATH=/home/atgdebug/tools/cmake-3.31.6/bin:$PATH \` to the command above.


### Run tests

First export necessary environment variables:
```bash
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export TRITON_TEST_OUTPUT_DIR=/path/to/triton_repo
export TRITON_HIP_LLD_PATH=/path/to/llvm-project/build/bin/ld.lld
export LLVM_BIN_PATH=/path/to/llvm-project/build/bin
export FFM_BIN_PATH=/path/to/ffm/_builds/Release/bin
```

Install hip-python from pypi:

```bash
pip install -i https://test.pypi.org/simple/ hip-python
```

You can just run any kernel by running the associated test script. Example for MXFA kernels:

```bash
# MXFP-FA
python3 mi400/test_mxfa_hipdriver.py

# or MXFP-GEMM
python3 mi400/test_mxgemm_hipdriver.py
```

This requires a pytorch installation. Use any ROCm pytorch nightly build:

```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.4
pip3 uninstall -y pytorch-triton-rocm
```

(doesn't matter if it doesn't support gfx1250 - the point is to only use the CPU part of pytorch and while `.cuda()` maps directly to the HIP runtime, which doesn't require gfx1250 specific support.)


### Compiler Explorer Setup

##### On your local machine

In order to display `compiler explorer` on your local machine (e.g., laptop) you need to establish a tunnel between your machine and a remote ssh server which you use for development (i.e., local port forwarding). For example:

```bash
ssh -L 3000:<server_name>:10240 <user_name>@<server_name>
```

This connects you to a `ssh-server`; opens port `10240` on the `ssh-server`; and forwards the traffic from that port to `3000` on your local machine.


##### On your remote machine

Make sure you sourced all necessary env. variables to run Triton (see above).

Get inside of your dev. container. Make sure that you mapped the host network ports inside your dev. container. The mapping is done when you spin your dev. container. For example,

```bash
docker run --rm -it -d --network host ...
```

You need to install `compiler explorer`. The provided Dockerfile (see above) already has `nodejs` and `npm` installed.

```bash
git clone https://github.com/compiler-explorer/compiler-explorer.git
cd compiler-explorer
```

Now you need to adjust the `compiler explorer` configuration with the location of your local Triton installation.

```bash
$ cat etc/config/triton.defaults.properties
compilers=&triton_amd
defaultCompiler=triton_amd_gfx1250
compilerType=triton
interpreted=true
supportsBinary=false
supportsExecute=false
isSemVer=true
notification=Experimental Triton support on Compiler Explorer. For tutorials, bugs reports, and feature requests, please visit <a href="https://github.com/ShawnZhong/compiler-explorer-triton">here</a>.

group.triton_amd.compilers=triton_amd_gfx1250
group.triton_amd.groupName=Triton (AMD)
group.triton_amd.options=--backend hip --arch gfx1250 --warp_size 64

compiler.triton_amd_gfx1250.name=Triton Custom (AMD)
compiler.triton_amd_gfx1250.exe=<home>/venv/bin/python3
```
Make sure to substitute `<home>/venv/bin/python3` with the absolute path of your python interpreter.

You will also need to copy `triton_wrapper.py` to the correct location.

```bash
make scripts
cp ./etc/scripts/triton_wrapper.py ./out/dist/etc/scripts/
```

After that you can compile and run `compiler-explorer` as follows:


```bash
make
```

##### On your local machine

Now go back to your local machine; open `http://localhost:3000/` with your browser; and select `Triton` in the drop-box on the left panel as the source language.
