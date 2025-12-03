

# Setup Multithreaded MI450 FFM with Triton

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
RUN amdgpu-repo --amdgpu-build=2241635 --rocm-build=compute-rocm-npi-mi450/695
RUN apt -y install rocm-llvm rocm-llvm-dev rocm-device-libs hsa-amd-aqlprofile rocprofiler-register comgr rocm-hip-runtime-dev rocblas rocblas-dev hipblas-common-dev hipblaslt hipblaslt-dev hipblas

RUN wget --no-check-certificate https://confluence.amd.com/download/attachments/1035169166/AMD_CA.crt

RUN pip config set global.break-system-packages true

RUN apt-get remove -y python3-distro && pip install conan einops requests Jinja2

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

#### Start Containers in CLI

In case you prefer starting containers in commandline, you can use this script:

```bash
docker build -t jitcu_docker:0.1 -f Dockerfile .

docker run -dt \
    --name jitcu_docker \
    --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined \
    --shm-size=16gb \
    --cap-add=SYS_PTRACE \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    -v ${SSH_AUTH_SOCK}:/ssh-agent \
    jitcu_docker:0.1 \
    /bin/bash
```

## Setup FFM
### Setup FFM-Lite and Environment

Please follow [Triton + FFM-Lite on MI450 - Machine Learning Software Engineering - Confluence](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1181562650/Triton+FFM-Lite+on+MI450) to install prerequisites and set up FFM-Lite.

### Setup via scripts

You can also setup the env with Danial's scripts: https://github.amd.com/GFX-IP-Arch/pre_silicon_sw_dev

### Use FFM Config File

You can change configurations to have fine-grained control over FFM:
<details>
<summary>FFM Config File: `ffm.config.toml`</summary>
<br>

```bash
$ cd <work>
$ cat ./ffm.config.toml

# forces itrace printouts to include all enabled threads/lanes instead of just the first and last active ones
# this will also dump WMMA matrix data to itrace file for MIs
[ffm_itrace_all_threads]
dona.component.jitcu_itrace.all_lanes=true

# FFM runs waves serially within a workgroup which causes waves to FFM-yield after signaling on a barrier or semaphore
# this allows other waves receiving the signal to run, simulating concurrently running waves
# useful for synchronisation app debugs
[ffm_debug_sync_signals]
dona.component.shader_complex.debug_barier_semaphore=true

# dump WMMA matrix data to itrace for MIs
# itrace must be enabled, good alternative to ffm_itrace_all_threads for debugging
[ffm_itrace_dump_matrix]
dona.component.jitcu_itrace.dump_matrix=true

# limits itrace dumps to only VMEM instructions
# itrace must be enabled
[ffm_itrace_vmem_only]
dona.component.jitcu_itrace.vmem_only=true

# use fast float implementations instead of the ones in Math Engine
# JIT-able and typically much faster but is not guaranteed to be bit accurate
# some operations do not have a fast float implementation; these will fall back to MathEn
[ffm_use_fast_float]
dona.component.shader_complex.use_fast_float=true

# enable JIT compile
# whether this is faster or not depends on the type of the workload
[ffm_use_jit]
dona.component.shader_complex.use_jit=true

# waves run round-robin, serially within a workgroup in-order of wave ID by default
# this option changes the execution order to random wave execution order
# useful for synchronisation app debug
[ffm_exec_order_random]
dona.component.shader_complex.exec_order=2

# waves run round-robin, serially within a workgroup in-order of wave ID by default
# this option changes the execution order to decreasing wave execution order
# useful for synchronisation app debug
[ffm_exec_order_descending]
dona.component.shader_complex.exec_order=1

# waves run round-robin, serially within a workgroup in-order of wave ID by default
# this option changes the execution order to ascending wave execution order, i.e. any time wave 0 can run it will run
# useful for synchronisation app debug
[ffm_exec_order_ascending]
dona.component.shader_complex.exec_order=3

# force memory dependency check for s_wait operations
# this is for wait counter debug for VGPR/LDS access
# search "Application Wait Count Debug" on Confluence for details
[ffm_force_mem_dep_check]
dona.component.shader_complex.force_mem_dep_check=true

# Allows waves / workgroups / clusters to be run in a timesliced manor.
# This will enabling coding patterns such as consumer / producer models
[ffm_enable_time_slicing]
dona.component.jitcu.enable_time_slicing=true

# only dump itrace for this wave, set to specified debug ID, in hex
[ffm_itrace_wave_id_filter]
dona.component.jitcu_itrace.wave_id=0x1

# if set, sgprs and vpgrs are initialized to a known value and an assert is triggered when trying to read from an
# uninitialized gpr
[ffm_validate_gprs_strict]
dona.component.shader_complex.validate_gprs_strict=true
```
</details>

To use it, you need to save the above to `ffm.config.toml` and set two environment variables:
```bash
# Where the config file is
export HSA_MODEL_TOML="/path/to/ffm.config.toml"

# Which knob you want to trigger(use space as separator)
export HSA_MODEL_ARGS="ffm_exec_order_random ffm_itrace_wave_id_filter"
```

### Disable Uninitialized SGPR/VGPR Reads Checks

You can do the following if you need to temprary disable the uninitialized SGPR/VGPR reads checks:

```bash
$ unset HSA_MODEL_TOML
$ unset HSA_MODEL_ARGS
```

**Note**: you need to test your kernels with the check enabled to avoid crashes on the AM.


## Sanity Check: Run simple.cpp test

<details>
<summary>Toy Problem: Source files</summary>
<br>

```cpp
// simple.cpp

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

```Makefile
# Makefile
CC=amdclang
TARGET_ARCH=gfx1250
ROCM_PATH=/opt/rocm

all:
        $(CC) -g -O1 -x hip --offload-arch=$(TARGET_ARCH) --rocm-path=$(ROCM_PATH) -rpath $(ROCM_PATH)/lib -lstdc++ simple.cpp -I$(ROCM_PATH)/include

clean:
        rm -rf ./a.out ./roc_capture*
```
</details>


Copy files from `Toy Problem` (see above) and execute the following steps:

```bash
$ cd <work> && mkdir toy && cd toy
# copy files from above
$ make
$ ./a.out
```

If the tests works, you should see a line of "100"'s, and then a second line with "110"'s, followed by the word "passed"

If it doesn't work, check:
- $TARGET_ARCH is gfx1200 for navi44, gfx1250 for mi450
- Check your $HSA_MODEL_TOPOLOGY - it should point to the folder that contains the generation_id file.

### Install Rocplay to capture CAP files

You can all `Rocplay` versions under [this URL address](https://atlartifactory.amd.com:8443/artifactory/HW-RocPlayCap-REL/). You can find `pre-release` and `release` versions there.

```bash
$ cd <work>
$ mkdir rocplaycap && cd rocplaycap
$ VERSION="4.5.1"
$ TYPE="releases"
$ wget https://atlartifactory.amd.com/artifactory/HW-RocPlayCap-REL/${TYPE}/rocplaycap-${VERSION}/rocplaycap-src-${VERSION}.tar.gz
$ tar -xvf ./rocplaycap-src-${VERSION}.tar.gz
$ cd rocplaycap-src-${VERSION}
$ mkdir cmake-build-debug && cd cmake-build-debug
$ CMAKE_BUILD_TYPE=debug CMAKE_DEBUG_TRACE=1 cmake .. -DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_PREFIX_PATH=/opt/rocm/ \
-DHSA_ROOT_DIR:PATH=/opt/rocm/hsa/
$ make all && make install
```

Make sure that you have the `hsa-amd-aqlprofile` package installed (see `Dockerfile` above). After installation, you can find executables under `<work>/rocplaycap/rocplaycap-src-${VERSION}/bin`. You can adjust your `PATH` env. variable to the aforementioned folder. For example,

```bash
export PATH=$(realpath ./rocplay/rocplaycap-src-4.*/bin):${PATH}
```

Test your rocplay installation using the `Toy Problem` (see above).

```bash
export HSA_KMT_MODEL_GPUVM_BASE=0x200000000 # NEEDED! do not forget.
export HSA_KMT_MODEL_GPUVM_SIZE=0xF00000000
```

```bash
$ cd <work>/toy
$ make
$ roccap capture --loglevel trace ./a.out
```

You will see `roc_capture_a.out.cap` generated in the directory where you launched the toy program. Usually you need to deliver cap files (like this one) to the AM team. Replay the cap file, to make sure that it is valid:

```bash
$ roccap play ./roc_capture_a.out.cap
...
[... INFO] Trace completed successfully.
```

You can find the installation details [there](https://amd.atlassian.net/wiki/spaces/~avulisha/pages/1068105750/Duplicate+of+Replaying+AQL+traces+captured+on+FFM+on+top+of+AM#Compiling-Rocplaycap-4.5.0). [Here](https://github.amd.com/GFX-IP-Arch/triton/blob/shared/gfx1250-dev-sept/mi400/test_mxfa_hipdriver.sh) you can find an example how to generate CAP files for Triton.

#### Filtering Kernels while capturing with Rocplay

The cap files may contain several kernel dispatches which are generated by the runtime and are not needed for the AM simulation. It **highly recommended** to remove them before sending a job to the AM. Here are a few examples for the runtime kernels which may be included in the trace - e.g., `__amd_rocclr_fillBufferAligned.kd`, `__amd_rocclr_copyBuffer.kd`.

Generate a dispatch log file as follows:

```bash
$ roccap capture --loglevel trace --dispatchlog <filename>.txt <you execution command>
```

Open the generated file and find the kernel id which corresponds to the kernel of your interest.
Exampel:

```bash
$ cat <filename>.txt
0: __amd_rocclr_copyBuffer.kd; Workgroup size xyz: (512,1,1); Grid size xyz: (1,65537,16843009); (no metadata) kernarg_size=48 6 Args: 0x35c1200000, 0x35c0c00000, 0x1000, 0x1000000000, 0x35c0c01000, 0x20000000200
1: __amd_rocclr_copyBuffer.kd; Workgroup size xyz: (512,1,1); Grid size xyz: (1,65537,16843009); (no metadata) kernarg_size=48 6 Args: 0x35c1201000, 0x35c0c01000, 0x80, 0x1000000000, 0x35c0c01080, 0x20000000200
2: __amd_rocclr_copyBuffer.kd; Workgroup size xyz: (512,1,1); Grid size xyz: (1,65537,16843009); (no metadata) kernarg_size=48 6 Args: 0x35c1500000, 0x35c0d01200, 0x2000, 0x1000000000, 0x35c0d03200, 0x20000000200
3: attn_fwd_pipelined_kernel.kd; Workgroup size xyz: (128,1,1); Grid size xyz: (1,2097153,218169601); (no metadata) kernarg_size=72 9 Args: 0x35c0c00000, 0x35c1a00000, 0x35c3e00000, 0x35c0c01000, 0x35c0c01200, 0x35c6200000, 0x35c0d01200, 0x0, 0x0
4: __amd_rocclr_copyBuffer.kd; Workgroup size xyz: (512,1,1); Grid size xyz: (1,65537,16843009); (no metadata) kernarg_size=48 6 Args: 0x35c0d01200, 0x35c1502000, 0x2000, 0x1000000000, 0x35c1504000, 0x20000000200
```

The kernel of our interest is called `attn_fwd_pipelined_kernel.kd` which has its `id` equal to `3`. We are going to run `rocpaly` the second time and capture this specific kernel by its id. To be consistent, remove the existing cap file:

```bash
$ rm -rf ./roc_capture*
$ roccap capture --loglevel trace --disp 3 <you execution command>
```

You can double-check that only a single kernel was captured:

```bash
$ tar -xvf ./roc_capture*.cap
$ cat roc_capture.json | grep "kernelName"
```

Make sure that there is only a single entry in the output.

#### Validation testing with Rocplay

Use `--autogold` option while recording the traces.

```bash
$ cd <work>/toy
$ make
$ roccap capture --loglevel trace --autogold ./a.out
```

Now you need to replay traces as you did it before:

```bash
$ roccap play ./roc_capture_a.out.cap
```

Note, `aqlplay_goldchecks.json` gets generated together with a bunch of `*.bin.lz4` files.

### Capture CAP files via Github Action Workflow
Now you can use our Github Action Workflow to capture CAP files following these steps:
1. Go to https://github.amd.com/GFX-IP-Arch/triton/actions/workflows/gfx1250-roccap.yml
2. Click 'Run workflow'.
3. Choose the branch where you want to capture CAP files.
4. Fill in other blanks. Meanings of options are listed below.
5. Click 'Run workflow' button to trigger the workflow.
6. Download artifacts from the webpage once the workflow finished.

#### Workflow options
| Name          | Description                                                                                                                                                                                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CapFileName   | Name of the cap file(without extension). The cap file will be generated as ${CapFileName}.cap                                                                                                                                                                                |
| CapFileRoot   | Where you want to put your cap file later. Make sure to put cap file to this directory and give rwx permission bits.                                                                                                                                                         |
| Enable ITrace | Enable itrace or not. Enabling itrace will take significant more time to finish.                                                                                                                                                                                             |
| Enable TTrace | Enable ttrace or not.                                                                                                                                                                                                                                                        |
| KernelRegex   | Specify the kernel name regex to capture. Kernel name regex should match the python function name decorated by @triton.jit / @gluon.jit. For example, attn_fwd_.*                                                                                                            |
| DispatchName  | Specify the dispatch number to capture. Dispatch number "0" means capture the first dispatch of the matched kernel.                                                                                                                                                          |
| GroupName     | [Optional]Change the group name to better represent your workload. It's fine to not change.                                                                                                                                                                                  |
| NumXCCs       | Number of XCCs. Choice of XCCs depends on the problem size. It's recommended to use 8 XCCs for kernel with more than 64 WGs and use 1 XCC for smaller kernel.                                                                                                                |
| Command       | Command to run the kernel. Triton source code is under /code/. Both absolute and relative path are supported. For example, you can run python3 ./third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py, or /code/third_party/amd/python/examples/gluon/mxfp_fa_gfx1250.py |

Please check https://github.amd.com/GFX-IP-Arch/triton/blob/shared/gfx1250-dev-sept/.github/workflows/gfx1250-roccap.yml for details.

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
