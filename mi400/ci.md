# Internal CI Flow

This doc lists our internal CI setup details and how to update components.

## Existing CI machines

Currently we have the following machines:

* Shark58 (Threadripper CPU machine)
* bg-1e715-b03-2 (Conductor mi350 machine)
* gbt350-odcdh1-b12-1 (Conductor mi350 machine)

We use GitHub Actions infrastructure to listen and dispatch workloads to CI
machines. Once a job arrives, GitHub Action runner on a local machine
runs `.github/workflows/gfx1250-ci.yml` and fires up the _local_
`ci/gfx1250-dev` docker, inside which `.github/workflows/gfx1250.sh`
gotten executed.

## Setup CI machine

### Set up directories

To add a new CI machine, first we set up the directories expected to mount to
the CI docker on the host file system:

```sh
sudo mkdir -p /data/ci/ffm /data/ci/llvm /data/ci/triton-cache /data/ci/ccache
```
where
* `/data/ci/ffm`: hosting FFM Lite packages
* `/data/ci/llvm`: hosting LLVM packages
* `/data/ci/triton-cache`: for mapping to `~/.triton`
* `/data/ci/ccache`: for mapping to `~/.ccache`

Then follow steps in [Update LLVM](#update-llvm) and [Update FFM](#update-ffm)
to populate `/data/ci/ffm` and `/data/ci/llvm` matching the current SHA and
version needed in `.github/workflows/gfx1250-ci.yml`.

We will set up GitHub Runner under your user account. So good to be
consistent regarding file ownership:

```sh
sudo chown $(id -u):$(id -g) /data/ci
```

### Build docker image

```sh
cd /data/ci/ffm/
docker build . -f /path/to/triton/mi400/gfx1250.Dockerfile -t ci/gfx1250-env \
  --build-arg DOCKER_USERID=$(id -u) --build-arg DOCKER_GROUPID=$(id -g) \
  --build-arg DOCKER_RENDERID=$(getent group render | cut -d: -f3)
```

The above creates a `mirror` user account inside the docker which has the same
UID and GID as your account. So whatever file it touches, it won't mess up
with ownership on the host.

### Configure GitHub Action runner

Then we need to set up GitHub Action runner. Download and configure:

```sh
cd /data/ci
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.329.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.329.0/actions-runner-linux-x64-2.329.0.tar.gz
echo "194f1e1e4bd02f80b7e9633fc546084d8d4e19f3928a324d512ea53430102e1d  actions-runner-linux-x64-2.329.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.329.0.tar.gz

./config.sh --url https://github.amd.com/GFX-IP-Arch/triton --token <token>
```
Get a token from https://github.amd.com/GFX-IP-Arch/triton/settings/actions/runners/new?arch=x64&os=linux.
Ask somebody with admin access if you cannot access the above page.

A list of labels will be asked to categorize the machine. You can see the
existing machines at https://github.amd.com/GFX-IP-Arch/triton/settings/actions/runners
to choose suitable ones. Note that if adding `ffm`, the machine will be
picked up for `CI/gfx1250` tasks; don't add that until proven the machine
is fully set up for it! These labels can be updated via GitHub UI anyway.

### Run GitHub Action runner

Finally start the runner in a `screen` to keep it always running (we might
want to configure it as a service but maybe later):

```sh
screen -S github-runner
# Inside the screen:
cd /data/ci/actions-runner
./run.sh
```

To detach from the current `screen`: press `Ctrl+A` and then press `d`.
To reattach, run `screen -r github-runner`.

## Update LLVM

If needing to update LLVM, compile LLVM from the SHA you need, say `abcd1234`.
Note that this needs to be in a Ubuntu 24.04 environment, which is what the
CI docker expects.

When configuring, make sure setting `CMAKE_INSTALL_PREFIX` to somewhere, say
`build/install`. Then `ninja install` to compile and copy over two binaries:

```sh
cd build/
cp bin/FileCheck install/bin/
cp bin/split-file install/bin/
```

Then mv the whole `install/` directory:

```sh
mv install/ /data/ci/llvm/abcd1234-debug
```

And then send pull request to update `.github/workflows/gfx1250-ci.sh` to use
the new path for mounting volume.

### Update FFM

Following https://amd.atlassian.net/wiki/spaces/MLSE/pages/1181562650/Triton+FFM-Lite+on+MI450,
download the new FFM package under `/data/ci/ffm` and unzip it and then
send pull request to update `.github/workflows/gfx1250-ci.sh` to use
the new path for mounting volume.

Common issues you may encounter:
1) When CI job is executing, you get "model: failed to load /ffm/lib/libhsakmtmodel.so: /ffm/lib/libhsakmtmodel.so: cannot open shared object file: Permission denied".
   Ensure that the correct permissions are set for the libhsakmtmodel.so in the FFM package. You may need to do:
    ```sh
    chmod o+rx libhsakmtmodel.so
    ```
2) When CI job is executing, you get "Error parsing TOML file: toml::parse: error opening file "/ffm/ffm_config.toml" model args not used".
   Ensure that HSA_MODEL_TOML env variable is set and correctly points to the location of the ffm_config.toml file in the FFM package. In addition,
   ensure that the correct permissions are set for ffm_config.toml. You may need to do:
   ```sh
   chmod o+r ffm_config.toml
   ```
