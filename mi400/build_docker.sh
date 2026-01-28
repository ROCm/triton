#!/bin/bash

# ========================================================
# Build docker images for gfx1250
#
# Prerequisites:
# 1. Run `docker login mkmhub.amd.com` to authenticate
# 2. Download ffmlite and place it under ./ffmlite
# ========================================================

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

VERSION=$1

set -euxo pipefail

REPO="mkmhub.amd.com/sw-aiggfx1250-dev"

# Public PyTorch
TAG="ci/gfx1250-env:${VERSION}"
docker build . -f ./gfx1250.Dockerfile -t "${REPO}/${TAG}"
docker push "${REPO}/${TAG}"

# NPI PyTorch
TAG="ci/gfx1250-pytorch-env:${VERSION}"
docker build . -f ./gfx1250.Dockerfile -t "${REPO}/${TAG}" --build-arg USE_NPI_TORCH=TRUE
docker push "${REPO}/${TAG}"

# Roccap
TAG="ci/gfx1250-roccap:${VERSION}"
docker build . -f ./gfx1250.Dockerfile -t "${REPO}/${TAG}" --build-arg USE_ROCCAP=TRUE
docker push "${REPO}/${TAG}"
