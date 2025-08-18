#!/bin/bash
set -eux

DOCKERFILE=Dockerfile
REPO=mtzguido/pulse-cuda-devcontainer2

docker build --no-cache -f "${DOCKERFILE}" -t "${REPO}" .

docker push "${REPO}"

echo Done
exit 0
