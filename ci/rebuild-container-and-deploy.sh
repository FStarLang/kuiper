#!/bin/bash
set -eux

# Manual fallback for .github/workflows/container.yml, which builds and
# publishes these same two images on any push to main touching ci/Dockerfile.
# Use this only to push an image without landing a Dockerfile change. Requires
# `docker login ghcr.io` with a token that has write:packages.

DOCKERFILE=Dockerfile
CUDA=ghcr.io/fstarlang/kuiper-base
NOCUDA=ghcr.io/fstarlang/kuiper-base-nocuda

# Only the first build passes --no-cache: the second reuses the nocuda stage
# we just rebuilt, rather than redoing the opam install to bolt CUDA on top.
docker build --no-cache --target nocuda -f "${DOCKERFILE}" -t "${NOCUDA}" .
docker build            --target cuda   -f "${DOCKERFILE}" -t "${CUDA}" .

docker push "${NOCUDA}"
docker push "${CUDA}"

echo Done
exit 0
