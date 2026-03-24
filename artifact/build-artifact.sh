#!/bin/bash

set -eux

NAME=kuiper               # name of project, will show up in tarball
TAG=kuiper-pldi2026       # name of docker tag and stem for filenames

# Create a clean checkout of kuiper with submodules
rm -rf $NAME
mkdir $NAME
git -C .. archive HEAD | tar -C $NAME -x
# Also export submodule contents
git -C .. submodule foreach --quiet --recursive \
  'git archive HEAD | tar -C "$toplevel/artifact/'"$NAME"'/$sm_path" -x'
mv $NAME/README.md $NAME/README-original.md
cp README.md $NAME/README.md

# Remove devcontainer definition, we use code-server instead
rm -rf $NAME/.devcontainer

# Build the base image (shared with CI)
docker build -t kuiper-base -f ../ci/Dockerfile ../ci/

# Build the artifact image on top
docker build -t $TAG .

docker save $TAG -o $TAG-docker.tar
gzip $TAG-docker.tar

echo 'Done!'
