#!/bin/bash
set -eux

DOCKERFILE=.devcontainer/minimal.Dockerfile
REPO=mtzguido/pulse-sandbox-devcontainer

# -f to detect worktrees too, where .git is not a directory
if ! [ -d .git ] && ! [ -f .git ]; then
	echo "This script must be run from the root of the repo" >&2
	exit 1
fi

if ! [ x"$(git clean -dnx)" == x"" ]; then
	echo "Repository seems dirty: aborting" >&2
	exit 1
fi

docker build -f "${DOCKERFILE}" -t "${REPO}" .

docker push "${REPO}"

echo Done
exit 0
