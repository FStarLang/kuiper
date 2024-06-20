#!/bin/bash

set -eu

PULSE_REPO="$1"
PULSE_HASH="$2"

if [[ -e pulse ]] && ! [[ -d pulse ]]; then
	echo "error: pulse exists and not a directory?"
	false
fi

if ! [[ -d pulse ]]; then
	git clone "${PULSE_REPO}" pulse/
fi

pushd pulse

git remote update
git reset --hard "${PULSE_HASH}"

popd

exit 0
