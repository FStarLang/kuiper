#!/bin/bash

for p in FStar karamel pulse; do
	pushd $p

	echo "$ git push -f origin HEAD:gpu"
	git push -f origin HEAD:gpu

	popd
done

set -eux

git add FStar karamel pulse
git commit -m 'bump submodules'
git push
