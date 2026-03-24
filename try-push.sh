#!/bin/bash

set -eux

for p in FStar karamel; do
	pushd $p

	echo "$ git push -f origin HEAD:refs/heads/gpu2"
	git push -f origin HEAD:refs/heads/gpu2

	popd
done

git add FStar karamel
git commit -m 'bump submodules'
git push
