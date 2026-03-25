#!/bin/bash

set -eux

for p in FStar karamel; do
	pushd $p

	echo "$ git push -f origin HEAD:refs/heads/_kuiper"
	git push -f origin HEAD:refs/heads/_kuiper

	popd
done

git add FStar karamel
git commit -m 'bump submodules'
git push
