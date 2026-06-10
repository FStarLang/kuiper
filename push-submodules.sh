#!/bin/bash

set -eux

for p in FStar karamel; do
	pushd $p

	echo "$ git push -f origin HEAD:refs/kuiper/objects"
	git push -f origin HEAD:refs/kuiper/objects

	popd
done
