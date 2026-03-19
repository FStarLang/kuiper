#!/bin/bash

set -eux

for p in FStar karamel; do
	pushd $p
	if ! git remote | grep -q upstream; then
		echo "Adding upstream remote"
		git remote add upstream https://github.com/FStarLang/$p
	fi

	if [ $p == FStar ]; then
		br=fstar2
	elif [ $p == karamel ]; then
		br=master
	else
		echo "ERROR: unknown project '$p'" >&2
		exit 1
	fi

	echo "$ git fetch upstream $br"
	git fetch upstream $br

	echo "$ git rebase upstream/$br"
	git rebase upstream/$br

	popd
done
