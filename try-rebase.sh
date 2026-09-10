#!/bin/bash

set -eux

for p in FStar; do
	pushd $p
	if ! git remote | grep -q upstream; then
		echo "Adding upstream remote"
		git remote add upstream https://github.com/FStarLang/$p
	fi

	br=master

	echo "$ git fetch upstream $br"
	git fetch upstream $br

	echo "$ git rebase upstream/$br"
	git rebase upstream/$br

	popd
done
