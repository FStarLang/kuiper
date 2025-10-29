#!/bin/bash

set -eux

for p in FStar karamel pulse; do
	pushd $p
	if ! git remote | grep -q upstream; then
		echo "Adding upstream remote"
		git remote add upstream https://github.com/FStarLang/$p
	fi

	if [ $p == pulse ]; then
		br=gebner_phase1_matcher
	else
		br=master
	fi

	echo "$ git fetch upstream $br"
	git fetch upstream $br

	echo "$ git rebase upstream/$br"
	git rebase upstream/$br

	popd
done
