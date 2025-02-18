#!/bin/bash

for p in FStar karamel pulse; do
	pushd $p
	if ! git remote | grep -q upstream; then
		echo "Adding upstream remote"
		git remote add upstream https://github.com/FStarLang/$p
	fi

	# if [ $p == pulse ]; then
	#         br=main
	# else
	#         br=master
	# fi

	echo "$ git push -f origin HEAD:gpu"
	git push -f origin HEAD:gpu

	popd
done

set -eux

git add FStar karamel pulse
git commit -m 'bump submodules'
git push
