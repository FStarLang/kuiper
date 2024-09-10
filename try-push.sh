#!/bin/bash

for p in FStar karamel pulse; do
	pushd $p
	if ! git remote | grep -q upstream; then
		echo "Adding upstream remote"
		git remote add upstream https://github.com/FStarLang/$p
	fi

	echo "$ git remote update"
	git remote update

	# if [ $p == pulse ]; then
	#         br=main
	# else
	#         br=master
	# fi

	echo "$ git push origin HEAD:gpu"
	git push origin HEAD:gpu

	popd
done
