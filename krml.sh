#!/bin/bash

SNAME="$0"

gcmd () {
	cd $(dirname $0)
	make -s echo-krml
}

exec $(gcmd) "$@"
