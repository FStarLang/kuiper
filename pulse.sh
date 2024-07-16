#!/bin/bash

# A helper to call F* with all the relevant flags to check a Pulse
# file in this repo.

SNAME="$0"

gcmd () {
	cd $(dirname $0)
	make -s echo-fstar
}

exec $(gcmd) "$@"
