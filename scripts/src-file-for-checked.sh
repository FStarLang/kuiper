#!/bin/bash

set -eu

CHK=$1
CHK=${CHK//\//\\\/} # escape slashes

cat .depend | sed -n "/^$CHK: \\\\$/{n;s,^[[:space:]]*\([^\\ ]*\) \\\\$,\1,p}"
