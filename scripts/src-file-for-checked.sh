#!/bin/bash

set -eu

CHK=$1

cat .depend | sed -n 's,^'"$1"': \([^\\]*\)\\$,\1,p'
