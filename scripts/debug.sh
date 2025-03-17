#!/bin/bash

# Debug an smt failure by running this script with an
# fst/fsti file as argument

# F="$1"
# B="$(basename "$F")"

# export O="$O --debug SMTFail --split_queries always"
# make -f verify.mk obj/$B.checked "$@"

make prepare -j$(nproc)
./fstar.sh --debug SMTFail --split_queries always "$@"
