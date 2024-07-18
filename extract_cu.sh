#!/bin/bash

unset MAKEFLAGS

set -eux

MOD="$1"
FILE=${MOD}.fst
OFILE="${MOD/./_}.c"
OUTD=out/${MOD}

mkdir -p "${OUTD}"

make -j32 -f verify.mk .cache/${FILE}.checked

V=1 ./pulse.sh src/examples/${FILE} --codegen krml --extract "-*,+GPU,+${MOD}" --odir "${OUTD}" # --debug extraction

R=$(pwd)

pushd "${OUTD}"
V=1 "${R}"/krml.sh out.krml

# Tweak file extension
ln -sf "${OFILE}" "${OFILE/.c/.cu}"

popd
exit 0
