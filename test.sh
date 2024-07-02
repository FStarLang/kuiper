#!/bin/bash

set -eux

MOD=GPU.DotProduct

FILE=${MOD}.fst
OFILE=GPU_DotProduct.c

make -j32 -f Makefile.verify .cache/${FILE}.checked

V=1 ./pulse.sh src/examples/${FILE} --codegen krml --extract GPU.Ref,GPU.Array,GPU.Base,${MOD} --debug extraction

/home/guido/r/karamel/krml \
	-add-early-include '"../GPU.h"' \
	-fc++-compat \
	-fcast-allocations \
	-verbose -dast \
	-skip-compilation -warn-error -2 -tmpdir kout \
	.out/out.krml

# Stuff to add to karamel:

# Somehow the kcall has an extra unit(?) arg
# sed -i 's/PULSE_KCALL\(.*\), (void \*)0U/PULSE_KCALL\1/' kout/"${OFILE}"

# File extension
ln -sf "${OFILE}" kout/final.cu

nvcc -c kout/final.cu -o kout/final.o -I /home/guido/r/karamel/include/ -I /home/guido/r/karamel/krmllib/dist/minimal/
