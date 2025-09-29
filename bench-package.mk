include .common.mk
include .configure.mk

default: build-targets
	./scripts/bench.sh

OUTDIR   := obj
include nvcc.mk
