include .common.mk
include .configure.mk

default: build-all
	./scripts/bench.sh

OUTDIR   := obj
include nvcc.mk
