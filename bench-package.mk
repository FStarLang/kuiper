include .common.mk
include .configure.mk

default: build-all test
	./scripts/bench.sh

OUTDIR   := obj
include nvcc.mk
