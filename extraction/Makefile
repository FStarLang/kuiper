# This makefile is not great and will repeat some work by checking ulib modules.

FSTAR_EXE ?= ../FStar/out/bin/fstar.exe
PULSE_ROOT ?= ../pulse
include $(PULSE_ROOT)/mk/locate.mk

TAG := extraction
SRC := .
CACHE_DIR := _cache
OUTPUT_DIR := dune/generated
CODEGEN := PluginNoLib
ROOTS := $(shell find $(SRC) -name '*.fst' -o -name '*.fsti')
FSTAR_OPTIONS += --with_fstarc
EXTRACT += --extract '-*,+ExtractKuiper'
FSTAR_OPTIONS += --lax --MLish --MLish_effect FStarC.Compiler.Effect

OTHERFLAGS += --already_cached 'Prims,FStarC'

include $(PULSE_ROOT)/mk/boot.mk

.DEFAULT_GOAL := all
all: build

extract: $(ALL_ML_FILES)

build: extract
	$(FSTAR_EXE) --ocamlenv dune build --root=dune
	dune install --root=dune
