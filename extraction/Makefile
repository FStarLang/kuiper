# This makefile is not great and will repeat some work by checking ulib modules.

FSTAR_EXE ?= ../inst/bin/fstar.exe
FSTAR_ROOT ?= ../FStar
PULSE_ROOT ?= ../FStar/pulse
# ^ should not be required

TAG := extraction
SRC := .
CACHE_DIR := _cache
OUTPUT_DIR := dune/generated
CODEGEN := PluginNoLib
ROOTS := $(shell find $(SRC) -name '*.fst' -o -name '*.fsti')
FSTAR_OPTIONS += --with_fstarc
EXTRACT += --extract '-*,+ExtractKuiper,+ExtractionUtils'
FSTAR_OPTIONS += --lax
FSTAR_OPTIONS += --warn_error -328 # spurious warn about let rec
FSTAR_OPTIONS += --warn_error -272 # top-level effects

OTHERFLAGS += --already_cached 'Prims,FStarC'

include $(PULSE_ROOT)/mk/boot.mk

.DEFAULT_GOAL := all
all: build

extract: $(ALL_ML_FILES)

build: extract
	$(FSTAR_EXE) --ocamlenv dune build --root=dune
	dune install --root=dune
