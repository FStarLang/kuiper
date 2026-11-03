# This makefile is not great and will repeat some work by checking ulib modules.

FSTAR_EXE ?= ../inst/bin/fstar.exe
FSTAR_ROOT ?= ../FStar
PULSE_ROOT ?= ../FStar/pulse
# ^ should not be required

TAG := extraction
SRC := .
CACHE_DIR := _cache
OUTPUT_DIR := dune/generated
ROOTS := $(shell find $(SRC) -name '*.fst' -o -name '*.fsti')
FSTAR_OPTIONS += --with_fstarc
FSTAR_OPTIONS += --lax
FSTAR_OPTIONS += --warn_error -328 # spurious warn about let rec
FSTAR_OPTIONS += --warn_error -272 # top-level effects

OTHERFLAGS += --already_cached 'Prims,FStarC'

# Extracted by a single Custard pass into one link unit, against the
# compiler's own (as F*'s pulse/mk/plugin.mk does). ExtractKuiper registers
# its rules in top-level effects, so it is an entry.
CUSTARD_UNIT    := KuiperExtr
CUSTARD_ROOT    := ExtractKuiper.fst
CUSTARD_ENTRIES := --custard_entry ExtractKuiper
# NB: deferred (=), FSTARC_CUI is defined by boot.mk.
CUSTARD_LINK     = $(FSTARC_CUI)

include $(PULSE_ROOT)/mk/boot.mk

.DEFAULT_GOAL := all
all: build

extract: custard

build: extract
	$(FSTAR_EXE) --ocamlenv dune build --root=dune
	dune install --root=dune
