all: build

FSTAR_HOME = ../FStar
FSTAR_EXE = $(FSTAR_HOME)/bin/fstar.exe

OTHERFLAGS += --MLish

# Assume FSTAR_HOME points to the F* source tree
include $(FSTAR_HOME)/.common.mk
include $(FSTAR_HOME)/ulib/gmake/z3.mk    # This pins $(Z3) ...
include $(FSTAR_HOME)/ulib/gmake/fstar.mk # and $(FSTAR) for all sub-make calls
include $(FSTAR_HOME)/src/Makefile.boot.common
include ../pulse/share/pulse/Makefile.locate-fstar

OUTPUT_DIRECTORY = $(CURDIR)/dune/generated
FSTAR_C=$(RUNLIM) $(FSTAR_EXE) $(SIL) $(FSTAR_BOOT_OPTIONS)

EXTRACT_FILES=ExtractPulseGPU.ml

# This refreshes the ML file, from the fst file.
extract: $(addprefix $(OUTPUT_DIRECTORY)/,$(EXTRACT_FILES))

# And then, in a separate invocation, from each .checked.lax we
# extract an .ml file. Must be run MANUALLY.
$(OUTPUT_DIRECTORY)/%.ml: %.fst
	mkdir -p $(OUTPUT_DIRECTORY)
	$(call msg, "EXTRACT", $(notdir $@))
	$(Q)$(BENCHMARK_PRE) $(FSTAR_C) $< \
		   --odir "$(OUTPUT_DIRECTORY)" \
		   --codegen OCaml \
		   --extract_module $(basename $(notdir $<))
	chmod -x $@

# This builds the plugin from the ML. This is run automatically.
build:
	cd dune/ && dune build
	cd dune/ && dune install
