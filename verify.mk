include .common.mk

# No default rules
.SUFFIXES:

ROOTS := $(shell find src/ -name '*.fst' -o -name '*.fsti')

CACHEDIR := .cache
OUTDIR   := .out

# Without .cmxs extension
PLUGIN=extraction/dune/_build/default/gpuextr

FSTAR_FLAGS += --cache_checked_modules
FSTAR_FLAGS += --cache_dir $(CACHEDIR)
FSTAR_FLAGS += --odir $(OUTDIR)
FSTAR_FLAGS += $(OTHERFLAGS)

FSTAR := fstar.exe					\
	$(SIL)						\
	--include pulse/lib/pulse/			\
	--include pulse/lib/pulse/core/			\
	--include pulse/lib/pulse/lib/			\
	--include pulse/lib/pulse/lib/class/		\
	--include src/lib/				\
	--include src/examples/				\
	--include src/examples/matmul/			\
	--load_cmxs pulse				\
	--load_cmxs $(PLUGIN)				\
	--warn_error -249-321				\
	$(FSTAR_FLAGS)
	
GPUH := $(realpath GPU.h)

KRML := $(KRML_HOME)/krml				\
	-add-early-include '"$(GPUH)"'			\
	-fc++-compat					\
	-fcast-allocations				\
	-verbose					\
	-skip-compilation				\
	-warn-error -2@4

# This sandwich is needed so all is the first rule (and not
# something in the include), and verify-all can refer to ALL_CHECKED_FILES,
# which is empty before including .depend. Sigh.
all: verify-all extraction-targets
include .depend
# verify-all: $(ALL_CHECKED_FILES)
	# ^ This is a bit excessive since it will traverse interfaces and
	# add them too. Instead, I'm using this expression below to turn the
	# $(ROOTS) into .checked. I don't like this since it involves choosing
	# the directory too and that is the job of --dep.
verify-all: $(foreach f, $(ROOTS), .cache/$(notdir $(f)).checked)

# Dependencies come from .depend. We still need this rule.
%.checked: | $(PLUGIN).cmxs
	@$(call msg, "CHECK", $(notdir $@))
	$(Q)$(FSTAR) $<
	@touch -c $@

$(PLUGIN).cmxs:
	+$(MAKE) -C extraction build

.PHONY: echo-fstar
echo-fstar:
	@echo $(FSTAR)

.PHONY: echo-krml
echo-krml:
	@echo $(KRML)

.depend: $(ROOTS)
	$(call msg, "DEPEND")
	$(Q)$(FSTAR) --dep full $(ROOTS) --output_deps_to $@

# Awful special casing
out/GPU.DotProduct/GPU_DotProduct.cu: .cache/GPU.DotProduct.fst.checked
	./extract_cu.sh GPU.DotProduct
out/GPU.Example1/GPU_Example1.cu: .cache/GPU.Example1.fst.checked
	./extract_cu.sh GPU.Example1
out/GPU.DotProduct2/GPU_DotProduct2.cu: .cache/GPU.DotProduct2.fst.checked
	./extract_cu.sh GPU.DotProduct2

%.o: %.cu
	nvcc -o $@ -c $< -I $(KRML_HOME)/include/ -I $(KRML_HOME)/krmllib/dist/minimal/

extraction-targets: \
	out/GPU.DotProduct/GPU_DotProduct.o \
	out/GPU.Example1/GPU_Example1.o \
	out/GPU.DotProduct2/GPU_DotProduct2.o \
