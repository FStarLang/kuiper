include .common.mk

# I HATE MAKE!
.SUFFIXES:
.PRECIOUS: out/%.c
.PRECIOUS: out/%.cu
.PRECIOUS: out/%.o
.PRECIOUS: out/%.output
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

ROOTS := $(shell find src/ -name '*.fst' -o -name '*.fsti')

CACHEDIR := .cache
OUTDIR   := out

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
	--warn_error -249-321				\
	$(FSTAR_FLAGS)
	
GPUH := $(realpath GPU.h)

KRML := $(KRML_HOME)/krml				\
	-add-early-include '"$(GPUH)"'			\
	-fc++-compat					\
	-fcast-allocations				\
	-skip-compilation				\
	-skip-makefiles					\
	$(if $(V), -verbose,-silent)			\
	-minimal					\
	-drop Prims					\
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
%.checked:
	@$(call msg, "CHECK",$(notdir $@))
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
	$(call msg,"DEPEND")
	$(Q)$(FSTAR) --dep full $(ROOTS) --output_deps_to $@

# Invalidate when plugin changes
$(OUTDIR)/%.krml: | $(PLUGIN).cmxs
	@# Stupid renaming!
	$(call msg,"EXTRACT",$@)
	$(Q)$(FSTAR) --codegen krml 						\
		--load_cmxs $(PLUGIN)						\
		--extract "-*" 							\
		--extract "$(subst _,.,$(patsubst $(OUTDIR)/%.krml,%,$@))"	\
		--odir $(shell dirname $@)					\
		$(patsubst .cache/%.checked,src/examples/%,$<)

$(OUTDIR)/%.c: $(OUTDIR)/%.krml
	$(call msg,"KRML",$@)
	$(Q)$(KRML) -tmpdir $(OUTDIR) $<

$(OUTDIR)/%.cu: $(OUTDIR)/%.c
	@ln -s $(realpath $<) $@

%.o: %.cu GPU.h
	$(call msg,"NVCC",$@)
	$(Q)nvcc -o $@ -c $<

$(OUTDIR)/%.exe: $(OUTDIR)/%.o test/Test_%.cu
	$(call msg,"NVCC",$@)
	$(Q)nvcc -I $(OUTDIR) -o $@ $^

$(OUTDIR)/%.output: $(OUTDIR)/%.exe
	$< > $@

$(OUTDIR)/%.test: test/%.output.expected $(OUTDIR)/%.output
	$(Q)diff -u $^
	$(call msg,"TEST OK",$@)
	@touch $@

$(OUTDIR)/%.accept: $(OUTDIR)/%.output
	$(call msg,"ACCEPT",$@)
	$(Q)cp $< $(patsubst $(OUTDIR)/%,test/%,$<).expected

extraction-targets: \
	out/GPU_DotProduct.o \
	out/GPU_Example1.exe \
	out/GPU_DotProduct2.exe \

TESTS+=GPU_Example1
TESTS+=GPU_DotProduct2

.PHONY: test
test: $(patsubst %,$(OUTDIR)/%.test,$(TESTS))
.PHONY: accept
accept: $(patsubst %,$(OUTDIR)/%.accept,$(TESTS))
