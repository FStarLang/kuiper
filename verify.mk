include .common.mk

# I HATE MAKE!
.SUFFIXES:
.PRECIOUS: out/%.c
.PRECIOUS: out/%.cu
.PRECIOUS: out/%.o
.PRECIOUS: out/%.output
.PRECIOUS: out/%.exe
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

ROOTS := $(shell find src/ -name '*.fst' -o -name '*.fsti')

FILTER_OUT = $(foreach v,$(2),$(if $(findstring $(1),$(v)),,$(v)))

ROOTS := $(call FILTER_OUT,MatMulOpt,$(ROOTS))

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
	--include src/examples/matmul-opt/			\
	--load_cmxs pulse				\
	--warn_error -249-321				\
	$(FSTAR_FLAGS)
	
GPUH := $(realpath include/GPU.h)

KRML := $(KRML_HOME)/krml				\
	-add-early-include '"$(GPUH)"'			\
	-fc++-compat					\
	-fcast-allocations				\
	-skip-compilation				\
	-skip-makefiles					\
	$(if $(V), -verbose,-silent)			\
	-drop Prims					\
	-minimal					\
	-warn-error -2@4-10@18

# 2: unimplemented function (we trick krml into extracting macros, and we cannot give a prototype)
# 4: type error / malformed input; krml usually skips the decl, we fail hard
# 10: do not warn about -drop being deprecated (though we should use -bundle instead)
# Warning 18: After bundling, two C files are named XXX

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
		--krmloutput $@							\
		$(patsubst .cache/%.checked,src/examples/%,$<)

$(OUTDIR)/%.c: $(OUTDIR)/%.krml
	$(call msg,"KRML",$@)
	@# Awful substitution here to get the module name, turning something like
	@# out/GPU_DotProduct2.krml into GPU.DotProduct2
	$(Q)MOD=$$(echo $< | sed 's,.*/,,' | sed 's/.krml$$//' | sed 's/_/./g') && \
	$(KRML) \
		-bundle "$${MOD}=$${MOD}.*" \
		-tmpdir $(OUTDIR) $<

$(OUTDIR)/%.cu: $(OUTDIR)/%.c
	@ln -sf $(realpath $<) $@

%.o: %.cu include/GPU.h
	$(call msg,"NVCC",$@)
	$(Q)nvcc -o $@ -c $<

$(OUTDIR)/%.exe: $(OUTDIR)/%.o test/Test_%.cu
	$(call msg,"NVCC",$@)
	$(Q)nvcc -I include -I $(OUTDIR) -o $@ $^

$(OUTDIR)/%.output: $(OUTDIR)/%.exe
	$< > $@

$(OUTDIR)/%.test: test/%.output.expected $(OUTDIR)/%.output
	$(Q)diff -u $^
	$(call msg,"TEST OK",$@)
	@touch $@

$(OUTDIR)/%.accept: $(OUTDIR)/%.output
	$(call msg,"ACCEPT",$@)
	$(Q)cp $< $(patsubst $(OUTDIR)/%,test/%,$<).expected

TESTS+=GPU_Example1
TESTS+=GPU_DotProduct2
TESTS+=GPU_MatMul
TESTS+=GPU_BasicFloat

extraction-targets: \
	out/GPU_DotProduct.o \
	out/GPU_Example1.exe \
	out/GPU_DotProduct2.exe \
	$(patsubst %,out/%.exe,$(TESTS))

.PHONY: test
test: $(patsubst %,$(OUTDIR)/%.test,$(TESTS))
.PHONY: accept
accept: $(patsubst %,$(OUTDIR)/%.accept,$(TESTS))
