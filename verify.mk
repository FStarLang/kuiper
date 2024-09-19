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

# We DO NOT read FSTAR_HOME externally.
FSTAR_HOME=$(PWD)/FStar
KRML_HOME=$(PWD)/karamel
FSTAR_EXE := $(FSTAR_HOME)/bin/fstar.exe

export FSTAR_HOME
export KRML_HOME

# Hack to print a newline in the $(error ...)
define newline


endef

.b_fstar: $(shell find FStar/ocaml/ -type f) FStar/Makefile
	@echo FSTAR
	$(MAKE) -C FStar 1
	@touch $@

FStar/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)

.b_karamel: $(shell find karamel/ -type f) karamel/Makefile
	@echo KRML
	$(MAKE) FSTAR_HOME=$(PWD)/FStar -C karamel minimal
	@touch $@

karamel/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)

.b_pulse: .b_fstar $(shell find pulse/ -type f) pulse/Makefile
	@echo PULSE
	$(MAKE) FSTAR_HOME=$(PWD)/FStar -C pulse/src build-ocaml
	@touch $@

pulse/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)

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
FSTAR_FLAGS += --cmi
FSTAR_FLAGS += --warn_error -249-321
FSTAR_FLAGS += --warn_error @242@250 # 242, 250: abort if could not extract something
FSTAR_FLAGS += --ext __unrefine
FSTAR_FLAGS += --ext context_pruning
FSTAR_FLAGS += --ext no_krml_private
FSTAR_FLAGS += --ext krml_inline_all
FSTAR_FLAGS += $(OTHERFLAGS)

FSTAR_NOPLUG = $(FSTAR_EXE)				\
	$(SIL)						\
	--include pulse					\
	--include src					\
	$(FSTAR_FLAGS)

FSTAR = $(FSTAR_NOPLUG) --load_cmxs pulse

GPUH := $(realpath include/kuiper.h)

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
%.checked: | .b_fstar .b_pulse
	@$(call msg,"CHECK")
	$(Q)$(FSTAR) --already_cached '*' $<
	@touch -c $@

# What the hell is going on!? This verifies fine locally from a clean
# build, but not on CI. Does the SMT encoding depend in any way on the machine,
# like filepaths? Anyway, bump the rlimit.
# -.cache/FStar.UInt.fst.checked: FSTAR_FLAGS+=--z3rlimit_factor 2
# That didn't help. To help with it, we do not verify anything from F*.
# I would add Pulse but we have some modules in the Pulse namespace here,
# rename themwould add Pulse but we have some modules in the Pulse namespace here.
.cache/FStar.%.checked: FSTAR_FLAGS+=--admit_smt_queries true

$(PLUGIN).cmxs: $(FSTAR_EXE)
	+$(MAKE) -C extraction build

.PHONY: echo-fstar
echo-fstar:
	@echo $(FSTAR)

.PHONY: echo-krml
echo-krml:
	@echo $(KRML)

# NB: The dependency analysis needs to parse the files, so it needs
# the Pulse plugin
.depend: $(ROOTS) .b_fstar .b_pulse
	$(call msg,"DEPEND",$@)
	$(Q)$(FSTAR) --codegen krml --dep full $(ROOTS) --output_deps_to $@

SRC_FILE_FOR_CHECKED = $(shell ./scripts/src-file-for-checked.sh $(1))

# FIXME: find a way to invalidate when plugin changes. The added dependency below does
# not do that.
$(OUTDIR)/%.krml: | $(PLUGIN).cmxs
	@# Stupid renaming!
	$(call msg,"EXTRACT")
	$(Q)$(FSTAR) --codegen krml 						\
		--load_cmxs $(PLUGIN)						\
		--extract "-*" 							\
		--extract "$(subst _,.,$(patsubst $(OUTDIR)/%.krml,%,$@))"	\
		--extract "+Kuiper"						\
		--odir $(shell dirname $@)					\
		--krmloutput $@							\
		$(call SRC_FILE_FOR_CHECKED,$<)

$(OUTDIR)/%.c: $(OUTDIR)/%.krml .b_karamel
	$(call msg,"KRML")
	@# Awful substitution here to get the module name, turning something like
	@# out/Kuiper_DotProduct2.krml into Kuiper.DotProduct2
	$(Q)MOD=$$(echo $< | sed 's,.*/,,' | sed 's/.krml$$//' | sed 's/_/./g') && \
	$(KRML) \
		-bundle "$${MOD}=*" \
		-tmpdir $(OUTDIR) $<

$(OUTDIR)/%.cu: $(OUTDIR)/%.c
	@ln -sf $(realpath $<) $@

%.o: %.cu include/*.h
	$(call msg,"NVCC")
	$(Q)nvcc $(CFLAGS) -O3 -o $@ -c $<

$(OUTDIR)/%.exe: $(OUTDIR)/%.o test/Test_%.cu
	$(call msg,"NVCC")
	$(Q)nvcc $(CFLAGS) -O3 -I include -I $(OUTDIR) -o $@ $^

$(OUTDIR)/startup.exe: test/startup.cu
	$(call msg,"NVCC")
	$(Q)nvcc -I include -I $(OUTDIR) -o $@ $^

$(OUTDIR)/%.output: $(OUTDIR)/%.exe
	$< > $@

$(OUTDIR)/%.test: test/%.output.expected $(OUTDIR)/%.output
	$(Q)diff -u $^
	$(call msg,"TEST OK")
	@touch $@

$(OUTDIR)/%.accept: $(OUTDIR)/%.output
	$(call msg,"ACCEPT")
	$(Q)cp $< $(patsubst $(OUTDIR)/%,test/%,$<).expected

TESTS+=Kuiper_Example1
TESTS+=Kuiper_DotProduct2
TESTS+=Kuiper_DotProduct3
TESTS+=Kuiper_MatMul
TESTS+=Kuiper_MatMulTile
TESTS+=Kuiper_MatMulTileF32
TESTS+=Kuiper_MatMulTile_Async
TESTS+=Kuiper_BasicFloat
TESTS+=Kuiper_AtomicReduce
TESTS+=Kuiper_HReduceU32Plus
TESTS+=Kuiper_HReduceU64Plus
TESTS+=Kuiper_HReduceF32Plus
TESTS+=Kuiper_HReduceF64Plus
TESTS+=Kuiper_ArrayReversal
TESTS+=Kuiper_Async1

extraction-targets: \
	out/Kuiper_DotProduct.o \
	out/Kuiper_Example1.exe \
	out/Kuiper_DotProduct2.exe \
	out/Kuiper_DotProduct3.exe \
	out/Kuiper_Reduction.cu \
	out/Kuiper_InnerGhostLem.cu \
	out/Kuiper_Polymorphism0.cu \
	out/Kuiper_Polymorphism1.cu \
	out/Kuiper_AtomicReduce.cu \
	$(patsubst %,out/%.exe,$(TESTS))

.PHONY: test
test: $(patsubst %,$(OUTDIR)/%.test,$(TESTS))
.PHONY: accept
accept: $(patsubst %,$(OUTDIR)/%.accept,$(TESTS))
