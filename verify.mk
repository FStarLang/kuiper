include .common.mk

.PHONY: .force
.force:

# I HATE MAKE!
.SUFFIXES:
.SECONDARY:
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

KRML_HOME := $(CURDIR)/karamel
FSTAR_EXE := $(CURDIR)/inst/bin/fstar.exe

export FSTAR_EXE
export KRML_HOME

# Hack to print a newline in the $(error ...)
define newline


endef

.fstar.touch: $(shell find FStar/src FStar/ulib -type f) FStar/Makefile
	@echo FSTAR
	$(MAKE) -C FStar ADMIT=1
	$(MAKE) -C FStar ADMIT=1 PREFIX=$(CURDIR)/inst install
	@touch $@

FStar/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)

.krml.touch: $(shell find karamel/ -type f)
	@echo KRML
	@# karamel needs builtin rules which we disable, so clear MAKEFLAGS but still set -j
	@# is minimal enough?
	$(MAKE) MAKEFLAGS=-j$(shell nproc) -C karamel ADMIT=1 minimal
	@touch $@

karamel/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)

.pulse.touch: .fstar.touch $(shell find pulse/ -type f) pulse/Makefile
	@echo PULSE
	$(MAKE) -C pulse FSTAR_EXE=$(FSTAR_EXE) ADMIT=1 plugin
	@touch $@

pulse/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)

.PHONY: prepare
prepare: .fstar.touch .krml.touch .pulse.touch

ROOTS := $(shell find src/ -name '*.fst' -o -name '*.fsti')

FILTER_OUT = $(foreach v,$(2),$(if $(findstring $(1),$(v)),,$(v)))

ROOTS := $(call FILTER_OUT,MatMulOpt,$(ROOTS))

CACHEDIR := obj
OUTDIR   := obj

ifneq ($(D),)
FSTAR_DEBUG := --debug $D
endif
ifneq ($(ADMIT),)
OTHERFLAGS += --admit_smt_queries true
endif
OTHERFLAGS += $O

FSTAR_FLAGS += --cache_dir $(CACHEDIR)
FSTAR_FLAGS += --odir $(OUTDIR)
FSTAR_FLAGS += --cmi
FSTAR_FLAGS += --warn_error -249-321
FSTAR_FLAGS += --warn_error @242@250 # 242, 250: abort if could not extract something
FSTAR_FLAGS += --z3version 4.13.3
FSTAR_FLAGS += --ext kuiper
FSTAR_FLAGS += --ext __unrefine
FSTAR_FLAGS += --ext context_pruning
FSTAR_FLAGS += --ext no_krml_private
FSTAR_FLAGS += --ext krml_inline_all
FSTAR_FLAGS += $(OTHERFLAGS)
FSTAR_FLAGS += $(FSTAR_DEBUG)

FSTAR = $(FSTAR_EXE)					\
	$(SIL)						\
	--include pulse/build/ocaml/installed/lib/pulse	\
	--include pulse/lib/common			\
	--include pulse/lib/pulse 			\
	--include src					\
	$(FSTAR_FLAGS)

GPUH := $(realpath include/kuiper.h)

KOTHERFLAGS += $(KO)

KRML := $(KRML_HOME)/krml				\
	-add-early-include '<kuiper.h>'			\
	-fc++-compat					\
	-fcast-allocations				\
	-skip-compilation				\
	-skip-makefiles					\
	-cuda						\
	$(if $(V),-verbose,-silent)			\
	-drop Prims					\
	-minimal					\
	-header /dev/null				\
	-warn-error -2@4-10@18				\
	$(KOTHERFLAGS)

# 2: unimplemented function (we trick krml into extracting macros, and we cannot give a prototype)
# 4: type error / malformed input; krml usually skips the decl, we fail hard
# 10: do not warn about -drop being deprecated (though we should use -bundle instead)
# Warning 18: After bundling, two C files are named XXX

# This sandwich is needed so all is the first rule (and not
# something in the include), and verify-all can refer to ALL_CHECKED_FILES,
# which is empty before including .depend. Sigh.
all: verify-all extraction-targets
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),echo-fstar)
ifneq ($(MAKECMDGOALS),echo-krml)
include .depend
endif
endif
endif
# verify-all: $(ALL_CHECKED_FILES)
	# ^ This is a bit excessive since it will traverse interfaces and
	# add them too. Instead, I'm using this expression below to turn the
	# $(ROOTS) into .checked. I don't like this since it involves choosing
	# the directory too and that is the job of --dep.
verify-all: $(foreach f, $(ROOTS), obj/$(notdir $(f)).checked)

$(CACHEDIR)/%.checked: | .fstar.touch .pulse.touch
	@$(call msg,"CHECK")
	$(Q)$(FSTAR) $(if $(findstring pulse/,$<),--admit_smt_queries true,) --already_cached '*' -c $< -o $@
	@touch -c $@

# Without .cmxs extension
PLUGIN=extraction/dune/_build/default/kuiper_extr

.plugin.touch: .fstar.touch $(shell find extraction -type f)
	+$(MAKE) -C extraction build
	touch $@

.PHONY: echo-fstar
echo-fstar:
	@echo $(FSTAR)

.PHONY: echo-krml
echo-krml:
	@echo $(KRML)

# NB: The dependency analysis needs to parse the files, so it needs
# the Pulse plugin
.depend: $(ROOTS) .fstar.touch .pulse.touch
	$(call msg,"DEPEND",$@)
	$(Q)$(FSTAR) --codegen krml --already_cached 'FStar,LowStar,Prims' --dep full $(ROOTS) -o $@

$(OUTDIR)/%.krml: MOD=$(subst _,.,$(basename $(notdir $@)))
$(OUTDIR)/%.krml: | .fstar.touch .plugin.touch
	@# Stupid renaming!
	$(call msg,"EXTRACT")
	$(Q)$(FSTAR) --codegen krml 						\
		--load_cmxs $(PLUGIN)						\
		--extract "-*" 							\
		--extract "$(MOD)"						\
		--extract "+Kuiper"						\
		-o $@								\
		$<

# Turning something like obj/Kuiper_DotProduct2.krml into Kuiper.DotProduct2
$(OUTDIR)/%.cu: MOD=$(subst _,.,$(basename $(notdir $<)))
$(OUTDIR)/%.cu: $(OUTDIR)/%.krml .krml.touch
	$(call msg,"KRML")
	$(KRML) -bundle "$(MOD)=*" \
		-tmpdir $(OUTDIR) $<

NVCC_FLAGS += -O3
NVCC_FLAGS += -I include
NVCC_FLAGS += -I obj # needed for files in test/ only..

%.o: %.cu include/*.h
	$(call msg,"NVCC")
	$(Q)nvcc $(NVCC_FLAGS) -o $@ -c $<

$(OUTDIR)/%.exe: $(OUTDIR)/%.o test/Test_%.cu
	$(call msg,"NVLD")
	$(Q)nvcc $(NVCC_FLAGS) $(NVLD_CFLAGS) -o $@ $^

$(OUTDIR)/startup.exe: test/startup.cu
	$(call msg,"NVCC")
	$(Q)nvcc $(NVCC_FLAGS) $(NVLD_FLAGS) -o $@ $^

$(OUTDIR)/%.output: $(OUTDIR)/%.exe
	$< > $@

test/%.output.expected:
	$(error You need to create the '$@' file)

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
TESTS+=Kuiper_MatMul_U64
# TESTS+=Kuiper_MatMulTile
# TESTS+=Kuiper_MatMulTileF32
# TESTS+=Kuiper_MatMulTile_Async
TESTS+=Kuiper_BasicFloat
TESTS+=Kuiper_AtomicReduce_U64
TESTS+=Kuiper_HReduceU32Plus
TESTS+=Kuiper_HReduceU64Plus
TESTS+=Kuiper_HReduceF32Plus
TESTS+=Kuiper_HReduceF64Plus
TESTS+=Kuiper_ArrayReversal
TESTS+=Kuiper_Async1
TESTS+=Kuiper_Softmax_F32
TESTS+=Kuiper_Softmax_F64

extraction-targets: \
	obj/Kuiper_Example1.exe \
	obj/Kuiper_DotProduct.o \
	obj/Kuiper_DotProduct.exe \
	$(subst _cu,.cu,$(subst .,_,$(patsubst src/examples/%.fst,obj/%.cu,$(wildcard src/examples/*.fst)))) \
	$(patsubst %,obj/%.exe,$(TESTS))
# ^ nasty
	# obj/Kuiper_MatMulTileF32_Async.cu \

.PHONY: test
test: $(patsubst %,$(OUTDIR)/%.test,$(TESTS))
.PHONY: accept
accept: $(patsubst %,$(OUTDIR)/%.accept,$(TESTS))
