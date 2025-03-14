default: all
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
# FSTAR_FLAGS += --error_contexts true
FSTAR_FLAGS += $(OTHERFLAGS)
FSTAR_FLAGS += $(FSTAR_DEBUG)

# abspath is important so the fstar.sh script can be run from anywhere
FSTAR = $(FSTAR_EXE)							\
	$(SIL)								\
	--include $(abspath pulse/build/ocaml/installed/lib/pulse)	\
	--include $(abspath pulse/lib/common)				\
	--include $(abspath pulse/lib/pulse)				\
	--include $(abspath src)					\
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

# Ignore some warnings from the Pulse library, it's out of scope for us.
# Also admit queries, we just want a quick build and it's supposed to be
# checked green by Pulse.
PULSE_LIB_FLAGS := --admit_smt_queries true --warn_error -288

$(CACHEDIR)/%.checked: | .fstar.touch .pulse.touch
	@$(call msg,"CHECK")
	$(Q)$(FSTAR) $(if $(findstring pulse/,$<),$(PULSE_LIB_FLAGS)) --already_cached '*' -c $< -o $@
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


depgraph: depend.pdf
depend.pdf: .depend .force
	$(call msg, "DEPEND GRAPH", $(SRC))
	$(FSTAR) --dep graph --codegen krml --already_cached 'FStar,LowStar,Prims' $(ROOTS) $(EXTRACT) $(DEPFLAGS) -o .depend.graph
	./FStar/.scripts/simpl_graph.py .depend.graph > .depend.simpl
	dot -Tpdf -o $@ .depend.simpl
	echo "Wrote $@"

$(OUTDIR)/%.krml: .plugin.touch
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
# .plugin.touch doesn't feel right to me (it's on the krml targets)
# but it triggers rebuilds when the plugin changes, which is good.
$(OUTDIR)/%.cu: $(OUTDIR)/%.krml .krml.touch .plugin.touch
	$(call msg,"KRML")
	$(KRML) -bundle "$(MOD)=*" \
		-tmpdir $(OUTDIR) $<

NVCC_FLAGS += -O3
NVCC_FLAGS += -I include
NVCC_FLAGS += -I obj # needed for files in test/ only..

%.o: %.cu include/*.h
	$(call msg,"NVCC")
	$(Q)nvcc $(NVCC_FLAGS) -o $@ -c $<

remove__ = $(firstword $(subst __, ,$(patsubst Test_%,%,$1)))

# argh
.SECONDEXPANSION:
$(OUTDIR)/%.exe: $(OUTDIR)/$$(call remove__, %).o test/%.cu
	$(call msg,"NVLD")
	$(Q)nvcc $(NVCC_FLAGS) $(NVLD_CFLAGS) -o $@ $^

$(OUTDIR)/%.output: $(OUTDIR)/%.exe
	$< > $@

test/%.output.expected:
	$(error You need to create the '$@' file)

$(OUTDIR)/%.test: $(OUTDIR)/%.output test/%.output.expected
	./scripts/diff.sh -u $^
	$(call msg,"TEST OK")
	@touch $@

$(OUTDIR)/%.accept: $(OUTDIR)/%.output
	$(call msg,"ACCEPT")
	$(Q)cp $< $(patsubst $(OUTDIR)/%,test/%,$<).expected

TESTS+=$(notdir $(basename $(wildcard test/*.cu)))
TESTS:=$(filter-out Kuiper_Softmax_F16, $(TESTS))
# Disable softmax 16. It works fine locally (outside of docker)
# but fails within in with undefined __hdiv. The nvcc there is slightly
# older. Suprisignly using / just works, but that fails for other
# operators. Forget it for now, but we should be principled about using
# the correct feature flags or whatever.

# matmultile is WIP
TESTS:=$(filter-out Test_Kuiper_MatMulTile_Async, $(TESTS))
TESTS:=$(filter-out Test_Kuiper_MatMulTile, $(TESTS))
TESTS:=$(filter-out Test_Kuiper_MatMulTileF32, $(TESTS))

# restore using poly impl
TESTS:=$(filter-out Test_Kuiper_DotProduct, $(TESTS))
TESTS:=$(filter-out Test_Kuiper_DotProduct2, $(TESTS))
TESTS:=$(filter-out Test_Kuiper_DotProduct3, $(TESTS))

extraction-targets: obj/Kuiper_ArrayView_Test1.cu
extraction-targets: obj/Kuiper_Example1.cu
extraction-targets: $(subst _cu,.cu,$(subst .,_,$(patsubst src/examples/%.fst,obj/%.cu,$(wildcard src/examples/*.fst))))
extraction-targets: $(subst _cu,.cu,$(subst .,_,$(patsubst src/lib/inst/%.fst,obj/%.cu,$(wildcard src/lib/inst/*.fst))))
extraction-targets: $(patsubst %,obj/%.exe,$(TESTS))

# ^ nasty
# obj/Kuiper_MatMulTileF32_Async.cu

.PHONY: test
test: $(patsubst %,$(OUTDIR)/%.test,$(TESTS))
.PHONY: accept
accept: $(patsubst %,$(OUTDIR)/%.accept,$(TESTS))
