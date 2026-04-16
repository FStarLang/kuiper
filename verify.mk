default: all
include .common.mk

all: verify-all
all: extract-all

minimal: verify-minimal
minimal: extract-minimal
minimal: build-minimal

.PHONY: .force
.force:

.configure.output: ./configure $(which nvcc)
	./configure $@

include .configure.output

ifeq (1, $(KUIPER_HAVE_NVCC))
all: build-all
endif

# I HATE MAKE!
.SUFFIXES:
# .SECONDARY:
# ^ Don't ask me why, but SECONDARY makes this makefile very slow on
# no-ops. NOTINTERMEDIATE has a similar effect.
.NOTINTERMEDIATE:
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

KRML_HOME := $(CURDIR)/karamel
FSTAR_EXE := $(CURDIR)/inst/bin/fstar.exe

export FSTAR_EXE
export KRML_HOME

# Hack to print a newline in the $(error ...)
define newline


endef

FStar/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)
karamel/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)
pulse/Makefile:
	$(error $@ not found${newline}Run `git submodule init && git submodule update` if you haven't)

.fstar.src.touch: .force
	[ -f $@ ] || touch $@
	find FStar/ -type f -newer $@ -exec touch $@ \; -quit

.fstar.touch: .fstar.src.touch
	@echo FSTAR
	$(MAKE) -C FStar ADMIT=1
	$(MAKE) -C FStar ADMIT=1 PREFIX=$(CURDIR)/inst install
	@touch .fstar.src.touch # building will change files
	@touch $@

.krml.src.touch: .force
	[ -f $@ ] || touch $@
	find karamel -type f -newer $@ -exec touch $@ \; -quit

.krml.touch: .krml.src.touch karamel/Makefile
	@echo KRML
	@# karamel needs builtin rules which we disable, so clear MAKEFLAGS but still set -j
	$(MAKE) MAKEFLAGS=-j$(shell nproc) -C karamel ADMIT=1 minimal
	@touch .krml.src.touch # building will change files
	@touch $@

.pulse.src.touch: .force
	[ -f $@ ] || touch $@
	find pulse -type f -newer $@ -exec touch $@ \; -quit

.pulse.touch: .fstar.touch .pulse.src.touch pulse/Makefile
	@echo PULSE
	$(MAKE) -C pulse FSTAR_EXE=$(FSTAR_EXE) ADMIT=1 plugin
	@touch .pulse.src.touch # building will change files
	@touch $@

.PHONY: prepare
prepare: .fstar.touch .krml.touch .pulse.touch

AUTOGEN_SCRIPTS := $(shell find src -name '*.fst.sh')
AUTOGEND := $(patsubst %.fst.sh,%.fst,$(AUTOGEN_SCRIPTS))

%.fst: %.fst.sh
	@$(call msg,"GEN",$@)
	./$< > $@

ROOTS := $(shell find src/ -name '*.fst' -o -name '*.fsti')
ROOTS += $(AUTOGEND)

FILTER_OUT = $(foreach v,$(2),$(if $(findstring $(1),$(v)),,$(v)))

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
FSTAR_FLAGS += --warn_error -291 # inspect_ln warnings, benign
FSTAR_FLAGS += --cmi
FSTAR_FLAGS += --warn_error -249-321
FSTAR_FLAGS += --warn_error @242@250 # 242, 250: abort if could not extract something
FSTAR_FLAGS += --z3version 4.13.3
FSTAR_FLAGS += --ext kuiper
FSTAR_FLAGS += --ext optimize_let_vc
FSTAR_FLAGS += --ext __unrefine
FSTAR_FLAGS += --ext context_pruning
FSTAR_FLAGS += --ext no_krml_private
# FSTAR_FLAGS += --ext core_phase2
FSTAR_FLAGS += --warn_error -288 # using has_type (we only use it in SMT patterns)
# FSTAR_FLAGS += --ext krml_inline_all
# FSTAR_FLAGS += --error_contexts true
FSTAR_FLAGS += --ext context_pruning_no_ambients
FSTAR_FLAGS += --ext fly_deps
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

KOTHERFLAGS += $(KO)

KRML_FLAGS :=
KRML_FLAGS += -add-early-include '<kuiper.h>'
KRML_FLAGS += -fc++-compat
KRML_FLAGS += -fcast-allocations
KRML_FLAGS += -skip-compilation
KRML_FLAGS += -skip-makefiles
KRML_FLAGS += -faggressive-inlining
KRML_FLAGS += -fauto-for-loops
KRML_FLAGS += -fnoshort-enums
KRML_FLAGS += -cuda
KRML_FLAGS += -dbacktrace
KRML_FLAGS += $(if $(V),-verbose,-silent)
KRML_FLAGS += -drop Prims
KRML_FLAGS += -minimal
KRML_FLAGS += -header /dev/null
KRML_FLAGS += -warn-error @6 # VLA
KRML_FLAGS += -warn-error -2@4-10@18
KRML_FLAGS += $(KOTHERFLAGS)

KRML := $(KRML_HOME)/krml $(KRML_FLAGS)

# 2: unimplemented function (we trick krml into extracting macros, and we cannot give a prototype)
# 4: type error / malformed input; krml usually skips the decl, we fail hard
# 10: do not warn about -drop being deprecated (though we should use -bundle instead)
# Warning 18: After bundling, two C files are named XXX

# This sandwich is needed so all is the first rule (and not
# something in the include), and verify-all can refer to ALL_CHECKED_FILES,
# which is empty before including .depend. Sigh.
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),echo-fstar)
ifneq ($(MAKECMDGOALS),echo-krml)
ifneq ($(MAKECMDGOALS),.fstar.touch)
ifneq ($(MAKECMDGOALS),.pulse.touch)
include .depend
endif
endif
endif
endif
endif
# verify-all: $(ALL_CHECKED_FILES)
	# ^ This is a bit excessive since it will traverse interfaces and
	# add them too. Instead, I'm using this expression below to turn the
	# $(ROOTS) into .checked. I don't like this since it involves choosing
	# the directory too and that is the job of --dep.

MY_CHECKED_FILES := $(foreach f, $(ROOTS), obj/$(notdir $(f)).checked)
verify-all: $(MY_CHECKED_FILES)

TENSORCORE_CHECKED_FILES := $(foreach f,$(MY_CHECKED_FILES),$(if $(findstring TensorCore,$(f)),$(f)))
MINIMAL_CHECKED_FILES := $(filter-out $(TENSORCORE_CHECKED_FILES),$(MY_CHECKED_FILES))
verify-minimal: $(MINIMAL_CHECKED_FILES)

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
	$(Q)$(FSTAR) --codegen krml --already_cached 'FStar,LowStar,Prims' --dep full $(ROOTS) -o $@.tmp
	# HUGE HACK: append (not prepend!) a .plugin.touch dependency for every krml file.
	sed ':outer; /krml: \\$$/{n;:inner;/[^\\]$$/{s/.*/& .plugin.touch/; b outer};n;b inner}' < $@.tmp > $@
	rm -f $@.tmp

depgraph: depend.pdf
depend.pdf: .depend .force
	$(call msg, "DEPEND GRAPH", $(SRC))
	$(FSTAR) --dep graph --codegen krml --already_cached 'FStar,LowStar,Prims' $(ROOTS) $(DEPFLAGS) -o .depend.graph
	./FStar/.scripts/simpl_graph.py .depend.graph > .depend.simpl
	# Tweak ratio
	sed -i 's/^digraph{/& ratio=1;/' .depend.simpl
	dot -Tpdf -o $@ .depend.simpl
	echo "Wrote $@"

# Does not work. See hack in .depend
# $(OUTDIR)/%.krml: .plugin.touch

$(OUTDIR)/%.krml: MOD=$(subst _,.,$(basename $(notdir $@)))
$(OUTDIR)/%.krml: | .fstar.touch
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
$(OUTDIR)/pre/%.cu $(OUTDIR)/pre/%.h: MOD=$(subst _,.,$(basename $(notdir $<)))
$(OUTDIR)/pre/%.cu $(OUTDIR)/pre/%.h: PRE=$(subst $(OUTDIR),$(OUTDIR)/pre,$@)
$(OUTDIR)/pre/%.cu $(OUTDIR)/pre/%.h: $(OUTDIR)/%.krml .krml.touch
	$(call msg,"KRML")
	# Output into prel/
	$(KRML) -bundle "$(MOD)=*" \
		-tmpdir $(OUTDIR)/pre/ $<

$(OUTDIR)/%.cu: $(OUTDIR)/pre/%.cu scripts/fixup.sed
	# Postprocess via sed and generate the actual target
	sed -f scripts/fixup.sed $< | indent -linux -i4 -nut > $@

$(OUTDIR)/%.h: $(OUTDIR)/pre/%.h scripts/fixup.sed
	# Same. Though no code in here there are still empty lines and whatnot
	sed -f scripts/fixup.sed $< | indent -linux -i4 -nut > $@

include nvcc.mk
