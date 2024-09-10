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

# Keep FStar built and pulse plugin updated
HACK:=$(shell $(MAKE) -C FStar 1)
HACK:=$(shell $(MAKE) FSTAR_HOME=$(PWD)/FStar -C karamel minimal)
HACK:=$(shell $(MAKE) FSTAR_HOME=$(PWD)/FStar -C pulse/src build-ocaml)

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
FSTAR_FLAGS += $(OTHERFLAGS)

FSTAR_NOPLUG := $(FSTAR_EXE)				\
	$(SIL)						\
	--include pulse					\
	--include src					\
	$(FSTAR_FLAGS)

FSTAR := $(FSTAR_NOPLUG) --load_cmxs pulse

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
.depend: $(ROOTS)
	$(call msg,"DEPEND")
	$(Q)$(FSTAR) --codegen krml --dep full $(ROOTS) --output_deps_to $@

dep.graph: $(DEPENDRSP)
	$(Q)$(FSTAR) --codegen krml --dep graph $(ROOTS) --output_deps_to $@

dep_filtered.graph: dep.graph
	$(Q) cat $< |					\
	  sed '/.*"fstar.*/d' |			\
	  sed '/.*"pulse.*/d' |			\
	  sed '/.*"prims.*/d' |			\
	  cat > $@

dep_simpl.graph: dep_filtered.graph
	$(Q)$(FSTAR_HOME)/.scripts/simpl_graph.py $< > $@

depgraph.pdf: dep_simpl.graph
	$(call msg, "DOT", $@)
	$(Q)dot -Tpdf -o $@ dep_simpl.graph

SRC_FILE_FOR_CHECKED = $(shell ./scripts/src-file-for-checked.sh $(1))

# FIXME: find a way to invalidate when plugin changes. The added dependency below does
# not do that.
$(OUTDIR)/%.krml: | $(PLUGIN).cmxs
	@# Stupid renaming!
	$(call msg,"EXTRACT",$@)
	$(Q)$(FSTAR) --codegen krml 						\
		--load_cmxs $(PLUGIN)						\
		--extract "-*" 							\
		--extract "$(subst _,.,$(patsubst $(OUTDIR)/%.krml,%,$@))"	\
		--extract "+GPU"						\
		--odir $(shell dirname $@)					\
		--krmloutput $@							\
		--ext extraction_inline_all					\
		$(call SRC_FILE_FOR_CHECKED,$<)

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

%.o: %.cu include/*.h
	$(call msg,"NVCC",$@)
	$(Q)nvcc -o $@ -c $<

$(OUTDIR)/%.exe: $(OUTDIR)/%.o test/Test_%.cu
	$(call msg,"NVCC",$@)
	$(Q)nvcc -I include -I $(OUTDIR) -o $@ $^

$(OUTDIR)/startup.exe: test/startup.cu
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
TESTS+=GPU_AtomicReduce

extraction-targets: \
	out/GPU_DotProduct.o \
	out/GPU_Example1.exe \
	out/GPU_DotProduct2.exe \
	out/GPU_Reduction.cu \
	out/GPU_InnerGhostLem.cu \
	out/GPU_Polymorphism0.cu \
	out/GPU_Polymorphism1.cu \
	out/GPU_AtomicReduce.cu \
	$(patsubst %,out/%.exe,$(TESTS))

.PHONY: test
test: $(patsubst %,$(OUTDIR)/%.test,$(TESTS))
.PHONY: accept
accept: $(patsubst %,$(OUTDIR)/%.accept,$(TESTS))
