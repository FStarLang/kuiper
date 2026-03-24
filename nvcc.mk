# Must be included after .common.mk, .configure.mk, and defininig OUTDIR

NVCC_FLAGS += -O3
NVCC_FLAGS += -I include
NVCC_FLAGS += -I obj # needed for files in test/ only..
NVCC_FLAGS += -arch=native
NVCC_FLAGS += -DKUIPER_CFG_TENSORCORES=$(KUIPER_CFG_TENSORCORES)

%.o: %.cu %.h include/*.h
	$(call msg,"NVCC")
	$(Q)nvcc $(NVCC_FLAGS) -o $@ -c $<

remove__ = $(firstword $(subst __, ,$(patsubst Test_%,%,$1)))

.SECONDEXPANSION:
$(OUTDIR)/Test_%.o: test/Test_%.cu test/test-common.h test/*.c.inc include/*.h $(OUTDIR)/$$(call remove__, Test_%).h
	$(call msg,"NVCC")
	$(Q)nvcc $(NVCC_FLAGS) -o $@ -c $<

# argh
.SECONDEXPANSION:
$(OUTDIR)/%.exe: $(OUTDIR)/%.o $(OUTDIR)/$$(call remove__, %).o
	$(call msg,"NVLD")
	$(Q)nvcc $(NVCC_FLAGS) $(NVLD_CFLAGS) -o $@ $^

.PHONY: nvidia-smi-check
nvidia-smi-check:
	nvidia-smi >/dev/null || (echo "*** nvidia-smi failed! Is CUDA set up properly?\n" >&2; false)

$(OUTDIR)/%.output: $(OUTDIR)/%.exe nvidia-smi-check
	timeout -k 1 180 $< > $@

test/%.output.expected:
	$(error You need to create the '$@' file)

$(OUTDIR)/%.test: $(OUTDIR)/%.output test/%.output.expected
	./scripts/diff.sh $^
	$(call msg,"TEST OK")
	@touch $@

$(OUTDIR)/%.accept: $(OUTDIR)/%.output
	$(call msg,"ACCEPT")
	$(Q)cp $< $(patsubst $(OUTDIR)/%,test/%,$<).expected

TESTS+=$(notdir $(basename $(wildcard test/Test_*.cu)))

NOTEST += Test_Kuiper_Softmax__F16
ifeq ($(KUIPER_CFG_TENSORCORES),0)
NOTEST += $(foreach f,$(TESTS),$(if $(findstring TensorCore,$(f)),$(f)))
endif

# Disable softmax 16. It works fine locally (outside of docker)
# but fails within in with undefined __hdiv. The nvcc there is slightly
# older. Suprisignly using / just works, but that fails for other
# operators. Forget it for now, but we should be principled about using
# the correct feature flags or whatever.

TESTS := $(filter-out $(NOTEST), $(TESTS))

EXTRACT :=
EXTRACT_MINIMAL :=

# Extract everything in src/examples
EXTRACT += $(wildcard src/examples/*.fst)
# Extract everything in src/lib/inst, they are the C api for the library
EXTRACT += $(wildcard src/lib/inst/*.fst)
# And src/lib/inst/gemm...
EXTRACT += $(wildcard src/lib/inst/gemm/*.fst)
EXTRACT += src/lib/graph/Kuiper.GraphDist.fst
EXTRACT += src/examples/Kuiper.Example2.fst

NOEXTRACT :=
NOEXTRACT += src/examples/Kuiper.Sparse.SPMM.fst

# The Inst.fst modules just contain an instantiation function, not to be extracted.
INST_MODULES := $(foreach f,$(EXTRACT),$(if $(findstring Inst.fst,$(f)),$(f)))
EXTRACT := $(filter-out $(INST_MODULES),$(EXTRACT))

EXTRACT := $(filter-out $(NOEXTRACT),$(EXTRACT))

extract-all: $(patsubst %,obj/%.cu,$(subst .,_,$(basename $(notdir $(EXTRACT)))))
extract-all: $(patsubst %,obj/%.h, $(subst .,_,$(basename $(notdir $(EXTRACT)))))

EXTRACT_MINIMAL := $(EXTRACT)
EXTRACT_MINIMAL := $(filter-out src/lib/inst/gemm/Kuiper.GEMM.TensorCore2D.fst, $(EXTRACT_MINIMAL))
EXTRACT_MINIMAL := $(filter-out src/lib/inst/gemm/Kuiper.GEMM.TensorCore.fst, $(EXTRACT_MINIMAL))
EXTRACT_MINIMAL := $(filter-out src/examples/Kuiper.Example.TensorCore.fst, $(EXTRACT_MINIMAL))

extract-minimal: $(patsubst %,obj/%.cu,$(subst .,_,$(basename $(notdir $(EXTRACT_MINIMAL)))))
extract-minimal: $(patsubst %,obj/%.h, $(subst .,_,$(basename $(notdir $(EXTRACT_MINIMAL)))))

BUILD :=
BUILD_MINIMAL :=

# *Build* every executable in test/, we can do this without a GPU
BUILD += $(patsubst %,obj/%.exe,$(TESTS))
BUILD += obj/Kuiper_Example2.o
ifeq ($(KUIPER_CFG_TENSORCORES),0)
TENSORCORE_BUILD := $(foreach f,$(BUILD),$(if $(findstring TensorCore,$(f)),$(f)))
BUILD := $(filter-out $(TENSORCORE_BUILD),$(BUILD))
endif

build-all: $(BUILD)

BUILD_MINIMAL := $(BUILD)
# For minimal, filter out everything mentioning tensorcore
TENSORCORE_BUILD := $(foreach f,$(BUILD),$(if $(findstring TensorCore,$(f)),$(f)))
BUILD_MINIMAL := $(filter-out $(TENSORCORE_BUILD),$(BUILD_MINIMAL))

build-minimal: $(BUILD_MINIMAL)

.PHONY: test
test: $(patsubst %,$(OUTDIR)/%.test,$(TESTS))

.PHONY: accept
accept: $(patsubst %,$(OUTDIR)/%.accept,$(TESTS))
