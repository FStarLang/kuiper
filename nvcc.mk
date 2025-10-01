# Must be included after .common.mk, .configure.mk, and defininig OUTDIR

NVCC_FLAGS += -O3
NVCC_FLAGS += -I include
NVCC_FLAGS += -I obj # needed for files in test/ only..
# NVCC_FLAGS += -arch=sm_75 # cc lower than 7.5 will be removed in future nvcc versions
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

$(OUTDIR)/%.output: $(OUTDIR)/%.exe
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

TESTS+=$(notdir $(basename $(wildcard test/*.cu)))

NOTEST += Test_Kuiper_Softmax__F16
ifeq ($(KUIPER_CFG_TENSORCORES),0)
NOTEST += $(foreach f,$(TESTS),$(if $(findstring TensorCore,$(f)),$(f)))
endif

# This does NOT work due to bad extraction of fragments (copy vs reference)
NOTEST += Test_Kuiper_GEMM_TensorCore2D__F16_F16_64x64x64_16x16x16_2x2
NOTEST += Test_Kuiper_GEMM_TensorCore2D__F16_F16_64x64x64_16x16x16_4x4

# Disable softmax 16. It works fine locally (outside of docker)
# but fails within in with undefined __hdiv. The nvcc there is slightly
# older. Suprisignly using / just works, but that fails for other
# operators. Forget it for now, but we should be principled about using
# the correct feature flags or whatever.

TESTS := $(filter-out $(NOTEST), $(TESTS))

EXTRACT :=

# Extract everything in src/examples
EXTRACT += $(wildcard src/examples/*.fst)
# Extract everything in src/lib/inst, they are the C api for the library
EXTRACT += $(wildcard src/lib/inst/*.fst)
# And src/lib/inst/gemm...
EXTRACT += $(wildcard src/lib/inst/gemm/*.fst)
EXTRACT += src/lib/graph/Kuiper.GraphDist.fst
EXTRACT += src/examples/Kuiper.Example2.fst

extraction-targets: $(patsubst %,obj/%.cu,$(subst .,_,$(basename $(notdir $(EXTRACT)))))
extraction-targets: $(patsubst %,obj/%.h, $(subst .,_,$(basename $(notdir $(EXTRACT)))))

BUILD :=

# *Build* every executable in test/, we can do this without a GPU
BUILD += $(patsubst %,obj/%.exe,$(TESTS))
BUILD += obj/Kuiper_Example2.o
ifeq ($(KUIPER_CFG_TENSORCORES),0)
TENSORCORE_BUILD := $(foreach f,$(BUILD),$(if $(findstring TensorCore,$(f)),$(f)))
BUILD := $(filter-out $(TENSORCORE_BUILD),$(BUILD))
endif

build-targets: $(BUILD)

.PHONY: test
test: $(patsubst %,$(OUTDIR)/%.test,$(TESTS))

.PHONY: accept
accept: $(patsubst %,$(OUTDIR)/%.accept,$(TESTS))
