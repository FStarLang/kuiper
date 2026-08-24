.SUFFIXES:

SPUTNIK_DIR = sputnik
KUIPER_DIR  = ../..

NVCC       = nvcc
NVCC_FLAGS = -O3 -arch=native -std=c++14
NVCC_FLAGS += -I $(SPUTNIK_DIR)
NVCC_FLAGS += -I $(KUIPER_DIR)/include
NVCC_FLAGS += -I $(KUIPER_DIR)/obj
NVCC_FLAGS += -DKUIPER_CFG_TENSORCORES=0

# NVML, for the clock-stability check in the benchmark driver. With the
# toolkit from NVIDIA's repo the only libnvidia-ml.so in the image is the
# link stub; the real one is injected by the container runtime.
CUDA_HOME  := $(patsubst %/bin/nvcc,%,$(shell command -v nvcc))
LDFLAGS    += $(addprefix -L,$(wildcard $(CUDA_HOME)/lib64/stubs)) -lnvidia-ml

.PHONY:
clone:
	git clone https://github.com/google-research/sputnik

.cloned:
	@[ -d $(SPUTNIK_DIR) ] || (echo '$(SPUTNIK_DIR) not found, run `make clone` to fetch it?'; false)
	touch .cloned

# Sputnik SpMM object.
#
# Built from our own TU, which #includes Sputnik's implementation and adds the
# explicit CudaSpmmEx<> instantiation for the tile shape the benchmark pins.
# Do NOT also compile cuda_spmm.cu.cc: it would define the same symbols.
sputnik_spmm.o: sputnik_spmm_ex.cu spmm_bench_config.h \
                $(SPUTNIK_DIR)/sputnik/spmm/cuda_spmm.cu.cc .cloned
	$(NVCC) $(NVCC_FLAGS) -I . -c $< -o $@

# Keep the extracted CUDA synchronized with the Kuiper sources before deciding
# whether the CUDA object itself needs rebuilding.
.PHONY: kuiper_spmm_cuda
kuiper_spmm_cuda:
	$(MAKE) -C $(KUIPER_DIR) obj/Klas_SPMM.cu

$(KUIPER_DIR)/obj/Klas_SPMM.cu: | kuiper_spmm_cuda

# Kuiper SpMM object (from obj/, which has the latest build including f32)
kuiper_spmm.o: $(KUIPER_DIR)/obj/Klas_SPMM.cu | kuiper_spmm_cuda
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Benchmark driver
spmm_bench.o: spmm_bench.cu spmm_bench_config.h .cloned
	$(NVCC) $(NVCC_FLAGS) -I . -c $< -o $@

spmm_bench: spmm_bench.o sputnik_spmm.o kuiper_spmm.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o spmm_bench

.DEFAULT_GOAL := spmm_bench
