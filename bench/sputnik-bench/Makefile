.SUFFIXES:

SPUTNIK_DIR = sputnik
KUIPER_DIR  = ../..

NVCC       = nvcc
NVCC_FLAGS = -O3 -arch=native -std=c++14
NVCC_FLAGS += -I $(SPUTNIK_DIR)
NVCC_FLAGS += -I $(KUIPER_DIR)/include
NVCC_FLAGS += -I $(KUIPER_DIR)/obj
NVCC_FLAGS += -DKUIPER_CFG_TENSORCORES=0

.PHONY:
clone:
	git clone https://github.com/google-research/sputnik

.cloned:
	@[ -d $(SPUTNIK_DIR) ] || (echo '$(SPUTNIK_DIR) not found, run `make clone` to fetch it?'; false)
	touch .cloned

# Sputnik SpMM object
sputnik_spmm.o: $(SPUTNIK_DIR)/sputnik/spmm/cuda_spmm.cu.cc .cloned
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

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
spmm_bench.o: spmm_bench.cu .cloned
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

spmm_bench: spmm_bench.o sputnik_spmm.o kuiper_spmm.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o spmm_bench

.DEFAULT_GOAL := spmm_bench
