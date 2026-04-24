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
	touch .cloned

.cloned:
	$(error $(SPUTNIK_DIR) not found, run `make clone` to fetch it?)

# Sputnik SpMM object
sputnik_spmm.o: $(SPUTNIK_DIR)/sputnik/spmm/cuda_spmm.cu.cc .cloned
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

# Kuiper SpMM object (from obj/, which has the latest build including f32)
kuiper_spmm.o: $(KUIPER_DIR)/obj/Kuiper_Example_Sparse_SPMM.cu
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
