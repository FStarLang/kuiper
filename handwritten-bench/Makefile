all: Bench_B2D.exe Bench_TC2D.exe

NVCC := nvcc -O3 -arch=native

%.o: %.cu
	$(NVCC) -c -I ../include -I ../test $< -o $@

Bench_B2D.exe: Bench_B2D.o BlockTiling2D.o
	$(NVCC) $^ -o $@

Bench_TC2D.exe: Bench_TC2D.o TensorCore2D.o
	$(NVCC) $^ -o $@
