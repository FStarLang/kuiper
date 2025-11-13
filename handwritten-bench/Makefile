all: Bench_B2D.exe

%.o: %.cu
	nvcc -O3 -c -I ../include -I ../test $< -o $@

Bench_B2D.exe: Bench_B2D.o BlockTiling2D.o
	nvcc -O3 $^ -o $@
