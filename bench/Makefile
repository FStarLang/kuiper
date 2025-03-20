.SUFFIXES:

bench: runner.o sgemm.o
	nvcc -O3 $^ -o $@ -lcublas

%.o: %.cu
	nvcc -I . -c $< -o $@
