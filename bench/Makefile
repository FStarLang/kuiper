.SUFFIXES:

bench: runner.o sgemm.o
	nvcc -arch=native -O3 $^ -o $@ -lcublas

%.o: %.cu *.cuh
	nvcc -arch=native -O3 -I . -c $< -o $@
