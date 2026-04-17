all: Bench_B2D.exe Bench_TC2D.exe

NVCC := nvcc -O3 -arch=native

%.o: %.cu
	$(NVCC) -c -I ../include -I ../test $< -o $@

Bench_B2D.exe: Bench_B2D.o BlockTiling2D.o
	$(NVCC) $^ -o $@

Bench_TC2D.exe: Bench_TC2D.o TensorCore2D.o
	$(NVCC) $^ -o $@

run: Bench_B2D.run Bench_TC2D.run

Bench_%.run: Bench_%.exe
	./Bench_$*.exe 100 1024 1024 1024 1 # 1000 laps, with final CPU check
