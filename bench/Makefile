bench: runner.cu sgemm.cu
	nvcc -O3 -I . runner.cu sgemm.cu -o $@ -lcublas
