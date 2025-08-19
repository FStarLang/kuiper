s/threadIdx_x/threadIdx.x/g
s/blockDim_x/blockDix.x/g
s/blockIdx_x/blockIdx.x/g
s/gridDim_x/gridDim.x/g
s/wmma__/wmma::/g
/__global__/{n;/^$/d}
/__device__/{n;/^$/d}
# cast the damn SHMEM
s/\([a-zA-Z_0-9]*\) \*\([a-zA-Z_0-9]*\) = \(.*\)KPR_SHMEM_AT(/\1 *\2 = (\1 *)\3KPR_SHMEM_AT(/
