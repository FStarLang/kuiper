:again
s/threadIdx_x/threadIdx.x/g
s/blockDim_x/blockDim.x/g
s/blockIdx_x/blockIdx.x/g
s/gridDim_x/gridDim.x/g
s/wmma__/wmma::/g
# Karamel emits a blank line after each CPrologue qualifier. Remove the
# gaps before clang-format, including the one between inline and __device__.
/^\(inline\|__global__\|__device__\|__host__\)$/{n;/^[[:space:]]*$/d; b again}
s/auto_AMP/auto\&/g
# an awful fix to rewrite auto& *foo into auto& foo
s/auto\& *\*/auto\&/g
/auto\&$/{n;s/^\( *\)\*/\1/; b again}
