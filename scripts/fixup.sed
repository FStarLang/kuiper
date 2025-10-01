:again
s/threadIdx_x/threadIdx.x/g
s/blockDim_x/blockDim.x/g
s/blockIdx_x/blockIdx.x/g
s/gridDim_x/gridDim.x/g
s/wmma__/wmma::/g
/__global__/{n;/^$/d}
/__device__/{n;/^$/d}
# try to unfold kpr_offset
s/kpr_offset(\([^ ()]*\), \(([^ ]*)[^ ]*\))/(\1 + \2)/g
s/auto_AMP/auto\&/g
# an awful fix to rewrite auto& *foo into auto& foo
s/auto\& *\*/auto\&/g
/auto\&$/{n;s/^\( *\)\*/\1/; b again}
