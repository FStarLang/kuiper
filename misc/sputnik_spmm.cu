
#include <assert.h>
#include <stdint.h>

typedef uint32_t scalar;

typedef struct smatrix {
    int nnz;
    scalar *elems;
    int *col_ind;
    int *row_off;
} smatrix;

__device__ __forceinline__
void predicateInit(uint32_t *predicates, int words) {
#pragma unroll
    for (int i = 0; i < words; i++) {
        predicates[i] = 0xffffffff;
    }
}

// In sputnik only the first 4 bits of each byte are used
// it's unclear why
// here we use all 8 bits
__device__ __forceinline__
void GetWordAndBitOffsets(int x, int *word, int *bit) {
    const int kWordOffset =
        (x / 8) / sizeof(uint32_t);
    const int kByteOffset =
        (x / 8) % sizeof(uint32_t);
    // is this correct?? doesn't seem right
    // should be kBitOffset = x % 8
    const int kBitOffset =
        (x % 8) % sizeof(uint32_t);

    *word = kWordOffset;
    *bit = kByteOffset * 8 + kBitOffset;
}

__device__ __forceinline__
void DisableBit(uint32_t *predicates, int x) {
    int word, bit;
    GetWordAndBitOffsets(x, &word, &bit);
    predicates[word] &= ~(1 << bit);
}

__device__ __forceinline__
bool GetBit(uint32_t *predicates, int x) {
    int word, bit;
    GetWordAndBitOffsets(x, &word, &bit);
    return (predicates[word] >> bit) & 1;
}

__device__ __forceinline__
void predicateSet(int n, int n_idx, uint32_t *predicates, int threadItemsX, int blockWidth) {
    int index = n_idx + threadIdx.x;

#pragma unroll
    for (int x = 0; x < threadItemsX; ++x) {
      if (index >= n) {
        DisableBit(predicates, x);
      }
      index += blockWidth;
    }
}

template<int blockItemsX, int blockItemsK, int blockWidth, int residueUnroll>

// if the parameters divide evenly this should work
__global__
void spmm_kernel(int rows,
                 int cols,
                 int shared,
                 // sparse matrix in CSR format
                 smatrix gA,
                 // dense matrices in row-major format
                 scalar *gB, scalar *gC)
{
    
    static_assert(blockWidth <= blockItemsK);
    static_assert(blockItemsK % blockWidth == 0);
    static_assert(blockWidth <= blockItemsX);
    static_assert(blockItemsX % blockWidth == 0);

    // grid of M x blockItemsX ==> tiles of C per row of length blockItemsX

    // block row
    const int m_idx = blockIdx.y;
    // first block column
    const int n_idx = blockIdx.x * blockItemsX;

    // this should always hold
    assert(m_idx < rows);
    assert(n_idx < cols);

    // row offset
    const int m_off = gA.row_off[m_idx];
    // number of non-zero elements in the row
    int nnz = gA.row_off[m_idx + 1] - m_off;

    // tiles of gA in shared memory
    scalar elems_tile[blockItemsK];
    int col_ind_tile[blockItemsK];

    const int threadItemsX = blockItemsX / blockWidth;

    // why align 16? it's like this in sputnik

    // 2D tile of gB
    __align__(16) scalar dense_fragment[blockItemsK * threadItemsX];
    // ^ do we need to zero-initialize? Maybe not, since everything is
    // filtered by the predicate.

    // 1D tile of gC
    // sputnik uses float here to accumulate results
    __align__(16) scalar output_fragment[threadItemsX] = {};

    // TODO remove
    // sanity check
    for (int i = 0; i < nnz; i++) {
        assert(output_fragment[i] == 0);
    }


    // Set predicate to mask out-of-range indices
    const int predicateBytes = (threadItemsX + 7) / 8;
    const int predicateWords =
        (predicateBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);

    uint32_t predicates[predicateWords];

    predicateInit(predicates, predicateWords);
    predicateSet(cols, n_idx, predicates, threadItemsX, blockWidth);


    // main loop

    // sparse matrix + block offset + thread offset
    scalar *elems = gA.elems + m_off + threadIdx.x;
    int *col_ind = gA.col_ind + m_off + threadIdx.x;

    int sparse_offset = m_off + threadIdx.x;

    // dense matrix + block offset + thread offset
    scalar *const dense = gB + n_idx + threadIdx.x;

    int dense_offset = n_idx + threadIdx.x;
    
    // Note: in sputnik they only synchronize when there are more than 32 threads
    for (; nnz >= blockItemsK; nnz -= blockItemsK) {

        // cooperatively load the sparse tile
        __syncthreads();

        int sparse_tile_offset = threadIdx.x;
#pragma unroll
        for (int k = 0; k < blockItemsK / blockWidth; k++) {
            elems_tile[sparse_tile_offset] = elems[sparse_offset];
            // column index ==> row index for loading dense matrix
            col_ind_tile[sparse_tile_offset] = cols * col_ind[sparse_offset];

            sparse_offset += blockWidth;
            sparse_tile_offset += blockWidth;
        }

        __syncthreads();

        // load the 2D dense tile for this thread
        int dense_fragment_offset = 0;
#pragma unroll
        for (int k = 0; k < blockItemsK; k++) {
            int dense_it = dense_offset + col_ind_tile[k];
#pragma unroll
            for (int x = 0; x < threadItemsX; x++) {
                if (GetBit(predicates, x)) {
                    // gB is dense
                    dense_fragment[dense_fragment_offset] =
                        gB[dense_it];
                    dense_it += blockWidth;
                    dense_fragment_offset++;

                    // Alternative:
                    // dense_fragment[dense_fragment_offset + x] =
                    //     gB[dense_it + blockWidth * x];
                }
            }
        }

        // compute the product
#pragma unroll
        for (int k = 0, dense_fragment_offset=0; k < blockItemsK; k++, dense_fragment_offset++) {
#pragma unroll
            for (int x = 0; x < threadItemsX; x++, dense_fragment_offset++) {
                output_fragment[x] += elems_tile[k] * dense_fragment[dense_fragment_offset];
                // dense_fragment_offset++;
            }
        }
        
    }

    // output_fragment holds partial products for this thread.
    // We still need to process some elements from row a, because we
    // stepped by blockItemsK, which doesn't necessarily divide nnz.

    // compute residual values

    // kernel precondition
    static_assert(residueUnroll > 0);
    static_assert(blockItemsK % residueUnroll == 0);

    // load sparse tile
    __syncthreads();

    if (residueUnroll > 1) {
        // zero out the sparse tile so we can operate without
        // checking bounds on each iteration
        int sparse_tile_offset = threadIdx.x;
#pragma unroll
        for (int k = 0; k < blockItemsK; k++) {
            elems_tile[sparse_tile_offset] = 0;
            col_ind_tile[sparse_tile_offset] = 0;
            sparse_tile_offset += blockWidth;
        }
        __syncthreads();
    }


    int sparse_tile_offset = threadIdx.x;
    int residue = nnz;
#pragma unroll
    for (int k = 0; k < blockItemsK / blockWidth; k++) {
        if (residue <= threadIdx.x) break;

        elems_tile[sparse_tile_offset] = elems[sparse_offset];
        col_ind_tile[sparse_tile_offset] = cols * col_ind[sparse_offset];

        residue -= blockWidth;
        
        sparse_offset += blockWidth;
        sparse_tile_offset += blockWidth;
    }

    __syncthreads();
    

    sparse_offset = 0;
    residue = nnz;
    // this is only so the loops can be unrolled
#pragma unroll
    for (int k_outer = 0; k_outer++ < blockItemsK / residueUnroll; k_outer++) {
        if (residue <= 0) break;
#pragma unroll
        for (int k_inner = 0; k_inner < residueUnroll; k_inner++) {
            int dense_offset = col_ind_tile[sparse_offset];
#pragma unroll
            for (int x = 0; x < threadItemsX; x++) {
                if (GetBit(predicates, x)) {
                    output_fragment[x] +=
                        elems_tile[sparse_offset] * dense[dense_offset];
                        dense_offset += blockWidth;
                }
            }
            sparse_offset++;
        }
        residue -= residueUnroll;
    }


    // accumulate results

    int out_offset = m_idx * cols + n_idx + threadIdx.x;
#pragma unroll
    for (int x = 0; x < threadItemsX; x++) {
        if (GetBit(predicates, x)) {
            gC[out_offset] = output_fragment[x]; 
            out_offset += blockWidth;
        }
    } 
}
