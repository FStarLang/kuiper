
#include <iostream>
#include <cstdint>
#include <cuda_fp16.h>

#include <mma.h>
using namespace nvcuda;

// #include <cuda_runtime.h>

//// BEGIN TUNING PARAMS
// BM divides rows
#define BM 128
// BN divides cols
#define BN 128
// BK divides shared
#define BK 64
// TM divides BM
#define TM 16
// TN divides BN
#define TN 16
// TK divides BK
#define TK 16
// WM*TM divides BM
#define WM 2
// WN*TN divides BN
#define WN 2
//// END TUNING PARAMS

#define WARP_SIZE 32
#define GRID_DIM(rows, cols) ((rows)/BM * ((cols)/BN))
#define BLOCK_DIM (BM/(WM*TM) * (BN/(WN*TN) * WARP_SIZE))

__global__
void tensorcore2d(uint32_t shared, uint32_t cols, half *gA, half *gB, half *gC)
{
    extern __shared__ half buffer[];
    half *sA = buffer;
    half *sB = &buffer[BM * BK];

    // allocate tensor core fragments
    wmma::fragment<wmma::matrix_a, TM, TN, TK, half, wmma::row_major> aFrags[WM];
    wmma::fragment<wmma::matrix_b, TM, TN, TK, half, wmma::row_major> bFrags[WN];
    wmma::fragment<wmma::accumulator, TM, TN, TK, half> accFrags[WM][WN];
    for (uint32_t i = 0; i < WM; i++) {
        for (uint32_t j = 0; j < WN; j++) {
            wmma::fill_fragment(accFrags[i][j], 0.0f);
        }
    }
    
    uint32_t num_n_tiles = cols / BN;
    // for each block, jump to beginning of tile row in gA
    uint32_t mrow = blockIdx.x / num_n_tiles;
    gA += shared * (mrow * BM);
    // for each block, jump to beginning of tile column in gB
    uint32_t mcol = blockIdx.x % num_n_tiles;
    gB += mcol * BN;
    // for each block, jump to the beginnin of the result tile in gC
    gC += mrow * BM * cols + mcol * BN;
    
    uint32_t warpIdx = threadIdx.x / WARP_SIZE;
    // row and column for each warp tile within the block tile
    uint32_t warpRow = warpIdx / (BN / (WN*TN));
    uint32_t warpCol = warpIdx % (BN / (WN*TN));

    for (uint32_t bkIdx = 0; bkIdx < shared; bkIdx += BK) {
        __syncthreads();

        // load A vectorized into shared memory
        for (uint32_t i = 0; i < BM * BK; i += BLOCK_DIM * 8) {
            uint32_t row = (i + threadIdx.x * 8) / BK;
            uint32_t col = (i + threadIdx.x * 8) % BK;

            reinterpret_cast<float4*>(&sA[row * BK + col])[0] =
                reinterpret_cast<float4*>(&gA[bkIdx + shared * row + col])[0];
        }
        
        // load B vectorized into shared memory
        for (uint32_t i = 0; i < BK * BN; i += BLOCK_DIM * 8) {
            uint32_t row = (i + threadIdx.x * 8) / BN;
            uint32_t col = (i + threadIdx.x * 8) % BN;

            reinterpret_cast<float4*>(&sB[row * BN + col])[0] =
              reinterpret_cast<float4*>(&gB[cols * bkIdx + cols * row + col])[0];
        }
        __syncthreads();

        // compute subproducts per warp tile using tensor cores
        for (uint32_t dotIdx = 0; dotIdx < BK / TK; dotIdx++) {
            
            // load tensor core tile from shared memory tile of A
            for (uint32_t i = 0; i < WM; i++) {
                // jump to the row of warp tiles for this warp
                half *rowWarpTilesA = &sA[BK * (warpRow * (WM*TM))];
                // jump to the current tensor core tile for this warp
                half *tc_tileA = &rowWarpTilesA[dotIdx * TK + BK * (i * TM)];
                wmma::load_matrix_sync(aFrags[i], tc_tileA, BK);
            }
            
            // load tensor core tile from shared memory tile of B
            for (uint32_t i = 0; i < WN; i++) {
                // jump to the col of warp tiles for this warp
                half *colWarpTilesB = &sB[warpCol * (WN*TN)];
                // jump to the current tensor core tile for this warp
                half *tc_tileB = &colWarpTilesB[BN * (dotIdx * TK) + i * TN];
                wmma::load_matrix_sync(bFrags[i], tc_tileB, BN);
            }
            
            // multiply all tiles using tensor core operations
            for (uint32_t resIdxM = 0; resIdxM < WM; resIdxM++) {
                for (uint32_t resIdxN = 0; resIdxN < WN; resIdxN++) {
                    wmma::mma_sync(
                        accFrags[resIdxM][resIdxN],
                        aFrags[resIdxM],
                        bFrags[resIdxN],
                        accFrags[resIdxM][resIdxN]
                    );
                }
            }
        }
    }
    
    // store results of gA*gB in gC
    for (uint32_t resIdxM = 0; resIdxM < WM; resIdxM++) {
        for (uint32_t resIdxN = 0; resIdxN < WN; resIdxN++) {
            half *outTcTile = 
                &gC[cols * (warpRow * (WM*TM) + (resIdxM * TM)) +
                        warpCol * (WN*TN) + resIdxN * TN];
            wmma::store_matrix_sync(
                outTcTile,
                accFrags[resIdxM][resIdxN],
                cols,
                wmma::mem_row_major);
        }
    }
}

void check_error(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
        std::cerr << "ERROR -- " << file << ":" << "line" << std::endl;
        std::cerr << "Reason: " << cudaGetErrorString(err);
		exit(1);
	}
}
#define ERROR(err) check_error(err, __FILE__, __LINE__)

void
tensorcore2d_host(
    uint32_t rows,
    uint32_t shared,
    uint32_t cols,
    half *gA,
    half *gB,
    half *gC
) {
    uint32_t nblk = GRID_DIM(rows, cols);
    uint32_t nthr = BLOCK_DIM;
    ERROR(cudaFuncSetAttribute(tensorcore2d,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (BM*BK + BK*BN) * sizeof(half)));
    tensorcore2d<<<nblk, nthr, (BM*BK + BK*BN) * sizeof(half)>>>(shared, cols, gA, gB, gC);
    ERROR(cudaDeviceSynchronize());
}
