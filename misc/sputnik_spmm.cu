
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

__device__ __forceinline__
void GetWordAndBitOffsets(int x, int *word, int *bit) {
    const int kWordOffset =
        (x / 8) / sizeof(uint32_t);
    const int kByteOffset =
        (x / 8) % sizeof(uint32_t);
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
void predicateSet(int n_idx, int n, uint32_t *predicates, int threadItemsX, int blockWidth) {
    int index = n_idx + threadIdx.x;

#pragma unroll
    for (int x = 0; x < threadItemsX; ++x) {
      if (index >= n) {
        DisableBit(predicates, x);
      }
      index += blockWidth;
    }
}

template<int blockItemsX, int blockItemsK, int blockWidth>

// creo que si los parámetros dividen bien esto debería andar
__global__
void spmm_kernel(int rows,
                 int cols,
                 int shared,
                 // matrix rala en formato CSR
                 smatrix gA,
                 // matrices densas en formato row major
                 scalar *gB, scalar *gC,
                 // largo del tile de C
                 //int blockItemsX,
                 // largo del tile de reduccion de A
                 //int blockItemsK,
                 // constante para factorizar loop residual y facilitar unrolling
                 int residueUnroll)
{
    
    static_assert(blockWidth <= blockItemsK);
    static_assert(blockItemsK % blockWidth == 0);
    static_assert(blockWidth <= blockItemsX);
    static_assert(blockItemsX % blockWidth == 0);

    // grid de M x blockItemsX ==> tiles de C por fila de largo blockItemsX

    // fila de bloque
    const int m_idx = blockIdx.y;
    // primer columna de bloque
    const int n_idx = blockIdx.x * blockItemsX;

    // creo que esto debería valer siempre
    assert(m_idx < rows);
    assert(n_idx < cols);

    // offset de fila
    const int m_off = gA.row_off[m_idx];
    // cantidad de elementos no nulos de la fila
    int nnz = gA.row_off[m_idx + 1] - m_off;

    // tiles de gA en shmem
    scalar elems_tile[blockItemsK];
    int col_ind_tile[blockItemsK];

    const int threadItemsX = blockItemsX / blockWidth;

    // por qué align 16? está así en sputnik

    // tile 2D de gB
    __align__(16) scalar dense_fragment[blockItemsK * threadItemsX];
    // Guido: ^ hay que inicializar a cero? Capaz no, porque se filtar todo
    // por el predicado.

    // tile 1D de gC
    // acá sputnik usa float para acumular los resultados
    __align__(16) scalar output_fragment[threadItemsX] = {};

    // TODO borrar
    // chequeo pavote
    for (int i = 0; i < nnz; i++) {
        assert(output_fragment[i] == 0);
    }


    // Seteamos predicado para enmascarar indices que caen fuera de rango
    const int predicateBytes = (threadItemsX + 7) / 8;
    const int predicateWords =
        (predicateBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);

    uint32_t predicates[predicateWords];

    predicateInit(predicates, predicateWords);
    predicateSet(cols, n_idx, predicates, threadItemsX, blockWidth);


    // main loop

    // matriz rala + offset de bloque + offset de thread
    scalar *elems = gA.elems + m_off + threadIdx.x;
    int *col_ind = gA.col_ind + m_off + threadIdx.x;

    int sparse_offset = m_off + threadIdx.x;

    // matriz densa + offset de bloque + offset de thread
    scalar *const dense = gB + n_idx + threadIdx.x;

    int dense_offset = n_idx + threadIdx.x;
    
    // Nota: en sputniik solo se sincroniza cuando hay mas de 32 threads
    for (; nnz >= blockItemsK; nnz -= blockItemsK) {

        // cargamos cooperativamente el tile sparse
        __syncthreads();

        int sparse_tile_offset = threadIdx.x;
        // TODO hace falta que dividan bien?
        // assert(blockItemsK % blockWidth == 0);
#pragma unroll
        for (int k = 0; k < blockItemsK / blockWidth; k++) {
            elems_tile[sparse_tile_offset] = elems[sparse_offset];
            // indice de columna ==> indice de fila para cargar matriz densa
            col_ind_tile[sparse_tile_offset] = cols * col_ind[sparse_offset];

            sparse_offset += blockWidth;
            sparse_tile_offset += blockWidth;
        }

        __syncthreads();

        // cargamos el tile 2D denso para este thread
        int dense_fragment_offset = 0;
#pragma unroll
        for (int k = 0; k < blockItemsK; k++) {
            int dense_it = dense_offset + col_ind_tile[k];
#pragma unroll
            for (int x = 0; x < threadItemsX; x++) {
                if (GetBit(predicates, x)) {
                    // Guido: gB es densa
                    dense_fragment[dense_fragment_offset] =
                        gB[dense_it];
                    dense_it += blockWidth;
                    dense_fragment_offset++;

                    // Alternativa:
                    // dense_fragment[dense_fragment_offset + x] =
                    //     gB[dense_it + blockWidth * x];
                }
            }
        }

        // calculamos el producto
#pragma unroll
        for (int k = 0, dense_fragment_offset=0; k < blockItemsK; k++, dense_fragment_offset++) {
#pragma unroll
            for (int x = 0; x < threadItemsX; x++, dense_fragment_offset++) {
                output_fragment[x] += elems_tile[k] * dense_fragment[dense_fragment_offset];
                // dense_fragment_offset++;
            }
        }
        
    }

    // output_fragment tiene productos parciales para este hilo
    // Nos faltan procesar algunos de la fila de a, porque fuimos
    // en pasos de blockItemsK, que no necesariamente divide nnz.

    // calculamos valores residuales

    // precondicion del kernel
    assert(residueUnroll > 0);
    assert(blockItemsK % residueUnroll == 0);

    // cargamos tile sparse
    __syncthreads();

    if (residueUnroll > 1) {
        // ponemos el tile sparse en cero para poder operar sin
        // chequear rangos en cada iteracion
        int sparse_tile_offset = threadIdx.x;
#pragma unroll
        for (int k = 0; k < blockItemsK; k++) {
            elems_tile[sparse_tile_offset] = 0;
            col_ind_tile[sparse_tile_offset] = 0;
            sparse_tile_offset += blockWidth;
        }
        __syncthreads();
    }


    sparse_offset = 0;
    int sparse_tile_offset = threadIdx.x;
#pragma unroll
    for (int k = 0; k < blockItemsK / blockWidth; k++) {
        if (nnz <= threadIdx.x) break;

        elems_tile[sparse_tile_offset] = elems[sparse_offset];
        col_ind_tile[sparse_tile_offset] = cols * col_ind[sparse_offset];

        nnz -= blockWidth;
        
        sparse_offset += blockWidth;
        sparse_tile_offset += blockWidth;
    }

    __syncthreads();
    

    sparse_offset = 0;
    // esto es solo para poder unrollear los loops
#pragma unroll
    for (int k_outer = 0; k_outer++ < blockItemsK / residueUnroll; k_outer++) {
        if (nnz <= 0) break;
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
        nnz -= residueUnroll;
    }


    // acumulamos resultados

    int out_offset = m_idx * cols + n_idx + threadIdx.x;
#pragma unroll
    for (int x = 0; x < threadItemsX; x++) {
        if (GetBit(predicates, x)) {
            gC[out_offset] = output_fragment[x]; 
            out_offset += blockWidth;
        }
    } 
}
