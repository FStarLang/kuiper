
#include "Kuiper_Example_Float32Order.h"

inline __host__ __device__ bool Kuiper_Example_Float32Order_compare(
    float x, float y)
{
    return x < y;
}
