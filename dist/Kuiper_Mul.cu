
#include "Kuiper_Mul.h"

__device__
    void Kuiper_Mul_kf(uint64_t *a1, uint64_t *a2, uint64_t *ar, uint32_t bid)
{
    ar[bid] = a1[bid] * a2[bid];
}
