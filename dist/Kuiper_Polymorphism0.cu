

#include "Kuiper_Polymorphism0.h"

__device__

void Kuiper_Polymorphism0_kswap_U64(uint64_t *r1, uint64_t *r2)
{
  uint64_t v1 = *r1;
  *r1 = *r2;
  *r2 = v1;
}

__device__

void Kuiper_Polymorphism0_kswap_F32(float_t *r1, float_t *r2)
{
  float_t v1 = *r1;
  *r1 = *r2;
  *r2 = v1;
}

