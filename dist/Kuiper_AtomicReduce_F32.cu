

#include "Kuiper_AtomicReduce_F32.h"

float_t Kuiper_AtomicReduce_F32_reduce(size_t n, float_t *a)
{
  KRML_MAYBE_UNUSED_VAR(n);
  KRML_MAYBE_UNUSED_VAR(a);
  KRML_HOST_EPRINTF("KaRaMeL abort at %s:%d\n%s\n", __FILE__, __LINE__, "");
  KRML_HOST_EXIT(255U);
}

