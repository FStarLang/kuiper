

#include "Kuiper_AtomicReduce_F64.h"

double_t Kuiper_AtomicReduce_F64_reduce(size_t n, double_t *a)
{
  KRML_MAYBE_UNUSED_VAR(n);
  KRML_MAYBE_UNUSED_VAR(a);
  KRML_HOST_EPRINTF("KaRaMeL abort at %s:%d\n%s\n", __FILE__, __LINE__, "");
  KRML_HOST_EXIT(255U);
}

