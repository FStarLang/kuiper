

#include "Kuiper_AtomicReduce_U64.h"

uint64_t Kuiper_AtomicReduce_U64_reduce(size_t n, uint64_t *a)
{
  KRML_MAYBE_UNUSED_VAR(n);
  KRML_MAYBE_UNUSED_VAR(a);
  KRML_HOST_EPRINTF("KaRaMeL abort at %s:%d\n%s\n", __FILE__, __LINE__, "");
  KRML_HOST_EXIT(255U);
}

