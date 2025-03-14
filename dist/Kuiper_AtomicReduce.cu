

#include "Kuiper_AtomicReduce.h"

uint32_t Kuiper_AtomicReduce_reduce_u32(size_t n, uint32_t *a)
{
  KRML_MAYBE_UNUSED_VAR(n);
  KRML_MAYBE_UNUSED_VAR(a);
  KRML_HOST_EPRINTF("KaRaMeL abort at %s:%d\n%s\n", __FILE__, __LINE__, "");
  KRML_HOST_EXIT(255U);
}

uint64_t Kuiper_AtomicReduce_reduce_u64(size_t n, uint64_t *a)
{
  KRML_MAYBE_UNUSED_VAR(n);
  KRML_MAYBE_UNUSED_VAR(a);
  KRML_HOST_EPRINTF("KaRaMeL abort at %s:%d\n%s\n", __FILE__, __LINE__, "");
  KRML_HOST_EXIT(255U);
}

float_t Kuiper_AtomicReduce_reduce_f32(size_t n, float_t *a)
{
  KRML_MAYBE_UNUSED_VAR(n);
  KRML_MAYBE_UNUSED_VAR(a);
  KRML_HOST_EPRINTF("KaRaMeL abort at %s:%d\n%s\n", __FILE__, __LINE__, "");
  KRML_HOST_EXIT(255U);
}

double_t Kuiper_AtomicReduce_reduce_f64(size_t n, double_t *a)
{
  KRML_MAYBE_UNUSED_VAR(n);
  KRML_MAYBE_UNUSED_VAR(a);
  KRML_HOST_EPRINTF("KaRaMeL abort at %s:%d\n%s\n", __FILE__, __LINE__, "");
  KRML_HOST_EXIT(255U);
}

