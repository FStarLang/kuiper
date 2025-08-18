

#include "Kuiper_Example_TensorCore.h"

typedef enum { FragA, FragB, FragAccum } fragment_kind;

typedef enum { FragLRM, FragLCM, FragLAccum } fragment_layout;

__device__

void Kuiper_Example_TensorCore_test(half_t *m1, half_t *m2, half_t *m3)
{
  KRML_MAYBE_UNUSED_VAR(m1);
  KRML_MAYBE_UNUSED_VAR(m2);
  KRML_MAYBE_UNUSED_VAR(m3);
  kpr_fragment buf = KPR_FRAGMENT(FragA, (size_t)16U, (size_t)16U, (size_t)16U, FragLRM);
  KRML_HOST_IGNORE(&buf);
}

