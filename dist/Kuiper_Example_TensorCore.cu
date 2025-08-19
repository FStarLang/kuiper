

#include "Kuiper_Example_TensorCore.h"

typedef enum { FragA, FragB, FragAccum } fragment_kind;

typedef enum { FragLRM, FragLCM, FragLAccum } fragment_layout;

__device__
void Kuiper_Example_TensorCore_test(half_t *m1, half_t *m2, half_t *m3)
{
  KRML_MAYBE_UNUSED_VAR(m3);
  kpr_fragment fragA = KPR_FRAGMENT(FragA, (size_t)16U, (size_t)16U, (size_t)16U, FragLRM);
  kpr_fragment fragB = KPR_FRAGMENT(FragB, (size_t)16U, (size_t)16U, (size_t)16U, FragLRM);
  kpr_fragment buf = KPR_FRAGMENT(FragAccum, (size_t)16U, (size_t)16U, (size_t)16U, FragLAccum);
  KRML_HOST_IGNORE(&buf);
  wmma::load_matrix_sync(fragA, m1, (size_t)0U);
  wmma::load_matrix_sync(fragB, m2, (size_t)0U);
}

