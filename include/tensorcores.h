#ifndef __KUIPER_TENSORCORE_H
#define __KUIPER_TENSORCORE_H 1

#include <mma.h>
using namespace nvcuda;
#define kpr_fragment wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>

// Huge hack: krml generates an assignment that we don't really want,
// so set it to 0.
#define KPR_FRAGMENT_INIT(et, kind, m, n, k, layout)				\
   (wmma::fragment<kind, m, n, k, et, layout>){0}
#define KPR_FRAGMENT_INIT_C(et, kind, m, n, k)					\
   (wmma::fragment<kind, m, n, k, et>){0}

#endif /* __KUIPER_TENSORCORE_H */
