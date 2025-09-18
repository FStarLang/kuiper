#ifndef __KUIPER_TENSORCORE_H
#define __KUIPER_TENSORCORE_H 1

#include <mma.h>
using namespace nvcuda;
#define kpr_fragment wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>

// Huge hack: krml generates an assignment that we don't really want,
// so set it to 0.
#define KPR_INIT(ty)\
   (ty){0}
#define KPR_FRAGMENT_TYPE(et, kind, m, n, k, layout) \
   wmma::fragment<kind, m, n, k, et, layout>
#define KPR_FRAGMENT_TYPE_C(et, kind, m, n, k) \
   wmma::fragment<kind, m, n, k, et>
#define KPR_ARRAY_FRAGMENT_TYPE(fragment_ty, size) \
   fragment_ty[size]

#endif /* __KUIPER_TENSORCORE_H */
