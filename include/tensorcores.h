#ifndef __KUIPER_TENSORCORE_H
#define __KUIPER_TENSORCORE_H 1

#include <mma.h>
using namespace nvcuda;

// Some hacky macros to work around not being able to emit fragment types
// directly during karamel extraction.
#define KPR_INIT(ty)                              (ty){0}
#define KPR_FRAG_TY(et, kind, m, n, k, layout)    wmma::fragment<kind, m, n, k, et, layout>
#define KPR_FRAG_TY_C(et, kind, m, n, k)          wmma::fragment<kind, m, n, k, et>
#define KPR_ARRAY(ty, size)                       ty[size]

#endif /* __KUIPER_TENSORCORE_H */
