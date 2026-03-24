#ifndef __KUIPER_TENSORCORE_H
#define __KUIPER_TENSORCORE_H 1

#include <mma.h>
using namespace nvcuda;

// Some macros to work around not being able to emit fragment types
// directly during karamel extraction.
#define kpr_fragment(...)                         wmma::fragment<__VA_ARGS__>
#define KPR_INIT(ty)                              (ty){0}
#define KPR_INIT_ARR(ty, size)                    (ty[size]){0}

#endif /* __KUIPER_TENSORCORE_H */
