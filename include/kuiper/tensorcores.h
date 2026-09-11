#ifndef KUIPER_TENSORCORES_H
#define KUIPER_TENSORCORES_H 1

/* Custard emits the union of every registered external's header into each
   generated header, so a unit with no tensor-core code still includes this
   one.  It therefore has to be harmless on a target without tensor cores:
   [nvcuda] would not exist there and even [using namespace nvcuda] is an
   error.  Same guard as kuiper.h applies to this include.  Modules that do
   use tensor cores are excluded from the build entirely in that
   configuration (nvcc.mk). */
#if (!defined(KUIPER_CFG_TENSORCORES) || KUIPER_CFG_TENSORCORES)

#include <mma.h>
#include "wgmma.h"
using namespace nvcuda;

// Some macros to work around not being able to emit fragment types
// directly during karamel extraction.
#define kpr_fragment(...)                         wmma::fragment<__VA_ARGS__>
#define KPR_INIT(ty)                              (ty){0}
#define KPR_INIT_ARR(ty, size)                    (ty[size]){0}

// Element-wise access to a wmma::fragment's register array. [fr] is a
// wmma::fragment (lvalue); [num_elements] and the register array [x] are
// members of wmma::fragment. KPR_FRAG_GET is emitted by the extraction of
// mma_loadA_map / mma_loadB_map / mma_store_comb (below) to read a register;
// the other two are used by the KPR_LOAD_MAP / KPR_STORE_COMB loops.
#define KPR_NELEM(fr)                             ((fr).num_elements)
#define KPR_FRAG_GET(fr, i)                       ((fr).x[(i)])
#define KPR_FRAG_SET(fr, i, v)                    ((fr).x[(i)] = (v))

// Load a wmma fragment, then apply an elementwise device map to every register.
// [mapped] is the (inlined) mapped value written in terms of the loop counter
// [_kpr_i], e.g. f(KPR_FRAG_GET(fr, _kpr_i)). The extraction plugin only has to
// build [mapped]; the load and the register loop live here. The counter carries
// a _kpr_ prefix so it cannot collide with karamel-generated locals (i, j, i0
// ...) that may appear inside [fr] or [mapped].
// register read to the named local [_kpr_in_v], and the plugin emits [mapped] in
// terms of it (e.g. f(_kpr_in_v)); an identity map leaves the register untouched.
#define KPR_LOAD_MAP(gm, fr, ldm, mapped)                                      \
  do {                                                                         \
    wmma::load_matrix_sync((fr), (gm), (ldm));                                 \
    for (uint32_t _kpr_i = 0; _kpr_i < KPR_NELEM(fr); _kpr_i++) {             \
      [[maybe_unused]] auto _kpr_in_v = KPR_FRAG_GET(fr, _kpr_i);             \
      KPR_FRAG_SET(fr, _kpr_i, (mapped));                                      \
    }                                                                          \
  } while (0)

// Read-modify-write store: copy the accumulator [fr] into a scratch fragment
// [_kpr_old] (a value copy, purely to obtain [fr]'s opaque 'auto' type), load
// the resident tile from [gm] into [_kpr_old], combine it register-wise with
// [fr] via the (inlined) [combined] expression, then store [fr]. The two register
// reads are bound here to named locals [_kpr_old_v] (old C) and [_kpr_new_v]
// (accumulator), and the plugin builds [combined] in terms of them (e.g.
// g(_kpr_old_v, _kpr_new_v)). Binding the reads to locals -- rather than emitting
// them inside [combined] -- is what lets an overwrite combine (g old new = new)
// drop [_kpr_old_v]: karamel treats a bare identifier as readonly, so an unused
// one is dropped instead of being hoisted out of this macro as a stray ignore.
#define KPR_STORE_COMB(gm, fr, ldm, combined)                                  \
  do {                                                                         \
    auto _kpr_old = (fr);                                                      \
    wmma::load_matrix_sync(_kpr_old, (gm), (ldm), wmma::mem_row_major);        \
    for (uint32_t _kpr_i = 0; _kpr_i < KPR_NELEM(fr); _kpr_i++) {             \
      [[maybe_unused]] auto _kpr_old_v = KPR_FRAG_GET(_kpr_old, _kpr_i);      \
      [[maybe_unused]] auto _kpr_new_v = KPR_FRAG_GET(fr, _kpr_i);            \
      KPR_FRAG_SET(fr, _kpr_i, (combined));                                    \
    }                                                                          \
    wmma::store_matrix_sync((gm), (fr), (ldm), wmma::mem_row_major);           \
  } while (0)

// Section 66 (Custard).  Two spellings whose trailing argument is a fixed
// token rather than a value.  Custard requires every symbol a rule refers to
// to be a declaration in the program (error 379), so a bare
// wmma::mem_row_major cannot be synthesized at the call site the way the
// karamel plugin does it.  Wrapping the token in a macro is what keeps the
// rule referring only to declarations.
#define KPR_LOAD_ACCUM(fr, gm, ldm)                                            \
  wmma::load_matrix_sync((fr), (gm), (ldm), wmma::mem_row_major)
#define KPR_STORE(gm, fr, ldm)                                                 \
  wmma::store_matrix_sync((gm), (fr), (ldm), wmma::mem_row_major)

#endif /* KUIPER_CFG_TENSORCORES */

#endif /* KUIPER_TENSORCORES_H */
