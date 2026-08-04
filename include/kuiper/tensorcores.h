#ifndef KUIPER_TENSORCORES_H
#define KUIPER_TENSORCORES_H 1

#include <mma.h>
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

// Number of registers actually backing a fragment's [x] array. Unevaluated
// (sizeof only), so it is usable in a constant expression -- unlike
// KPR_NELEM(fr), which goes through a member access on an object.
#define KPR_FRAG_STORAGE(fr)                      (sizeof((fr).x) / sizeof((fr).x[0]))

// A fragment with the same use, shape and layout as [Frag] but a different
// element type. KPR_STORE_COMB needs to name a fragment matching its
// accumulator's shape, and the extraction plugin cannot spell that shape: the
// fragment dimensions are `erased nat` at the mma_store_comb call site, so they
// extract to units. Recovering them from the accumulator's own type sidesteps
// that, and is stronger than passing them explicitly -- the two fragments agree
// on use/shape/layout by construction.
template <typename Frag, typename T> struct kpr_retype_frag;

template <typename Use, int m, int n, int k, typename T0, typename L, typename T>
struct kpr_retype_frag<wmma::fragment<Use, m, n, k, T0, L>, T> {
  using type = wmma::fragment<Use, m, n, k, T, L>;
};

// [decltype] of a subscript or parenthesized expression is a reference, and the
// accumulator often reaches the macro as an array element, so strip qualifiers.
template <typename Frag, typename T>
struct kpr_retype_frag<Frag &, T> : kpr_retype_frag<Frag, T> {};
template <typename Frag, typename T>
struct kpr_retype_frag<const Frag, T> : kpr_retype_frag<Frag, T> {};

#define KPR_RETYPE_FRAG(fr, T) typename kpr_retype_frag<decltype(fr), T>::type

// Read-modify-write store: declare a scratch accumulator fragment [_kpr_old]
// whose element type is the DESTINATION matrix's, load the resident tile from
// [gm] into it, combine it register-wise with the accumulator [fr] via the
// (inlined) [combined] expression, and store the result. The trailing variadic
// parameter is the destination's *element* type (half / float / ...), supplied
// by the extraction plugin; the fragment's shape comes from [fr].
//
// Typing [_kpr_old] by the destination rather than by [fr] is what makes a
// heterogeneous epilogue (e.g. f32 accumulator into an f16 C matrix) work at
// all: wmma has no converting load/store, so both the load of the old tile and
// the store of the result must go through a fragment whose element type is
// exactly [gm]'s. The combine itself does the conversion, register-wise. When
// the accumulator and the destination agree this degenerates to the previous
// behaviour.
//
// ASSUMPTION: for a given (kind, m, n, k), a fragment's register-to-matrix-cell
// mapping does not depend on its element type, so [_kpr_old.x[i]] and
// [fr.x[i]] name the same cell. CUDA treats fragment layouts as opaque and does
// not guarantee this, so the static_assert below at least catches a change in
// the register count; a mismatch in the mapping itself would be silent.
//
// The two register reads are bound here to named locals [_kpr_old_v] (old C)
// and [_kpr_new_v] (accumulator), and the plugin builds [combined] in terms of
// them (e.g. g(_kpr_old_v, _kpr_new_v)). Binding the reads to locals -- rather
// than emitting them inside [combined] -- is what lets an overwrite combine
// (g old new = new) drop [_kpr_old_v]: karamel treats a bare identifier as
// readonly, so an unused one is dropped instead of being hoisted out of this
// macro as a stray ignore. For the same reason the plugin must hand us
// [combined] as a single C *expression*: a let-binding would be hoisted to a
// statement outside this macro, where [_kpr_old_v] does not exist.
#define KPR_STORE_COMB(gm, fr, ldm, combined, ...)                             \
  do {                                                                         \
    KPR_RETYPE_FRAG(fr, __VA_ARGS__) _kpr_old;                                 \
    static_assert(KPR_FRAG_STORAGE(_kpr_old) == KPR_FRAG_STORAGE(fr),          \
                  "KPR_STORE_COMB: the destination and accumulator fragments "  \
                  "disagree on register count; the register-wise combine "     \
                  "below would not be cell-to-cell.");                         \
    wmma::load_matrix_sync(_kpr_old, (gm), (ldm), wmma::mem_row_major);        \
    for (uint32_t _kpr_i = 0; _kpr_i < KPR_NELEM(fr); _kpr_i++) {             \
      [[maybe_unused]] auto _kpr_old_v = KPR_FRAG_GET(_kpr_old, _kpr_i);      \
      [[maybe_unused]] auto _kpr_new_v = KPR_FRAG_GET(fr, _kpr_i);            \
      KPR_FRAG_SET(_kpr_old, _kpr_i, (combined));                              \
    }                                                                          \
    wmma::store_matrix_sync((gm), _kpr_old, (ldm), wmma::mem_row_major);       \
  } while (0)

#endif /* KUIPER_TENSORCORES_H */
