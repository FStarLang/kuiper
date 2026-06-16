module Klas.Rotg

(* BLAS level-1 rotg: GENERATE a Givens rotation (c, s, r) from a pair (a, b).
   Corresponds to cublasSrotg/Drotg (real cosine/sine).

   ============================================================================
   BIG NOTE — why this is the "convention-relaxed" Givens generator, and why it
   CANNOT bit-match cuBLAS/reference-BLAS rotg.
   ============================================================================

   The textbook/cuBLAS/LAPACK srotg fixes a SIGN CONVENTION for the output:
   it returns r = sign(a)*hypot(a,b) when |a|>|b| (resp. sign(b)*... otherwise),
   plus a "z" reconstruction scalar, all of which are defined through copysign /
   the sign bit. Our floating-point model (Kuiper.Floating.Base) *deliberately
   cannot observe the sign bit*:

     1. The sign of zero is unobservable by ANY arithmetic test — this is IEEE
        754, not just our model: `+0.0 == -0.0` is TRUE and `-0.0 < +0.0` is
        FALSE, so eq/lt/lte (the only comparison ops in the `floating` class)
        cannot distinguish +0 from -0. Only signbit/copysign can.
     2. `neg x` is defined as `zero `sub` x` (see neg_kind/neg_neg), and
        `sub zero zero` is +0 under IEEE subtraction — so you cannot even
        *construct* a observable -0 this way.
     3. The model additionally *identifies* +0 and -0 propositionally (`==`
        conflates them), precisely so the clean axioms hold (neg_neg, the total
        preorder on non-NaNs, lt_neg_flip, ...). Hence NO theorem can depend on
        the sign of a zero.
     4. `copysign` exists in the class but is left UNAXIOMATIZED (a primitive
        for extraction only — you can call it, but can prove nothing about it),
        and `signbit` is not provided at all.

   Therefore cuBLAS's exact signed result (the sign of r, and the z parameter)
   is NOT verifiable in this model. We instead generate a *functionally valid*
   Givens rotation with the canonical NON-NEGATIVE radius:

        r = +sqrt(a*a + b*b)  >= 0,   c = a / r,   s = b / r.

   This is a genuine Givens rotation — the matrix [[c, s], [-s, c]] sends
   (a, b) to (r, 0) — and it is what `Klas.Rot` (cublasSrot) consumes. It just
   does not reproduce cuBLAS's sign of r / z when a (or b) is negative.

   Two further simplifications, both documented so callers are not surprised:
     * We OMIT the z reconstruction scalar (it is sign-convention defined).
     * We use the naive a*a + b*b (no max-scaling), so very large inputs may
       overflow. Reference BLAS scales by max(|a|,|b|) for overflow safety;
       that needs fabs+branch and buys nothing for the spec, so we skip it.

   The verified guarantee (see rotg_f32/_f64 below and s_rotg_rotates) is the
   real-number correctness of the result: the returned c, s, r APPROXIMATE the
   exact real Givens parameters of (a, b), which form a true rotation taking
   (a, b) to (r, 0). The degenerate all-zero input (a == b == 0, where r == 0
   and c, s are undefined) is EXCLUDED by precondition.
   ============================================================================ *)

#lang-pulse
open Kuiper

(* Result bundle. cuBLAS returns these through the a/b/c/s pointers; we return
   them by value (r overwrites a in cuBLAS; we expose it explicitly as rr). *)
noeq
type rotg_out (et:Type0) = {
  rc : et;   (* cosine  c *)
  rs : et;   (* sine    s *)
  rr : et;   (* radius  r = sqrt(a*a + b*b) >= 0 *)
}

(* --- Real-number specification of the (convention-relaxed) Givens generator. *)

(* Non-negative radius. (unfold: proofs and specs see straight through.) *)
unfold let s_rotg_r (ra rb : real) : real = realsqrt (ra *. ra +. rb *. rb)

(* Cosine / sine. Defined only when the radius is non-zero (i.e. not both
   inputs are zero), which the radius-refinement makes explicit. *)
unfold let s_rotg_c (ra rb : real { s_rotg_r ra rb =!= 0.0R }) : real = ra /. s_rotg_r ra rb
unfold let s_rotg_s (ra rb : real { s_rotg_r ra rb =!= 0.0R }) : real = rb /. s_rotg_r ra rb

(* The spec really is a Givens rotation: [[c,s],[-s,c]] * (a,b) = (r,0). *)
val s_rotg_rotates (ra rb : real)
  : Lemma (requires s_rotg_r ra rb =!= 0.0R)
          (ensures
            (s_rotg_c ra rb *. ra) +. (s_rotg_s ra rb *. rb) == s_rotg_r ra rb /\
            (s_rotg_c ra rb *. rb) -. (s_rotg_s ra rb *. ra) == 0.0R)

(* Generic typed signature: given a, b whose real radius is non-zero, the
   floating-point outputs approximate the exact real Givens parameters. *)
inline_for_extraction noextract
type rotg_ty (et:Type0) {| floating et, real_like et, floating_real_like et |} =
  a:et -> b:et ->
  Pure (rotg_out et)
    (requires s_rotg_r (to_real a) (to_real b) =!= 0.0R)
    (ensures fun o ->
       o.rr %~ s_rotg_r (to_real a) (to_real b) /\
       o.rc %~ s_rotg_c (to_real a) (to_real b) /\
       o.rs %~ s_rotg_s (to_real a) (to_real b))

(* NOTE: no f16 instance. rotg is a pure HOST scalar computation (not a kernel),
   and half-precision arithmetic (__hmul/__hadd/__hdiv/hsqrt) is __device__-only,
   so a host rotg_f16 fails to compile (calling a __device__ function from a
   __host__ one). cuBLAS rotg is S/D/C/Z anyway -- there is no half rotg. *)
val rotg_f32 : rotg_ty f32
val rotg_f64 : rotg_ty f64
