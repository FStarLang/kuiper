module Kuiper.Spec.SigmoidOptimized

module F = Kuiper.Float32.Exact
module U = FStar.UInt32

(* Exact binary32 encodings of the decimal constants in optimized_s.cu. *)
inline_for_extraction noextract
let c5 = 0xb6113926ul  // -2.1639948e-6f
inline_for_extraction noextract
let c4 = 0x37b319baul  //  2.1350443e-5f
inline_for_extraction noextract
let c3 = 0xb95d0dd1ul  // -2.1081349e-4f
inline_for_extraction noextract
let c2 = 0x3b088888ul  //  2.0833333e-3f
inline_for_extraction noextract
let c1 = 0xbcaaaaabul  // -2.0833334e-2f
inline_for_extraction noextract
let c0 = 0x3e800000ul  //  2.5e-1f
inline_for_extraction noextract
let half = 0x3f000000ul
inline_for_extraction noextract
let one = 0x3f800000ul
inline_for_extraction noextract
let minus_one = 0xbf800000ul

(* No numerical assumptions: this is the actual bit transformation. It
   preserves the sign of zero and leaves normals, infinities and NaNs alone. *)
let flush_bits (b : U.t) : Tot U.t =
  if U.eq (U.logand b 0x7f800000ul) 0ul then U.logand b 0x80000000ul else b

(* A bit-level postcondition, including signed zeros, infinities and NaNs:
   exponent != 0, or all magnitude bits are zero. *)
let is_flushed (b : U.t) : prop =
  not (U.logand b 0x7f800000ul == 0ul) \/ U.logand b 0x7ffffffful == 0ul

let sign_mask_properties (b : U.t) : Lemma
  (ensures U.logand (U.logand b 0x80000000ul) 0x7ffffffful == 0ul /\
           U.logand (U.logand b 0x80000000ul) 0x7f800000ul == 0ul /\
           U.logand (U.logand b 0x80000000ul) 0x80000000ul == U.logand b 0x80000000ul)
=
  let open FStar.UInt in
  assert_norm (Prims.pow2 31 == 0x80000000);
  logand_mask #32 0x80000000 31;
  logand_mask #32 0x7f800000 31;
  assert (logand #32 0x80000000 0x7fffffff == 0);
  assert (logand #32 0x7f800000 0x7fffffff == 0x7f800000);
  logand_commutative #32 0x7f800000 0x7fffffff;
  logand_associative #32 0x80000000 0x7fffffff 0x7f800000;
  logand_commutative #32 0 0x7f800000;
  logand_lemma_1 #32 0x7f800000;
  assert (logand #32 0x80000000 0x7f800000 == 0);
  logand_associative #32 (U.v b) 0x80000000 0x7fffffff;
  logand_associative #32 (U.v b) 0x80000000 0x7f800000;
  logand_associative #32 (U.v b) 0x80000000 0x80000000;
  logand_lemma_1 #32 (U.v b);
  logand_self #32 0x80000000

(* These are proved from integer bit operations, with no new float axioms. *)
let flush_properties (b : U.t) : Lemma
  (ensures is_flushed (flush_bits b) /\
           U.logand (flush_bits b) 0x80000000ul == U.logand b 0x80000000ul /\
           flush_bits (flush_bits b) == flush_bits b)
= sign_mask_properties b

(* The degree-11 odd polynomial around 1/2, evaluated in Horner order.
   Each FMA has one rounding step, followed by the explicit flush. *)
let polynomial_bits (x : U.t) : GTot U.t =
  let x2 = flush_bits (F.mul_rn_bits x x) in
  let p4 = flush_bits (F.fma_bits c5 x2 c4) in
  let p3 = flush_bits (F.fma_bits p4 x2 c3) in
  let p2 = flush_bits (F.fma_bits p3 x2 c2) in
  let p1 = flush_bits (F.fma_bits p2 x2 c1) in
  let p0 = flush_bits (F.fma_bits p1 x2 c0) in
  flush_bits (F.fma_bits x p0 half)

let fallback_bits (x : U.t) : GTot U.t =
  let e = flush_bits (F.exp_fast_bits (F.neg_bits x)) in
  let d = flush_bits (F.add_rn_bits one e) in
  flush_bits (F.div_fast_bits one d)

(* Inclusive [-1,1] test after the input flush. le_bits has CUDA comparison
   semantics, so NaNs take the fallback branch. No input is excluded. *)
let polynomial_branch (x : U.t) : GTot bool =
  F.le_bits minus_one x && F.le_bits x one

let sigmoid_bits (b : U.t) : GTot U.t =
  let x = flush_bits b in
  if polynomial_branch x then polynomial_bits x else fallback_bits x
