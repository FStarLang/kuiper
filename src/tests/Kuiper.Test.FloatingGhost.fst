module Kuiper.Test.FloatingGhost

#lang-pulse
open Kuiper
module A = Kuiper.Approximates.Base
module F = Kuiper.Floating.Base
module FP = Kuiper.Float32
module Fast = Kuiper.Float32.FastMath
module RO = Kuiper.Float.Realops

#set-options "--fuel 3 --ifuel 1 --z3rlimit 10"

(* The original recursive proof must fail at semantic equality, not a missing
   eq_approx name or a reference to another rejected lemma. *)
[@@expect_failure [19]]
let rec infinity_down (p:real) (n:nat)
  : Lemma
      (requires A.v_approximates (F.infinity #FP.t) p)
      (ensures A.v_approximates (F.infinity #FP.t)
        (p -. FStar.Real.of_int n))
      (decreases n)
  = if n = 0 then ()
    else (
      infinity_down p (n - 1);
      let _ = A.a1 #FP.t in
      let difference = Fast.pure_sub_rn (F.infinity #FP.t) F.one in
      assert (difference == (F.infinity #FP.t));
      assert (A.v_approximates difference
        ((p -. FStar.Real.of_int (n - 1)) -. 1.0R)))

(* Zero is not ruled out as infinity's chosen ideal, but cannot be forced. *)
[@@expect_failure [19]]
let infinity_approximates_zero ()
  : Lemma (A.v_approximates (F.infinity #FP.t) 0.0R)
  = A.to_real_ok (F.infinity #FP.t)

(* Even granting the preceding claim, zero bits do not identify canonical zero. *)
[@@expect_failure [19]]
let zero_approximates_one ()
  : Lemma
      (requires A.v_approximates (F.infinity #FP.t) 0.0R)
      (ensures A.v_approximates (F.zero #FP.t) 1.0R)
  = let _ = A.approx_neg #FP.t in
    A.approx_apply (F.neg #FP.t) RO.r_neg F.infinity 0.0R;
    let _ = A.approx_fexp #FP.t in
    A.approx_apply (F.fexp #FP.t) exp (F.neg F.infinity) 0.0R;
    F.exp_special_values (F.neg (F.infinity #FP.t));
    exp_base ()

inline_for_extraction noextract
let constant_zero (_:FP.t) : FP.t = F.zero

[@@expect_failure [19]]
let one_approximates_zero ()
  : Lemma (A.v_approximates (F.one #FP.t) 0.0R) = ()

[@@expect_failure [19]]
let finite_fact_transfers_to_zero (x:FP.t) (r:real)
  : Lemma
      (requires Finite? (F.kind x) /\ A.v_approximates x r)
      (ensures A.v_approximates (F.zero #FP.t) r)
  = A.to_real_ok x;
    let difference = Fast.pure_sub_rn x x in
    assert (difference == (F.zero #FP.t));
    assert (A.v_approximates difference (r -. 0.0R))

[@@expect_failure [19]]
let negative_control ()
  : Lemma
      (A.v_approximates (constant_zero (F.one #FP.t)) 1.0R /\
       not (F.eq (constant_zero (F.one #FP.t)) F.one))
  = F.eq_spec (F.zero #FP.t) F.one

[@@expect_failure [19]]
inline_for_extraction noextract
fn constant_zero_with_identity_contract (x:FP.t)
  requires gpu ** pure (Finite? (F.kind x))
  returns result:FP.t
  ensures gpu ** pure (forall (r:real).
    A.v_approximates x r ==> A.v_approximates result r)
{
  F.zero #FP.t
}
