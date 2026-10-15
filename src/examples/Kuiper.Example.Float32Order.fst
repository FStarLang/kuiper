module Kuiper.Example.Float32Order

#lang-pulse

open Kuiper
module F = Kuiper.Floating
module F32 = Kuiper.Float32

let comparison_excludes_nan (x y : f32)
  : Lemma
      (requires lt x y)
      (ensures not (F.is_nan x) /\ not (F.is_nan y))
  = F32.lt_ordered x y

let comparison_is_transitive (x y z : f32)
  : Lemma
      (requires lt x y /\ lt y z)
      (ensures lt x z)
  = F32.lt_transitive x y z

let no_strict_cycle (x y z : f32)
  : Lemma
      (requires lt x y /\ lt y z /\ lt z x)
      (ensures False)
  = F32.lt_transitive x y z;
    F32.lt_transitive x z x;
    F32.lt_ordered x x;
    F.eq_spec x x;
    F.lte_is_lt_or_eq x x;
    F.negate_lt_is_lte x x

[@@expect_failure [19]]
let unordered_values_need_not_be_comparable (x y : f32)
  : Lemma (lt x y \/ F.ieee_eq x y \/ lt y x)
  = ()

[@@expect_failure [19]]
let numerical_equality_is_not_representation_equality (x y : f32)
  : Lemma
      (requires F.ieee_eq x y)
      (ensures F.bit_eq x y)
  = F.eq_spec x y;
    F.bit_eq_spec x y

[@@CPrologue "inline"; CPrologue "__host__"; CPrologue "__device__"]
let compare (x y : f32) : bool = lt x y
