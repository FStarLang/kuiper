module Kuiper.Test.TensorCoreApprox

open Kuiper
open Kuiper.Chest
open Kuiper.Spec.GEMM
open Kuiper.TensorCore.Base

[@@expect_failure [19]]
let axioms_do_not_prove_false () : Lemma False = ()

let supported_mma
  (c : chest2 f32 16 16)
  (a b : chest2 f16 16 16)
  (rc ra rb : chest2 real 16 16)
  : Lemma (requires c %~ rc /\ a %~ ra /\ b %~ rb)
          (ensures emma c a b %~ matplus rc (matmul ra rb))
  = emma_approx_lemma c a b rc ra rb

[@@expect_failure [228]]
let no_modular_inputs
  (c : chest2 f32 1 1)
  (a b : chest2 u8 1 1)
  (rc ra rb : chest2 real 1 1)
  : Lemma (requires c %~ rc /\ a %~ ra /\ b %~ rb)
          (ensures emma c a b %~ matplus rc (matmul ra rb))
  = emma_approx_lemma c a b rc ra rb

[@@expect_failure [228]]
let no_arbitrary_float_interpretation
  {| source : real_like f16 |}
  (c : chest2 f32 16 16)
  (a b : chest2 f16 16 16)
  (rc ra rb : chest2 real 16 16)
  : Lemma (requires c %~ rc /\ a %~ ra /\ b %~ rb)
          (ensures emma c a b %~ matplus rc (matmul ra rb))
  = emma_approx_lemma c a b rc ra rb
