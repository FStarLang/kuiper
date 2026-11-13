module KPlay.Proof.Optimized.Decay

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 50"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module FP = Kuiper.Float32
module Neg = Kuiper.Float32.NativeNeg
module Cast = Kuiper.Float.Casts.Base
module SZ = Kuiper.SizeT
module C = FStar.Classical
module Scalar = KPlay.Proof.Optimized.Scalar
module Softplus = Luce.Softplus.Fast
module Model = KPlay.Proof.Shared.Model
module R = KPlay.Proof.Shared.Recurrent
module Exp = FStar.Math.Exp

inline_for_extraction noextract
let interpretation = lseq real 21555360 & real

inline_for_extraction noextract
let inputs (w:lseq bf16 21555360) (alpha:f32) (m:interpretation) : prop =
  w %~ fst m /\ alpha %~ snd m

inline_for_extraction noextract
let model (head:nat{head < 16}) (m:interpretation) : GTot real =
  Exp.exp ((0.0R -. Exp.exp (S.index (fst m) (19458048 + head))) *.
    R.softplus (snd m +. S.index (fst m) (19458064 + head)))

let negative_exponential (w:lseq bf16 21555360) (alpha exponential:f32)
  (head:nat{head < 16})
  : Lemma
      (requires forall (m:interpretation). inputs w alpha m ==>
        exponential %~ Exp.exp (S.index (fst m) (19458048 + head)))
      (ensures forall (m:interpretation). inputs w alpha m ==>
        Neg.neg exponential %~ (0.0R -. Exp.exp (S.index (fst m) (19458048 + head)))) =
  let related (m:interpretation{inputs w alpha m})
    : Lemma (Neg.neg exponential %~
        (0.0R -. Exp.exp (S.index (fst m) (19458048 + head)))) =
    Neg.neg_approx exponential (Exp.exp (S.index (fst m) (19458048 + head)))
  in C.forall_intro related

noextract
fn decay (weights:larray bf16 21555360) (head:sz{head < 16}) (alpha:f32)
  (#wv:erased (lseq bf16 21555360)) (#fw:perm)
  preserves gpu ** weights |-> Frac fw wv
  returns result:f32
  ensures pure (forall (m:interpretation). inputs wv alpha m ==>
    result %~ model (SZ.v head) m)
{
  let bias = Cast.cast_bf16_to_f32 (weights.(SZ.uint32_to_sizet 19458064ul +^ head));
  let x = FP.add_rn_ftz alpha bias;
  lift_add #interpretation (inputs wv alpha) (fun m -> snd m)
    (fun m -> S.index (fst m) (19458064 + SZ.v head)) alpha bias x;
  let sp = Softplus.softplus20 x;
  C.forall_intro Model.authorized_softplus_model;
  let a = Cast.cast_bf16_to_f32 (weights.(SZ.uint32_to_sizet 19458048ul +^ head));
  let exponential = Scalar.exp32 a;
  negative_exponential wv alpha exponential (SZ.v head);
  let negative = Neg.neg exponential;
  let exponent = FP.mul_rn_ftz negative sp;
  lift_mul #interpretation (inputs wv alpha)
    (fun m -> 0.0R -. Exp.exp (S.index (fst m) (19458048 + SZ.v head)))
    (fun m -> R.softplus (snd m +. S.index (fst m) (19458064 + SZ.v head)))
    negative sp exponent;
  Scalar.exp32 exponent
}
