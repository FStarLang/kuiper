module KPlay.Proof.Base.SoftmaxScale

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost
open FStar.Real

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Literal = Kuiper.Float32.ConstantFold
module Cert = Kuiper.Float32.ConstantFold.Certificates
module S = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module Norm = KPlay.Proof.Base.Norm
module Math = KPlay.Proof.Base.AttentionAlgebra
module L2 = KPlay.Proof.Base.L2
module Transfer = KPlay.Proof.Base.Transfer
module Vocabulary = KPlay.Proof.Base.Vocabulary

inline_for_extraction noextract
let coefficient : FP.t = F.of_literal "0.0625"

inline_for_extraction noextract
let coefficient_approx ()
  : Lemma (Approx.v_approximates coefficient 0.0625R) =
  assert_norm (Cert.exact_interval "0.0625" 1 16 1 16 8388608 (-27));
  Literal.literal_approx "0.0625" 1 16 1 16 8388608 (-27) 0.0625R

inline_for_extraction noextract
let scaled (x:BF.t) : FP.t =
  F.add (L2.rounded (F.mul (Cast.cast_bf16_to_f32 x) coefficient)) (F.zero #FP.t)

inline_for_extraction noextract
let scaled_approx (x:BF.t) (r:real)
  : Lemma
      (requires Approx.v_approximates x r)
      (ensures Approx.v_approximates (scaled x) (r *. 0.0625R)) =
  coefficient_approx ();
  Cast.cast_bf16_to_f32_ok x r;
  Approx.a_mul (Cast.cast_bf16_to_f32 x) coefficient r 0.0625R;
  L2.rounded_approx (F.mul (Cast.cast_bf16_to_f32 x) coefficient) (r *. 0.0625R);
  Approx.a_add (L2.rounded (F.mul (Cast.cast_bf16_to_f32 x) coefficient)) (F.zero #FP.t)
    (r *. 0.0625R) 0.0R

inline_for_extraction noextract
let choose (before current:FP.t) : FP.t =
  if F.lt before current then current else before

inline_for_extraction noextract
let rec maximum (hidden:Transfer.scratch) (offset count:nat{offset + count <= 16000})
  : GTot FP.t (decreases count) =
  if count = 0 then Vocabulary.negative_infinity
  else choose (maximum hidden offset (count - 1)) (scaled (Seq.index hidden (offset + count - 1)))

inline_for_extraction noextract
let writes (before:lseq FP.t 2048) (hidden:Transfer.scratch)
  (offset:nat) (count:nat{count <= 647 /\ offset + count <= 16000}) : GTot (lseq FP.t 2048) =
  Seq.init_ghost 2048 (fun i -> if i < count then scaled (Seq.index hidden (offset + i)) else Seq.index before i)

inline_for_extraction noextract
let writes_zero (before:lseq FP.t 2048) (hidden:Transfer.scratch) (offset:nat{offset <= 16000})
  : Lemma (writes before hidden offset 0 == before) =
  assert (Seq.equal (writes before hidden offset 0) before);
  Seq.lemma_eq_elim (writes before hidden offset 0) before

inline_for_extraction noextract
let writes_step (before:lseq FP.t 2048) (hidden:Transfer.scratch)
  (offset:nat) (i:nat{i < 647 /\ offset + i < 16000})
  : Lemma (
      Seq.upd (writes before hidden offset i) i (scaled (Seq.index hidden (offset + i))) ==
      writes before hidden offset (i + 1)) =
  assert (Seq.equal
    (Seq.upd (writes before hidden offset i) i (scaled (Seq.index hidden (offset + i))))
    (writes before hidden offset (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (writes before hidden offset i) i (scaled (Seq.index hidden (offset + i))))
    (writes before hidden offset (i + 1))

inline_for_extraction noextract
let hidden_rep (n:pos{n <= 647}) (before:Transfer.scratch) (offset:nat{offset + n <= 16000})
  (logical:S.vector n) : prop =
  forall (i:S.index n). Approx.v_approximates (Seq.index before (offset + i)) (logical i)

inline_for_extraction noextract
let values_rep (n:pos{n <= 647}) (values:lseq FP.t 2048) (logical:S.vector n) : prop =
  forall (i:S.index n). Approx.v_approximates (Seq.index values i) (logical i)

inline_for_extraction noextract
let values_approx (scope:prop) (n:pos{n <= 647}) (before:lseq FP.t 2048) (hidden:Transfer.scratch)
  (offset:nat{offset + n <= 16000}) (logical:S.vector n)
  : Lemma
      (requires scope ==> hidden_rep n hidden offset logical)
      (ensures scope ==> values_rep n (writes before hidden offset n) (Math.scaled_scores n logical)) =
  let under (_:scope)
    : Lemma (values_rep n (writes before hidden offset n) (Math.scaled_scores n logical)) =
    let at_index (i:S.index n)
      : Lemma (Approx.v_approximates (Seq.index (writes before hidden offset n) i)
          (Math.scaled_scores n logical i)) =
      scaled_approx (Seq.index hidden (offset + i)) (logical i)
    in
    FStar.Classical.forall_intro at_index
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
fn scale_and_maximum
  (hidden:larray BF.t 16000) (values:larray FP.t 2048)
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647}) (offset:sz{SZ.v offset + SZ.v n <= 16000})
  (#sx:erased Transfer.scratch) (#sv:erased (lseq FP.t 2048)) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction sx
  requires values |-> sv
  returns result:FP.t
  ensures values |-> writes sv sx (SZ.v offset) (SZ.v n)
  ensures pure (result == maximum sx (SZ.v offset) (SZ.v n))
{
  writes_zero sv sx (SZ.v offset);
  let mut best:FP.t = Vocabulary.negative_infinity;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{SZ.v j <= SZ.v n}) (current:FP.t).
      i |-> j ** best |-> current ** values |-> writes sv sx (SZ.v offset) (SZ.v j) **
      pure (current == maximum sx (SZ.v offset) (SZ.v j))
    invariant gpu ** A.pts_to hidden #fraction sx
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    let input = hidden.(SZ.add offset j);
    let value = scaled input;
    writes_step sv sx (SZ.v offset) (SZ.v j);
    values.(j) <- value;
    let old = !best;
    norm_spec [delta_once ["KPlay.Proof.Base.SoftmaxScale.maximum"]]
      (maximum sx (SZ.v offset) (SZ.v j + 1));
    if (F.lt old value) { best := value; };
    i := SZ.add j 1sz;
  };
  !best
}

inline_for_extraction noextract
fn source_width (n:sz{0 < SZ.v n /\ SZ.v n <= 647})
  preserves gpu
  returns result:sz
  ensures pure (
    0 < SZ.v result /\ Norm.tree_size (SZ.v result) /\ SZ.v result <= 32 /\
    (SZ.v n <= SZ.v result \/ SZ.v result = 32))
{
  let mut width = 1sz;
  while (!width <^ n && !width <^ 32sz)
    invariant exists* (w:sz{0 < SZ.v w /\ Norm.tree_size (SZ.v w) /\ SZ.v w <= 32}).
      width |-> w
    invariant gpu
    decreases (32 - SZ.v !width)
  {
    let w = !width;
    width := SZ.mul w 2sz;
  };
  !width
}
