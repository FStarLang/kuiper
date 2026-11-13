module KPlay.Proof.Base.AttentionRotation

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost
open FStar.Real

module Array = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Neg = Kuiper.Float32.NativeNeg
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module L2 = KPlay.Proof.Base.L2
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let head_offset (head:P.index 10) : (offset:nat{offset + 256 <= 16000}) =
  if head < 8 then 2048 + head * 256 else 6144 + (head - 8) * 256

inline_for_extraction noextract
let pair (x y cosine sine:BF.t) (upper:bool) : BF.t =
  let xf = Cast.cast_bf16_to_f32 x in
  let yf = Cast.cast_bf16_to_f32 y in
  let c = Cast.cast_bf16_to_f32 cosine in
  let s = Cast.cast_bf16_to_f32 sine in
  if upper then Cast.cast_f32_to_bf16
    (F.add (L2.rounded (F.mul yf c)) (L2.rounded (F.mul xf s)))
  else Cast.cast_f32_to_bf16
    (F.add (L2.rounded (F.mul xf c)) (L2.rounded (F.mul (Neg.neg yf) s)))

inline_for_extraction noextract
let pair_approx (x y cosine sine:BF.t) (xr yr cr sr:real) (upper:bool)
  : Lemma
      (requires Approx.v_approximates x xr /\ Approx.v_approximates y yr /\
        Approx.v_approximates cosine cr /\ Approx.v_approximates sine sr)
      (ensures Approx.v_approximates (pair x y cosine sine upper)
        (if upper then yr *. cr +. xr *. sr else xr *. cr +. (0.0R -. yr) *. sr)) =
  Cast.cast_bf16_to_f32_ok x xr;
  Cast.cast_bf16_to_f32_ok y yr;
  Cast.cast_bf16_to_f32_ok cosine cr;
  Cast.cast_bf16_to_f32_ok sine sr;
  let xf = Cast.cast_bf16_to_f32 x in
  let yf = Cast.cast_bf16_to_f32 y in
  let c = Cast.cast_bf16_to_f32 cosine in
  let s = Cast.cast_bf16_to_f32 sine in
  if upper then (
    Approx.a_mul yf c yr cr;
    Approx.a_mul xf s xr sr;
    L2.rounded_approx (F.mul yf c) (yr *. cr);
    L2.rounded_approx (F.mul xf s) (xr *. sr);
    Approx.a_add (L2.rounded (F.mul yf c)) (L2.rounded (F.mul xf s)) (yr *. cr) (xr *. sr);
    Cast.cast_f32_to_bf16_ok (F.add (L2.rounded (F.mul yf c)) (L2.rounded (F.mul xf s))) (yr *. cr +. xr *. sr)
  ) else (
    Neg.neg_approx yf yr;
    Approx.a_mul xf c xr cr;
    Approx.a_mul (Neg.neg yf) s (0.0R -. yr) sr;
    L2.rounded_approx (F.mul xf c) (xr *. cr);
    L2.rounded_approx (F.mul (Neg.neg yf) s) ((0.0R -. yr) *. sr);
    Approx.a_add (L2.rounded (F.mul xf c)) (L2.rounded (F.mul (Neg.neg yf) s)) (xr *. cr) ((0.0R -. yr) *. sr);
    Cast.cast_f32_to_bf16_ok (F.add (L2.rounded (F.mul xf c)) (L2.rounded (F.mul (Neg.neg yf) s)))
      (xr *. cr +. (0.0R -. yr) *. sr)
  )

inline_for_extraction noextract
let value (before:Transfer.scratch) (head:P.index 10) (i:P.index 32) (upper:bool) : GTot BF.t =
  pair (Seq.index before (head_offset head + i)) (Seq.index before (head_offset head + i + 32))
    (Seq.index before (15600 + i)) (Seq.index before (15632 + i)) upper

inline_for_extraction noextract
let head_after (before:Transfer.scratch) (head:P.index 10) (count:nat{count <= 32}) : GTot Transfer.scratch =
  let offset = head_offset head in
  Seq.init_ghost 16000 (fun i ->
    if offset <= i && i < offset + 64 && (i - offset) % 32 < count
    then value before head ((i - offset) % 32) (i >= offset + 32)
    else Seq.index before i)

inline_for_extraction noextract
let head_zero (before:Transfer.scratch) (head:P.index 10)
  : Lemma (head_after before head 0 == before) =
  assert (Seq.equal (head_after before head 0) before);
  Seq.lemma_eq_elim (head_after before head 0) before

inline_for_extraction noextract
let head_step (before:Transfer.scratch) (head:P.index 10) (i:P.index 32)
  : Lemma (
      Seq.index (head_after before head i) (head_offset head + i) == Seq.index before (head_offset head + i) /\
      Seq.index (head_after before head i) (head_offset head + i + 32) == Seq.index before (head_offset head + i + 32) /\
      Seq.index (head_after before head i) (15600 + i) == Seq.index before (15600 + i) /\
      Seq.index (head_after before head i) (15632 + i) == Seq.index before (15632 + i) /\
      Seq.upd (Seq.upd (head_after before head i) (head_offset head + i) (value before head i false))
        (head_offset head + i + 32) (value before head i true) ==
      head_after before head (i + 1)) =
  assert (Seq.equal
    (Seq.upd (Seq.upd (head_after before head i) (head_offset head + i) (value before head i false))
      (head_offset head + i + 32) (value before head i true))
    (head_after before head (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (Seq.upd (head_after before head i) (head_offset head + i) (value before head i false))
      (head_offset head + i + 32) (value before head i true))
    (head_after before head (i + 1))

inline_for_extraction noextract
fn apply_head
  (hidden:larray BF.t 16000) (head:sz{SZ.v head < 10}) (#before:erased Transfer.scratch)
  preserves gpu
  requires Array.pts_to hidden before
  ensures Array.pts_to hidden (head_after before (SZ.v head) 32)
{
  head_zero before (SZ.v head);
  let offset = if (head <^ 8sz) { SZ.add 2048sz (SZ.mul head 256sz) }
    else { SZ.add 6144sz (SZ.mul (head -^ 8sz) 256sz) };
  let mut i = 0sz;
  while (!i <^ 32sz)
    invariant exists* (j:sz{SZ.v j <= 32}).
      i |-> j ** Array.pts_to hidden (head_after before (SZ.v head) (SZ.v j))
    invariant gpu
    decreases (32 - SZ.v !i)
  {
    let j = !i;
    head_step before (SZ.v head) (SZ.v j);
    let x = hidden.(SZ.add offset j);
    let y = hidden.(SZ.add (SZ.add offset j) 32sz);
    let cosine = hidden.(SZ.add 15600sz j);
    let sine = hidden.(SZ.add 15632sz j);
    let xf = Cast.cast_bf16_to_f32 x;
    let yf = Cast.cast_bf16_to_f32 y;
    let c = Cast.cast_bf16_to_f32 cosine;
    let s = Cast.cast_bf16_to_f32 sine;
    let xc = L2.rounded (F.mul xf c);
    let ys = L2.rounded (F.mul (Neg.neg yf) s);
    let yc = L2.rounded (F.mul yf c);
    let xs = L2.rounded (F.mul xf s);
    hidden.(SZ.add offset j) <- Cast.cast_f32_to_bf16 (F.add xc ys);
    hidden.(SZ.add (SZ.add offset j) 32sz) <- Cast.cast_f32_to_bf16 (F.add yc xs);
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
let rec after (before:Transfer.scratch) (count:nat{count <= 10}) : GTot Transfer.scratch (decreases count) =
  if count = 0 then before else head_after (after before (count - 1)) (count - 1) 32

inline_for_extraction noextract
fn apply
  (hidden:larray BF.t 16000) (#before:erased Transfer.scratch)
  preserves gpu
  requires Array.pts_to hidden before
  ensures Array.pts_to hidden (after before 10)
{
  let mut head = 0sz;
  while (!head <^ 10sz)
    invariant exists* (h:sz{SZ.v h <= 10}).
      head |-> h ** Array.pts_to hidden (after before (SZ.v h))
    invariant gpu
    decreases (10 - SZ.v !head)
  {
    let h = !head;
    apply_head hidden h;
    norm_spec [delta_once ["KPlay.Proof.Base.AttentionRotation.after"]] (after before (SZ.v h + 1));
    head := SZ.add h 1sz;
  }
}

inline_for_extraction noextract
let head_approx (before:Transfer.scratch) (head:P.index 10) (position:nat) (input:A.vector 256) (i:P.index 256)
  : Lemma
      (requires
        (forall (d:P.index 256). Approx.v_approximates (Seq.index before (head_offset head + d)) (input d)) /\
        (forall (j:P.index 32).
          Approx.v_approximates (Seq.index before (15600 + j)) (A.rope_cos position j) /\
          Approx.v_approximates (Seq.index before (15632 + j)) (A.rope_sin position j)))
      (ensures Approx.v_approximates (Seq.index (head_after before head 32) (head_offset head + i))
        (A.rotate_partial position input i)) =
  if i < 64 then (
    let j = i % 32 in
    pair_approx (Seq.index before (head_offset head + j)) (Seq.index before (head_offset head + j + 32))
      (Seq.index before (15600 + j)) (Seq.index before (15632 + j))
      (input j) (input (j + 32)) (A.rope_cos position j) (A.rope_sin position j) (i >= 32)
  )

inline_for_extraction noextract
let head_frame (before:Transfer.scratch) (head:P.index 10) (i:P.index 16000)
  : Lemma (
      ~(head_offset head <= i /\ i < head_offset head + 64) ==>
      Seq.index (head_after before head 32) i == Seq.index before i) = ()

inline_for_extraction noextract
let rec rotary_frame (before:Transfer.scratch) (count:nat{count <= 10}) (i:P.index 32)
  : Lemma (
      Seq.index (after before count) (15600 + i) == Seq.index before (15600 + i) /\
      Seq.index (after before count) (15632 + i) == Seq.index before (15632 + i))
      (decreases count) =
  if count > 0 then (
    rotary_frame before (count - 1) i;
    head_frame (after before (count - 1)) (count - 1) (15600 + i);
    head_frame (after before (count - 1)) (count - 1) (15632 + i);
    norm_spec [delta_once ["KPlay.Proof.Base.AttentionRotation.after"]] (after before count)
  )

inline_for_extraction noextract
let disjoint (left right:P.index 10{left <> right}) (i:P.index 256)
  : Lemma (
      head_offset left + i < head_offset right \/ head_offset right + 256 <= head_offset left + i) = ()

inline_for_extraction noextract
let rec unread_head (before:Transfer.scratch) (count:nat{count <= 10})
  (head:P.index 10{count <= head}) (i:P.index 256)
  : Lemma (
      Seq.index (after before count) (head_offset head + i) ==
      Seq.index before (head_offset head + i))
      (decreases count) =
  if count > 0 then (
    unread_head before (count - 1) head i;
    disjoint head (count - 1) i;
    head_frame (after before (count - 1)) (count - 1) (head_offset head + i);
    norm_spec [delta_once ["KPlay.Proof.Base.AttentionRotation.after"]] (after before count)
  )

inline_for_extraction noextract
let rec processed_head (before:Transfer.scratch) (count:pos{count <= 10})
  (head:nat{head < count}) (i:P.index 256)
  : Lemma (
      Seq.index (after before count) (head_offset head + i) ==
      Seq.index (head_after before head 32) (head_offset head + i))
      (decreases count) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionRotation.after"]] (after before count);
  if head < count - 1 then (
    processed_head before (count - 1) head i;
    disjoint head (count - 1) i;
    head_frame (after before (count - 1)) (count - 1) (head_offset head + i)
  ) else (
    unread_head before head head i;
    if i < 64 then (
      let j = i % 32 in
      unread_head before head head j;
      unread_head before head head (j + 32);
      rotary_frame before head j
    )
  )

inline_for_extraction noextract
let complete_head_approx (before:Transfer.scratch) (head:P.index 10)
  (position:nat) (input:A.vector 256) (i:P.index 256)
  : Lemma
      (requires
        (forall (d:P.index 256). Approx.v_approximates (Seq.index before (head_offset head + d)) (input d)) /\
        (forall (j:P.index 32).
          Approx.v_approximates (Seq.index before (15600 + j)) (A.rope_cos position j) /\
          Approx.v_approximates (Seq.index before (15632 + j)) (A.rope_sin position j)))
      (ensures Approx.v_approximates (Seq.index (after before 10) (head_offset head + i))
        (A.rotate_partial position input i)) =
  processed_head before 10 head i;
  head_approx before head position input i

inline_for_extraction noextract
let rec outside_frame (before:Transfer.scratch) (count:nat{count <= 10}) (i:P.index 16000)
  : Lemma
      (requires ~(2048 <= i /\ i < 4096) /\ ~(6144 <= i /\ i < 6656))
      (ensures Seq.index (after before count) i == Seq.index before i)
      (decreases count) =
  if count > 0 then (
    outside_frame before (count - 1) i;
    head_frame (after before (count - 1)) (count - 1) i;
    norm_spec [delta_once ["KPlay.Proof.Base.AttentionRotation.after"]] (after before count)
  )
