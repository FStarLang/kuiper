module KPlay.Proof.Base.Rotary

#lang-pulse

open Kuiper
open Kuiper.Float.Casts
open FStar.Ghost

module S = FStar.Seq
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module Array = Pulse.Lib.Array
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Attention = KPlay.Proof.Shared.Attention
module Recurrent = KPlay.Proof.Shared.Recurrent
module Model = KPlay.Proof.Shared.Model
module Source = KPlay.Proof.RoPE.Source
module Consumer = KPlay.Proof.RoPE.Consumer

inline_for_extraction noextract
let snapshot = lseq (option bf16) 16000

inline_for_extraction noextract
let rotary_mask (i:nat) : prop = 15600 <= i /\ i < 15664

inline_for_extraction noextract
let source_position (call:Model.invocation) (p:nat{p < Model.calls call})
  : Source.position =
  Model.full_profile call;
  I64.int_to_t p

inline_for_extraction noextract
let source_position_value (call:Model.invocation) (p:nat{p < Model.calls call})
  : Lemma (I64.v (source_position call p) == p /\ p < 647) =
  Model.full_profile call

(* The source writes cosine then sine at each index, not two staged passes. *)
inline_for_extraction noextract
let rec source_writes (before:snapshot) (p:Source.position)
  (count:nat{count <= 32}) : Tot snapshot (decreases count) =
  if count = 0 then before
  else
    let i = count - 1 in
    let previous = source_writes before p i in
    let cosine = S.upd previous (15600 + i) (Some (Source.rotary_component p i)) in
    S.upd cosine (15632 + i) (Some (Source.rotary_component p (32 + i)))

inline_for_extraction noextract
let rec source_write_cell (before:snapshot) (p:Source.position)
  (count:nat{count <= 32}) (i:nat{i < 16000})
  : Lemma
      (ensures S.index (source_writes before p count) i ==
        (if 15600 <= i && i < 15600 + count then
           Some (Source.rotary_component p (i - 15600))
         else if 15632 <= i && i < 15632 + count then
           Some (Source.rotary_component p (32 + i - 15632))
         else S.index before i))
      (decreases count) =
  if count > 0 then source_write_cell before p (count - 1) i

inline_for_extraction noextract
let initialized (stored:snapshot) : prop =
  forall (d:nat{d < 64}). Some? (S.index stored (15600 + d))

inline_for_extraction noextract
let source_initialized (before:snapshot) (p:Source.position)
  : Lemma (initialized (source_writes before p 32)) =
  let at_cell (d:nat{d < 64})
    : Lemma (Some? (S.index (source_writes before p 32) (15600 + d))) =
    source_write_cell before p 32 (15600 + d)
  in
  FStar.Classical.forall_intro at_cell

inline_for_extraction noextract
let observed_cells (stored:snapshot{initialized stored}) : GTot (lseq bf16 64) =
  S.init_ghost 64 (fun d -> Recurrent.present_value (S.index stored (15600 + d)))

inline_for_extraction noextract
let produced_cells (before:snapshot) (p:Source.position) : GTot (lseq bf16 64) =
  source_initialized before p;
  observed_cells (source_writes before p 32)

(* The accepted full-attention caller predicate, with only namespace/type aliases. *)
inline_for_extraction noextract
let rope_represents (position:nat) (storage:lseq bf16 64) : prop =
  (forall (j:Attention.index 32).
    Approx.v_approximates (S.index storage j) (Attention.rope_cos position j)) /\
  (forall (j:Attention.index 32).
    Approx.v_approximates (S.index storage (32 + j)) (Attention.rope_sin position j))

inline_for_extraction noextract
let source_post (before:snapshot) (p:Source.position)
  : Lemma (Source.rotary_post p (produced_cells before p)) =
  let at_cell (d:nat{d < 64})
    : Lemma (S.index (produced_cells before p) d == Source.rotary_component p d) =
    source_write_cell before p 32 (15600 + d)
  in
  FStar.Classical.forall_intro at_cell

inline_for_extraction noextract
let consumed_pair (before:snapshot) (p:Source.position) (j:nat{j < 32})
  : GTot (f32 & f32) =
  (Cast.cast_bf16_to_f32 (S.index (produced_cells before p) j),
   Cast.cast_bf16_to_f32 (S.index (produced_cells before p) (32 + j)))

inline_for_extraction noextract
let source_connection (before:snapshot) (p:Source.position)
  : Lemma (
      Source.rotary_post p (produced_cells before p) /\
      rope_represents (I64.v p) (produced_cells before p) /\
      (forall (j:nat{j < 32}).
        let cosine, sine = consumed_pair before p j in
        Approx.v_approximates cosine (Attention.rope_cos (I64.v p) j) /\
        Approx.v_approximates sine (Attention.rope_sin (I64.v p) j)) /\
      (forall (i:nat{i < 16000}).
        ~(rotary_mask i) ==>
        S.index (source_writes before p 32) i == S.index before i)) =
  source_post before p;
  let at_frequency (j:nat{j < 32})
    : Lemma (
        Approx.v_approximates (S.index (produced_cells before p) j)
          (Attention.rope_cos (I64.v p) j) /\
        Approx.v_approximates (S.index (produced_cells before p) (32 + j))
          (Attention.rope_sin (I64.v p) j) /\
        (let cosine, sine = consumed_pair before p j in
          Approx.v_approximates cosine (Attention.rope_cos (I64.v p) j) /\
          Approx.v_approximates sine (Attention.rope_sin (I64.v p) j))) =
    Consumer.producer_consumer p (produced_cells before p) j;
    Cast.cast_bf16_to_f32_ok (S.index (produced_cells before p) j)
      (Attention.rope_cos (I64.v p) j);
    Cast.cast_bf16_to_f32_ok (S.index (produced_cells before p) (32 + j))
      (Attention.rope_sin (I64.v p) j)
  in
  FStar.Classical.forall_intro at_frequency;
  let outside (i:nat{i < 16000})
    : Lemma (
        ~(rotary_mask i) ==>
        S.index (source_writes before p 32) i == S.index before i) =
    source_write_cell before p 32 i
  in
  FStar.Classical.forall_intro outside

inline_for_extraction noextract
let earlier_write_preserves (before:snapshot) (p:Source.position)
  (address:nat{address < 15600}) (value:bf16)
  : Lemma (
      forall (d:nat{d < 64}).
        S.index (S.upd (source_writes before p 32) address (Some value)) (15600 + d) ==
        S.index (source_writes before p 32) (15600 + d)) = ()

(* Only the 64 destination cells are owned; None is allowed on entry.
   No untouched scratch cell is read or asserted initialized. *)
inline_for_extraction noextract
fn position_embedding (hidden:larray bf16 16000) (p:Source.position)
  (#before:erased snapshot)
  preserves gpu
  requires Array.pts_to_mask hidden before rotary_mask
  ensures Array.pts_to_mask hidden (source_writes before p 32) rotary_mask
  ensures pure (
    Source.rotary_post p (produced_cells before p) /\
    rope_represents (I64.v p) (produced_cells before p))
{
  let mut i = 0sz;
  while (!i <^ 32sz)
    invariant exists* (current:sz{SZ.v current <= 32}).
      i |-> current **
      Array.pts_to_mask hidden (source_writes before p (SZ.v current)) rotary_mask
    invariant gpu
    decreases (32 - SZ.v !i)
  {
    let current = !i;
    let angle = mul (Source.inverse_frequency (SZ.v current)) (of_int p);
    let cosine:bf16 = fcast (cos angle);
    Array.mask_write hidden (SZ.add 15600sz current) cosine;
    let sine:bf16 = fcast (sin angle);
    Array.mask_write hidden (SZ.add 15632sz current) sine;
    assert pure (
      S.upd
        (S.upd (source_writes before p (SZ.v current))
          (15600 + SZ.v current) (Some cosine))
        (15632 + SZ.v current) (Some sine) ==
      source_writes before p (SZ.v current + 1));
    i := SZ.add current 1sz;
  };
  source_connection before p;
  assert pure (Source.rotary_post p (produced_cells before p));
  assert pure (rope_represents (I64.v p) (produced_cells before p));
}

inline_for_extraction noextract
fn read_source_pair
  (hidden:larray bf16 16000) (p:Source.position) (j:sz{SZ.v j < 32})
  (#before:snapshot) (#fraction:perm)
  preserves gpu
  preserves Array.pts_to_mask hidden #fraction (source_writes before p 32) rotary_mask
  returns result:(f32 & f32)
  ensures pure (
    result == consumed_pair before p (SZ.v j) /\
    (let cosine, sine = result in
      Approx.v_approximates cosine (Attention.rope_cos (I64.v p) (SZ.v j)) /\
      Approx.v_approximates sine (Attention.rope_sin (I64.v p) (SZ.v j))))
{
  source_write_cell before p 32 (15600 + SZ.v j);
  source_write_cell before p 32 (15632 + SZ.v j);
  let cosine_cell = Array.mask_read hidden (SZ.add 15600sz j);
  let cosine = Cast.cast_bf16_to_f32 cosine_cell;
  let sine_cell = Array.mask_read hidden (SZ.add 15632sz j);
  let sine = Cast.cast_bf16_to_f32 sine_cell;
  source_connection before p;
  assert pure (cosine_cell == S.index (produced_cells before p) (SZ.v j));
  assert pure (sine_cell == S.index (produced_cells before p) (32 + SZ.v j));
  (cosine, sine)
}
