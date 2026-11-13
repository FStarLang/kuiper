module KPlay.Functional

#lang-pulse

open FStar.Real
open Pulse.Lib.Core
open Pulse.Lib.Send

module PA = Pulse.Lib.Array
module PC = Pulse.Lib.Core
module S = FStar.Seq.Base
module SP = FStar.Seq.Properties
module BF16 = Kuiper.BFloat16
module NativeNeg = Kuiper.Float32.NativeNeg
module KL = Kuiper.Locs
module KA = Kuiper.Array.Core
module Permission = PulseCore.FractionalPermission

inline_for_extraction noextract
let source_window_sequence_partition (contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  : Lemma (
      S.slice (S.slice contents lo weight_elements) 0 (hi - lo) ==
        S.slice contents lo hi /\
      S.slice (S.slice contents lo weight_elements) (hi - lo) (weight_elements - lo) ==
        S.slice contents hi weight_elements /\
      S.append (S.append (S.slice contents 0 lo) (S.slice contents lo hi))
        (S.slice contents hi weight_elements) == contents) =
  SP.slice_slice contents lo weight_elements 0 (hi - lo);
  SP.slice_slice contents lo weight_elements (hi - lo) (weight_elements - lo);
  assert (S.equal
    (S.append (S.append (S.slice contents 0 lo) (S.slice contents lo hi))
      (S.slice contents hi weight_elements)) contents);
  S.lemma_eq_elim
    (S.append (S.append (S.slice contents 0 lo) (S.slice contents lo hi))
      (S.slice contents hi weight_elements)) contents

inline_for_extraction noextract
ghost
fn source_take_weight_window_local
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires PA.pts_to weights #fraction contents
  ensures source_weight_window_resources weights contents lo hi fraction
{
  source_weight_window_shape weights lo hi;
  source_window_sequence_partition contents lo hi;
  PA.pts_to_range_intro weights fraction contents;
  PA.pts_to_range_split weights 0 lo weight_elements;
  PA.pts_to_range_split weights lo hi weight_elements;
  with middle. assert PA.pts_to_range weights lo hi #fraction middle;
  PA.unfold_pts_to_range weights lo hi #fraction middle;
  PA.from_mask (PA.gsub weights lo hi);
  with window_values. assert PA.pts_to (PA.gsub weights lo hi) #fraction window_values;
  assert pure (S.equal window_values (S.slice contents lo hi));
  S.lemma_eq_elim window_values (S.slice contents lo hi);
  fold source_weight_window_resources weights contents lo hi fraction;
}

inline_for_extraction noextract
ghost
fn source_return_weight_window_local
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires source_weight_window_resources weights contents lo hi fraction
  ensures PA.pts_to weights #fraction contents
{
  source_window_sequence_partition contents lo hi;
  unfold source_weight_window_resources weights contents lo hi fraction;
  PA.to_mask (PA.gsub weights lo hi);
  PA.fold_pts_to_range weights lo hi #fraction (S.slice contents lo hi);
  PA.pts_to_range_join weights 0 lo hi;
  PA.pts_to_range_join weights 0 hi weight_elements;
  PA.pts_to_range_elim weights fraction contents;
}

inline_for_extraction noextract
ghost
fn take_source_weight_window
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires PC.on KL.gpu_loc (PA.pts_to weights #fraction contents)
  ensures PC.on KL.gpu_loc
    (source_weight_window_resources weights contents lo hi fraction)
{
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (PA.pts_to weights #fraction contents))
    (PC.on KL.gpu_loc (source_weight_window_resources weights contents lo hi fraction))
    fn () {
      PC.on_elim (PA.pts_to weights #fraction contents);
      source_take_weight_window_local weights #contents lo hi #fraction;
      PC.on_intro (source_weight_window_resources weights contents lo hi fraction);
    };
}

inline_for_extraction noextract
ghost
fn return_source_weight_window
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires PC.on KL.gpu_loc
    (source_weight_window_resources weights contents lo hi fraction)
  ensures PC.on KL.gpu_loc (PA.pts_to weights #fraction contents)
{
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (source_weight_window_resources weights contents lo hi fraction))
    (PC.on KL.gpu_loc (PA.pts_to weights #fraction contents))
    fn () {
      PC.on_elim (source_weight_window_resources weights contents lo hi fraction);
      source_return_weight_window_local weights #contents lo hi #fraction;
      PC.on_intro (PA.pts_to weights #fraction contents);
    };
}

inline_for_extraction noextract
ghost
fn split_first_decoder_alias
  (hidden:PA.larray BF16.t 7168)
  (#contents:values BF16.t 7168)
  requires PC.on KL.gpu_loc (PA.pts_to hidden contents)
  ensures PC.on KL.gpu_loc
    (PA.pts_to hidden #0.5R contents ** PA.pts_to hidden #0.5R contents)
{
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (PA.pts_to hidden contents))
    (PC.on KL.gpu_loc
      (PA.pts_to hidden #0.5R contents ** PA.pts_to hidden #0.5R contents))
    fn () {
      PC.on_elim (PA.pts_to hidden contents);
      PA.share hidden #contents #1.0R;
      PC.on_intro
        (PA.pts_to hidden #0.5R contents ** PA.pts_to hidden #0.5R contents);
    };
}

inline_for_extraction noextract
ghost
fn reassemble_first_decoder_alias
  (hidden:PA.larray BF16.t 7168)
  (#left #right:values BF16.t 7168)
  requires PC.on KL.gpu_loc
    (PA.pts_to hidden #0.5R left ** PA.pts_to hidden #0.5R right)
  ensures PC.on KL.gpu_loc (PA.pts_to hidden left) ** PC.pure (left == right)
{
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc
      (PA.pts_to hidden #0.5R left ** PA.pts_to hidden #0.5R right))
    (PC.on KL.gpu_loc (PA.pts_to hidden left) ** PC.pure (left == right))
    fn () {
      PC.on_elim
        (PA.pts_to hidden #0.5R left ** PA.pts_to hidden #0.5R right);
      PA.gather hidden #left #right #0.5R #0.5R;
      PC.on_intro (PA.pts_to hidden left);
    };
}

inline_for_extraction noextract
ghost
fn source_embedding_call_residency
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (ambient:PC.slprop)
  preserves KL.cpu ** ambient ** PC.on KL.gpu_loc (PA.pts_to weights contents)
  ensures PC.pure (KA.is_global_array (source_embedding_window weights))
{
  take_source_weight_window weights #contents 0 embedding_elements #1.0R;
  source_weight_window_shape weights 0 embedding_elements;
  assert pure (KA.is_global_array (source_embedding_window weights));
  return_source_weight_window weights #contents 0 embedding_elements #1.0R;
}

inline_for_extraction noextract
ghost
fn expose_source_weight_window
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires PC.on KL.gpu_loc
    (source_weight_window_resources weights contents lo hi fraction)
  ensures source_weight_window_frame weights contents lo hi fraction **
    PC.on KL.gpu_loc
      (PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi))
{
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (source_weight_window_resources weights contents lo hi fraction))
    (source_weight_window_frame weights contents lo hi fraction **
      PC.on KL.gpu_loc
        (PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi)))
    fn () {
      PC.on_elim (source_weight_window_resources weights contents lo hi fraction);
      unfold source_weight_window_resources weights contents lo hi fraction;
      PC.on_intro (PA.pts_to_range weights 0 lo #fraction (S.slice contents 0 lo));
      PC.on_intro
        (PA.pts_to_range weights hi weight_elements #fraction (S.slice contents hi weight_elements));
      PC.on_intro
        (PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi));
      fold source_weight_window_frame weights contents lo hi fraction;
    };
}

inline_for_extraction noextract
ghost
fn conceal_source_weight_window
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= weight_elements})
  (#fraction:Permission.perm)
  requires source_weight_window_frame weights contents lo hi fraction **
    PC.on KL.gpu_loc
      (PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi))
  ensures PC.on KL.gpu_loc
    (source_weight_window_resources weights contents lo hi fraction)
{
  ghost_impersonate KL.gpu_loc
    (source_weight_window_frame weights contents lo hi fraction **
      PC.on KL.gpu_loc
        (PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi)))
    (PC.on KL.gpu_loc (source_weight_window_resources weights contents lo hi fraction))
    fn () {
      unfold source_weight_window_frame weights contents lo hi fraction;
      PC.on_elim (PA.pts_to_range weights 0 lo #fraction (S.slice contents 0 lo));
      PC.on_elim
        (PA.pts_to_range weights hi weight_elements #fraction (S.slice contents hi weight_elements));
      PC.on_elim
        (PA.pts_to (source_weight_window weights lo hi) #fraction (S.slice contents lo hi));
      fold source_weight_window_resources weights contents lo hi fraction;
      PC.on_intro (source_weight_window_resources weights contents lo hi fraction);
    };
}

module Send = Pulse.Lib.Send

inline_for_extraction noextract
let source_metadata_mask (contents:values BF16.t weight_elements)
  : GTot (values (option BF16.t) weight_elements) =
  S.init_ghost weight_elements (fun i -> Some (S.index contents i))

inline_for_extraction noextract
ghost
fn source_empty_mask_localize
  (weights:packed_weights_address)
  (#contents:values BF16.t weight_elements)
  (ambient:PC.slprop)
  preserves KL.cpu ** ambient ** PC.on KL.gpu_loc (PA.pts_to weights contents)
  ensures PA.pts_to_mask weights #1.0R (source_metadata_mask contents) (fun _ -> False)
{
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (PA.pts_to weights contents))
    (PC.on KL.gpu_loc (PA.pts_to weights contents) **
      PC.on KL.gpu_loc
        (PA.pts_to_mask weights #1.0R (source_metadata_mask contents) (fun _ -> False)))
    fn () {
      PC.on_elim (PA.pts_to weights contents);
      PA.to_mask weights;
      PA.mask_vext weights (source_metadata_mask contents);
      PA.split_mask weights #1.0R #(source_metadata_mask contents)
        #(fun _ -> True) (fun _ -> False);
      PA.mask_mext weights #1.0R #(source_metadata_mask contents)
        #(PA.mask_isect (fun _ -> True) (fun _ -> False)) (fun _ -> False);
      PC.on_intro
        (PA.pts_to_mask weights #1.0R (source_metadata_mask contents) (fun _ -> False));
      PA.mask_mext weights #1.0R #(source_metadata_mask contents)
        #(PA.mask_diff (fun _ -> True) (fun _ -> False)) (fun _ -> True);
      PA.from_mask weights;
      with restored. assert PA.pts_to weights restored;
      assert pure (S.equal restored contents);
      S.lemma_eq_elim restored contents;
      PC.on_intro (PA.pts_to weights contents);
    };
  Send.placeless_on_elim
    (PA.pts_to_mask weights #1.0R (source_metadata_mask contents) (fun _ -> False))
    KL.gpu_loc;
}

module Ghost = FStar.Ghost
module HostSize = FStar.SizeT

inline_for_extraction noextract
fn materialize_source_weight_window
  (weights:packed_weights_address)
  (lo:HostSize.t)
  (hi:Ghost.erased nat{HostSize.v lo <= hi /\ hi <= weight_elements})
  (#contents:Ghost.erased (values BF16.t weight_elements))
  (ambient:PC.slprop)
  preserves KL.cpu ** ambient ** PC.on KL.gpu_loc (PA.pts_to weights contents)
  returns window:PA.array BF16.t
  ensures rewrites_to window (source_weight_window weights (HostSize.v lo) hi)
  ensures pure (PA.length window == hi - HostSize.v lo /\ KA.is_global_array window)
{
  source_weight_window_shape weights (HostSize.v lo) hi;
  source_weight_window_residency weights (HostSize.v lo) hi;
  source_empty_mask_localize weights #contents ambient;
  let window = PA.sub weights #1.0R #(fun _ -> False) lo hi
    #(Ghost.hide (source_metadata_mask contents <: S.seq (option BF16.t)));
  PA.mask_mext weights (fun _ -> False);
  drop_ (PA.pts_to_mask weights #1.0R (source_metadata_mask contents) (fun _ -> False));
  PA.mask_mext window (fun _ -> False);
  drop_ (PA.pts_to_mask window #1.0R
    (S.slice (source_metadata_mask contents) (HostSize.v lo) hi) (fun _ -> False));
  window
}

open Kuiper
module Shared = KPlay.SharedSpec
module SZ = Kuiper.SizeT
module IC = FStar.Int.Cast
module FC = Kuiper.Float.Casts
module I64 = FStar.Int64
module F32 = Kuiper.Float32
module R = FStar.Real
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry
module C = FStar.Classical
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Kernel = Kuiper.Kernel
module Pledge = Pulse.Lib.Pledge
open FStar.Tactics.Typeclasses { solve }

#push-options "--fuel 4 --ifuel 2 --z3rlimit 0"

noextract
let source_row_major_unique
  (width:pos) (r s:nat) (c d:nat{d < width})
  : Lemma
      (requires c < width)
      (ensures (r * width + c == s * width + d) <==> (r == s /\ c == d)) =
  Kuiper.Math.div_mod_of_mul_add width r c;
  Kuiper.Math.div_mod_of_mul_add width s d

noextract
let source_updated_index
  (contents:S.seq BF16.t) (at:nat{at < S.length contents})
  (value:BF16.t) (index:nat{index < S.length contents})
  : Lemma (
      S.index (S.upd contents at value) index ==
      (if index = at then value else S.index contents index)) =
  if index = at then FStar.Seq.lemma_index_upd1 contents at value
  else FStar.Seq.lemma_index_upd2 contents at value index

noextract
let source_embedding_ready
  (weights:source_embedding_values) (ts:source_token_values)
  (hidden:source_hidden_values) (upto:nat) : prop =
  forall (p:Shared.index 7) (h:Shared.index 1024).
    p * 1024 + h < upto ==>
    S.index hidden (p * 1024 + h) ==
      S.index weights (I64.v (S.index ts p) * 1024 + h)

noextract
let source_embedding_step
  (weights:source_embedding_values) (ts:source_token_values) (hidden:source_hidden_values)
  (p:Shared.index 7) (h:Shared.index 1024) (value:BF16.t)
  : Lemma
      (requires source_embedding_ready weights ts hidden (p * 1024 + h) /\
        value == S.index weights (I64.v (S.index ts p) * 1024 + h))
      (ensures source_embedding_ready weights ts
        (S.upd hidden (p * 1024 + h) value) (p * 1024 + h + 1)) =
  let after = S.upd hidden (p * 1024 + h) value in
  let at_cell (r:Shared.index 7) (c:Shared.index 1024)
    : Lemma (r * 1024 + c < p * 1024 + h + 1 ==>
        S.index after (r * 1024 + c) ==
          S.index weights (I64.v (S.index ts r) * 1024 + c)) =
    source_row_major_unique 1024 r p c h;
    source_updated_index hidden (p * 1024 + h) value (r * 1024 + c)
  in C.forall_intro_2 at_cell

noextract
let source_rotary_ready (rotary:source_rotary_values) (upto:nat) : prop =
  forall (p:Shared.index 7) (d:Shared.index 128).
    p * 128 + d < upto ==>
    v_approximates (S.index rotary (p * 128 + d)) (embedding_cosine p d) /\
    v_approximates (S.index rotary (896 + p * 128 + d)) (embedding_sine p d)

noextract
let source_rotary_step
  (rotary:source_rotary_values) (p:Shared.index 7) (d:Shared.index 128)
  (cosine sine:BF16.t)
  : Lemma
      (requires source_rotary_ready rotary (p * 128 + d) /\
        v_approximates cosine (embedding_cosine p d) /\
        v_approximates sine (embedding_sine p d))
      (ensures source_rotary_ready
        (S.upd (S.upd rotary (p * 128 + d) cosine) (896 + p * 128 + d) sine)
        (p * 128 + d + 1)) =
  let a = p * 128 + d in
  let b = 896 + p * 128 + d in
  let first = S.upd rotary a cosine in
  let after = S.upd first b sine in
  let at_cell (r:Shared.index 7) (c:Shared.index 128)
    : Lemma (r * 128 + c < p * 128 + d + 1 ==>
        v_approximates (S.index after (r * 128 + c)) (embedding_cosine r c) /\
        v_approximates (S.index after (896 + r * 128 + c)) (embedding_sine r c)) =
    source_row_major_unique 128 r p c d;
    source_updated_index rotary a cosine (r * 128 + c);
    source_updated_index rotary a cosine (896 + r * 128 + c);
    source_updated_index first b sine (r * 128 + c);
    source_updated_index first b sine (896 + r * 128 + c)
  in C.forall_intro_2 at_cell

noextract
let source_frequency_refinement (d:Shared.index 128)
  : Lemma (
      d % 64 == embedding_frequency_coordinate d /\
      R.of_int (d % 64) /. 64.0R ==
        R.of_int (2 * embedding_frequency_coordinate d) /. 128.0R) =
  if d < 64 then Kuiper.Math.div_mod_of_mul_add 64 0 d
  else Kuiper.Math.div_mod_of_mul_add 64 1 (d - 64)

inline_for_extraction noextract
let source_integer_index (x:I64.t{0 <= I64.v x /\ SZ.fits (I64.v x)})
  : i:sz{SZ.v i == I64.v x} =
  assert_norm (pow2 32 == 0x100000000);
  SZ.uint32_to_sizet (IC.int64_to_uint32 x)

inline_for_extraction noextract
let source_coordinate (i:sz)
  : x:f32{v_approximates x (R.of_int (SZ.v i)) /\
      x == of_int #f32 (I64.int_to_t (SZ.v i))} =
  let integer = IC.uint32_to_int64 (SZ.sizet_to_u32 i) in
  of_int_approx #f32 integer;
  of_int #f32 integer

inline_for_extraction noextract
let source_rotary_angle
  (p:sz{SZ.v p < 7})
  (d:sz{SZ.v d < 128})
  (first_position:source_prefill_position)
  : result:f32{v_approximates result (embedding_angle (SZ.v p) (SZ.v d)) /\
      result == source_angle32 (SZ.v p) (SZ.v d)} =
  let frequency = d %^ 64sz in
  let base = of_int #f32 1000000L in
  let denominator = of_int #f32 64L in
  let numerator = source_coordinate frequency in
  let exponent = div numerator denominator in
  let power = pow base exponent in
  let inverse = div (one #f32) power in
  let position = source_coordinate (p +^ source_integer_index first_position) in
  let angle = mul position inverse in
  let real_exponent = R.of_int (SZ.v frequency) /. 64.0R in
  let real_power = Pow.pow 1000000.0R real_exponent in
  let real_angle = R.of_int (SZ.v p) *. (1.0R /. real_power) in
  of_int_approx #f32 1000000L;
  of_int_approx #f32 64L;
  div_approx numerator denominator (R.of_int (SZ.v frequency)) 64.0R;
  F32.pow_approx base exponent 1000000.0R real_exponent;
  div_approx (one #f32) power 1.0R real_power;
  a_mul position inverse (R.of_int (SZ.v p)) (1.0R /. real_power);
  source_frequency_refinement (SZ.v d);
  assert (real_angle == embedding_angle (SZ.v p) (SZ.v d));
  assert (inverse == source_inverse_frequency (SZ.v d % 64));
  assert (position == source_position32 (SZ.v p));
  angle

noextract
let source_rotary_casts_refinement
  (angle:f32) (p:Shared.index 7) (d:Shared.index 128)
  : Lemma
      (requires v_approximates angle (embedding_angle p d))
      (ensures
        v_approximates (FC.fcast #f32 #BF16.t (cos angle)) (embedding_cosine p d) /\
        v_approximates (FC.fcast #f32 #BF16.t (sin angle)) (embedding_sine p d)) =
  F32.cos_approx angle (embedding_angle p d);
  F32.sin_approx angle (embedding_angle p d);
  FC.fcast_approx #f32 #BF16.t (cos angle) (Trig.cos (embedding_angle p d));
  FC.fcast_approx #f32 #BF16.t (sin angle) (Trig.sin (embedding_angle p d))

inline_for_extraction noextract
fn source_embed
  (weights:larray BF16.t embedding_elements)
  (tokens:larray I64.t 7)
  (hidden:larray BF16.t 7168)
  (#w:erased source_embedding_values)
  (#ts:erased (values I64.t 7){valid_prompt ts})
  (#before:erased source_hidden_values)
  preserves gpu ** weights |-> w ** tokens |-> ts
  requires hidden |-> before
  ensures exists* (after:source_hidden_values).
    hidden |-> after ** pure (source_embedding_ready w ts after 7168)
{
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (contents:source_hidden_values).
      row |-> r ** hidden |-> contents **
      pure (source_embedding_ready w ts contents (SZ.v r * 1024))
    decreases (7 - SZ.v !row)
  {
    let r = !row;
    let token = tokens.(r);
    assert pure (0 <= I64.v token /\ I64.v token < 151936);
    let selected = source_integer_index token;
    let mut column = 0sz;
    while (!column <^ 1024sz)
      invariant exists* (c:sz{c <= 1024}) (contents:source_hidden_values).
        column |-> c ** hidden |-> contents **
        pure (source_embedding_ready w ts contents (SZ.v r * 1024 + SZ.v c))
      decreases (1024 - SZ.v !column)
    {
      let c = !column;
      let value = weights.(selected *^ 1024sz +^ c);
      with old. assert hidden |-> old;
      hidden.(r *^ 1024sz +^ c) <- value;
      source_embedding_step w ts old (SZ.v r) (SZ.v c) value;
      column := !column +^ 1sz;
    };
    row := !row +^ 1sz;
  }
}

inline_for_extraction noextract
let source_exact_rotary_step (rotary:source_rotary_values)
  (p:Shared.index 7) (d:Shared.index 128) (cosine sine:BF16.t)
  : Lemma
      (requires source_exact_rotary rotary (p * 128 + d) /\
        cosine == source_rotary_cell 0 p d /\ sine == source_rotary_cell 1 p d)
      (ensures source_exact_rotary
        (S.upd (S.upd rotary (p * 128 + d) cosine) (896 + p * 128 + d) sine)
        (p * 128 + d + 1)) =
  let at (r:Shared.index 7) (c:Shared.index 128)
    : Lemma (r * 128 + c < p * 128 + d + 1 ==>
        S.index (S.upd (S.upd rotary (p * 128 + d) cosine) (896 + p * 128 + d) sine)
          (r * 128 + c) == source_rotary_cell 0 r c /\
        S.index (S.upd (S.upd rotary (p * 128 + d) cosine) (896 + p * 128 + d) sine)
          (896 + r * 128 + c) == source_rotary_cell 1 r c) =
    source_row_major_unique 128 r p c d
  in C.forall_intro_2 at

inline_for_extraction noextract
fn source_make_rotary
  (first_position:source_prefill_position)
  (rotary:larray BF16.t 1792)
  (#before:erased source_rotary_values)
  preserves gpu
  requires rotary |-> before
  ensures exists* (after:source_rotary_values).
    rotary |-> after ** pure (source_rotary_ready after 896 /\ source_exact_rotary after 896)
{
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (contents:source_rotary_values).
      row |-> r ** rotary |-> contents **
      pure (source_rotary_ready contents (SZ.v r * 128) /\
        source_exact_rotary contents (SZ.v r * 128))
    decreases (7 - SZ.v !row)
  {
    let r = !row;
    let mut column = 0sz;
    while (!column <^ 128sz)
      invariant exists* (d:sz{d <= 128}) (contents:source_rotary_values).
        column |-> d ** rotary |-> contents **
        pure (source_rotary_ready contents (SZ.v r * 128 + SZ.v d) /\
          source_exact_rotary contents (SZ.v r * 128 + SZ.v d))
      decreases (128 - SZ.v !column)
    {
      let d = !column;
      let angle = source_rotary_angle r d first_position;
      source_rotary_casts_refinement angle (SZ.v r) (SZ.v d);
      let cosine = FC.fcast #f32 #BF16.t (cos angle);
      let sine = FC.fcast #f32 #BF16.t (sin angle);
      with old. assert rotary |-> old;
      source_exact_rotary_step old (SZ.v r) (SZ.v d) cosine sine;
      let offset = r *^ 128sz +^ d;
      rotary.(offset) <- cosine;
      pts_to_len rotary;
      rotary.(896sz +^ offset) <- sine;
      source_rotary_step old (SZ.v r) (SZ.v d) cosine sine;
      column := !column +^ 1sz;
    };
    row := !row +^ 1sz;
  }
}

inline_for_extraction noextract
fn source_bound_embedding_kernel
  (weights:larray BF16.t embedding_elements)
  (tokens:larray I64.t 7)
  (first_position:source_prefill_position)
  (hidden:larray BF16.t 7168)
  (rotary:larray BF16.t 1792)
  (#w:erased source_embedding_values)
  (#ts:erased (values I64.t 7){valid_prompt ts})
  (#hidden_before:erased source_hidden_values)
  (#rotary_before:erased source_rotary_values)
  preserves gpu ** weights |-> w ** tokens |-> ts
  requires hidden |-> hidden_before ** rotary |-> rotary_before
  ensures exists* (h:source_hidden_values) (r:source_rotary_values).
    hidden |-> h ** rotary |-> r **
    pure (accepted_embedding_pointwise w ts h r /\ source_exact_rotary r 896)
{
  source_embed weights tokens hidden;
  source_make_rotary first_position rotary;
}

inline_for_extraction noextract
let source_embedding_raw_post
  (addresses:source_input_addresses) (outputs:source_embedding_addresses)
  (input:source_inputs) : PC.slprop =
  PA.pts_to addresses.weights_address input.weights **
  PA.pts_to addresses.prompt_address input.prompt **
  (exists* (hidden:source_hidden_values) (rotary:source_rotary_values).
    PA.pts_to outputs.embedded_address hidden ** PA.pts_to outputs.rotary_address rotary **
    PC.pure (accepted_embedding_pointwise (S.slice input.weights 0 embedding_elements)
      input.prompt hidden rotary /\ source_exact_rotary rotary 896))

inline_for_extraction noextract
fn source_embedding_native_call
  (addresses:source_input_addresses)
  (outputs:source_embedding_addresses)
  (window:PA.larray BF16.t embedding_elements{
    window == source_embedding_window addresses.weights_address})
  (#input:Ghost.erased source_inputs)
  preserves gpu
  requires source_embedding_kernel_pre addresses outputs input
  ensures source_embedding_raw_post addresses outputs input
{
  unfold source_embedding_kernel_pre addresses outputs input;
  assert rewrites_to window (source_weight_window addresses.weights_address 0 embedding_elements);
  with h0 r0. assert
    (PA.pts_to outputs.embedded_address h0 ** PA.pts_to outputs.rotary_address r0);
  source_take_weight_window_local addresses.weights_address
    #(Ghost.reveal input).weights 0 embedding_elements #1.0R;
  unfold source_weight_window_resources addresses.weights_address
    (Ghost.reveal input).weights 0 embedding_elements 1.0R;
  source_bound_embedding_kernel window addresses.prompt_address 0L
    outputs.embedded_address outputs.rotary_address
    #(Ghost.hide (S.slice (Ghost.reveal input).weights 0 embedding_elements))
    #(Ghost.hide (Ghost.reveal input).prompt) #h0 #r0;
  fold source_weight_window_resources addresses.weights_address
    (Ghost.reveal input).weights 0 embedding_elements 1.0R;
  source_return_weight_window_local addresses.weights_address
    #(Ghost.reveal input).weights 0 embedding_elements #1.0R;
  fold source_embedding_raw_post addresses outputs input;
}

inline_for_extraction noextract
ghost
fn source_embedding_post_adapter
  (addresses:source_input_addresses) (outputs:source_embedding_addresses)
  (#input:Ghost.erased source_inputs)
  ()
  requires source_embedding_raw_post addresses outputs input
  ensures source_embedding_kernel_post addresses outputs input
{
  unfold source_embedding_raw_post addresses outputs input;
  with h r. assert
    (PA.pts_to outputs.embedded_address h ** PA.pts_to outputs.rotary_address r);
  source_exact_rotary_generation r;
  fold source_embedding_kernel_post addresses outputs input;
}

inline_for_extraction noextract
instance source_send_global_array_contents
  (#a:Type0) (arr:PA.array a{KA.is_global_array arr})
  (#f:Permission.perm) (contents:S.seq a)
  : is_send_across KL.gpu_of (PA.pts_to arr #f contents) =
  KA.is_send_pts_to arr #f contents

inline_for_extraction noextract
let source_embedding_descriptor
  (addresses:source_input_addresses)
  (outputs:source_embedding_addresses)
  (window:PA.larray BF16.t embedding_elements{
    window == source_embedding_window addresses.weights_address})
  (#input:Ghost.erased source_inputs)
  : Kernel.kernel_desc
      (source_embedding_kernel_pre addresses outputs input)
      (source_embedding_kernel_post addresses outputs input) =
  let raw : Kernel.kernel_desc_1_1
    (source_embedding_kernel_pre addresses outputs input)
    (source_embedding_raw_post addresses outputs input) = {
    f = (fun () -> source_embedding_native_call addresses outputs window #input);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in
  Kernel.kd_weaken_post (Kernel.k11_as_kfull raw)
    (source_embedding_post_adapter addresses outputs #input)

inline_for_extraction noextract
fn source_allocate_embedding_outputs (ambient:PC.slprop)
  preserves KL.cpu ** ambient
  returns outputs:source_embedding_addresses
  ensures exists* (hidden:source_hidden_values) (rotary:source_rotary_values).
    PC.on KL.gpu_loc
      (PA.pts_to outputs.embedded_address hidden ** PA.pts_to outputs.rotary_address rotary)
{
  let hidden = KA.gpu_array_alloc #BF16.t 7168sz;
  let rotary = KA.gpu_array_alloc #BF16.t 1792sz;
  let outputs = { embedded_address = hidden; rotary_address = rotary };
  with (h:source_hidden_values). assert PC.on KL.gpu_loc (PA.pts_to hidden h);
  with (r:source_rotary_values). assert PC.on KL.gpu_loc (PA.pts_to rotary r);
  rewrite PC.on KL.gpu_loc (PA.pts_to hidden h)
    as PC.on KL.gpu_loc (PA.pts_to outputs.embedded_address h);
  rewrite PC.on KL.gpu_loc (PA.pts_to rotary r)
    as PC.on KL.gpu_loc (PA.pts_to outputs.rotary_address r);
  outputs
}

inline_for_extraction noextract
fn source_enqueue_embedding
  (addresses:source_input_addresses)
  (outputs:source_embedding_addresses)
  (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t)
  (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** ambient
  requires Epoch.epoch_live stream epoch **
    PC.on KL.gpu_loc (source_embedding_kernel_pre addresses outputs input)
  ensures Epoch.epoch_live stream (Epoch.epoch_next epoch) **
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch))
      (PC.on KL.gpu_loc (source_embedding_kernel_post addresses outputs input))
{
  let rest = PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
    (exists* (h:source_hidden_values) (r:source_rotary_values).
      PA.pts_to outputs.embedded_address h ** PA.pts_to outputs.rotary_address r);
  PC.on_star_eq KL.gpu_loc
    (PA.pts_to addresses.weights_address (Ghost.reveal input).weights) rest;
  rewrite PC.on KL.gpu_loc (source_embedding_kernel_pre addresses outputs input) as
    (PC.on KL.gpu_loc (PA.pts_to addresses.weights_address (Ghost.reveal input).weights) **
      PC.on KL.gpu_loc rest);
  assert pure (HostSize.v (HostSize.uint32_to_sizet 0ul) == 0);
  let window = materialize_source_weight_window addresses.weights_address
    (HostSize.uint32_to_sizet 0ul) (Ghost.hide embedding_elements)
    #(Ghost.hide (Ghost.reveal input).weights) ambient;
  rewrite
    (PC.on KL.gpu_loc (PA.pts_to addresses.weights_address (Ghost.reveal input).weights) **
      PC.on KL.gpu_loc rest) as
    PC.on KL.gpu_loc (source_embedding_kernel_pre addresses outputs input);
  let descriptor = source_embedding_descriptor addresses outputs window #input;
  Kernel.launch_kernel_full_owned descriptor stream #epoch;
}
#pop-options

module RT = Kuiper.Tensor.Runtime
module RuntimeHelpers = Kuiper.Tensor.Runtime.Helpers
module List = FStar.List.Tot

inline_for_extraction noextract
ghost
fn source_empty_allocation_storage
  (token:RT.tensor I64.t) (frame:PC.slprop) ()
  requires RT.contents token ** frame
  ensures source_allocated_storage ({ bf16_buffers = []; token_buffer = token }) frame
{
  fold RuntimeHelpers.all_contents #BF16.t [];
  fold source_allocated_storage ({ bf16_buffers = []; token_buffer = token }) frame;
}

inline_for_extraction noextract
ghost
fn source_cons_allocation_storage
  (value:RT.tensor BF16.t) (buffers:source_allocated_buffers)
  (frame:PC.slprop) ()
  requires source_allocated_storage buffers (RT.contents value ** frame)
  ensures source_allocated_storage
    ({ bf16_buffers = value::buffers.bf16_buffers; token_buffer = buffers.token_buffer }) frame
{
  unfold source_allocated_storage buffers (RT.contents value ** frame);
  fold RuntimeHelpers.all_contents (value::buffers.bf16_buffers);
  fold source_allocated_storage
    ({ bf16_buffers = value::buffers.bf16_buffers; token_buffer = buffers.token_buffer }) frame;
}

inline_for_extraction noextract
fn rec source_allocate_bf16_shapes
  (shapes:list source_allocation_shape)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  (frame:PC.slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream start
  requires Pledge.pledge0 (Epoch.epoch_done stream start) frame
  returns outcome:source_allocation_outcome
  ensures (
    match outcome with
    | SourceAllocationFailed _ -> Pledge.pledge0 (Epoch.epoch_done stream start) frame
    | SourceAllocated buffers ->
      PC.pure (source_allocation_shapes_match shapes buffers.bf16_buffers /\
        RT.metadata buffers.token_buffer [1; 8]) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_allocated_storage buffers frame))
  decreases shapes
{
  match shapes {
    [] -> {
      let result = RT.empty RT.Int64 stream #start [1; 8] frame ();
      match result {
        RT.Failed error -> { SourceAllocationFailed error }
        RT.Allocated token -> {
          let buffers = { bf16_buffers = []; token_buffer = token };
          fold RuntimeHelpers.all_owners #BF16.t stream [];
          rewrite RuntimeHelpers.all_owners #BF16.t stream [] as
            RuntimeHelpers.all_owners stream buffers.bf16_buffers;
          rewrite RT.owner stream token as RT.owner stream buffers.token_buffer;
          fold source_allocated_owners stream buffers;
          Pledge.rewrite_pledge (RT.contents token ** frame)
            (source_allocated_storage buffers frame)
            (source_empty_allocation_storage token frame);
          SourceAllocated buffers
        }
      }
    }
    shape::rest -> {
      let result = RT.empty RT.BFloat16 stream #start shape frame ();
      match result {
        RT.Failed error -> { SourceAllocationFailed error }
        RT.Allocated value -> {
          let following = source_allocate_bf16_shapes rest stream #start
            (RT.contents value ** frame) ();
          match following {
            SourceAllocationFailed error -> {
              RT.release stream #start value frame ();
              SourceAllocationFailed error
            }
            SourceAllocated buffers -> {
              let combined = {
                bf16_buffers = value::buffers.bf16_buffers;
                token_buffer = buffers.token_buffer
              };
              Pledge.rewrite_pledge
                (source_allocated_storage buffers (RT.contents value ** frame))
                (source_allocated_storage combined frame)
                (source_cons_allocation_storage value buffers frame);
              unfold source_allocated_owners stream buffers;
              fold RuntimeHelpers.all_owners stream (value::buffers.bf16_buffers);
              rewrite RuntimeHelpers.all_owners stream (value::buffers.bf16_buffers) as
                RuntimeHelpers.all_owners stream combined.bf16_buffers;
              rewrite RT.owner stream buffers.token_buffer as
                RT.owner stream combined.token_buffer;
              fold source_allocated_owners stream combined;
              SourceAllocated combined
            }
          }
        }
      }
    }
  }
}

inline_for_extraction noextract
fn source_prepare_allocations
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  (frame:PC.slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream start
  requires Pledge.pledge0 (Epoch.epoch_done stream start) frame
  returns outcome:source_allocation_outcome
  ensures (
    match outcome with
    | SourceAllocationFailed _ -> Pledge.pledge0 (Epoch.epoch_done stream start) frame
    | SourceAllocated buffers ->
      PC.pure (source_allocation_metadata buffers) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_allocated_storage buffers frame))
{
  source_allocation_count ();
  source_allocate_bf16_shapes source_bf16_allocation_shapes stream #start frame ()
}

inline_for_extraction noextract
instance source_runtime_contents_placeless (#a:Type0) (value:RT.tensor a)
  : placeless (RT.contents value) = solve

inline_for_extraction noextract
let rec source_all_contents_placeless_aux (buffers:list (RT.tensor BF16.t))
  : Tot (placeless (RuntimeHelpers.all_contents buffers)) (decreases buffers) =
  match buffers with
  | [] -> placeless_emp
  | value::rest ->
    placeless_star (RT.contents value) (RuntimeHelpers.all_contents rest)
      #(source_runtime_contents_placeless value) #(source_all_contents_placeless_aux rest)

inline_for_extraction noextract
instance source_all_contents_placeless (buffers:list (RT.tensor BF16.t))
  : placeless (RuntimeHelpers.all_contents buffers) =
  source_all_contents_placeless_aux buffers

inline_for_extraction noextract
ghost
fn source_allocations_to_embedding_pre
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:Ghost.erased source_inputs) ()
  requires source_allocated_storage buffers (source_borrowed_inputs addresses input)
  ensures PC.on KL.gpu_loc
    (source_embedding_kernel_pre addresses (source_allocated_embedding buffers) input **
      source_remaining_storage buffers)
{
  source_allocations_head_tail buffers;
  let hidden = List.index buffers.bf16_buffers 0;
  let rotary = List.index buffers.bf16_buffers 1;
  let rest = source_remaining_bf16 buffers;
  let outputs = source_allocated_embedding buffers;
  unfold source_allocated_storage buffers (source_borrowed_inputs addresses input);
  rewrite RuntimeHelpers.all_contents buffers.bf16_buffers as
    RuntimeHelpers.all_contents (hidden::rotary::rest);
  unfold RuntimeHelpers.all_contents (hidden::rotary::rest);
  unfold RuntimeHelpers.all_contents (rotary::rest);
  rewrite RuntimeHelpers.all_contents rest as
    RuntimeHelpers.all_contents (source_remaining_bf16 buffers);
  fold source_remaining_storage buffers;
  ghost_impersonate KL.gpu_loc
    (RT.contents rotary ** RT.contents hidden ** source_borrowed_inputs addresses input)
    (PC.on KL.gpu_loc (source_embedding_kernel_pre addresses outputs input))
    fn () {
      unfold RT.contents rotary;
      unfold RT.contents hidden;
      unfold source_borrowed_inputs addresses input;
      with r. assert PC.on KL.gpu_loc (PA.pts_to (RT.data rotary) r);
      with h. assert PC.on KL.gpu_loc (PA.pts_to (RT.data hidden) h);
      PC.on_elim (PA.pts_to (RT.data rotary) r);
      PC.on_elim (PA.pts_to (RT.data hidden) h);
      PC.on_elim (PA.pts_to addresses.weights_address (Ghost.reveal input).weights);
      PC.on_elim (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt);
      PA.pts_to_len (RT.data rotary) #1.0R #r;
      PA.pts_to_len (RT.data hidden) #1.0R #h;
      rewrite PA.pts_to (RT.data rotary) r as PA.pts_to outputs.rotary_address r;
      rewrite PA.pts_to (RT.data hidden) h as PA.pts_to outputs.embedded_address h;
      fold source_embedding_kernel_pre addresses outputs input;
      PC.on_intro (source_embedding_kernel_pre addresses outputs input);
    };
  rewrite PC.on KL.gpu_loc (source_embedding_kernel_pre addresses outputs input) as
    PC.on KL.gpu_loc
      (source_embedding_kernel_pre addresses (source_allocated_embedding buffers) input);
  Send.placeless_on_intro (source_remaining_storage buffers) KL.gpu_loc;
  Send.on_star_intro
    (source_embedding_kernel_pre addresses (source_allocated_embedding buffers) input)
    (source_remaining_storage buffers);
}

inline_for_extraction noextract
fn source_framed_embedding_body
  (addresses:source_input_addresses) (outputs:source_embedding_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (window:PA.larray BF16.t embedding_elements{
    window == source_embedding_window addresses.weights_address})
  (#input:Ghost.erased source_inputs)
  preserves gpu
  requires source_embedding_kernel_pre addresses outputs input ** source_remaining_storage buffers
  ensures source_embedding_kernel_post addresses outputs input ** source_remaining_storage buffers
{
  source_embedding_native_call addresses outputs window #input;
  source_embedding_post_adapter addresses outputs #input ();
}

inline_for_extraction noextract
let source_framed_embedding_descriptor
  (addresses:source_input_addresses) (outputs:source_embedding_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (window:PA.larray BF16.t embedding_elements{
    window == source_embedding_window addresses.weights_address})
  (#input:Ghost.erased source_inputs)
  : Kernel.kernel_desc
      (source_embedding_kernel_pre addresses outputs input ** source_remaining_storage buffers)
      (source_embedding_kernel_post addresses outputs input ** source_remaining_storage buffers) =
  let descriptor : Kernel.kernel_desc_1_1
    (source_embedding_kernel_pre addresses outputs input ** source_remaining_storage buffers)
    (source_embedding_kernel_post addresses outputs input ** source_remaining_storage buffers) = {
    f = (fun () -> source_framed_embedding_body addresses outputs buffers window #input);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in
  Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
ghost
fn source_embedding_frame_out
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (outputs:source_embedding_addresses{outputs == source_allocated_embedding buffers})
  (#input:Ghost.erased source_inputs) ()
  requires PC.on KL.gpu_loc
    (source_embedding_kernel_post addresses outputs input ** source_remaining_storage buffers)
  ensures source_after_embedding addresses input buffers
{
  Send.on_star_elim
    (source_embedding_kernel_post addresses outputs input) (source_remaining_storage buffers);
  Send.placeless_on_elim (source_remaining_storage buffers) KL.gpu_loc;
  rewrite PC.on KL.gpu_loc (source_embedding_kernel_post addresses outputs input) as
    PC.on KL.gpu_loc
      (source_embedding_kernel_post addresses (source_allocated_embedding buffers) input);
  fold source_after_embedding addresses input buffers;
}

inline_for_extraction noextract
fn source_enqueue_allocated_embedding
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (window:PA.larray BF16.t embedding_elements{
    window == source_embedding_window addresses.weights_address})
  (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    source_allocated_owners stream buffers
  requires Epoch.epoch_live stream start **
    Pledge.pledge0 (Epoch.epoch_done stream start)
      (source_allocated_storage buffers (source_borrowed_inputs addresses input))
  ensures Epoch.epoch_live stream (Epoch.epoch_next start) **
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next start))
      (source_after_embedding addresses input buffers)
{
  source_allocations_head_tail buffers;
  let hidden = List.index buffers.bf16_buffers 0;
  let rotary = List.index buffers.bf16_buffers 1;
  let rest = source_remaining_bf16 buffers;
  unfold source_allocated_owners stream buffers;
  rewrite RuntimeHelpers.all_owners stream buffers.bf16_buffers as
    RuntimeHelpers.all_owners stream (hidden::rotary::rest);
  unfold RuntimeHelpers.all_owners stream (hidden::rotary::rest);
  unfold RuntimeHelpers.all_owners stream (rotary::rest);
  let hidden_address = RT.data_ptr stream hidden;
  let rotary_address = RT.data_ptr stream rotary;
  let outputs = { embedded_address = hidden_address; rotary_address = rotary_address };
  assert pure (outputs == source_allocated_embedding buffers);
  fold RuntimeHelpers.all_owners stream (rotary::rest);
  fold RuntimeHelpers.all_owners stream (hidden::rotary::rest);
  rewrite RuntimeHelpers.all_owners stream (hidden::rotary::rest) as
    RuntimeHelpers.all_owners stream buffers.bf16_buffers;
  fold source_allocated_owners stream buffers;
  Pledge.rewrite_pledge
    (source_allocated_storage buffers (source_borrowed_inputs addresses input))
    (PC.on KL.gpu_loc
      (source_embedding_kernel_pre addresses (source_allocated_embedding buffers) input **
        source_remaining_storage buffers))
    (source_allocations_to_embedding_pre addresses buffers #input);
  rewrite Pledge.pledge0 (Epoch.epoch_done stream start)
    (PC.on KL.gpu_loc
      (source_embedding_kernel_pre addresses (source_allocated_embedding buffers) input **
        source_remaining_storage buffers)) as
    Pledge.pledge0 (Epoch.epoch_done stream start)
      (PC.on KL.gpu_loc
        (source_embedding_kernel_pre addresses outputs input ** source_remaining_storage buffers));
  let descriptor = source_framed_embedding_descriptor addresses outputs buffers window #input;
  Kernel.launch_kernel_full descriptor stream #start;
  Pledge.rewrite_pledge
    (PC.on KL.gpu_loc
      (source_embedding_kernel_post addresses outputs input ** source_remaining_storage buffers))
    (source_after_embedding addresses input buffers)
    (source_embedding_frame_out addresses buffers outputs #input);
}

inline_for_extraction noextract
fn source_prepare_and_enqueue
  (addresses:source_input_addresses) (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** source_borrowed_inputs addresses input
  returns outcome:source_prefix_outcome
  ensures (
    match outcome with
    | SourcePrefixFailed _ ->
      Epoch.epoch_live stream start **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_borrowed_inputs addresses input)
    | SourcePrefixEnqueued buffers ->
      Epoch.epoch_live stream (Epoch.epoch_next start) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next start))
        (source_after_embedding addresses input buffers))
{
  unfold source_borrowed_inputs addresses input;
  assert pure (HostSize.v (HostSize.uint32_to_sizet 0ul) == 0);
  let window = materialize_source_weight_window addresses.weights_address
    (HostSize.uint32_to_sizet 0ul) (Ghost.hide embedding_elements)
    #(Ghost.hide (Ghost.reveal input).weights) ambient;
  fold source_borrowed_inputs addresses input;
  Pledge.return_pledge (Epoch.epoch_done stream start)
    (source_borrowed_inputs addresses input) #solve;
  let allocations = source_prepare_allocations stream #start
    (source_borrowed_inputs addresses input) ();
  match allocations {
    SourceAllocationFailed error -> { SourcePrefixFailed error }
    SourceAllocated buffers -> {
      source_enqueue_allocated_embedding addresses buffers window #input stream #start;
      SourcePrefixEnqueued buffers
    }
  }
}

(* Inlined checked SourceMMA.Layout; provenance is in seed-bindings.json. *)
open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided


let source_mma_layout_group (lane : natlt 32) : natlt 8 = lane / 4
let source_mma_layout_pair (lane : natlt 32) : natlt 7 = (lane % 4) * 2

let source_mma_layout_a_row (lane : natlt 32) (reg : natlt 4) : natlt 16 =
  source_mma_layout_group lane + (reg % 2) * 8

let source_mma_layout_a_col (lane : natlt 32) (reg : natlt 4) (half : natlt 2) : natlt 16 =
  source_mma_layout_pair lane + (reg / 2) * 8 + half

let source_mma_layout_b_row (lane : natlt 32) (reg : natlt 2) (half : natlt 2) : natlt 16 =
  source_mma_layout_pair lane + reg * 8 + half

let source_mma_layout_c_row (lane : natlt 32) (reg : natlt 4) : natlt 16 =
  source_mma_layout_group lane + (reg / 2) * 8

let source_mma_layout_c_col (lane : natlt 32) (reg : natlt 4) : natlt 8 =
  source_mma_layout_pair lane + reg % 2

let source_mma_layout_a_inverse (i : natlt 16) (j : natlt 16)
  : Lemma
      (source_mma_layout_a_row ((i % 8) * 4 + (j % 8) / 2) (i / 8 + (j / 8) * 2) == i /\
       source_mma_layout_a_col ((i % 8) * 4 + (j % 8) / 2) (i / 8 + (j / 8) * 2) (j % 2) == j)
  = ()

let source_mma_layout_b_inverse (i : natlt 16) (j : natlt 8)
  : Lemma
      (source_mma_layout_group (j * 4 + (i % 8) / 2) == j /\
       source_mma_layout_b_row (j * 4 + (i % 8) / 2) (i / 8) (i % 2) == i)
  = ()

let source_mma_layout_c_inverse (i : natlt 16) (j : natlt 8)
  : Lemma
      (source_mma_layout_c_row ((i % 8) * 4 + j / 2) ((i / 8) * 2 + j % 2) == i /\
       source_mma_layout_c_col ((i % 8) * 4 + j / 2) ((i / 8) * 2 + j % 2) == j)
  = ()

let source_mma_layout_c_unique (lane : natlt 32) (reg : natlt 4)
  : Lemma
      ((source_mma_layout_c_row lane reg % 8) * 4 + source_mma_layout_c_col lane reg / 2 == lane /\
       (source_mma_layout_c_row lane reg / 8) * 2 + source_mma_layout_c_col lane reg % 2 == reg)
  = ()

let source_mma_layout_a_address (l : layout2 16 16) (s : strided_row_major l)
  (lane : natlt 32) (reg : natlt 4) (half : natlt 2)
  : Lemma
      (cell_of_pos l (source_mma_layout_a_row lane reg) (source_mma_layout_a_col lane reg half) ==
       s.offset + s.stride * source_mma_layout_a_row lane reg + source_mma_layout_a_col lane reg half)
  = s.pf (source_mma_layout_a_row lane reg) (source_mma_layout_a_col lane reg half)

let source_mma_layout_b_address (l : layout2 16 7) (s : strided_col_major l)
  (lane : natlt 32 { source_mma_layout_group lane < 7 }) (reg : natlt 2) (half : natlt 2)
  : Lemma
      (cell_of_pos l (source_mma_layout_b_row lane reg half) (source_mma_layout_group lane) ==
       s.offset + s.stride * source_mma_layout_group lane + source_mma_layout_b_row lane reg half)
  = s.pf (source_mma_layout_b_row lane reg half) (source_mma_layout_group lane)

let source_mma_layout_c_address (l : layout2 16 7) (s : strided_col_major l)
  (lane : natlt 32) (reg : natlt 4 { source_mma_layout_c_col lane reg < 7 })
  : Lemma
      (cell_of_pos l (source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg) ==
       s.offset + s.stride * source_mma_layout_c_col lane reg + source_mma_layout_c_row lane reg)
  = s.pf (source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg)

let source_mma_layout_padding (b : chest2 bf16 16 7)
  (lane : natlt 32) (reg : natlt 2) (half : natlt 2)
  : Lemma
      (acc2 (Kuiper.TensorCore.MMA.Registers.pad_b7 b) (source_mma_layout_b_row lane reg half) (source_mma_layout_group lane) ==
       (if source_mma_layout_group lane < 7 then acc2 b (source_mma_layout_b_row lane reg half) (source_mma_layout_group lane)
        else zero #bf16))
  = ()

let source_mma_layout_stored_value (v : chest2 float 16 8)
  (lane : natlt 32) (reg : natlt 4 { source_mma_layout_c_col lane reg < 7 })
  : Lemma
      (acc2 (Kuiper.TensorCore.MMA.Registers.rounded7 v) (source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg) ==
       Kuiper.Float.Casts.Base.cast_f32_to_bf16 (acc2 v (source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg)))
  = ()

let source_mma_layout_added_value (d s : chest2 float 16 8)
  (lane : natlt 32) (reg : natlt 4)
  : Lemma
      (acc2 (Kuiper.TensorCore.MMA.Registers.add_values d s) (source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg) ==
       Kuiper.Scalars.Base.add
         (acc2 d (source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg))
         (acc2 s (source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg)))
  = ()

let source_mma_layout_packed_word (lo hi : natlt 65536) : natlt 4294967296 =
  lo + 65536 * hi

let source_mma_layout_packed_halves (lo hi : natlt 65536)
  : Lemma
      (source_mma_layout_packed_word lo hi % 65536 == lo /\
       source_mma_layout_packed_word lo hi / 65536 == hi /\
       source_mma_layout_packed_word 0 0 == 0)
  = ()

let source_mma_layout_source_masks (lane : natlt 32)
  : Lemma
      ((source_mma_layout_group lane < 7 <==> lane < 28) /\
       source_mma_layout_c_col lane 0 < 7 /\ source_mma_layout_c_col lane 2 < 7 /\
       (source_mma_layout_c_col lane 1 < 7 <==> source_mma_layout_pair lane + 1 < 7) /\
       (source_mma_layout_c_col lane 3 < 7 <==> source_mma_layout_pair lane + 1 < 7))
  = ()

(* Inlined checked SourceMMA.Kernel; provenance is in seed-bindings.json. *)
open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Pulse.Lib.Trade


inline_for_extraction noextract
instance source_mma_kernel_c7 : concrete_sz 7 = { x = 7sz; }

let source_mma_kernel_zeros : chest2 float 16 8 = const (16 @| 8 @| INil) (zero #float)

let rec source_mma_kernel_k_prefix
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (a : chest2 bf16 columns width) (b : chest2 bf16 width 7)
  (tile : natlt (columns / 16))
  (start : natle (width / 16))
  (count : nat { start + count <= width / 16 })
  : Tot (chest2 float 16 8) (decreases count) =
  if count = 0 then source_mma_kernel_zeros
  else Kuiper.TensorCore.MMA.emma (source_mma_kernel_k_prefix a b tile start (count - 1))
    (ematrix_subtile a 16 16 tile (start + count - 1))
    (Kuiper.TensorCore.MMA.Registers.pad_b7 (ematrix_subtile b 16 7 (start + count - 1) 0))

let source_mma_kernel_k_step
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (a : chest2 bf16 columns width) (b : chest2 bf16 width 7)
  (tile : natlt (columns / 16)) (start : natle (width / 16))
  (count : nat { start + count < width / 16 })
  : Lemma
      (source_mma_kernel_k_prefix a b tile start (count + 1) ==
       Kuiper.TensorCore.MMA.emma (source_mma_kernel_k_prefix a b tile start count)
         (ematrix_subtile a 16 16 tile (start + count))
         (Kuiper.TensorCore.MMA.Registers.pad_b7 (ematrix_subtile b 16 7 (start + count) 0)))
  = ()

let rec source_mma_kernel_split_prefix
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (a : chest2 bf16 columns width) (b : chest2 bf16 width 7)
  (tile : natlt (columns / 16))
  (steps : pos) (count : nat { count * steps <= width / 16 })
  : Tot (chest2 float 16 8) (decreases count) =
  if count = 0 then source_mma_kernel_zeros
  else
    let part = source_mma_kernel_k_prefix a b tile ((count - 1) * steps) steps in
    if count = 1 then part
    else Kuiper.TensorCore.MMA.Registers.add_values (source_mma_kernel_split_prefix a b tile steps (count - 1)) part

let source_mma_kernel_split_step
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (a : chest2 bf16 columns width) (b : chest2 bf16 width 7)
  (tile : natlt (columns / 16)) (steps : pos)
  (count : nat { (count + 1) * steps <= width / 16 })
  : Lemma
      (source_mma_kernel_split_prefix a b tile steps (count + 1) ==
       (if count = 0 then source_mma_kernel_k_prefix a b tile (count * steps) steps
        else Kuiper.TensorCore.MMA.Registers.add_values (source_mma_kernel_split_prefix a b tile steps count)
          (source_mma_kernel_k_prefix a b tile (count * steps) steps)))
  = ()

let source_mma_kernel_output_prefix
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (a : chest2 bf16 columns width) (b : chest2 bf16 width 7)
  (steps : pos) (splits : pos { splits * steps == width / 16 })
  (old : chest2 bf16 columns 7) (done : natle (columns / 16))
  : chest2 bf16 columns 7 =
  mk2 #bf16 #columns #7 (fun i j ->
    if i / 16 < done then
      acc2 (Kuiper.TensorCore.MMA.Registers.rounded7 (source_mma_kernel_split_prefix a b (i / 16) steps splits)) (i % 16) j
    else acc2 old i j)

let source_mma_kernel_output_zero
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (a : chest2 bf16 columns width) (b : chest2 bf16 width 7)
  (steps : pos) (splits : pos { splits * steps == width / 16 })
  (old : chest2 bf16 columns 7)
  : Lemma (source_mma_kernel_output_prefix a b steps splits old 0 == old)
  = Kuiper.Chest.ext (source_mma_kernel_output_prefix a b steps splits old 0) old

let source_mma_kernel_output_step
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (a : chest2 bf16 columns width) (b : chest2 bf16 width 7)
  (steps : pos) (splits : pos { splits * steps == width / 16 })
  (old : chest2 bf16 columns 7) (tile : natlt (columns / 16))
  : Lemma
      (update_tile (source_mma_kernel_output_prefix a b steps splits old tile) 16 7 tile 0
         (Kuiper.TensorCore.MMA.Registers.rounded7 (source_mma_kernel_split_prefix a b tile steps splits)) ==
       source_mma_kernel_output_prefix a b steps splits old (tile + 1))
  = Kuiper.Chest.ext
      (update_tile (source_mma_kernel_output_prefix a b steps splits old tile) 16 7 tile 0
        (Kuiper.TensorCore.MMA.Registers.rounded7 (source_mma_kernel_split_prefix a b tile steps splits)))
      (source_mma_kernel_output_prefix a b steps splits old (tile + 1))

let source_mma_kernel_source_loop_coordinates
  (width splits steps : pos { width == splits * steps * 16 })
  (split : natlt splits) (k : natlt steps)
  : Lemma
      (width / splits == steps * 16 /\
       split * (width / splits) + k * 16 == (split * steps + k) * 16 /\
       (split * steps + k) * 16 < (split + 1) * (width / splits) /\
       (split * steps + steps) * 16 == (split + 1) * (width / splits))
  = FStar.Math.Lemmas.cancel_mul_div (steps * 16) splits

inline_for_extraction noextract
fn source_mma_kernel_accumulate_segment
  (#columns #width : erased pos { 16 /?+ columns /\ 16 /?+ width })
  (#la : layout2 columns width) {| strided_row_major la |}
  (a : array2 bf16 la)
  (#lb : layout2 width 7) {| strided_col_major lb |}
  (b : array2 bf16 lb)
  (fr : Kuiper.TensorCore.MMA.fragment)
  (tile : szlt (columns / 16)) (start : szle (width / 16))
  (steps : sz { start + steps <= width / 16 })
  (#fa #fb : perm)
  (#va : chest2 bf16 columns width) (#vb : chest2 bf16 width 7)
  (#vc : chest2 float 16 8)
  preserves gpu
  preserves a |-> Frac fa va
  preserves b |-> Frac fb vb
  requires fr |-> vc
  ensures fr |-> source_mma_kernel_k_prefix va vb tile start steps
{
  tensor_pts_to_ref a;
  tensor_pts_to_ref b;
  Kuiper.TensorCore.MMA.fill fr (zero #float);
  let mut k = 0sz;
  while (!k <^ steps)
    invariant exists* (i : sz { i <= steps }).
      k |-> i ** fr |-> source_mma_kernel_k_prefix va vb tile start i
    invariant a |-> Frac fa va ** b |-> Frac fb vb ** gpu
    decreases (steps - !k)
  {
    let ki = start +^ !k;
    let at = array2_extract_tile_ro' a 16 16 (Kuiper.SizeT.v tile) (Kuiper.SizeT.v ki);
    let bt = array2_extract_tile_ro' b 16 7 (Kuiper.SizeT.v ki) 0;
    Kuiper.TensorCore.MMA.Registers.mma_sync7 at bt fr;
    ambig_trade_elim ();
    ambig_trade_elim ();
    source_mma_kernel_k_step va vb tile start !k;
    k := !k +^ 1sz;
  };
}

inline_for_extraction noextract
fn source_mma_kernel_accumulate_tile
  (#columns #width : erased pos { 16 /?+ columns /\ 16 /?+ width })
  (#la : layout2 columns width) {| strided_row_major la |}
  (a : array2 bf16 la)
  (#lb : layout2 width 7) {| strided_col_major lb |}
  (b : array2 bf16 lb)
  (accum : Kuiper.TensorCore.MMA.fragment) (tile : szlt (columns / 16))
  (steps : szp) (splits : szp { splits * steps == width / 16 })
  (#fa #fb : perm)
  (#va : chest2 bf16 columns width) (#vb : chest2 bf16 width 7)
  (#vc : chest2 float 16 8)
  preserves gpu
  preserves a |-> Frac fa va
  preserves b |-> Frac fb vb
  requires accum |-> vc
  ensures accum |-> source_mma_kernel_split_prefix va vb tile steps splits
{
  tensor_pts_to_ref a;
  let fr = Kuiper.TensorCore.MMA.alloc_fragment ();
  Kuiper.TensorCore.MMA.fill accum (zero #float);
  let mut split = 0sz;
  while (!split <^ splits)
    invariant exists* (i : sz { i <= splits }).
      split |-> i ** accum |-> source_mma_kernel_split_prefix va vb tile steps i
    invariant exists* v. fr |-> v
    invariant a |-> Frac fa va ** b |-> Frac fb vb ** gpu
    decreases (splits - !split)
  {
    source_mma_kernel_accumulate_segment a b fr tile (!split *^ steps) steps;
    if (!split = 0sz) { Kuiper.TensorCore.MMA.Registers.copy accum fr; }
    else { Kuiper.TensorCore.MMA.Registers.add accum fr; };
    source_mma_kernel_split_step va vb tile steps !split;
    split := !split +^ 1sz;
  };
  with v. assert fr |-> v;
  drop_ (fr |-> v);
}

inline_for_extraction noextract
fn source_mma_kernel_linear
  (columns width : szp { 16 /?+ columns /\ 16 /?+ width })
  (#la : layout2 columns width) {| strided_row_major la |}
  (a : array2 bf16 la)
  (#lb : layout2 width 7) {| strided_col_major lb |}
  (b : array2 bf16 lb)
  (#lc : layout2 columns 7) {| strided_col_major lc |}
  (c : array2 bf16 lc)
  (steps : szp) (splits : szp { splits * steps == width / 16 })
  (#fa #fb : perm)
  (#va : chest2 bf16 columns width) (#vb : chest2 bf16 width 7)
  (#vc : chest2 bf16 columns 7)
  preserves gpu
  preserves a |-> Frac fa va
  preserves b |-> Frac fb vb
  requires c |-> Frac (1.0R /. 32) vc
  ensures c |-> Frac (1.0R /. 32)
    (source_mma_kernel_output_prefix va vb steps splits vc (columns / 16))
{
  tensor_pts_to_ref a;
  tensor_pts_to_ref b;
  tensor_pts_to_ref c;
  source_mma_kernel_output_zero va vb steps splits vc;
  let accum = Kuiper.TensorCore.MMA.alloc_fragment ();
  let mut tile = 0sz;
  while (!tile <^ columns /^ 16sz)
    invariant exists* (i : sz { i <= columns / 16 }).
      tile |-> i **
      c |-> Frac (1.0R /. 32) (source_mma_kernel_output_prefix va vb steps splits vc i)
    invariant exists* v. accum |-> v
    invariant a |-> Frac fa va ** b |-> Frac fb vb ** gpu
    decreases (columns / 16 - !tile)
  {
    source_mma_kernel_accumulate_tile a b accum !tile steps splits;
    let ct = array2_extract_tile_st c 16 7 (Kuiper.SizeT.v !tile) 0;
    Kuiper.TensorCore.MMA.Registers.store7_bf16 accum ct;
    Pulse.Lib.Forall.elim_forall (Kuiper.TensorCore.MMA.Registers.rounded7 (source_mma_kernel_split_prefix va vb !tile steps splits));
    ambig_trade_elim ();
    source_mma_kernel_output_step va vb steps splits vc !tile;
    tile := !tile +^ 1sz;
  };
  with v. assert accum |-> v;
  drop_ (accum |-> v);
}

(* Inlined checked SourceMMA.Arrays; provenance is in seed-bindings.json. *)
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg


let source_mma_arrays_array_result
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (#n : nat) (weights : lseq bf16 n)
  (input : lseq bf16 (width * 7)) (old : lseq bf16 (columns * 7))
  (offset : nat { offset + columns * width <= n })
  (steps : pos) (splits : pos { steps * splits == width / 16 })
  : GTot (lseq bf16 (columns * 7)) =
  let va = from_seq (l2_row_major columns width)
    (FStar.Seq.slice weights offset (offset + columns * width)) in
  let vb = from_seq (l2_col_major width 7) input in
  let vc = from_seq (l2_col_major columns 7) old in
  to_seq (l2_col_major columns 7)
    (source_mma_kernel_output_prefix va vb steps splits vc (columns / 16))

ghost
fn source_mma_arrays_tensor_concr_unchanged
  (#et : Type0) (#r : nat) (#d : shape r)
  (l : full_tlayout d) (arr : larray et (sizeof d))
  (#f : perm) (#s : lseq et (sizeof d))
  requires from_array l arr |-> Frac f (from_seq l s)
  ensures arr |-> Frac f s
{
  tensor_concr (from_array l arr);
  rewrite each core (from_array l arr) as arr;
  to_from l s;
  rewrite (arr |-> Frac f (to_seq l (from_seq l s))) as (arr |-> Frac f s);
}

inline_for_extraction noextract
fn source_mma_arrays_linear_arrays
  (columns width : szp { 16 /?+ columns /\ 16 /?+ width })
  (#n : erased nat)
  (weights : larray bf16 n)
  (input : larray bf16 (width * 7))
  (output : larray bf16 (columns * 7))
  (offset : sz { offset + columns * width <= n })
  (steps : szp) (splits : szp { steps * splits == width / 16 })
  (#fw #fx : perm)
  (#wv : erased (lseq bf16 n))
  (#xv : erased (lseq bf16 (width * 7)))
  (#ov : erased (lseq bf16 (columns * 7)))
  preserves gpu
  preserves weights |-> Frac fw wv
  preserves input |-> Frac fx xv
  requires output |-> Frac (1.0R /. 32) ov
  ensures output |-> Frac (1.0R /. 32) (source_mma_arrays_array_result #columns #width #n wv xv ov offset steps splits)
{
  Pulse.Lib.Array.to_mask weights;
  with wm. assert Pulse.Lib.Array.pts_to_mask weights #fw wm (fun _ -> True);
  let window = Pulse.Lib.Array.sub weights offset (offset + columns * width);
  Pulse.Lib.Array.from_mask window;
  with ws. assert window |-> Frac fw ws;
  FStar.Seq.lemma_eq_elim ws (FStar.Seq.slice wv offset (offset + columns * width));
  assert pure (ws == FStar.Seq.slice wv offset (offset + columns * width));

  tensor_abs' (l2_row_major columns width) window;
  tensor_abs' (l2_col_major width 7) input;
  tensor_abs' (l2_col_major columns 7) output;
  let a = from_array (l2_row_major columns width) window;
  let b = from_array (l2_col_major width 7) input;
  let c = from_array (l2_col_major columns 7) output;
  assert rewrites_to a (from_array (l2_row_major columns width) window);
  assert rewrites_to b (from_array (l2_col_major width 7) input);
  assert rewrites_to c (from_array (l2_col_major columns 7) output);
  source_mma_kernel_linear columns width a b c steps splits;
  source_mma_arrays_tensor_concr_unchanged (l2_row_major columns width) window;
  source_mma_arrays_tensor_concr_unchanged (l2_col_major width 7) input;
  tensor_concr c;
  rewrite each core (from_array (l2_col_major columns 7) output) as output;

  Pulse.Lib.Array.to_mask window;
  Pulse.Lib.Array.return_sub weights;
  Pulse.Lib.Array.from_mask weights;
  with restored. assert weights |-> Frac fw restored;
  assert pure (FStar.Seq.length restored == FStar.Seq.length wv);
  assert pure (forall (i : nat { i < FStar.Seq.length wv }).
    FStar.Seq.index restored i == FStar.Seq.index wv i);
  FStar.Seq.lemma_eq_elim restored wv;
  assert pure (restored == wv);
}

(* reduced.cu:522. The input sequence is a legitimate precondition, not a
   claim that the preceding rmsnorm launch has been composed. All 32 lanes
   must call this body convergently with published input writes. *)
inline_for_extraction noextract
fn source_mma_arrays_first_linear
  (weights : larray bf16 596049920)
  (norm0 : larray bf16 7168)
  (qkv0 : larray bf16 28672)
  (#wv : erased (lseq bf16 596049920))
  (#xv : erased (lseq bf16 7168))
  (#ov : erased (lseq bf16 28672))
  preserves gpu
  preserves weights |-> Frac (1.0R /. 32) wv
  preserves norm0 |-> Frac (1.0R /. 32) xv
  requires qkv0 |-> Frac (1.0R /. 32) ov
  ensures qkv0 |-> Frac (1.0R /. 32)
    (source_mma_arrays_array_result #4096 #1024 #596049920 wv xv ov 155585536 64 1)
{
  let offset = Kuiper.SizeT.uint_to_t 155585536;
  source_mma_arrays_linear_arrays 4096sz 1024sz #596049920 weights norm0 qkv0 offset 64sz 1sz;
}

(* reduced.cu:528, including the first-split copy and later explicit FP32 add. *)
inline_for_extraction noextract
fn source_mma_arrays_first_split_linear
  (weights : larray bf16 596049920)
  (activated0 : larray bf16 21504)
  (down0 : larray bf16 7168)
  (#wv : erased (lseq bf16 596049920))
  (#xv : erased (lseq bf16 21504))
  (#ov : erased (lseq bf16 7168))
  preserves gpu
  preserves weights |-> Frac (1.0R /. 32) wv
  preserves activated0 |-> Frac (1.0R /. 32) xv
  requires down0 |-> Frac (1.0R /. 32) ov
  ensures down0 |-> Frac (1.0R /. 32)
    (source_mma_arrays_array_result #1024 #3072 #596049920 wv xv ov 168168704 96 2)
{
  let offset = Kuiper.SizeT.uint_to_t 168168704;
  source_mma_arrays_linear_arrays 1024sz 3072sz #596049920 weights activated0 down0 offset 96sz 2sz;
}

(* Inlined checked SourceMMA.SourceBinding; provenance is in seed-bindings.json. *)
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling


let source_mma_sourcebinding_to_seq_at (#et : Type0) (#r : nat) (#d : shape r)
  (l : full_tlayout d) (m : chest d et) (i : abs d)
  : Lemma (FStar.Seq.index (to_seq l m) (l.imap.f i) == acc m i)
  = Kuiper.Injection.lem_pat l.imap i

let source_mma_sourcebinding_weight_value
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (#n : nat) (weights : lseq bf16 n)
  (offset : nat { offset + columns * width <= n })
  (tile : natlt (columns / 16)) (k : natlt (width / 16))
  (lane : natlt 32) (reg : natlt 4) (half : natlt 2)
  : Lemma
      (acc2 (ematrix_subtile
        (from_seq (l2_row_major columns width)
          (FStar.Seq.slice weights offset (offset + columns * width)))
        16 16 tile k) (source_mma_layout_a_row lane reg) (source_mma_layout_a_col lane reg half) ==
       FStar.Seq.index weights
         (offset + (tile * 16 + source_mma_layout_a_row lane reg) * width +
          k * 16 + source_mma_layout_a_col lane reg half))
  = ()

let source_mma_sourcebinding_source_a_offset
  (width : pos) (offset tile k : nat)
  (lane : natlt 32) (reg : natlt 4) (half : natlt 2)
  : Lemma
      (offset + (tile + source_mma_layout_a_row lane reg) * width + k + source_mma_layout_a_col lane reg half ==
       offset + (tile + source_mma_layout_group lane) * width + k + source_mma_layout_pair lane +
         (if reg = 0 then 0
          else if reg = 1 then 8 * width
          else if reg = 2 then 8
          else 8 * width + 8) + half)
  = ()

let source_mma_sourcebinding_input_value
  (#width : pos { 16 /?+ width })
  (input : lseq bf16 (width * 7)) (k : natlt (width / 16))
  (lane : natlt 32) (reg : natlt 2) (half : natlt 2)
  : Lemma
      (acc2 (Kuiper.TensorCore.MMA.Registers.pad_b7
        (ematrix_subtile (from_seq (l2_col_major width 7) input) 16 7 k 0))
        (source_mma_layout_b_row lane reg half) (source_mma_layout_group lane) ==
       (if source_mma_layout_group lane < 7 then
          FStar.Seq.index input (source_mma_layout_group lane * width + k * 16 + source_mma_layout_pair lane + reg * 8 + half)
        else zero #bf16))
  =
    let vb = from_seq (l2_col_major width 7) input in
    source_mma_layout_padding (ematrix_subtile vb 16 7 k 0) lane reg half;
    if source_mma_layout_group lane < 7 then
      subtile_acc2 vb 16 7 k 0 (source_mma_layout_b_row lane reg half) (source_mma_layout_group lane)
    else ()

let source_mma_sourcebinding_output_value
  (#columns #width : pos { 16 /?+ columns /\ 16 /?+ width })
  (#n : nat) (weights : lseq bf16 n)
  (input : lseq bf16 (width * 7)) (old : lseq bf16 (columns * 7))
  (offset : nat { offset + columns * width <= n })
  (steps : pos) (splits : pos { steps * splits == width / 16 })
  (tile : natlt (columns / 16)) (lane : natlt 32)
  (reg : natlt 4 { source_mma_layout_c_col lane reg < 7 })
  : Lemma
      (FStar.Seq.index (source_mma_arrays_array_result #columns #width #n weights input old offset steps splits)
         (source_mma_layout_c_col lane reg * columns + tile * 16 + source_mma_layout_c_row lane reg) ==
       Kuiper.Float.Casts.Base.cast_f32_to_bf16
         (acc2 (source_mma_kernel_split_prefix
           (from_seq (l2_row_major columns width)
             (FStar.Seq.slice weights offset (offset + columns * width)))
           (from_seq (l2_col_major width 7) input) tile steps splits)
           (source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg)))
  =
    let va = from_seq (l2_row_major columns width)
      (FStar.Seq.slice weights offset (offset + columns * width)) in
    let vb = from_seq (l2_col_major width 7) input in
    let vc = from_seq (l2_col_major columns 7) old in
    source_mma_sourcebinding_to_seq_at (l2_col_major columns 7)
      (source_mma_kernel_output_prefix va vb steps splits vc (columns / 16))
      (idx2 #columns #7 (tile * 16 + source_mma_layout_c_row lane reg) (source_mma_layout_c_col lane reg))

let source_mma_sourcebinding_first_linear_extents ()
  : Lemma
      (1024 * 7 == 7168 /\ 4096 * 7 == 28672 /\
       155585536 + 4096 * 1024 <= 596049920 /\
       4096 / 16 == 256 /\ 1024 / 16 == 64 /\
       96 * 2 * 16 == 3072 /\ 3072 * 7 == 21504 /\
       168168704 + 1024 * 3072 <= 596049920)
  = ()

module F = Kuiper.Floating.Base
module SourceCasts = Kuiper.Float.Casts
module DeviceSize = Kuiper.SizeT

inline_for_extraction noextract
let source_tree_level (n:nat) : prop =
  n == 1 \/ n == 2 \/ n == 4 \/ n == 8 \/
  n == 16 \/ n == 32 \/ n == 64 \/ n == 128

noextract
let rec source_tree (leaves:S.seq F32.t)
  (span:pos{source_tree_level span /\ span <= S.length leaves})
  (level:pos{source_tree_level level /\ level <= span})
  (lane:nat{lane < level}) : GTot F32.t (decreases (span - level)) =
  if level = span then S.index leaves lane
  else F.add (source_tree leaves span (2 * level) lane)
    (source_tree leaves span (2 * level) (lane + level))

inline_for_extraction noextract
let source_tree_ready (leaves:S.seq F32.t)
  (current:S.seq F32.t{S.length current == S.length leaves})
  (span:pos{source_tree_level span /\ span <= S.length leaves})
  (level:pos{source_tree_level level /\ level <= span}) : prop =
  forall (lane:nat{lane < level}).
    S.index current lane == source_tree leaves span level lane

inline_for_extraction noextract
let source_tree_pair_ready (leaves:S.seq F32.t)
  (current:S.seq F32.t{S.length current == S.length leaves})
  (span:pos{source_tree_level span /\ span <= S.length leaves})
  (stride:pos{source_tree_level (2 * stride) /\ 2 * stride <= span})
  (done:nat{done <= stride}) : prop =
  (forall (lane:nat{lane < done}).
    S.index current lane == source_tree leaves span stride lane) /\
  (forall (lane:nat{done <= lane /\ lane < 2 * stride}).
    S.index current lane == source_tree leaves span (2 * stride) lane)

inline_for_extraction noextract
let source_tree_pair_step (leaves:S.seq F32.t)
  (current:S.seq F32.t{S.length current == S.length leaves})
  (span:pos{source_tree_level span /\ span <= S.length leaves})
  (stride:pos{source_tree_level (2 * stride) /\ 2 * stride <= span})
  (lane:nat{lane < stride})
  : Lemma
      (requires source_tree_pair_ready leaves current span stride lane)
      (ensures source_tree_pair_ready leaves
        (S.upd current lane (F.add (S.index current lane) (S.index current (lane + stride))))
        span stride (lane + 1)) =
  assert (source_tree leaves span stride lane ==
    F.add (source_tree leaves span (2 * stride) lane)
      (source_tree leaves span (2 * stride) (lane + stride)))

inline_for_extraction noextract
let source_tree_pair_completed (leaves:S.seq F32.t)
  (current:S.seq F32.t{S.length current == S.length leaves})
  (span:pos{source_tree_level span /\ span <= S.length leaves})
  (stride:pos{source_tree_level (2 * stride) /\ 2 * stride <= span})
  : Lemma
      (requires source_tree_pair_ready leaves current span stride stride)
      (ensures source_tree_ready leaves current span stride) = ()

inline_for_extraction noextract
fn source_reduce_partial
  (#capacity:erased nat)
  (partial:PA.larray F32.t capacity)
  (span:szp{source_tree_level span /\ span <= capacity})
  (#leaves:erased (S.seq F32.t){S.length leaves == capacity})
  preserves gpu
  requires PA.pts_to partial leaves
  returns result:F32.t
  ensures exists* (current:values F32.t capacity).
    PA.pts_to partial current **
    pure (result == source_tree leaves span 1 0 /\ S.index current 0 == result)
{
  open Pulse.Lib.Array;
  let mut stride = span /^ 2sz;
  while (!stride >^ 0sz)
    invariant exists* (s:sz) (current:values F32.t capacity).
      stride |-> s ** PA.pts_to partial current **
      pure ((DeviceSize.v s == 0 /\ source_tree_ready leaves current span 1) \/
        (s > 0 /\ source_tree_level (2 * s) /\ 2 * s <= span /\
          source_tree_ready leaves current span (2 * s)))
    decreases (DeviceSize.v !stride)
  {
    let s = !stride;
    let mut lane = 0sz;
    while (!lane <^ s)
      invariant exists* (l:sz{l <= s}) (current:values F32.t capacity).
        lane |-> l ** PA.pts_to partial current **
        pure (source_tree_pair_ready leaves current span s l)
      decreases (s - !lane)
    {
      let l = !lane;
      with current. assert PA.pts_to partial current;
      let left = partial.(l);
      let right = partial.(l +^ s);
      source_tree_pair_step leaves current span s l;
      partial.(l) <- F.add left right;
      lane := l +^ 1sz;
    };
    with current. assert PA.pts_to partial current;
    source_tree_pair_completed leaves current span s;
    assert pure (DeviceSize.v s == 1 \/
      (DeviceSize.v s > 1 /\ 2 * (DeviceSize.v s / 2) == DeviceSize.v s));
    stride := s /^ 2sz;
  };
  partial.(0sz)
}


inline_for_extraction noextract
let source_rms_component_step
  (input:source_hidden_values) (row:Shared.index 7)
  (lane:Shared.index 128) (component:Shared.index 4) (count:nat{count < 2})
  : Lemma (
      let value = SourceCasts.fcast #BF16.t #F32.t
        (S.index input (row * 1024 + lane * 4 + component + count * 512)) in
      source_rms_component input row lane component (count + 1) ==
        F.add (source_rms_component input row lane component count) (F.mul value value)) = ()

inline_for_extraction noextract
let source_rms_partial_ready (input:source_hidden_values) (row:Shared.index 7)
  (partial:values F32.t 128) (count:pos{source_rms_level count}) : prop =
  forall (lane:nat{lane < count}).
    S.index partial lane == source_rms_tree input row count lane

inline_for_extraction noextract
let source_rms_pair_ready (input:source_hidden_values) (row:Shared.index 7)
  (partial:values F32.t 128) (count:pos{source_rms_level count /\ count > 1})
  (done:nat{done <= count / 2}) : prop =
  (forall (lane:nat{lane < done}).
    S.index partial lane == source_rms_tree input row (count / 2) lane) /\
  (forall (lane:nat{done <= lane /\ lane < count}).
    S.index partial lane == source_rms_tree input row count lane)

inline_for_extraction noextract
let source_rms_pair_step (input:source_hidden_values) (row:Shared.index 7)
  (partial:values F32.t 128) (count:pos{source_rms_level count /\ count > 1})
  (lane:nat{lane < count / 2})
  : Lemma
      (requires source_rms_pair_ready input row partial count lane)
      (ensures source_rms_pair_ready input row
        (S.upd partial lane (F.add (S.index partial lane) (S.index partial (lane + count / 2))))
        count (lane + 1)) =
  assert (count == 2 * (count / 2));
  assert (source_rms_tree input row (count / 2) lane ==
    F.add (source_rms_tree input row count lane)
      (source_rms_tree input row count (lane + count / 2)))

inline_for_extraction noextract
let source_rms_pair_completed (input:source_hidden_values) (row:Shared.index 7)
  (partial:values F32.t 128) (count:pos{source_rms_level count /\ count > 1})
  : Lemma
      (requires source_rms_pair_ready input row partial count (count / 2))
      (ensures source_rms_partial_ready input row partial (count / 2)) = ()

inline_for_extraction noextract
let rec source_rms_tree_from_leaves
  (input:source_hidden_values) (row:Shared.index 7) (leaves:values F32.t 128)
  (level:pos{source_tree_level level}) (lane:nat{lane < level})
  : Lemma
      (requires forall (i:Shared.index 128). S.index leaves i == source_rms_lane input row i)
      (ensures source_tree leaves 128 level lane == source_rms_tree input row level lane)
      (decreases (128 - level)) =
  if level < 128 then (
    source_rms_tree_from_leaves input row leaves (2 * level) lane;
    source_rms_tree_from_leaves input row leaves (2 * level) (lane + level))

inline_for_extraction noextract
let source_rms_output_ready (weights:values BF16.t weight_elements)
  (input output:source_hidden_values) (offset:nat{offset + 1024 <= weight_elements})
  (done:nat{done <= 7168}) : prop =
  forall (row:Shared.index 7) (column:Shared.index 1024).
    row * 1024 + column < done ==>
    S.index output (row * 1024 + column) == source_rms_cell weights input offset row column

inline_for_extraction noextract
let source_rms_output_step (weights:values BF16.t weight_elements)
  (input output:source_hidden_values) (offset:nat{offset + 1024 <= weight_elements})
  (row:Shared.index 7) (column:Shared.index 1024)
  : Lemma
      (requires source_rms_output_ready weights input output offset (row * 1024 + column))
      (ensures source_rms_output_ready weights input
        (S.upd output (row * 1024 + column) (source_rms_cell weights input offset row column))
        offset (row * 1024 + column + 1)) =
  let point (r:Shared.index 7) (c:Shared.index 1024)
    : Lemma (r * 1024 + c < row * 1024 + column + 1 ==>
        S.index
          (S.upd output (row * 1024 + column) (source_rms_cell weights input offset row column))
          (r * 1024 + c) == source_rms_cell weights input offset r c) =
    source_row_major_unique 1024 r row c column;
    source_updated_index output (row * 1024 + column)
      (source_rms_cell weights input offset row column) (r * 1024 + c)
  in C.forall_intro_2 point

#push-options "--fuel 4 --ifuel 2 --z3rlimit 0"
inline_for_extraction noextract
fn source_rmsnorm_body
  (weights:packed_weights_address)
  (input output:PA.larray BF16.t 7168)
  (offset:DeviceSize.t{DeviceSize.v offset + 1024 <= weight_elements})
  (#wv:Ghost.erased (values BF16.t weight_elements))
  (#xv:Ghost.erased source_hidden_values)
  preserves KL.gpu
  requires source_rms_kernel_pre weights input output wv xv
  ensures source_rms_kernel_post weights input output wv xv (DeviceSize.v offset)
{
  open Pulse.Lib.Array;
  unfold source_rms_kernel_pre weights input output wv xv;
  with old. assert PA.pts_to output old;
  let mut partial : PA.array F32.t = [| 128sz |];
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (out:source_hidden_values).
      row |-> r ** PA.pts_to output out ** PA.pts_to_uninit partial 128 **
      pure (source_rms_output_ready wv xv out offset (r * 1024))
    decreases (7 - !row)
  {
    let r = !row;
    unfold PA.pts_to_uninit partial 128;
    let mut lane = 0sz;
    while (!lane <^ 128sz)
      invariant exists* (l:sz{l <= 128}) (pv:S.seq (option F32.t)).
        lane |-> l ** PA.pts_to_mask partial pv (fun _ -> True) **
        pure (S.length pv == 128 /\
          (forall (i:nat{i < l}). S.index pv i == Some (source_rms_lane xv r i)))
      decreases (128 - !lane)
    {
      let l = !lane;
      let mut v0 : F32.t = F.zero;
      let mut v1 : F32.t = F.zero;
      let mut v2 : F32.t = F.zero;
      let mut v3 : F32.t = F.zero;
      let mut column = l *^ 4sz;
      while (!column <^ 1024sz)
        invariant exists*
          (c:sz{DeviceSize.v c == DeviceSize.v l * 4 \/
            DeviceSize.v c == DeviceSize.v l * 4 + 512 \/
            DeviceSize.v c == DeviceSize.v l * 4 + 1024})
          (a b c2 d:F32.t).
          column |-> c ** v0 |-> a ** v1 |-> b ** v2 |-> c2 ** v3 |-> d **
          pure (
            a == source_rms_component xv r l 0 ((DeviceSize.v c - DeviceSize.v l * 4) / 512) /\
            b == source_rms_component xv r l 1 ((DeviceSize.v c - DeviceSize.v l * 4) / 512) /\
            c2 == source_rms_component xv r l 2 ((DeviceSize.v c - DeviceSize.v l * 4) / 512) /\
            d == source_rms_component xv r l 3 ((DeviceSize.v c - DeviceSize.v l * 4) / 512))
        decreases (1536 - !column)
      {
        let c = !column;
        let a = SourceCasts.fcast #BF16.t #F32.t input.(r *^ 1024sz +^ c);
        let b = SourceCasts.fcast #BF16.t #F32.t input.(r *^ 1024sz +^ c +^ 1sz);
        let cc = SourceCasts.fcast #BF16.t #F32.t input.(r *^ 1024sz +^ c +^ 2sz);
        let d = SourceCasts.fcast #BF16.t #F32.t input.(r *^ 1024sz +^ c +^ 3sz);
        let k = Ghost.hide ((DeviceSize.v c - DeviceSize.v l * 4) / 512);
        source_rms_component_step xv r l 0 k;
        source_rms_component_step xv r l 1 k;
        source_rms_component_step xv r l 2 k;
        source_rms_component_step xv r l 3 k;
        v0 := F.add !v0 (F.mul a a);
        v1 := F.add !v1 (F.mul b b);
        v2 := F.add !v2 (F.mul cc cc);
        v3 := F.add !v3 (F.mul d d);
        column := c +^ 512sz;
      };
      let value = F.add (F.add (F.add !v0 !v1) !v2) !v3;
      assert pure (value == source_rms_lane xv r l);
      PA.mask_write partial l value;
      lane := l +^ 1sz;
    };
    PA.from_mask partial;
    with pv. assert PA.pts_to partial pv;
    assert pure (S.length pv == 128);
    assert pure (source_rms_partial_ready xv r pv 128);
    let squared_sum = source_reduce_partial #128 partial 128sz #pv;
    source_rms_tree_from_leaves xv r pv 1 0;
    let scale = F.rsqrt
      (F.add (F.mul squared_sum (F.div (F.one #F32.t) (F.of_int #F32.t 1024L)))
        (F.of_literal #F32.t "0.000001"));
    assert pure (scale == source_rms_scale xv r);
    let mut column = 0sz;
    while (!column <^ 1024sz)
      invariant exists* (c:sz{c <= 1024}) (out:source_hidden_values).
        column |-> c ** PA.pts_to output out **
        pure (source_rms_output_ready wv xv out offset (r * 1024 + c))
      decreases (1024 - !column)
    {
      let c = !column;
      let x = input.(r *^ 1024sz +^ c);
      let normalized = SourceCasts.fcast #F32.t #BF16.t
        (F.mul (SourceCasts.fcast #BF16.t #F32.t x) scale);
      let weight = weights.(offset +^ c);
      let value = SourceCasts.fcast #F32.t #BF16.t
        (F.mul (SourceCasts.fcast #BF16.t #F32.t normalized)
          (SourceCasts.fcast #BF16.t #F32.t weight));
      assert pure (value == source_rms_cell wv xv offset r c);
      with out. assert PA.pts_to output out;
      source_rms_output_step wv xv out offset r c;
      output.(r *^ 1024sz +^ c) <- value;
      column := c +^ 1sz;
    };
    PA.to_mask partial;
    fold PA.pts_to_uninit partial 128;
    row := r +^ 1sz;
  };
  fold source_rms_kernel_post weights input output wv xv offset;
}
#pop-options

inline_for_extraction noextract
let source_rmsnorm_descriptor
  (weights:packed_weights_address)
  (input:PA.larray BF16.t 7168{KA.is_global_array input})
  (output:PA.larray BF16.t 7168{KA.is_global_array output})
  (offset:DeviceSize.t{DeviceSize.v offset + 1024 <= weight_elements})
  (#wv:Ghost.erased (values BF16.t weight_elements))
  (#xv:Ghost.erased source_hidden_values)
  : Kernel.kernel_desc
      (source_rms_kernel_pre weights input output wv xv)
      (source_rms_kernel_post weights input output wv xv (DeviceSize.v offset)) =
  let descriptor : Kernel.kernel_desc_1_1
    (source_rms_kernel_pre weights input output wv xv)
    (source_rms_kernel_post weights input output wv xv (DeviceSize.v offset)) = {
    f = (fun () -> source_rmsnorm_body weights input output offset #wv #xv);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in
  Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
ghost
fn source_embedding_to_first_rms_pre
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:Ghost.erased source_inputs) ()
  requires source_after_embedding addresses input buffers
  ensures PC.on KL.gpu_loc
    (source_first_rms_pre addresses (source_allocated_embedding buffers)
      (source_allocated_norm0 buffers) input ** source_remaining_rms_storage buffers)
{
  source_allocations_first_three buffers;
  let outputs = source_allocated_embedding buffers;
  let norm = List.index buffers.bf16_buffers 2;
  let rest = source_remaining_after_rms buffers;
  unfold source_after_embedding addresses input buffers;
  unfold source_remaining_storage buffers;
  rewrite RuntimeHelpers.all_contents (source_remaining_bf16 buffers) as
    RuntimeHelpers.all_contents (norm::rest);
  unfold RuntimeHelpers.all_contents (norm::rest);
  rewrite RuntimeHelpers.all_contents rest as
    RuntimeHelpers.all_contents (source_remaining_after_rms buffers);
  fold source_remaining_rms_storage buffers;
  rewrite PC.on KL.gpu_loc
    (source_embedding_kernel_post addresses (source_allocated_embedding buffers) input) as
    PC.on KL.gpu_loc (source_embedding_kernel_post addresses outputs input);
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (source_embedding_kernel_post addresses outputs input) ** RT.contents norm)
    (PC.on KL.gpu_loc
      (source_first_rms_pre addresses outputs (source_allocated_norm0 buffers) input))
    fn () {
      PC.on_elim (source_embedding_kernel_post addresses outputs input);
      unfold RT.contents norm;
      with old. assert PC.on KL.gpu_loc (PA.pts_to (RT.data norm) old);
      PC.on_elim (PA.pts_to (RT.data norm) old);
      PA.pts_to_len (RT.data norm) #1.0R #old;
      rewrite PA.pts_to (RT.data norm) old as
        PA.pts_to (source_allocated_norm0 buffers) old;
      fold source_first_rms_pre addresses outputs (source_allocated_norm0 buffers) input;
      PC.on_intro
        (source_first_rms_pre addresses outputs (source_allocated_norm0 buffers) input);
    };
  rewrite PC.on KL.gpu_loc
    (source_first_rms_pre addresses outputs (source_allocated_norm0 buffers) input) as
    PC.on KL.gpu_loc
      (source_first_rms_pre addresses (source_allocated_embedding buffers)
        (source_allocated_norm0 buffers) input);
  Send.placeless_on_intro (source_remaining_rms_storage buffers) KL.gpu_loc;
  Send.on_star_intro
    (source_first_rms_pre addresses (source_allocated_embedding buffers)
      (source_allocated_norm0 buffers) input)
    (source_remaining_rms_storage buffers);
}

inline_for_extraction noextract
fn source_framed_first_rms_body
  (addresses:source_input_addresses) (#outputs:erased source_embedding_addresses)
  (hidden_address:PA.larray BF16.t 7168{
    hidden_address == (Ghost.reveal outputs).embedded_address})
  (norm:PA.larray BF16.t 7168{KA.is_global_array norm})
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:Ghost.erased source_inputs)
  preserves KL.gpu
  requires source_first_rms_pre addresses outputs norm input ** source_remaining_rms_storage buffers
  ensures source_first_rms_post addresses outputs norm input ** source_remaining_rms_storage buffers
{
  unfold source_first_rms_pre addresses outputs norm input;
  unfold source_embedding_kernel_post addresses outputs input;
  with hidden rotary.
    assert PA.pts_to (Ghost.reveal outputs).embedded_address hidden **
      PA.pts_to (Ghost.reveal outputs).rotary_address rotary **
      PC.pure (source_embedding_result input hidden rotary);
  rewrite PA.pts_to (Ghost.reveal outputs).embedded_address hidden as PA.pts_to hidden_address hidden;
  fold source_rms_kernel_pre addresses.weights_address hidden_address norm
    (Ghost.reveal input).weights hidden;
  let offset = DeviceSize.uint32_to_sizet 155583488ul;
  source_rmsnorm_body addresses.weights_address hidden_address norm offset
    #(Ghost.hide (Ghost.reveal input).weights) #hidden;
  assert pure (DeviceSize.v offset == 155583488);
  rewrite source_rms_kernel_post addresses.weights_address hidden_address norm
    (Ghost.reveal input).weights hidden (DeviceSize.v offset) as
    source_rms_kernel_post addresses.weights_address hidden_address norm
      (Ghost.reveal input).weights hidden 155583488;
  unfold source_rms_kernel_post addresses.weights_address hidden_address norm
    (Ghost.reveal input).weights hidden 155583488;
  rewrite PA.pts_to hidden_address hidden as PA.pts_to (Ghost.reveal outputs).embedded_address hidden;
  fold source_first_rms_post addresses outputs norm input;
}

inline_for_extraction noextract
let source_framed_first_rms_descriptor
  (addresses:source_input_addresses) (#outputs:erased source_embedding_addresses)
  (hidden_address:PA.larray BF16.t 7168{
    hidden_address == (Ghost.reveal outputs).embedded_address})
  (norm:PA.larray BF16.t 7168{KA.is_global_array norm})
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:Ghost.erased source_inputs)
  : Kernel.kernel_desc
      (source_first_rms_pre addresses outputs norm input ** source_remaining_rms_storage buffers)
      (source_first_rms_post addresses outputs norm input ** source_remaining_rms_storage buffers) =
  let descriptor : Kernel.kernel_desc_1_1
    (source_first_rms_pre addresses outputs norm input ** source_remaining_rms_storage buffers)
    (source_first_rms_post addresses outputs norm input ** source_remaining_rms_storage buffers) = {
    f = (fun () -> source_framed_first_rms_body addresses #outputs hidden_address norm buffers #input);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in
  Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
ghost
fn source_first_rms_frame_out
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:Ghost.erased source_inputs) ()
  requires PC.on KL.gpu_loc
    (source_first_rms_post addresses (source_allocated_embedding buffers)
      (source_allocated_norm0 buffers) input ** source_remaining_rms_storage buffers)
  ensures source_after_rmsnorm addresses input buffers
{
  Send.on_star_elim
    (source_first_rms_post addresses (source_allocated_embedding buffers)
      (source_allocated_norm0 buffers) input) (source_remaining_rms_storage buffers);
  Send.placeless_on_elim (source_remaining_rms_storage buffers) KL.gpu_loc;
  fold source_after_rmsnorm addresses input buffers;
}

inline_for_extraction noextract
fn source_enqueue_first_rmsnorm
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    source_allocated_owners stream buffers
  requires Epoch.epoch_live stream (Epoch.epoch_next start) **
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next start))
      (source_after_embedding addresses input buffers)
  ensures Epoch.epoch_live stream (start + 2) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 2))
      (source_after_rmsnorm addresses input buffers)
{
  source_allocations_first_three buffers;
  let hidden = List.index buffers.bf16_buffers 0;
  let rotary = List.index buffers.bf16_buffers 1;
  let norm = List.index buffers.bf16_buffers 2;
  let rest = source_remaining_after_rms buffers;
  unfold source_allocated_owners stream buffers;
  rewrite RuntimeHelpers.all_owners stream buffers.bf16_buffers as
    RuntimeHelpers.all_owners stream (hidden::rotary::norm::rest);
  unfold RuntimeHelpers.all_owners stream (hidden::rotary::norm::rest);
  unfold RuntimeHelpers.all_owners stream (rotary::norm::rest);
  unfold RuntimeHelpers.all_owners stream (norm::rest);
  let hidden_address = RT.data_ptr stream hidden;
  let norm_address = RT.data_ptr stream norm;
  let outputs = Ghost.hide (source_allocated_embedding buffers);
  assert pure (hidden_address == (Ghost.reveal outputs).embedded_address /\
    norm_address == source_allocated_norm0 buffers);
  fold RuntimeHelpers.all_owners stream (norm::rest);
  fold RuntimeHelpers.all_owners stream (rotary::norm::rest);
  fold RuntimeHelpers.all_owners stream (hidden::rotary::norm::rest);
  rewrite RuntimeHelpers.all_owners stream (hidden::rotary::norm::rest) as
    RuntimeHelpers.all_owners stream buffers.bf16_buffers;
  fold source_allocated_owners stream buffers;
  Pledge.rewrite_pledge
    (source_after_embedding addresses input buffers)
    (PC.on KL.gpu_loc
      (source_first_rms_pre addresses (source_allocated_embedding buffers)
        (source_allocated_norm0 buffers) input ** source_remaining_rms_storage buffers))
    (source_embedding_to_first_rms_pre addresses buffers #input);
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next start))
    (PC.on KL.gpu_loc
      (source_first_rms_pre addresses (source_allocated_embedding buffers)
        (source_allocated_norm0 buffers) input ** source_remaining_rms_storage buffers)) as
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next start))
      (PC.on KL.gpu_loc
        (source_first_rms_pre addresses outputs norm_address input **
          source_remaining_rms_storage buffers));
  let descriptor = source_framed_first_rms_descriptor addresses #outputs hidden_address norm_address buffers #input;
  Kernel.launch_kernel_full descriptor stream #(Epoch.epoch_next start);
  rewrite Epoch.epoch_live stream (Epoch.epoch_next (Epoch.epoch_next start)) as
    Epoch.epoch_live stream (start + 2);
  rewrite Pledge.pledge0
    (Epoch.epoch_done stream (Epoch.epoch_next (Epoch.epoch_next start)))
    (PC.on KL.gpu_loc
      (source_first_rms_post addresses outputs norm_address input **
        source_remaining_rms_storage buffers)) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + 2))
      (PC.on KL.gpu_loc
        (source_first_rms_post addresses (source_allocated_embedding buffers)
          (source_allocated_norm0 buffers) input ** source_remaining_rms_storage buffers));
  Pledge.rewrite_pledge
    (PC.on KL.gpu_loc
      (source_first_rms_post addresses (source_allocated_embedding buffers)
        (source_allocated_norm0 buffers) input ** source_remaining_rms_storage buffers))
    (source_after_rmsnorm addresses input buffers)
    (source_first_rms_frame_out addresses buffers #input);
}

inline_for_extraction noextract
fn source_prepare_and_enqueue_rms
  (addresses:source_input_addresses) (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** source_borrowed_inputs addresses input
  returns outcome:source_prefix_outcome
  ensures (
    match outcome with
    | SourcePrefixFailed _ ->
      Epoch.epoch_live stream start **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_borrowed_inputs addresses input)
    | SourcePrefixEnqueued buffers ->
      Epoch.epoch_live stream (start + 2) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream (start + 2))
        (source_after_rmsnorm addresses input buffers))
{
  let outcome = source_prepare_and_enqueue addresses #input stream #start ambient;
  match outcome {
    SourcePrefixFailed error -> { SourcePrefixFailed error }
    SourcePrefixEnqueued buffers -> {
      source_enqueue_first_rmsnorm addresses buffers #input stream #start;
      SourcePrefixEnqueued buffers
    }
  }
}

module ArrayExtra = Kuiper.Array.Extra

inline_for_extraction noextract
let source_linear_old_independent
  (#columns #width:pos{16 /?+ columns /\ 16 /?+ width}) (#n:nat)
  (weights:values BF16.t n) (input:values BF16.t (width * 7))
  (left right:values BF16.t (columns * 7))
  (offset:nat{offset + columns * width <= n})
  (steps:pos) (splits:pos{steps * splits == width / 16})
  : Lemma (
      source_mma_arrays_array_result #columns #width #n weights input left offset steps splits ==
      source_mma_arrays_array_result #columns #width #n weights input right offset steps splits) =
  let a = from_seq (l2_row_major columns width)
    (S.slice weights offset (offset + columns * width)) in
  let b = from_seq (l2_col_major width 7) input in
  Kuiper.Chest.ext
    (source_mma_kernel_output_prefix a b steps splits (from_seq (l2_col_major columns 7) left) (columns / 16))
    (source_mma_kernel_output_prefix a b steps splits (from_seq (l2_col_major columns 7) right) (columns / 16))

inline_for_extraction noextract
let source_linear_pre
  (#columns #width #n:nat)
  (weights:PA.larray BF16.t n)
  (input:PA.larray BF16.t (width * 7))
  (output:PA.larray BF16.t (columns * 7))
  (wv:values BF16.t n) (xv:values BF16.t (width * 7))
  (ov:values BF16.t (columns * 7)) : PC.slprop =
  PA.pts_to weights wv ** PA.pts_to input xv ** PA.pts_to output ov

inline_for_extraction noextract
let source_linear_post
  (#columns #width:pos{16 /?+ columns /\ 16 /?+ width}) (#n:nat)
  (weights:PA.larray BF16.t n)
  (input:PA.larray BF16.t (width * 7))
  (output:PA.larray BF16.t (columns * 7))
  (wv:values BF16.t n) (xv:values BF16.t (width * 7))
  (ov:values BF16.t (columns * 7))
  (offset:nat{offset + columns * width <= n})
  (steps:pos) (splits:pos{steps * splits == width / 16}) : PC.slprop =
  PA.pts_to weights wv ** PA.pts_to input xv **
  PA.pts_to output (source_mma_arrays_array_result #columns #width #n wv xv ov offset steps splits)

inline_for_extraction noextract
ghost
fn source_linear_setup
  (#columns #width #n:nat)
  (weights:PA.larray BF16.t n)
  (input:PA.larray BF16.t (width * 7))
  (output:PA.larray BF16.t (columns * 7))
  (#wv:values BF16.t n) (#xv:values BF16.t (width * 7))
  (#ov:values BF16.t (columns * 7)) ()
  norewrite
  requires source_linear_pre #columns #width #n weights input output wv xv ov
  ensures (forall+ (_:natlt 32).
    PA.pts_to weights #(1.0R /. 32) wv **
    PA.pts_to input #(1.0R /. 32) xv **
    PA.pts_to output #(1.0R /. 32) ov) ** emp
{
  unfold source_linear_pre #columns #width #n weights input output wv xv ov;
  ArrayExtra.array_share weights #wv #1.0R 32;
  ArrayExtra.array_share input #xv #1.0R 32;
  ArrayExtra.array_share output #ov #1.0R 32;
  forevery_zip3
    (fun (_:natlt 32) -> PA.pts_to weights #(1.0R /. 32) wv)
    (fun (_:natlt 32) -> PA.pts_to input #(1.0R /. 32) xv)
    (fun (_:natlt 32) -> PA.pts_to output #(1.0R /. 32) ov);
}

inline_for_extraction noextract
ghost
fn source_linear_teardown
  (#columns #width:pos{16 /?+ columns /\ 16 /?+ width}) (#n:nat)
  (weights:PA.larray BF16.t n)
  (input:PA.larray BF16.t (width * 7))
  (output:PA.larray BF16.t (columns * 7))
  (#wv:values BF16.t n) (#xv:values BF16.t (width * 7))
  (#ov:values BF16.t (columns * 7))
  (offset:nat{offset + columns * width <= n})
  (steps:pos) (splits:pos{steps * splits == width / 16}) ()
  norewrite
  requires (forall+ (_:natlt 32).
    PA.pts_to weights #(1.0R /. 32) wv **
    PA.pts_to input #(1.0R /. 32) xv **
    PA.pts_to output #(1.0R /. 32)
      (source_mma_arrays_array_result #columns #width #n wv xv ov offset steps splits)) ** emp
  ensures source_linear_post #columns #width #n weights input output wv xv ov offset steps splits
{
  forevery_unzip3
    (fun (_:natlt 32) -> PA.pts_to weights #(1.0R /. 32) wv)
    (fun (_:natlt 32) -> PA.pts_to input #(1.0R /. 32) xv)
    (fun (_:natlt 32) -> PA.pts_to output #(1.0R /. 32)
      (source_mma_arrays_array_result #columns #width #n wv xv ov offset steps splits));
  ArrayExtra.array_gather weights #wv #1.0R 32;
  ArrayExtra.array_gather input #xv #1.0R 32;
  ArrayExtra.array_gather output
    #(source_mma_arrays_array_result #columns #width #n wv xv ov offset steps splits) #1.0R 32;
  fold source_linear_post #columns #width #n weights input output wv xv ov offset steps splits;
}

inline_for_extraction noextract
fn source_linear_thread
  (columns width:szp{16 /?+ columns /\ 16 /?+ width}) (#n:erased nat)
  (weights:PA.larray BF16.t n)
  (input:PA.larray BF16.t (width * 7))
  (output:PA.larray BF16.t (columns * 7))
  (offset:sz{offset + columns * width <= n})
  (steps:szp) (splits:szp{steps * splits == width / 16})
  (#wv:erased (values BF16.t n)) (#xv:erased (values BF16.t (width * 7)))
  (#ov:erased (values BF16.t (columns * 7)))
  (tid:szlt 32) ()
  norewrite
  requires gpu **
    (PA.pts_to weights #(1.0R /. 32) wv **
      PA.pts_to input #(1.0R /. 32) xv **
      PA.pts_to output #(1.0R /. 32) ov) ** thread_id 32 tid
  ensures gpu **
    (PA.pts_to weights #(1.0R /. 32) wv **
      PA.pts_to input #(1.0R /. 32) xv **
      PA.pts_to output #(1.0R /. 32)
        (source_mma_arrays_array_result #columns #width #n wv xv ov offset steps splits)) ** thread_id 32 tid
{
  source_mma_arrays_linear_arrays columns width #n weights input output offset steps splits
    #(1.0R /. 32) #(1.0R /. 32) #wv #xv #ov;
}

inline_for_extraction noextract
let source_linear_descriptor
  (columns width:szp{16 /?+ columns /\ 16 /?+ width}) (#n:erased nat)
  (weights:PA.larray BF16.t n{KA.is_global_array weights})
  (input:PA.larray BF16.t (width * 7){KA.is_global_array input})
  (output:PA.larray BF16.t (columns * 7){KA.is_global_array output})
  (offset:sz{offset + columns * width <= n})
  (steps:szp) (splits:szp{steps * splits == width / 16})
  (#wv:erased (values BF16.t n)) (#xv:erased (values BF16.t (width * 7)))
  (#ov:erased (values BF16.t (columns * 7)))
  : Kernel.kernel_desc
      (source_linear_pre #columns #width #n weights input output wv xv ov)
      (source_linear_post #columns #width #n weights input output wv xv ov offset steps splits) =
  let descriptor : Kernel.kernel_desc_1_n
    (source_linear_pre #columns #width #n weights input output wv xv ov)
    (source_linear_post #columns #width #n weights input output wv xv ov offset steps splits) = {
    nthr = 32sz;
    frame = emp;
    kpre = (fun _ ->
      PA.pts_to weights #(1.0R /. 32) wv **
      PA.pts_to input #(1.0R /. 32) xv **
      PA.pts_to output #(1.0R /. 32) ov);
    kpost = (fun _ ->
      PA.pts_to weights #(1.0R /. 32) wv **
      PA.pts_to input #(1.0R /. 32) xv **
      PA.pts_to output #(1.0R /. 32)
        (source_mma_arrays_array_result #columns #width #n wv xv ov offset steps splits));
    f = source_linear_thread columns width #n weights input output offset steps splits #wv #xv #ov;
    block_setup = source_linear_setup #columns #width #n weights input output #wv #xv #ov;
    block_teardown = source_linear_teardown #columns #width #n weights input output #wv #xv #ov offset steps splits;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve
  } in
  Kernel.k1n_as_kfull descriptor

inline_for_extraction noextract
let source_flat_prefix (#a:Type0) (output:S.seq a)
  (expected:(i:nat{i < S.length output}) -> GTot a)
  (done:nat{done <= S.length output}) : prop =
  forall (i:nat{i < done}). S.index output i == expected i

inline_for_extraction noextract
let source_flat_prefix_step (#a:Type0) (output:S.seq a)
  (expected:(i:nat{i < S.length output}) -> GTot a)
  (at:nat{at < S.length output})
  : Lemma
      (requires source_flat_prefix output expected at)
      (ensures source_flat_prefix (S.upd output at (expected at)) expected (at + 1)) = ()

inline_for_extraction noextract
let source_mask_prefix (#a:Type0) (output:S.seq (option a))
  (expected:(i:nat{i < S.length output}) -> GTot a)
  (done:nat{done <= S.length output}) : prop =
  forall (i:nat{i < done}). S.index output i == Some (expected i)

inline_for_extraction noextract
let source_mask_prefix_step (#a:Type0) (output:S.seq (option a))
  (expected:(i:nat{i < S.length output}) -> GTot a)
  (at:nat{at < S.length output})
  : Lemma
      (requires source_mask_prefix output expected at)
      (ensures source_mask_prefix (S.upd output at (Some (expected at))) expected (at + 1)) = ()

inline_for_extraction noextract
fn source_silu_product_body
  (input:PA.larray BF16.t 43008) (output:PA.larray BF16.t 21504)
  (#xv:erased (values BF16.t 43008))
  (#old:erased (values BF16.t 21504))
  preserves gpu ** PA.pts_to input xv
  requires PA.pts_to output old
  ensures exists* (result:values BF16.t 21504).
    PA.pts_to output result ** pure (source_silu_result xv result)
{
  open Pulse.Lib.Array;
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (out:values BF16.t 21504).
      row |-> r ** PA.pts_to output out **
      pure (source_flat_prefix out
        (fun i -> source_silu_cell xv (i / 3072) (i % 3072)) (r * 3072))
    decreases (7 - !row)
  {
    let r = !row;
    let mut column = 0sz;
    while (!column <^ 3072sz)
      invariant exists* (c:sz{c <= 3072}) (out:values BF16.t 21504).
        column |-> c ** PA.pts_to output out **
        pure (source_flat_prefix out
          (fun i -> source_silu_cell xv (i / 3072) (i % 3072)) (r * 3072 + c))
      decreases (3072 - !column)
    {
      let c = !column;
      let stored_gate = input.(r *^ 6144sz +^ c);
      let gate = SourceCasts.fcast #BF16.t #F32.t stored_gate;
      let activated = SourceCasts.fcast #F32.t #BF16.t
        (F.div gate (F.add (F.one #F32.t) (F.fexp (NativeNeg.neg gate))));
      let stored_up = input.(r *^ 6144sz +^ 3072sz +^ c);
      let result = SourceCasts.fcast #F32.t #BF16.t
        (F.mul (SourceCasts.fcast #BF16.t #F32.t activated)
          (SourceCasts.fcast #BF16.t #F32.t stored_up));
      assert pure (result == source_silu_cell xv r c);
      Kuiper.Math.div_mod_of_mul_add 3072 r c;
      with out. assert PA.pts_to output out;
      source_flat_prefix_step out (fun i -> source_silu_cell xv (i / 3072) (i % 3072))
        (r * 3072 + c);
      output.(r *^ 3072sz +^ c) <- result;
      column := c +^ 1sz;
    };
    row := r +^ 1sz;
  };
}

inline_for_extraction noextract
let source_addnorm_components (input:source_hidden_values)
  (row:Shared.index 7) (lane:Shared.index 128)
  (count:nat{count <= 2}) (sums:values F32.t 4) : prop =
  forall (j:Shared.index 4). S.index sums j == source_rms_component input row lane j count

inline_for_extraction noextract
let source_addnorm_component_prefix (input:source_hidden_values)
  (row:Shared.index 7) (lane:Shared.index 128)
  (count:nat{count < 2}) (done:nat{done <= 4}) (sums:values F32.t 4) : prop =
  forall (j:Shared.index 4).
    S.index sums j == source_rms_component input row lane j
      (if j < done then count + 1 else count)

inline_for_extraction noextract
let source_addnorm_component_step (input:source_hidden_values)
  (row:Shared.index 7) (lane:Shared.index 128)
  (count:nat{count < 2}) (at:Shared.index 4) (sums:values F32.t 4)
  : Lemma
      (requires source_addnorm_component_prefix input row lane count at sums)
      (ensures (
        let value = SourceCasts.fcast #BF16.t #F32.t
          (S.index input (row * 1024 + lane * 4 + count * 512 + at)) in
        source_addnorm_component_prefix input row lane count (at + 1)
          (S.upd sums at (F.add (S.index sums at) (F.mul value value))))) =
  source_rms_component_step input row lane at count

inline_for_extraction noextract
fn source_add_norm_body
  (weights:packed_weights_address)
  (left right combined normalized:PA.larray BF16.t 7168)
  (offset:sz{offset + 1024 <= weight_elements})
  (#wv:erased (values BF16.t weight_elements))
  (#lv #rv #old_combined #old_normalized:erased source_hidden_values)
  preserves gpu ** PA.pts_to weights wv ** PA.pts_to left lv ** PA.pts_to right rv
  requires PA.pts_to combined old_combined ** PA.pts_to normalized old_normalized
  ensures exists* (cv nv:source_hidden_values).
    PA.pts_to combined cv ** PA.pts_to normalized nv **
    pure (source_add_norm_result wv lv rv cv nv offset)
{
  open Pulse.Lib.Array;
  let expected = Ghost.hide (source_added_values lv rv);
  let mut values_buffer : PA.array F32.t = [| 1024sz |];
  let mut partial : PA.array F32.t = [| 128sz |];
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (cv nv:source_hidden_values).
      row |-> r ** PA.pts_to combined cv ** PA.pts_to normalized nv **
      PA.pts_to_uninit values_buffer 1024 ** PA.pts_to_uninit partial 128 **
      pure (source_flat_prefix cv (fun i -> S.index expected i) (r * 1024) /\
        source_rms_output_ready wv expected nv offset (r * 1024))
    decreases (7 - !row)
  {
    let r = !row;
    unfold PA.pts_to_uninit values_buffer 1024;
    let mut column = 0sz;
    while (!column <^ 1024sz)
      invariant exists* (c:sz{c <= 1024}) (cv:source_hidden_values)
        (vv:values (option F32.t) 1024).
        column |-> c ** PA.pts_to combined cv **
        PA.pts_to_mask values_buffer vv (fun _ -> True) **
        pure (source_flat_prefix cv (fun i -> S.index expected i) (r * 1024 + c) /\
          source_mask_prefix vv
            (fun i -> SourceCasts.fcast #BF16.t #F32.t (S.index expected (r * 1024 + i))) c)
      decreases (1024 - !column)
    {
      let c = !column;
      let a = left.(r *^ 1024sz +^ c);
      let b = right.(r *^ 1024sz +^ c);
      let value = SourceCasts.fcast #F32.t #BF16.t
        (F.add (SourceCasts.fcast #BF16.t #F32.t a)
          (SourceCasts.fcast #BF16.t #F32.t b));
      assert pure (value == S.index expected (r * 1024 + c));
      with cv. assert PA.pts_to combined cv;
      source_flat_prefix_step cv (fun i -> S.index expected i) (r * 1024 + c);
      combined.(r *^ 1024sz +^ c) <- value;
      with vv. assert PA.pts_to_mask values_buffer vv (fun _ -> True);
      source_mask_prefix_step vv
        (fun i -> SourceCasts.fcast #BF16.t #F32.t (S.index expected (r * 1024 + i))) c;
      PA.mask_write values_buffer c (SourceCasts.fcast #BF16.t #F32.t value);
      column := c +^ 1sz;
    };
    PA.from_mask values_buffer;
    with vv. assert PA.pts_to values_buffer vv;
    assert pure (S.length vv == 1024 /\
      (forall (i:Shared.index 1024).
        S.index vv i == SourceCasts.fcast #BF16.t #F32.t (S.index expected (r * 1024 + i))));
    unfold PA.pts_to_uninit partial 128;
    let mut lane = 0sz;
    while (!lane <^ 128sz)
      invariant exists* (l:sz{l <= 128}) (pv:values (option F32.t) 128).
        lane |-> l ** PA.pts_to_mask partial pv (fun _ -> True) **
        pure (source_mask_prefix pv (source_rms_lane expected r) l)
      decreases (128 - !lane)
    {
      let l = !lane;
      let initial:F32.t = F.zero;
      let mut sums = [| initial; 4sz |];
      let mut column = l *^ 4sz;
      while (!column <^ 1024sz)
        invariant exists*
          (c:sz{DeviceSize.v c == DeviceSize.v l * 4 \/
            DeviceSize.v c == DeviceSize.v l * 4 + 512 \/
            DeviceSize.v c == DeviceSize.v l * 4 + 1024})
          (sv:values F32.t 4).
          column |-> c ** PA.pts_to sums sv **
          pure (source_addnorm_components expected r l
            ((DeviceSize.v c - DeviceSize.v l * 4) / 512) sv)
        decreases (1536 - !column)
      {
        let c = !column;
        let count = Ghost.hide ((DeviceSize.v c - DeviceSize.v l * 4) / 512);
        let mut j = 0sz;
        while (!j <^ 4sz)
          invariant exists* (done:sz{done <= 4}) (sv:values F32.t 4).
            j |-> done ** PA.pts_to sums sv **
            pure (source_addnorm_component_prefix expected r l count done sv)
          decreases (4 - !j)
        {
          let at = !j;
          let value = values_buffer.(c +^ at);
          let previous = sums.(at);
          with sv. assert PA.pts_to sums sv;
          source_addnorm_component_step expected r l count at sv;
          sums.(at) <- F.add previous (F.mul value value);
          j := at +^ 1sz;
        };
        column := c +^ 512sz;
      };
      let a = sums.(0sz);
      let b = sums.(1sz);
      let c = sums.(2sz);
      let d = sums.(3sz);
      let value = F.add (F.add (F.add a b) c) d;
      assert pure (value == source_rms_lane expected r l);
      with pv. assert PA.pts_to_mask partial pv (fun _ -> True);
      source_mask_prefix_step pv (source_rms_lane expected r) l;
      PA.mask_write partial l value;
      lane := l +^ 1sz;
    };
    PA.from_mask partial;
    with pv. assert PA.pts_to partial pv;
    assert pure (S.length pv == 128 /\
      (forall (i:Shared.index 128). S.index pv i == source_rms_lane expected r i));
    let squared_sum = source_reduce_partial #128 partial 128sz #pv;
    source_rms_tree_from_leaves expected r pv 1 0;
    let scale = F.rsqrt
      (F.add (F.mul squared_sum (F.div (F.one #F32.t) (F.of_int #F32.t 1024L)))
        (F.of_literal #F32.t "0.000001"));
    assert pure (scale == source_rms_scale expected r);
    let mut column = 0sz;
    while (!column <^ 1024sz)
      invariant exists* (c:sz{c <= 1024}) (nv:source_hidden_values).
        column |-> c ** PA.pts_to normalized nv **
        pure (source_rms_output_ready wv expected nv offset (r * 1024 + c))
      decreases (1024 - !column)
    {
      let c = !column;
      let x = values_buffer.(c);
      let value = SourceCasts.fcast #F32.t #BF16.t (F.mul x scale);
      let weight = weights.(offset +^ c);
      let result = SourceCasts.fcast #F32.t #BF16.t
        (F.mul (SourceCasts.fcast #BF16.t #F32.t value)
          (SourceCasts.fcast #BF16.t #F32.t weight));
      assert pure (result == source_rms_cell wv expected offset r c);
      with nv. assert PA.pts_to normalized nv;
      source_rms_output_step wv expected nv offset r c;
      normalized.(r *^ 1024sz +^ c) <- result;
      column := c +^ 1sz;
    };
    PA.to_mask values_buffer;
    PA.to_mask partial;
    fold PA.pts_to_uninit values_buffer 1024;
    fold PA.pts_to_uninit partial 128;
    row := r +^ 1sz;
  };
  with cv. assert PA.pts_to combined cv;
  assert pure (S.equal cv expected);
  S.lemma_eq_elim cv expected;
}

module N = Kuiper.Approximates.Base
module Root = FStar.Math.Sqrt
module Exp = FStar.Math.Exp
module Sum = Kuiper.Sum
module Native = Kuiper.Float32.FastMath

noextract
let rec source_sum (n:nat) (f:nat -> GTot real) : GTot real (decreases n) =
  if n = 0 then 0.0R else source_sum (n - 1) f +. f (n - 1)

inline_for_extraction noextract
let source_sum_step (n:pos) (f:nat -> GTot real)
  : Lemma (source_sum n f == source_sum (n - 1) f +. f (n - 1)) = ()

inline_for_extraction noextract
let rec source_sum_ext (n:nat) (f g:nat -> GTot real)
  : Lemma
      (requires forall (i:nat{i < n}). f i == g i)
      (ensures source_sum n f == source_sum n g) (decreases n) =
  if n > 0 then source_sum_ext (n - 1) f g

inline_for_extraction noextract
let rec source_sum_add (n:nat) (f g:nat -> GTot real)
  : Lemma (source_sum n (fun i -> f i +. g i) == source_sum n f +. source_sum n g)
      (decreases n) =
  if n > 0 then source_sum_add (n - 1) f g

inline_for_extraction noextract
let rec source_sum_append (n m:nat) (f:nat -> GTot real)
  : Lemma (source_sum (n + m) f ==
      source_sum n f +. source_sum m (fun i -> f (n + i))) (decreases m) =
  if m > 0 then source_sum_append n (m - 1) f

inline_for_extraction noextract
let rec source_sum_even_odd (n:nat) (f:nat -> GTot real)
  : Lemma (source_sum (2 * n) f ==
      source_sum n (fun i -> f (2 * i)) +.
      source_sum n (fun i -> f (2 * i + 1))) (decreases n) =
  if n > 0 then source_sum_even_odd (n - 1) f

inline_for_extraction noextract
let rec source_sum_flat (n m:nat) (f:nat -> GTot real)
  : Lemma (source_sum n (fun i -> source_sum m (fun j -> f (i * m + j))) ==
      source_sum (n * m) f) (decreases n) =
  if n > 0 then (
    source_sum_flat (n - 1) m f;
    source_sum_append ((n - 1) * m) m f;
    assert ((n - 1) * m + m == n * m);
    source_sum_step n (fun i -> source_sum m (fun j -> f (i * m + j))))

inline_for_extraction noextract
let rec source_sum_zero (n:nat)
  : Lemma (source_sum n (fun _ -> 0.0R) == 0.0R) (decreases n) =
  if n > 0 then source_sum_zero (n - 1)

inline_for_extraction noextract
let rec source_sum_swap (n m:nat) (f:nat -> nat -> GTot real)
  : Lemma (source_sum n (fun i -> source_sum m (f i)) ==
      source_sum m (fun j -> source_sum n (fun i -> f i j))) (decreases n) =
  if n = 0 then (
    source_sum_zero m;
    source_sum_ext m (fun j -> source_sum n (fun i -> f i j)) (fun _ -> 0.0R);
    assert (source_sum n (fun i -> source_sum m (f i)) == 0.0R))
  else (
    source_sum_swap (n - 1) m f;
    source_sum_step n (fun i -> source_sum m (f i));
    source_sum_ext m (fun j -> source_sum n (fun i -> f i j))
      (fun j -> source_sum (n - 1) (fun i -> f i j) +. f (n - 1) j);
    source_sum_add m (fun j -> source_sum (n - 1) (fun i -> f i j)) (f (n - 1)))

inline_for_extraction noextract
let rec source_sum_shared (n:nat) (f:Shared.vector real n) (g:nat -> GTot real)
  : Lemma
      (requires forall (i:Shared.index n). f i == g i)
      (ensures Shared.sum n f == source_sum n g) (decreases n) =
  if n > 0 then (
    source_sum_shared (n - 1) f g;
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 n f)

noextract
let source_padded (n:pos) (f:Shared.vector real n) (i:nat) : GTot real =
  if i < n then f i else 0.0R

inline_for_extraction noextract
let rec source_sum_padding (n:pos{n <= 7})
  (count:nat{n <= count /\ count <= 8}) (f:Shared.vector real n)
  : Lemma (source_sum count (source_padded n f) == Shared.sum n f)
      (decreases count) =
  if count = n then source_sum_shared n f (source_padded n f)
  else (
    source_sum_padding n (count - 1) f;
    source_sum_step count (source_padded n f))

inline_for_extraction noextract
let source_sum_strided_128 (term:nat -> GTot real)
  : Lemma (source_sum 16 (fun lane -> source_sum 8 (fun i -> term (lane + 16 * i))) ==
      source_sum 128 term) =
  source_sum_swap 16 8 (fun lane i -> term (lane + 16 * i));
  let at (i:nat{i < 8})
    : Lemma (source_sum 16 (fun lane -> term (lane + 16 * i)) ==
        source_sum 16 (fun lane -> term (i * 16 + lane))) =
    source_sum_ext 16 (fun lane -> term (lane + 16 * i)) (fun lane -> term (i * 16 + lane))
  in
  C.forall_intro at;
  source_sum_ext 8 (fun i -> source_sum 16 (fun lane -> term (lane + 16 * i)))
    (fun i -> source_sum 16 (fun lane -> term (i * 16 + lane)));
  source_sum_flat 8 16 term

inline_for_extraction noextract
let source_tree_end (leaves:S.seq F32.t)
  (span:pos{source_tree_level span /\ span <= S.length leaves})
  (lane:nat{lane < span})
  : Lemma (source_tree leaves span span lane == S.index leaves lane) = ()

inline_for_extraction noextract
let source_tree_expand (leaves:S.seq F32.t)
  (span:pos{source_tree_level span /\ span <= S.length leaves})
  (level:pos{source_tree_level level /\ level < span})
  (lane:nat{lane < level})
  : Lemma (source_tree leaves span level lane ==
      F.add (source_tree leaves span (2 * level) lane)
        (source_tree leaves span (2 * level) (lane + level))) = ()

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
let rec source_tree_numeric (#capacity:nat) (leaves:values F32.t capacity)
  (span:pos{source_tree_level span /\ span <= capacity})
  (level:pos{source_tree_level level /\ level <= span})
  (lane:nat{lane < level}) (logical:nat -> GTot real)
  : Lemma
      (requires forall (i:nat{i < span}). N.v_approximates (S.index leaves i) (logical i))
      (ensures N.v_approximates (source_tree leaves span level lane)
        (source_sum (span / level) (fun i -> logical (lane + i * level))))
      (decreases (span - level)) =
  if level = span then (
    source_tree_end leaves span lane;
    source_sum_step 1 (fun i -> logical (lane + i * level)))
  else (
    source_tree_expand leaves span level lane;
    let next = 2 * level in
    let count = span / next in
    source_tree_numeric leaves span next lane logical;
    source_tree_numeric leaves span next (lane + level) logical;
    N.a_add (source_tree leaves span next lane) (source_tree leaves span next (lane + level))
      (source_sum count (fun i -> logical (lane + i * next)))
      (source_sum count (fun i -> logical (lane + level + i * next)));
    assert (span / level == 2 * count);
    source_sum_even_odd count (fun i -> logical (lane + i * level));
    source_sum_ext count (fun i -> logical (lane + (2 * i) * level))
      (fun i -> logical (lane + i * next));
    source_sum_ext count (fun i -> logical (lane + (2 * i + 1) * level))
      (fun i -> logical (lane + level + i * next)))
#pop-options

inline_for_extraction noextract
let source_fma_contract (a b c result:F32.t) : prop =
  forall (ar br cr:real).
    N.v_approximates a ar /\ N.v_approximates b br /\ N.v_approximates c cr ==>
    N.v_approximates result (ar *. br +. cr)

inline_for_extraction noextract
let source_qk_load (qk:values BF16.t 21504) : GTot (Shared.heads F32.t 24) =
  fun row head column ->
    SourceCasts.fcast #BF16.t #F32.t (S.index qk (row * 3072 + head * 128 + column))

inline_for_extraction noextract
let source_score_term (q:Shared.heads real 24)
  (row key:Shared.index 7) (head:Shared.index 16) (column:nat) : GTot real =
  if column < 128 then q row head column *. q key (16 + head / 2) column else 0.0R

inline_for_extraction noextract
let source_score_lane_relation (qk:values BF16.t 21504)
  (row key:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 16)
  (count:nat{count <= 8}) (acc:F32.t) : prop =
  forall (q:Shared.heads real 24).
    Shared.heads_approximates (source_qk_load qk) q ==>
    N.v_approximates acc
      (source_sum count (fun i -> source_score_term q row key head (lane + 16 * i)))

inline_for_extraction noextract
let source_score_lane_step (qk:values BF16.t 21504)
  (row key:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 16)
  (count:nat{count < 8}) (acc result:F32.t)
  : Lemma
      (requires
        source_score_lane_relation qk row key head lane count acc /\
        source_fma_contract
          (source_qk_load qk row head (lane + 16 * count))
          (source_qk_load qk key (16 + head / 2) (lane + 16 * count))
          acc result)
      (ensures source_score_lane_relation qk row key head lane (count + 1) result) =
  let related (q:Shared.heads real 24)
    : Lemma
        (requires Shared.heads_approximates (source_qk_load qk) q)
        (ensures N.v_approximates result
          (source_sum (count + 1) (fun i -> source_score_term q row key head (lane + 16 * i)))) =
    let column = lane + 16 * count in
    let previous = source_sum count (fun i -> source_score_term q row key head (lane + 16 * i)) in
    assert (N.v_approximates (source_qk_load qk row head column) (q row head column));
    assert (N.v_approximates (source_qk_load qk key (16 + head / 2) column)
      (q key (16 + head / 2) column));
    assert (N.v_approximates acc previous);
    assert (N.v_approximates result
      (q row head column *. q key (16 + head / 2) column +. previous));
    source_sum_step (count + 1) (fun i -> source_score_term q row key head (lane + 16 * i))
  in C.forall_intro (C.move_requires related)

inline_for_extraction noextract
fn source_score_lane_body
  (qk:PA.larray BF16.t 21504)
  (row key:szlt 7) (head lane:szlt 16)
  (#qv:erased (values BF16.t 21504))
  preserves gpu ** PA.pts_to qk qv
  returns result:F32.t
  ensures pure (source_score_lane_relation qv row key head lane 8 result)
{
  open Pulse.Lib.Array;
  let mut acc:F32.t = F.zero;
  let mut column:sz = lane;
  while (!column <^ 128sz)
    invariant exists* (c:sz{c >= lane /\ (c - lane) % 16 == 0 /\ c <= lane + 128})
      (value:F32.t).
      column |-> c ** acc |-> value **
      pure (source_score_lane_relation qv row key head lane ((c - lane) / 16) value)
    decreases (144 - !column)
  {
    let c = !column;
    let a = qk.(row *^ 3072sz +^ head *^ 128sz +^ c);
    let b = qk.(key *^ 3072sz +^ 2048sz +^ (head /^ 2sz) *^ 128sz +^ c);
    let x = SourceCasts.fcast #BF16.t #F32.t a;
    let y = SourceCasts.fcast #BF16.t #F32.t b;
    let previous = !acc;
    let result = Native.fma_rn x y previous;
    source_score_lane_step qv row key head lane ((c - lane) / 16) previous result;
    acc := result;
    column := c +^ 16sz;
  };
  !acc
}

inline_for_extraction noextract
let source_qk_partial (qkv:values BF16.t 28672)
  (row:Shared.index 7) (head:Shared.index 24) (lane:Shared.index 32) : F32.t =
  let a = SourceCasts.fcast #BF16.t #F32.t (S.index qkv (row * 4096 + head * 128 + lane * 4)) in
  let b = SourceCasts.fcast #BF16.t #F32.t (S.index qkv (row * 4096 + head * 128 + lane * 4 + 1)) in
  let c = SourceCasts.fcast #BF16.t #F32.t (S.index qkv (row * 4096 + head * 128 + lane * 4 + 2)) in
  let d = SourceCasts.fcast #BF16.t #F32.t (S.index qkv (row * 4096 + head * 128 + lane * 4 + 3)) in
  F.add (F.add (F.add (F.mul a a) (F.mul b b)) (F.mul c c)) (F.mul d d)

inline_for_extraction noextract
let source_qk_scale (qkv:values BF16.t 28672)
  (row:Shared.index 7) (head:Shared.index 24) : GTot F32.t =
  let leaves = S.init_ghost 32 (source_qk_partial qkv row head) in
  F.rsqrt (F.add (F.mul (source_tree leaves 32 1 0)
    (F.div (F.one #F32.t) (F.of_int #F32.t 128L))) (F.of_literal #F32.t "0.000001"))

inline_for_extraction noextract
let source_qk_normalized (weights:values BF16.t weight_elements)
  (qkv:values BF16.t 28672) (offset:nat{offset + 256 <= weight_elements})
  (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128) : GTot BF16.t =
  let value = SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t
      (S.index qkv (row * 4096 + head * 128 + column))) (source_qk_scale qkv row head)) in
  SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t value)
      (SourceCasts.fcast #BF16.t #F32.t
        (S.index weights (offset + (if head < 16 then 0 else 128) + column))))

inline_for_extraction noextract
let source_qk_cell (weights:values BF16.t weight_elements)
  (qkv:values BF16.t 28672) (rotary:source_rotary_values)
  (offset:nat{offset + 256 <= weight_elements})
  (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128) : GTot BF16.t =
  let value = source_qk_normalized weights qkv offset row head column in
  let partner = SourceCasts.fcast #BF16.t #F32.t
    (source_qk_normalized weights qkv offset row head ((column + 64) % 128)) in
  let rotated = if column < 64 then NativeNeg.neg partner else partner in
  let a = SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t value)
      (SourceCasts.fcast #BF16.t #F32.t (S.index rotary (row * 128 + column)))) in
  let b = SourceCasts.fcast #F32.t #BF16.t
    (F.mul rotated
      (SourceCasts.fcast #BF16.t #F32.t (S.index rotary (896 + row * 128 + column)))) in
  SourceCasts.fcast #F32.t #BF16.t
    (F.add (SourceCasts.fcast #BF16.t #F32.t a) (SourceCasts.fcast #BF16.t #F32.t b))

inline_for_extraction noextract
let source_qk_values (weights:values BF16.t weight_elements)
  (qkv:values BF16.t 28672) (rotary:source_rotary_values)
  (offset:nat{offset + 256 <= weight_elements}) : GTot (values BF16.t 21504) =
  S.init_ghost 21504 (fun i -> source_qk_cell weights qkv rotary offset
    (i / 3072) ((i % 3072) / 128) (i % 128))

inline_for_extraction noextract
let source_qk_index (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128)
  : Lemma (
    (row * 3072 + head * 128 + column) / 3072 == row /\
    ((row * 3072 + head * 128 + column) % 3072) / 128 == head /\
    (row * 3072 + head * 128 + column) % 128 == column) =
  Kuiper.Math.div_mod_of_mul_add 3072 row (head * 128 + column);
  Kuiper.Math.div_mod_of_mul_add 128 head column;
  Kuiper.Math.div_mod_of_mul_add 128 (row * 24 + head) column

inline_for_extraction noextract
fn source_prepare_qk
  (weights:packed_weights_address)
  (qkv:PA.larray BF16.t 28672) (rotary:PA.larray BF16.t 1792)
  (qk:PA.larray BF16.t 21504) (partial:PA.larray F32.t 32)
  (normalized:PA.larray BF16.t 128)
  (offset:sz{offset + 256 <= weight_elements})
  (#wv:erased (values BF16.t weight_elements))
  (#qv:erased (values BF16.t 28672)) (#rv:erased source_rotary_values)
  preserves gpu ** PA.pts_to weights wv ** PA.pts_to qkv qv ** PA.pts_to rotary rv
  requires PA.pts_to_uninit qk 21504 ** PA.pts_to_uninit partial 32 **
    PA.pts_to_uninit normalized 128
  ensures PA.pts_to qk (source_qk_values wv qv rv offset) **
    (exists* (pv:values F32.t 32). PA.pts_to partial pv) **
    PA.pts_to_uninit normalized 128
{
  open Pulse.Lib.Array;
  let expected = Ghost.hide (source_qk_values wv qv rv offset);
  unfold PA.pts_to_uninit qk 21504;
  unfold PA.pts_to_uninit partial 32;
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (out:values (option BF16.t) 21504)
      (pv:values (option F32.t) 32).
      row |-> r ** PA.pts_to_mask qk out (fun _ -> True) **
      PA.pts_to_mask partial pv (fun _ -> True) ** PA.pts_to_uninit normalized 128 **
      pure (source_mask_prefix out (fun i -> S.index expected i) (r * 3072) /\
        (r > 0 ==> forall (i:Shared.index 32). Some? (S.index pv i)))
    decreases (7 - !row)
  {
    let r = !row;
    let mut head = 0sz;
    while (!head <^ 24sz)
      invariant exists* (h:sz{h <= 24}) (out:values (option BF16.t) 21504)
        (pv:values (option F32.t) 32).
        head |-> h ** PA.pts_to_mask qk out (fun _ -> True) **
        PA.pts_to_mask partial pv (fun _ -> True) ** PA.pts_to_uninit normalized 128 **
        pure (source_mask_prefix out (fun i -> S.index expected i) (r * 3072 + h * 128) /\
          (h > 0 ==> forall (i:Shared.index 32). Some? (S.index pv i)))
      decreases (24 - !head)
    {
      let h = !head;
      let start = r *^ 4096sz +^ h *^ 128sz;
      let mut lane = 0sz;
      while (!lane <^ 32sz)
        invariant exists* (l:sz{l <= 32}) (pv:values (option F32.t) 32).
          lane |-> l ** PA.pts_to_mask partial pv (fun _ -> True) **
          pure (source_mask_prefix pv (source_qk_partial qv r h) l)
        decreases (32 - !lane)
      {
        let l = !lane;
        let a = SourceCasts.fcast #BF16.t #F32.t qkv.(start +^ l *^ 4sz);
        let b = SourceCasts.fcast #BF16.t #F32.t qkv.(start +^ l *^ 4sz +^ 1sz);
        let c = SourceCasts.fcast #BF16.t #F32.t qkv.(start +^ l *^ 4sz +^ 2sz);
        let d = SourceCasts.fcast #BF16.t #F32.t qkv.(start +^ l *^ 4sz +^ 3sz);
        let value = F.add (F.add (F.add (F.mul a a) (F.mul b b)) (F.mul c c)) (F.mul d d);
        assert pure (value == source_qk_partial qv r h l);
        with pv. assert PA.pts_to_mask partial pv (fun _ -> True);
        source_mask_prefix_step pv (source_qk_partial qv r h) l;
        PA.mask_write partial l value;
        lane := l +^ 1sz;
      };
      PA.from_mask partial;
      with pv. assert PA.pts_to partial pv;
      let leaves = Ghost.hide (S.init_ghost 32 (source_qk_partial qv r h));
      assert pure (S.equal pv leaves);
      S.lemma_eq_elim pv leaves;
      let squared_sum = source_reduce_partial #32 partial 32sz #pv;
      let scale = F.rsqrt (F.add
        (F.mul squared_sum (F.div (F.one #F32.t) (F.of_int #F32.t 128L)))
        (F.of_literal #F32.t "0.000001"));
      assert pure (scale == source_qk_scale qv r h);
      unfold PA.pts_to_uninit normalized 128;
      let mut column = 0sz;
      while (!column <^ 128sz)
        invariant exists* (c:sz{c <= 128}) (nv:values (option BF16.t) 128).
          column |-> c ** PA.pts_to_mask normalized nv (fun _ -> True) **
          pure (source_mask_prefix nv (source_qk_normalized wv qv offset r h) c)
        decreases (128 - !column)
      {
        let c = !column;
        let x = qkv.(start +^ c);
        let value = SourceCasts.fcast #F32.t #BF16.t
          (F.mul (SourceCasts.fcast #BF16.t #F32.t x) scale);
        let weight_offset = offset +^ (if h <^ 16sz then 0sz else 128sz) +^ c;
        let weight = weights.(weight_offset);
        let result = SourceCasts.fcast #F32.t #BF16.t
          (F.mul (SourceCasts.fcast #BF16.t #F32.t value)
            (SourceCasts.fcast #BF16.t #F32.t weight));
        assert pure (result == source_qk_normalized wv qv offset r h c);
        with nv. assert PA.pts_to_mask normalized nv (fun _ -> True);
        source_mask_prefix_step nv (source_qk_normalized wv qv offset r h) c;
        PA.mask_write normalized c result;
        column := c +^ 1sz;
      };
      PA.from_mask normalized;
      with nv. assert PA.pts_to normalized nv;
      let mut column = 0sz;
      while (!column <^ 128sz)
        invariant exists* (c:sz{c <= 128}) (out:values (option BF16.t) 21504).
          column |-> c ** PA.pts_to_mask qk out (fun _ -> True) **
          pure (source_mask_prefix out (fun i -> S.index expected i)
            (r * 3072 + h * 128 + c))
        decreases (128 - !column)
      {
        let c = !column;
        let other = normalized.((c +^ 64sz) %^ 128sz);
        let partner = SourceCasts.fcast #BF16.t #F32.t other;
        let rotated = (if c <^ 64sz then NativeNeg.neg partner else partner);
        let value = normalized.(c);
        let cosine = rotary.(r *^ 128sz +^ c);
        let a = SourceCasts.fcast #F32.t #BF16.t
          (F.mul (SourceCasts.fcast #BF16.t #F32.t value)
            (SourceCasts.fcast #BF16.t #F32.t cosine));
        let sine = rotary.(896sz +^ r *^ 128sz +^ c);
        let b = SourceCasts.fcast #F32.t #BF16.t
          (F.mul rotated (SourceCasts.fcast #BF16.t #F32.t sine));
        let result = SourceCasts.fcast #F32.t #BF16.t
          (F.add (SourceCasts.fcast #BF16.t #F32.t a)
            (SourceCasts.fcast #BF16.t #F32.t b));
        source_qk_index r h c;
        assert pure (result == S.index expected (r * 3072 + h * 128 + c));
        with out. assert PA.pts_to_mask qk out (fun _ -> True);
        source_mask_prefix_step out (fun i -> S.index expected i) (r * 3072 + h * 128 + c);
        PA.mask_write qk (r *^ 3072sz +^ h *^ 128sz +^ c) result;
        column := c +^ 1sz;
      };
      PA.to_mask normalized;
      fold PA.pts_to_uninit normalized 128;
      PA.to_mask partial;
      head := h +^ 1sz;
    };
    row := r +^ 1sz;
  };
  PA.from_mask qk;
  with out. assert PA.pts_to qk out;
  assert pure (S.equal out expected);
  S.lemma_eq_elim out expected;
  rewrite PA.pts_to qk out as PA.pts_to qk (source_qk_values wv qv rv offset);
  PA.from_mask partial;
  PA.pts_to_len partial;
}

inline_for_extraction noextract
let source_attention_factor_related ()
  : Lemma (N.v_approximates (F.of_literal #F32.t "0.08838834764831845")
      (Root.rsqrt 128.0R)) =
  let lo = 8838834764831844.0R /. 100000000000000000.0R in
  let hi = 8838834764831845.0R /. 100000000000000000.0R in
  Root.sqrt_square 128.0R;
  Root.sqrt_positive 128.0R;
  assert (Root.rsqrt 128.0R *. Root.rsqrt 128.0R *. 128.0R == 1.0R);
  assert (lo *. lo *. 128.0R <. 1.0R /\ hi *. hi *. 128.0R >. 1.0R);
  assert (lo <=. Root.rsqrt 128.0R /\ Root.rsqrt 128.0R <=. hi);
  assert_norm (Kuiper.Float32.Literals.common_normal_cell "0.08838834764831845"
    8838834764831844 100000000000000000 8838834764831845 100000000000000000
    11863283 (-27));
  F32.literal_interval_approx "0.08838834764831845"
    8838834764831844 100000000000000000 8838834764831845 100000000000000000
    11863283 (-27) (Root.rsqrt 128.0R)

inline_for_extraction noextract
let source_score_term_inside (q:Shared.heads real 24)
  (row key:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128)
  : Lemma (source_score_term q row key head column ==
      q row head column *. q key (16 + head / 2) column) = ()

#push-options "--fuel 0 --ifuel 0"
inline_for_extraction noextract
let source_score_lanes_sum (q:Shared.heads real 24)
  (row key:Shared.index 7) (head:Shared.index 16)
  : Lemma (
      source_sum 16 (fun lane -> source_sum 8
        (fun i -> source_score_term q row key head (lane + 16 * i))) ==
      Shared.dot (q row head) (q key (16 + head / 2))) =
  let term = fun column -> source_score_term q row key head column in
  C.forall_intro (source_score_term_inside q row key head);
  source_sum_strided_128 term;
  let same_lane (lane:nat{lane < 16})
    : Lemma (
        source_sum 8 (fun i -> source_score_term q row key head (lane + 16 * i)) ==
        source_sum 8 (fun i -> term (lane + 16 * i))) =
    source_sum_ext 8 (fun i -> source_score_term q row key head (lane + 16 * i))
      (fun i -> term (lane + 16 * i))
  in
  C.forall_intro same_lane;
  source_sum_ext 16
    (fun lane -> source_sum 8 (fun i -> source_score_term q row key head (lane + 16 * i)))
    (fun lane -> source_sum 8 (fun i -> term (lane + 16 * i)));
  assert (source_sum 16 (fun lane -> source_sum 8
    (fun i -> source_score_term q row key head (lane + 16 * i))) == source_sum 128 term);
  source_sum_shared 128 (fun d -> q row head d *. q key (16 + head / 2) d) term;
  assert (source_sum 128 term == Shared.sum 128 (fun d -> q row head d *. q key (16 + head / 2) d));
  assert (Shared.dot (q row head) (q key (16 + head / 2)) ==
    Shared.sum 128 (fun d -> q row head d *. q key (16 + head / 2) d)) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.dot]];
     FStar.Tactics.V2.trivial ())
#pop-options

inline_for_extraction noextract
let source_score_dot_relation (qk:values BF16.t 21504)
  (row key:Shared.index 7) (head:Shared.index 16) (value:F32.t) : prop =
  forall (q:Shared.heads real 24).
    Shared.heads_approximates (source_qk_load qk) q ==>
    N.v_approximates value (Shared.dot (q row head) (q key (16 + head / 2)))

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
let source_score_lane_at (qk:values BF16.t 21504)
  (row key:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 16)
  (value:F32.t) (q:Shared.heads real 24)
  : Lemma
      (requires source_score_lane_relation qk row key head lane 8 value /\
        Shared.heads_approximates (source_qk_load qk) q)
      (ensures N.v_approximates value
        (source_sum 8 (fun i -> source_score_term q row key head (lane + 16 * i)))) = ()

inline_for_extraction noextract
let source_score_tree_related (qk:values BF16.t 21504)
  (row key:Shared.index 7) (head:Shared.index 16) (leaves:values F32.t 32)
  : Lemma
      (requires forall (lane:Shared.index 16).
        source_score_lane_relation qk row key head lane 8 (S.index leaves lane))
      (ensures source_score_dot_relation qk row key head (source_tree leaves 16 1 0)) =
  let related (q:Shared.heads real 24)
    : Lemma
        (requires Shared.heads_approximates (source_qk_load qk) q)
        (ensures N.v_approximates (source_tree leaves 16 1 0)
          (Shared.dot (q row head) (q key (16 + head / 2)))) =
    let logical = fun (lane:nat) -> source_sum 8
      (fun i -> source_score_term q row key head (lane + 16 * i)) in
    let leaf (lane:Shared.index 16)
      : Lemma (N.v_approximates (S.index leaves lane) (logical lane)) =
      source_score_lane_at qk row key head lane (S.index leaves lane) q
    in
    C.forall_intro leaf;
    source_tree_numeric #32 leaves 16 1 0 logical;
    source_sum_ext 16 (fun i -> logical (0 + i * 1)) logical;
    source_sum_ext 16 logical
      (fun lane -> source_sum 8
        (fun i -> source_score_term q row key head (lane + 16 * i)));
    source_score_lanes_sum q row key head;
    assert (source_sum 16 logical == Shared.dot (q row head) (q key (16 + head / 2)))
  in C.forall_intro (C.move_requires related)
#pop-options

inline_for_extraction noextract
let source_scaled_score (value:F32.t) : F32.t =
  let dot = SourceCasts.fcast #F32.t #BF16.t value in
  let scaled = SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t dot)
      (F.of_literal #F32.t "0.08838834764831845")) in
  SourceCasts.fcast #BF16.t #F32.t scaled

inline_for_extraction noextract
let source_score_relation (qk:values BF16.t 21504)
  (row key:Shared.index 7) (head:Shared.index 16) (value:F32.t) : prop =
  forall (q:Shared.heads real 24).
    Shared.heads_approximates (source_qk_load qk) q ==>
    N.v_approximates value
      (Shared.dot (q row head) (q key (16 + head / 2)) *. Root.rsqrt 128.0R)

inline_for_extraction noextract
let source_scaled_score_related (qk:values BF16.t 21504)
  (row key:Shared.index 7) (head:Shared.index 16) (value:F32.t)
  : Lemma
      (requires source_score_dot_relation qk row key head value)
      (ensures source_score_relation qk row key head (source_scaled_score value)) =
  source_attention_factor_related ();
  let related (q:Shared.heads real 24)
    : Lemma
        (requires Shared.heads_approximates (source_qk_load qk) q)
        (ensures N.v_approximates (source_scaled_score value)
          (Shared.dot (q row head) (q key (16 + head / 2)) *. Root.rsqrt 128.0R)) =
    let real_dot = Shared.dot (q row head) (q key (16 + head / 2)) in
    let dot = SourceCasts.fcast #F32.t #BF16.t value in
    SourceCasts.fcast_approx #F32.t #BF16.t value real_dot;
    SourceCasts.fcast_approx #BF16.t #F32.t dot real_dot;
    let wide = SourceCasts.fcast #BF16.t #F32.t dot in
    let factor = F.of_literal #F32.t "0.08838834764831845" in
    N.a_mul wide factor real_dot (Root.rsqrt 128.0R);
    SourceCasts.fcast_approx #F32.t #BF16.t (F.mul wide factor) (real_dot *. Root.rsqrt 128.0R);
    SourceCasts.fcast_approx #BF16.t #F32.t
      (SourceCasts.fcast #F32.t #BF16.t (F.mul wide factor)) (real_dot *. Root.rsqrt 128.0R)
  in C.forall_intro (C.move_requires related)

inline_for_extraction noextract
fn source_attention_score
  (qk:PA.larray BF16.t 21504) (partial:PA.larray F32.t 32)
  (row key:szlt 7) (head:szlt 16)
  (#qv:erased (values BF16.t 21504))
  (#old:erased (S.seq F32.t){S.length old == 32})
  preserves gpu ** PA.pts_to qk qv
  requires PA.pts_to partial old
  returns result:F32.t
  ensures (exists* (pv:values F32.t 32). PA.pts_to partial pv) **
    pure (source_score_relation qv row key head result)
{
  open Pulse.Lib.Array;
  let mut lane = 0sz;
  while (!lane <^ 16sz)
    invariant exists* (l:sz{l <= 16}) (pv:values F32.t 32).
      lane |-> l ** PA.pts_to partial pv **
      pure (forall (i:nat{i < l}).
        source_score_lane_relation qv row key head i 8 (S.index pv i))
    decreases (16 - !lane)
  {
    let l = !lane;
    let value = source_score_lane_body qk row key head l #qv;
    partial.(l) <- value;
    lane := l +^ 1sz;
  };
  with leaves. assert PA.pts_to partial leaves;
  let snapshot = Ghost.hide #(S.seq F32.t) leaves;
  let value = source_reduce_partial #32 partial 16sz #snapshot;
  source_score_tree_related qv row key head leaves;
  source_scaled_score_related qv row key head value;
  source_scaled_score value
}

inline_for_extraction noextract
let source_scores_represented (row:Shared.index 7)
  (scores:values (option F32.t) 8) (logical:Shared.vector real (row + 1)) : prop =
  forall (i:Shared.index (row + 1)).
    exists (value:F32.t). S.index scores i == Some value /\ N.v_approximates value (logical i)

inline_for_extraction noextract
let source_probabilities_represented (row:Shared.index 7)
  (probabilities:values (option BF16.t) 7) (logical:Shared.vector real (row + 1)) : prop =
  forall (i:Shared.index (row + 1)).
    exists (value:BF16.t).
      S.index probabilities i == Some value /\
      N.v_approximates (SourceCasts.fcast #BF16.t #F32.t value) (logical i)

inline_for_extraction noextract
let source_shifted_sum (row:Shared.index 7)
  (logical:Shared.vector real (row + 1)) (maximum:real) : GTot real =
  Shared.sum (row + 1) (fun i -> Exp.exp (logical i -. maximum))

inline_for_extraction noextract
let source_exp_terms (row:Shared.index 7)
  (logical:Shared.vector real (row + 1)) (maximum:real) (i:nat) : GTot real =
  if i <= row then Exp.exp (logical i -. maximum) else 0.0R

inline_for_extraction noextract
let source_padded_exp_sum (row:Shared.index 7)
  (logical:Shared.vector real (row + 1)) (maximum:real)
  (count:nat{row + 1 <= count /\ count <= 8})
  : Lemma (source_sum count (source_exp_terms row logical maximum) ==
      source_shifted_sum row logical maximum) =
  let f = fun i -> Exp.exp (logical i -. maximum) in
  source_sum_padding (row + 1) count f;
  source_sum_ext count (source_exp_terms row logical maximum) (source_padded (row + 1) f)

inline_for_extraction noextract
let rec source_sum_divide (count:nat) (f:nat -> GTot real) (d:real{d >. 0.0R})
  : Lemma (source_sum count (fun i -> f i /. d) == source_sum count f /. d)
      (decreases count) =
  if count > 0 then (
    source_sum_divide (count - 1) f d;
    source_sum_step count (fun i -> f i /. d);
    source_sum_step count f)

inline_for_extraction noextract
let rec source_shared_sum_equal (n:nat) (left right:Shared.vector real n)
  : Lemma
      (requires forall (i:Shared.index n). left i == right i)
      (ensures Shared.sum n left == Shared.sum n right) (decreases n) =
  if n > 0 then (
    source_shared_sum_equal (n - 1) left right;
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 n left;
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 n right)

inline_for_extraction noextract
let source_ratio_cancel (a:real) (b:real{b >. 0.0R}) (c:real{c >. 0.0R})
  : Lemma ((a /. c) /. (b /. c) == a /. b) = ()

inline_for_extraction noextract
let source_prefix_softmax_at (row:Shared.index 7)
  (logical:Shared.vector real (row + 1)) (key:Shared.index (row + 1))
  : Lemma
      (requires Shared.sum (row + 1) (fun i -> Exp.exp (logical i)) >. 0.0R)
      (ensures Shared.prefix_softmax row logical key ==
        Exp.exp (logical key) /. Shared.sum (row + 1) (fun i -> Exp.exp (logical i))) =
  assert (Shared.softmax logical key ==
    Exp.exp (logical key) /. Shared.sum (row + 1) (fun i -> Exp.exp (logical i))) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.softmax]];
     FStar.Tactics.V2.trivial ());
  assert (Shared.prefix_softmax row logical key ==
    Exp.exp (logical key) /. Shared.sum (row + 1) (fun i -> Exp.exp (logical i))) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.prefix_softmax]];
     FStar.Tactics.V2.smt ())

#push-options "--fuel 4 --ifuel 1"
inline_for_extraction noextract
let source_shift_cancel (row:Shared.index 7)
  (logical:Shared.vector real (row + 1)) (maximum:real)
  : Lemma (
      source_shifted_sum row logical maximum >. 0.0R /\
      (forall (i:Shared.index (row + 1)).
        Exp.exp (logical i -. maximum) /. source_shifted_sum row logical maximum ==
          Shared.prefix_softmax row logical i)) =
  let plain:Shared.vector real (row + 1) = fun i -> Exp.exp (logical i) in
  let shifted:Shared.vector real (row + 1) = fun i -> Exp.exp (logical i -. maximum) in
  Shared.sum_positive (row + 1) plain;
  Shared.sum_positive (row + 1) shifted;
  let at (i:Shared.index (row + 1))
    : Lemma (shifted i == plain i /. Exp.exp maximum) =
    Exp.exp_sub (logical i) maximum
  in
  C.forall_intro at;
  source_shared_sum_equal (row + 1) shifted (fun i -> plain i /. Exp.exp maximum);
  let padded = source_padded (row + 1) plain in
  source_sum_shared (row + 1) plain padded;
  source_sum_shared (row + 1) (fun i -> plain i /. Exp.exp maximum)
    (fun i -> padded i /. Exp.exp maximum);
  source_sum_divide (row + 1) padded (Exp.exp maximum);
  assert (source_shifted_sum row logical maximum ==
    Shared.sum (row + 1) plain /. Exp.exp maximum);
  let probability (i:Shared.index (row + 1))
    : Lemma (Exp.exp (logical i -. maximum) /. source_shifted_sum row logical maximum ==
        Shared.prefix_softmax row logical i) =
    Exp.exp_sub (logical i) maximum;
    source_ratio_cancel (plain i) (Shared.sum (row + 1) plain) (Exp.exp maximum);
    source_prefix_softmax_at row logical i
  in C.forall_intro probability
#pop-options

inline_for_extraction noextract
let source_exponential_relation (row:Shared.index 7)
  (scores:values (option F32.t) 8) (maximum:F32.t)
  (key:Shared.index 8) (result:F32.t) : prop =
  if key <= row then
    forall (logical:Shared.vector real (row + 1)).
      source_scores_represented row scores logical ==>
      N.v_approximates result (Exp.exp (logical key -. N.to_real maximum))
  else result == F.zero #F32.t

inline_for_extraction noextract
let source_exponential_after_sub (row:Shared.index 7)
  (scores:values (option F32.t) 8) (maximum value shifted:F32.t)
  (key:Shared.index (row + 1))
  : Lemma
      (requires S.index scores key == Some value /\
        (forall (a b:real).
          N.v_approximates value a /\ N.v_approximates maximum b ==>
          N.v_approximates shifted (a -. b)))
      (ensures source_exponential_relation row scores maximum key (F.fexp shifted)) =
  N.to_real_ok maximum;
  let related (logical:Shared.vector real (row + 1))
    : Lemma
        (requires source_scores_represented row scores logical)
        (ensures N.v_approximates (F.fexp shifted) (Exp.exp (logical key -. N.to_real maximum))) =
    assert (N.v_approximates value (logical key));
    assert (N.v_approximates shifted (logical key -. N.to_real maximum));
    N.exp_approx shifted (logical key -. N.to_real maximum)
  in C.forall_intro (C.move_requires related)

inline_for_extraction noextract
let source_exponential_prefix (row:Shared.index 7)
  (scores:values (option F32.t) 8) (maximum:F32.t)
  (partial:values F32.t 32) (exponentials:values (option F32.t) 8)
  (count:nat{count <= 8}) : prop =
  forall (key:Shared.index 8{key < count}).
    S.index exponentials key == Some (S.index partial key) /\
    source_exponential_relation row scores maximum key (S.index partial key)

inline_for_extraction noextract
let source_exponential_write (row:Shared.index 7)
  (scores:values (option F32.t) 8) (maximum:F32.t)
  (partial:values F32.t 32) (exponentials:values (option F32.t) 8)
  (key:Shared.index 8) (value:F32.t)
  : Lemma
      (requires source_exponential_prefix row scores maximum partial exponentials key /\
        source_exponential_relation row scores maximum key value)
      (ensures source_exponential_prefix row scores maximum
        (S.upd partial key value) (S.upd exponentials key (Some value)) (key + 1)) = ()

inline_for_extraction noextract
let source_denominator_relation (row:Shared.index 7)
  (scores:values (option F32.t) 8) (maximum denominator:F32.t) : prop =
  forall (logical:Shared.vector real (row + 1)).
    source_scores_represented row scores logical ==>
    N.v_approximates denominator (source_shifted_sum row logical (N.to_real maximum))

inline_for_extraction noextract
let source_denominator_from_tree (row:Shared.index 7)
  (scores:values (option F32.t) 8) (maximum:F32.t) (leaves:values F32.t 32)
  : Lemma
      (requires forall (key:Shared.index 8).
        source_exponential_relation row scores maximum key (S.index leaves key))
      (ensures source_denominator_relation row scores maximum (source_tree leaves 8 1 0)) =
  let related (logical:Shared.vector real (row + 1))
    : Lemma
        (requires source_scores_represented row scores logical)
        (ensures N.v_approximates (source_tree leaves 8 1 0)
          (source_shifted_sum row logical (N.to_real maximum))) =
    let terms = source_exp_terms row logical (N.to_real maximum) in
    let cell (key:Shared.index 8)
      : Lemma (N.v_approximates (S.index leaves key) (terms key)) =
      if key > row then (let _ = N.a0 #F32.t in ())
    in
    C.forall_intro cell;
    source_tree_numeric #32 leaves 8 1 0 terms;
    source_sum_ext 8 (fun i -> terms (0 + i * 1)) terms;
    source_padded_exp_sum row logical (N.to_real maximum) 8
  in C.forall_intro (C.move_requires related)

inline_for_extraction noextract
let source_probability_relation (row:Shared.index 7)
  (scores:values (option F32.t) 8) (key:Shared.index (row + 1)) (value:BF16.t) : prop =
  forall (logical:Shared.vector real (row + 1)).
    source_scores_represented row scores logical ==>
    N.v_approximates (SourceCasts.fcast #BF16.t #F32.t value)
      (Shared.prefix_softmax row logical key)

inline_for_extraction noextract
let source_probability_value_related (row:Shared.index 7)
  (scores:values (option F32.t) 8) (maximum numerator denominator:F32.t)
  (key:Shared.index (row + 1))
  : Lemma
      (requires source_exponential_relation row scores maximum key numerator /\
        source_denominator_relation row scores maximum denominator)
      (ensures source_probability_relation row scores key
        (SourceCasts.fcast #F32.t #BF16.t (F.div numerator denominator))) =
  let related (logical:Shared.vector real (row + 1))
    : Lemma
        (requires source_scores_represented row scores logical)
        (ensures N.v_approximates
          (SourceCasts.fcast #BF16.t #F32.t
            (SourceCasts.fcast #F32.t #BF16.t (F.div numerator denominator)))
          (Shared.prefix_softmax row logical key)) =
    source_shift_cancel row logical (N.to_real maximum);
    let n = Exp.exp (logical key -. N.to_real maximum) in
    let d = source_shifted_sum row logical (N.to_real maximum) in
    assert (N.v_approximates numerator n /\ N.v_approximates denominator d);
    N.div_approx numerator denominator n d;
    SourceCasts.fcast_approx #F32.t #BF16.t (F.div numerator denominator) (n /. d);
    SourceCasts.fcast_approx #BF16.t #F32.t
      (SourceCasts.fcast #F32.t #BF16.t (F.div numerator denominator)) (n /. d)
  in C.forall_intro (C.move_requires related)

inline_for_extraction noextract
let source_probability_prefix (row:Shared.index 7)
  (scores:values (option F32.t) 8) (probabilities:values (option BF16.t) 7)
  (count:nat{count <= row + 1}) : prop =
  forall (key:Shared.index (row + 1){key < count}).
    exists (value:BF16.t).
      S.index probabilities key == Some value /\ source_probability_relation row scores key value

inline_for_extraction noextract
let source_probability_write (row:Shared.index 7)
  (scores:values (option F32.t) 8) (probabilities:values (option BF16.t) 7)
  (key:Shared.index (row + 1)) (value:BF16.t)
  : Lemma
      (requires source_probability_prefix row scores probabilities key /\
        source_probability_relation row scores key value)
      (ensures source_probability_prefix row scores
        (S.upd probabilities key (Some value)) (key + 1)) = ()

inline_for_extraction noextract
fn source_attention_softmax
  (scores:PA.larray F32.t 8) (exponentials:PA.larray F32.t 8)
  (partial:PA.larray F32.t 32) (probabilities:PA.larray BF16.t 7)
  (row:szlt 7)
  (#sv:erased (S.seq (option F32.t)){S.length sv == 8})
  (#old:erased (S.seq F32.t){S.length old == 32})
  preserves gpu ** PA.pts_to_mask scores sv (fun _ -> True)
  requires PA.pts_to_uninit exponentials 8 ** PA.pts_to_uninit probabilities 7 **
    PA.pts_to partial old **
    pure (forall (key:Shared.index (row + 1)). Some? (S.index sv key))
  ensures PA.pts_to_uninit exponentials 8 **
    (exists* (pv:values F32.t 32). PA.pts_to partial pv) **
    (exists* (prob:values (option BF16.t) 7).
      PA.pts_to_mask probabilities prob (fun _ -> True) **
      pure (source_probability_prefix row sv prob (row + 1)))
{
  open Pulse.Lib.Array;
  let initial = PA.mask_read scores 0sz;
  let mut maximum:F32.t = initial;
  let mut key = 1sz;
  while (!key <^ row +^ 1sz)
    invariant exists* (k:sz{1 <= k /\ k <= row + 1}) (m:F32.t).
      key |-> k ** maximum |-> m
    decreases (row + 1 - !key)
  {
    let k = !key;
    let value = PA.mask_read scores k;
    maximum := F.fmax !maximum value;
    key := k +^ 1sz;
  };
  let m = !maximum;
  unfold PA.pts_to_uninit exponentials 8;
  let mut key = 0sz;
  while (!key <^ 8sz)
    invariant exists* (k:sz{k <= 8}) (pv:values F32.t 32)
      (ev:values (option F32.t) 8).
      key |-> k ** PA.pts_to partial pv ** PA.pts_to_mask exponentials ev (fun _ -> True) **
      pure (source_exponential_prefix row sv m pv ev k)
    decreases (8 - !key)
  {
    let k = !key;
    let value = if (k <^ (row +^ 1sz)) {
      let score = PA.mask_read scores k;
      let shifted = Native.sub_rn score m;
      source_exponential_after_sub row sv m score shifted k;
      F.fexp shifted
    } else {
      let z:F32.t = F.zero;
      z
    };
    assert pure (source_exponential_relation row sv m k value);
    with pv. assert PA.pts_to partial pv;
    with ev. assert PA.pts_to_mask exponentials ev (fun _ -> True);
    source_exponential_write row sv m pv ev k value;
    PA.mask_write exponentials k value;
    partial.(k) <- value;
    key := k +^ 1sz;
  };
  with leaves. assert PA.pts_to partial leaves;
  PA.from_mask exponentials;
  with ev. assert PA.pts_to exponentials ev;
  assert pure (S.length ev == 8 /\
    (forall (key:Shared.index 8).
      source_exponential_relation row sv m key (S.index ev key)));
  let snapshot = Ghost.hide #(S.seq F32.t) leaves;
  let denominator = source_reduce_partial #32 partial 8sz #snapshot;
  source_denominator_from_tree row sv m leaves;
  unfold PA.pts_to_uninit probabilities 7;
  let mut key = 0sz;
  while (!key <^ row +^ 1sz)
    invariant exists* (k:sz{k <= row + 1}) (prob:values (option BF16.t) 7).
      key |-> k ** PA.pts_to_mask probabilities prob (fun _ -> True) **
      pure (source_probability_prefix row sv prob k)
    decreases (row + 1 - !key)
  {
    let k = !key;
    let numerator = exponentials.(k);
    let divisor = partial.(0sz);
    assert pure (divisor == denominator);
    source_probability_value_related row sv m numerator divisor k;
    let value = SourceCasts.fcast #F32.t #BF16.t (F.div numerator divisor);
    with prob. assert PA.pts_to_mask probabilities prob (fun _ -> True);
    source_probability_write row sv prob k value;
    PA.mask_write probabilities k value;
    key := k +^ 1sz;
  };
  PA.to_mask exponentials;
  fold PA.pts_to_uninit exponentials 8;
}

inline_for_extraction noextract
let source_v_load (qkv:values BF16.t 28672) : GTot (Shared.heads F32.t 8) =
  fun row head column -> SourceCasts.fcast #BF16.t #F32.t
    (S.index qkv (row * 4096 + 3072 + head * 128 + column))

inline_for_extraction noextract
let source_mixed_term (row:Shared.index 7) (head:Shared.index 16)
  (column:Shared.index 128) (v:Shared.heads real 8)
  (probability:Shared.vector real (row + 1)) (key:nat) : GTot real =
  if key <= row then probability key *. v key (head / 2) column else 0.0R

inline_for_extraction noextract
let source_mixed_prefix (qkv:values BF16.t 28672)
  (probabilities:values (option BF16.t) 7)
  (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128)
  (count:nat{count <= row + 1}) (acc:F32.t) : prop =
  forall (v:Shared.heads real 8) (probability:Shared.vector real (row + 1)).
    Shared.heads_approximates (source_v_load qkv) v /\
    source_probabilities_represented row probabilities probability ==>
    N.v_approximates acc (source_sum count (source_mixed_term row head column v probability))

inline_for_extraction noextract
let source_mixed_step (qkv:values BF16.t 28672)
  (probabilities:values (option BF16.t) 7)
  (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128)
  (count:Shared.index (row + 1)) (prob:BF16.t) (acc result:F32.t)
  : Lemma
      (requires source_mixed_prefix qkv probabilities row head column count acc /\
        S.index probabilities count == Some prob /\
        source_fma_contract (SourceCasts.fcast #BF16.t #F32.t prob)
          (source_v_load qkv count (head / 2) column) acc result)
      (ensures source_mixed_prefix qkv probabilities row head column (count + 1) result) =
  let related (v:Shared.heads real 8) (probability:Shared.vector real (row + 1))
    : Lemma
        (requires Shared.heads_approximates (source_v_load qkv) v /\
          source_probabilities_represented row probabilities probability)
        (ensures N.v_approximates result
          (source_sum (count + 1) (source_mixed_term row head column v probability))) =
    let previous = source_sum count (source_mixed_term row head column v probability) in
    assert (N.v_approximates (SourceCasts.fcast #BF16.t #F32.t prob) (probability count));
    assert (N.v_approximates (source_v_load qkv count (head / 2) column)
      (v count (head / 2) column));
    assert (N.v_approximates acc previous);
    assert (N.v_approximates result
      (probability count *. v count (head / 2) column +. previous));
    source_sum_step (count + 1) (source_mixed_term row head column v probability)
  in C.forall_intro_2 (fun v -> C.move_requires (related v))

inline_for_extraction noextract
let source_mixed_value_relation (qkv:values BF16.t 28672)
  (probabilities:values (option BF16.t) 7)
  (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128)
  (value:BF16.t) : prop =
  forall (v:Shared.heads real 8) (probability:Shared.vector real (row + 1)).
    Shared.heads_approximates (source_v_load qkv) v /\
    source_probabilities_represented row probabilities probability ==>
    N.v_approximates value
      (source_sum (row + 1) (source_mixed_term row head column v probability))

inline_for_extraction noextract
let source_mixed_cast (qkv:values BF16.t 28672)
  (probabilities:values (option BF16.t) 7)
  (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128)
  (acc:F32.t)
  : Lemma
      (requires source_mixed_prefix qkv probabilities row head column (row + 1) acc)
      (ensures source_mixed_value_relation qkv probabilities row head column
        (SourceCasts.fcast #F32.t #BF16.t acc)) =
  let related (v:Shared.heads real 8) (probability:Shared.vector real (row + 1))
    : Lemma
        (requires Shared.heads_approximates (source_v_load qkv) v /\
          source_probabilities_represented row probabilities probability)
        (ensures N.v_approximates (SourceCasts.fcast #F32.t #BF16.t acc)
          (source_sum (row + 1) (source_mixed_term row head column v probability))) =
    SourceCasts.fcast_approx #F32.t #BF16.t acc
      (source_sum (row + 1) (source_mixed_term row head column v probability))
  in C.forall_intro_2 (fun v -> C.move_requires (related v))

inline_for_extraction noextract
fn source_attention_mixed_value
  (qkv:PA.larray BF16.t 28672) (probabilities:PA.larray BF16.t 7)
  (row:szlt 7) (head:szlt 16) (column:szlt 128)
  (#qv:erased (values BF16.t 28672))
  (#pm:erased (S.seq (option BF16.t)){S.length pm == 7})
  preserves gpu ** PA.pts_to qkv qv ** PA.pts_to_mask probabilities pm (fun _ -> True)
  requires pure (forall (key:Shared.index (row + 1)). Some? (S.index pm key))
  returns result:BF16.t
  ensures pure (source_mixed_value_relation qv pm row head column result)
{
  open Pulse.Lib.Array;
  let mut acc:F32.t = F.zero;
  let mut key = 0sz;
  while (!key <^ (row +^ 1sz))
    invariant exists* (k:sz{k <= row + 1}) (value:F32.t).
      key |-> k ** acc |-> value **
      pure (source_mixed_prefix qv pm row head column k value)
    decreases (row + 1 - !key)
  {
    let k = !key;
    let p = PA.mask_read probabilities k;
    let v = qkv.(k *^ 4096sz +^ 3072sz +^ (head /^ 2sz) *^ 128sz +^ column);
    let pf = SourceCasts.fcast #BF16.t #F32.t p;
    let vf = SourceCasts.fcast #BF16.t #F32.t v;
    let previous = !acc;
    let value = Native.fma_rn pf vf previous;
    source_mixed_step qv pm row head column k p previous value;
    acc := value;
    key := k +^ 1sz;
  };
  let value = !acc;
  source_mixed_cast qv pm row head column value;
  SourceCasts.fcast #F32.t #BF16.t value
}

inline_for_extraction noextract
let source_attention_scores_prefix (qk:values BF16.t 21504)
  (row:Shared.index 7) (head:Shared.index 16) (scores:values (option F32.t) 8)
  (count:nat{count <= row + 1}) : prop =
  forall (key:Shared.index (row + 1){key < count}).
    exists (value:F32.t). S.index scores key == Some value /\
      source_score_relation qk row key head value

inline_for_extraction noextract
let source_attention_score_write (qk:values BF16.t 21504)
  (row:Shared.index 7) (head:Shared.index 16) (scores:values (option F32.t) 8)
  (key:Shared.index (row + 1)) (value:F32.t)
  : Lemma
      (requires source_attention_scores_prefix qk row head scores key /\
        source_score_relation qk row key head value)
      (ensures source_attention_scores_prefix qk row head
        (S.upd scores key (Some value)) (key + 1)) = ()

inline_for_extraction noextract
let source_attention_score_model (q:Shared.heads real 24)
  (row:Shared.index 7) (head:Shared.index 16)
  : GTot (Shared.vector real (row + 1)) =
  fun key -> Shared.dot (q row head) (q key (16 + head / 2)) *. Root.rsqrt 128.0R

inline_for_extraction noextract
let source_attention_cell_relation (qkv:values BF16.t 28672) (qk:values BF16.t 21504)
  (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128) (value:BF16.t) : prop =
  forall (q:Shared.heads real 24) (v:Shared.heads real 8).
    Shared.heads_approximates (source_qk_load qk) q /\
    Shared.heads_approximates (source_v_load qkv) v ==>
    N.v_approximates value
      (source_sum (row + 1) (source_mixed_term row head column v
        (fun key -> Shared.prefix_softmax row (source_attention_score_model q row head) key)))

inline_for_extraction noextract
let source_attention_cell_from_parts (qkv:values BF16.t 28672) (qk:values BF16.t 21504)
  (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128)
  (scores:values (option F32.t) 8) (probabilities:values (option BF16.t) 7) (value:BF16.t)
  : Lemma
      (requires
        source_attention_scores_prefix qk row head scores (row + 1) /\
        source_probability_prefix row scores probabilities (row + 1) /\
        source_mixed_value_relation qkv probabilities row head column value)
      (ensures source_attention_cell_relation qkv qk row head column value) =
  let related (q:Shared.heads real 24) (v:Shared.heads real 8)
    : Lemma
        (requires Shared.heads_approximates (source_qk_load qk) q /\
          Shared.heads_approximates (source_v_load qkv) v)
        (ensures N.v_approximates value
          (source_sum (row + 1) (source_mixed_term row head column v
            (fun key -> Shared.prefix_softmax row (source_attention_score_model q row head) key)))) =
    let logical = source_attention_score_model q row head in
    assert (source_scores_represented row scores logical);
    assert (source_probabilities_represented row probabilities
      (fun key -> Shared.prefix_softmax row logical key))
  in C.forall_intro_2 (fun q -> C.move_requires (related q))

inline_for_extraction noextract
let source_attention_result
  (weights:values BF16.t weight_elements) (qkv:values BF16.t 28672)
  (rotary:source_rotary_values) (qk:values BF16.t 21504)
  (output:values BF16.t 14336) (offset:nat{offset + 256 <= weight_elements}) : GTot prop =
  qk == source_qk_values weights qkv rotary offset /\
  (forall (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128).
    source_attention_cell_relation qkv qk row head column
      (S.index output (row * 2048 + head * 128 + column)))

inline_for_extraction noextract
let source_attention_output_ready (qkv:values BF16.t 28672) (qk:values BF16.t 21504)
  (output:values BF16.t 14336) (done:nat{done <= 14336}) : prop =
  forall (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128).
    (head * 7 + row) * 128 + column < done ==>
    source_attention_cell_relation qkv qk row head column
      (S.index output (row * 2048 + head * 128 + column))

inline_for_extraction noextract
let source_attention_output_step (qkv:values BF16.t 28672) (qk:values BF16.t 21504)
  (output:values BF16.t 14336)
  (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128) (value:BF16.t)
  : Lemma
      (requires source_attention_output_ready qkv qk output ((head * 7 + row) * 128 + column) /\
        source_attention_cell_relation qkv qk row head column value)
      (ensures source_attention_output_ready qkv qk
        (S.upd output (row * 2048 + head * 128 + column) value)
        ((head * 7 + row) * 128 + column + 1)) =
  let point (r:Shared.index 7) (h:Shared.index 16) (c:Shared.index 128)
    : Lemma ((h * 7 + r) * 128 + c < (head * 7 + row) * 128 + column + 1 ==>
        source_attention_cell_relation qkv qk r h c
          (S.index (S.upd output (row * 2048 + head * 128 + column) value)
            (r * 2048 + h * 128 + c))) =
    source_row_major_unique 2048 r row (h * 128 + c) (head * 128 + column);
    source_row_major_unique 128 h head c column;
    source_updated_index output (row * 2048 + head * 128 + column) value
      (r * 2048 + h * 128 + c)
  in C.forall_intro_3 point

inline_for_extraction noextract
fn source_attention_body
  (weights:packed_weights_address)
  (qkv:PA.larray BF16.t 28672) (rotary:PA.larray BF16.t 1792)
  (output:PA.larray BF16.t 14336)
  (offset:sz{offset + 256 <= weight_elements})
  (#wv:erased (values BF16.t weight_elements))
  (#qv:erased (values BF16.t 28672)) (#rv:erased source_rotary_values)
  (#old:erased (values BF16.t 14336))
  preserves gpu ** PA.pts_to weights wv ** PA.pts_to qkv qv ** PA.pts_to rotary rv
  requires PA.pts_to output old
  returns observation:erased (values BF16.t 21504)
  ensures exists* (result:values BF16.t 14336).
    PA.pts_to output result ** pure (source_attention_result wv qv rv observation result offset)
{
  open Pulse.Lib.Array;
  let mut qk : PA.array BF16.t = [| 21504sz |];
  let mut partial : PA.array F32.t = [| 32sz |];
  let mut normalized : PA.array BF16.t = [| 128sz |];
  source_prepare_qk weights qkv rotary qk partial normalized offset #wv #qv #rv;
  let observed = Ghost.hide (source_qk_values wv qv rv offset);
  rewrite PA.pts_to qk (source_qk_values wv qv rv offset) as PA.pts_to qk observed;
  let mut scores : PA.array F32.t = [| 8sz |];
  let mut exponentials : PA.array F32.t = [| 8sz |];
  let mut probabilities : PA.array BF16.t = [| 7sz |];
  let mut head = 0sz;
  while (!head <^ 16sz)
    invariant exists* (h:sz{h <= 16}) (out:values BF16.t 14336)
      (pv:values F32.t 32).
      head |-> h ** PA.pts_to output out ** PA.pts_to partial pv **
      PA.pts_to_uninit scores 8 ** PA.pts_to_uninit exponentials 8 **
      PA.pts_to_uninit probabilities 7 **
      pure (source_attention_output_ready qv observed out (h * 7 * 128))
    decreases (16 - !head)
  {
    let h = !head;
    let mut row = 0sz;
    while (!row <^ 7sz)
      invariant exists* (r:sz{r <= 7}) (out:values BF16.t 14336)
        (pv:values F32.t 32).
        row |-> r ** PA.pts_to output out ** PA.pts_to partial pv **
        PA.pts_to_uninit scores 8 ** PA.pts_to_uninit exponentials 8 **
        PA.pts_to_uninit probabilities 7 **
        pure (source_attention_output_ready qv observed out ((h * 7 + r) * 128))
      decreases (7 - !row)
    {
      let r = !row;
      unfold PA.pts_to_uninit scores 8;
      let mut key = 0sz;
      while (!key <^ (r +^ 1sz))
        invariant exists* (k:sz{k <= r + 1}) (pv:values F32.t 32)
          (sv:values (option F32.t) 8).
          key |-> k ** PA.pts_to partial pv ** PA.pts_to_mask scores sv (fun _ -> True) **
          pure (source_attention_scores_prefix observed r h sv k)
        decreases (r + 1 - !key)
      {
        let k = !key;
        with pv. assert PA.pts_to partial pv;
        let partial_values = Ghost.hide #(S.seq F32.t) pv;
        let value = source_attention_score qk partial r k h #observed #partial_values;
        with sv. assert PA.pts_to_mask scores sv (fun _ -> True);
        source_attention_score_write observed r h sv k value;
        PA.mask_write scores k value;
        key := k +^ 1sz;
      };
      with sv. assert PA.pts_to_mask scores sv (fun _ -> True);
      with pv. assert PA.pts_to partial pv;
      assert pure (forall (k:Shared.index (r + 1)). Some? (S.index sv k));
      let score_values = Ghost.hide #(S.seq (option F32.t)) sv;
      let partial_values = Ghost.hide #(S.seq F32.t) pv;
      source_attention_softmax scores exponentials partial probabilities r #score_values #partial_values;
      with pm. assert PA.pts_to_mask probabilities pm (fun _ -> True);
      assert pure (forall (k:Shared.index (r + 1)). Some? (S.index pm k));
      let probability_values = Ghost.hide #(S.seq (option BF16.t)) pm;
      let mut column = 0sz;
      while (!column <^ 128sz)
        invariant exists* (c:sz{c <= 128}) (out:values BF16.t 14336).
          column |-> c ** PA.pts_to output out **
          pure (source_attention_output_ready qv observed out ((h * 7 + r) * 128 + c))
        decreases (128 - !column)
      {
        let c = !column;
        let value = source_attention_mixed_value qkv probabilities r h c #qv #probability_values;
        source_attention_cell_from_parts qv observed r h c sv pm value;
        with out. assert PA.pts_to output out;
        source_attention_output_step qv observed out r h c value;
        output.(r *^ 2048sz +^ h *^ 128sz +^ c) <- value;
        column := c +^ 1sz;
      };
      fold PA.pts_to_uninit scores 8;
      fold PA.pts_to_uninit probabilities 7;
      row := r +^ 1sz;
    };
    head := h +^ 1sz;
  };
  PA.to_mask qk;
  PA.to_mask partial;
  fold PA.pts_to_uninit qk 21504;
  fold PA.pts_to_uninit partial 32;
  observed
}

inline_for_extraction noextract
let source_sum_four (f:nat -> GTot real)
  : Lemma (source_sum 4 f == ((f 0 +. f 1) +. f 2) +. f 3) =
  source_sum_step 4 f;
  source_sum_step 3 f;
  source_sum_step 2 f;
  source_sum_step 1 f

inline_for_extraction noextract
let source_sum_strided (width count:nat) (term:nat -> GTot real)
  : Lemma (
      source_sum width (fun lane -> source_sum count (fun i -> term (lane + width * i))) ==
      source_sum (width * count) term) =
  source_sum_swap width count (fun lane i -> term (lane + width * i));
  let at (i:nat{i < count})
    : Lemma (source_sum width (fun lane -> term (lane + width * i)) ==
        source_sum width (fun lane -> term (i * width + lane))) =
    source_sum_ext width (fun lane -> term (lane + width * i))
      (fun lane -> term (i * width + lane))
  in
  C.forall_intro at;
  source_sum_ext count
    (fun i -> source_sum width (fun lane -> term (lane + width * i)))
    (fun i -> source_sum width (fun lane -> term (i * width + lane)));
  source_sum_flat count width term

inline_for_extraction noextract
let source_epsilon_related ()
  : Lemma (N.v_approximates (F.of_literal #F32.t "0.000001") 0.000001R) =
  assert_norm (Kuiper.Float32.Literals.common_normal_cell
    "0.000001" 1 1000000 1 1000000 8796093 (-43));
  F32.literal_interval_approx "0.000001" 1 1000000 1 1000000 8796093 (-43) 0.000001R

inline_for_extraction noextract
let source_scale_related (squared_sum:F32.t) (width:I64.t{I64.v width > 0})
  (squares:real{squares >=. 0.0R})
  : Lemma
      (requires N.v_approximates squared_sum squares)
      (ensures N.v_approximates
        (F.rsqrt (F.add
          (F.mul squared_sum (F.div (F.one #F32.t) (F.of_int #F32.t width)))
          (F.of_literal #F32.t "0.000001")))
        (Root.rsqrt (squares /. FStar.Real.of_int (I64.v width) +. 0.000001R))) =
  source_epsilon_related ();
  let denominator = F.of_int #F32.t width in
  let reciprocal = F.div (F.one #F32.t) denominator in
  let real_width = FStar.Real.of_int (I64.v width) in
  N.of_int_approx #F32.t width;
  N.div_approx (F.one #F32.t) denominator 1.0R real_width;
  N.a_mul squared_sum reciprocal squares (1.0R /. real_width);
  N.a_add (F.mul squared_sum reciprocal) (F.of_literal #F32.t "0.000001")
    (squares /. real_width) 0.000001R;
  N.rsqrt_approx
    (F.add (F.mul squared_sum reciprocal) (F.of_literal #F32.t "0.000001"))
    (squares /. real_width +. 0.000001R)

inline_for_extraction noextract
let source_normalized_scalar (x weight:BF16.t) (scale:F32.t) : BF16.t =
  let normalized = SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t x) scale) in
  SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t normalized)
      (SourceCasts.fcast #BF16.t #F32.t weight))

inline_for_extraction noextract
let source_normalized_scalar_related (x weight:BF16.t) (scale:F32.t) (xr wr sr:real)
  : Lemma
      (requires source_loaded_approximates x xr /\ N.v_approximates weight wr /\
        N.v_approximates scale sr)
      (ensures N.v_approximates (source_normalized_scalar x weight scale) (wr *. (xr *. sr))) =
  source_loaded_widen x xr;
  let product = F.mul (SourceCasts.fcast #BF16.t #F32.t x) scale in
  N.a_mul (SourceCasts.fcast #BF16.t #F32.t x) scale xr sr;
  SourceCasts.fcast_approx #F32.t #BF16.t product (xr *. sr);
  SourceCasts.fcast_approx #BF16.t #F32.t
    (SourceCasts.fcast #F32.t #BF16.t product) (xr *. sr);
  SourceCasts.fcast_approx #BF16.t #F32.t weight wr;
  let normalized = SourceCasts.fcast #BF16.t #F32.t
    (SourceCasts.fcast #F32.t #BF16.t product) in
  let wide_weight = SourceCasts.fcast #BF16.t #F32.t weight in
  N.a_mul normalized wide_weight (xr *. sr) wr;
  SourceCasts.fcast_approx #F32.t #BF16.t (F.mul normalized wide_weight) ((xr *. sr) *. wr)

noextract
let source_square_at (#width:pos) (x:Shared.vector real width) (i:nat) : GTot real =
  if i < width then x i *. x i else 0.0R

noextract
let source_real_squares (#width:pos) (x:Shared.vector real width) : GTot Root.rnonneg =
  let squares:Shared.vector real width = fun i -> x i *. x i in
  Shared.sum_nonnegative width squares;
  Shared.sum width squares

inline_for_extraction noextract
let source_real_squares_equation (#width:pos) (x:Shared.vector real width)
  : Lemma (source_real_squares #width x == Shared.sum width (fun i -> x i *. x i)) =
  assert (source_real_squares #width x == Shared.sum width (fun i -> x i *. x i)) by
    (FStar.Tactics.V2.norm [delta_only [`%source_real_squares]];
     FStar.Tactics.V2.trivial ())

noextract
let source_real_scale (#width:pos) (x:Shared.vector real width) : GTot real =
  Root.rsqrt (source_real_squares #width x /. FStar.Real.of_int width +. 0.000001R)

inline_for_extraction noextract
let source_frozen_rms_point (#width:pos) (weight x:Shared.vector real width)
  (i:Shared.index width)
  : Lemma (Shared.rms_norm #width weight x i == weight i *. (x i *. source_real_scale #width x)) =
  let squares:Shared.vector real width = fun j -> x j *. x j in
  Shared.sum_nonnegative width squares;
  assert_norm (Shared.rms_norm #width weight x i ==
    weight i *. (x i *. Root.rsqrt
      (Shared.sum width squares /. FStar.Real.of_int width +. 0.000001R)))

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
let rec source_rms_component_related (input:source_hidden_values)
  (row:Shared.index 7) (x:Shared.vector real 1024)
  (lane:Shared.index 128) (component:Shared.index 4) (count:nat{count <= 2})
  : Lemma
      (requires forall (i:Shared.index 1024).
        source_loaded_approximates (S.index input (row * 1024 + i)) (x i))
      (ensures N.v_approximates (source_rms_component input row lane component count)
        (source_sum count (fun i -> source_square_at #1024 x (lane * 4 + component + i * 512))))
      (decreases count) =
  if count > 0 then (
    let previous = count - 1 in
    let index = lane * 4 + component + previous * 512 in
    let native = S.index input (row * 1024 + index) in
    let wide = SourceCasts.fcast #BF16.t #F32.t native in
    let terms = fun (i:nat) -> source_square_at #1024 x (lane * 4 + component + i * 512) in
    source_rms_component_related input row x lane component previous;
    source_loaded_widen native (x index);
    N.a_mul wide wide (x index) (x index);
    N.a_add (source_rms_component input row lane component previous) (F.mul wide wide)
      (source_sum previous terms) (source_square_at #1024 x index);
    source_rms_component_step input row lane component previous;
    source_sum_step count terms)
#pop-options

inline_for_extraction noextract
let source_rms_lane_related (input:source_hidden_values) (row:Shared.index 7)
  (x:Shared.vector real 1024) (lane:Shared.index 128)
  : Lemma
      (requires forall (i:Shared.index 1024).
        source_loaded_approximates (S.index input (row * 1024 + i)) (x i))
      (ensures N.v_approximates (source_rms_lane input row lane)
        (source_sum 4 (fun component ->
          source_sum 2 (fun part -> source_square_at #1024 x (lane * 4 + component + part * 512))))) =
  let term = fun (component:nat) ->
    source_sum 2 (fun part -> source_square_at #1024 x (lane * 4 + component + part * 512)) in
  source_rms_component_related input row x lane 0 2;
  source_rms_component_related input row x lane 1 2;
  source_rms_component_related input row x lane 2 2;
  source_rms_component_related input row x lane 3 2;
  let a = source_rms_component input row lane 0 2 in
  let b = source_rms_component input row lane 1 2 in
  let c = source_rms_component input row lane 2 2 in
  let d = source_rms_component input row lane 3 2 in
  N.a_add a b (term 0) (term 1);
  N.a_add (F.add a b) c (term 0 +. term 1) (term 2);
  N.a_add (F.add (F.add a b) c) d ((term 0 +. term 1) +. term 2) (term 3);
  source_sum_four term

inline_for_extraction noextract
let source_four_terms (term:nat -> GTot real) (start:nat) : GTot real =
  ((term start +. term (start + 1)) +. term (start + 2)) +. term (start + 3)

inline_for_extraction noextract
let source_four_terms_sum (term:nat -> GTot real) (start:nat)
  : Lemma (source_four_terms term start == source_sum 4 (fun i -> term (start + i))) =
  source_sum_four (fun i -> term (start + i))

inline_for_extraction noextract
let source_rms_lanes_sum (term:nat -> GTot real)
  : Lemma (
      source_sum 128 (fun lane -> source_sum 4
        (fun component -> source_sum 2 (fun part -> term (lane * 4 + component + part * 512)))) ==
      source_sum 1024 term) =
  let low = fun (lane:nat) -> source_four_terms term (lane * 4) in
  let high = fun (lane:nat) -> source_four_terms term (lane * 4 + 512) in
  let pair (start:nat)
    : Lemma (source_sum 2 (fun part -> term (start + part * 512)) ==
        term start +. term (start + 512)) =
    source_sum_step 2 (fun part -> term (start + part * 512));
    source_sum_step 1 (fun part -> term (start + part * 512))
  in
  let lane (i:nat)
    : Lemma (source_sum 4 (fun component ->
          source_sum 2 (fun part -> term (i * 4 + component + part * 512))) ==
        low i +. high i) =
    let component (c:nat)
      : Lemma (source_sum 2 (fun part -> term (i * 4 + c + part * 512)) ==
          term (i * 4 + c) +. term (i * 4 + c + 512)) =
      pair (i * 4 + c)
    in
    C.forall_intro component;
    source_sum_ext 4
      (fun component -> source_sum 2 (fun part -> term (i * 4 + component + part * 512)))
      (fun component -> term (i * 4 + component) +. term (i * 4 + component + 512));
    source_sum_add 4 (fun c -> term (i * 4 + c)) (fun c -> term (i * 4 + c + 512));
    source_sum_ext 4 (fun c -> term (i * 4 + c + 512)) (fun c -> term (i * 4 + 512 + c));
    source_four_terms_sum term (i * 4);
    source_four_terms_sum term (i * 4 + 512)
  in
  C.forall_intro lane;
  source_sum_ext 128
    (fun lane -> source_sum 4 (fun component ->
      source_sum 2 (fun part -> term (lane * 4 + component + part * 512))))
    (fun lane -> low lane +. high lane);
  source_sum_add 128 low high;
  let low_ok (i:nat)
    : Lemma (low i == source_sum 4 (fun j -> term (i * 4 + j))) =
    source_four_terms_sum term (i * 4)
  in
  let high_ok (i:nat)
    : Lemma (high i == source_sum 4 (fun j -> term (i * 4 + j + 512))) =
    source_four_terms_sum (fun j -> term (j + 512)) (i * 4)
  in
  C.forall_intro low_ok;
  C.forall_intro high_ok;
  source_sum_ext 128 low (fun i -> source_sum 4 (fun j -> term (i * 4 + j)));
  source_sum_ext 128 high (fun i -> source_sum 4 (fun j -> term (i * 4 + j + 512)));
  source_sum_flat 128 4 term;
  source_sum_flat 128 4 (fun i -> term (i + 512));
  source_sum_append 512 512 term;
  source_sum_ext 512 (fun i -> term (512 + i)) (fun i -> term (i + 512))

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
let source_rms_squares_related (input:source_hidden_values) (row:Shared.index 7)
  (x:Shared.vector real 1024)
  : Lemma
      (requires forall (i:Shared.index 1024).
        source_loaded_approximates (S.index input (row * 1024 + i)) (x i))
      (ensures N.v_approximates (source_rms_tree input row 1 0) (source_real_squares #1024 x)) =
  let leaves = S.init_ghost 128 (source_rms_lane input row) in
  let term = fun (i:nat) -> source_square_at #1024 x i in
  let logical = fun (lane:nat) -> source_sum 4 (fun component ->
    source_sum 2 (fun part -> term (lane * 4 + component + part * 512))) in
  let cell (lane:Shared.index 128)
    : Lemma (N.v_approximates (S.index leaves lane) (logical lane)) =
    source_rms_lane_related input row x lane;
    let component (c:nat)
      : Lemma (
          source_sum 2 (fun part -> source_square_at #1024 x (lane * 4 + c + part * 512)) ==
          source_sum 2 (fun part -> term (lane * 4 + c + part * 512))) =
      source_sum_ext 2 (fun part -> source_square_at #1024 x (lane * 4 + c + part * 512))
        (fun part -> term (lane * 4 + c + part * 512))
    in
    C.forall_intro component;
    source_sum_ext 4
      (fun c -> source_sum 2 (fun part -> source_square_at #1024 x (lane * 4 + c + part * 512)))
      (fun c -> source_sum 2 (fun part -> term (lane * 4 + c + part * 512)))
  in
  C.forall_intro cell;
  source_tree_numeric #128 leaves 128 1 0 logical;
  source_sum_ext 128 (fun i -> logical (0 + i * 1)) logical;
  source_sum_ext 128 logical (fun lane -> source_sum 4 (fun component ->
    source_sum 2 (fun part -> term (lane * 4 + component + part * 512))));
  source_rms_lanes_sum term;
  assert (source_sum 128 logical == source_sum 1024 term);
  source_sum_shared 1024 (fun i -> x i *. x i) term;
  source_real_squares_equation #1024 x;
  assert (source_sum 1024 term == source_real_squares #1024 x);
  source_rms_tree_from_leaves input row leaves 1 0;
  assert (N.v_approximates (source_rms_tree input row 1 0) (source_real_squares #1024 x))

inline_for_extraction noextract
let source_rms_scale_related (input:source_hidden_values) (row:Shared.index 7)
  (x:Shared.vector real 1024)
  : Lemma
      (requires forall (i:Shared.index 1024).
        source_loaded_approximates (S.index input (row * 1024 + i)) (x i))
      (ensures N.v_approximates (source_rms_scale input row) (source_real_scale #1024 x)) =
  source_rms_squares_related input row x;
  source_scale_related (source_rms_tree input row 1 0) 1024L (source_real_squares #1024 x)
#pop-options

inline_for_extraction noextract
let source_rms_result_related (weights:values BF16.t weight_elements)
  (input output:source_hidden_values) (offset:nat{offset + 1024 <= weight_elements})
  (wr:Shared.vector real 1024) (xr:Shared.hidden real)
  : Lemma
      (requires source_rms_result weights input output offset /\
        (forall (i:Shared.index 1024). N.v_approximates (S.index weights (offset + i)) (wr i)) /\
        source_loaded_hidden input xr)
      (ensures Shared.matrix_approximates (hidden_view output) (Shared.normalize_hidden wr xr)) =
  let row (r:Shared.index 7)
    : Lemma (Shared.vector_approximates (hidden_view output r)
        (Shared.normalize_hidden wr xr r)) =
    source_rms_scale_related input r (xr r);
    let cell (c:Shared.index 1024)
      : Lemma (N.v_approximates (S.index output (r * 1024 + c))
          (Shared.rms_norm #1024 wr (xr r) c)) =
      source_normalized_scalar_related (S.index input (r * 1024 + c))
        (S.index weights (offset + c)) (source_rms_scale input r)
        (xr r c) (wr c) (source_real_scale #1024 (xr r));
      source_frozen_rms_point #1024 wr (xr r) c
    in C.forall_intro cell
  in C.forall_intro row

inline_for_extraction noextract
let source_square_related (x:BF16.t) (xr:real)
  : Lemma
      (requires N.v_approximates x xr)
      (ensures N.v_approximates
        (F.mul (SourceCasts.fcast #BF16.t #F32.t x) (SourceCasts.fcast #BF16.t #F32.t x))
        (xr *. xr)) =
  SourceCasts.fcast_approx #BF16.t #F32.t x xr;
  N.a_mul (SourceCasts.fcast #BF16.t #F32.t x) (SourceCasts.fcast #BF16.t #F32.t x) xr xr

inline_for_extraction noextract
let source_qk_partial_related (qkv:values BF16.t 28672)
  (row:Shared.index 7) (head:Shared.index 24) (x:Shared.vector real 128)
  (lane:Shared.index 32)
  : Lemma
      (requires forall (i:Shared.index 128).
        N.v_approximates (S.index qkv (row * 4096 + head * 128 + i)) (x i))
      (ensures N.v_approximates (source_qk_partial qkv row head lane)
        (source_sum 4 (fun i -> source_square_at #128 x (lane * 4 + i)))) =
  let base = row * 4096 + head * 128 + lane * 4 in
  let a = SourceCasts.fcast #BF16.t #F32.t (S.index qkv base) in
  let b = SourceCasts.fcast #BF16.t #F32.t (S.index qkv (base + 1)) in
  let c = SourceCasts.fcast #BF16.t #F32.t (S.index qkv (base + 2)) in
  let d = SourceCasts.fcast #BF16.t #F32.t (S.index qkv (base + 3)) in
  let term = fun (i:nat) -> source_square_at #128 x (lane * 4 + i) in
  source_square_related (S.index qkv base) (x (lane * 4));
  source_square_related (S.index qkv (base + 1)) (x (lane * 4 + 1));
  source_square_related (S.index qkv (base + 2)) (x (lane * 4 + 2));
  source_square_related (S.index qkv (base + 3)) (x (lane * 4 + 3));
  N.a_add (F.mul a a) (F.mul b b) (term 0) (term 1);
  N.a_add (F.add (F.mul a a) (F.mul b b)) (F.mul c c) (term 0 +. term 1) (term 2);
  N.a_add (F.add (F.add (F.mul a a) (F.mul b b)) (F.mul c c)) (F.mul d d)
    ((term 0 +. term 1) +. term 2) (term 3);
  source_sum_four term

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
let source_qk_scale_related (qkv:values BF16.t 28672)
  (row:Shared.index 7) (head:Shared.index 24) (x:Shared.vector real 128)
  : Lemma
      (requires forall (i:Shared.index 128).
        N.v_approximates (S.index qkv (row * 4096 + head * 128 + i)) (x i))
      (ensures N.v_approximates (source_qk_scale qkv row head) (source_real_scale #128 x)) =
  let leaves = S.init_ghost 32 (source_qk_partial qkv row head) in
  let term = fun (i:nat) -> source_square_at #128 x i in
  let logical = fun (lane:nat) -> source_sum 4 (fun i -> term (lane * 4 + i)) in
  let cell (lane:Shared.index 32)
    : Lemma (N.v_approximates (S.index leaves lane) (logical lane)) =
    source_qk_partial_related qkv row head x lane;
    source_sum_ext 4 (fun i -> source_square_at #128 x (lane * 4 + i))
      (fun i -> term (lane * 4 + i))
  in
  C.forall_intro cell;
  source_tree_numeric #32 leaves 32 1 0 logical;
  source_sum_ext 32 (fun i -> logical (0 + i * 1)) logical;
  source_sum_ext 32 logical (fun lane -> source_sum 4 (fun i -> term (lane * 4 + i)));
  source_sum_flat 32 4 term;
  assert (source_sum 32 logical == source_sum 128 term);
  source_sum_shared 128 (fun i -> x i *. x i) term;
  source_real_squares_equation #128 x;
  assert (N.v_approximates (source_tree leaves 32 1 0) (source_real_squares #128 x));
  source_scale_related (source_tree leaves 32 1 0) 128L (source_real_squares #128 x)
#pop-options

inline_for_extraction noextract
let source_qk_normalized_related (weights:values BF16.t weight_elements)
  (qkv:values BF16.t 28672) (offset:nat{offset + 256 <= weight_elements})
  (row:Shared.index 7) (head:Shared.index 24) (wr x:Shared.vector real 128)
  : Lemma
      (requires
        (forall (i:Shared.index 128).
          N.v_approximates (S.index qkv (row * 4096 + head * 128 + i)) (x i)) /\
        (forall (i:Shared.index 128).
          N.v_approximates (S.index weights (offset + (if head < 16 then 0 else 128) + i)) (wr i)))
      (ensures forall (i:Shared.index 128).
        N.v_approximates (source_qk_normalized weights qkv offset row head i)
          (Shared.rms_norm #128 wr x i)) =
  source_qk_scale_related qkv row head x;
  let cell (i:Shared.index 128)
    : Lemma (N.v_approximates (source_qk_normalized weights qkv offset row head i)
        (Shared.rms_norm #128 wr x i)) =
    source_normalized_scalar_related (S.index qkv (row * 4096 + head * 128 + i))
      (S.index weights (offset + (if head < 16 then 0 else 128) + i))
      (source_qk_scale qkv row head) (x i) (wr i) (source_real_scale #128 x);
    source_frozen_rms_point #128 wr x i
  in C.forall_intro cell

inline_for_extraction noextract
let source_rotated_scalar (value partner cosine sine:BF16.t) (negative:bool) : BF16.t =
  let wide_partner = SourceCasts.fcast #BF16.t #F32.t partner in
  let rotated = if negative then NativeNeg.neg wide_partner else wide_partner in
  let a = SourceCasts.fcast #F32.t #BF16.t
    (F.mul (SourceCasts.fcast #BF16.t #F32.t value) (SourceCasts.fcast #BF16.t #F32.t cosine)) in
  let b = SourceCasts.fcast #F32.t #BF16.t
    (F.mul rotated (SourceCasts.fcast #BF16.t #F32.t sine)) in
  SourceCasts.fcast #F32.t #BF16.t
    (F.add (SourceCasts.fcast #BF16.t #F32.t a) (SourceCasts.fcast #BF16.t #F32.t b))

inline_for_extraction noextract
let source_rotated_scalar_related (value partner cosine sine:BF16.t) (negative:bool)
  (vr pr cr sr:real)
  : Lemma
      (requires N.v_approximates value vr /\ N.v_approximates partner pr /\
        source_loaded_approximates cosine cr /\ source_loaded_approximates sine sr)
      (ensures N.v_approximates (source_rotated_scalar value partner cosine sine negative)
        (vr *. cr +. (if negative then 0.0R -. pr else pr) *. sr)) =
  let vf = SourceCasts.fcast #BF16.t #F32.t value in
  let pf = SourceCasts.fcast #BF16.t #F32.t partner in
  let cf = SourceCasts.fcast #BF16.t #F32.t cosine in
  let sf = SourceCasts.fcast #BF16.t #F32.t sine in
  SourceCasts.fcast_approx #BF16.t #F32.t value vr;
  SourceCasts.fcast_approx #BF16.t #F32.t partner pr;
  source_loaded_widen cosine cr;
  source_loaded_widen sine sr;
  let rotated = if negative then NativeNeg.neg pf else pf in
  let rr = if negative then 0.0R -. pr else pr in
  if negative then NativeNeg.neg_approx pf pr;
  N.a_mul vf cf vr cr;
  N.a_mul rotated sf rr sr;
  SourceCasts.fcast_approx #F32.t #BF16.t (F.mul vf cf) (vr *. cr);
  SourceCasts.fcast_approx #F32.t #BF16.t (F.mul rotated sf) (rr *. sr);
  let a = SourceCasts.fcast #F32.t #BF16.t (F.mul vf cf) in
  let b = SourceCasts.fcast #F32.t #BF16.t (F.mul rotated sf) in
  SourceCasts.fcast_approx #BF16.t #F32.t a (vr *. cr);
  SourceCasts.fcast_approx #BF16.t #F32.t b (rr *. sr);
  N.a_add (SourceCasts.fcast #BF16.t #F32.t a) (SourceCasts.fcast #BF16.t #F32.t b)
    (vr *. cr) (rr *. sr);
  SourceCasts.fcast_approx #F32.t #BF16.t
    (F.add (SourceCasts.fcast #BF16.t #F32.t a) (SourceCasts.fcast #BF16.t #F32.t b))
    (vr *. cr +. rr *. sr)

inline_for_extraction noextract
let source_qk_real (qkv:Shared.heads real 24)
  (qw kw:Shared.vector real 128) (rotary:Shared.rotary_tables real)
  : GTot (Shared.heads real 24) =
  Shared.rotary #24 rotary (fun row head ->
    Shared.rms_norm #128 (if head < 16 then qw else kw) (qkv row head))

inline_for_extraction noextract
let source_frozen_rotary_point (#count:pos)
  (rotary:Shared.rotary_tables real) (data:Shared.heads real count)
  (row:Shared.index 7) (head:Shared.index count) (column:Shared.index 128)
  : Lemma (Shared.rotary #count rotary data row head column ==
      data row head column *. rotary.cosine row column +.
        (if column < 64 then 0.0R -. data row head (column + 64)
         else data row head (column - 64)) *. rotary.sine row column) =
  assert (Shared.rotary #count rotary data row head column ==
    data row head column *. rotary.cosine row column +.
      (if column < 64 then 0.0R -. data row head (column + 64)
       else data row head (column - 64)) *. rotary.sine row column) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.rotary]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let source_qk_real_at (qkv:Shared.heads real 24)
  (qw kw:Shared.vector real 128) (rotary:Shared.rotary_tables real)
  (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128)
  : Lemma (
      let normalized = Shared.rms_norm #128 (if head < 16 then qw else kw)
        (qkv row head) in
      let partner = (column + 64) % 128 in
      source_qk_real qkv qw kw rotary row head column ==
        normalized column *. rotary.cosine row column +.
          (if column < 64 then 0.0R -. normalized partner else normalized partner) *. rotary.sine row column) =
  let data:Shared.heads real 24 = fun r h ->
    Shared.rms_norm #128 (if h < 16 then qw else kw) (qkv r h) in
  assert (source_qk_real qkv qw kw rotary row head column ==
    Shared.rotary #24 rotary data row head column) by
    (FStar.Tactics.V2.norm [delta_only [`%source_qk_real]];
     FStar.Tactics.V2.smt ());
  source_frozen_rotary_point #24 rotary data row head column

inline_for_extraction noextract
let source_qk_rotated_scalar (weights:values BF16.t weight_elements)
  (qkv:values BF16.t 28672) (rotary:source_rotary_values)
  (offset:nat{offset + 256 <= weight_elements})
  (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128)
  : Lemma (source_qk_cell weights qkv rotary offset row head column ==
      source_rotated_scalar (source_qk_normalized weights qkv offset row head column)
        (source_qk_normalized weights qkv offset row head ((column + 64) % 128))
        (S.index rotary (row * 128 + column)) (S.index rotary (896 + row * 128 + column))
        (column < 64)) = ()

inline_for_extraction noextract
let source_qk_bf16_load (qk:values BF16.t 21504) : GTot (Shared.heads BF16.t 24) =
  fun row head column -> S.index qk (row * 3072 + head * 128 + column)

inline_for_extraction noextract
let source_qk_values_related (weights:values BF16.t weight_elements)
  (qkv:values BF16.t 28672) (rotary:source_rotary_values) (qk:values BF16.t 21504)
  (offset:nat{offset + 256 <= weight_elements})
  (xr:Shared.heads real 24) (qw kw:Shared.vector real 128) (dr:Shared.rotary_tables real)
  : Lemma
      (requires qk == source_qk_values weights qkv rotary offset /\
        (forall (row:Shared.index 7) (head:Shared.index 24) (i:Shared.index 128).
          N.v_approximates (S.index qkv (row * 4096 + head * 128 + i)) (xr row head i)) /\
        (forall (i:Shared.index 128).
          N.v_approximates (S.index weights (offset + i)) (qw i) /\
          N.v_approximates (S.index weights (offset + 128 + i)) (kw i)) /\
        source_loaded_rotary rotary dr)
      (ensures
        Shared.heads_approximates (source_qk_bf16_load qk) (source_qk_real xr qw kw dr) /\
        Shared.heads_approximates (source_qk_load qk) (source_qk_real xr qw kw dr)) =
  let head (row:Shared.index 7) (h:Shared.index 24)
    : Lemma (
        Shared.vector_approximates (source_qk_bf16_load qk row h) (source_qk_real xr qw kw dr row h) /\
        Shared.vector_approximates (source_qk_load qk row h) (source_qk_real xr qw kw dr row h)) =
    let wr:Shared.vector real 128 = if h < 16 then qw else kw in
    let x:Shared.vector real 128 = xr row h in
    let normalized = Shared.rms_norm #128 wr x in
    source_qk_normalized_related weights qkv offset row h wr x;
    let cell (c:Shared.index 128)
      : Lemma (
          N.v_approximates (source_qk_bf16_load qk row h c) (source_qk_real xr qw kw dr row h c) /\
          N.v_approximates (source_qk_load qk row h c) (source_qk_real xr qw kw dr row h c)) =
      let partner = (c + 64) % 128 in
      source_rotated_scalar_related
        (source_qk_normalized weights qkv offset row h c)
        (source_qk_normalized weights qkv offset row h partner)
        (S.index rotary (row * 128 + c)) (S.index rotary (896 + row * 128 + c))
        (c < 64) (normalized c) (normalized partner) (dr.cosine row c) (dr.sine row c);
      source_qk_rotated_scalar weights qkv rotary offset row h c;
      source_qk_index row h c;
      assert (S.index qk (row * 3072 + h * 128 + c) == source_qk_cell weights qkv rotary offset row h c);
      source_qk_real_at xr qw kw dr row h c;
      assert (source_qk_real xr qw kw dr row h c ==
        normalized c *. dr.cosine row c +.
          (if c < 64 then 0.0R -. normalized partner else normalized partner) *. dr.sine row c);
      SourceCasts.fcast_approx #BF16.t #F32.t (S.index qk (row * 3072 + h * 128 + c))
        (source_qk_real xr qw kw dr row h c)
    in C.forall_intro cell
  in C.forall_intro_2 head

module SourceChest = Kuiper.Chest
module SourceMMA = Kuiper.TensorCore.MMA
module SourceRegisters = Kuiper.TensorCore.MMA.Registers
module SourceGEMM = Kuiper.Spec.GEMM

noextract
let source_mma_real_term (a:chest2 real 16 16) (b:chest2 real 16 8)
  (row:natlt 16) (column:natlt 8) (k:nat) : GTot real =
  if k < 16 then SourceChest.acc2 a row k *. SourceChest.acc2 b k column else 0.0R

inline_for_extraction noextract
let rec source_mma_matmul_sum (a:chest2 real 16 16) (b:chest2 real 16 8)
  (row:natlt 16) (column:natlt 8) (count:nat{count <= 16})
  : Lemma (SourceGEMM.__matmul_single a b row column count ==
      source_sum count (source_mma_real_term a b row column)) (decreases count) =
  if count = 0 then SourceGEMM.matmul_zero_lemma a b row column
  else (
    source_mma_matmul_sum a b row column (count - 1);
    SourceGEMM.matmul_single_lemma a b row column count;
    source_sum_step count (source_mma_real_term a b row column))

noextract
let source_mma_dot_term (a b:values real 16) (k:nat) : GTot real =
  if k < 16 then S.index a k *. S.index b k else 0.0R

inline_for_extraction noextract
let source_mma_cell_related (a:chest2 bf16 16 16) (b:chest2 bf16 16 8)
  (c:chest2 float 16 8) (row:natlt 16) (column:natlt 8)
  (ar br:values real 16) (cr:real)
  : Lemma
      (requires
        (forall (k:natlt 16). N.v_approximates (SourceChest.acc2 a row k) (S.index ar k)) /\
        (forall (k:natlt 16). N.v_approximates (SourceChest.acc2 b k column) (S.index br k)) /\
        N.v_approximates (SourceChest.acc2 c row column) cr)
      (ensures N.v_approximates (SourceChest.acc2 (SourceMMA.emma c a b) row column)
        (cr +. source_sum 16 (source_mma_dot_term ar br))) =
  let ra:chest2 real 16 16 = SourceChest.mk2
    (fun i k -> if i = row then S.index ar k else N.to_real (SourceChest.acc2 a i k)) in
  let rb:chest2 real 16 8 = SourceChest.mk2
    (fun k j -> if j = column then S.index br k else N.to_real (SourceChest.acc2 b k j)) in
  let rc:chest2 real 16 8 = SourceChest.mk2
    (fun i j -> if i = row && j = column then cr else N.to_real (SourceChest.acc2 c i j)) in
  SourceChest.lemma_to_real_chest_approximates a;
  SourceChest.lemma_to_real_chest_approximates b;
  SourceChest.lemma_to_real_chest_approximates c;
  assert (a %~ ra /\ b %~ rb /\ c %~ rc);
  SourceMMA.emma_approx_lemma c a b rc ra rb;
  SourceGEMM.lemma_matmul_index ra rb row column;
  source_mma_matmul_sum ra rb row column 16;
  source_sum_ext 16 (source_mma_real_term ra rb row column) (source_mma_dot_term ar br);
  assert (SourceChest.acc2 (SourceGEMM.matplus rc (SourceGEMM.matmul ra rb)) row column ==
    cr +. source_sum 16 (source_mma_dot_term ar br))

noextract
let source_linear_term (#width:pos) (a b:Shared.vector real width) (start k:nat) : GTot real =
  if start + k < width then a (start + k) *. b (start + k) else 0.0R

inline_for_extraction noextract
let rec source_mma_k_related
  (#columns #width:pos{16 /?+ columns /\ 16 /?+ width})
  (a:chest2 bf16 columns width) (b:chest2 bf16 width 7)
  (tile:natlt (columns / 16)) (start:natle (width / 16))
  (count:nat{start + count <= width / 16}) (row:natlt 16) (column:natlt 7)
  (ar br:Shared.vector real width)
  : Lemma
      (requires
        (forall (k:natlt width). N.v_approximates (SourceChest.acc2 a (tile * 16 + row) k) (ar k)) /\
        (forall (k:natlt width). N.v_approximates (SourceChest.acc2 b k column) (br k)))
      (ensures N.v_approximates
        (SourceChest.acc2 (source_mma_kernel_k_prefix a b tile start count) row column)
        (source_sum (count * 16) (source_linear_term #width ar br (start * 16))))
      (decreases count) =
  if count > 0 then (
    let k = start + count - 1 in
    let ra = S.init_ghost 16 (fun i -> ar (k * 16 + i)) in
    let rb = S.init_ghost 16 (fun i -> br (k * 16 + i)) in
    source_mma_k_related a b tile start (count - 1) row column ar br;
    let cell (i:natlt 16)
      : Lemma (
          N.v_approximates (SourceChest.acc2 (ematrix_subtile a 16 16 tile k) row i) (S.index ra i) /\
          N.v_approximates (SourceChest.acc2
            (SourceRegisters.pad_b7 (ematrix_subtile b 16 7 k 0)) i column) (S.index rb i)) =
      subtile_acc2 a 16 16 tile k row i;
      subtile_acc2 b 16 7 k 0 i column
    in
    C.forall_intro cell;
    source_mma_cell_related (ematrix_subtile a 16 16 tile k)
      (SourceRegisters.pad_b7 (ematrix_subtile b 16 7 k 0))
      (source_mma_kernel_k_prefix a b tile start (count - 1)) row column ra rb
      (source_sum ((count - 1) * 16) (source_linear_term #width ar br (start * 16)));
    source_sum_ext 16 (source_mma_dot_term ra rb)
      (fun i -> source_linear_term #width ar br (start * 16) ((count - 1) * 16 + i));
    source_sum_append ((count - 1) * 16) 16 (source_linear_term #width ar br (start * 16));
    source_mma_kernel_k_step a b tile start (count - 1))

inline_for_extraction noextract
let rec source_mma_splits_related
  (#columns #width:pos{16 /?+ columns /\ 16 /?+ width})
  (a:chest2 bf16 columns width) (b:chest2 bf16 width 7)
  (tile:natlt (columns / 16)) (steps:pos) (count:nat{count * steps <= width / 16})
  (row:natlt 16) (column:natlt 7) (ar br:Shared.vector real width)
  : Lemma
      (requires
        (forall (k:natlt width). N.v_approximates (SourceChest.acc2 a (tile * 16 + row) k) (ar k)) /\
        (forall (k:natlt width). N.v_approximates (SourceChest.acc2 b k column) (br k)))
      (ensures N.v_approximates
        (SourceChest.acc2 (source_mma_kernel_split_prefix a b tile steps count) row column)
        (source_sum (count * steps * 16) (source_linear_term #width ar br 0)))
      (decreases count) =
  if count > 0 then (
    let previous = count - 1 in
    let start = previous * steps in
    let part = source_mma_kernel_k_prefix a b tile start steps in
    let terms = source_linear_term #width ar br 0 in
    source_mma_k_related a b tile start steps row column ar br;
    source_mma_kernel_split_step a b tile steps previous;
    source_sum_ext (steps * 16) (source_linear_term #width ar br (start * 16))
      (fun i -> terms (previous * steps * 16 + i));
    if count > 1 then (
      source_mma_splits_related a b tile steps previous row column ar br;
      N.a_add (SourceChest.acc2 (source_mma_kernel_split_prefix a b tile steps previous) row column)
        (SourceChest.acc2 part row column)
        (source_sum (previous * steps * 16) terms)
        (source_sum (steps * 16) (fun i -> terms (previous * steps * 16 + i)));
      source_sum_append (previous * steps * 16) (steps * 16) terms))

inline_for_extraction noextract
let source_linear_operand_views
  (#columns #width:pos{16 /?+ columns /\ 16 /?+ width}) (#n:nat)
  (weights:values BF16.t n) (input:values BF16.t (width * 7))
  (offset:nat{offset + columns * width <= n})
  (row:natlt 7) (channel:natlt columns) (k:natlt width)
  : Lemma (
      SourceChest.acc2
        (from_seq (l2_row_major columns width) (S.slice weights offset (offset + columns * width)))
        channel k == S.index weights (offset + channel * width + k) /\
      SourceChest.acc2 (from_seq (l2_col_major width 7) input) k row ==
        S.index input (row * width + k)) = ()

inline_for_extraction noextract
let source_frozen_linear_point (#columns #width:pos)
  (wr:Shared.matrix real columns width) (xr:Shared.matrix real 7 width)
  (row:natlt 7) (channel:natlt columns)
  : Lemma (Shared.linear_rows #7 #width #columns wr xr row channel ==
      Shared.sum width (fun k -> xr row k *. wr channel k)) =
  assert (Shared.linear_rows #7 #width #columns wr xr row channel ==
    Shared.sum width (fun k -> xr row k *. wr channel k)) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.linear_rows; `%Shared.linear; `%Shared.dot]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let source_linear_array_result_related
  (#columns #width:pos{16 /?+ columns /\ 16 /?+ width}) (#n:nat)
  (weights:values BF16.t n) (input:values BF16.t (width * 7)) (old:values BF16.t (columns * 7))
  (offset:nat{offset + columns * width <= n})
  (steps:pos) (splits:pos{steps * splits == width / 16})
  (wr:Shared.matrix real columns width) (xr:Shared.matrix real 7 width)
  : Lemma
      (requires
        (forall (channel:natlt columns) (k:natlt width).
          N.v_approximates (S.index weights (offset + channel * width + k)) (wr channel k)) /\
        (forall (row:natlt 7) (k:natlt width).
          N.v_approximates (S.index input (row * width + k)) (xr row k)))
      (ensures forall (row:natlt 7) (channel:natlt columns).
        N.v_approximates
          (S.index (source_mma_arrays_array_result #columns #width #n weights input old offset steps splits)
            (row * columns + channel))
          (Shared.linear_rows #7 #width #columns wr xr row channel)) =
  let a = from_seq (l2_row_major columns width) (S.slice weights offset (offset + columns * width)) in
  let b = from_seq (l2_col_major width 7) input in
  let cell (row:natlt 7) (channel:natlt columns)
    : Lemma (N.v_approximates
        (S.index (source_mma_arrays_array_result #columns #width #n weights input old offset steps splits)
          (row * columns + channel))
        (Shared.linear_rows #7 #width #columns wr xr row channel)) =
    let operand (k:natlt width)
      : Lemma (
          N.v_approximates (SourceChest.acc2 a channel k) (wr channel k) /\
          N.v_approximates (SourceChest.acc2 b k row) (xr row k)) =
      source_linear_operand_views #columns #width #n weights input offset row channel k
    in
    C.forall_intro operand;
    let tile = channel / 16 in
    let at = channel % 16 in
    source_mma_splits_related a b tile steps splits at row (wr channel) (xr row);
    let terms = source_linear_term #width (wr channel) (xr row) 0 in
    source_sum_shared width (fun k -> xr row k *. wr channel k) terms;
    source_frozen_linear_point #columns #width wr xr row channel;
    let lane = (at % 8) * 4 + row / 2 in
    let reg = (at / 8) * 2 + row % 2 in
    source_mma_layout_c_inverse at row;
    source_mma_sourcebinding_output_value #columns #width #n weights input old offset steps splits tile lane reg;
    Kuiper.Float.Casts.Base.cast_f32_to_bf16_ok
      (SourceChest.acc2 (source_mma_kernel_split_prefix a b tile steps splits) at row)
      (Shared.linear_rows #7 #width #columns wr xr row channel)
  in C.forall_intro_2 cell

inline_for_extraction noextract
let source_added_values_related (left right:source_hidden_values) (lr rr:Shared.hidden real)
  : Lemma
      (requires source_loaded_hidden left lr /\ source_loaded_hidden right rr)
      (ensures Shared.matrix_approximates (hidden_view (source_added_values left right))
        (Shared.residual_add lr rr)) =
  let cell (row:Shared.index 7) (column:Shared.index 1024)
    : Lemma (N.v_approximates (S.index (source_added_values left right) (row * 1024 + column))
        (lr row column +. rr row column)) =
    let l = S.index left (row * 1024 + column) in
    let r = S.index right (row * 1024 + column) in
    source_loaded_widen l (lr row column);
    source_loaded_widen r (rr row column);
    N.a_add (SourceCasts.fcast #BF16.t #F32.t l) (SourceCasts.fcast #BF16.t #F32.t r)
      (lr row column) (rr row column);
    SourceCasts.fcast_approx #F32.t #BF16.t
      (F.add (SourceCasts.fcast #BF16.t #F32.t l) (SourceCasts.fcast #BF16.t #F32.t r))
      (lr row column +. rr row column)
  in C.forall_intro_2 cell

inline_for_extraction noextract
let source_silu_cell_related (input:values BF16.t 43008)
  (row:Shared.index 7) (column:Shared.index 3072) (gr ur:real)
  : Lemma
      (requires N.v_approximates (S.index input (row * 6144 + column)) gr /\
        N.v_approximates (S.index input (row * 6144 + 3072 + column)) ur)
      (ensures N.v_approximates (source_silu_cell input row column) (Shared.silu gr *. ur)) =
  let gate = SourceCasts.fcast #BF16.t #F32.t (S.index input (row * 6144 + column)) in
  let up = SourceCasts.fcast #BF16.t #F32.t (S.index input (row * 6144 + 3072 + column)) in
  SourceCasts.fcast_approx #BF16.t #F32.t (S.index input (row * 6144 + column)) gr;
  SourceCasts.fcast_approx #BF16.t #F32.t (S.index input (row * 6144 + 3072 + column)) ur;
  NativeNeg.neg_approx gate gr;
  N.exp_approx (NativeNeg.neg gate) (0.0R -. gr);
  let exponential = F.fexp (NativeNeg.neg gate) in
  N.a_add (F.one #F32.t) exponential 1.0R (Exp.exp (0.0R -. gr));
  N.div_approx gate (F.add (F.one #F32.t) exponential) gr (1.0R +. Exp.exp (0.0R -. gr));
  let activated = F.div gate (F.add (F.one #F32.t) exponential) in
  SourceCasts.fcast_approx #F32.t #BF16.t activated (Shared.silu gr);
  SourceCasts.fcast_approx #BF16.t #F32.t (SourceCasts.fcast #F32.t #BF16.t activated) (Shared.silu gr);
  let wide = SourceCasts.fcast #BF16.t #F32.t (SourceCasts.fcast #F32.t #BF16.t activated) in
  N.a_mul wide up (Shared.silu gr) ur;
  SourceCasts.fcast_approx #F32.t #BF16.t (F.mul wide up) (Shared.silu gr *. ur)

inline_for_extraction noextract
let source_join_qk (query:Shared.heads real 16) (key:Shared.heads real 8)
  : GTot (Shared.heads real 24) =
  fun row head -> if head < 16 then query row head else key row (head - 16)

inline_for_extraction noextract
let source_normalize_heads_at (#count:pos) (weight:Shared.vector real 128)
  (data:Shared.heads real count) (row:Shared.index 7) (head:Shared.index count)
  : Lemma (Shared.normalize_heads #count weight data row head ==
      Shared.rms_norm #128 weight (data row head)) =
  assert (Shared.normalize_heads #count weight data row head ==
    Shared.rms_norm #128 weight (data row head)) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.normalize_heads]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let source_qk_reference_view (qk:values BF16.t 21504)
  (query:Shared.heads real 16) (key:Shared.heads real 8)
  (qw kw:Shared.vector real 128) (rotary:Shared.rotary_tables real)
  : Lemma
      (requires
        Shared.heads_approximates (source_qk_bf16_load qk)
          (source_qk_real (source_join_qk query key) qw kw rotary) /\
        Shared.heads_approximates (source_qk_load qk)
          (source_qk_real (source_join_qk query key) qw kw rotary))
      (ensures
        Shared.heads_approximates (source_qk_bf16_load qk)
          (source_join_qk
            (Shared.rotary #16 rotary (Shared.normalize_heads #16 qw query))
            (Shared.rotary #8 rotary (Shared.normalize_heads #8 kw key))) /\
        Shared.heads_approximates (source_qk_load qk)
          (source_join_qk
            (Shared.rotary #16 rotary (Shared.normalize_heads #16 qw query))
            (Shared.rotary #8 rotary (Shared.normalize_heads #8 kw key)))) =
  let cell (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128)
    : Lemma (
        N.v_approximates (source_qk_bf16_load qk row head column)
          (source_join_qk
            (Shared.rotary #16 rotary (Shared.normalize_heads #16 qw query))
            (Shared.rotary #8 rotary (Shared.normalize_heads #8 kw key)) row head column) /\
        N.v_approximates (source_qk_load qk row head column)
        (source_join_qk
          (Shared.rotary #16 rotary (Shared.normalize_heads #16 qw query))
          (Shared.rotary #8 rotary (Shared.normalize_heads #8 kw key)) row head column)) =
    source_qk_real_at (source_join_qk query key) qw kw rotary row head column;
    if head < 16 then (
      source_normalize_heads_at #16 qw query row head;
      source_frozen_rotary_point #16 rotary (Shared.normalize_heads #16 qw query) row head column)
    else (
      source_normalize_heads_at #8 kw key row (head - 16);
      source_frozen_rotary_point #8 rotary (Shared.normalize_heads #8 kw key) row (head - 16) column)
  in C.forall_intro_3 cell

inline_for_extraction noextract
let source_prefix_softmax_ext (row:Shared.index 7)
  (left right:Shared.vector real (row + 1)) (key:Shared.index (row + 1))
  : Lemma
      (requires forall (i:Shared.index (row + 1)). left i == right i)
      (ensures Shared.prefix_softmax row left key == Shared.prefix_softmax row right key) =
  let l:Shared.vector real (row + 1) = fun i -> Exp.exp (left i) in
  let r:Shared.vector real (row + 1) = fun i -> Exp.exp (right i) in
  Shared.sum_positive (row + 1) l;
  Shared.sum_positive (row + 1) r;
  source_shared_sum_equal (row + 1) l r;
  source_prefix_softmax_at row left key;
  source_prefix_softmax_at row right key

inline_for_extraction noextract
let source_dot_ext (#width:nat)
  (x y u v:Shared.vector real width)
  : Lemma
      (requires forall (i:Shared.index width). x i == u i /\ y i == v i)
      (ensures Shared.dot #width x y == Shared.dot #width u v) =
  source_shared_sum_equal width (fun i -> x i *. y i) (fun i -> u i *. v i);
  assert (Shared.dot #width x y == Shared.sum width (fun i -> x i *. y i)) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.dot]];
     FStar.Tactics.V2.trivial ());
  assert (Shared.dot #width u v == Shared.sum width (fun i -> u i *. v i)) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.dot]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let source_frozen_probabilities_at (query key:Shared.heads real 16)
  (row:Shared.index 7) (head:Shared.index 16) (position:Shared.index (row + 1))
  : Lemma (Shared.attention_probabilities query key row head position ==
      Shared.prefix_softmax row
        (fun k -> Shared.dot #128 (query row head) (key k head) *. Root.rsqrt 128.0R) position) =
  assert (Shared.attention_probabilities query key row head position ==
    Shared.prefix_softmax row
      (fun k -> Shared.dot #128 (query row head) (key k head) *. Root.rsqrt 128.0R) position) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.attention_probabilities]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let source_attention_probability_at (query:Shared.heads real 16) (key:Shared.heads real 8)
  (row:Shared.index 7) (head:Shared.index 16) (position:Shared.index (row + 1))
  : Lemma (Shared.attention_probabilities query (Shared.repeat_kv key) row head position ==
      Shared.prefix_softmax row (source_attention_score_model (source_join_qk query key) row head) position) =
  let repeated = Shared.repeat_kv key in
  let scores:Shared.vector real (row + 1) =
    fun k -> Shared.dot #128 (query row head) (repeated k head) *. Root.rsqrt 128.0R in
  let score (k:Shared.index (row + 1))
    : Lemma (scores k == source_attention_score_model (source_join_qk query key) row head k) =
    source_dot_ext #128 (query row head) (repeated k head)
      (source_join_qk query key row head) (source_join_qk query key k (16 + head / 2))
  in
  C.forall_intro score;
  source_prefix_softmax_ext row scores (source_attention_score_model (source_join_qk query key) row head) position;
  source_frozen_probabilities_at query repeated row head position

inline_for_extraction noextract
let source_attention_cell_apply (qkv:values BF16.t 28672) (qk:values BF16.t 21504)
  (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128) (actual:BF16.t)
  (q:Shared.heads real 24) (v:Shared.heads real 8)
  : Lemma
      (requires source_attention_cell_relation qkv qk row head column actual /\
        Shared.heads_approximates (source_qk_load qk) q /\
        Shared.heads_approximates (source_v_load qkv) v)
      (ensures N.v_approximates actual
        (source_sum (row + 1) (source_mixed_term row head column v
          (fun key -> Shared.prefix_softmax row (source_attention_score_model q row head) key)))) = ()

inline_for_extraction noextract
let source_attention_to_reference
  (weights:values BF16.t weight_elements) (qkv:values BF16.t 28672) (rotary:source_rotary_values)
  (qk:values BF16.t 21504) (output:values BF16.t 14336)
  (offset:nat{offset + 256 <= weight_elements})
  (query:Shared.heads real 16) (key value:Shared.heads real 8)
  : Lemma
      (requires source_attention_result weights qkv rotary qk output offset /\
        Shared.heads_approximates (source_qk_load qk) (source_join_qk query key) /\
        Shared.heads_approximates (source_v_load qkv) value)
      (ensures forall (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128).
        N.v_approximates (S.index output (row * 2048 + head * 128 + column))
          (Shared.attention_values (Shared.attention_probabilities query (Shared.repeat_kv key))
            (Shared.repeat_kv value) row head column)) =
  let cell (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128)
    : Lemma (N.v_approximates (S.index output (row * 2048 + head * 128 + column))
        (Shared.attention_values (Shared.attention_probabilities query (Shared.repeat_kv key))
          (Shared.repeat_kv value) row head column)) =
    let probability = fun (k:Shared.index (row + 1)) ->
      Shared.prefix_softmax row (source_attention_score_model (source_join_qk query key) row head) k in
    let term = source_mixed_term row head column value probability in
    source_attention_cell_apply qkv qk row head column
      (S.index output (row * 2048 + head * 128 + column)) (source_join_qk query key) value;
    assert (N.v_approximates (S.index output (row * 2048 + head * 128 + column)) (source_sum (row + 1) term));
    let at (k:Shared.index (row + 1))
      : Lemma (
          Shared.attention_probabilities query (Shared.repeat_kv key) row head k *.
            Shared.repeat_kv value k head column == term k) =
      source_attention_probability_at query key row head k
    in
    C.forall_intro at;
    source_sum_shared (row + 1)
      (fun k -> Shared.attention_probabilities query (Shared.repeat_kv key) row head k *.
        Shared.repeat_kv value k head column) term;
    assert (Shared.attention_values (Shared.attention_probabilities query (Shared.repeat_kv key))
        (Shared.repeat_kv value) row head column ==
      Shared.sum (row + 1)
        (fun k -> Shared.attention_probabilities query (Shared.repeat_kv key) row head k *.
          Shared.repeat_kv value k head column)) by
      (FStar.Tactics.V2.norm [delta_only [`%Shared.attention_values]];
       FStar.Tactics.V2.trivial ())
  in C.forall_intro_3 cell

inline_for_extraction noextract
let source_qkv_weight (w:Shared.parameters real) : GTot (Shared.matrix real 4096 1024) =
  fun channel ->
    if channel < 2048 then w.q_proj channel
    else if channel < 3072 then w.k_proj (channel - 2048)
    else w.v_proj (channel - 3072)

inline_for_extraction noextract
let source_gateup_weight (w:Shared.parameters real) : GTot (Shared.matrix real 6144 1024) =
  fun channel -> if channel < 3072 then w.gate_proj channel else w.up_proj (channel - 3072)

inline_for_extraction noextract
let source_qkv_projection_at (w:Shared.parameters real) (x:Shared.hidden real)
  (row:Shared.index 7) (channel:Shared.index 4096)
  : Lemma (
      Shared.linear_rows #7 #1024 #4096 (source_qkv_weight w) x row channel ==
      (if channel < 2048 then Shared.linear_rows #7 #1024 #2048 w.q_proj x row channel
       else if channel < 3072 then Shared.linear_rows #7 #1024 #1024 w.k_proj x row (channel - 2048)
       else Shared.linear_rows #7 #1024 #1024 w.v_proj x row (channel - 3072))) =
  source_frozen_linear_point #4096 #1024 (source_qkv_weight w) x row channel;
  if channel < 2048 then (
    source_frozen_linear_point #2048 #1024 w.q_proj x row channel;
    source_shared_sum_equal 1024
      (fun k -> x row k *. source_qkv_weight w channel k) (fun k -> x row k *. w.q_proj channel k))
  else if channel < 3072 then (
    source_frozen_linear_point #1024 #1024 w.k_proj x row (channel - 2048);
    source_shared_sum_equal 1024
      (fun k -> x row k *. source_qkv_weight w channel k) (fun k -> x row k *. w.k_proj (channel - 2048) k))
  else (
    source_frozen_linear_point #1024 #1024 w.v_proj x row (channel - 3072);
    source_shared_sum_equal 1024
      (fun k -> x row k *. source_qkv_weight w channel k) (fun k -> x row k *. w.v_proj (channel - 3072) k))

inline_for_extraction noextract
let source_gateup_projection_at (w:Shared.parameters real) (x:Shared.hidden real)
  (row:Shared.index 7) (channel:Shared.index 6144)
  : Lemma (
      Shared.linear_rows #7 #1024 #6144 (source_gateup_weight w) x row channel ==
      (if channel < 3072 then Shared.linear_rows #7 #1024 #3072 w.gate_proj x row channel
       else Shared.linear_rows #7 #1024 #3072 w.up_proj x row (channel - 3072))) =
  source_frozen_linear_point #6144 #1024 (source_gateup_weight w) x row channel;
  if channel < 3072 then (
    source_frozen_linear_point #3072 #1024 w.gate_proj x row channel;
    source_shared_sum_equal 1024
      (fun k -> x row k *. source_gateup_weight w channel k) (fun k -> x row k *. w.gate_proj channel k))
  else (
    source_frozen_linear_point #3072 #1024 w.up_proj x row (channel - 3072);
    source_shared_sum_equal 1024
      (fun k -> x row k *. source_gateup_weight w channel k) (fun k -> x row k *. w.up_proj (channel - 3072) k))

inline_for_extraction noextract
let source_linear_values_related
  (columns width:pos{16 /?+ columns /\ 16 /?+ width})
  (weights:values BF16.t weight_elements) (input:values BF16.t (width * 7))
  (offset:nat{offset + columns * width <= weight_elements})
  (steps:pos) (splits:pos{steps * splits == width / 16})
  (wr:Shared.matrix real columns width) (xr:Shared.matrix real 7 width)
  : Lemma
      (requires
        (forall (channel:Shared.index columns) (k:Shared.index width).
          N.v_approximates (S.index weights (offset + channel * width + k)) (wr channel k)) /\
        (forall (row:Shared.index 7) (k:Shared.index width).
          N.v_approximates (S.index input (row * width + k)) (xr row k)))
      (ensures forall (row:Shared.index 7) (channel:Shared.index columns).
        N.v_approximates
          (S.index (source_linear_values columns width weights input offset steps splits) (row * columns + channel))
          (Shared.linear_rows #7 #width #columns wr xr row channel)) =
  source_linear_array_result_related #columns #width #weight_elements weights input
    (S.create (columns * 7) (F.zero #BF16.t)) offset steps splits wr xr

inline_for_extraction noextract
let source_first_block_weights_related (input:source_inputs) (wr:Shared.parameters real)
  : Lemma
      (requires Shared.parameters_approximates (source_parameters input 0) wr)
      (ensures
        (forall (d:Shared.index 1024).
          N.v_approximates (S.index input.weights (155583488 + d)) (wr.input_layernorm d) /\
          N.v_approximates (S.index input.weights (155584512 + d)) (wr.post_attention_layernorm d)) /\
        (forall (r:Shared.index 4096) (d:Shared.index 1024).
          N.v_approximates (S.index input.weights (155585536 + r * 1024 + d)) (source_qkv_weight wr r d)) /\
        (forall (d:Shared.index 128).
          N.v_approximates (S.index input.weights (161876992 + d)) (wr.q_norm d) /\
          N.v_approximates (S.index input.weights (161877120 + d)) (wr.k_norm d)) /\
        (forall (r:Shared.index 1024) (d:Shared.index 2048).
          N.v_approximates (S.index input.weights (159779840 + r * 2048 + d)) (wr.o_proj r d)) /\
        (forall (r:Shared.index 6144) (d:Shared.index 1024).
          N.v_approximates (S.index input.weights (161877248 + r * 1024 + d)) (source_gateup_weight wr r d)) /\
        (forall (r:Shared.index 1024) (d:Shared.index 3072).
          N.v_approximates (S.index input.weights (168168704 + r * 3072 + d)) (wr.down_proj r d))) =
  assert_norm (layer_start 0 == 155583488)

inline_for_extraction noextract
let source_next_norm_weight_related (input:source_inputs) (wr:Shared.vector real 1024)
  : Lemma
      (requires Shared.vector_approximates (source_parameters input 1).input_layernorm wr)
      (ensures forall (d:Shared.index 1024).
        N.v_approximates (S.index input.weights (171314432 + d)) (wr d)) =
  assert_norm (layer_start 1 == 171314432)

inline_for_extraction noextract
let source_first_block_arithmetic (input:source_inputs) (v:source_block_values) : GTot prop =
  source_rms_result input.weights v.hidden0 v.norm0 155583488 /\
  v.qkv0 == source_linear_values 4096 1024 input.weights v.norm0 155585536 64 1 /\
  source_attention_result input.weights v.qkv0 v.rotary0
    (source_qk_values input.weights v.qkv0 v.rotary0 161876992) v.attn0 161876992 /\
  v.projected0 == source_linear_values 1024 2048 input.weights v.attn0 159779840 128 1 /\
  source_add_norm_result input.weights v.hidden0 v.projected0 v.residual0 v.postnorm0 155584512 /\
  v.gateup0 == source_linear_values 6144 1024 input.weights v.postnorm0 161877248 64 1 /\
  source_silu_result v.gateup0 v.activated0 /\
  v.down0 == source_linear_values 1024 3072 input.weights v.activated0 168168704 96 2 /\
  source_add_norm_result input.weights v.residual0 v.down0 v.hidden1 v.norm1 171314432

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
let source_first_block_mlp_related (input:source_inputs) (v:source_block_values)
  (wr:Shared.parameters real) (residual:Shared.hidden real)
  : Lemma
      (requires source_first_block_arithmetic input v /\
        Shared.parameters_approximates (source_parameters input 0) wr /\
        Shared.matrix_approximates (hidden_view v.residual0) residual)
      (ensures Shared.matrix_approximates (hidden_view v.hidden1)
        (Shared.residual_add residual
          (Shared.feed_forward wr (Shared.normalize_hidden wr.post_attention_layernorm residual)))) =
  source_first_block_weights_related input wr;
  source_rms_result_related input.weights v.residual0 v.postnorm0 155584512 wr.post_attention_layernorm residual;
  let mlp_input = Shared.normalize_hidden wr.post_attention_layernorm residual in
  source_linear_values_related 6144 1024 input.weights v.postnorm0 161877248 64 1
    (source_gateup_weight wr) mlp_input;
  let gate = Shared.linear_rows #7 #1024 #3072 wr.gate_proj mlp_input in
  let up = Shared.linear_rows #7 #1024 #3072 wr.up_proj mlp_input in
  let activated:Shared.matrix real 7 3072 = fun row column ->
    Shared.silu (Shared.linear_rows #7 #1024 #3072 wr.gate_proj mlp_input row column) *.
      Shared.linear_rows #7 #1024 #3072 wr.up_proj mlp_input row column in
  let activated_cell (row:Shared.index 7) (column:Shared.index 3072)
    : Lemma (N.v_approximates (S.index v.activated0 (row * 3072 + column)) (activated row column)) =
    source_gateup_projection_at wr mlp_input row column;
    source_gateup_projection_at wr mlp_input row (3072 + column);
    source_silu_cell_related v.gateup0 row column (gate row column) (up row column)
  in
  C.forall_intro_2 activated_cell;
  source_linear_values_related 1024 3072 input.weights v.activated0 168168704 96 2 wr.down_proj activated;
  let down = Shared.linear_rows #7 #3072 #1024 wr.down_proj activated in
  assert (Shared.matrix_approximates (hidden_view v.down0) down);
  source_added_values_related v.residual0 v.down0 residual down;
  assert (Shared.matrix_approximates (hidden_view v.hidden1) (Shared.residual_add residual down));
  assert (Shared.feed_forward wr mlp_input == down) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.feed_forward]];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let source_first_block_qkv_related (input:source_inputs) (v:source_block_values)
  (wr:Shared.parameters real) (normalized:Shared.hidden real)
  : Lemma
      (requires source_first_block_arithmetic input v /\
        Shared.parameters_approximates (source_parameters input 0) wr /\
        Shared.matrix_approximates (hidden_view v.norm0) normalized)
      (ensures (
        let query = Shared.split_heads #real #16
          (Shared.linear_rows #7 #1024 #2048 wr.q_proj normalized) in
        let key = Shared.split_heads #real #8
          (Shared.linear_rows #7 #1024 #1024 wr.k_proj normalized) in
        let value = Shared.split_heads #real #8
          (Shared.linear_rows #7 #1024 #1024 wr.v_proj normalized) in
        (forall (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128).
          N.v_approximates (S.index v.qkv0 (row * 4096 + head * 128 + column))
            (source_join_qk query key row head column)) /\
        (forall (row:Shared.index 7) (head:Shared.index 8) (column:Shared.index 128).
          N.v_approximates (S.index v.qkv0 (row * 4096 + 3072 + head * 128 + column))
            (value row head column)) /\
        Shared.heads_approximates (source_v_load v.qkv0) value)) =
  source_first_block_weights_related input wr;
  source_linear_values_related 4096 1024 input.weights v.norm0 155585536 64 1
    (source_qkv_weight wr) normalized;
  let query = Shared.split_heads #real #16
    (Shared.linear_rows #7 #1024 #2048 wr.q_proj normalized) in
  let key = Shared.split_heads #real #8
    (Shared.linear_rows #7 #1024 #1024 wr.k_proj normalized) in
  let value = Shared.split_heads #real #8
    (Shared.linear_rows #7 #1024 #1024 wr.v_proj normalized) in
  let qkv_cell (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128)
    : Lemma (N.v_approximates (S.index v.qkv0 (row * 4096 + head * 128 + column))
        (source_join_qk query key row head column)) =
    source_qkv_projection_at wr normalized row (head * 128 + column)
  in
  C.forall_intro_3 qkv_cell;
  let value_cell (row:Shared.index 7) (head:Shared.index 8) (column:Shared.index 128)
    : Lemma (
        N.v_approximates (S.index v.qkv0 (row * 4096 + 3072 + head * 128 + column)) (value row head column) /\
        N.v_approximates (source_v_load v.qkv0 row head column) (value row head column)) =
    source_qkv_projection_at wr normalized row (3072 + head * 128 + column);
    SourceCasts.fcast_approx #BF16.t #F32.t
      (S.index v.qkv0 (row * 4096 + 3072 + head * 128 + column)) (value row head column)
  in C.forall_intro_3 value_cell

inline_for_extraction noextract
let source_first_block_keys_related (v:source_block_values) (qk:values BF16.t 21504)
  (query:Shared.heads real 16) (key:Shared.heads real 8)
  : Lemma
      (requires Shared.heads_approximates (source_qk_bf16_load qk) (source_join_qk query key))
      (ensures Shared.heads_approximates (source_first_block_native_layer v qk).keys key) =
  let cell (row:Shared.index 7) (head:Shared.index 8) (column:Shared.index 128)
    : Lemma (N.v_approximates (S.index qk (row * 3072 + 2048 + head * 128 + column))
        (key row head column)) =
    assert (N.v_approximates (source_qk_bf16_load qk row (16 + head) column)
      (source_join_qk query key row (16 + head) column))
  in C.forall_intro_3 cell

inline_for_extraction noextract
let source_attention_concatenated (output:values BF16.t 14336) (mixed:Shared.heads real 16)
  : Lemma
      (requires forall (row:Shared.index 7) (head:Shared.index 16) (column:Shared.index 128).
        N.v_approximates (S.index output (row * 2048 + head * 128 + column))
          (mixed row head column))
      (ensures forall (row:Shared.index 7) (column:Shared.index 2048).
        N.v_approximates (S.index output (row * 2048 + column))
          (Shared.concatenate_heads mixed row column)) =
  let cell (row:Shared.index 7) (column:Shared.index 2048)
    : Lemma (N.v_approximates (S.index output (row * 2048 + column))
        (Shared.concatenate_heads mixed row column)) =
    assert (N.v_approximates
      (S.index output (row * 2048 + (column / 128) * 128 + column % 128))
      (mixed row (column / 128) (column % 128)))
  in C.forall_intro_2 cell

inline_for_extraction noextract
let source_first_block_layer_related (input:source_inputs) (v:source_block_values)
  (qk:values BF16.t 21504)
  (wr:Shared.parameters real) (xr:Shared.hidden real) (dr:Shared.rotary_tables real)
  : Lemma
      (requires source_first_block_arithmetic input v /\
        qk == source_qk_values input.weights v.qkv0 v.rotary0 161876992 /\
        Shared.parameters_approximates (source_parameters input 0) wr /\
        Shared.matrix_approximates (hidden_view v.hidden0) xr /\
        Shared.rotary_approximates (rotary_view v.rotary0) dr)
      (ensures Shared.layer_approximates (source_first_block_native_layer v qk)
        (Shared.decoder_layer wr dr xr)) =
  assert (source_attention_result input.weights v.qkv0 v.rotary0 qk v.attn0 161876992);
  source_first_block_weights_related input wr;
  let normalized = Shared.normalize_hidden wr.input_layernorm xr in
  source_rms_result_related input.weights v.hidden0 v.norm0 155583488 wr.input_layernorm xr;
  source_first_block_qkv_related input v wr normalized;
  let query_input = Shared.split_heads #real #16
    (Shared.linear_rows #7 #1024 #2048 wr.q_proj normalized) in
  let key_input = Shared.split_heads #real #8
    (Shared.linear_rows #7 #1024 #1024 wr.k_proj normalized) in
  let value = Shared.split_heads #real #8
    (Shared.linear_rows #7 #1024 #1024 wr.v_proj normalized) in
  let qk_input = source_join_qk query_input key_input in
  source_qk_values_related input.weights v.qkv0 v.rotary0 qk 161876992 qk_input wr.q_norm wr.k_norm dr;
  source_qk_reference_view qk query_input key_input wr.q_norm wr.k_norm dr;
  let query = Shared.rotary #16 dr (Shared.normalize_heads #16 wr.q_norm query_input) in
  let key = Shared.rotary #8 dr (Shared.normalize_heads #8 wr.k_norm key_input) in
  source_first_block_keys_related v qk query key;
  assert (Shared.heads_approximates (source_first_block_native_layer v qk).values value);
  source_attention_to_reference input.weights v.qkv0 v.rotary0 qk v.attn0 161876992 query key value;
  let probabilities = Shared.attention_probabilities query (Shared.repeat_kv key) in
  let mixed = Shared.attention_values probabilities (Shared.repeat_kv value) in
  source_attention_concatenated v.attn0 mixed;
  let attention = Shared.concatenate_heads mixed in
  source_linear_values_related 1024 2048 input.weights v.attn0 159779840 128 1 wr.o_proj attention;
  let projected = Shared.linear_rows #7 #2048 #1024 wr.o_proj attention in
  assert (Shared.matrix_approximates (hidden_view v.projected0) projected);
  source_added_values_related v.hidden0 v.projected0 xr projected;
  let residual = Shared.residual_add xr projected in
  assert (Shared.matrix_approximates (hidden_view v.residual0) residual);
  source_first_block_mlp_related input v wr residual;
  let mlp_input = Shared.normalize_hidden wr.post_attention_layernorm residual in
  let expected:Shared.layer_values real = {
    hidden_after = Shared.residual_add residual (Shared.feed_forward wr mlp_input);
    keys = key;
    values = value
  } in
  assert (Shared.decoder_layer wr dr xr == expected) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.decoder_layer]];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let source_first_block_next_norm_related (input:source_inputs) (v:source_block_values)
  (wr:Shared.vector real 1024) (xr:Shared.hidden real)
  : Lemma
      (requires source_first_block_arithmetic input v /\
        Shared.vector_approximates (source_parameters input 1).input_layernorm wr /\
        Shared.matrix_approximates (hidden_view v.hidden1) xr)
      (ensures Shared.matrix_approximates (hidden_view v.norm1) (Shared.normalize_hidden wr xr)) =
  source_next_norm_weight_related input wr;
  source_rms_result_related input.weights v.hidden1 v.norm1 171314432 wr xr

inline_for_extraction noextract
let source_first_block_numerical_from_results (input:source_inputs) (v:source_block_values)
  : Lemma
      (requires source_first_block_arithmetic input v)
      (ensures source_first_block_numerical input v) =
  let qk = source_qk_values input.weights v.qkv0 v.rotary0 161876992 in
  let related (wr:Shared.parameters real) (xr:Shared.hidden real) (dr:Shared.rotary_tables real)
    : Lemma
        (requires Shared.parameters_approximates (source_parameters input 0) wr /\
          Shared.matrix_approximates (hidden_view v.hidden0) xr /\
          Shared.rotary_approximates (rotary_view v.rotary0) dr)
        (ensures Shared.layer_approximates (source_first_block_native_layer v qk)
          (Shared.decoder_layer wr dr xr)) =
    source_first_block_layer_related input v qk wr xr dr
  in
  C.forall_intro_3 (fun wr xr -> C.move_requires (related wr xr));
  let next_norm (wr:Shared.vector real 1024) (xr:Shared.hidden real)
    : Lemma
        (requires Shared.vector_approximates (source_parameters input 1).input_layernorm wr /\
          Shared.matrix_approximates (hidden_view v.hidden1) xr)
        (ensures Shared.matrix_approximates (hidden_view v.norm1) (Shared.normalize_hidden wr xr)) =
    source_first_block_next_norm_related input v wr xr
  in C.forall_intro_2 (fun wr -> C.move_requires (next_norm wr))
#pop-options

inline_for_extraction noextract
let source_first_block_numerical_from_stage (input:source_inputs) (v:source_block_values)
  : Lemma
      (requires source_block_stage input v 8)
      (ensures source_first_block_numerical input v) =
  assert (source_first_block_arithmetic input v);
  source_first_block_numerical_from_results input v

noextract
let rec source_bank_storage (arrays:list (PA.array BF16.t))
  (contents:list (S.seq BF16.t)) (fraction:Permission.perm) : PC.slprop =
  match arrays, contents with
  | [], [] -> emp
  | a::rest, v::tail -> PA.pts_to a #fraction v ** source_bank_storage rest tail fraction
  | _ -> PC.pure False

inline_for_extraction noextract
let source_block_storage_bank (a:source_block_addresses) (v:source_block_values)
  (fraction:Permission.perm)
  : Lemma (
      List.length (source_block_arrays a) == 13 /\
      List.length (source_block_sequences v) == 13 /\
      source_block_storage a v fraction ==
        source_bank_storage (source_block_arrays a) (source_block_sequences v) fraction) =
  assert_norm (List.length (source_block_arrays a) == 13);
  assert_norm (List.length (source_block_sequences v) == 13);
  assert (source_block_storage a v fraction ==
    source_bank_storage (source_block_arrays a) (source_block_sequences v) fraction) by
    (FStar.Tactics.V2.norm [delta_only
      [`%source_block_storage; `%source_bank_storage;
       `%source_block_arrays; `%source_block_sequences]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
ghost
fn rec source_bank_share (arrays:list (PA.array BF16.t))
  (#contents:list (S.seq BF16.t){List.length contents == List.length arrays})
  (#fraction:Permission.perm) (count:pos)
  requires source_bank_storage arrays contents fraction
  ensures forall+ (_:natlt count).
    source_bank_storage arrays contents (fraction /. FStar.Real.of_int count)
  decreases arrays
{
  match arrays {
    [] -> {
      assert pure (contents == []);
      rewrite source_bank_storage arrays contents fraction as source_bank_storage [] [] fraction;
      unfold source_bank_storage [] [] fraction;
      forevery_emp_intro (natlt count);
      forevery_ext (fun (_:natlt count) -> emp)
        (fun _ -> source_bank_storage arrays contents (fraction /. FStar.Real.of_int count));
    }
    a::rest -> {
      let v = List.hd contents;
      let tail = List.tail contents;
      assert pure (contents == v::tail);
      rewrite source_bank_storage arrays contents fraction as
        source_bank_storage (a::rest) (v::tail) fraction;
      unfold source_bank_storage (a::rest) (v::tail) fraction;
      ArrayExtra.array_share a #v #fraction count;
      source_bank_share rest #tail #fraction count;
      forevery_zip
        (fun (_:natlt count) -> PA.pts_to a #(fraction /. FStar.Real.of_int count) v)
        (fun (_:natlt count) -> source_bank_storage rest tail (fraction /. FStar.Real.of_int count));
      forevery_ext
        (fun (_:natlt count) -> PA.pts_to a #(fraction /. FStar.Real.of_int count) v **
          source_bank_storage rest tail (fraction /. FStar.Real.of_int count))
        (fun (_:natlt count) -> source_bank_storage arrays contents (fraction /. FStar.Real.of_int count));
    }
  }
}

inline_for_extraction noextract
ghost
fn rec source_bank_gather (arrays:list (PA.array BF16.t))
  (#contents:list (S.seq BF16.t){List.length contents == List.length arrays})
  (#fraction:Permission.perm) (count:pos)
  requires forall+ (_:natlt count).
    source_bank_storage arrays contents (fraction /. FStar.Real.of_int count)
  ensures source_bank_storage arrays contents fraction
  decreases arrays
{
  match arrays {
    [] -> {
      assert pure (contents == []);
      forevery_ext
        (fun (_:natlt count) -> source_bank_storage arrays contents (fraction /. FStar.Real.of_int count))
        (fun (_:natlt count) -> emp);
      forevery_emp_elim (natlt count);
      fold source_bank_storage [] [] fraction;
      rewrite source_bank_storage [] [] fraction as source_bank_storage arrays contents fraction;
    }
    a::rest -> {
      let v = List.hd contents;
      let tail = List.tail contents;
      assert pure (contents == v::tail);
      forevery_ext
        (fun (_:natlt count) -> source_bank_storage arrays contents (fraction /. FStar.Real.of_int count))
        (fun (_:natlt count) -> PA.pts_to a #(fraction /. FStar.Real.of_int count) v **
          source_bank_storage rest tail (fraction /. FStar.Real.of_int count));
      forevery_unzip
        (fun (_:natlt count) -> PA.pts_to a #(fraction /. FStar.Real.of_int count) v)
        (fun (_:natlt count) -> source_bank_storage rest tail (fraction /. FStar.Real.of_int count));
      ArrayExtra.array_gather a #v #fraction count;
      source_bank_gather rest #tail #fraction count;
      fold source_bank_storage (a::rest) (v::tail) fraction;
      rewrite source_bank_storage (a::rest) (v::tail) fraction as
        source_bank_storage arrays contents fraction;
    }
  }
}

inline_for_extraction noextract
ghost
fn source_block_share (a:source_block_addresses) (#v:source_block_values)
  (#fraction:Permission.perm) (count:pos)
  requires source_block_storage a v fraction
  ensures forall+ (_:natlt count).
    source_block_storage a v (fraction /. FStar.Real.of_int count)
{
  source_block_storage_bank a v fraction;
  rewrite source_block_storage a v fraction as
    source_bank_storage (source_block_arrays a) (source_block_sequences v) fraction;
  source_bank_share (source_block_arrays a) #(source_block_sequences v) #fraction count;
  source_block_storage_bank a v (fraction /. FStar.Real.of_int count);
  forevery_ext
    (fun (_:natlt count) -> source_bank_storage (source_block_arrays a)
      (source_block_sequences v) (fraction /. FStar.Real.of_int count))
    (fun (_:natlt count) -> source_block_storage a v (fraction /. FStar.Real.of_int count));
}

inline_for_extraction noextract
ghost
fn source_block_gather (a:source_block_addresses) (#v:source_block_values)
  (#fraction:Permission.perm) (count:pos)
  requires forall+ (_:natlt count).
    source_block_storage a v (fraction /. FStar.Real.of_int count)
  ensures source_block_storage a v fraction
{
  source_block_storage_bank a v (fraction /. FStar.Real.of_int count);
  forevery_ext
    (fun (_:natlt count) -> source_block_storage a v (fraction /. FStar.Real.of_int count))
    (fun (_:natlt count) -> source_bank_storage (source_block_arrays a)
      (source_block_sequences v) (fraction /. FStar.Real.of_int count));
  source_bank_gather (source_block_arrays a) #(source_block_sequences v) #fraction count;
  source_block_storage_bank a v fraction;
  rewrite source_bank_storage (source_block_arrays a) (source_block_sequences v) fraction as
    source_block_storage a v fraction;
}

inline_for_extraction noextract
ghost
fn source_block_values_agree (a:source_block_addresses)
  (#left #right:source_block_values) (#f #g:Permission.perm)
  preserves source_block_storage a left f ** source_block_storage a right g
  ensures pure (left == right)
{
  unfold source_block_storage a left f;
  unfold source_block_storage a right g;
  PA.pts_to_injective_eq #BF16.t #f #g #left.hidden0 #right.hidden0 a.b_hidden0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.rotary0 #right.rotary0 a.b_rotary0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.norm0 #right.norm0 a.b_norm0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.qkv0 #right.qkv0 a.b_qkv0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.attn0 #right.attn0 a.b_attn0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.projected0 #right.projected0 a.b_projected0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.residual0 #right.residual0 a.b_residual0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.postnorm0 #right.postnorm0 a.b_postnorm0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.gateup0 #right.gateup0 a.b_gateup0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.activated0 #right.activated0 a.b_activated0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.down0 #right.down0 a.b_down0;
  PA.pts_to_injective_eq #BF16.t #f #g #left.hidden1 #right.hidden1 a.b_hidden1;
  PA.pts_to_injective_eq #BF16.t #f #g #left.norm1 #right.norm1 a.b_norm1;
  fold source_block_storage a left f;
  fold source_block_storage a right g;
}

inline_for_extraction noextract
let source_block_condition (a:source_block_addresses)
  (fact:source_block_values -> prop) (fraction:Permission.perm) : PC.slprop =
  exists* (v:source_block_values).
    source_block_storage a v fraction ** pure (fact v)

inline_for_extraction noextract
ghost
fn source_block_pack (a:source_block_addresses)
  (fact:source_block_values -> prop)
  (#v:source_block_values{fact v}) (#fraction:Permission.perm) (_:natlt 32)
  requires source_block_storage a v fraction
  ensures source_block_condition a fact fraction
{
  fold source_block_condition a fact fraction;
}

inline_for_extraction noextract
ghost
fn source_block_distribute (a:source_block_addresses)
  (fact:source_block_values -> prop) ()
  norewrite
  requires source_block_condition a fact 1.0R
  ensures (forall+ (_:natlt 32). source_block_condition a fact (1.0R /. 32)) ** emp
{
  unfold source_block_condition a fact 1.0R;
  with v. assert source_block_storage a v 1.0R ** pure (fact v);
  source_block_share a #v #1.0R 32;
  forevery_map
    (fun (_:natlt 32) -> source_block_storage a v (1.0R /. 32))
    (fun (_:natlt 32) -> source_block_condition a fact (1.0R /. 32))
    (source_block_pack a fact #v #(1.0R /. 32));
}

inline_for_extraction noextract
ghost
fn source_block_reveal (a:source_block_addresses)
  (fact:source_block_values -> prop) (fraction:Permission.perm)
  requires source_block_condition a fact fraction
  returns observed:erased source_block_values
  ensures source_block_storage a observed fraction ** pure (fact observed)
{
  unfold source_block_condition a fact fraction;
  with v. assert source_block_storage a v fraction ** pure (fact v);
  Ghost.hide (Ghost.reveal v)
}

inline_for_extraction noextract
ghost
fn source_block_align (a:source_block_addresses)
  (fact:source_block_values -> prop)
  (#v:source_block_values) (_:natlt 31)
  requires source_block_storage a v (1.0R /. 32) **
    source_block_condition a fact (1.0R /. 32)
  ensures source_block_storage a v (1.0R /. 32) **
    source_block_storage a v (1.0R /. 32)
{
  let other = source_block_reveal a fact (1.0R /. 32);
  source_block_values_agree a #v #other #(1.0R /. 32) #(1.0R /. 32);
  rewrite source_block_storage a other (1.0R /. 32) as source_block_storage a v (1.0R /. 32);
}

inline_for_extraction noextract
ghost
fn source_block_reassemble (a:source_block_addresses)
  (fact:source_block_values -> prop) ()
  norewrite
  requires (forall+ (_:natlt 32). source_block_condition a fact (1.0R /. 32)) ** emp
  ensures source_block_condition a fact 1.0R
{
  forevery_natlt_pop 32 (fun (_:natlt 32) -> source_block_condition a fact (1.0R /. 32));
  unfold source_block_condition a fact (1.0R /. 32);
  with v. assert source_block_storage a v (1.0R /. 32) ** pure (fact v);
  forevery_map_extra
    (source_block_storage a v (1.0R /. 32))
    (fun (_:natlt 31) -> source_block_condition a fact (1.0R /. 32))
    (fun (_:natlt 31) -> source_block_storage a v (1.0R /. 32))
    (source_block_align a fact #v);
  forevery_natlt_push 32 (fun (_:natlt 32) -> source_block_storage a v (1.0R /. 32));
  source_block_gather a #v #1.0R 32;
  fold source_block_condition a fact 1.0R;
}

inline_for_extraction noextract
ghost
fn source_descriptor_frame_setup (#pre #post:PC.slprop)
  (descriptor:Kernel.kernel_desc pre post) (frame:PC.slprop) ()
  norewrite
  requires pre ** frame
  ensures (forall+ (bid:natlt descriptor.nblk). descriptor.block_pre bid) **
    (descriptor.frame ** frame)
{
  let setup = descriptor.setup;
  setup ();
}

inline_for_extraction noextract
ghost
fn source_descriptor_frame_teardown (#pre #post:PC.slprop)
  (descriptor:Kernel.kernel_desc pre post) (frame:PC.slprop) ()
  norewrite
  requires (forall+ (bid:natlt descriptor.nblk). descriptor.block_post bid) **
    (descriptor.frame ** frame)
  ensures post ** frame
{
  let teardown = descriptor.teardown;
  teardown ();
}

inline_for_extraction noextract
let source_descriptor_with_frame (#pre #post:PC.slprop)
  (descriptor:Kernel.kernel_desc pre post) (frame:PC.slprop)
  : Kernel.kernel_desc (pre ** frame) (post ** frame) =
  { descriptor with
    frame = descriptor.frame ** frame;
    setup = source_descriptor_frame_setup descriptor frame;
    teardown = source_descriptor_frame_teardown descriptor frame }

inline_for_extraction noextract
let source_block_linear_width (slot:Shared.index 4) : pos =
  if slot = 1 then 2048 else if slot = 3 then 3072 else 1024

inline_for_extraction noextract
let source_block_linear_columns (slot:Shared.index 4) : pos =
  if slot = 0 then 4096 else if slot = 2 then 6144 else 1024

inline_for_extraction noextract
let source_block_linear_offset (slot:Shared.index 4) : nat =
  if slot = 0 then 155585536 else if slot = 1 then 159779840
  else if slot = 2 then 161877248 else 168168704

inline_for_extraction noextract
let source_block_linear_splits (slot:Shared.index 4) : pos =
  if slot = 3 then 2 else 1

inline_for_extraction noextract
let source_block_linear_steps (slot:Shared.index 4) : pos =
  if slot = 1 then 128 else if slot = 3 then 96 else 64

inline_for_extraction noextract
let source_block_linear_dimensions (slot:Shared.index 4)
  : Lemma (
      16 /?+ source_block_linear_width slot /\
      16 /?+ source_block_linear_columns slot /\
      source_block_linear_steps slot * source_block_linear_splits slot ==
        source_block_linear_width slot / 16 /\
      source_block_linear_offset slot +
        source_block_linear_width slot * source_block_linear_columns slot <= weight_elements) = ()

inline_for_extraction noextract
let source_block_linear_input (a:source_block_addresses) (slot:Shared.index 4)
  : GTot (PA.larray BF16.t (source_block_linear_width slot * 7)) =
  if slot = 0 then a.b_norm0 else if slot = 1 then a.b_attn0
  else if slot = 2 then a.b_postnorm0 else a.b_activated0

inline_for_extraction noextract
let source_block_linear_output (a:source_block_addresses) (slot:Shared.index 4)
  : GTot (PA.larray BF16.t (source_block_linear_columns slot * 7)) =
  if slot = 0 then a.b_qkv0 else if slot = 1 then a.b_projected0
  else if slot = 2 then a.b_gateup0 else a.b_down0

inline_for_extraction noextract
let source_block_linear_x (v:source_block_values) (slot:Shared.index 4)
  : GTot (values BF16.t (source_block_linear_width slot * 7)) =
  if slot = 0 then v.norm0 else if slot = 1 then v.attn0
  else if slot = 2 then v.postnorm0 else v.activated0

inline_for_extraction noextract
let source_block_linear_old (v:source_block_values) (slot:Shared.index 4)
  : GTot (values BF16.t (source_block_linear_columns slot * 7)) =
  if slot = 0 then v.qkv0 else if slot = 1 then v.projected0
  else if slot = 2 then v.gateup0 else v.down0

inline_for_extraction noextract
let source_block_linear_update (v:source_block_values) (slot:Shared.index 4)
  (result:values BF16.t (source_block_linear_columns slot * 7)) : GTot source_block_values =
  if slot = 0 then {v with qkv0 = result}
  else if slot = 1 then {v with projected0 = result}
  else if slot = 2 then {v with gateup0 = result}
  else {v with down0 = result}

inline_for_extraction noextract
let source_block_linear_pair (a:source_block_addresses) (v:source_block_values)
  (slot:Shared.index 4) (fraction:Permission.perm) : PC.slprop =
  PA.pts_to (source_block_linear_input a slot) #fraction (source_block_linear_x v slot) **
  PA.pts_to (source_block_linear_output a slot) #fraction (source_block_linear_old v slot)

inline_for_extraction noextract
let source_block_linear_common (a:source_block_addresses) (v:source_block_values)
  (fraction:Permission.perm) : PC.slprop =
  PA.pts_to a.b_hidden0 #fraction v.hidden0 ** PA.pts_to a.b_rotary0 #fraction v.rotary0 **
  PA.pts_to a.b_residual0 #fraction v.residual0 ** PA.pts_to a.b_hidden1 #fraction v.hidden1 **
  PA.pts_to a.b_norm1 #fraction v.norm1

inline_for_extraction noextract
let source_block_linear_frame (a:source_block_addresses) (v:source_block_values)
  (slot:Shared.index 4) (fraction:Permission.perm) : PC.slprop =
  source_block_linear_common a v fraction **
  (if slot = 0 then emp else source_block_linear_pair a v 0 fraction) **
  (if slot = 1 then emp else source_block_linear_pair a v 1 fraction) **
  (if slot = 2 then emp else source_block_linear_pair a v 2 fraction) **
  (if slot = 3 then emp else source_block_linear_pair a v 3 fraction)

inline_for_extraction noextract
ghost
fn source_block_open_linear (a:source_block_addresses) (#v:source_block_values)
  (slot:Shared.index 4) (#fraction:Permission.perm)
  requires source_block_storage a v fraction
  ensures source_block_linear_pair a v slot fraction ** source_block_linear_frame a v slot fraction
{
  unfold source_block_storage a v fraction;
  fold source_block_linear_common a v fraction;
  fold source_block_linear_pair a v 0 fraction;
  fold source_block_linear_pair a v 1 fraction;
  fold source_block_linear_pair a v 2 fraction;
  fold source_block_linear_pair a v 3 fraction;
  if (slot = 0) {
    fold source_block_linear_frame a v 0 fraction;
    rewrite source_block_linear_frame a v 0 fraction as source_block_linear_frame a v slot fraction;
    rewrite source_block_linear_pair a v 0 fraction as source_block_linear_pair a v slot fraction;
  } else if (slot = 1) {
    fold source_block_linear_frame a v 1 fraction;
    rewrite source_block_linear_frame a v 1 fraction as source_block_linear_frame a v slot fraction;
    rewrite source_block_linear_pair a v 1 fraction as source_block_linear_pair a v slot fraction;
  } else if (slot = 2) {
    fold source_block_linear_frame a v 2 fraction;
    rewrite source_block_linear_frame a v 2 fraction as source_block_linear_frame a v slot fraction;
    rewrite source_block_linear_pair a v 2 fraction as source_block_linear_pair a v slot fraction;
  } else {
    fold source_block_linear_frame a v 3 fraction;
    rewrite source_block_linear_frame a v 3 fraction as source_block_linear_frame a v slot fraction;
    rewrite source_block_linear_pair a v 3 fraction as source_block_linear_pair a v slot fraction;
  };
}

inline_for_extraction noextract
ghost
fn source_block_close_linear (a:source_block_addresses) (#v:source_block_values)
  (slot:Shared.index 4)
  (#result:values BF16.t (source_block_linear_columns slot * 7))
  (#fraction:Permission.perm)
  requires
    PA.pts_to (source_block_linear_input a slot) #fraction (source_block_linear_x v slot) **
    PA.pts_to (source_block_linear_output a slot) #fraction result **
    source_block_linear_frame a v slot fraction
  ensures source_block_storage a (source_block_linear_update v slot result) fraction
{
  if (slot = 0) {
    rewrite source_block_linear_frame a v slot fraction as source_block_linear_frame a v 0 fraction;
    unfold source_block_linear_frame a v 0 fraction;
    unfold source_block_linear_common a v fraction;
    unfold source_block_linear_pair a v 1 fraction;
    unfold source_block_linear_pair a v 2 fraction;
    unfold source_block_linear_pair a v 3 fraction;
    rewrite PA.pts_to (source_block_linear_input a slot) #fraction (source_block_linear_x v slot) as
      PA.pts_to a.b_norm0 #fraction v.norm0;
    rewrite PA.pts_to (source_block_linear_output a slot) #fraction result as
      PA.pts_to a.b_qkv0 #fraction result;
    fold source_block_storage a (source_block_linear_update v 0 result) fraction;
    rewrite source_block_storage a (source_block_linear_update v 0 result) fraction as
      source_block_storage a (source_block_linear_update v slot result) fraction;
  } else if (slot = 1) {
    rewrite source_block_linear_frame a v slot fraction as source_block_linear_frame a v 1 fraction;
    unfold source_block_linear_frame a v 1 fraction;
    unfold source_block_linear_common a v fraction;
    unfold source_block_linear_pair a v 0 fraction;
    unfold source_block_linear_pair a v 2 fraction;
    unfold source_block_linear_pair a v 3 fraction;
    rewrite PA.pts_to (source_block_linear_input a slot) #fraction (source_block_linear_x v slot) as
      PA.pts_to a.b_attn0 #fraction v.attn0;
    rewrite PA.pts_to (source_block_linear_output a slot) #fraction result as
      PA.pts_to a.b_projected0 #fraction result;
    fold source_block_storage a (source_block_linear_update v 1 result) fraction;
    rewrite source_block_storage a (source_block_linear_update v 1 result) fraction as
      source_block_storage a (source_block_linear_update v slot result) fraction;
  } else if (slot = 2) {
    rewrite source_block_linear_frame a v slot fraction as source_block_linear_frame a v 2 fraction;
    unfold source_block_linear_frame a v 2 fraction;
    unfold source_block_linear_common a v fraction;
    unfold source_block_linear_pair a v 0 fraction;
    unfold source_block_linear_pair a v 1 fraction;
    unfold source_block_linear_pair a v 3 fraction;
    rewrite PA.pts_to (source_block_linear_input a slot) #fraction (source_block_linear_x v slot) as
      PA.pts_to a.b_postnorm0 #fraction v.postnorm0;
    rewrite PA.pts_to (source_block_linear_output a slot) #fraction result as
      PA.pts_to a.b_gateup0 #fraction result;
    fold source_block_storage a (source_block_linear_update v 2 result) fraction;
    rewrite source_block_storage a (source_block_linear_update v 2 result) fraction as
      source_block_storage a (source_block_linear_update v slot result) fraction;
  } else {
    rewrite source_block_linear_frame a v slot fraction as source_block_linear_frame a v 3 fraction;
    unfold source_block_linear_frame a v 3 fraction;
    unfold source_block_linear_common a v fraction;
    unfold source_block_linear_pair a v 0 fraction;
    unfold source_block_linear_pair a v 1 fraction;
    unfold source_block_linear_pair a v 2 fraction;
    rewrite PA.pts_to (source_block_linear_input a slot) #fraction (source_block_linear_x v slot) as
      PA.pts_to a.b_activated0 #fraction v.activated0;
    rewrite PA.pts_to (source_block_linear_output a slot) #fraction result as
      PA.pts_to a.b_down0 #fraction result;
    fold source_block_storage a (source_block_linear_update v 3 result) fraction;
    rewrite source_block_storage a (source_block_linear_update v 3 result) fraction as
      source_block_storage a (source_block_linear_update v slot result) fraction;
  };
}

inline_for_extraction noextract
let source_block_linear_progress (input:source_inputs) (v:source_block_values)
  (slot:Shared.index 4)
  : Lemma
      (requires source_block_stage input v (2 * slot))
      (ensures
        source_block_stage input
          (source_block_linear_update v slot
            (source_mma_arrays_array_result
              #(source_block_linear_columns slot) #(source_block_linear_width slot) #weight_elements
              input.weights (source_block_linear_x v slot) (source_block_linear_old v slot)
              (source_block_linear_offset slot) (source_block_linear_steps slot)
              (source_block_linear_splits slot))) (2 * slot + 1)) =
  source_block_linear_dimensions slot;
  source_linear_old_independent
    #(source_block_linear_columns slot) #(source_block_linear_width slot) #weight_elements
    input.weights (source_block_linear_x v slot) (source_block_linear_old v slot)
    (S.create (source_block_linear_columns slot * 7) (F.zero #BF16.t))
    (source_block_linear_offset slot) (source_block_linear_steps slot) (source_block_linear_splits slot)

inline_for_extraction noextract
fn source_block_linear_thread
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (#slot:erased (Shared.index 4))
  (weights:packed_weights_address)
  (x:PA.larray BF16.t (source_block_linear_width slot * 7){
    x == source_block_linear_input a slot})
  (output:PA.larray BF16.t (source_block_linear_columns slot * 7){
    output == source_block_linear_output a slot})
  (columns:szp{DeviceSize.v columns == source_block_linear_columns slot})
  (width:szp{DeviceSize.v width == source_block_linear_width slot})
  (offset:sz{DeviceSize.v offset == source_block_linear_offset slot})
  (steps:szp{DeviceSize.v steps == source_block_linear_steps slot})
  (splits:szp{DeviceSize.v splits == source_block_linear_splits slot})
  (tid:szlt 32) ()
  norewrite
  requires gpu ** (PA.pts_to weights #(1.0R /. 32) (Ghost.reveal input).weights **
    source_block_condition a (fun v -> source_block_stage input v (2 * slot)) (1.0R /. 32)) **
    thread_id 32 tid
  ensures gpu ** (PA.pts_to weights #(1.0R /. 32) (Ghost.reveal input).weights **
    source_block_condition a (fun v -> source_block_stage input v (2 * slot + 1)) (1.0R /. 32)) **
    thread_id 32 tid
{
  source_block_linear_dimensions slot;
  unfold source_block_condition a (fun v -> source_block_stage input v (2 * slot)) (1.0R /. 32);
  with before. assert source_block_storage a before (1.0R /. 32) **
    pure (source_block_stage input before (2 * slot));
  source_block_open_linear a #before slot #(1.0R /. 32);
  unfold source_block_linear_pair a before slot (1.0R /. 32);
  rewrite PA.pts_to (source_block_linear_input a slot) #(1.0R /. 32)
    (source_block_linear_x before slot) as PA.pts_to x #(1.0R /. 32) (source_block_linear_x before slot);
  rewrite PA.pts_to (source_block_linear_output a slot) #(1.0R /. 32)
    (source_block_linear_old before slot) as
    PA.pts_to output #(1.0R /. 32) (source_block_linear_old before slot);
  source_mma_arrays_linear_arrays columns width #weight_elements weights x output offset steps splits
    #(1.0R /. 32) #(1.0R /. 32) #(Ghost.hide (Ghost.reveal input).weights)
    #(Ghost.hide (source_block_linear_x before slot)) #(Ghost.hide (source_block_linear_old before slot));
  let result = Ghost.hide (source_mma_arrays_array_result #columns #width #weight_elements
    (Ghost.reveal input).weights (source_block_linear_x before slot) (source_block_linear_old before slot)
    offset steps splits);
  rewrite PA.pts_to x #(1.0R /. 32) (source_block_linear_x before slot) as
    PA.pts_to (source_block_linear_input a slot) #(1.0R /. 32) (source_block_linear_x before slot);
  rewrite PA.pts_to output #(1.0R /. 32) result as
    PA.pts_to (source_block_linear_output a slot) #(1.0R /. 32) result;
  source_block_close_linear a #before slot #result #(1.0R /. 32);
  source_block_linear_progress input before slot;
  fold source_block_condition a (fun v -> source_block_stage input v (2 * slot + 1)) (1.0R /. 32);
}

inline_for_extraction noextract
let source_block_core (weights:packed_weights_address) (a:source_block_addresses)
  (input:source_inputs) (stage:nat{stage <= 8}) (fraction:Permission.perm) : PC.slprop =
  PA.pts_to weights #fraction input.weights **
  source_block_condition a (fun v -> source_block_stage input v stage) fraction

inline_for_extraction noextract
let source_block_attention_progress (input:source_inputs) (before:source_block_values)
  (result:values BF16.t 14336) (observation:values BF16.t 21504)
  : Lemma
      (requires source_block_stage input before 1 /\
        source_attention_result input.weights before.qkv0 before.rotary0 observation result 161876992)
      (ensures source_block_stage input {before with attn0 = result} 2) =
  C.exists_intro
    (fun qk -> source_attention_result input.weights before.qkv0 before.rotary0 qk result 161876992)
    observation

inline_for_extraction noextract
let source_block_postnorm_progress (input:source_inputs) (before:source_block_values)
  (residual normalized:source_hidden_values)
  : Lemma
      (requires source_block_stage input before 3 /\
        source_add_norm_result input.weights before.hidden0 before.projected0
          residual normalized 155584512)
      (ensures source_block_stage input
        {before with residual0 = residual; postnorm0 = normalized} 4) = ()

inline_for_extraction noextract
let source_block_silu_progress (input:source_inputs) (before:source_block_values)
  (result:values BF16.t 21504)
  : Lemma
      (requires source_block_stage input before 5 /\ source_silu_result before.gateup0 result)
      (ensures source_block_stage input {before with activated0 = result} 6) = ()

inline_for_extraction noextract
let source_block_nextnorm_progress (input:source_inputs) (before:source_block_values)
  (hidden normalized:source_hidden_values)
  : Lemma
      (requires source_block_stage input before 7 /\
        source_add_norm_result input.weights before.residual0 before.down0
          hidden normalized 171314432)
      (ensures source_block_stage input {before with hidden1 = hidden; norm1 = normalized} 8) = ()

inline_for_extraction noextract
fn source_block_attention_thread
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address)
  (qkv:PA.larray BF16.t 28672{qkv == (Ghost.reveal a).b_qkv0})
  (rotary:PA.larray BF16.t 1792{rotary == (Ghost.reveal a).b_rotary0})
  (output:PA.larray BF16.t 14336{output == (Ghost.reveal a).b_attn0})
  preserves gpu
  requires source_block_core weights a input 1 1.0R
  ensures source_block_core weights a input 2 1.0R
{
  unfold source_block_core weights a input 1 1.0R;
  unfold source_block_condition a (fun v -> source_block_stage input v 1) 1.0R;
  with before. assert source_block_storage a before 1.0R ** pure (source_block_stage input before 1);
  unfold source_block_storage a before 1.0R;
  rewrite each (Ghost.reveal a).b_qkv0 as qkv;
  rewrite each (Ghost.reveal a).b_rotary0 as rotary;
  rewrite each (Ghost.reveal a).b_attn0 as output;
  let offset = DeviceSize.uint32_to_sizet 161876992ul;
  let observed = source_attention_body weights qkv rotary output offset
    #(Ghost.hide (Ghost.reveal input).weights) #(Ghost.hide before.qkv0)
    #(Ghost.hide before.rotary0) #(Ghost.hide before.attn0);
  with result. assert PA.pts_to output result;
  source_block_attention_progress input before result observed;
  rewrite each qkv as (Ghost.reveal a).b_qkv0;
  rewrite each rotary as (Ghost.reveal a).b_rotary0;
  rewrite each output as (Ghost.reveal a).b_attn0;
  let after = Ghost.hide {before with attn0 = result};
  fold source_block_storage a after 1.0R;
  fold source_block_condition a (fun v -> source_block_stage input v 2) 1.0R;
  fold source_block_core weights a input 2 1.0R;
}

inline_for_extraction noextract
fn source_block_postnorm_thread
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address)
  (hidden:PA.larray BF16.t 7168{hidden == (Ghost.reveal a).b_hidden0})
  (projected:PA.larray BF16.t 7168{projected == (Ghost.reveal a).b_projected0})
  (residual:PA.larray BF16.t 7168{residual == (Ghost.reveal a).b_residual0})
  (normalized:PA.larray BF16.t 7168{normalized == (Ghost.reveal a).b_postnorm0})
  preserves gpu
  requires source_block_core weights a input 3 1.0R
  ensures source_block_core weights a input 4 1.0R
{
  unfold source_block_core weights a input 3 1.0R;
  unfold source_block_condition a (fun v -> source_block_stage input v 3) 1.0R;
  with before. assert source_block_storage a before 1.0R ** pure (source_block_stage input before 3);
  unfold source_block_storage a before 1.0R;
  rewrite each (Ghost.reveal a).b_hidden0 as hidden;
  rewrite each (Ghost.reveal a).b_projected0 as projected;
  rewrite each (Ghost.reveal a).b_residual0 as residual;
  rewrite each (Ghost.reveal a).b_postnorm0 as normalized;
  let offset = DeviceSize.uint32_to_sizet 155584512ul;
  source_add_norm_body weights hidden projected residual normalized offset
    #(Ghost.hide (Ghost.reveal input).weights) #(Ghost.hide before.hidden0)
    #(Ghost.hide before.projected0) #(Ghost.hide before.residual0) #(Ghost.hide before.postnorm0);
  with cv nv. assert PA.pts_to residual cv ** PA.pts_to normalized nv;
  source_block_postnorm_progress input before cv nv;
  rewrite each hidden as (Ghost.reveal a).b_hidden0;
  rewrite each projected as (Ghost.reveal a).b_projected0;
  rewrite each residual as (Ghost.reveal a).b_residual0;
  rewrite each normalized as (Ghost.reveal a).b_postnorm0;
  let after = Ghost.hide {before with residual0 = cv; postnorm0 = nv};
  fold source_block_storage a after 1.0R;
  fold source_block_condition a (fun v -> source_block_stage input v 4) 1.0R;
  fold source_block_core weights a input 4 1.0R;
}

inline_for_extraction noextract
fn source_block_silu_thread
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address)
  (gateup:PA.larray BF16.t 43008{gateup == (Ghost.reveal a).b_gateup0})
  (activated:PA.larray BF16.t 21504{activated == (Ghost.reveal a).b_activated0})
  preserves gpu
  requires source_block_core weights a input 5 1.0R
  ensures source_block_core weights a input 6 1.0R
{
  unfold source_block_core weights a input 5 1.0R;
  unfold source_block_condition a (fun v -> source_block_stage input v 5) 1.0R;
  with before. assert source_block_storage a before 1.0R ** pure (source_block_stage input before 5);
  unfold source_block_storage a before 1.0R;
  rewrite each (Ghost.reveal a).b_gateup0 as gateup;
  rewrite each (Ghost.reveal a).b_activated0 as activated;
  source_silu_product_body gateup activated #(Ghost.hide before.gateup0) #(Ghost.hide before.activated0);
  with result. assert PA.pts_to activated result;
  source_block_silu_progress input before result;
  rewrite each gateup as (Ghost.reveal a).b_gateup0;
  rewrite each activated as (Ghost.reveal a).b_activated0;
  let after = Ghost.hide {before with activated0 = result};
  fold source_block_storage a after 1.0R;
  fold source_block_condition a (fun v -> source_block_stage input v 6) 1.0R;
  fold source_block_core weights a input 6 1.0R;
}

inline_for_extraction noextract
fn source_block_nextnorm_thread
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address)
  (residual:PA.larray BF16.t 7168{residual == (Ghost.reveal a).b_residual0})
  (down:PA.larray BF16.t 7168{down == (Ghost.reveal a).b_down0})
  (hidden:PA.larray BF16.t 7168{hidden == (Ghost.reveal a).b_hidden1})
  (normalized:PA.larray BF16.t 7168{normalized == (Ghost.reveal a).b_norm1})
  preserves gpu
  requires source_block_core weights a input 7 1.0R
  ensures source_block_core weights a input 8 1.0R
{
  unfold source_block_core weights a input 7 1.0R;
  unfold source_block_condition a (fun v -> source_block_stage input v 7) 1.0R;
  with before. assert source_block_storage a before 1.0R ** pure (source_block_stage input before 7);
  unfold source_block_storage a before 1.0R;
  rewrite each (Ghost.reveal a).b_residual0 as residual;
  rewrite each (Ghost.reveal a).b_down0 as down;
  rewrite each (Ghost.reveal a).b_hidden1 as hidden;
  rewrite each (Ghost.reveal a).b_norm1 as normalized;
  let offset = DeviceSize.uint32_to_sizet 171314432ul;
  source_add_norm_body weights residual down hidden normalized offset
    #(Ghost.hide (Ghost.reveal input).weights) #(Ghost.hide before.residual0)
    #(Ghost.hide before.down0) #(Ghost.hide before.hidden1) #(Ghost.hide before.norm1);
  with hv nv. assert PA.pts_to hidden hv ** PA.pts_to normalized nv;
  source_block_nextnorm_progress input before hv nv;
  rewrite each residual as (Ghost.reveal a).b_residual0;
  rewrite each down as (Ghost.reveal a).b_down0;
  rewrite each hidden as (Ghost.reveal a).b_hidden1;
  rewrite each normalized as (Ghost.reveal a).b_norm1;
  let after = Ghost.hide {before with hidden1 = hv; norm1 = nv};
  fold source_block_storage a after 1.0R;
  fold source_block_condition a (fun v -> source_block_stage input v 8) 1.0R;
  fold source_block_core weights a input 8 1.0R;
}

inline_for_extraction noextract
instance source_block_storage_send (a:source_block_addresses) (v:source_block_values)
  (fraction:Permission.perm) : is_send_across KL.gpu_of (source_block_storage a v fraction) = solve

inline_for_extraction noextract
instance source_block_condition_send (a:source_block_addresses)
  (fact:source_block_values -> prop) (fraction:Permission.perm)
  : is_send_across KL.gpu_of (source_block_condition a fact fraction) = solve

inline_for_extraction noextract
instance source_block_core_send (weights:packed_weights_address) (a:source_block_addresses)
  (input:source_inputs) (stage:nat{stage <= 8}) (fraction:Permission.perm)
  : is_send_across KL.gpu_of (source_block_core weights a input stage fraction) = solve

inline_for_extraction noextract
ghost
fn source_block_core_distribute (weights:packed_weights_address)
  (a:source_block_addresses) (input:source_inputs) (stage:nat{stage <= 8}) ()
  norewrite
  requires source_block_core weights a input stage 1.0R
  ensures (forall+ (_:natlt 32).
    PA.pts_to weights #(1.0R /. 32) input.weights **
    source_block_condition a (fun v -> source_block_stage input v stage) (1.0R /. 32)) ** emp
{
  unfold source_block_core weights a input stage 1.0R;
  ArrayExtra.array_share weights #input.weights #1.0R 32;
  source_block_distribute a (fun v -> source_block_stage input v stage) ();
  forevery_zip
    (fun (_:natlt 32) -> PA.pts_to weights #(1.0R /. 32) input.weights)
    (fun (_:natlt 32) ->
      source_block_condition a (fun v -> source_block_stage input v stage) (1.0R /. 32));
}

inline_for_extraction noextract
ghost
fn source_block_core_reassemble (weights:packed_weights_address)
  (a:source_block_addresses) (input:source_inputs) (stage:nat{stage <= 8}) ()
  norewrite
  requires (forall+ (_:natlt 32).
    PA.pts_to weights #(1.0R /. 32) input.weights **
    source_block_condition a (fun v -> source_block_stage input v stage) (1.0R /. 32)) ** emp
  ensures source_block_core weights a input stage 1.0R
{
  forevery_unzip
    (fun (_:natlt 32) -> PA.pts_to weights #(1.0R /. 32) input.weights)
    (fun (_:natlt 32) ->
      source_block_condition a (fun v -> source_block_stage input v stage) (1.0R /. 32));
  ArrayExtra.array_gather weights #input.weights #1.0R 32;
  source_block_reassemble a (fun v -> source_block_stage input v stage) ();
  fold source_block_core weights a input stage 1.0R;
}

inline_for_extraction noextract
let source_block_linear_descriptor
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (#slot:erased (Shared.index 4))
  (weights:packed_weights_address)
  (x:PA.larray BF16.t (source_block_linear_width slot * 7){
    x == source_block_linear_input a slot})
  (output:PA.larray BF16.t (source_block_linear_columns slot * 7){
    output == source_block_linear_output a slot})
  (columns:szp{DeviceSize.v columns == source_block_linear_columns slot})
  (width:szp{DeviceSize.v width == source_block_linear_width slot})
  (offset:sz{DeviceSize.v offset == source_block_linear_offset slot})
  (steps:szp{DeviceSize.v steps == source_block_linear_steps slot})
  (splits:szp{DeviceSize.v splits == source_block_linear_splits slot})
  : Kernel.kernel_desc
      (source_block_core weights a input (2 * slot) 1.0R)
      (source_block_core weights a input (2 * slot + 1) 1.0R) =
  let descriptor : Kernel.kernel_desc_1_n
    (source_block_core weights a input (2 * slot) 1.0R)
    (source_block_core weights a input (2 * slot + 1) 1.0R) = {
    nthr = 32sz;
    frame = emp;
    kpre = (fun _ -> PA.pts_to weights #(1.0R /. 32) (Ghost.reveal input).weights **
      source_block_condition a (fun v -> source_block_stage input v (2 * slot)) (1.0R /. 32));
    kpost = (fun _ -> PA.pts_to weights #(1.0R /. 32) (Ghost.reveal input).weights **
      source_block_condition a (fun v -> source_block_stage input v (2 * slot + 1)) (1.0R /. 32));
    f = source_block_linear_thread #a #input #slot weights x output columns width offset steps splits;
    block_setup = source_block_core_distribute weights a input (2 * slot);
    block_teardown = source_block_core_reassemble weights a input (2 * slot + 1);
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve
  } in Kernel.k1n_as_kfull descriptor

inline_for_extraction noextract
let source_block_attention_descriptor
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address)
  (qkv:PA.larray BF16.t 28672{qkv == (Ghost.reveal a).b_qkv0})
  (rotary:PA.larray BF16.t 1792{rotary == (Ghost.reveal a).b_rotary0})
  (output:PA.larray BF16.t 14336{output == (Ghost.reveal a).b_attn0})
  : Kernel.kernel_desc (source_block_core weights a input 1 1.0R)
      (source_block_core weights a input 2 1.0R) =
  let descriptor : Kernel.kernel_desc_1_1 (source_block_core weights a input 1 1.0R)
    (source_block_core weights a input 2 1.0R) = {
    f = (fun () -> source_block_attention_thread #a #input weights qkv rotary output);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
let source_block_postnorm_descriptor
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address)
  (hidden:PA.larray BF16.t 7168{hidden == (Ghost.reveal a).b_hidden0})
  (projected:PA.larray BF16.t 7168{projected == (Ghost.reveal a).b_projected0})
  (residual:PA.larray BF16.t 7168{residual == (Ghost.reveal a).b_residual0})
  (normalized:PA.larray BF16.t 7168{normalized == (Ghost.reveal a).b_postnorm0})
  : Kernel.kernel_desc (source_block_core weights a input 3 1.0R)
      (source_block_core weights a input 4 1.0R) =
  let descriptor : Kernel.kernel_desc_1_1 (source_block_core weights a input 3 1.0R)
    (source_block_core weights a input 4 1.0R) = {
    f = (fun () -> source_block_postnorm_thread #a #input weights hidden projected residual normalized);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
let source_block_silu_descriptor
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address)
  (gateup:PA.larray BF16.t 43008{gateup == (Ghost.reveal a).b_gateup0})
  (activated:PA.larray BF16.t 21504{activated == (Ghost.reveal a).b_activated0})
  : Kernel.kernel_desc (source_block_core weights a input 5 1.0R)
      (source_block_core weights a input 6 1.0R) =
  let descriptor : Kernel.kernel_desc_1_1 (source_block_core weights a input 5 1.0R)
    (source_block_core weights a input 6 1.0R) = {
    f = (fun () -> source_block_silu_thread #a #input weights gateup activated);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
let source_block_nextnorm_descriptor
  (#a:erased source_block_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address)
  (residual:PA.larray BF16.t 7168{residual == (Ghost.reveal a).b_residual0})
  (down:PA.larray BF16.t 7168{down == (Ghost.reveal a).b_down0})
  (hidden:PA.larray BF16.t 7168{hidden == (Ghost.reveal a).b_hidden1})
  (normalized:PA.larray BF16.t 7168{normalized == (Ghost.reveal a).b_norm1})
  : Kernel.kernel_desc (source_block_core weights a input 7 1.0R)
      (source_block_core weights a input 8 1.0R) =
  let descriptor : Kernel.kernel_desc_1_1 (source_block_core weights a input 7 1.0R)
    (source_block_core weights a input 8 1.0R) = {
    f = (fun () -> source_block_nextnorm_thread #a #input weights residual down hidden normalized);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

#push-options "--fuel 20 --ifuel 1"
inline_for_extraction noextract
let source_first_block_partition
  (buffers:source_allocated_buffers{
    source_allocation_metadata buffers /\ List.length buffers.bf16_buffers == 284})
  : Lemma (
      List.length (source_remaining_after_block buffers) == 271 /\
      source_remaining_after_rms buffers ==
        List.index buffers.bf16_buffers 3 :: List.index buffers.bf16_buffers 4 ::
        List.index buffers.bf16_buffers 5 :: List.index buffers.bf16_buffers 6 ::
        List.index buffers.bf16_buffers 7 :: List.index buffers.bf16_buffers 8 ::
        List.index buffers.bf16_buffers 9 :: List.index buffers.bf16_buffers 10 ::
        List.index buffers.bf16_buffers 11 :: List.index buffers.bf16_buffers 12 ::
        source_remaining_after_block buffers) =
  List.lemma_splitAt_snd_length 13 buffers.bf16_buffers
#pop-options

inline_for_extraction noextract
let source_first_block_aliases
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Lemma (
      (source_allocated_first_block buffers).b_hidden0 ==
        (source_allocated_embedding buffers).embedded_address /\
      (source_allocated_first_block buffers).b_rotary0 ==
        (source_allocated_embedding buffers).rotary_address /\
      (source_allocated_first_block buffers).b_norm0 == source_allocated_norm0 buffers) =
  source_allocation_count ();
  source_allocations_first_three buffers;
  source_first_block_shapes ()

inline_for_extraction noextract
ghost
fn source_rms_to_block
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) ()
  requires source_after_rmsnorm addresses input buffers
  ensures PC.on KL.gpu_loc (
    source_block_core addresses.weights_address (source_allocated_first_block buffers) input 0 1.0R **
    (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers))
{
  source_allocations_head_tail buffers;
  source_first_block_partition buffers;
  source_first_block_aliases buffers;
  let a = source_allocated_first_block buffers;
  let t3 = List.index buffers.bf16_buffers 3;
  let t4 = List.index buffers.bf16_buffers 4;
  let t5 = List.index buffers.bf16_buffers 5;
  let t6 = List.index buffers.bf16_buffers 6;
  let t7 = List.index buffers.bf16_buffers 7;
  let t8 = List.index buffers.bf16_buffers 8;
  let t9 = List.index buffers.bf16_buffers 9;
  let t10 = List.index buffers.bf16_buffers 10;
  let t11 = List.index buffers.bf16_buffers 11;
  let t12 = List.index buffers.bf16_buffers 12;
  let rest = source_remaining_after_block buffers;
  unfold source_after_rmsnorm addresses input buffers;
  unfold source_remaining_rms_storage buffers;
  rewrite RuntimeHelpers.all_contents (source_remaining_after_rms buffers) as
    RuntimeHelpers.all_contents (t3::t4::t5::t6::t7::t8::t9::t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t3::t4::t5::t6::t7::t8::t9::t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t4::t5::t6::t7::t8::t9::t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t5::t6::t7::t8::t9::t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t6::t7::t8::t9::t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t7::t8::t9::t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t8::t9::t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t9::t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t10::t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t11::t12::rest);
  unfold RuntimeHelpers.all_contents (t12::rest);
  rewrite RuntimeHelpers.all_contents rest as RuntimeHelpers.all_contents (source_remaining_after_block buffers);
  fold source_remaining_block_storage buffers;
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (source_first_rms_post addresses (source_allocated_embedding buffers)
        (source_allocated_norm0 buffers) input) **
      RT.contents t3 ** RT.contents t4 ** RT.contents t5 ** RT.contents t6 ** RT.contents t7 **
      RT.contents t8 ** RT.contents t9 ** RT.contents t10 ** RT.contents t11 ** RT.contents t12 **
      source_remaining_block_storage buffers)
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 0 1.0R **
      (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers)))
    fn () {
      PC.on_elim (source_first_rms_post addresses (source_allocated_embedding buffers)
        (source_allocated_norm0 buffers) input);
      unfold source_first_rms_post addresses (source_allocated_embedding buffers)
        (source_allocated_norm0 buffers) input;
      with hidden rotary norm.
        assert PA.pts_to (source_allocated_embedding buffers).embedded_address hidden **
          PA.pts_to (source_allocated_embedding buffers).rotary_address rotary **
          PA.pts_to (source_allocated_norm0 buffers) norm **
          pure (source_embedding_result input hidden rotary /\
            source_rms_result (Ghost.reveal input).weights hidden norm 155583488);
      unfold RT.contents t3;
      unfold RT.contents t4;
      unfold RT.contents t5;
      unfold RT.contents t6;
      unfold RT.contents t7;
      unfold RT.contents t8;
      unfold RT.contents t9;
      unfold RT.contents t10;
      unfold RT.contents t11;
      unfold RT.contents t12;
      with qkv. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t3) qkv);
      with attn. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t4) attn);
      with projected. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t5) projected);
      with residual. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t6) residual);
      with postnorm. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t7) postnorm);
      with gateup. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t8) gateup);
      with activated. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t9) activated);
      with down. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t10) down);
      with hidden1. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t11) hidden1);
      with norm1. assert PC.on KL.gpu_loc (PA.pts_to (RT.data t12) norm1);
      PC.on_elim (PA.pts_to (RT.data t3) qkv);
      PC.on_elim (PA.pts_to (RT.data t4) attn);
      PC.on_elim (PA.pts_to (RT.data t5) projected);
      PC.on_elim (PA.pts_to (RT.data t6) residual);
      PC.on_elim (PA.pts_to (RT.data t7) postnorm);
      PC.on_elim (PA.pts_to (RT.data t8) gateup);
      PC.on_elim (PA.pts_to (RT.data t9) activated);
      PC.on_elim (PA.pts_to (RT.data t10) down);
      PC.on_elim (PA.pts_to (RT.data t11) hidden1);
      PC.on_elim (PA.pts_to (RT.data t12) norm1);
      PA.pts_to_len (RT.data t3);
      PA.pts_to_len (RT.data t4);
      PA.pts_to_len (RT.data t5);
      PA.pts_to_len (RT.data t6);
      PA.pts_to_len (RT.data t7);
      PA.pts_to_len (RT.data t8);
      PA.pts_to_len (RT.data t9);
      PA.pts_to_len (RT.data t10);
      PA.pts_to_len (RT.data t11);
      PA.pts_to_len (RT.data t12);
      rewrite each (source_allocated_embedding buffers).embedded_address as a.b_hidden0;
      rewrite each (source_allocated_embedding buffers).rotary_address as a.b_rotary0;
      rewrite each (source_allocated_norm0 buffers) as a.b_norm0;
      rewrite each RT.data t3 as a.b_qkv0;
      rewrite each RT.data t4 as a.b_attn0;
      rewrite each RT.data t5 as a.b_projected0;
      rewrite each RT.data t6 as a.b_residual0;
      rewrite each RT.data t7 as a.b_postnorm0;
      rewrite each RT.data t8 as a.b_gateup0;
      rewrite each RT.data t9 as a.b_activated0;
      rewrite each RT.data t10 as a.b_down0;
      rewrite each RT.data t11 as a.b_hidden1;
      rewrite each RT.data t12 as a.b_norm1;
      let v = Ghost.hide {
        hidden0 = hidden; rotary0 = rotary; norm0 = norm; qkv0 = qkv; attn0 = attn;
        projected0 = projected; residual0 = residual; postnorm0 = postnorm;
        gateup0 = gateup; activated0 = activated; down0 = down; hidden1 = hidden1; norm1 = norm1
      };
      fold source_block_storage a v 1.0R;
      fold source_block_condition a (fun v -> source_block_stage input v 0) 1.0R;
      fold source_block_core addresses.weights_address a input 0 1.0R;
      PC.on_intro (source_block_core addresses.weights_address a input 0 1.0R **
        (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers));
    };
  rewrite PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 0 1.0R **
    (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers)) as
    PC.on KL.gpu_loc (
      source_block_core addresses.weights_address (source_allocated_first_block buffers) input 0 1.0R **
      (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers));
}

inline_for_extraction noextract
fn rec source_buffer_data_ptr (stream:Stream.stream_t) (buffers:list (RT.tensor BF16.t))
  (index:nat{index < List.length buffers})
  preserves KL.cpu ** RuntimeHelpers.all_owners stream buffers
  returns address:PA.array BF16.t
  ensures pure (address == RT.data (List.index buffers index))
  decreases index
{
  match buffers {
    [] -> { unreachable (); }
    value::rest -> {
      unfold RuntimeHelpers.all_owners stream buffers;
      let address = if (index = 0) {
        RT.data_ptr stream value
      } else {
        source_buffer_data_ptr stream rest (index - 1)
      };
      fold RuntimeHelpers.all_owners stream buffers;
      address
    }
  }
}

inline_for_extraction noextract
fn source_block_data_ptr (stream:Stream.stream_t)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (index:nat{index < 13 /\ index < List.length buffers.bf16_buffers /\
    index < List.length source_bf16_allocation_shapes})
  preserves KL.cpu ** source_allocated_owners stream buffers
  returns address:PA.array BF16.t
  ensures pure (address == source_allocated_array_at buffers index)
{
  source_allocations_head_tail buffers;
  unfold source_allocated_owners stream buffers;
  let address = source_buffer_data_ptr stream buffers.bf16_buffers index;
  fold source_allocated_owners stream buffers;
  address
}

inline_for_extraction noextract
ghost
fn source_queue_step_label
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (offset:nat) (post:PC.slprop)
  requires Epoch.epoch_live stream (Epoch.epoch_next (start + offset)) **
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next (start + offset))) post
  ensures Epoch.epoch_live stream (start + (offset + 1)) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + (offset + 1))) post
{
  rewrite Epoch.epoch_live stream (Epoch.epoch_next (start + offset)) as
    Epoch.epoch_live stream (start + (offset + 1));
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next (start + offset))) post as
    Pledge.pledge0 (Epoch.epoch_done stream (start + (offset + 1))) post;
}

inline_for_extraction noextract
ghost
fn source_block_result_out
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) ()
  requires PC.on KL.gpu_loc (
    source_block_core addresses.weights_address (source_allocated_first_block buffers) input 8 1.0R **
    (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers))
  ensures source_after_first_block addresses input buffers
{
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (
      source_block_core addresses.weights_address (source_allocated_first_block buffers) input 8 1.0R **
      (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers)))
    (source_after_first_block addresses input buffers)
    fn () {
      PC.on_elim (
        source_block_core addresses.weights_address (source_allocated_first_block buffers) input 8 1.0R **
        (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers));
      unfold source_block_core addresses.weights_address (source_allocated_first_block buffers) input 8 1.0R;
      unfold source_block_condition (source_allocated_first_block buffers)
        (fun v -> source_block_stage input v 8) 1.0R;
      with v. assert source_block_storage (source_allocated_first_block buffers) v 1.0R **
        pure (source_block_stage input v 8);
      let qk = Ghost.hide (source_qk_values (Ghost.reveal input).weights v.qkv0 v.rotary0 161876992);
      assert pure (source_attention_result (Ghost.reveal input).weights v.qkv0 v.rotary0 qk v.attn0 161876992);
      source_first_block_cache_frame v qk;
      source_first_block_numerical_from_stage input v;
      PC.intro_exists (fun (snapshot:source_block_values) ->
        source_block_storage (source_allocated_first_block buffers) snapshot 1.0R **
        pure (source_first_block_observation input snapshot)) v;
      PC.on_intro (
        PA.pts_to addresses.weights_address (Ghost.reveal input).weights **
        PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
        (exists* (v:source_block_values).
          source_block_storage (source_allocated_first_block buffers) v 1.0R **
          pure (source_first_block_observation input v)));
      fold source_after_first_block addresses input buffers;
    };
}

inline_for_extraction noextract
fn source_enqueue_first_block
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    source_allocated_owners stream buffers
  requires Epoch.epoch_live stream (start + 2) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 2)) (source_after_rmsnorm addresses input buffers)
  ensures Epoch.epoch_live stream (start + 10) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 10)) (source_after_first_block addresses input buffers)
{
  source_allocation_count ();
  source_allocations_head_tail buffers;
  let a = Ghost.hide (source_allocated_first_block buffers);
  let frame = PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
    source_remaining_block_storage buffers;
  Pledge.rewrite_pledge (source_after_rmsnorm addresses input buffers)
    (PC.on KL.gpu_loc (
      source_block_core addresses.weights_address (source_allocated_first_block buffers) input 0 1.0R **
      (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers)))
    (source_rms_to_block addresses buffers #input);
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + 2))
    (PC.on KL.gpu_loc (
      source_block_core addresses.weights_address (source_allocated_first_block buffers) input 0 1.0R **
      (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers))) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + 2))
      (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 0 1.0R ** frame));

  let norm0 = source_block_data_ptr stream buffers 2;
  let qkv0 = source_block_data_ptr stream buffers 3;
  let qkv_desc = source_block_linear_descriptor #a #input #(Ghost.hide #(Shared.index 4) 0)
    addresses.weights_address norm0 qkv0 4096sz 1024sz
    (DeviceSize.uint32_to_sizet 155585536ul) 64sz 1sz;
  let qkv_framed = source_descriptor_with_frame qkv_desc frame;
  Kernel.launch_kernel_full qkv_framed stream #(start + 2);
  source_queue_step_label stream #start 2
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 1 1.0R ** frame));

  let qkv0 = source_block_data_ptr stream buffers 3;
  let rotary0 = source_block_data_ptr stream buffers 1;
  let attn0 = source_block_data_ptr stream buffers 4;
  let attention_desc = source_block_attention_descriptor #a #input addresses.weights_address qkv0 rotary0 attn0;
  let attention_framed = source_descriptor_with_frame attention_desc frame;
  Kernel.launch_kernel_full attention_framed stream #(start + 3);
  source_queue_step_label stream #start 3
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 2 1.0R ** frame));

  let attn0 = source_block_data_ptr stream buffers 4;
  let projected0 = source_block_data_ptr stream buffers 5;
  let projected_desc = source_block_linear_descriptor #a #input #(Ghost.hide #(Shared.index 4) 1)
    addresses.weights_address attn0 projected0 1024sz 2048sz
    (DeviceSize.uint32_to_sizet 159779840ul) 128sz 1sz;
  let projected_framed = source_descriptor_with_frame projected_desc frame;
  Kernel.launch_kernel_full projected_framed stream #(start + 4);
  source_queue_step_label stream #start 4
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 3 1.0R ** frame));

  let hidden0 = source_block_data_ptr stream buffers 0;
  let projected0 = source_block_data_ptr stream buffers 5;
  let residual0 = source_block_data_ptr stream buffers 6;
  let postnorm0 = source_block_data_ptr stream buffers 7;
  let postnorm_desc = source_block_postnorm_descriptor #a #input addresses.weights_address
    hidden0 projected0 residual0 postnorm0;
  let postnorm_framed = source_descriptor_with_frame postnorm_desc frame;
  Kernel.launch_kernel_full postnorm_framed stream #(start + 5);
  source_queue_step_label stream #start 5
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 4 1.0R ** frame));

  let postnorm0 = source_block_data_ptr stream buffers 7;
  let gateup0 = source_block_data_ptr stream buffers 8;
  let gateup_desc = source_block_linear_descriptor #a #input #(Ghost.hide #(Shared.index 4) 2)
    addresses.weights_address postnorm0 gateup0 6144sz 1024sz
    (DeviceSize.uint32_to_sizet 161877248ul) 64sz 1sz;
  let gateup_framed = source_descriptor_with_frame gateup_desc frame;
  Kernel.launch_kernel_full gateup_framed stream #(start + 6);
  source_queue_step_label stream #start 6
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 5 1.0R ** frame));

  let gateup0 = source_block_data_ptr stream buffers 8;
  let activated0 = source_block_data_ptr stream buffers 9;
  let silu_desc = source_block_silu_descriptor #a #input addresses.weights_address gateup0 activated0;
  let silu_framed = source_descriptor_with_frame silu_desc frame;
  Kernel.launch_kernel_full silu_framed stream #(start + 7);
  source_queue_step_label stream #start 7
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 6 1.0R ** frame));

  let activated0 = source_block_data_ptr stream buffers 9;
  let down0 = source_block_data_ptr stream buffers 10;
  let down_desc = source_block_linear_descriptor #a #input #(Ghost.hide #(Shared.index 4) 3)
    addresses.weights_address activated0 down0 1024sz 3072sz
    (DeviceSize.uint32_to_sizet 168168704ul) 96sz 2sz;
  let down_framed = source_descriptor_with_frame down_desc frame;
  Kernel.launch_kernel_full down_framed stream #(start + 8);
  source_queue_step_label stream #start 8
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 7 1.0R ** frame));

  let residual0 = source_block_data_ptr stream buffers 6;
  let down0 = source_block_data_ptr stream buffers 10;
  let hidden1 = source_block_data_ptr stream buffers 11;
  let norm1 = source_block_data_ptr stream buffers 12;
  let nextnorm_desc = source_block_nextnorm_descriptor #a #input addresses.weights_address residual0 down0 hidden1 norm1;
  let nextnorm_framed = source_descriptor_with_frame nextnorm_desc frame;
  Kernel.launch_kernel_full nextnorm_framed stream #(start + 9);
  source_queue_step_label stream #start 9
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 8 1.0R ** frame));

  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + 10))
    (PC.on KL.gpu_loc (source_block_core addresses.weights_address a input 8 1.0R ** frame)) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + 10))
      (PC.on KL.gpu_loc (
        source_block_core addresses.weights_address (source_allocated_first_block buffers) input 8 1.0R **
        (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers)));
  Pledge.rewrite_pledge
    (PC.on KL.gpu_loc (
      source_block_core addresses.weights_address (source_allocated_first_block buffers) input 8 1.0R **
      (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** source_remaining_block_storage buffers)))
    (source_after_first_block addresses input buffers)
    (source_block_result_out addresses buffers #input);
}

inline_for_extraction noextract
fn first_block_refines
  (addresses:source_input_addresses) (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** source_borrowed_inputs addresses input
  returns outcome:source_prefix_outcome
  ensures (
    match outcome with
    | SourcePrefixFailed _ ->
      Epoch.epoch_live stream start **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_borrowed_inputs addresses input)
    | SourcePrefixEnqueued buffers ->
      Epoch.epoch_live stream (start + 10) **
      source_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream (start + 10))
        (source_after_first_block addresses input buffers))
{
  let outcome = source_prepare_and_enqueue_rms addresses #input stream #start ambient;
  match outcome {
    SourcePrefixFailed error -> { SourcePrefixFailed error }
    SourcePrefixEnqueued buffers -> {
      source_enqueue_first_block addresses buffers #input stream #start;
      SourcePrefixEnqueued buffers
    }
  }
}

module Cast = Kuiper.Float.Casts.Base
module ListProps = FStar.List.Tot.Properties

inline_for_extraction noextract
fn source_residual_add_body
  (left right output:PA.larray BF16.t 7168)
  (#lv #rv #old:Ghost.erased source_hidden_values)
  preserves gpu ** PA.pts_to left lv ** PA.pts_to right rv
  requires PA.pts_to output old
  ensures PA.pts_to output (source_added_values lv rv)
{
  open Pulse.Lib.Array;
  let expected = Ghost.hide (source_added_values lv rv);
  let mut index = 0sz;
  while (!index <^ 7168sz)
    invariant exists* (i:sz{i <= 7168}) (out:source_hidden_values).
      index |-> i ** PA.pts_to output out **
      pure (source_flat_prefix out (fun j -> S.index expected j) i)
    decreases (7168 - !index)
  {
    let i = !index;
    let a = left.(i);
    let b = right.(i);
    let result = SourceCasts.fcast #F32.t #BF16.t
      (F.add (SourceCasts.fcast #BF16.t #F32.t a) (SourceCasts.fcast #BF16.t #F32.t b));
    with out. assert PA.pts_to output out;
    source_flat_prefix_step out (fun j -> S.index expected j) i;
    output.(i) <- result;
    index := i +^ 1sz;
  };
  with result. assert PA.pts_to output result;
  assert pure (S.equal result expected);
  S.lemma_eq_elim result expected;
}

inline_for_extraction noextract
let source_selector_irreflexive (x:F32.t)
  : Lemma (not (F.lt x x)) =
  if F.lt x x then (
    F32.lt_ordered x x;
    F.eq_spec x x;
    F.lte_is_lt_or_eq x x;
    F.negate_lt_is_lte x x)

inline_for_extraction noextract
let source_selector_trichotomy (x y:F32.t)
  : Lemma
      (requires not (F.NaN? (F.kind x)) /\ not (F.NaN? (F.kind y)))
      (ensures
        (F.lt x y \/ F.eq x y \/ F.lt y x) /\
        (F.eq x y <==> not (F.lt x y) /\ not (F.lt y x))) =
  F.eq_spec x y;
  F.eq_spec y x;
  F.lte_is_lt_or_eq x y;
  F.lte_is_lt_or_eq y x;
  F.negate_lt_is_lte x y;
  F.negate_lt_is_lte y x

inline_for_extraction noextract
let source_selector_lt_congruent (z x y:F32.t)
  : Lemma
      (requires not (F.NaN? (F.kind z)) /\ not (F.NaN? (F.kind x)) /\
        not (F.NaN? (F.kind y)) /\ F.eq x y)
      (ensures F.lt z x <==> F.lt z y) =
  source_selector_trichotomy x y;
  source_selector_trichotomy z x;
  source_selector_trichotomy z y;
  if F.lt z x && not (F.lt z y) then (
    if F.lt y z then F32.lt_transitive y z x
    else (F.eq_spec z y; F.eq_spec x y; F.eq_spec z x))
  else if F.lt z y && not (F.lt z x) then (
    if F.lt x z then F32.lt_transitive x z y
    else (F.eq_spec z x; F.eq_spec x y; F.eq_spec z y))

inline_for_extraction noextract
let source_selector_asymmetric (x y:F32.t)
  : Lemma (requires F.lt x y) (ensures not (F.lt y x)) =
  if F.lt y x then (
    F32.lt_transitive x y x;
    source_selector_irreflexive x)

inline_for_extraction noextract
let source_selector_le_lt (x y z:F32.t)
  : Lemma
      (requires not (F.NaN? (F.kind x)) /\ not (F.NaN? (F.kind y)) /\
        not (F.NaN? (F.kind z)) /\ not (F.lt y x) /\ F.lt y z)
      (ensures F.lt x z) =
  source_selector_trichotomy x z;
  if F.lt z x then F32.lt_transitive y z x
  else if F.eq x z then source_selector_lt_congruent y x z

inline_for_extraction noextract
let source_winner_prefix (scores:values F32.t 151936)
  (count:pos{count <= 151936}) (winner:nat{winner < count}) : prop =
  if F.NaN? (F.kind (S.index scores winner)) then
    forall (j:nat{j < winner}). not (F.NaN? (F.kind (S.index scores j)))
  else
    (forall (j:nat{j < count}).
      not (F.NaN? (F.kind (S.index scores j))) /\
      not (F.lt (S.index scores winner) (S.index scores j))) /\
    (forall (j:nat{j < winner}). F.lt (S.index scores j) (S.index scores winner))

noextract
let rec source_select_prefix (scores:values F32.t 151936)
  (count:pos{count <= 151936}) : GTot (winner:nat{winner < count}) (decreases count) =
  if count = 1 then 0
  else
    let previous = source_select_prefix scores (count - 1) in
    if not (F.NaN? (F.kind (S.index scores previous))) &&
       (F.NaN? (F.kind (S.index scores (count - 1))) ||
        F.lt (S.index scores previous) (S.index scores (count - 1)))
    then count - 1
    else previous

inline_for_extraction noextract
let rec source_select_prefix_correct (scores:values F32.t 151936)
  (count:pos{count <= 151936})
  : Lemma (source_winner_prefix scores count (source_select_prefix scores count))
      (decreases count) =
  if count = 1 then source_selector_irreflexive (S.index scores 0)
  else (
    source_select_prefix_correct scores (count - 1);
    let previous = source_select_prefix scores (count - 1) in
    let best = S.index scores previous in
    let current = S.index scores (count - 1) in
    if not (F.NaN? (F.kind best)) && not (F.NaN? (F.kind current)) && F.lt best current then (
      let earlier (j:nat{j < count - 1})
        : Lemma (F.lt (S.index scores j) current /\ not (F.lt current (S.index scores j))) =
        source_selector_le_lt (S.index scores j) best current;
        source_selector_asymmetric (S.index scores j) current
      in
      C.forall_intro earlier;
      source_selector_irreflexive current))

inline_for_extraction noextract
let source_score_is_nan (value:F32.t) : result:bool{result == F.NaN? (F.kind value)} =
  Kuiper.Floating.ieee_eq_refl value;
  not (Kuiper.Floating.ieee_eq value value)

inline_for_extraction noextract
let source_token_as_int64 (index:szlt 151936)
  : token:I64.t{I64.v token == DeviceSize.v index} =
  FStar.Int.Cast.uint32_to_int64 (DeviceSize.sizet_to_u32 index)

noextract
let source_selector_state (scores:values F32.t 151936)
  (count winner:sz) (best:F32.t) : GTot prop =
  if 1 <= DeviceSize.v count && DeviceSize.v count <= 151936 && DeviceSize.v winner < DeviceSize.v count
  then DeviceSize.v winner == source_select_prefix scores (DeviceSize.v count) /\
    best == S.index scores (DeviceSize.v winner)
  else False

inline_for_extraction noextract
fn source_select_token_body
  (logits:PA.larray BF16.t (7 * 151936)) (prompt:PA.larray I64.t 7)
  (output:PA.larray I64.t 8)
  (#lv:Ghost.erased (values BF16.t (7 * 151936)))
  (#pv:Ghost.erased (values I64.t 7))
  (#old:Ghost.erased (S.seq I64.t){S.length old == 8})
  preserves gpu ** PA.pts_to logits lv ** PA.pts_to prompt pv
  requires PA.pts_to output old
  ensures exists* (result:values I64.t 8).
    PA.pts_to output result ** pure (source_selection_result lv pv result)
{
  open Pulse.Lib.Array;
  let mut position = 0sz;
  while (!position <^ 7sz)
    invariant exists* (p:sz{p <= 7}) (out:values I64.t 8).
      position |-> p ** PA.pts_to output out **
      pure (forall (i:nat{i < p}). S.index out i == S.index pv i)
    decreases (7 - !position)
  {
    let p = !position;
    let token = prompt.(p);
    output.(p) <- token;
    position := p +^ 1sz;
  };
  let scores = Ghost.hide (selector_row_scores lv);
  let limit = DeviceSize.uint32_to_sizet 151936ul;
  let offset = 6sz *^ limit;
  let initial = logits.(offset);
  let mut best = Cast.cast_bf16_to_f32 initial;
  let mut winner = 0sz;
  let mut count = 1sz;
  while (!count <^ limit)
    invariant exists* (k w:sz) (b:F32.t).
      count |-> k ** winner |-> w ** best |-> b **
      pure (source_selector_state scores k w b)
    decreases (151936 - DeviceSize.v !count)
  {
    let k = !count;
    let w = !winner;
    let previous = !best;
    let stored = logits.(offset +^ k);
    let current = Cast.cast_bf16_to_f32 stored;
    let replace = not (source_score_is_nan previous) &&
      (source_score_is_nan current || F.lt previous current);
    assert pure (source_select_prefix scores (DeviceSize.v k + 1) ==
      (if replace then DeviceSize.v k else DeviceSize.v w));
    if replace {
      best := current;
      winner := k;
    };
    count := k +^ 1sz;
  };
  let selected = !winner;
  let token = source_token_as_int64 selected;
  source_select_prefix_correct scores 151936;
  assert pure (selector_first_winner scores selected);
  with copied. assert PA.pts_to output copied;
  C.exists_intro
    (fun (chosen:Shared.token_id) ->
      I64.v (S.index (S.upd copied 7 token) 7) == chosen /\
      selector_first_winner (selector_row_scores lv) chosen) (DeviceSize.v selected);
  output.(7sz) <- token;
}

inline_for_extraction noextract
ghost
fn source_program_take
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#v:source_program_values) (#fraction:Permission.perm)
  (mask:Shared.index 284 -> prop) (index:Shared.index 284{mask index})
  requires source_program_masked_storage buffers v fraction mask
  ensures PA.pts_to (source_program_address buffers index) #fraction (source_program_slot v index) **
    source_program_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 284) i index))
{
  unfold source_program_masked_storage buffers v fraction mask;
  forevery_remove' #(Shared.index 284) mask
    (fun (i:Shared.index 284) ->
      PA.pts_to (source_program_address buffers i) #fraction (source_program_slot v i)) index;
  rewrite
    (forall+ (i:Shared.index 284{mask i /\ ~(eq2 #(Shared.index 284) i index)}).
      PA.pts_to (source_program_address buffers i) #fraction (source_program_slot v i))
    as source_program_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 284) i index));
}

inline_for_extraction noextract
ghost
fn source_program_put
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#v:source_program_values) (#fraction:Permission.perm)
  (mask:Shared.index 284 -> prop) (index:Shared.index 284{mask index})
  (#value:values BF16.t (source_program_size index))
  requires PA.pts_to (source_program_address buffers index) #fraction value **
    source_program_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 284) i index))
  ensures source_program_masked_storage buffers (source_program_replace v index value) fraction mask
{
  let after = Ghost.hide (source_program_replace v index value);
  unfold source_program_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 284) i index));
  forevery_ext
    (fun (i:Shared.index 284{mask i /\ ~(eq2 #(Shared.index 284) i index)}) ->
      PA.pts_to (source_program_address buffers i) #fraction (source_program_slot v i))
    (fun (i:Shared.index 284{mask i /\ ~(eq2 #(Shared.index 284) i index)}) ->
      PA.pts_to (source_program_address buffers i) #fraction (source_program_slot after i));
  rewrite PA.pts_to (source_program_address buffers index) #fraction value as
    PA.pts_to (source_program_address buffers index) #fraction (source_program_slot after index);
  forevery_insert #(Shared.index 284) #(fun i -> mask i /\ ~(eq2 #(Shared.index 284) i index))
    (fun i -> PA.pts_to (source_program_address buffers i) #fraction (source_program_slot after i)) index;
  forevery_refine_ext #(Shared.index 284)
    #(fun i -> (mask i /\ ~(eq2 #(Shared.index 284) i index)) \/ eq2 #(Shared.index 284) index i) mask
    (fun i -> PA.pts_to (source_program_address buffers i) #fraction (source_program_slot after i));
  fold source_program_masked_storage buffers after fraction mask;
  rewrite source_program_masked_storage buffers after fraction mask as
    source_program_masked_storage buffers (source_program_replace v index value) fraction mask;
}

inline_for_extraction noextract
let source_program_replace_same (v:source_program_values) (index:Shared.index 284)
  : Lemma (source_program_replace v index (source_program_slot v index) == v) =
  assert (S.equal (source_program_replace v index (source_program_slot v index)) v);
  S.lemma_eq_elim (source_program_replace v index (source_program_slot v index)) v

inline_for_extraction noextract
ghost
fn source_program_share
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#v:source_program_values) (#fraction:Permission.perm) (count:pos)
  requires source_program_storage buffers v fraction
  ensures forall+ (_:natlt count).
    source_program_storage buffers v (fraction /. FStar.Real.of_int count)
{
  source_program_storage_equation buffers v (fraction /. FStar.Real.of_int count);
  unfold source_program_storage buffers v fraction;
  forevery_map
    (fun (i:Shared.index 284) -> PA.pts_to (source_program_address buffers i) #fraction (source_program_slot v i))
    (fun (i:Shared.index 284) -> forall+ (_:natlt count).
      PA.pts_to (source_program_address buffers i) #(fraction /. FStar.Real.of_int count) (source_program_slot v i))
    fn i {
      ArrayExtra.array_share (source_program_address buffers i) #(source_program_slot v i) #fraction count;
    };
  forevery_commute (fun (i:Shared.index 284) (_:natlt count) ->
    PA.pts_to (source_program_address buffers i) #(fraction /. FStar.Real.of_int count) (source_program_slot v i));
  forevery_ext
    (fun (_:natlt count) -> forall+ (i:Shared.index 284).
      PA.pts_to (source_program_address buffers i) #(fraction /. FStar.Real.of_int count) (source_program_slot v i))
    (fun (_:natlt count) -> source_program_storage buffers v (fraction /. FStar.Real.of_int count));
}

inline_for_extraction noextract
ghost
fn source_program_gather
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#v:source_program_values) (#fraction:Permission.perm) (count:pos)
  requires forall+ (_:natlt count).
    source_program_storage buffers v (fraction /. FStar.Real.of_int count)
  ensures source_program_storage buffers v fraction
{
  source_program_storage_equation buffers v (fraction /. FStar.Real.of_int count);
  forevery_ext
    (fun (_:natlt count) -> source_program_storage buffers v (fraction /. FStar.Real.of_int count))
    (fun (_:natlt count) -> forall+ (i:Shared.index 284).
      PA.pts_to (source_program_address buffers i) #(fraction /. FStar.Real.of_int count) (source_program_slot v i));
  forevery_commute (fun (_:natlt count) (i:Shared.index 284) ->
    PA.pts_to (source_program_address buffers i) #(fraction /. FStar.Real.of_int count) (source_program_slot v i));
  forevery_map
    (fun (i:Shared.index 284) -> forall+ (_:natlt count).
      PA.pts_to (source_program_address buffers i) #(fraction /. FStar.Real.of_int count) (source_program_slot v i))
    (fun (i:Shared.index 284) -> PA.pts_to (source_program_address buffers i) #fraction (source_program_slot v i))
    fn i {
      ArrayExtra.array_gather (source_program_address buffers i) #(source_program_slot v i) #fraction count;
    };
  fold source_program_storage buffers v fraction;
}

inline_for_extraction noextract
ghost
fn source_program_values_agree
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#left #right:source_program_values) (#f #g:Permission.perm)
  preserves source_program_storage buffers left f ** source_program_storage buffers right g
  ensures pure (left == right)
{
  unfold source_program_storage buffers left f;
  unfold source_program_storage buffers right g;
  forevery_zip
    (fun (i:Shared.index 284) -> PA.pts_to (source_program_address buffers i) #f (source_program_slot left i))
    (fun (i:Shared.index 284) -> PA.pts_to (source_program_address buffers i) #g (source_program_slot right i));
  forevery_map
    (fun (i:Shared.index 284) ->
      PA.pts_to (source_program_address buffers i) #f (source_program_slot left i) **
      PA.pts_to (source_program_address buffers i) #g (source_program_slot right i))
    (fun (i:Shared.index 284) ->
      (PA.pts_to (source_program_address buffers i) #f (source_program_slot left i) **
       PA.pts_to (source_program_address buffers i) #g (source_program_slot right i)) **
      pure (S.index left i == S.index right i))
    fn i {
      PA.pts_to_injective_eq #BF16.t #f #g #(source_program_slot left i) #(source_program_slot right i)
        (source_program_address buffers i);
    };
  forevery_unzip
    (fun (i:Shared.index 284) ->
      PA.pts_to (source_program_address buffers i) #f (source_program_slot left i) **
      PA.pts_to (source_program_address buffers i) #g (source_program_slot right i))
    (fun (i:Shared.index 284) -> pure (S.index left i == S.index right i));
  forevery_elim_pure (fun (i:Shared.index 284) -> S.index left i == S.index right i);
  assert pure (S.equal left right);
  S.lemma_eq_elim left right;
  forevery_unzip
    (fun (i:Shared.index 284) -> PA.pts_to (source_program_address buffers i) #f (source_program_slot left i))
    (fun (i:Shared.index 284) -> PA.pts_to (source_program_address buffers i) #g (source_program_slot right i));
  fold source_program_storage buffers left f;
  fold source_program_storage buffers right g;
}

inline_for_extraction noextract
ghost
fn source_program_pack
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228})
  (#v:source_program_values{source_program_stage input v calls})
  (#fraction:Permission.perm) (_:natlt 32)
  requires source_program_storage buffers v fraction
  ensures source_program_condition buffers input calls fraction
{
  fold source_program_condition buffers input calls fraction;
}

inline_for_extraction noextract
ghost
fn source_program_distribute
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228}) ()
  norewrite
  requires source_program_condition buffers input calls 1.0R
  ensures (forall+ (_:natlt 32). source_program_condition buffers input calls (1.0R /. 32)) ** emp
{
  unfold source_program_condition buffers input calls 1.0R;
  with v. assert source_program_storage buffers v 1.0R ** pure (source_program_stage input v calls);
  source_program_share buffers #v #1.0R 32;
  forevery_map
    (fun (_:natlt 32) -> source_program_storage buffers v (1.0R /. 32))
    (fun (_:natlt 32) -> source_program_condition buffers input calls (1.0R /. 32))
    (source_program_pack buffers input calls #v #(1.0R /. 32));
}

inline_for_extraction noextract
ghost
fn source_program_reveal
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228}) (fraction:Permission.perm)
  requires source_program_condition buffers input calls fraction
  returns observed:erased source_program_values
  ensures source_program_storage buffers observed fraction ** pure (source_program_stage input observed calls)
{
  unfold source_program_condition buffers input calls fraction;
  with v. assert source_program_storage buffers v fraction ** pure (source_program_stage input v calls);
  Ghost.hide (Ghost.reveal v)
}

inline_for_extraction noextract
ghost
fn source_program_align
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228})
  (#v:source_program_values) (_:natlt 31)
  requires source_program_storage buffers v (1.0R /. 32) **
    source_program_condition buffers input calls (1.0R /. 32)
  ensures source_program_storage buffers v (1.0R /. 32) **
    source_program_storage buffers v (1.0R /. 32)
{
  let other = source_program_reveal buffers input calls (1.0R /. 32);
  source_program_values_agree buffers #v #other #(1.0R /. 32) #(1.0R /. 32);
  rewrite source_program_storage buffers other (1.0R /. 32) as
    source_program_storage buffers v (1.0R /. 32);
}

inline_for_extraction noextract
ghost
fn source_program_reassemble
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228}) ()
  norewrite
  requires (forall+ (_:natlt 32). source_program_condition buffers input calls (1.0R /. 32)) ** emp
  ensures source_program_condition buffers input calls 1.0R
{
  forevery_natlt_pop 32 (fun (_:natlt 32) -> source_program_condition buffers input calls (1.0R /. 32));
  let v = source_program_reveal buffers input calls (1.0R /. 32);
  forevery_map_extra
    (source_program_storage buffers v (1.0R /. 32))
    (fun (_:natlt 31) -> source_program_condition buffers input calls (1.0R /. 32))
    (fun (_:natlt 31) -> source_program_storage buffers v (1.0R /. 32))
    (source_program_align buffers input calls #v);
  forevery_natlt_push 32 (fun (_:natlt 32) -> source_program_storage buffers v (1.0R /. 32));
  source_program_gather buffers #v #1.0R 32;
  fold source_program_condition buffers input calls 1.0R;
}

inline_for_extraction noextract
let source_program_linear_input_index (layer:Shared.index 28) (slot:Shared.index 4) : Shared.index 284 =
  source_program_index layer (if slot = 0 then 2 else if slot = 1 then 4 else if slot = 2 then 7 else 9)

inline_for_extraction noextract
let source_program_linear_output_index (layer:Shared.index 28) (slot:Shared.index 4) : Shared.index 284 =
  source_program_index layer (if slot = 0 then 3 else if slot = 1 then 5 else if slot = 2 then 8 else 10)

inline_for_extraction noextract
let source_program_linear_offset (layer:Shared.index 28) (slot:Shared.index 4) : nat =
  layer_start layer + (if slot = 0 then 2048 else if slot = 1 then 4196352 else if slot = 2 then 6293760 else 12585216)

inline_for_extraction noextract
let source_program_linear_dimensions (layer:Shared.index 28) (slot:Shared.index 4)
  : Lemma (
      16 /?+ source_block_linear_columns slot /\ 16 /?+ source_block_linear_width slot /\
      source_program_size (source_program_linear_input_index layer slot) == source_block_linear_width slot * 7 /\
      source_program_size (source_program_linear_output_index layer slot) == source_block_linear_columns slot * 7 /\
      source_program_linear_input_index layer slot <> source_program_linear_output_index layer slot /\
      source_program_linear_offset layer slot + source_block_linear_columns slot * source_block_linear_width slot <= weight_elements /\
      source_block_linear_steps slot * source_block_linear_splits slot == source_block_linear_width slot / 16) =
  source_spans layer;
  source_block_linear_dimensions slot

inline_for_extraction noextract
let source_program_linear_values (input:source_inputs) (v:source_program_values)
  (layer:Shared.index 28) (slot:Shared.index 4)
  : GTot (values BF16.t (source_program_size (source_program_linear_output_index layer slot))) =
  source_program_linear_dimensions layer slot;
  source_linear_values (source_block_linear_columns slot) (source_block_linear_width slot)
    input.weights (source_program_slot v (source_program_linear_input_index layer slot))
    (source_program_linear_offset layer slot) (source_block_linear_steps slot) (source_block_linear_splits slot)

inline_for_extraction noextract
let source_layer_linear_progress (input:source_inputs) (v:source_block_values)
  (layer:Shared.index 28) (slot:Shared.index 4)
  : Lemma
      (requires source_layer_stage input layer v (2 * slot))
      (ensures source_layer_stage input layer
        (source_block_linear_update v slot
          (source_linear_values (source_block_linear_columns slot) (source_block_linear_width slot)
            input.weights (source_block_linear_x v slot) (source_program_linear_offset layer slot)
            (source_block_linear_steps slot) (source_block_linear_splits slot))) (2 * slot + 1)) =
  source_program_linear_dimensions layer slot

inline_for_extraction noextract
let source_program_linear_progress (input:source_inputs) (v:source_program_values)
  (layer:Shared.index 28) (slot:Shared.index 4)
  : Lemma
      (requires source_program_stage input v ((2 + 2 * slot) + 8 * layer))
      (ensures source_program_stage input
        (source_program_replace v (source_program_linear_output_index layer slot)
          (source_program_linear_values input v layer slot)) ((3 + 2 * slot) + 8 * layer)) =
  source_program_linear_dimensions layer slot;
  let output = source_program_linear_output_index layer slot in
  let result = source_program_linear_values input v layer slot in
  let after = source_program_replace v output result in
  let before_layer = source_program_layer v layer in
  source_layer_linear_progress input before_layer layer slot;
  assert (source_program_layer after layer ==
    source_block_linear_update before_layer slot result);
  let prior (other:Shared.index 28)
    : Lemma ((3 + 2 * slot) + 8 * layer >= 2 + 8 * other ==>
        source_layer_stage input other (source_program_layer after other)
          (if (3 + 2 * slot) + 8 * layer >= 10 + 8 * other then 8
           else (3 + 2 * slot) + 8 * layer - (2 + 8 * other))) =
    if other < layer then (
      assert (source_program_layer after other == source_program_layer v other);
      assert (source_layer_stage input other (source_program_layer v other) 8))
    else if other = layer then ()
  in
  C.forall_intro prior

inline_for_extraction noextract
ghost
fn source_program_restore
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#v:source_program_values) (#fraction:Permission.perm)
  (mask:Shared.index 284 -> prop) (index:Shared.index 284{mask index})
  requires PA.pts_to (source_program_address buffers index) #fraction (source_program_slot v index) **
    source_program_masked_storage buffers v fraction
      (fun i -> mask i /\ ~(eq2 #(Shared.index 284) i index))
  ensures source_program_masked_storage buffers v fraction mask
{
  source_program_put buffers #v #fraction mask index #(source_program_slot v index);
  source_program_replace_same v index;
  rewrite source_program_masked_storage buffers
    (source_program_replace v index (source_program_slot v index)) fraction mask as
    source_program_masked_storage buffers v fraction mask;
}

inline_for_extraction noextract
fn source_program_linear_thread
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (#layer:erased (Shared.index 28)) (#slot:erased (Shared.index 4))
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (x:PA.array BF16.t{x == source_program_address buffers (source_program_linear_input_index layer slot)})
  (output:PA.array BF16.t{output == source_program_address buffers (source_program_linear_output_index layer slot)})
  (columns:szp{DeviceSize.v columns == source_block_linear_columns slot})
  (width:szp{DeviceSize.v width == source_block_linear_width slot})
  (offset:sz{DeviceSize.v offset == source_program_linear_offset layer slot})
  (steps:szp{DeviceSize.v steps == source_block_linear_steps slot})
  (splits:szp{DeviceSize.v splits == source_block_linear_splits slot})
  (tid:szlt 32) ()
  norewrite
  requires gpu ** source_program_core addresses buffers input ((2 + 2 * slot) + 8 * layer) (1.0R /. 32) **
    thread_id 32 tid
  ensures gpu ** source_program_core addresses buffers input ((3 + 2 * slot) + 8 * layer) (1.0R /. 32) **
    thread_id 32 tid
{
  source_program_linear_dimensions layer slot;
  unfold source_program_core addresses buffers input ((2 + 2 * slot) + 8 * layer) (1.0R /. 32);
  let before = source_program_reveal buffers input ((2 + 2 * slot) + 8 * layer) (1.0R /. 32);
  let xi = Ghost.hide (source_program_linear_input_index layer slot);
  let yi = Ghost.hide (source_program_linear_output_index layer slot);
  source_program_storage_equation buffers before (1.0R /. 32);
  rewrite source_program_storage buffers before (1.0R /. 32) as
    source_program_masked_storage buffers before (1.0R /. 32) (fun _ -> True);
  source_program_take buffers #before #(1.0R /. 32) (fun _ -> True) xi;
  source_program_take buffers #before #(1.0R /. 32)
    (fun i -> True /\ ~(eq2 #(Shared.index 284) i xi)) yi;
  rewrite each source_program_address buffers xi as x;
  rewrite each source_program_address buffers yi as output;
  rewrite each (Ghost.reveal addresses).weights_address as weights;
  let xv = Ghost.hide #(values BF16.t (DeviceSize.v width * 7)) (source_program_slot before xi);
  let old = Ghost.hide #(values BF16.t (DeviceSize.v columns * 7)) (source_program_slot before yi);
  rewrite PA.pts_to x #(1.0R /. 32) (source_program_slot before xi) as
    PA.pts_to x #(1.0R /. 32) xv;
  rewrite PA.pts_to output #(1.0R /. 32) (source_program_slot before yi) as
    PA.pts_to output #(1.0R /. 32) old;
  source_mma_arrays_linear_arrays columns width #weight_elements weights x output offset steps splits
    #(1.0R /. 32) #(1.0R /. 32) #(Ghost.hide (Ghost.reveal input).weights) #xv #old;
  source_linear_old_independent #columns #width #weight_elements (Ghost.reveal input).weights xv old
    (S.create (columns * 7) (F.zero #BF16.t)) offset steps splits;
  let result = Ghost.hide (source_program_linear_values input before layer slot);
  rewrite PA.pts_to output #(1.0R /. 32)
    (source_mma_arrays_array_result #columns #width #weight_elements
      (Ghost.reveal input).weights xv old offset steps splits) as
    PA.pts_to output #(1.0R /. 32) result;
  rewrite each x as source_program_address buffers xi;
  rewrite each output as source_program_address buffers yi;
  source_program_put buffers #before #(1.0R /. 32)
    (fun i -> True /\ ~(eq2 #(Shared.index 284) i xi)) yi #result;
  let after = Ghost.hide (source_program_replace before yi result);
  rewrite source_program_masked_storage buffers (source_program_replace before yi result) (1.0R /. 32)
    (fun i -> True /\ ~(eq2 #(Shared.index 284) i xi)) as
    source_program_masked_storage buffers after (1.0R /. 32) (fun i -> True /\ ~(eq2 #(Shared.index 284) i xi));
  rewrite PA.pts_to (source_program_address buffers xi) #(1.0R /. 32) xv as
    PA.pts_to (source_program_address buffers xi) #(1.0R /. 32) (source_program_slot after xi);
  source_program_restore buffers #after #(1.0R /. 32) (fun _ -> True) xi;
  source_program_storage_equation buffers after (1.0R /. 32);
  rewrite source_program_masked_storage buffers after (1.0R /. 32) (fun _ -> True) as
    source_program_storage buffers after (1.0R /. 32);
  source_program_linear_progress input before layer slot;
  fold source_program_condition buffers input ((3 + 2 * slot) + 8 * layer) (1.0R /. 32);
  rewrite each weights as (Ghost.reveal addresses).weights_address;
  fold source_program_core addresses buffers input ((3 + 2 * slot) + 8 * layer) (1.0R /. 32);
}

inline_for_extraction noextract
instance source_program_storage_send
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (v:source_program_values) (fraction:Permission.perm)
  : is_send_across KL.gpu_of (source_program_storage buffers v fraction) = solve

inline_for_extraction noextract
instance source_program_condition_send
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228}) (fraction:Permission.perm)
  : is_send_across KL.gpu_of (source_program_condition buffers input calls fraction) = solve

inline_for_extraction noextract
instance source_program_core_send (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228}) (fraction:Permission.perm)
  : is_send_across KL.gpu_of (source_program_core addresses buffers input calls fraction) = solve

inline_for_extraction noextract
ghost
fn source_program_core_distribute (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228}) ()
  norewrite
  requires source_program_core addresses buffers input calls 1.0R
  ensures (forall+ (_:natlt 32). source_program_core addresses buffers input calls (1.0R /. 32)) ** emp
{
  unfold source_program_core addresses buffers input calls 1.0R;
  ArrayExtra.array_share addresses.weights_address #input.weights #1.0R 32;
  source_program_distribute buffers input calls ();
  forevery_zip
    (fun (_:natlt 32) -> PA.pts_to addresses.weights_address #(1.0R /. 32) input.weights)
    (fun (_:natlt 32) -> source_program_condition buffers input calls (1.0R /. 32));
  forevery_ext
    (fun (_:natlt 32) -> PA.pts_to addresses.weights_address #(1.0R /. 32) input.weights **
      source_program_condition buffers input calls (1.0R /. 32))
    (fun (_:natlt 32) -> source_program_core addresses buffers input calls (1.0R /. 32));
}

inline_for_extraction noextract
ghost
fn source_program_core_reassemble (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (input:source_inputs) (calls:nat{2 <= calls /\ calls <= 228}) ()
  norewrite
  requires (forall+ (_:natlt 32). source_program_core addresses buffers input calls (1.0R /. 32)) ** emp
  ensures source_program_core addresses buffers input calls 1.0R
{
  forevery_ext
    (fun (_:natlt 32) -> source_program_core addresses buffers input calls (1.0R /. 32))
    (fun (_:natlt 32) -> PA.pts_to addresses.weights_address #(1.0R /. 32) input.weights **
      source_program_condition buffers input calls (1.0R /. 32));
  forevery_unzip
    (fun (_:natlt 32) -> PA.pts_to addresses.weights_address #(1.0R /. 32) input.weights)
    (fun (_:natlt 32) -> source_program_condition buffers input calls (1.0R /. 32));
  ArrayExtra.array_gather addresses.weights_address #input.weights #1.0R 32;
  source_program_reassemble buffers input calls ();
  fold source_program_core addresses buffers input calls 1.0R;
}

inline_for_extraction noextract
let source_program_linear_descriptor
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (#layer:erased (Shared.index 28)) (#slot:erased (Shared.index 4))
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (x:PA.array BF16.t{x == source_program_address buffers (source_program_linear_input_index layer slot)})
  (output:PA.array BF16.t{output == source_program_address buffers (source_program_linear_output_index layer slot)})
  (columns:szp{DeviceSize.v columns == source_block_linear_columns slot})
  (width:szp{DeviceSize.v width == source_block_linear_width slot})
  (offset:sz{DeviceSize.v offset == source_program_linear_offset layer slot})
  (steps:szp{DeviceSize.v steps == source_block_linear_steps slot})
  (splits:szp{DeviceSize.v splits == source_block_linear_splits slot})
  : Kernel.kernel_desc
      (source_program_core addresses buffers input ((2 + 2 * slot) + 8 * layer) 1.0R)
      (source_program_core addresses buffers input ((3 + 2 * slot) + 8 * layer) 1.0R) =
  let descriptor:Kernel.kernel_desc_1_n
    (source_program_core addresses buffers input ((2 + 2 * slot) + 8 * layer) 1.0R)
    (source_program_core addresses buffers input ((3 + 2 * slot) + 8 * layer) 1.0R) = {
    nthr = 32sz;
    frame = emp;
    kpre = (fun _ -> source_program_core addresses buffers input ((2 + 2 * slot) + 8 * layer) (1.0R /. 32));
    kpost = (fun _ -> source_program_core addresses buffers input ((3 + 2 * slot) + 8 * layer) (1.0R /. 32));
    f = source_program_linear_thread #buffers #addresses #input #layer #slot weights x output columns width offset steps splits;
    block_setup = source_program_core_distribute addresses buffers input ((2 + 2 * slot) + 8 * layer);
    block_teardown = source_program_core_reassemble addresses buffers input ((3 + 2 * slot) + 8 * layer);
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve
  } in Kernel.k1n_as_kfull descriptor

inline_for_extraction noextract
let source_program_attention_progress (input:source_inputs) (v:source_program_values)
  (layer:Shared.index 28) (qk:values BF16.t 21504) (result:values BF16.t 14336)
  : Lemma
      (requires source_program_stage input v (3 + 8 * layer) /\
        source_attention_result input.weights (source_program_layer v layer).qkv0
          (source_program_layer v layer).rotary0 qk result (layer_start layer + 6293504))
      (ensures source_program_stage input
        (source_program_replace v (source_program_index layer 4) result) (4 + 8 * layer)) =
  let after = source_program_replace v (source_program_index layer 4) result in
  assert (source_program_layer after layer == {(source_program_layer v layer) with attn0 = result});
  assert (source_layer_stage input layer (source_program_layer after layer) 2);
  let prior (other:Shared.index 28)
    : Lemma (4 + 8 * layer >= 2 + 8 * other ==>
        source_layer_stage input other (source_program_layer after other)
          (if 4 + 8 * layer >= 10 + 8 * other then 8 else 4 + 8 * layer - (2 + 8 * other))) =
    if other < layer then assert (source_program_layer after other == source_program_layer v other)
  in C.forall_intro prior

inline_for_extraction noextract
let source_program_postnorm_progress (input:source_inputs) (v:source_program_values)
  (layer:Shared.index 28) (residual normalized:source_hidden_values)
  : Lemma
      (requires source_program_stage input v (5 + 8 * layer) /\
        source_add_norm_result input.weights (source_program_layer v layer).hidden0
          (source_program_layer v layer).projected0 residual normalized (layer_start layer + 1024))
      (ensures source_program_stage input
        (source_program_replace (source_program_replace v (source_program_index layer 6) residual)
          (source_program_index layer 7) normalized) (6 + 8 * layer)) =
  let after = source_program_replace (source_program_replace v (source_program_index layer 6) residual)
    (source_program_index layer 7) normalized in
  assert (source_program_layer after layer ==
    {(source_program_layer v layer) with residual0 = residual; postnorm0 = normalized});
  assert (source_layer_stage input layer (source_program_layer after layer) 4);
  let prior (other:Shared.index 28)
    : Lemma (6 + 8 * layer >= 2 + 8 * other ==>
        source_layer_stage input other (source_program_layer after other)
          (if 6 + 8 * layer >= 10 + 8 * other then 8 else 6 + 8 * layer - (2 + 8 * other))) =
    if other < layer then assert (source_program_layer after other == source_program_layer v other)
  in C.forall_intro prior

inline_for_extraction noextract
let source_program_silu_progress (input:source_inputs) (v:source_program_values)
  (layer:Shared.index 28) (result:values BF16.t 21504)
  : Lemma
      (requires source_program_stage input v (7 + 8 * layer) /\
        source_silu_result (source_program_layer v layer).gateup0 result)
      (ensures source_program_stage input
        (source_program_replace v (source_program_index layer 9) result) (8 + 8 * layer)) =
  let after = source_program_replace v (source_program_index layer 9) result in
  assert (source_program_layer after layer == {(source_program_layer v layer) with activated0 = result});
  assert (source_layer_stage input layer (source_program_layer after layer) 6);
  let prior (other:Shared.index 28)
    : Lemma (8 + 8 * layer >= 2 + 8 * other ==>
        source_layer_stage input other (source_program_layer after other)
          (if 8 + 8 * layer >= 10 + 8 * other then 8 else 8 + 8 * layer - (2 + 8 * other))) =
    if other < layer then assert (source_program_layer after other == source_program_layer v other)
  in C.forall_intro prior

inline_for_extraction noextract
let source_program_nextnorm_progress (input:source_inputs) (v:source_program_values)
  (layer:Shared.index 27) (hidden normalized:source_hidden_values)
  : Lemma
      (requires source_program_stage input v (9 + 8 * layer) /\
        source_add_norm_result input.weights (source_program_layer v layer).residual0
          (source_program_layer v layer).down0 hidden normalized (layer_start layer + layer_weight_elements))
      (ensures source_program_stage input
        (source_program_replace (source_program_replace v (source_program_index layer 11) hidden)
          (source_program_index layer 12) normalized) (10 + 8 * layer)) =
  let after = source_program_replace (source_program_replace v (source_program_index layer 11) hidden)
    (source_program_index layer 12) normalized in
  assert (source_program_layer after layer ==
    {(source_program_layer v layer) with hidden1 = hidden; norm1 = normalized});
  assert (source_layer_stage input layer (source_program_layer after layer) 8);
  assert ((source_program_layer after (layer + 1)).hidden0 == hidden /\
    (source_program_layer after (layer + 1)).norm0 == normalized);
  assert (source_layer_stage input (layer + 1) (source_program_layer after (layer + 1)) 0);
  let prior (other:Shared.index 28)
    : Lemma (10 + 8 * layer >= 2 + 8 * other ==>
        source_layer_stage input other (source_program_layer after other)
          (if 10 + 8 * layer >= 10 + 8 * other then 8 else 10 + 8 * layer - (2 + 8 * other))) =
    if other < layer then assert (source_program_layer after other == source_program_layer v other)
  in C.forall_intro prior

inline_for_extraction noextract
let source_program_residual_progress (input:source_inputs) (v:source_program_values)
  : Lemma
      (requires source_program_stage input v 225)
      (ensures source_program_stage input
        (source_program_replace v 281
          (source_added_values (source_program_slot v 276) (source_program_slot v 280))) 226) =
  let after = source_program_replace v 281
    (source_added_values (source_program_slot v 276) (source_program_slot v 280)) in
  assert (source_layer_stage input 27 (source_program_layer after 27) 8);
  let prior (layer:Shared.index 28)
    : Lemma (source_layer_stage input layer (source_program_layer after layer) 8) =
    if layer < 27 then assert (source_program_layer after layer == source_program_layer v layer)
  in C.forall_intro prior

inline_for_extraction noextract
let source_program_final_norm_progress (input:source_inputs) (v:source_program_values)
  (result:source_hidden_values)
  : Lemma
      (requires source_program_stage input v 226 /\
        source_rms_result input.weights (source_program_slot v 281) result 155582464)
      (ensures source_program_stage input (source_program_replace v 282 result) 227) =
  let after = source_program_replace v 282 result in
  let prior (layer:Shared.index 28)
    : Lemma (source_layer_stage input layer (source_program_layer after layer) 8) =
    if layer < 27 then assert (source_program_layer after layer == source_program_layer v layer)
    else assert (source_program_layer after layer == {(source_program_layer v layer) with norm1 = result})
  in C.forall_intro prior

inline_for_extraction noextract
let source_program_projection_progress (input:source_inputs) (v:source_program_values)
  : Lemma
      (requires source_program_stage input v 227)
      (ensures source_program_stage input
        (source_program_replace v 283
          (source_linear_values 151936 1024 input.weights (source_program_slot v 282) 0 64 1)) 228) =
  let after = source_program_replace v 283
    (source_linear_values 151936 1024 input.weights (source_program_slot v 282) 0 64 1) in
  let prior (layer:Shared.index 28)
    : Lemma (source_layer_stage input layer (source_program_layer after layer) 8) =
    assert (source_program_layer after layer == source_program_layer v layer)
  in C.forall_intro prior

inline_for_extraction noextract
fn source_program_attention_thread
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (#layer:erased (Shared.index 28))
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (qkv:PA.array BF16.t{qkv == source_program_address buffers (source_program_index layer 3)})
  (rotary:PA.array BF16.t{rotary == source_program_address buffers 1})
  (output:PA.array BF16.t{output == source_program_address buffers (source_program_index layer 4)})
  (offset:sz{DeviceSize.v offset == layer_start layer + 6293504})
  preserves gpu
  requires source_program_core addresses buffers input (3 + 8 * layer) 1.0R
  ensures source_program_core addresses buffers input (4 + 8 * layer) 1.0R
{
  unfold source_program_core addresses buffers input (3 + 8 * layer) 1.0R;
  let before = source_program_reveal buffers input (3 + 8 * layer) 1.0R;
  let qi = Ghost.hide (source_program_index layer 3);
  let yi = Ghost.hide (source_program_index layer 4);
  source_program_storage_equation buffers before 1.0R;
  rewrite source_program_storage buffers before 1.0R as source_program_masked_storage buffers before 1.0R (fun _ -> True);
  source_program_take buffers #before #1.0R (fun _ -> True) qi;
  source_program_take buffers #before #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i qi)) 1;
  source_program_take buffers #before #1.0R
    (fun i -> (True /\ ~(eq2 #(Shared.index 284) i qi)) /\ ~(eq2 #(Shared.index 284) i 1)) yi;
  rewrite each source_program_address buffers qi as qkv;
  rewrite each source_program_address buffers 1 as rotary;
  rewrite each source_program_address buffers yi as output;
  rewrite each (Ghost.reveal addresses).weights_address as weights;
  let observed = source_attention_body weights qkv rotary output offset
    #(Ghost.hide (Ghost.reveal input).weights)
    #(Ghost.hide #(values BF16.t 28672) (source_program_slot before qi))
    #(Ghost.hide #source_rotary_values (source_program_slot before 1))
    #(Ghost.hide #(values BF16.t 14336) (source_program_slot before yi));
  with result. assert PA.pts_to output result;
  source_program_attention_progress input before layer observed result;
  rewrite each qkv as source_program_address buffers qi;
  rewrite each rotary as source_program_address buffers 1;
  rewrite each output as source_program_address buffers yi;
  source_program_put buffers #before #1.0R
    (fun i -> (True /\ ~(eq2 #(Shared.index 284) i qi)) /\ ~(eq2 #(Shared.index 284) i 1)) yi #result;
  let after = Ghost.hide (source_program_replace before yi result);
  rewrite each source_program_replace before yi result as after;
  rewrite PA.pts_to (source_program_address buffers 1) (source_program_slot before 1) as
    PA.pts_to (source_program_address buffers 1) (source_program_slot after 1);
  source_program_restore buffers #after #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i qi)) 1;
  rewrite PA.pts_to (source_program_address buffers qi) (source_program_slot before qi) as
    PA.pts_to (source_program_address buffers qi) (source_program_slot after qi);
  source_program_restore buffers #after #1.0R (fun _ -> True) qi;
  source_program_storage_equation buffers after 1.0R;
  rewrite source_program_masked_storage buffers after 1.0R (fun _ -> True) as source_program_storage buffers after 1.0R;
  fold source_program_condition buffers input (4 + 8 * layer) 1.0R;
  rewrite each weights as (Ghost.reveal addresses).weights_address;
  fold source_program_core addresses buffers input (4 + 8 * layer) 1.0R;
}

inline_for_extraction noextract
fn source_program_silu_thread
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (#layer:erased (Shared.index 28))
  (gateup:PA.array BF16.t{gateup == source_program_address buffers (source_program_index layer 8)})
  (output:PA.array BF16.t{output == source_program_address buffers (source_program_index layer 9)})
  preserves gpu
  requires source_program_core addresses buffers input (7 + 8 * layer) 1.0R
  ensures source_program_core addresses buffers input (8 + 8 * layer) 1.0R
{
  unfold source_program_core addresses buffers input (7 + 8 * layer) 1.0R;
  let before = source_program_reveal buffers input (7 + 8 * layer) 1.0R;
  let xi = Ghost.hide (source_program_index layer 8);
  let yi = Ghost.hide (source_program_index layer 9);
  source_program_storage_equation buffers before 1.0R;
  rewrite source_program_storage buffers before 1.0R as source_program_masked_storage buffers before 1.0R (fun _ -> True);
  source_program_take buffers #before #1.0R (fun _ -> True) xi;
  source_program_take buffers #before #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i xi)) yi;
  rewrite each source_program_address buffers xi as gateup;
  rewrite each source_program_address buffers yi as output;
  source_silu_product_body gateup output
    #(Ghost.hide #(values BF16.t 43008) (source_program_slot before xi))
    #(Ghost.hide #(values BF16.t 21504) (source_program_slot before yi));
  with result. assert PA.pts_to output result;
  source_program_silu_progress input before layer result;
  rewrite each gateup as source_program_address buffers xi;
  rewrite each output as source_program_address buffers yi;
  source_program_put buffers #before #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i xi)) yi #result;
  let after = Ghost.hide (source_program_replace before yi result);
  rewrite each source_program_replace before yi result as after;
  rewrite PA.pts_to (source_program_address buffers xi) (source_program_slot before xi) as
    PA.pts_to (source_program_address buffers xi) (source_program_slot after xi);
  source_program_restore buffers #after #1.0R (fun _ -> True) xi;
  source_program_storage_equation buffers after 1.0R;
  rewrite source_program_masked_storage buffers after 1.0R (fun _ -> True) as source_program_storage buffers after 1.0R;
  fold source_program_condition buffers input (8 + 8 * layer) 1.0R;
  fold source_program_core addresses buffers input (8 + 8 * layer) 1.0R;
}

inline_for_extraction noextract
let source_program_addnorm_left (layer:Shared.index 28) (next:bool) : Shared.index 284 =
  source_program_index layer (if next then 6 else 0)

inline_for_extraction noextract
let source_program_addnorm_right (layer:Shared.index 28) (next:bool) : Shared.index 284 =
  source_program_index layer (if next then 10 else 5)

inline_for_extraction noextract
let source_program_addnorm_output (layer:Shared.index 28) (next:bool) : Shared.index 284 =
  source_program_index layer (if next then 11 else 6)

inline_for_extraction noextract
let source_program_addnorm_normalized (layer:Shared.index 28) (next:bool) : Shared.index 284 =
  source_program_index layer (if next then 12 else 7)

inline_for_extraction noextract
let source_program_addnorm_offset (layer:Shared.index 28) (next:bool{next ==> layer < 27}) : nat =
  layer_start layer + (if next then layer_weight_elements else 1024)

inline_for_extraction noextract
let source_program_addnorm_progress (input:source_inputs) (v:source_program_values)
  (layer:Shared.index 28) (next:bool{next ==> layer < 27})
  (combined normalized:source_hidden_values)
  : Lemma
      (requires source_program_stage input v ((5 + (if next then 4 else 0)) + 8 * layer) /\
        source_add_norm_result input.weights
          (source_program_slot v (source_program_addnorm_left layer next))
          (source_program_slot v (source_program_addnorm_right layer next))
          combined normalized (source_program_addnorm_offset layer next))
      (ensures source_program_stage input
        (source_program_replace
          (source_program_replace v (source_program_addnorm_output layer next) combined)
          (source_program_addnorm_normalized layer next) normalized)
        ((6 + (if next then 4 else 0)) + 8 * layer)) =
  if next then source_program_nextnorm_progress input v layer combined normalized
  else source_program_postnorm_progress input v layer combined normalized

inline_for_extraction noextract
fn source_program_addnorm_thread
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (#layer:erased (Shared.index 28)) (#next:erased bool{Ghost.reveal next == true ==> Ghost.reveal layer < 27})
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (left:PA.array BF16.t{left == source_program_address buffers (source_program_addnorm_left layer next)})
  (right:PA.array BF16.t{right == source_program_address buffers (source_program_addnorm_right layer next)})
  (combined:PA.array BF16.t{combined == source_program_address buffers (source_program_addnorm_output layer next)})
  (normalized:PA.array BF16.t{normalized == source_program_address buffers (source_program_addnorm_normalized layer next)})
  (offset:sz{DeviceSize.v offset == source_program_addnorm_offset layer next})
  preserves gpu
  requires source_program_core addresses buffers input ((5 + (if next then 4 else 0)) + 8 * layer) 1.0R
  ensures source_program_core addresses buffers input ((6 + (if next then 4 else 0)) + 8 * layer) 1.0R
{
  unfold source_program_core addresses buffers input ((5 + (if next then 4 else 0)) + 8 * layer) 1.0R;
  let before = source_program_reveal buffers input ((5 + (if next then 4 else 0)) + 8 * layer) 1.0R;
  let li = Ghost.hide (source_program_addnorm_left layer next);
  let ri = Ghost.hide (source_program_addnorm_right layer next);
  let ci = Ghost.hide (source_program_addnorm_output layer next);
  let ni = Ghost.hide (source_program_addnorm_normalized layer next);
  source_program_storage_equation buffers before 1.0R;
  rewrite source_program_storage buffers before 1.0R as source_program_masked_storage buffers before 1.0R (fun _ -> True);
  source_program_take buffers #before #1.0R (fun _ -> True) li;
  source_program_take buffers #before #1.0R
    (fun i -> True /\ ~(eq2 #(Shared.index 284) i li)) ri;
  source_program_take buffers #before #1.0R
    (fun i -> (True /\ ~(eq2 #(Shared.index 284) i li)) /\ ~(eq2 #(Shared.index 284) i ri)) ni;
  source_program_take buffers #before #1.0R
    (fun i -> ((True /\ ~(eq2 #(Shared.index 284) i li)) /\ ~(eq2 #(Shared.index 284) i ri)) /\
      ~(eq2 #(Shared.index 284) i ni)) ci;
  rewrite each source_program_address buffers li as left;
  rewrite each source_program_address buffers ri as right;
  rewrite each source_program_address buffers ci as combined;
  rewrite each source_program_address buffers ni as normalized;
  rewrite each (Ghost.reveal addresses).weights_address as weights;
  source_add_norm_body weights left right combined normalized offset
    #(Ghost.hide (Ghost.reveal input).weights)
    #(Ghost.hide #source_hidden_values (source_program_slot before li))
    #(Ghost.hide #source_hidden_values (source_program_slot before ri))
    #(Ghost.hide #source_hidden_values (source_program_slot before ci))
    #(Ghost.hide #source_hidden_values (source_program_slot before ni));
  with cv nv. assert PA.pts_to combined cv ** PA.pts_to normalized nv;
  source_program_addnorm_progress input before layer next cv nv;
  rewrite each left as source_program_address buffers li;
  rewrite each right as source_program_address buffers ri;
  rewrite each combined as source_program_address buffers ci;
  rewrite each normalized as source_program_address buffers ni;
  source_program_put buffers #before #1.0R
    (fun i -> ((True /\ ~(eq2 #(Shared.index 284) i li)) /\ ~(eq2 #(Shared.index 284) i ri)) /\
      ~(eq2 #(Shared.index 284) i ni)) ci #cv;
  let middle = Ghost.hide (source_program_replace before ci cv);
  rewrite each source_program_replace before ci cv as middle;
  source_program_put buffers #middle #1.0R
    (fun i -> (True /\ ~(eq2 #(Shared.index 284) i li)) /\ ~(eq2 #(Shared.index 284) i ri)) ni #nv;
  let after = Ghost.hide (source_program_replace middle ni nv);
  rewrite each source_program_replace middle ni nv as after;
  rewrite PA.pts_to (source_program_address buffers ri) (source_program_slot before ri) as
    PA.pts_to (source_program_address buffers ri) (source_program_slot after ri);
  source_program_restore buffers #after #1.0R
    (fun i -> True /\ ~(eq2 #(Shared.index 284) i li)) ri;
  rewrite PA.pts_to (source_program_address buffers li) (source_program_slot before li) as
    PA.pts_to (source_program_address buffers li) (source_program_slot after li);
  source_program_restore buffers #after #1.0R (fun _ -> True) li;
  source_program_storage_equation buffers after 1.0R;
  rewrite source_program_masked_storage buffers after 1.0R (fun _ -> True) as source_program_storage buffers after 1.0R;
  fold source_program_condition buffers input ((6 + (if next then 4 else 0)) + 8 * layer) 1.0R;
  rewrite each weights as (Ghost.reveal addresses).weights_address;
  fold source_program_core addresses buffers input ((6 + (if next then 4 else 0)) + 8 * layer) 1.0R;
}

inline_for_extraction noextract
let source_program_attention_descriptor
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (#layer:erased (Shared.index 28))
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (qkv:PA.array BF16.t{qkv == source_program_address buffers (source_program_index layer 3)})
  (rotary:PA.array BF16.t{rotary == source_program_address buffers 1})
  (output:PA.array BF16.t{output == source_program_address buffers (source_program_index layer 4)})
  (offset:sz{DeviceSize.v offset == layer_start layer + 6293504})
  : Kernel.kernel_desc
      (source_program_core addresses buffers input (3 + 8 * layer) 1.0R)
      (source_program_core addresses buffers input (4 + 8 * layer) 1.0R) =
  let descriptor:Kernel.kernel_desc_1_1
    (source_program_core addresses buffers input (3 + 8 * layer) 1.0R)
    (source_program_core addresses buffers input (4 + 8 * layer) 1.0R) = {
    f = (fun () -> source_program_attention_thread #buffers #addresses #input #layer weights qkv rotary output offset);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
let source_program_silu_descriptor
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (#layer:erased (Shared.index 28))
  (gateup:PA.array BF16.t{gateup == source_program_address buffers (source_program_index layer 8)})
  (output:PA.array BF16.t{output == source_program_address buffers (source_program_index layer 9)})
  : Kernel.kernel_desc
      (source_program_core addresses buffers input (7 + 8 * layer) 1.0R)
      (source_program_core addresses buffers input (8 + 8 * layer) 1.0R) =
  let descriptor:Kernel.kernel_desc_1_1
    (source_program_core addresses buffers input (7 + 8 * layer) 1.0R)
    (source_program_core addresses buffers input (8 + 8 * layer) 1.0R) = {
    f = (fun () -> source_program_silu_thread #buffers #addresses #input #layer gateup output);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
let source_program_addnorm_descriptor
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (#layer:erased (Shared.index 28)) (#next:erased bool{Ghost.reveal next == true ==> Ghost.reveal layer < 27})
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (left:PA.array BF16.t{left == source_program_address buffers (source_program_addnorm_left layer next)})
  (right:PA.array BF16.t{right == source_program_address buffers (source_program_addnorm_right layer next)})
  (combined:PA.array BF16.t{combined == source_program_address buffers (source_program_addnorm_output layer next)})
  (normalized:PA.array BF16.t{normalized == source_program_address buffers (source_program_addnorm_normalized layer next)})
  (offset:sz{DeviceSize.v offset == source_program_addnorm_offset layer next})
  : Kernel.kernel_desc
      (source_program_core addresses buffers input ((5 + (if next then 4 else 0)) + 8 * layer) 1.0R)
      (source_program_core addresses buffers input ((6 + (if next then 4 else 0)) + 8 * layer) 1.0R) =
  let descriptor:Kernel.kernel_desc_1_1
    (source_program_core addresses buffers input ((5 + (if next then 4 else 0)) + 8 * layer) 1.0R)
    (source_program_core addresses buffers input ((6 + (if next then 4 else 0)) + 8 * layer) 1.0R) = {
    f = (fun () -> source_program_addnorm_thread #buffers #addresses #input #layer #next
      weights left right combined normalized offset);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
let source_layer_weights_related (input:source_inputs) (layer:Shared.index 28) (wr:Shared.parameters real)
  : Lemma
      (requires Shared.parameters_approximates (source_parameters input layer) wr)
      (ensures
        (forall (d:Shared.index 1024).
          N.v_approximates (S.index input.weights (layer_start layer + d)) (wr.input_layernorm d) /\
          N.v_approximates (S.index input.weights (layer_start layer + 1024 + d)) (wr.post_attention_layernorm d)) /\
        (forall (r:Shared.index 4096) (d:Shared.index 1024).
          N.v_approximates (S.index input.weights (layer_start layer + 2048 + r * 1024 + d))
            (source_qkv_weight wr r d)) /\
        (forall (d:Shared.index 128).
          N.v_approximates (S.index input.weights (layer_start layer + 6293504 + d)) (wr.q_norm d) /\
          N.v_approximates (S.index input.weights (layer_start layer + 6293632 + d)) (wr.k_norm d)) /\
        (forall (r:Shared.index 1024) (d:Shared.index 2048).
          N.v_approximates (S.index input.weights (layer_start layer + 4196352 + r * 2048 + d)) (wr.o_proj r d)) /\
        (forall (r:Shared.index 6144) (d:Shared.index 1024).
          N.v_approximates (S.index input.weights (layer_start layer + 6293760 + r * 1024 + d))
            (source_gateup_weight wr r d)) /\
        (forall (r:Shared.index 1024) (d:Shared.index 3072).
          N.v_approximates (S.index input.weights (layer_start layer + 12585216 + r * 3072 + d))
            (wr.down_proj r d))) =
  source_spans layer

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
let source_layer_mlp_related (input:source_inputs) (layer:Shared.index 28) (v:source_block_values)
  (wr:Shared.parameters real) (residual:Shared.hidden real)
  : Lemma
      (requires source_layer_stage input layer v 8 /\
        Shared.parameters_approximates (source_parameters input layer) wr /\
        Shared.matrix_approximates (hidden_view v.residual0) residual)
      (ensures Shared.matrix_approximates (hidden_view v.hidden1)
        (Shared.residual_add residual
          (Shared.feed_forward wr (Shared.normalize_hidden wr.post_attention_layernorm residual)))) =
  source_layer_weights_related input layer wr;
  source_rms_result_related input.weights v.residual0 v.postnorm0
    (layer_start layer + 1024) wr.post_attention_layernorm residual;
  let mlp_input = Shared.normalize_hidden wr.post_attention_layernorm residual in
  source_linear_values_related 6144 1024 input.weights v.postnorm0 (layer_start layer + 6293760) 64 1
    (source_gateup_weight wr) mlp_input;
  let gate = Shared.linear_rows #7 #1024 #3072 wr.gate_proj mlp_input in
  let up = Shared.linear_rows #7 #1024 #3072 wr.up_proj mlp_input in
  let activated:Shared.matrix real 7 3072 = fun row column ->
    Shared.silu (Shared.linear_rows #7 #1024 #3072 wr.gate_proj mlp_input row column) *.
      Shared.linear_rows #7 #1024 #3072 wr.up_proj mlp_input row column in
  let activated_cell (row:Shared.index 7) (column:Shared.index 3072)
    : Lemma (N.v_approximates (S.index v.activated0 (row * 3072 + column)) (activated row column)) =
    source_gateup_projection_at wr mlp_input row column;
    source_gateup_projection_at wr mlp_input row (3072 + column);
    source_silu_cell_related v.gateup0 row column (gate row column) (up row column)
  in
  C.forall_intro_2 activated_cell;
  source_linear_values_related 1024 3072 input.weights v.activated0 (layer_start layer + 12585216)
    96 2 wr.down_proj activated;
  let down = Shared.linear_rows #7 #3072 #1024 wr.down_proj activated in
  assert (Shared.matrix_approximates (hidden_view v.down0) down);
  source_added_values_related v.residual0 v.down0 residual down;
  assert (Shared.matrix_approximates (hidden_view v.hidden1) (Shared.residual_add residual down));
  assert (Shared.feed_forward wr mlp_input == down) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.feed_forward]];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let source_layer_qkv_related (input:source_inputs) (layer:Shared.index 28) (v:source_block_values)
  (wr:Shared.parameters real) (normalized:Shared.hidden real)
  : Lemma
      (requires source_layer_stage input layer v 8 /\
        Shared.parameters_approximates (source_parameters input layer) wr /\
        Shared.matrix_approximates (hidden_view v.norm0) normalized)
      (ensures (
        let query = Shared.split_heads #real #16
          (Shared.linear_rows #7 #1024 #2048 wr.q_proj normalized) in
        let key = Shared.split_heads #real #8
          (Shared.linear_rows #7 #1024 #1024 wr.k_proj normalized) in
        let value = Shared.split_heads #real #8
          (Shared.linear_rows #7 #1024 #1024 wr.v_proj normalized) in
        (forall (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128).
          N.v_approximates (S.index v.qkv0 (row * 4096 + head * 128 + column))
            (source_join_qk query key row head column)) /\
        (forall (row:Shared.index 7) (head:Shared.index 8) (column:Shared.index 128).
          N.v_approximates (S.index v.qkv0 (row * 4096 + 3072 + head * 128 + column))
            (value row head column)) /\
        Shared.heads_approximates (source_v_load v.qkv0) value)) =
  source_layer_weights_related input layer wr;
  source_linear_values_related 4096 1024 input.weights v.norm0 (layer_start layer + 2048) 64 1
    (source_qkv_weight wr) normalized;
  let query = Shared.split_heads #real #16
    (Shared.linear_rows #7 #1024 #2048 wr.q_proj normalized) in
  let key = Shared.split_heads #real #8
    (Shared.linear_rows #7 #1024 #1024 wr.k_proj normalized) in
  let value = Shared.split_heads #real #8
    (Shared.linear_rows #7 #1024 #1024 wr.v_proj normalized) in
  let qkv_cell (row:Shared.index 7) (head:Shared.index 24) (column:Shared.index 128)
    : Lemma (N.v_approximates (S.index v.qkv0 (row * 4096 + head * 128 + column))
        (source_join_qk query key row head column)) =
    source_qkv_projection_at wr normalized row (head * 128 + column)
  in
  C.forall_intro_3 qkv_cell;
  let value_cell (row:Shared.index 7) (head:Shared.index 8) (column:Shared.index 128)
    : Lemma (
        N.v_approximates (S.index v.qkv0 (row * 4096 + 3072 + head * 128 + column)) (value row head column) /\
        N.v_approximates (source_v_load v.qkv0 row head column) (value row head column)) =
    source_qkv_projection_at wr normalized row (3072 + head * 128 + column);
    SourceCasts.fcast_approx #BF16.t #F32.t
      (S.index v.qkv0 (row * 4096 + 3072 + head * 128 + column)) (value row head column)
  in C.forall_intro_3 value_cell

inline_for_extraction noextract
let source_layer_related_loaded (input:source_inputs) (layer:Shared.index 28) (v:source_block_values)
  (qk:values BF16.t 21504) (wr:Shared.parameters real) (xr:Shared.hidden real)
  (dr:Shared.rotary_tables real)
  : Lemma
      (requires source_layer_stage input layer v 8 /\
        qk == source_qk_values input.weights v.qkv0 v.rotary0 (layer_start layer + 6293504) /\
        Shared.parameters_approximates (source_parameters input layer) wr /\
        source_loaded_hidden v.hidden0 xr /\ source_loaded_rotary v.rotary0 dr)
      (ensures Shared.layer_approximates (source_first_block_native_layer v qk)
        (Shared.decoder_layer wr dr xr)) =
  source_layer_weights_related input layer wr;
  let normalized = Shared.normalize_hidden wr.input_layernorm xr in
  source_rms_result_related input.weights v.hidden0 v.norm0 (layer_start layer) wr.input_layernorm xr;
  source_layer_qkv_related input layer v wr normalized;
  let query_input = Shared.split_heads #real #16
    (Shared.linear_rows #7 #1024 #2048 wr.q_proj normalized) in
  let key_input = Shared.split_heads #real #8
    (Shared.linear_rows #7 #1024 #1024 wr.k_proj normalized) in
  let value = Shared.split_heads #real #8
    (Shared.linear_rows #7 #1024 #1024 wr.v_proj normalized) in
  let qk_input = source_join_qk query_input key_input in
  source_qk_values_related input.weights v.qkv0 v.rotary0 qk (layer_start layer + 6293504)
    qk_input wr.q_norm wr.k_norm dr;
  source_qk_reference_view qk query_input key_input wr.q_norm wr.k_norm dr;
  let query = Shared.rotary #16 dr (Shared.normalize_heads #16 wr.q_norm query_input) in
  let key = Shared.rotary #8 dr (Shared.normalize_heads #8 wr.k_norm key_input) in
  source_first_block_keys_related v qk query key;
  assert (Shared.heads_approximates (source_first_block_native_layer v qk).values value);
  source_attention_to_reference input.weights v.qkv0 v.rotary0 qk v.attn0
    (layer_start layer + 6293504) query key value;
  let probabilities = Shared.attention_probabilities query (Shared.repeat_kv key) in
  let mixed = Shared.attention_values probabilities (Shared.repeat_kv value) in
  source_attention_concatenated v.attn0 mixed;
  let attention = Shared.concatenate_heads mixed in
  source_linear_values_related 1024 2048 input.weights v.attn0 (layer_start layer + 4196352)
    128 1 wr.o_proj attention;
  let projected = Shared.linear_rows #7 #2048 #1024 wr.o_proj attention in
  assert (Shared.matrix_approximates (hidden_view v.projected0) projected);
  source_added_values_related v.hidden0 v.projected0 xr projected;
  let residual = Shared.residual_add xr projected in
  assert (Shared.matrix_approximates (hidden_view v.residual0) residual);
  source_layer_mlp_related input layer v wr residual;
  let mlp_input = Shared.normalize_hidden wr.post_attention_layernorm residual in
  let expected:Shared.layer_values real = {
    hidden_after = Shared.residual_add residual (Shared.feed_forward wr mlp_input);
    keys = key;
    values = value
  } in
  assert (Shared.decoder_layer wr dr xr == expected) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.decoder_layer]];
     FStar.Tactics.V2.smt ())
#pop-options

inline_for_extraction noextract
let source_layer_observation_widen (v:source_block_values) (qk:values BF16.t 21504)
  (ideal:Shared.layer_values real)
  : Lemma
      (requires Shared.layer_approximates (source_first_block_native_layer v qk) ideal)
      (ensures Shared.layer_approximates (source_first_block_layer v qk) ideal) =
  let hidden (row:Shared.index 7) (column:Shared.index 1024)
    : Lemma (N.v_approximates ((source_first_block_layer v qk).hidden_after row column)
        (ideal.hidden_after row column)) =
    SourceCasts.fcast_approx #BF16.t #F32.t (S.index v.hidden1 (row * 1024 + column))
      (ideal.hidden_after row column)
  in
  let cache (row:Shared.index 7) (head:Shared.index 8) (column:Shared.index 128)
    : Lemma (
        N.v_approximates ((source_first_block_layer v qk).keys row head column) (ideal.keys row head column) /\
        N.v_approximates ((source_first_block_layer v qk).values row head column) (ideal.values row head column)) =
    SourceCasts.fcast_approx #BF16.t #F32.t (S.index qk (row * 3072 + 2048 + head * 128 + column))
      (ideal.keys row head column);
    SourceCasts.fcast_approx #BF16.t #F32.t (S.index v.qkv0 (row * 4096 + 3072 + head * 128 + column))
      (ideal.values row head column)
  in
  C.forall_intro_2 hidden;
  C.forall_intro_3 cache

inline_for_extraction noextract
let source_layer_numerical32 (input:source_inputs) (layer:Shared.index 28) (v:source_block_values)
  : Lemma
      (requires source_layer_stage input layer v 8)
      (ensures (
        let qk = source_qk_values input.weights v.qkv0 v.rotary0 (layer_start layer + 6293504) in
        Shared.decoder_relation (source_parameters input layer) (hidden_view v.hidden0)
          (source_observed_rotary v.rotary0) (source_first_block_layer v qk) /\
        Shared.decoder_relation (source_parameters input layer) (source_hidden32 v.hidden0)
          (source_observed_rotary v.rotary0) (source_first_block_layer v qk))) =
  let qk = source_qk_values input.weights v.qkv0 v.rotary0 (layer_start layer + 6293504) in
  let raw (wr:Shared.parameters real) (xr:Shared.hidden real) (dr:Shared.rotary_tables real)
    : Lemma
        (requires Shared.parameters_approximates (source_parameters input layer) wr /\
          Shared.matrix_approximates (hidden_view v.hidden0) xr /\
          Shared.rotary_approximates (source_observed_rotary v.rotary0) dr)
        (ensures Shared.layer_approximates (source_first_block_layer v qk) (Shared.decoder_layer wr dr xr)) =
    source_layer_related_loaded input layer v qk wr xr dr;
    source_layer_observation_widen v qk (Shared.decoder_layer wr dr xr)
  in
  C.forall_intro_3 (fun wr xr -> C.move_requires (raw wr xr));
  let loaded (wr:Shared.parameters real) (xr:Shared.hidden real) (dr:Shared.rotary_tables real)
    : Lemma
        (requires Shared.parameters_approximates (source_parameters input layer) wr /\
          Shared.matrix_approximates (source_hidden32 v.hidden0) xr /\
          Shared.rotary_approximates (source_observed_rotary v.rotary0) dr)
        (ensures Shared.layer_approximates (source_first_block_layer v qk) (Shared.decoder_layer wr dr xr)) =
    source_layer_related_loaded input layer v qk wr xr dr;
    source_layer_observation_widen v qk (Shared.decoder_layer wr dr xr)
  in C.forall_intro_3 (fun wr xr -> C.move_requires (loaded wr xr))

inline_for_extraction noextract
let source_program_decoder_edge (input:source_inputs) (v:source_program_values)
  (selected:Shared.token_id) (layer:Shared.index 28)
  : Lemma
      (requires source_program_stage input v 228)
      (ensures (
        let run = source_program_execution input v selected in
        Shared.decoder_relation (source_parameters input layer)
          (if layer = 0 then source_hidden32 (source_program_slot v 0)
           else (run.layer (layer - 1)).hidden_after)
          run.rope (run.layer layer))) =
  let values = source_program_layer v layer in
  source_layer_numerical32 input layer values;
  if layer > 0 then (
    assert (Shared.same_matrix
      (source_program_layer32 input v (layer - 1)).hidden_after (source_hidden32 values.hidden0));
    let related (wr:Shared.parameters real) (xr:Shared.hidden real) (dr:Shared.rotary_tables real)
      : Lemma
          (requires Shared.parameters_approximates (source_parameters input layer) wr /\
            Shared.matrix_approximates (source_program_layer32 input v (layer - 1)).hidden_after xr /\
            Shared.rotary_approximates (source_observed_rotary (source_program_slot v 1)) dr)
          (ensures Shared.layer_approximates (source_program_layer32 input v layer) (Shared.decoder_layer wr dr xr)) =
      assert (Shared.matrix_approximates (source_hidden32 values.hidden0) xr)
    in C.forall_intro_3 (fun wr xr -> C.move_requires (related wr xr)))

inline_for_extraction noextract
let source_program_final_normalization (input:source_inputs) (v:source_program_values)
  : Lemma
      (requires source_program_stage input v 228)
      (ensures Shared.final_norm_relation (input_mapping input).weight.final_norm
        (source_program_layer32 input v 27).hidden_after (hidden_view (source_program_slot v 282))) =
  let related (wr:Shared.vector real 1024) (xr:Shared.hidden real)
    : Lemma
        (requires Shared.vector_approximates (input_mapping input).weight.final_norm wr /\
          Shared.matrix_approximates (source_program_layer32 input v 27).hidden_after xr)
        (ensures Shared.matrix_approximates (hidden_view (source_program_slot v 282))
          (Shared.normalize_hidden wr xr)) =
    assert (source_loaded_hidden (source_program_slot v 281) xr);
    source_rms_result_related input.weights (source_program_slot v 281) (source_program_slot v 282)
      155582464 wr xr
  in C.forall_intro_2 (fun wr -> C.move_requires (related wr))

inline_for_extraction noextract
let source_program_projection_related (input:source_inputs) (v:source_program_values)
  (selected:Shared.token_id)
  : Lemma
      (requires source_program_stage input v 228)
      (ensures Shared.projection_relation (input_mapping input).weight.embedding
        (hidden_view (source_program_slot v 282)) (source_program_execution input v selected).scores) =
  let related (wr:Shared.matrix real 151936 1024) (xr:Shared.hidden real)
    : Lemma
        (requires Shared.matrix_approximates (input_mapping input).weight.embedding wr /\
          Shared.matrix_approximates (hidden_view (source_program_slot v 282)) xr)
        (ensures Shared.vector_approximates (source_program_execution input v selected).scores
          (Shared.linear #1024 #151936 wr (xr 6))) =
    source_linear_values_related 151936 1024 input.weights (source_program_slot v 282) 0 64 1 wr xr;
    let point (token:Shared.token_id)
      : Lemma (N.v_approximates ((source_program_execution input v selected).scores token)
          (Shared.linear #1024 #151936 wr (xr 6) token)) =
      SourceCasts.fcast_approx #BF16.t #F32.t
        (S.index (source_program_slot v 283) (6 * 151936 + token))
        (Shared.linear_rows #7 #1024 #151936 wr xr 6 token)
    in C.forall_intro point
  in C.forall_intro_2 (fun wr -> C.move_requires (related wr))

inline_for_extraction noextract
let source_program_numerical (input:source_inputs) (v:source_program_values)
  (selected:Shared.token_id)
  : Lemma
      (requires source_program_stage input v 228 /\ source_rotary_domain ())
      (ensures Shared.numerical_execution (input_mapping input) (source_program_execution input v selected)) =
  source_layer_numerical32 input 0 (source_program_layer v 0);
  let edge (layer:Shared.index 28{layer > 0})
    : Lemma (Shared.decoder_relation (source_parameters input layer)
        (source_program_layer32 input v (layer - 1)).hidden_after
        (source_observed_rotary (source_program_slot v 1)) (source_program_layer32 input v layer)) =
    source_program_decoder_edge input v selected layer
  in
  C.forall_intro edge;
  source_program_final_normalization input v;
  source_program_projection_related input v selected

inline_for_extraction noextract
let source_program_cache_execution (input:source_inputs) (v:source_program_values) (selected:Shared.token_id)
  : Lemma (Shared.cache_execution (source_program_execution input v selected)) =
  let step (layer:Shared.index 28) (other:Shared.index 28)
    : Lemma (Shared.same_cache_slot (source_program_cache input v (layer + 1) other)
        (Shared.update_cache (source_program_cache input v layer) layer
          (source_program_layer32 input v layer) other)) = ()
  in C.forall_intro_2 step

inline_for_extraction noextract
let source_program_history (input:source_inputs) (v:source_program_values)
  (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens})
  : Lemma (source_program_tokens input v tokens ==
      Shared.append_token (input_mapping input).prompt (I64.v (S.index tokens 7))) =
  let copied = source_program_tokens input v tokens in
  let expected = Shared.append_token (input_mapping input).prompt (I64.v (S.index tokens 7)) in
  assert (S.equal copied expected);
  S.lemma_eq_elim copied expected

inline_for_extraction noextract
let source_program_structural (#frame_t:Type0)
  (input:source_inputs) (v:source_program_values)
  (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens})
  (frame:frame_t)
  : Lemma
      (requires source_program_stage input v 228)
      (ensures Shared.structural_postcondition (input_mapping input) frame
        (source_program_execution input v (I64.v (S.index tokens 7)))
        (source_program_outputs input v tokens) (source_program_after_state input v tokens frame)) =
  source_program_history input v tokens;
  source_program_cache_execution input v (I64.v (S.index tokens 7));
  let embedding (p:Shared.index 7) (d:Shared.index 1024)
    : Lemma (S.index (source_program_slot v 0) (p * 1024 + d) ==
        (input_mapping input).weight.embedding (S.index (input_mapping input).prompt p) d) =
    assert (S.index (source_program_slot v 0) (p * 1024 + d) ==
      S.index (S.slice input.weights 0 embedding_elements) (I64.v (S.index input.prompt p) * 1024 + d))
  in C.forall_intro_2 embedding

inline_for_extraction noextract
let source_program_selector_maximal (input:source_inputs) (v:source_program_values)
  (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens})
  : Lemma
      (requires Shared.selector_domain
        (source_program_execution input v (I64.v (S.index tokens 7))).scores)
      (ensures Shared.maximal_actual_index
        (source_program_execution input v (I64.v (S.index tokens 7))).scores (I64.v (S.index tokens 7))) =
  let scores = selector_row_scores (source_program_slot v 283) in
  let selected:Shared.token_id = I64.v (S.index tokens 7) in
  let point (token:Shared.token_id)
    : Lemma (F.lte (S.index scores token) (S.index scores selected)) =
    F.negate_lt_is_lte (S.index scores selected) (S.index scores token)
  in C.forall_intro point

inline_for_extraction noextract
let source_program_semantic_result (#frame_t:Type0)
  (input:source_inputs) (v:source_program_values)
  (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens})
  (frame:frame_t)
  : Lemma
      (requires source_program_stage input v 228)
      (ensures source_program_result input v tokens frame) =
  let selected:Shared.token_id = I64.v (S.index tokens 7) in
  let run = source_program_execution input v selected in
  source_program_structural input v tokens frame;
  if t2b (source_rotary_domain ()) then source_program_numerical input v selected;
  if t2b (Shared.selector_domain run.scores) then source_program_selector_maximal input v tokens;
  if t2b (source_rotary_domain () /\ Shared.selector_domain run.scores) then
    Shared.numerical_execution_related (input_mapping input) run

inline_for_extraction noextract
let rec source_list_drop_step (#a:Type0) (items:list a) (index:nat{index < List.length items})
  : Lemma ((List.splitAt index items)._2 ==
      List.index items index :: (List.splitAt (index + 1) items)._2) (decreases index) =
  if index > 0 then source_list_drop_step (List.tail items) (index - 1)

inline_for_extraction noextract
let source_program_tail
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) (start:nat{start <= 284})
  : GTot (items:list (RT.tensor BF16.t){List.length items == 284 - start}) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  List.lemma_splitAt_snd_length start buffers.bf16_buffers;
  (List.splitAt start buffers.bf16_buffers)._2

inline_for_extraction noextract
type source_program_partial_values (start:nat{start <= 284}) =
  s:values (S.seq BF16.t) 284{
    forall (i:Shared.index 284{start <= i}). S.length (S.index s i) == source_program_size i}

inline_for_extraction noextract
let source_program_collected
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (start:nat{start <= 284}) (v:source_program_partial_values start) : PC.slprop =
  forall+ (i:Shared.index 284{start <= i}).
    PA.pts_to (source_program_address buffers i) (S.index v i)

inline_for_extraction noextract
let source_program_collected_equation
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (start:nat{start <= 284}) (v:source_program_partial_values start)
  : Lemma (source_program_collected buffers start v ==
      (forall+ (i:Shared.index 284{start <= i}).
        PA.pts_to (source_program_address buffers i) (S.index v i))) =
  assert (source_program_collected buffers start v ==
    (forall+ (i:Shared.index 284{start <= i}).
      PA.pts_to (source_program_address buffers i) (S.index v i))) by
    (FStar.Tactics.V2.norm [delta_only [`%source_program_collected]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
ghost
fn rec source_program_collect_remaining
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) (start:nat{start <= 284})
  preserves PC.loc KL.gpu_loc
  requires RuntimeHelpers.all_contents (source_program_tail buffers start)
  returns observed:erased (source_program_partial_values start)
  ensures source_program_collected buffers start observed
  decreases (284 - start)
{
  if (start = 284) {
    assert pure (source_program_tail buffers start == []);
    rewrite RuntimeHelpers.all_contents (source_program_tail buffers start) as
      RuntimeHelpers.all_contents #BF16.t [];
    unfold RuntimeHelpers.all_contents #BF16.t [];
    let values = Ghost.hide #(source_program_partial_values start)
      (S.create #(S.seq BF16.t) 284 (S.empty #BF16.t));
    source_program_collected_equation buffers start values;
    forevery_intro_empty (fun (i:Shared.index 284{start <= i}) ->
      PA.pts_to (source_program_address buffers i) (S.index values i));
    rewrite
      (forall+ (i:Shared.index 284{start <= i}).
        PA.pts_to (source_program_address buffers i) (S.index values i))
      as source_program_collected buffers start values;
    values
  } else {
    source_allocation_count ();
    source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
    source_list_drop_step buffers.bf16_buffers start;
    let handle = List.index buffers.bf16_buffers start;
    rewrite RuntimeHelpers.all_contents (source_program_tail buffers start) as
      RT.contents handle ** RuntimeHelpers.all_contents (source_program_tail buffers (start + 1));
    unfold RT.contents handle;
    with head_values. assert PC.on KL.gpu_loc (PA.pts_to (RT.data handle) head_values);
    PC.on_elim (PA.pts_to (RT.data handle) head_values);
    rewrite each RT.data handle as source_program_address buffers start;
    PA.pts_to_len (source_program_address buffers start);
    let tail_values = source_program_collect_remaining buffers (start + 1);
    let values = Ghost.hide #(source_program_partial_values start)
      (S.upd #(S.seq BF16.t) tail_values start head_values);
    unfold source_program_collected buffers (start + 1) tail_values;
    forevery_ext
      (fun (i:Shared.index 284{start + 1 <= i}) ->
        PA.pts_to (source_program_address buffers i) (S.index tail_values i))
      (fun (i:Shared.index 284{start + 1 <= i}) ->
        PA.pts_to (source_program_address buffers i) (S.index values i));
    rewrite PA.pts_to (source_program_address buffers start) head_values as
      PA.pts_to (source_program_address buffers start) (S.index values start);
    forevery_insert #(Shared.index 284) #(fun i -> start + 1 <= i)
      (fun i -> PA.pts_to (source_program_address buffers i) (S.index values i)) start;
    forevery_refine_ext #(Shared.index 284) #(fun i -> start + 1 <= i \/ eq2 #(Shared.index 284) start i)
      (fun i -> start <= i) (fun i -> PA.pts_to (source_program_address buffers i) (S.index values i));
    source_program_collected_equation buffers start values;
    fold source_program_collected buffers start values;
    values
  }
}

#push-options "--fuel 20 --ifuel 1"
inline_for_extraction noextract
let source_program_initial_values (first:source_block_values) (tail:source_program_partial_values 13)
  : GTot source_program_values =
  assert_norm (List.length (source_block_sequences first) == 13);
  let cell (i:Shared.index 13)
    : Lemma (S.length (List.index (source_block_sequences first) i) == source_program_size i) = ()
  in
  C.forall_intro cell;
  let prefix = source_block_sequences first in
  S.init_ghost 284 (fun i -> if i < 13 then List.index prefix i else S.index tail i)

inline_for_extraction noextract
let source_program_initial_layer_views (first:source_block_values) (tail:source_program_partial_values 13)
  : Lemma (
      source_program_layer (source_program_initial_values first tail) 0 == first /\
      (source_program_layer (source_program_initial_values first tail) 1).hidden0 == first.hidden1 /\
      (source_program_layer (source_program_initial_values first tail) 1).norm0 == first.norm1) = ()
#pop-options

inline_for_extraction noextract
let source_program_initial_progress (input:source_inputs) (first:source_block_values)
  (tail:source_program_partial_values 13)
  : Lemma
      (requires source_first_block_observation input first)
      (ensures source_program_stage input (source_program_initial_values first tail) 10) =
  let values = source_program_initial_values first tail in
  source_program_initial_layer_views first tail;
  assert (source_program_layer values 0 == first);
  assert (source_layer_stage input 0 first 8);
  assert ((source_program_layer values 1).hidden0 == first.hidden1 /\
    (source_program_layer values 1).norm0 == first.norm1);
  assert (source_layer_stage input 1 (source_program_layer values 1) 0)

inline_for_extraction noextract
ghost
fn source_program_insert_collected
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#v:source_program_values) (index:Shared.index 284)
  requires PA.pts_to (source_program_address buffers index) (source_program_slot v index) **
    source_program_collected buffers (index + 1) v
  ensures source_program_collected buffers index v
{
  unfold source_program_collected buffers (index + 1) v;
  forevery_insert #(Shared.index 284) #(fun i -> index + 1 <= i)
    (fun i -> PA.pts_to (source_program_address buffers i) (source_program_slot v i)) index;
  forevery_refine_ext #(Shared.index 284)
    #(fun i -> index + 1 <= i \/ eq2 #(Shared.index 284) index i)
    (fun i -> index <= i)
    (fun i -> PA.pts_to (source_program_address buffers i) (source_program_slot v i));
  source_program_collected_equation buffers index v;
  fold source_program_collected buffers index v;
}

inline_for_extraction noextract
ghost
fn source_program_join_first_block
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#first:source_block_values) (#tail:source_program_partial_values 13)
  requires source_block_storage (source_allocated_first_block buffers) first 1.0R **
    source_program_collected buffers 13 tail
  ensures source_program_storage buffers (source_program_initial_values first tail) 1.0R
{
  let values = Ghost.hide (source_program_initial_values first tail);
  source_program_initial_layer_views first tail;
  unfold source_program_collected buffers 13 tail;
  forevery_ext
    (fun (i:Shared.index 284{13 <= i}) -> PA.pts_to (source_program_address buffers i) (S.index tail i))
    (fun (i:Shared.index 284{13 <= i}) -> PA.pts_to (source_program_address buffers i) (S.index values i));
  fold source_program_collected buffers 13 values;
  unfold source_block_storage (source_allocated_first_block buffers) first 1.0R;
  source_program_insert_collected buffers #values 12;
  source_program_insert_collected buffers #values 11;
  source_program_insert_collected buffers #values 10;
  source_program_insert_collected buffers #values 9;
  source_program_insert_collected buffers #values 8;
  source_program_insert_collected buffers #values 7;
  source_program_insert_collected buffers #values 6;
  source_program_insert_collected buffers #values 5;
  source_program_insert_collected buffers #values 4;
  source_program_insert_collected buffers #values 3;
  source_program_insert_collected buffers #values 2;
  source_program_insert_collected buffers #values 1;
  source_program_insert_collected buffers #values 0;
  unfold source_program_collected buffers 0 values;
  forevery_unrefine (fun (i:Shared.index 284) -> PA.pts_to (source_program_address buffers i) (source_program_slot values i));
  fold source_program_storage buffers values 1.0R;
  rewrite source_program_storage buffers values 1.0R as
    source_program_storage buffers (source_program_initial_values first tail) 1.0R;
}

inline_for_extraction noextract
let source_after_program_stage (addresses:source_input_addresses) (input:source_inputs)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (calls:nat{2 <= calls /\ calls <= 228}) : PC.slprop =
  PC.on KL.gpu_loc (source_program_core addresses buffers input calls 1.0R **
    (PA.pts_to addresses.prompt_address input.prompt ** RT.contents buffers.token_buffer))

inline_for_extraction noextract
ghost
fn source_program_from_first_block (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) ()
  requires source_after_first_block addresses input buffers
  ensures source_after_program_stage addresses input buffers 10
{
  ghost_impersonate KL.gpu_loc (source_after_first_block addresses input buffers)
    (source_after_program_stage addresses input buffers 10)
    fn () {
      unfold source_after_first_block addresses input buffers;
      unfold source_remaining_block_storage buffers;
      PC.on_elim (
        PA.pts_to addresses.weights_address (Ghost.reveal input).weights **
        PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
        (exists* (v:source_block_values).
          source_block_storage (source_allocated_first_block buffers) v 1.0R **
          pure (source_first_block_observation input v)));
      with first. assert source_block_storage (source_allocated_first_block buffers) first 1.0R **
        pure (source_first_block_observation input first);
      rewrite RuntimeHelpers.all_contents (source_remaining_after_block buffers) as
        RuntimeHelpers.all_contents (source_program_tail buffers 13);
      let tail = source_program_collect_remaining buffers 13;
      source_program_initial_progress input first tail;
      source_program_join_first_block buffers #first #tail;
      fold source_program_condition buffers input 10 1.0R;
      fold source_program_core addresses buffers input 10 1.0R;
      PC.on_intro (source_program_core addresses buffers input 10 1.0R **
        (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** RT.contents buffers.token_buffer));
      fold source_after_program_stage addresses input buffers 10;
    };
}

inline_for_extraction noextract
fn source_program_data_ptr (stream:Stream.stream_t)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) (index:Shared.index 284)
  preserves KL.cpu ** source_allocated_owners stream buffers
  returns address:PA.array BF16.t
  ensures pure (address == source_program_address buffers index)
{
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  unfold source_allocated_owners stream buffers;
  let address = source_buffer_data_ptr stream buffers.bf16_buffers index;
  fold source_allocated_owners stream buffers;
  address
}

inline_for_extraction noextract
ghost
fn source_program_queue_step
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  (before:nat) (after:nat{after == before + 1}) (post:PC.slprop)
  requires Epoch.epoch_live stream (Epoch.epoch_next (start + before)) **
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next (start + before))) post
  ensures Epoch.epoch_live stream (start + after) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + after)) post
{
  rewrite Epoch.epoch_live stream (Epoch.epoch_next (start + before)) as
    Epoch.epoch_live stream (start + after);
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next (start + before))) post as
    Pledge.pledge0 (Epoch.epoch_done stream (start + after)) post;
}

inline_for_extraction noextract
fn source_enqueue_layer_seven (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) (layer:Shared.index 28)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    source_allocated_owners stream buffers
  requires Epoch.epoch_live stream (start + (2 + 8 * layer)) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + (2 + 8 * layer)))
      (source_after_program_stage addresses input buffers (2 + 8 * layer))
  ensures Epoch.epoch_live stream (start + (9 + 8 * layer)) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + (9 + 8 * layer)))
      (source_after_program_stage addresses input buffers (9 + 8 * layer))
{
  source_spans layer;
  let frame = Ghost.hide (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
    RT.contents buffers.token_buffer);
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + (2 + 8 * layer)))
    (source_after_program_stage addresses input buffers (2 + 8 * layer)) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + (2 + 8 * layer)))
      (PC.on KL.gpu_loc (source_program_core addresses buffers input (2 + 8 * layer) 1.0R ** frame));
  let norm = source_program_data_ptr stream buffers (source_program_index layer 2);
  let qkv = source_program_data_ptr stream buffers (source_program_index layer 3);
  let qkv_offset = DeviceSize.uint_to_t (layer_start layer + 2048);
  let qkv_desc = source_program_linear_descriptor #buffers #addresses #input #layer #0
    addresses.weights_address norm qkv 4096sz 1024sz qkv_offset 64sz 1sz;
  let qkv_framed = source_descriptor_with_frame qkv_desc frame;
  Kernel.launch_kernel_full qkv_framed stream #(start + (2 + 8 * layer));
  source_program_queue_step stream #start (2 + 8 * layer) (3 + 8 * layer)
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (3 + 8 * layer) 1.0R ** frame));

  let qkv = source_program_data_ptr stream buffers (source_program_index layer 3);
  let rotary = source_program_data_ptr stream buffers 1;
  let attn = source_program_data_ptr stream buffers (source_program_index layer 4);
  let qk_offset = DeviceSize.uint_to_t (layer_start layer + 6293504);
  let attention_desc = source_program_attention_descriptor #buffers #addresses #input #layer
    addresses.weights_address qkv rotary attn qk_offset;
  let attention_framed = source_descriptor_with_frame attention_desc frame;
  Kernel.launch_kernel_full attention_framed stream #(start + (3 + 8 * layer));
  source_program_queue_step stream #start (3 + 8 * layer) (4 + 8 * layer)
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (4 + 8 * layer) 1.0R ** frame));

  let attn = source_program_data_ptr stream buffers (source_program_index layer 4);
  let projected = source_program_data_ptr stream buffers (source_program_index layer 5);
  let projected_offset = DeviceSize.uint_to_t (layer_start layer + 4196352);
  let projected_desc = source_program_linear_descriptor #buffers #addresses #input #layer #1
    addresses.weights_address attn projected 1024sz 2048sz projected_offset 128sz 1sz;
  let projected_framed = source_descriptor_with_frame projected_desc frame;
  Kernel.launch_kernel_full projected_framed stream #(start + (4 + 8 * layer));
  source_program_queue_step stream #start (4 + 8 * layer) (5 + 8 * layer)
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (5 + 8 * layer) 1.0R ** frame));

  let hidden = source_program_data_ptr stream buffers (source_program_index layer 0);
  let projected = source_program_data_ptr stream buffers (source_program_index layer 5);
  let residual = source_program_data_ptr stream buffers (source_program_index layer 6);
  let postnorm = source_program_data_ptr stream buffers (source_program_index layer 7);
  let post_offset = DeviceSize.uint_to_t (layer_start layer + 1024);
  let post_desc = source_program_addnorm_descriptor #buffers #addresses #input #layer #false
    addresses.weights_address hidden projected residual postnorm post_offset;
  let post_framed = source_descriptor_with_frame post_desc frame;
  Kernel.launch_kernel_full post_framed stream #(start + (5 + 8 * layer));
  source_program_queue_step stream #start (5 + 8 * layer) (6 + 8 * layer)
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (6 + 8 * layer) 1.0R ** frame));

  let postnorm = source_program_data_ptr stream buffers (source_program_index layer 7);
  let gateup = source_program_data_ptr stream buffers (source_program_index layer 8);
  let gateup_offset = DeviceSize.uint_to_t (layer_start layer + 6293760);
  let gateup_desc = source_program_linear_descriptor #buffers #addresses #input #layer #2
    addresses.weights_address postnorm gateup 6144sz 1024sz gateup_offset 64sz 1sz;
  let gateup_framed = source_descriptor_with_frame gateup_desc frame;
  Kernel.launch_kernel_full gateup_framed stream #(start + (6 + 8 * layer));
  source_program_queue_step stream #start (6 + 8 * layer) (7 + 8 * layer)
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (7 + 8 * layer) 1.0R ** frame));

  let gateup = source_program_data_ptr stream buffers (source_program_index layer 8);
  let activated = source_program_data_ptr stream buffers (source_program_index layer 9);
  let silu_desc = source_program_silu_descriptor #buffers #addresses #input #layer gateup activated;
  let silu_framed = source_descriptor_with_frame silu_desc frame;
  Kernel.launch_kernel_full silu_framed stream #(start + (7 + 8 * layer));
  source_program_queue_step stream #start (7 + 8 * layer) (8 + 8 * layer)
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (8 + 8 * layer) 1.0R ** frame));

  let activated = source_program_data_ptr stream buffers (source_program_index layer 9);
  let down = source_program_data_ptr stream buffers (source_program_index layer 10);
  let down_offset = DeviceSize.uint_to_t (layer_start layer + 12585216);
  let down_desc = source_program_linear_descriptor #buffers #addresses #input #layer #3
    addresses.weights_address activated down 1024sz 3072sz down_offset 96sz 2sz;
  let down_framed = source_descriptor_with_frame down_desc frame;
  Kernel.launch_kernel_full down_framed stream #(start + (8 + 8 * layer));
  source_program_queue_step stream #start (8 + 8 * layer) (9 + 8 * layer)
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (9 + 8 * layer) 1.0R ** frame));
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + (9 + 8 * layer)))
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (9 + 8 * layer) 1.0R ** frame)) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + (9 + 8 * layer)))
      (source_after_program_stage addresses input buffers (9 + 8 * layer));
}

inline_for_extraction noextract
fn source_enqueue_layer_nextnorm (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) (layer:Shared.index 27)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    source_allocated_owners stream buffers
  requires Epoch.epoch_live stream (start + (9 + 8 * layer)) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + (9 + 8 * layer)))
      (source_after_program_stage addresses input buffers (9 + 8 * layer))
  ensures Epoch.epoch_live stream (start + (2 + 8 * (layer + 1))) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + (2 + 8 * (layer + 1))))
      (source_after_program_stage addresses input buffers (2 + 8 * (layer + 1)))
{
  let frame = Ghost.hide (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
    RT.contents buffers.token_buffer);
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + (9 + 8 * layer)))
    (source_after_program_stage addresses input buffers (9 + 8 * layer)) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + (9 + 8 * layer)))
      (PC.on KL.gpu_loc (source_program_core addresses buffers input (9 + 8 * layer) 1.0R ** frame));
  let residual = source_program_data_ptr stream buffers (source_program_index layer 6);
  let down = source_program_data_ptr stream buffers (source_program_index layer 10);
  let hidden = source_program_data_ptr stream buffers (source_program_index layer 11);
  let normalized = source_program_data_ptr stream buffers (source_program_index layer 12);
  let offset = DeviceSize.uint_to_t (layer_start layer + layer_weight_elements);
  let descriptor = source_program_addnorm_descriptor #buffers #addresses #input #layer #true
    addresses.weights_address residual down hidden normalized offset;
  let framed = source_descriptor_with_frame descriptor frame;
  Kernel.launch_kernel_full framed stream #(start + (9 + 8 * layer));
  source_program_queue_step stream #start (9 + 8 * layer) (10 + 8 * layer)
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (10 + 8 * layer) 1.0R ** frame));
  rewrite Epoch.epoch_live stream (start + (10 + 8 * layer)) as
    Epoch.epoch_live stream (start + (2 + 8 * (layer + 1)));
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + (10 + 8 * layer)))
    (PC.on KL.gpu_loc (source_program_core addresses buffers input (10 + 8 * layer) 1.0R ** frame)) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + (2 + 8 * (layer + 1))))
      (source_after_program_stage addresses input buffers (2 + 8 * (layer + 1)));
}

inline_for_extraction noextract
fn source_enqueue_remaining_layers (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    source_allocated_owners stream buffers
  requires Epoch.epoch_live stream (start + 10) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 10)) (source_after_first_block addresses input buffers)
  ensures Epoch.epoch_live stream (start + 225) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 225)) (source_after_program_stage addresses input buffers 225)
{
  Pledge.rewrite_pledge (source_after_first_block addresses input buffers)
    (source_after_program_stage addresses input buffers 10)
    (source_program_from_first_block addresses buffers #input);
  let mut layer:nat = 1;
  while (!layer < 27)
    invariant exists* (current:nat{1 <= current /\ current <= 27}).
      layer |-> current ** Epoch.epoch_live stream (start + (2 + 8 * current)) **
      Pledge.pledge0 (Epoch.epoch_done stream (start + (2 + 8 * current)))
        (source_after_program_stage addresses input buffers (2 + 8 * current))
    decreases (27 - !layer)
  {
    let current = !layer;
    source_enqueue_layer_seven addresses buffers #input current stream #start;
    source_enqueue_layer_nextnorm addresses buffers #input current stream #start;
    layer := current + 1;
  };
  with current. assert layer |-> current;
  assert pure (current == 27);
  rewrite Epoch.epoch_live stream (start + (2 + 8 * current)) as Epoch.epoch_live stream (start + 218);
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + (2 + 8 * current)))
    (source_after_program_stage addresses input buffers (2 + 8 * current)) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + 218))
      (source_after_program_stage addresses input buffers 218);
  source_enqueue_layer_seven addresses buffers #input 27 stream #start;
}

inline_for_extraction noextract
fn source_program_residual_thread
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (left:PA.array BF16.t{left == source_program_address buffers 276})
  (right:PA.array BF16.t{right == source_program_address buffers 280})
  (output:PA.array BF16.t{output == source_program_address buffers 281})
  preserves gpu
  requires source_program_core addresses buffers input 225 1.0R
  ensures source_program_core addresses buffers input 226 1.0R
{
  unfold source_program_core addresses buffers input 225 1.0R;
  let before = source_program_reveal buffers input 225 1.0R;
  source_program_storage_equation buffers before 1.0R;
  rewrite source_program_storage buffers before 1.0R as source_program_masked_storage buffers before 1.0R (fun _ -> True);
  source_program_take buffers #before #1.0R (fun _ -> True) 276;
  source_program_take buffers #before #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i 276)) 280;
  source_program_take buffers #before #1.0R
    (fun i -> (True /\ ~(eq2 #(Shared.index 284) i 276)) /\ ~(eq2 #(Shared.index 284) i 280)) 281;
  rewrite each source_program_address buffers 276 as left;
  rewrite each source_program_address buffers 280 as right;
  rewrite each source_program_address buffers 281 as output;
  source_residual_add_body left right output
    #(Ghost.hide #source_hidden_values (source_program_slot before 276))
    #(Ghost.hide #source_hidden_values (source_program_slot before 280))
    #(Ghost.hide #source_hidden_values (source_program_slot before 281));
  let result = Ghost.hide (source_added_values (source_program_slot before 276) (source_program_slot before 280));
  source_program_residual_progress input before;
  rewrite each left as source_program_address buffers 276;
  rewrite each right as source_program_address buffers 280;
  rewrite each output as source_program_address buffers 281;
  source_program_put buffers #before #1.0R
    (fun i -> (True /\ ~(eq2 #(Shared.index 284) i 276)) /\ ~(eq2 #(Shared.index 284) i 280)) 281 #result;
  let after = Ghost.hide (source_program_replace before 281 result);
  rewrite each source_program_replace before 281 result as after;
  rewrite PA.pts_to (source_program_address buffers 280) (source_program_slot before 280) as
    PA.pts_to (source_program_address buffers 280) (source_program_slot after 280);
  source_program_restore buffers #after #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i 276)) 280;
  rewrite PA.pts_to (source_program_address buffers 276) (source_program_slot before 276) as
    PA.pts_to (source_program_address buffers 276) (source_program_slot after 276);
  source_program_restore buffers #after #1.0R (fun _ -> True) 276;
  source_program_storage_equation buffers after 1.0R;
  rewrite source_program_masked_storage buffers after 1.0R (fun _ -> True) as source_program_storage buffers after 1.0R;
  fold source_program_condition buffers input 226 1.0R;
  fold source_program_core addresses buffers input 226 1.0R;
}

inline_for_extraction noextract
fn source_program_final_norm_thread
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (hidden:PA.array BF16.t{hidden == source_program_address buffers 281})
  (output:PA.array BF16.t{output == source_program_address buffers 282})
  preserves gpu
  requires source_program_core addresses buffers input 226 1.0R
  ensures source_program_core addresses buffers input 227 1.0R
{
  unfold source_program_core addresses buffers input 226 1.0R;
  let before = source_program_reveal buffers input 226 1.0R;
  source_program_storage_equation buffers before 1.0R;
  rewrite source_program_storage buffers before 1.0R as source_program_masked_storage buffers before 1.0R (fun _ -> True);
  source_program_take buffers #before #1.0R (fun _ -> True) 281;
  source_program_take buffers #before #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i 281)) 282;
  rewrite each source_program_address buffers 281 as hidden;
  rewrite each source_program_address buffers 282 as output;
  rewrite each (Ghost.reveal addresses).weights_address as weights;
  let offset = DeviceSize.uint32_to_sizet 155582464ul;
  fold source_rms_kernel_pre weights hidden output (Ghost.reveal input).weights (source_program_slot before 281);
  source_rmsnorm_body weights hidden output offset
    #(Ghost.hide (Ghost.reveal input).weights)
    #(Ghost.hide #source_hidden_values (source_program_slot before 281));
  rewrite source_rms_kernel_post weights hidden output (Ghost.reveal input).weights
    (source_program_slot before 281) (DeviceSize.v offset) as
    source_rms_kernel_post weights hidden output (Ghost.reveal input).weights
      (source_program_slot before 281) 155582464;
  unfold source_rms_kernel_post weights hidden output (Ghost.reveal input).weights
    (source_program_slot before 281) 155582464;
  with result. assert PA.pts_to output result **
    pure (source_rms_result (Ghost.reveal input).weights (source_program_slot before 281) result 155582464);
  source_program_final_norm_progress input before result;
  rewrite each hidden as source_program_address buffers 281;
  rewrite each output as source_program_address buffers 282;
  source_program_put buffers #before #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i 281)) 282 #result;
  let after = Ghost.hide (source_program_replace before 282 result);
  rewrite each source_program_replace before 282 result as after;
  rewrite PA.pts_to (source_program_address buffers 281) (source_program_slot before 281) as
    PA.pts_to (source_program_address buffers 281) (source_program_slot after 281);
  source_program_restore buffers #after #1.0R (fun _ -> True) 281;
  source_program_storage_equation buffers after 1.0R;
  rewrite source_program_masked_storage buffers after 1.0R (fun _ -> True) as source_program_storage buffers after 1.0R;
  fold source_program_condition buffers input 227 1.0R;
  rewrite each weights as (Ghost.reveal addresses).weights_address;
  fold source_program_core addresses buffers input 227 1.0R;
}

inline_for_extraction noextract
fn source_program_projection_thread
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (hidden:PA.array BF16.t{hidden == source_program_address buffers 282})
  (output:PA.array BF16.t{output == source_program_address buffers 283})
  (tid:szlt 32) ()
  norewrite
  requires gpu ** source_program_core addresses buffers input 227 (1.0R /. 32) ** thread_id 32 tid
  ensures gpu ** source_program_core addresses buffers input 228 (1.0R /. 32) ** thread_id 32 tid
{
  unfold source_program_core addresses buffers input 227 (1.0R /. 32);
  let before = source_program_reveal buffers input 227 (1.0R /. 32);
  source_program_storage_equation buffers before (1.0R /. 32);
  rewrite source_program_storage buffers before (1.0R /. 32) as
    source_program_masked_storage buffers before (1.0R /. 32) (fun _ -> True);
  source_program_take buffers #before #(1.0R /. 32) (fun _ -> True) 282;
  source_program_take buffers #before #(1.0R /. 32) (fun i -> True /\ ~(eq2 #(Shared.index 284) i 282)) 283;
  rewrite each source_program_address buffers 282 as hidden;
  rewrite each source_program_address buffers 283 as output;
  rewrite each (Ghost.reveal addresses).weights_address as weights;
  let columns = DeviceSize.uint32_to_sizet 151936ul;
  let xv = Ghost.hide #source_hidden_values (source_program_slot before 282);
  let old = Ghost.hide #(values BF16.t (DeviceSize.v columns * 7)) (source_program_slot before 283);
  rewrite PA.pts_to hidden #(1.0R /. 32) (source_program_slot before 282) as PA.pts_to hidden #(1.0R /. 32) xv;
  rewrite PA.pts_to output #(1.0R /. 32) (source_program_slot before 283) as PA.pts_to output #(1.0R /. 32) old;
  source_mma_arrays_linear_arrays columns 1024sz #weight_elements weights hidden output 0sz 64sz 1sz
    #(1.0R /. 32) #(1.0R /. 32) #(Ghost.hide (Ghost.reveal input).weights) #xv #old;
  source_linear_old_independent #columns #1024 #weight_elements (Ghost.reveal input).weights xv old
    (S.create (columns * 7) (F.zero #BF16.t)) 0 64 1;
  let result = Ghost.hide (source_linear_values 151936 1024 (Ghost.reveal input).weights xv 0 64 1);
  rewrite PA.pts_to output #(1.0R /. 32)
    (source_mma_arrays_array_result #columns #1024 #weight_elements (Ghost.reveal input).weights xv old 0 64 1) as
    PA.pts_to output #(1.0R /. 32) result;
  source_program_projection_progress input before;
  rewrite each hidden as source_program_address buffers 282;
  rewrite each output as source_program_address buffers 283;
  source_program_put buffers #before #(1.0R /. 32) (fun i -> True /\ ~(eq2 #(Shared.index 284) i 282)) 283 #result;
  let after = Ghost.hide (source_program_replace before 283 result);
  rewrite each source_program_replace before 283 result as after;
  rewrite PA.pts_to (source_program_address buffers 282) #(1.0R /. 32) xv as
    PA.pts_to (source_program_address buffers 282) #(1.0R /. 32) (source_program_slot after 282);
  source_program_restore buffers #after #(1.0R /. 32) (fun _ -> True) 282;
  source_program_storage_equation buffers after (1.0R /. 32);
  rewrite source_program_masked_storage buffers after (1.0R /. 32) (fun _ -> True) as
    source_program_storage buffers after (1.0R /. 32);
  fold source_program_condition buffers input 228 (1.0R /. 32);
  rewrite each weights as (Ghost.reveal addresses).weights_address;
  fold source_program_core addresses buffers input 228 (1.0R /. 32);
}

inline_for_extraction noextract
let source_program_residual_descriptor
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (left:PA.array BF16.t{left == source_program_address buffers 276})
  (right:PA.array BF16.t{right == source_program_address buffers 280})
  (output:PA.array BF16.t{output == source_program_address buffers 281})
  : Kernel.kernel_desc (source_program_core addresses buffers input 225 1.0R)
      (source_program_core addresses buffers input 226 1.0R) =
  let descriptor:Kernel.kernel_desc_1_1 (source_program_core addresses buffers input 225 1.0R)
    (source_program_core addresses buffers input 226 1.0R) = {
    f = (fun () -> source_program_residual_thread #buffers #addresses #input left right output);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
let source_program_final_norm_descriptor
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (hidden:PA.array BF16.t{hidden == source_program_address buffers 281})
  (output:PA.array BF16.t{output == source_program_address buffers 282})
  : Kernel.kernel_desc (source_program_core addresses buffers input 226 1.0R)
      (source_program_core addresses buffers input 227 1.0R) =
  let descriptor:Kernel.kernel_desc_1_1 (source_program_core addresses buffers input 226 1.0R)
    (source_program_core addresses buffers input 227 1.0R) = {
    f = (fun () -> source_program_final_norm_thread #buffers #addresses #input weights hidden output);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
let source_program_projection_descriptor
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (weights:packed_weights_address{weights == (Ghost.reveal addresses).weights_address})
  (hidden:PA.array BF16.t{hidden == source_program_address buffers 282})
  (output:PA.array BF16.t{output == source_program_address buffers 283})
  : Kernel.kernel_desc (source_program_core addresses buffers input 227 1.0R)
      (source_program_core addresses buffers input 228 1.0R) =
  let descriptor:Kernel.kernel_desc_1_n (source_program_core addresses buffers input 227 1.0R)
    (source_program_core addresses buffers input 228 1.0R) = {
    nthr = 32sz;
    frame = emp;
    kpre = (fun _ -> source_program_core addresses buffers input 227 (1.0R /. 32));
    kpost = (fun _ -> source_program_core addresses buffers input 228 (1.0R /. 32));
    f = source_program_projection_thread #buffers #addresses #input weights hidden output;
    block_setup = source_program_core_distribute addresses buffers input 227;
    block_teardown = source_program_core_reassemble addresses buffers input 228;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve
  } in Kernel.k1n_as_kfull descriptor

inline_for_extraction noextract
let source_program_selector_pre (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) (input:source_inputs) : PC.slprop =
  source_program_core addresses buffers input 228 1.0R **
  PA.pts_to addresses.prompt_address input.prompt **
  (exists* (tokens:values I64.t 8). PA.pts_to (RT.data buffers.token_buffer) tokens)

inline_for_extraction noextract
let source_program_terminal (#frame_t:Type0) (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) (input:source_inputs)
  (frame:frame_t) : PC.slprop =
  PA.pts_to addresses.weights_address input.weights **
  PA.pts_to addresses.prompt_address input.prompt **
  (exists* (v:source_program_values)
    (tokens:values I64.t 8{source_selection_result (source_program_slot v 283) input.prompt tokens}).
    source_program_storage buffers v 1.0R ** PA.pts_to (RT.data buffers.token_buffer) tokens **
    pure (source_program_stage input v 228 /\ source_program_result input v tokens frame))

inline_for_extraction noextract
fn source_program_selector_thread (#frame_t:Type0) (#frame:erased frame_t)
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (logits:PA.array BF16.t{logits == source_program_address buffers 283})
  (prompt:PA.array I64.t{prompt == (Ghost.reveal addresses).prompt_address})
  (output:PA.array I64.t{output == RT.data (Ghost.reveal buffers).token_buffer})
  preserves gpu
  requires source_program_selector_pre addresses buffers input
  ensures source_program_terminal addresses buffers input frame
{
  unfold source_program_selector_pre addresses buffers input;
  unfold source_program_core addresses buffers input 228 1.0R;
  let values = source_program_reveal buffers input 228 1.0R;
  source_program_storage_equation buffers values 1.0R;
  rewrite source_program_storage buffers values 1.0R as source_program_masked_storage buffers values 1.0R (fun _ -> True);
  source_program_take buffers #values #1.0R (fun _ -> True) 283;
  with old. assert PA.pts_to (RT.data (Ghost.reveal buffers).token_buffer) old;
  rewrite each source_program_address buffers 283 as logits;
  rewrite each (Ghost.reveal addresses).prompt_address as prompt;
  rewrite each RT.data (Ghost.reveal buffers).token_buffer as output;
  let old_values = Ghost.hide #(S.seq I64.t) old;
  source_select_token_body logits prompt output
    #(Ghost.hide #(s:S.seq BF16.t{S.length s == 7 * 151936}) (source_program_slot values 283))
    #(Ghost.hide #(s:S.seq I64.t{S.length s == 7}) (Ghost.reveal input).prompt) #old_values;
  with tokens. assert PA.pts_to output tokens **
    pure (source_selection_result (source_program_slot values 283) (Ghost.reveal input).prompt tokens);
  source_program_semantic_result input values tokens frame;
  rewrite each logits as source_program_address buffers 283;
  rewrite each prompt as (Ghost.reveal addresses).prompt_address;
  rewrite each output as RT.data (Ghost.reveal buffers).token_buffer;
  source_program_restore buffers #values #1.0R (fun _ -> True) 283;
  source_program_storage_equation buffers values 1.0R;
  rewrite source_program_masked_storage buffers values 1.0R (fun _ -> True) as source_program_storage buffers values 1.0R;
  fold source_program_terminal addresses buffers input frame;
}

inline_for_extraction noextract
instance source_program_selector_pre_send (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) (input:source_inputs)
  : is_send_across KL.gpu_of (source_program_selector_pre addresses buffers input) = solve

inline_for_extraction noextract
instance source_program_terminal_send (#frame_t:Type0) (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers}) (input:source_inputs) (frame:frame_t)
  : is_send_across KL.gpu_of (source_program_terminal addresses buffers input frame) = solve

inline_for_extraction noextract
let source_program_selector_descriptor (#frame_t:Type0) (#frame:erased frame_t)
  (#buffers:erased (b:source_allocated_buffers{source_allocation_metadata b}))
  (#addresses:erased source_input_addresses) (#input:erased source_inputs)
  (logits:PA.array BF16.t{logits == source_program_address buffers 283})
  (prompt:PA.array I64.t{prompt == (Ghost.reveal addresses).prompt_address})
  (output:PA.array I64.t{output == RT.data (Ghost.reveal buffers).token_buffer})
  : Kernel.kernel_desc (source_program_selector_pre addresses buffers input)
      (source_program_terminal addresses buffers input frame) =
  let descriptor:Kernel.kernel_desc_1_1 (source_program_selector_pre addresses buffers input)
    (source_program_terminal addresses buffers input frame) = {
    f = (fun () -> source_program_selector_thread #frame_t #frame #buffers #addresses #input logits prompt output);
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in Kernel.k11_as_kfull descriptor

inline_for_extraction noextract
ghost
fn source_program_prepare_selector (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) ()
  requires source_after_program_stage addresses input buffers 228
  ensures PC.on KL.gpu_loc (source_program_selector_pre addresses buffers input)
{
  ghost_impersonate KL.gpu_loc (source_after_program_stage addresses input buffers 228)
    (PC.on KL.gpu_loc (source_program_selector_pre addresses buffers input))
    fn () {
      unfold source_after_program_stage addresses input buffers 228;
      PC.on_elim (source_program_core addresses buffers input 228 1.0R **
        (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt ** RT.contents buffers.token_buffer));
      unfold RT.contents buffers.token_buffer;
      with old. assert PC.on KL.gpu_loc (PA.pts_to (RT.data buffers.token_buffer) old);
      PC.on_elim (PA.pts_to (RT.data buffers.token_buffer) old);
      PA.pts_to_len (RT.data buffers.token_buffer);
      fold source_program_selector_pre addresses buffers input;
      PC.on_intro (source_program_selector_pre addresses buffers input);
    };
}

inline_for_extraction noextract
fn source_enqueue_program_suffix (#frame_t:Type0) (#logical_frame:erased frame_t)
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    source_allocated_owners stream buffers
  requires Epoch.epoch_live stream (start + 225) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 225)) (source_after_program_stage addresses input buffers 225)
  ensures Epoch.epoch_live stream (start + 229) **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 229))
      (PC.on KL.gpu_loc (source_program_terminal addresses buffers input logical_frame))
{
  let frame = Ghost.hide (PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
    RT.contents buffers.token_buffer);
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + 225))
    (source_after_program_stage addresses input buffers 225) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + 225))
      (PC.on KL.gpu_loc (source_program_core addresses buffers input 225 1.0R ** frame));
  let residual = source_program_data_ptr stream buffers 276;
  let down = source_program_data_ptr stream buffers 280;
  let hidden = source_program_data_ptr stream buffers 281;
  let residual_desc = source_program_residual_descriptor #buffers #addresses #input residual down hidden;
  let residual_framed = source_descriptor_with_frame residual_desc frame;
  Kernel.launch_kernel_full residual_framed stream #(start + 225);
  source_queue_step_label stream #start 225
    (PC.on KL.gpu_loc (source_program_core addresses buffers input 226 1.0R ** frame));

  let hidden = source_program_data_ptr stream buffers 281;
  let normalized = source_program_data_ptr stream buffers 282;
  let norm_desc = source_program_final_norm_descriptor #buffers #addresses #input
    addresses.weights_address hidden normalized;
  let norm_framed = source_descriptor_with_frame norm_desc frame;
  Kernel.launch_kernel_full norm_framed stream #(start + 226);
  source_queue_step_label stream #start 226
    (PC.on KL.gpu_loc (source_program_core addresses buffers input 227 1.0R ** frame));

  let normalized = source_program_data_ptr stream buffers 282;
  let logits = source_program_data_ptr stream buffers 283;
  let projection_desc = source_program_projection_descriptor #buffers #addresses #input
    addresses.weights_address normalized logits;
  let projection_framed = source_descriptor_with_frame projection_desc frame;
  Kernel.launch_kernel_full projection_framed stream #(start + 227);
  source_queue_step_label stream #start 227
    (PC.on KL.gpu_loc (source_program_core addresses buffers input 228 1.0R ** frame));
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + 228))
    (PC.on KL.gpu_loc (source_program_core addresses buffers input 228 1.0R ** frame)) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + 228))
      (source_after_program_stage addresses input buffers 228);
  Pledge.rewrite_pledge (source_after_program_stage addresses input buffers 228)
    (PC.on KL.gpu_loc (source_program_selector_pre addresses buffers input))
    (source_program_prepare_selector addresses buffers #input);
  let logits = source_program_data_ptr stream buffers 283;
  unfold source_allocated_owners stream buffers;
  let output = RT.data_ptr stream buffers.token_buffer;
  fold source_allocated_owners stream buffers;
  let selector = source_program_selector_descriptor #frame_t #logical_frame #buffers #addresses #input
    logits addresses.prompt_address output;
  Kernel.launch_kernel_full selector stream #(start + 228);
  source_queue_step_label stream #start 228
    (PC.on KL.gpu_loc (source_program_terminal addresses buffers input logical_frame));
}

inline_for_extraction noextract
let rec source_list_split (#a:Type0) (items:list a) (count:nat{count <= List.length items})
  : Lemma (
      List.length (List.splitAt count items)._1 == count /\
      List.length (List.splitAt count items)._2 == List.length items - count /\
      List.append (List.splitAt count items)._1 (List.splitAt count items)._2 == items)
      (decreases count) =
  if count > 0 then source_list_split (List.tail items) (count - 1)

inline_for_extraction noextract
let rec source_list_prefix_step (#a:Type0) (items:list a)
  (count:pos{count <= List.length items})
  : Lemma ((List.splitAt count items)._1 ==
      List.append (List.splitAt (count - 1) items)._1 [List.index items (count - 1)])
      (decreases count) =
  if count > 1 then source_list_prefix_step (List.tail items) (count - 1)

inline_for_extraction noextract
let source_program_returned
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Tot (output:source_returned_buffers{source_returned_metadata output}) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  source_matching_allocation_at source_bf16_allocation_shapes buffers.bf16_buffers 282;
  source_layer_allocation_count 28;
  source_list_index_append [[7; 1024]; [2; 7; 128]]
    (List.append (source_layer_allocations 28) [[7; 1024]; [7; 151936]]) 282;
  source_list_index_append (source_layer_allocations 28) [[7; 1024]; [7; 151936]] 280;
  {
    returned_tokens = buffers.token_buffer;
    returned_hidden = List.index buffers.bf16_buffers 282
  }

inline_for_extraction noextract
let source_program_return_addresses
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Lemma (
      source_program_address buffers 282 == RT.data (source_program_returned buffers).returned_hidden /\
      RT.data buffers.token_buffer == RT.data (source_program_returned buffers).returned_tokens /\
      RT.metadata (source_program_returned buffers).returned_hidden [7; 1024] /\
      RT.metadata (source_program_returned buffers).returned_tokens [1; 8]) =
  assert (source_returned_metadata (source_program_returned buffers))

inline_for_extraction noextract
let source_program_retirement
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Tot (list (RT.tensor BF16.t)) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  List.index buffers.bf16_buffers 283 :: List.rev' (List.splitAt 282 buffers.bf16_buffers)._1

inline_for_extraction noextract
let source_program_retirement_binding
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  : Lemma (
      List.length buffers.bf16_buffers == 284 /\
      List.length (List.splitAt 282 buffers.bf16_buffers)._1 == 282 /\
      buffers.bf16_buffers == List.append (List.splitAt 282 buffers.bf16_buffers)._1
        [(source_program_returned buffers).returned_hidden; List.index buffers.bf16_buffers 283] /\
      List.length (source_program_retirement buffers) == 283) =
  source_allocation_count ();
  source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
  source_list_split buffers.bf16_buffers 282;
  let left, right = List.splitAt 282 buffers.bf16_buffers in
  source_list_index_append left right 282;
  source_list_index_append left right 283;
  assert (right == [List.index buffers.bf16_buffers 282; List.index buffers.bf16_buffers 283]);
  ListProps.rev_length left;
  ListProps.rev_rev' left

inline_for_extraction noextract
ghost
fn rec source_owners_append (stream:Stream.stream_t) (left right:list (RT.tensor BF16.t))
  requires RuntimeHelpers.all_owners stream left ** RuntimeHelpers.all_owners stream right
  ensures RuntimeHelpers.all_owners stream (List.append left right)
  decreases left
{
  match left {
    [] -> {
      unfold RuntimeHelpers.all_owners #BF16.t stream [];
      rewrite RuntimeHelpers.all_owners stream right as RuntimeHelpers.all_owners stream (List.append left right);
    }
    head::tail -> {
      unfold RuntimeHelpers.all_owners stream left;
      source_owners_append stream tail right;
      fold RuntimeHelpers.all_owners stream (head :: List.append tail right);
      rewrite RuntimeHelpers.all_owners stream (head :: List.append tail right) as
        RuntimeHelpers.all_owners stream (List.append left right);
    }
  }
}

inline_for_extraction noextract
ghost
fn rec source_owners_split (stream:Stream.stream_t) (left right:list (RT.tensor BF16.t))
  requires RuntimeHelpers.all_owners stream (List.append left right)
  ensures RuntimeHelpers.all_owners stream left ** RuntimeHelpers.all_owners stream right
  decreases left
{
  match left {
    [] -> {
      rewrite RuntimeHelpers.all_owners stream (List.append left right) as RuntimeHelpers.all_owners stream right;
      fold RuntimeHelpers.all_owners #BF16.t stream [];
    }
    head::tail -> {
      rewrite RuntimeHelpers.all_owners stream (List.append left right) as
        RuntimeHelpers.all_owners stream (head :: List.append tail right);
      unfold RuntimeHelpers.all_owners stream (head :: List.append tail right);
      source_owners_split stream tail right;
      fold RuntimeHelpers.all_owners stream left;
    }
  }
}

inline_for_extraction noextract
ghost
fn rec source_owners_reverse (stream:Stream.stream_t) (items:list (RT.tensor BF16.t))
  requires RuntimeHelpers.all_owners stream items
  ensures RuntimeHelpers.all_owners stream (List.rev' items)
  decreases items
{
  match items {
    [] -> { rewrite RuntimeHelpers.all_owners stream items as RuntimeHelpers.all_owners stream (List.rev' items); }
    head::tail -> {
      unfold RuntimeHelpers.all_owners stream items;
      source_owners_reverse stream tail;
      fold RuntimeHelpers.all_owners #BF16.t stream [];
      fold RuntimeHelpers.all_owners stream [head];
      source_owners_append stream (List.rev' tail) [head];
      rewrite RuntimeHelpers.all_owners stream (List.append (List.rev' tail) [head]) as
        RuntimeHelpers.all_owners stream (List.rev' items);
    }
  }
}

inline_for_extraction noextract
ghost
fn rec source_contents_append (left right:list (RT.tensor BF16.t))
  requires RuntimeHelpers.all_contents left ** RuntimeHelpers.all_contents right
  ensures RuntimeHelpers.all_contents (List.append left right)
  decreases left
{
  match left {
    [] -> {
      unfold RuntimeHelpers.all_contents #BF16.t [];
      rewrite RuntimeHelpers.all_contents right as RuntimeHelpers.all_contents (List.append left right);
    }
    head::tail -> {
      unfold RuntimeHelpers.all_contents left;
      source_contents_append tail right;
      fold RuntimeHelpers.all_contents (head :: List.append tail right);
      rewrite RuntimeHelpers.all_contents (head :: List.append tail right) as
        RuntimeHelpers.all_contents (List.append left right);
    }
  }
}

inline_for_extraction noextract
ghost
fn rec source_contents_reverse (items:list (RT.tensor BF16.t))
  requires RuntimeHelpers.all_contents items
  ensures RuntimeHelpers.all_contents (List.rev' items)
  decreases items
{
  match items {
    [] -> { rewrite RuntimeHelpers.all_contents items as RuntimeHelpers.all_contents (List.rev' items); }
    head::tail -> {
      unfold RuntimeHelpers.all_contents items;
      source_contents_reverse tail;
      fold RuntimeHelpers.all_contents #BF16.t [];
      fold RuntimeHelpers.all_contents [head];
      source_contents_append (List.rev' tail) [head];
      rewrite RuntimeHelpers.all_contents (List.append (List.rev' tail) [head]) as
        RuntimeHelpers.all_contents (List.rev' items);
    }
  }
}

inline_for_extraction noextract
ghost
fn source_program_retirement_owners (stream:Stream.stream_t)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  requires source_allocated_owners stream buffers
  ensures RuntimeHelpers.all_owners stream (source_program_retirement buffers) **
    RT.owner stream (source_program_returned buffers).returned_tokens **
    RT.owner stream (source_program_returned buffers).returned_hidden
{
  source_program_retirement_binding buffers;
  unfold source_allocated_owners stream buffers;
  let prefix = (List.splitAt 282 buffers.bf16_buffers)._1;
  let logits = List.index buffers.bf16_buffers 283;
  let hidden = (source_program_returned buffers).returned_hidden;
  rewrite RuntimeHelpers.all_owners stream buffers.bf16_buffers as
    RuntimeHelpers.all_owners stream (List.append prefix [hidden; logits]);
  source_owners_split stream prefix [hidden; logits];
  unfold RuntimeHelpers.all_owners stream [hidden; logits];
  unfold RuntimeHelpers.all_owners stream [logits];
  unfold RuntimeHelpers.all_owners #BF16.t stream [];
  source_owners_reverse stream prefix;
  fold RuntimeHelpers.all_owners stream (logits :: List.rev' prefix);
  rewrite RuntimeHelpers.all_owners stream (logits :: List.rev' prefix) as
    RuntimeHelpers.all_owners stream (source_program_retirement buffers);
  rewrite RT.owner stream buffers.token_buffer as RT.owner stream (source_program_returned buffers).returned_tokens;
  rewrite RT.owner stream hidden as RT.owner stream (source_program_returned buffers).returned_hidden;
}

inline_for_extraction noextract
ghost
fn rec source_program_prefix_to_runtime
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#v:source_program_values) (count:nat{count <= 282})
  preserves PC.loc KL.gpu_loc
  requires forall+ (i:Shared.index 284{i < count}).
    PA.pts_to (source_program_address buffers i) (source_program_slot v i)
  ensures RuntimeHelpers.all_contents (List.splitAt count buffers.bf16_buffers)._1
  decreases count
{
  if (count = 0) {
    forevery_elim_empty (fun (i:Shared.index 284{i < count}) ->
      PA.pts_to (source_program_address buffers i) (source_program_slot v i));
    fold RuntimeHelpers.all_contents #BF16.t [];
    rewrite RuntimeHelpers.all_contents #BF16.t [] as RuntimeHelpers.all_contents (List.splitAt count buffers.bf16_buffers)._1;
  } else {
    let last = count - 1;
    forevery_remove' #(Shared.index 284) (fun i -> i < count)
      (fun i -> PA.pts_to (source_program_address buffers i) (source_program_slot v i)) last;
    forevery_refine_ext #(Shared.index 284) #(fun i -> i < count /\ ~(eq2 #(Shared.index 284) i last))
      (fun i -> i < last) (fun i -> PA.pts_to (source_program_address buffers i) (source_program_slot v i));
    source_program_prefix_to_runtime buffers #v last;
    PC.on_intro (PA.pts_to (source_program_address buffers last) (source_program_slot v last));
    source_allocation_count ();
    source_matching_allocation_count source_bf16_allocation_shapes buffers.bf16_buffers;
    source_matching_allocation_at source_bf16_allocation_shapes buffers.bf16_buffers last;
    source_program_shape_at last;
    let handle = List.index buffers.bf16_buffers last;
    rewrite each source_program_address buffers last as RT.data handle;
    fold RT.contents handle;
    fold RuntimeHelpers.all_contents #BF16.t [];
    fold RuntimeHelpers.all_contents [handle];
    source_contents_append (List.splitAt last buffers.bf16_buffers)._1 [handle];
    source_list_prefix_step buffers.bf16_buffers count;
    rewrite RuntimeHelpers.all_contents (List.append (List.splitAt last buffers.bf16_buffers)._1 [handle]) as
      RuntimeHelpers.all_contents (List.splitAt count buffers.bf16_buffers)._1;
  }
}

inline_for_extraction noextract
ghost
fn source_program_pack_return (#frame_t:Type0) (#frame:erased frame_t)
  (addresses:source_input_addresses) (#input:erased source_inputs)
  (output:source_returned_buffers{source_returned_metadata output})
  (#v:source_program_values)
  (#tokens:values I64.t 8{source_selection_result (source_program_slot v 283) (Ghost.reveal input).prompt tokens})
  preserves PC.loc KL.gpu_loc
  requires PA.pts_to addresses.weights_address (Ghost.reveal input).weights **
    PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
    PA.pts_to (RT.data output.returned_tokens) tokens **
    PA.pts_to (RT.data output.returned_hidden) (source_program_slot v 282) **
    pure (source_program_stage input v 228 /\ source_program_result input v tokens frame)
  ensures source_program_return_frame addresses input output frame
{
  PC.intro_exists
    (fun (ts:values I64.t 8{source_selection_result (source_program_slot v 283) (Ghost.reveal input).prompt ts}) ->
      PA.pts_to (RT.data output.returned_tokens) ts **
      PA.pts_to (RT.data output.returned_hidden) (source_program_slot v 282) **
      pure (source_program_stage input v 228 /\ source_program_result input v ts frame)) tokens;
  PC.intro_exists
    (fun (snapshot:source_program_values) ->
      exists* (ts:values I64.t 8{
        source_selection_result (source_program_slot snapshot 283) (Ghost.reveal input).prompt ts}).
        PA.pts_to (RT.data output.returned_tokens) ts **
        PA.pts_to (RT.data output.returned_hidden) (source_program_slot snapshot 282) **
        pure (source_program_stage input snapshot 228 /\ source_program_result input snapshot ts frame)) v;
  PC.on_intro (
    PA.pts_to addresses.weights_address (Ghost.reveal input).weights **
    PA.pts_to addresses.prompt_address (Ghost.reveal input).prompt **
    (exists* (snapshot:source_program_values)
      (ts:values I64.t 8{
        source_selection_result (source_program_slot snapshot 283) (Ghost.reveal input).prompt ts}).
      PA.pts_to (RT.data output.returned_tokens) ts **
      PA.pts_to (RT.data output.returned_hidden) (source_program_slot snapshot 282) **
      pure (source_program_stage input snapshot 228 /\ source_program_result input snapshot ts frame)));
  fold source_program_return_frame addresses input output frame;
}

inline_for_extraction noextract
ghost
fn source_program_retirement_contents (#frame_t:Type0) (#frame:erased frame_t)
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) ()
  requires PC.on KL.gpu_loc (source_program_terminal addresses buffers input frame)
  ensures RuntimeHelpers.all_contents (source_program_retirement buffers) **
    source_program_return_frame addresses input (source_program_returned buffers) frame
{
  ghost_impersonate KL.gpu_loc
    (PC.on KL.gpu_loc (source_program_terminal addresses buffers input frame))
    (RuntimeHelpers.all_contents (source_program_retirement buffers) **
      source_program_return_frame addresses input (source_program_returned buffers) frame)
    fn () {
      PC.on_elim (source_program_terminal addresses buffers input frame);
      unfold source_program_terminal addresses buffers input frame;
      with v tokens.
        assert source_program_storage buffers v 1.0R ** PA.pts_to (RT.data buffers.token_buffer) tokens **
          pure (source_program_stage input v 228 /\ source_program_result input v tokens frame);
      source_program_storage_equation buffers v 1.0R;
      rewrite source_program_storage buffers v 1.0R as source_program_masked_storage buffers v 1.0R (fun _ -> True);
      source_program_take buffers #v #1.0R (fun _ -> True) 282;
      source_program_take buffers #v #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 284) i 282)) 283;
      unfold source_program_masked_storage buffers v 1.0R
        (fun i -> (True /\ ~(eq2 #(Shared.index 284) i 282)) /\ ~(eq2 #(Shared.index 284) i 283));
      forevery_refine_ext #(Shared.index 284)
        #(fun i -> (True /\ ~(eq2 #(Shared.index 284) i 282)) /\ ~(eq2 #(Shared.index 284) i 283))
        (fun i -> i < 282)
        (fun i -> PA.pts_to (source_program_address buffers i) (source_program_slot v i));
      source_program_prefix_to_runtime buffers #v 282;
      source_contents_reverse (List.splitAt 282 buffers.bf16_buffers)._1;
      PC.on_intro (PA.pts_to (source_program_address buffers 283) (source_program_slot v 283));
      source_program_retirement_binding buffers;
      source_allocation_count ();
      source_matching_allocation_at source_bf16_allocation_shapes buffers.bf16_buffers 283;
      source_program_shape_at 283;
      let logits = List.index buffers.bf16_buffers 283;
      rewrite each source_program_address buffers 283 as RT.data logits;
      fold RT.contents logits;
      fold RuntimeHelpers.all_contents (logits :: List.rev' (List.splitAt 282 buffers.bf16_buffers)._1);
      rewrite RuntimeHelpers.all_contents (logits :: List.rev' (List.splitAt 282 buffers.bf16_buffers)._1) as
        RuntimeHelpers.all_contents (source_program_retirement buffers);
      source_program_return_addresses buffers;
      let output = source_program_returned buffers;
      assert pure (RT.metadata output.returned_hidden [7; 1024] /\ RT.metadata output.returned_tokens [1; 8]);
      rewrite each source_program_address buffers 282 as RT.data output.returned_hidden;
      rewrite each RT.data buffers.token_buffer as RT.data output.returned_tokens;
      source_program_pack_return #frame_t #frame addresses #input output #v #tokens;
      rewrite source_program_return_frame addresses input output frame as
        source_program_return_frame addresses input (source_program_returned buffers) frame;
    };
}

inline_for_extraction noextract
instance source_program_return_frame_send (#frame_t:Type0)
  (addresses:source_input_addresses) (input:source_inputs)
  (output:source_returned_buffers{source_returned_metadata output}) (frame:frame_t)
  : is_send (source_program_return_frame addresses input output frame) = solve

inline_for_extraction noextract
fn source_program_return (#frame_t:Type0) (#frame:erased frame_t)
  (addresses:source_input_addresses)
  (buffers:source_allocated_buffers{source_allocation_metadata buffers})
  (#input:erased source_inputs) (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream (start + 229)
  requires source_allocated_owners stream buffers **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 229))
      (PC.on KL.gpu_loc (source_program_terminal addresses buffers input frame))
  returns output:source_returned_result
  ensures RT.owner stream output.returned_tokens ** RT.owner stream output.returned_hidden **
    Pledge.pledge0 (Epoch.epoch_done stream (start + 229))
      (source_program_return_frame addresses input output frame)
{
  source_program_retirement_binding buffers;
  assert pure (start + 220 <= start + 229);
  let output = source_program_returned buffers;
  source_program_retirement_owners stream buffers;
  rewrite RT.owner stream (source_program_returned buffers).returned_tokens as RT.owner stream output.returned_tokens;
  rewrite RT.owner stream (source_program_returned buffers).returned_hidden as RT.owner stream output.returned_hidden;
  Pledge.rewrite_pledge
    (PC.on KL.gpu_loc (source_program_terminal addresses buffers input frame))
    (RuntimeHelpers.all_contents (source_program_retirement buffers) **
      source_program_return_frame addresses input (source_program_returned buffers) frame)
    (source_program_retirement_contents #frame_t #frame addresses buffers #input);
  RuntimeHelpers.release_ordered stream #(start + 229) (source_program_retirement buffers)
    (source_program_return_frame addresses input (source_program_returned buffers) frame) ();
  rewrite Pledge.pledge0 (Epoch.epoch_done stream (start + 229))
    (source_program_return_frame addresses input (source_program_returned buffers) frame) as
    Pledge.pledge0 (Epoch.epoch_done stream (start + 229))
      (source_program_return_frame addresses input output frame);
  output
}

inline_for_extraction noextract
fn program_refines (#frame_t:Type0) (#logical_frame:Ghost.erased frame_t)
  (addresses:source_input_addresses) (#input:Ghost.erased source_inputs)
  (stream:Stream.stream_t) (#start:Epoch.epoch_t) (ambient:PC.slprop)
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** source_borrowed_inputs addresses input
  returns outcome:source_program_outcome
  ensures (
    match outcome with
    | SourceProgramFailed _ ->
      Epoch.epoch_live stream start **
      Pledge.pledge0 (Epoch.epoch_done stream start) (source_borrowed_inputs addresses input)
    | SourceProgramReturned output ->
      Epoch.epoch_live stream (start + 229) **
      RT.owner stream output.returned_tokens ** RT.owner stream output.returned_hidden **
      Pledge.pledge0 (Epoch.epoch_done stream (start + 229))
        (source_program_return_frame addresses input output logical_frame))
{
  let first = first_block_refines addresses #input stream #start ambient;
  match first {
    SourcePrefixFailed error -> { SourceProgramFailed error }
    SourcePrefixEnqueued buffers -> {
      source_enqueue_remaining_layers addresses buffers #input stream #start;
      source_enqueue_program_suffix #frame_t #logical_frame addresses buffers #input stream #start;
      let output = source_program_return #frame_t #logical_frame addresses buffers #input stream #start;
      SourceProgramReturned output
    }
  }
}
