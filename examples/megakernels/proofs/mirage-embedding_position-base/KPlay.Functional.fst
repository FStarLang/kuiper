module KPlay.Functional

#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module BF16 = Kuiper.BFloat16
module Shared = KPlay.SharedSpec
module SZ = Kuiper.SizeT
module IC = FStar.Int.Cast
module FC = Kuiper.Float.Casts
module F32 = Kuiper.Float32
module R = FStar.Real
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry
module C = FStar.Classical

#set-options "--fuel 4 --ifuel 2 --z3rlimit 40"

noextract
let row_major_unique
  (width:pos) (r s:nat) (c d:nat{d < width})
  : Lemma
      (requires c < width)
      (ensures (r * width + c == s * width + d) <==> (r == s /\ c == d)) =
  Kuiper.Math.div_mod_of_mul_add width r c;
  Kuiper.Math.div_mod_of_mul_add width s d

noextract
let updated_index
  (values:seq BF16.t) (at:nat{at < Seq.length values})
  (value:BF16.t) (index:nat{index < Seq.length values})
  : Lemma (
      Seq.index (Seq.upd values at value) index ==
      (if index = at then value else Seq.index values index)) =
  if index = at then Seq.lemma_index_upd1 values at value
  else Seq.lemma_index_upd2 values at value index

noextract
let embedding_ready
  (weights:embedding_values) (ts:token_values)
  (hidden:hidden_values) (upto:nat) : prop =
  forall (p:Shared.prompt_index) (h:Shared.hidden_index).
    hidden_offset p h < upto ==>
    Seq.index hidden (hidden_offset p h) ==
      Seq.index weights (embedding_offset (token_at ts p) h)

noextract
let embedding_step
  (weights:embedding_values) (ts:token_values) (hidden:hidden_values)
  (p:Shared.prompt_index) (h:Shared.hidden_index) (value:BF16.t)
  : Lemma
      (requires
        embedding_ready weights ts hidden (hidden_offset p h) /\
        value == Seq.index weights (embedding_offset (token_at ts p) h))
      (ensures
        embedding_ready weights ts
          (Seq.upd hidden (hidden_offset p h) value) (hidden_offset p h + 1)) =
  let after = Seq.upd hidden (hidden_offset p h) value in
  let at_cell (r:Shared.prompt_index) (c:Shared.hidden_index)
    : Lemma (
        hidden_offset r c < hidden_offset p h + 1 ==>
        Seq.index after (hidden_offset r c) ==
          Seq.index weights (embedding_offset (token_at ts r) c)) =
    row_major_unique Shared.hidden_size r p c h;
    updated_index hidden (hidden_offset p h) value (hidden_offset r c);
    if hidden_offset r c < hidden_offset p h + 1 then (
      if hidden_offset r c = hidden_offset p h then
        assert (r == p /\ c == h)
      else
        assert (Seq.index hidden (hidden_offset r c) ==
          Seq.index weights (embedding_offset (token_at ts r) c))
    )
  in
  C.forall_intro_2 at_cell

noextract
let rotary_ready (rotary:rotary_values) (upto:nat) : prop =
  forall (p:Shared.prompt_index) (d:Shared.rotary_index).
    p * Shared.head_dim + d < upto ==>
    v_approximates (Seq.index rotary (rotary_offset 0 p d)) (Shared.cosine p d) /\
    v_approximates (Seq.index rotary (rotary_offset 1 p d)) (Shared.sine p d)

noextract
let rotary_step
  (rotary:rotary_values) (p:Shared.prompt_index) (d:Shared.rotary_index)
  (cosine sine:BF16.t)
  : Lemma
      (requires
        rotary_ready rotary (p * Shared.head_dim + d) /\
        v_approximates cosine (Shared.cosine p d) /\
        v_approximates sine (Shared.sine p d))
      (ensures
        rotary_ready
          (Seq.upd (Seq.upd rotary (rotary_offset 0 p d) cosine)
            (rotary_offset 1 p d) sine)
          (p * Shared.head_dim + d + 1)) =
  let a = rotary_offset 0 p d in
  let b = rotary_offset 1 p d in
  let first = Seq.upd rotary a cosine in
  let after = Seq.upd first b sine in
  let at_cell (r:Shared.prompt_index) (c:Shared.rotary_index)
    : Lemma (
        r * Shared.head_dim + c < p * Shared.head_dim + d + 1 ==>
        v_approximates (Seq.index after (rotary_offset 0 r c)) (Shared.cosine r c) /\
        v_approximates (Seq.index after (rotary_offset 1 r c)) (Shared.sine r c)) =
    row_major_unique Shared.head_dim r p c d;
    updated_index rotary a cosine (rotary_offset 0 r c);
    updated_index rotary a cosine (rotary_offset 1 r c);
    updated_index first b sine (rotary_offset 0 r c);
    updated_index first b sine (rotary_offset 1 r c);
    if r * Shared.head_dim + c < p * Shared.head_dim + d + 1 then (
      if r * Shared.head_dim + c = p * Shared.head_dim + d then
        assert (r == p /\ c == d)
      else
        assert (
          v_approximates (Seq.index rotary (rotary_offset 0 r c)) (Shared.cosine r c) /\
          v_approximates (Seq.index rotary (rotary_offset 1 r c)) (Shared.sine r c))
    )
  in
  C.forall_intro_2 at_cell

noextract
let frequency_refinement (d:Shared.rotary_index)
  : Lemma (
      d % 64 == Shared.frequency_of_coordinate d /\
      R.of_int (d % 64) /. 64.0R ==
        R.of_int (2 * Shared.frequency_of_coordinate d) /. R.of_int Shared.head_dim) =
  if d < 64 then Kuiper.Math.div_mod_of_mul_add 64 0 d
  else Kuiper.Math.div_mod_of_mul_add 64 1 (d - 64)

inline_for_extraction noextract
let integer_index (x:I64.t{0 <= I64.v x /\ SZ.fits (I64.v x)})
  : i:sz{SZ.v i == I64.v x} =
  assert_norm (pow2 32 == 0x100000000);
  SZ.uint32_to_sizet (IC.int64_to_uint32 x)

inline_for_extraction noextract
let coordinate (i:sz)
  : x:f32{v_approximates x (R.of_int (SZ.v i))} =
  let integer = IC.uint32_to_int64 (SZ.sizet_to_u32 i) in
  of_int_approx #f32 integer;
  of_int #f32 integer

inline_for_extraction noextract
let rotary_angle
  (p:sz{SZ.v p < Shared.prompt_length})
  (d:sz{SZ.v d < Shared.head_dim})
  (first_position:prefill_position)
  : result:f32{v_approximates result (Shared.angle (SZ.v p) (SZ.v d))} =
  let frequency = d %^ 64sz in
  let base = of_int #f32 1000000L in
  let denominator = of_int #f32 64L in
  let numerator = coordinate frequency in
  let exponent = div numerator denominator in
  let power = pow base exponent in
  let inverse = div (one #f32) power in
  let position = coordinate (p +^ integer_index first_position) in
  let angle = mul position inverse in
  let real_exponent = R.of_int (SZ.v frequency) /. 64.0R in
  let real_power = Pow.pow Shared.rope_theta real_exponent in
  let real_angle = R.of_int (SZ.v p) *. (1.0R /. real_power) in
  of_int_approx #f32 1000000L;
  of_int_approx #f32 64L;
  div_approx numerator denominator (R.of_int (SZ.v frequency)) 64.0R;
  F32.pow_approx base exponent Shared.rope_theta real_exponent;
  div_approx (one #f32) power 1.0R real_power;
  a_mul position inverse (R.of_int (SZ.v p)) (1.0R /. real_power);
  frequency_refinement (SZ.v d);
  assert (real_angle == Shared.angle (SZ.v p) (SZ.v d));
  angle

noextract
let rotary_casts_refinement
  (angle:f32) (p:Shared.prompt_index) (d:Shared.rotary_index)
  : Lemma
      (requires v_approximates angle (Shared.angle p d))
      (ensures
        v_approximates (FC.fcast #f32 #BF16.t (cos angle)) (Shared.cosine p d) /\
        v_approximates (FC.fcast #f32 #BF16.t (sin angle)) (Shared.sine p d)) =
  F32.cos_approx angle (Shared.angle p d);
  F32.sin_approx angle (Shared.angle p d);
  FC.fcast_approx #f32 #BF16.t (cos angle) (Trig.cos (Shared.angle p d));
  FC.fcast_approx #f32 #BF16.t (sin angle) (Trig.sin (Shared.angle p d))

inline_for_extraction noextract
fn embed
  (weights:larray BF16.t embedding_count)
  (tokens:larray I64.t Shared.prompt_length)
  (hidden:larray BF16.t hidden_count)
  (#w:erased embedding_values)
  (#ts:erased (lseq I64.t Shared.prompt_length){valid_tokens ts})
  (#before:erased hidden_values)
  preserves gpu ** weights |-> w ** tokens |-> ts
  requires hidden |-> before
  ensures exists* (after:hidden_values).
    hidden |-> after ** pure (embedding_ready w ts after hidden_count)
{
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (values:hidden_values).
      row |-> r ** hidden |-> values **
      pure (embedding_ready w ts values (SZ.v r * Shared.hidden_size))
    decreases (7 - SZ.v !row)
  {
    let r = !row;
    let token = tokens.(r);
    assert pure (0 <= I64.v token /\ I64.v token < Shared.vocab_size);
    let selected = integer_index token;
    let mut column = 0sz;
    while (!column <^ 1024sz)
      invariant exists* (c:sz{c <= 1024}) (values:hidden_values).
        column |-> c ** hidden |-> values **
        pure (embedding_ready w ts values (SZ.v r * Shared.hidden_size + SZ.v c))
      decreases (1024 - SZ.v !column)
    {
      let c = !column;
      let value = weights.(selected *^ 1024sz +^ c);
      with old. assert hidden |-> old;
      hidden.(r *^ 1024sz +^ c) <- value;
      embedding_step w ts old (SZ.v r) (SZ.v c) value;
      column := !column +^ 1sz;
    };
    row := !row +^ 1sz;
  }
}

inline_for_extraction noextract
fn make_rotary
  (first_position:prefill_position)
  (rotary:larray BF16.t rotary_count)
  (#before:erased rotary_values)
  preserves gpu
  requires rotary |-> before
  ensures exists* (after:rotary_values).
    rotary |-> after ** pure (rotary_ready after rotary_plane_count)
{
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (values:rotary_values).
      row |-> r ** rotary |-> values **
      pure (rotary_ready values (SZ.v r * Shared.head_dim))
    decreases (7 - SZ.v !row)
  {
    let r = !row;
    let mut column = 0sz;
    while (!column <^ 128sz)
      invariant exists* (d:sz{d <= 128}) (values:rotary_values).
        column |-> d ** rotary |-> values **
        pure (rotary_ready values (SZ.v r * Shared.head_dim + SZ.v d))
      decreases (128 - SZ.v !column)
    {
      let d = !column;
      let angle = rotary_angle r d first_position;
      rotary_casts_refinement angle (SZ.v r) (SZ.v d);
      let cosine = FC.fcast #f32 #BF16.t (cos angle);
      let sine = FC.fcast #f32 #BF16.t (sin angle);
      with old. assert rotary |-> old;
      let offset = r *^ 128sz +^ d;
      rotary.(offset) <- cosine;
      pts_to_len rotary;
      rotary.(896sz +^ offset) <- sine;
      rotary_step old (SZ.v r) (SZ.v d) cosine sine;
      column := !column +^ 1sz;
    };
    row := !row +^ 1sz;
  }
}

inline_for_extraction noextract
fn family_kernel
  (weights:larray BF16.t embedding_count)
  (tokens:larray I64.t Shared.prompt_length)
  (first_position:prefill_position)
  (hidden:larray BF16.t hidden_count)
  (rotary:larray BF16.t rotary_count)
  (#w:erased embedding_values)
  (#ts:erased (lseq I64.t Shared.prompt_length){valid_tokens ts})
  (#hidden_before:erased hidden_values)
  (#rotary_before:erased rotary_values)
  preserves gpu ** weights |-> w ** tokens |-> ts
  requires hidden |-> hidden_before ** rotary |-> rotary_before
  ensures exists* (h:hidden_values) (r:rotary_values).
    hidden |-> h ** rotary |-> r **
    pure (pointwise_postcondition w ts h r)
{
  embed weights tokens hidden;
  make_rotary first_position rotary;
}

instance send_global_array_contents
  (#a:Type0) (arr:array a{is_global_array arr})
  (#f:perm) (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents) =
  Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (weights:larray BF16.t embedding_count{is_global_array weights})
  (tokens:larray I64.t Shared.prompt_length{is_global_array tokens})
  (first_position:prefill_position)
  (hidden:larray BF16.t hidden_count{is_global_array hidden})
  (rotary:larray BF16.t rotary_count{is_global_array rotary})
  (#weights_before:erased embedding_values)
  (#tokens_before:erased (lseq I64.t Shared.prompt_length){
    valid_tokens tokens_before})
  preserves cpu **
    on gpu_loc (weights |-> weights_before ** tokens |-> tokens_before)
  requires exists* (hidden_before:hidden_values) (rotary_before:rotary_values).
    on gpu_loc (hidden |-> hidden_before ** rotary |-> rotary_before)
  ensures exists* (hidden_after:hidden_values) (rotary_after:rotary_values).
    on gpu_loc (hidden |-> hidden_after ** rotary |-> rotary_after) **
    pure (semantic_postcondition weights_before tokens_before first_position
      hidden_after rotary_after)
{
  with hidden_before rotary_before.
    assert on gpu_loc (hidden |-> hidden_before ** rotary |-> rotary_before);
  launch_kernel_1 (fun () ->
    family_kernel weights tokens first_position hidden rotary
      #weights_before #tokens_before #hidden_before #rotary_before);
  with hidden_after rotary_after.
    assert on gpu_loc (hidden |-> hidden_after ** rotary |-> rotary_after);
  establish_semantic_postcondition weights_before tokens_before first_position
    hidden_after rotary_after;
}
