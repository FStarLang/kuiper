module KPlay.Functional

open FStar.Real
module Shared = KPlay.SharedSpec
module Sc = Kuiper.Scalars
module N = Kuiper.Approximates.Base
module F = Kuiper.Floating.Base
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module Cast = Kuiper.Float.Casts
module C = FStar.Classical
module S = FStar.Seq.Base
module I64 = FStar.Int64
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry

inline_for_extraction noextract
let source_inverse_frequency (r:Shared.index 64) : F32.t =
  F.div (F.one #F32.t)
    (F.pow (F.of_int #F32.t 1000000L)
      (F.div (F.of_int #F32.t (I64.int_to_t r)) (F.of_int #F32.t 64L)))

inline_for_extraction noextract
let source_inverse_frequency_related (r:Shared.index 64)
  : Lemma (N.v_approximates (source_inverse_frequency r) (Shared.inverse_frequency r)) =
  let theta = F.of_int #F32.t 1000000L in
  let numerator = F.of_int #F32.t (I64.int_to_t r) in
  let denominator = F.of_int #F32.t 64L in
  let exponent = F.div numerator denominator in
  let exponent_real = of_int r /. 64.0R in
  N.of_int_approx #F32.t 1000000L;
  N.of_int_approx #F32.t (I64.int_to_t r);
  N.of_int_approx #F32.t 64L;
  N.div_approx numerator denominator (of_int r) 64.0R;
  F32.pow_approx theta exponent 1000000.0R exponent_real;
  N.div_approx (F.one #F32.t) (F.pow theta exponent) 1.0R
    (Pow.pow 1000000.0R exponent_real);
  assert (exponent_real == of_int (2 * r) /. 128.0R)

inline_for_extraction noextract
let source_angle (row:Shared.index 4096) (col:Shared.index 128) : F32.t =
  F.mul (source_inverse_frequency (col % 64)) (F.of_int #F32.t (I64.int_to_t row))

inline_for_extraction noextract
let source_cell (component:Shared.index 2) (row:Shared.index 4096) (col:Shared.index 128)
  : BF16.t =
  Cast.fcast #F32.t #BF16.t
    (if component = 0 then F.cos (source_angle row col) else F.sin (source_angle row col))

inline_for_extraction noextract
let source_rotary_state () : GTot (Shared.rotary_state F32.t) = {
  inv_freq = source_inverse_frequency; original_inv_freq = source_inverse_frequency;
  attention_scaling = F.one #F32.t;
  max_seq_len_cached = 40960; original_max_seq_len = 40960
}

inline_for_extraction noextract
let source_rotary_domain () : prop =
  forall (row:Shared.index 7) (col:Shared.index 128).
    not (F.NaN? (F.kind (F.cos (source_angle row col)))) /\
    not (F.NaN? (F.kind (F.sin (source_angle row col))))

inline_for_extraction noextract
let source_cell_related (component:Shared.index 2) (row:Shared.index 7) (col:Shared.index 128)
  (frequency factor:real)
  : Lemma
      (requires N.v_approximates (source_inverse_frequency (col % 64)) frequency /\
        N.v_approximates (F.one #F32.t) factor /\ source_rotary_domain ())
      (ensures N.v_approximates (source_cell component row col)
        ((if component = 0 then Trig.cos (of_int row *. frequency)
          else Trig.sin (of_int row *. frequency)) *. factor)) =
  N.of_int_approx #F32.t (I64.int_to_t row);
  N.a_mul (source_inverse_frequency (col % 64)) (F.of_int #F32.t (I64.int_to_t row))
    frequency (of_int row);
  assert (frequency *. of_int row == of_int row *. frequency);
  let value = if component = 0 then F.cos (source_angle row col) else F.sin (source_angle row col) in
  let target = if component = 0 then Trig.cos (of_int row *. frequency)
    else Trig.sin (of_int row *. frequency) in
  if component = 0 then F32.cos_approx (source_angle row col) (of_int row *. frequency)
  else F32.sin_approx (source_angle row col) (of_int row *. frequency);
  F.mul_one value;
  N.a_mul value (F.one #F32.t) target factor;
  Cast.fcast_approx #F32.t #BF16.t value (target *. factor)

inline_for_extraction noextract
let source_rotary_generation (table:Shared.rotary_tables BF16.t)
  : Lemma
      (requires
        (forall (row:Shared.index 7) (col:Shared.index 128).
          table.cosine row col == source_cell 0 row col /\
          table.sine row col == source_cell 1 row col))
      (ensures Shared.rotary_initialized (source_rotary_state ()) /\
        (source_rotary_domain () ==> Shared.mixed_rotary_generation (source_rotary_state ()) table)) =
  C.forall_intro source_inverse_frequency_related;
  if t2b (source_rotary_domain ()) then (
    let related (frequency:Shared.vector real 64) (factor:real)
      : Lemma
          (requires Shared.vector_approximates source_inverse_frequency frequency /\
            N.v_approximates (F.one #F32.t) factor)
          (ensures Shared.rotary_approximates table (Shared.tables_from_frequency frequency factor)) =
      let cell (row:Shared.index 7) (col:Shared.index 128)
        : Lemma (
            N.v_approximates (table.cosine row col)
              ((Shared.tables_from_frequency frequency factor).cosine row col) /\
            N.v_approximates (table.sine row col)
              ((Shared.tables_from_frequency frequency factor).sine row col)) =
        source_cell_related 0 row col (frequency (col % 64)) factor;
        source_cell_related 1 row col (frequency (col % 64)) factor
      in C.forall_intro_2 cell
    in C.forall_intro_2 (fun frequency -> C.move_requires (related frequency)))
