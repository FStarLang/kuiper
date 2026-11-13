module KPlay.Functional

#lang-pulse

open Kuiper
open FStar.Real
open Kuiper.Approximates.Base

module ObShared = KPlay.SharedSpec
module ObExp = FStar.Math.Exp
module ObPow = FStar.Math.Pow
module ObSum = Kuiper.Sum
module ObF = Kuiper.Floating.Base
module ObF32 = Kuiper.Float32
module ObCasts = Kuiper.Float.Casts
module ObI64 = FStar.Int64
module ObTrig = Kuiper.Real.Trigonometry

(* Reuse the historical sum proof, not its obsolete attention equation. *)
inline_for_extraction noextract
let sum_seven (f:ObShared.vector 7)
  : Lemma
      (ObShared.sum 7 f ==
        (f 0 +. f 4 +. (f 2 +. f 6)) +. (f 1 +. f 5 +. f 3)) =
  ObSum.sum_pop_left 0 7 f;
  ObSum.sum_pop_left 1 7 f;
  ObSum.sum_pop_left 2 7 f;
  ObSum.sum_pop_left 3 7 f;
  ObSum.sum_pop_left 4 7 f;
  ObSum.sum_pop_left 5 7 f;
  ObSum.sum_pop_left 6 7 f

inline_for_extraction noextract
let rec sum_extensional (n:nat) (left right:ObShared.vector n)
  : Lemma
      (requires forall (i:ObShared.index n). left i == right i)
      (ensures ObShared.sum n left == ObShared.sum n right)
      (decreases n)
  =
  if n > 0 then (
    sum_extensional (n - 1) left right;
    ObSum.sum_pop_right 0 n left;
    ObSum.sum_pop_right 0 n right
  )

inline_for_extraction noextract
let real_add_div (a b:real) (denominator:real{denominator >. 0.0R})
  : Lemma (a /. denominator +. b /. denominator == (a +. b) /. denominator)
  = ()

inline_for_extraction noextract
let real_div_cancel (a:real) (b:real{b >. 0.0R}) (c:real{c >. 0.0R})
  : Lemma ((a /. c) /. (b /. c) == a /. b)
  = ()

#push-options "--fuel 4 --ifuel 1 --z3rlimit 30"
inline_for_extraction noextract
let rec sum_divide (n:nat) (values:ObShared.vector n) (denominator:real{denominator >. 0.0R})
  : Lemma
      (ObShared.sum n (fun i -> values i /. denominator) ==
        ObShared.sum n values /. denominator)
      (decreases n)
  =
  let divided : ObShared.vector n = fun i -> values i /. denominator in
  if n > 0 then (
    sum_divide (n - 1) values denominator;
    sum_extensional (n - 1) divided (fun i -> values i /. denominator);
    assert (ObShared.sum (n - 1) divided ==
      ObShared.sum (n - 1) values /. denominator);
    ObSum.sum_pop_right 0 n values;
    ObSum.sum_pop_right 0 n divided;
    real_add_div (ObShared.sum (n - 1) values) (values (n - 1)) denominator
  )
#pop-options

inline_for_extraction noextract
let rec sum_zero_tail (n:nat) (cut:nat{cut <= n}) (values:ObShared.vector n)
  : Lemma
      (requires forall (i:ObShared.index n). cut <= i ==> values i == 0.0R)
      (ensures ObShared.sum n values == ObShared.sum cut values)
      (decreases n)
  =
  if cut < n then (
    sum_zero_tail (n - 1) cut values;
    ObSum.sum_pop_right 0 n values
  )

#push-options "--fuel 4 --ifuel 1"
inline_for_extraction noextract
let shared_softmax_at (n:pos) (scores:ObShared.vector n) (i:ObShared.index n)
  : Lemma
      (requires ObShared.sum n (fun j -> ObExp.exp (scores j)) >. 0.0R)
      (ensures ObShared.softmax scores i ==
        ObExp.exp (scores i) /. ObShared.sum n (fun j -> ObExp.exp (scores j)))
  =
  ObShared.sum_positive n (fun j -> ObExp.exp (scores j));
  assert_norm (ObShared.softmax scores i ==
    ObExp.exp (scores i) /. ObShared.sum n (fun j -> ObExp.exp (scores j)))
#pop-options

inline_for_extraction noextract
let prefix_normalize_shifted
  (token:ObShared.index 7) (scores:ObShared.vector (token + 1))
  (shift:real) (key:ObShared.index 7{key <= token})
  : Lemma
      (let shifted : ObShared.vector (token + 1) =
         fun i -> ObExp.exp (scores i -. shift) in
       ObShared.sum (token + 1) shifted >. 0.0R /\
       shifted key /. ObShared.sum (token + 1) shifted ==
         ObShared.prefix_softmax token scores key)
  =
  let original : ObShared.vector (token + 1) = fun i -> ObExp.exp (scores i) in
  let shifted : ObShared.vector (token + 1) = fun i -> ObExp.exp (scores i -. shift) in
  let divided : ObShared.vector (token + 1) = fun i -> original i /. ObExp.exp shift in
  let at (i:ObShared.index (token + 1))
    : Lemma (shifted i == divided i) = ObExp.exp_sub (scores i) shift
  in
  FStar.Classical.forall_intro at;
  sum_extensional (token + 1) shifted divided;
  sum_divide (token + 1) original (ObExp.exp shift);
  ObShared.sum_positive (token + 1) original;
  ObShared.sum_positive (token + 1) shifted;
  shared_softmax_at (token + 1) scores key;
  at key;
  assert (ObShared.sum (token + 1) shifted ==
    ObShared.sum (token + 1) original /. ObExp.exp shift);
  real_div_cancel (original key) (ObShared.sum (token + 1) original) (ObExp.exp shift)

inline_for_extraction noextract
let padded_real (values:ObShared.vector 7) (key:ObShared.index 64) : GTot real =
  if key < 7 then values key else 0.0R

(* This is an algebraic reduction identity, not a native IEEE identity. *)
inline_for_extraction noextract
let dense_value_equals_prefix
  (token:ObShared.index 7) (probabilities values:ObShared.vector 7)
  : Lemma
      (requires forall (key:ObShared.index 7).
        token < key ==> probabilities key == 0.0R)
      (ensures
        ObShared.sum 64 (fun key ->
          padded_real probabilities key *. padded_real values key) ==
        ObShared.sum (token + 1) (fun key -> probabilities key *. values key))
  =
  let term : ObShared.vector 64 =
    fun key -> padded_real probabilities key *. padded_real values key in
  let tail (key:ObShared.index 64)
    : Lemma (token + 1 <= key ==> term key == 0.0R) = ()
  in
  FStar.Classical.forall_intro tail;
  sum_zero_tail 64 (token + 1) term;
  sum_extensional (token + 1) term (fun key -> probabilities key *. values key)

inline_for_extraction noextract
let shared_dense_value_equals_prefix
  (data:ObShared.attention_data) (query repeated_key repeated_value:ObShared.heads 16)
  (token:ObShared.index 7) (head:ObShared.index 16) (lane:ObShared.index 128)
  : Lemma
      (let p = ObShared.attention_probabilities data query repeated_key in
       ObShared.sum 64 (fun key ->
         padded_real (p token head) key *.
         padded_real (fun source_token -> repeated_value source_token head lane) key) ==
       ObShared.attention_values p repeated_value token head lane)
  =
  let p = ObShared.attention_probabilities data query repeated_key in
  let future (key:ObShared.index 7)
    : Lemma (token < key ==> p token head key == 0.0R) =
    if token < key then ObShared.attention_future_zero data query repeated_key token head key
  in
  FStar.Classical.forall_intro future;
  let values : ObShared.vector 7 = fun source_token -> repeated_value source_token head lane in
  dense_value_equals_prefix token (p token head) values;
  sum_extensional (token + 1)
    (fun key -> p token head key *. values key)
    (fun key -> p token head key *. repeated_value key head lane);
  sum_extensional 64
    (fun key -> padded_real (p token head) key *. padded_real values key)
    (fun key ->
      padded_real (ObShared.attention_probabilities data query repeated_key token head) key *.
      padded_real (fun source_token -> repeated_value source_token head lane) key);
  assert_norm (ObShared.attention_values p repeated_value token head lane ==
    ObShared.sum (token + 1) (fun key -> p token head key *. repeated_value key head lane))

inline_for_extraction noextract
let nonzero_rescale_changes_singleton (rescale:real{rescale >. 0.0R})
  : Lemma (1.0R /. (rescale +. 1.0R) =!= 1.0R)
  = ()

(* Scalar producer arithmetic only; this does not prove the table's ownership. *)
inline_for_extraction noextract
let producer_rotary_cell
  (position:ObI64.t{0 <= ObI64.v position /\ ObI64.v position < 4096})
  (column:ObI64.t{0 <= ObI64.v column /\ ObI64.v column < 64})
  : bf16 & bf16 =
  let exponent = ObF.div #ObF32.t (ObF.of_int column) (ObF.of_int 64L) in
  let power = ObF.pow #ObF32.t (ObF.of_int 1000000L) exponent in
  let inverse = ObF.div #ObF32.t ObF.one power in
  let angle = ObF.mul #ObF32.t inverse (ObF.of_int position) in
  (ObCasts.fcast #ObF32.t #bf16 (ObF.cos angle),
   ObCasts.fcast #ObF32.t #bf16 (ObF.sin angle))

#push-options "--fuel 4 --ifuel 1 --z3rlimit 30"
inline_for_extraction noextract
let producer_rotary_cell_represents
  (position:ObI64.t{0 <= ObI64.v position /\ ObI64.v position < 4096})
  (column:ObI64.t{0 <= ObI64.v column /\ ObI64.v column < 64})
  (lane:ObShared.index 128{lane % 64 == ObI64.v column})
  : Lemma
      (let c, s = producer_rotary_cell position column in
       v_approximates c (ObTrig.cos (ObShared.rope_angle (ObI64.v position) lane)) /\
       v_approximates s (ObTrig.sin (ObShared.rope_angle (ObI64.v position) lane)))
  =
  of_int_approx #ObF32.t column;
  of_int_approx #ObF32.t position;
  of_int_approx #ObF32.t 64L;
  of_int_approx #ObF32.t 1000000L;
  let exponent = ObF.div #ObF32.t (ObF.of_int column) (ObF.of_int 64L) in
  let exponent_real = of_int (ObI64.v column) /. 64.0R in
  div_approx (ObF.of_int #ObF32.t column) (ObF.of_int #ObF32.t 64L)
    (of_int (ObI64.v column)) 64.0R;
  let power = ObF.pow #ObF32.t (ObF.of_int 1000000L) exponent in
  ObF32.pow_approx (ObF.of_int 1000000L) exponent 1000000.0R exponent_real;
  let inverse = ObF.div #ObF32.t ObF.one power in
  let one_ok = a1 #ObF32.t in
  div_approx (ObF.one #ObF32.t) power 1.0R (ObPow.pow 1000000.0R exponent_real);
  let angle = ObF.mul #ObF32.t inverse (ObF.of_int position) in
  a_mul inverse (ObF.of_int #ObF32.t position)
    (1.0R /. ObPow.pow 1000000.0R exponent_real) (of_int (ObI64.v position));
  assert (exponent_real == of_int (2 * (lane % 64)) /. 128.0R);
  assert (v_approximates angle (ObShared.rope_angle (ObI64.v position) lane));
  ObF32.cos_approx angle (ObShared.rope_angle (ObI64.v position) lane);
  ObF32.sin_approx angle (ObShared.rope_angle (ObI64.v position) lane);
  ObCasts.fcast_approx #ObF32.t #bf16 (ObF.cos angle)
    (ObTrig.cos (ObShared.rope_angle (ObI64.v position) lane));
  ObCasts.fcast_approx #ObF32.t #bf16 (ObF.sin angle)
    (ObTrig.sin (ObShared.rope_angle (ObI64.v position) lane))
#pop-options

inline_for_extraction noextract
let prefill_rotary_cell_represents
  (token:ObShared.index 7) (lane:ObShared.index 128)
  : Lemma
      (let c, s = producer_rotary_cell (ObI64.int_to_t token) (ObI64.int_to_t (lane % 64)) in
       v_approximates c (ObShared.prefill_cos token lane) /\
       v_approximates s (ObShared.prefill_sin token lane))
  =
  producer_rotary_cell_represents (ObI64.int_to_t token) (ObI64.int_to_t (lane % 64)) lane

inline_for_extraction noextract
let embedding_establishes_rotary_representation
  (table:bf16_values rotary_elements)
  : Lemma
      (requires forall (token:ObShared.index 7) (lane:ObShared.index 128).
        let cosine, sine = producer_rotary_cell (ObI64.int_to_t token) (ObI64.int_to_t (lane % 64)) in
        FStar.Seq.index table (token * 128 + lane) == cosine /\
        FStar.Seq.index table (524288 + token * 128 + lane) == sine)
      (ensures rotary_representation table 0 (ObShared.prefill_attention_data ())) =
  let at (token:ObShared.index 7) (lane:ObShared.index 128)
    : Lemma
        (FStar.Seq.index table (token * 128 + lane) %~ ObShared.prefill_cos token lane /\
         FStar.Seq.index table (524288 + token * 128 + lane) %~ ObShared.prefill_sin token lane) =
    prefill_rotary_cell_represents token lane
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let source_column (part:ObShared.index 4) (step:ObShared.index 16)
  : col:nat{col < 64} =
  part * 2 + (step / 2) * 8 + step % 2

inline_for_extraction noextract
let source_column_inverse (col:ObShared.index 64)
  : Lemma (source_column ((col % 8) / 2) ((col / 8) * 2 + col % 2) == col) =
  ()

inline_for_extraction noextract
let source_column_injective
  (p q:ObShared.index 4) (i j:ObShared.index 16)
  : Lemma (source_column p i == source_column q j ==> p == q /\ i == j) =
  ()

inline_for_extraction noextract
let source_live (row:ObShared.index 16) (col:ObShared.index 64) : bool =
  row < 14 && col <= row / 2

inline_for_extraction noextract
let logical_prefix_is_live
  (token:ObShared.index 7) (query_in_group:ObShared.index 2)
  (key:ObShared.index 7)
  : Lemma (source_live (2 * token + query_in_group) key <==> key <= token) =
  ()

inline_for_extraction noextract
let future_is_masked
  (token:ObShared.index 7) (query_in_group:ObShared.index 2)
  (key:ObShared.index 7{token < key})
  : Lemma (not (source_live (2 * token + query_in_group) key)) =
  logical_prefix_is_live token query_in_group key

inline_for_extraction noextract
let padding_is_masked (row:ObShared.index 16) (col:ObShared.index 64{7 <= col})
  : Lemma (not (source_live row col)) =
  ()

inline_for_extraction noextract
let empty_rows_are_masked (row:ObShared.index 16{14 <= row}) (col:ObShared.index 64)
  : Lemma (not (source_live row col)) =
  ()

inline_for_extraction noextract
let negative_infinity () : ObF32.t =
  ObF.sub ObF.zero ObF.infinity

inline_for_extraction noextract
let source_masked_score
  (row:ObShared.index 16) (col:ObShared.index 64) (score:ObF32.t) : ObF32.t =
  if source_live row col then score else negative_infinity ()

inline_for_extraction noextract
let negative_infinity_compares_equal ()
  : Lemma (ieee_eq (negative_infinity ()) (negative_infinity ())) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  ObF.eq_spec (negative_infinity ()) (negative_infinity ())

inline_for_extraction noextract
let masked_exponent_branch (value active_result:ObF32.t) : ObF32.t =
  if ieee_eq value (negative_infinity ()) then ObF.zero else active_result

inline_for_extraction noextract
let masked_exponent_is_zero (active_result:ObF32.t)
  : Lemma (masked_exponent_branch (negative_infinity ()) active_result == (ObF.zero #ObF32.t)) =
  negative_infinity_compares_equal ()

inline_for_extraction noextract
let source_future_exponent_is_zero
  (token:ObShared.index 7) (query_in_group:ObShared.index 2)
  (key:ObShared.index 7{token < key})
  (score active_result:ObF32.t)
  : Lemma
      (masked_exponent_branch
        (source_masked_score (2 * token + query_in_group) key score)
        active_result == (ObF.zero #ObF32.t)) =
  future_is_masked token query_in_group key;
  masked_exponent_is_zero active_result

inline_for_extraction noextract
let source_future_probability_represents_shared
  (token:ObShared.index 7) (query_in_group:ObShared.index 2)
  (key:ObShared.index 7{token < key})
  (scores:ObShared.vector (token + 1)) (score active_result:ObF32.t)
  : Lemma
      (v_approximates
        (ObCasts.fcast #ObF32.t #bf16
          (masked_exponent_branch
            (source_masked_score (2 * token + query_in_group) key score) active_result))
        (ObShared.prefix_softmax token scores key)) =
  source_future_exponent_is_zero token query_in_group key score active_result;
  ObShared.prefix_softmax_future_zero token scores key;
  let zero_ok = a0 #ObF32.t in
  ObCasts.fcast_approx #ObF32.t #bf16 ObF.zero 0.0R

inline_for_extraction noextract
let empty_rescale_branch (maximum nonempty_result:ObF32.t) : ObF32.t =
  if ieee_eq maximum (negative_infinity ()) then ObF.one else nonempty_result

inline_for_extraction noextract
let empty_rescale_is_one (nonempty_result:ObF32.t)
  : Lemma (empty_rescale_branch (negative_infinity ()) nonempty_result == (ObF.one #ObF32.t)) =
  negative_infinity_compares_equal ()

inline_for_extraction noextract
let ordinary_expf_negative_infinity_is_zero ()
  : Lemma (ObF.fexp (negative_infinity ()) == (ObF.zero #ObF32.t)) =
  ObF32.exp_special_values (negative_infinity ())

inline_for_extraction noextract
let source_attention_literal () : ObF32.t =
  ObF.of_literal "0.12751743197441101074"

inline_for_extraction noextract
let source_attention_literal_is_positive_normal ()
  : Lemma (ObF32.positive_normal (source_attention_literal ())) =
  assert_norm (Kuiper.Float32.Literals.positive_normal_decimal "0.12751743197441101074");
  ObF32.positive_normal_of_literal "0.12751743197441101074"

inline_for_extraction noextract
let source_addend_guard (negative_scaled:ObF32.t) : prop =
  Finite? (kind negative_scaled) \/ negative_scaled == negative_infinity ()

inline_for_extraction noextract
let source_rescale_tail_contract (negative_scaled result:ObF32.t) : prop =
  (exists (argument:ObF32.t).
    (forall (xr yr zr:real).
      v_approximates (negative_infinity ()) xr /\
      v_approximates (source_attention_literal ()) yr /\
      v_approximates negative_scaled zr ==>
      v_approximates argument (xr *. yr +. zr)) /\
    (forall (xr:real).
      v_approximates argument xr ==> v_approximates result (ObPow.exp2 xr))) /\
  (source_addend_guard negative_scaled ==>
    result == (ObF.zero #ObF32.t) /\ v_approximates result 0.0R) /\
  (negative_scaled == (ObF.infinity #ObF32.t) \/ NaN? (kind negative_scaled) ==>
    NaN? (kind result))

(* The guard is conditional, not a new decoder precondition. *)
inline_for_extraction noextract
fn source_rescale_tail (negative_scaled:ObF32.t)
  preserves gpu
  returns result:ObF32.t
  ensures pure (source_rescale_tail_contract negative_scaled result)
{
  source_attention_literal_is_positive_normal ();
  let zero_ok = a0 #ObF32.t;
  let argument = ObF32.fma_rn_ftz (negative_infinity ())
    (source_attention_literal ()) negative_scaled;
  let result = ObF32.exp2_approx_ftz argument;
  assert pure (exists (a:ObF32.t).
    (forall (xr yr zr:real).
      v_approximates (negative_infinity ()) xr /\
      v_approximates (source_attention_literal ()) yr /\
      v_approximates negative_scaled zr ==>
      v_approximates a (xr *. yr +. zr)) /\
    (forall (xr:real).
      v_approximates a xr ==> v_approximates result (ObPow.exp2 xr)));
  result
}

inline_for_extraction noextract
fn source_exp2_negative_infinity_zero ()
  preserves gpu
  returns result:ObF32.t
  ensures pure (result == (ObF.zero #ObF32.t) /\ v_approximates result 0.0R)
{
  to_real_ok (negative_infinity ());
  let result = ObF32.exp2_approx_ftz (negative_infinity ());
  assert pure (v_approximates result (ObPow.exp2 (to_real (negative_infinity ()))));
  assert pure (result == (ObF.zero #ObF32.t));
  result
}

inline_for_extraction noextract
fn source_rescale_for_guarded_addend (negative_scaled:ObF32.t)
  preserves gpu
  requires pure (source_addend_guard negative_scaled)
  returns result:ObF32.t
  ensures pure (result == (ObF.zero #ObF32.t) /\ v_approximates result 0.0R)
{
  source_rescale_tail negative_scaled
}

inline_for_extraction noextract
fn source_rescale_at_zero_shift ()
  preserves gpu
  returns result:ObF32.t
  ensures pure (result == (ObF.zero #ObF32.t) /\ v_approximates result 0.0R)
{
  source_rescale_for_guarded_addend ObF.zero
}

inline_for_extraction noextract
fn source_rescale_at_negative_infinity_shift ()
  preserves gpu
  returns result:ObF32.t
  ensures pure (result == (ObF.zero #ObF32.t) /\ v_approximates result 0.0R)
{
  source_rescale_for_guarded_addend (negative_infinity ())
}

inline_for_extraction noextract
fn source_rescale_infinite_cancellation ()
  preserves gpu
  returns result:ObF32.t
  ensures pure (NaN? (kind result))
{
  source_rescale_tail ObF.infinity
}

inline_for_extraction noextract
fn store_masked_probability (output:gpu_ref bf16) (#old:erased bf16)
  preserves gpu
  requires output |-> old
  ensures output |-> ObCasts.fcast #ObF32.t #bf16 ObF.zero
{
  output := ObCasts.fcast #ObF32.t #bf16 ObF.zero;
}

fn run_masked_probability (output:gpu_ref bf16) (#old:erased bf16)
  preserves cpu
  requires on gpu_loc (output |-> old)
  ensures on gpu_loc (output |-> ObCasts.fcast #ObF32.t #bf16 ObF.zero)
{
  launch_kernel_1 (fun () -> store_masked_probability output #old);
}

inline_for_extraction noextract
fn store_source_rescale_tail
  (negative_scaled:ObF32.t) (output:gpu_ref ObF32.t) (#old:erased ObF32.t)
  preserves gpu
  requires output |-> old
  ensures exists* (result:ObF32.t).
    output |-> result ** pure (source_rescale_tail_contract negative_scaled result)
{
  let result = source_rescale_tail negative_scaled;
  output := result;
}

fn run_source_rescale_tail
  (negative_scaled:ObF32.t) (output:gpu_ref ObF32.t) (#old:erased ObF32.t)
  preserves cpu
  requires on gpu_loc (output |-> old)
  ensures exists* (result:ObF32.t).
    on gpu_loc (output |-> result) ** pure (source_rescale_tail_contract negative_scaled result)
{
  launch_kernel_1 (fun () -> store_source_rescale_tail negative_scaled output #old);
}

(* A scalar model of decoder.cu:235-246, not a score-MMA reachability theorem. *)
inline_for_extraction noextract
let source_row_score
  (row:ObShared.index 16) (scores:lseq ObF32.t 8) (col:ObShared.index 64)
  : ObF32.t =
  if source_live row col then FStar.Seq.index scores col else negative_infinity ()

inline_for_extraction noextract
let rec source_part_maximum
  (row:ObShared.index 16) (scores:lseq ObF32.t 8)
  (part:ObShared.index 4) (steps:nat{steps <= 16})
  : Tot ObF32.t (decreases steps) =
  if steps = 0 then negative_infinity ()
  else
    ObF.fmax (source_part_maximum row scores part (steps - 1))
      (source_row_score row scores (source_column part (steps - 1)))

inline_for_extraction noextract
let source_maximum (row:ObShared.index 16) (scores:lseq ObF32.t 8) : ObF32.t =
  ObF.fmax
    (ObF.fmax (source_part_maximum row scores 0 16) (source_part_maximum row scores 1 16))
    (ObF.fmax (source_part_maximum row scores 2 16) (source_part_maximum row scores 3 16))

inline_for_extraction noextract
let fmax_non_nan_when_both_non_nan (left right:ObF32.t)
  : Lemma
      (requires not (NaN? (kind left)) /\ not (NaN? (kind right)))
      (ensures not (NaN? (kind (ObF.fmax left right)))) =
  ObF.fmax_spec left right;
  ObF.eq_spec (ObF.fmax left right) (if ObF.lt left right then right else left)

inline_for_extraction noextract
let rec source_part_maximum_non_nan_when_scores_non_nan
  (row:ObShared.index 16) (scores:lseq ObF32.t 8)
  (part:ObShared.index 4) (steps:nat{steps <= 16})
  : Lemma
      (requires forall (col:ObShared.index 8). not (NaN? (kind (FStar.Seq.index scores col))))
      (ensures not (NaN? (kind (source_part_maximum row scores part steps))))
      (decreases steps)
  =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  if steps > 0 then (
    source_part_maximum_non_nan_when_scores_non_nan row scores part (steps - 1);
    fmax_non_nan_when_both_non_nan
      (source_part_maximum row scores part (steps - 1))
      (source_row_score row scores (source_column part (steps - 1)))
  )

inline_for_extraction noextract
let source_maximum_non_nan_when_scores_non_nan
  (row:ObShared.index 16) (scores:lseq ObF32.t 8)
  : Lemma
      (requires forall (col:ObShared.index 8). not (NaN? (kind (FStar.Seq.index scores col))))
      (ensures not (NaN? (kind (source_maximum row scores))))
  =
  source_part_maximum_non_nan_when_scores_non_nan row scores 0 16;
  source_part_maximum_non_nan_when_scores_non_nan row scores 1 16;
  source_part_maximum_non_nan_when_scores_non_nan row scores 2 16;
  source_part_maximum_non_nan_when_scores_non_nan row scores 3 16;
  fmax_non_nan_when_both_non_nan
    (source_part_maximum row scores 0 16) (source_part_maximum row scores 1 16);
  fmax_non_nan_when_both_non_nan
    (source_part_maximum row scores 2 16) (source_part_maximum row scores 3 16);
  fmax_non_nan_when_both_non_nan
    (ObF.fmax (source_part_maximum row scores 0 16) (source_part_maximum row scores 1 16))
    (ObF.fmax (source_part_maximum row scores 2 16) (source_part_maximum row scores 3 16))

inline_for_extraction noextract
let fmax_negative_infinity ()
  : Lemma
      (ObF.fmax (negative_infinity ()) (negative_infinity ()) == negative_infinity ()) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  ObF.fmax_spec (negative_infinity ()) (negative_infinity ());
  ObF.eq_spec (ObF.fmax (negative_infinity ()) (negative_infinity ())) (negative_infinity ())

inline_for_extraction noextract
let fmax_non_nan_left (left right:ObF32.t)
  : Lemma
      (requires not (NaN? (kind left)))
      (ensures not (NaN? (kind (ObF.fmax left right)))) =
  ObF32.fmax_special_values left right;
  if not (NaN? (kind right)) then
    fmax_non_nan_when_both_non_nan left right

inline_for_extraction noextract
let rec source_part_maximum_non_nan
  (row:ObShared.index 16) (scores:lseq ObF32.t 8)
  (part:ObShared.index 4) (steps:nat{steps <= 16})
  : Lemma (not (NaN? (kind (source_part_maximum row scores part steps))))
      (decreases steps) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  if steps > 0 then (
    source_part_maximum_non_nan row scores part (steps - 1);
    fmax_non_nan_left
      (source_part_maximum row scores part (steps - 1))
      (source_row_score row scores (source_column part (steps - 1)))
  )

inline_for_extraction noextract
let source_maximum_non_nan (row:ObShared.index 16) (scores:lseq ObF32.t 8)
  : Lemma (not (NaN? (kind (source_maximum row scores)))) =
  source_part_maximum_non_nan row scores 0 16;
  source_part_maximum_non_nan row scores 1 16;
  source_part_maximum_non_nan row scores 2 16;
  source_part_maximum_non_nan row scores 3 16;
  fmax_non_nan_left
    (source_part_maximum row scores 0 16) (source_part_maximum row scores 1 16);
  fmax_non_nan_left
    (source_part_maximum row scores 2 16) (source_part_maximum row scores 3 16);
  fmax_non_nan_left
    (ObF.fmax (source_part_maximum row scores 0 16) (source_part_maximum row scores 1 16))
    (ObF.fmax (source_part_maximum row scores 2 16) (source_part_maximum row scores 3 16))

inline_for_extraction noextract
let source_scale_upper_bound ()
  : Lemma (ObF.lte (source_attention_literal ()) ObF.one) =
  assert_norm (Kuiper.Float32.Literals.positive_normal_decimal "0.12751743197441101074");
  assert_norm (Kuiper.Float32.Literals.at_most_one_decimal "0.12751743197441101074");
  ObF32.positive_literal_at_most_one "0.12751743197441101074"

inline_for_extraction noextract
let source_maximum_cases (maximum:ObF32.t)
  : Lemma (not (NaN? (kind maximum)) ==>
      Finite? (kind maximum) \/ maximum == ObF.infinity \/ maximum == negative_infinity ()) =
  if not (NaN? (kind maximum)) then ObF32.non_nan_classification maximum

inline_for_extraction noextract
let rec source_part_maximum_empty
  (row:ObShared.index 16{14 <= row}) (scores:lseq ObF32.t 8)
  (part:ObShared.index 4) (steps:nat{steps <= 16})
  : Lemma
      (source_part_maximum row scores part steps == negative_infinity ())
      (decreases steps)
  =
  if steps > 0 then (
    source_part_maximum_empty row scores part (steps - 1);
    empty_rows_are_masked row (source_column part (steps - 1));
    fmax_negative_infinity ()
  )

inline_for_extraction noextract
let source_maximum_empty
  (row:ObShared.index 16{14 <= row}) (scores:lseq ObF32.t 8)
  : Lemma
      (source_maximum row scores == negative_infinity () /\
       ieee_eq (source_maximum row scores) (negative_infinity ()))
  =
  source_part_maximum_empty row scores 0 16;
  source_part_maximum_empty row scores 1 16;
  source_part_maximum_empty row scores 2 16;
  source_part_maximum_empty row scores 3 16;
  fmax_negative_infinity ();
  negative_infinity_compares_equal ()

inline_for_extraction noextract
let source_scaled_rescale_contract (maximum result:ObF32.t) : prop =
  exists (scaled:ObF32.t).
    (forall (mr ar:real).
      v_approximates maximum mr /\ v_approximates (source_attention_literal ()) ar ==>
      v_approximates scaled (mr *. ar)) /\
    (ieee_eq maximum (negative_infinity ()) ==> result == (ObF.one #ObF32.t)) /\
    (not (ieee_eq maximum (negative_infinity ())) ==>
      source_rescale_tail_contract (ObF.sub ObF.zero scaled) result) /\
    (not (NaN? (kind maximum)) /\ not (ieee_eq maximum (negative_infinity ())) ==>
      result == (ObF.zero #ObF32.t) /\ v_approximates result 0.0R)

(* The product is computed before the branch, as in decoder.cu:247,259-260. *)
inline_for_extraction noextract
fn source_rescale_from_maximum (maximum:ObF32.t)
  preserves gpu
  returns result:ObF32.t
  ensures pure (source_scaled_rescale_contract maximum result)
{
  source_attention_literal_is_positive_normal ();
  source_scale_upper_bound ();
  source_maximum_cases maximum;
  negative_infinity_compares_equal ();
  let scaled = ObF32.mul_rn_ftz maximum (source_attention_literal ());
  ObF.neg_kind scaled;
  assert pure (not (NaN? (kind maximum)) /\ not (ieee_eq maximum (negative_infinity ())) ==>
    source_addend_guard (ObF.sub ObF.zero scaled));
  let one : ObF32.t = ObF.one;
  let result = if ieee_eq maximum (negative_infinity ()) {
    one
  } else {
    source_rescale_tail (ObF.sub ObF.zero scaled)
  };
  assert pure (source_scaled_rescale_contract maximum result);
  result
}

inline_for_extraction noextract
fn store_source_rescale_from_maximum
  (maximum:ObF32.t) (output:gpu_ref ObF32.t) (#old:erased ObF32.t)
  preserves gpu
  requires output |-> old
  ensures exists* (result:ObF32.t).
    output |-> result ** pure (source_scaled_rescale_contract maximum result)
{
  let result = source_rescale_from_maximum maximum;
  output := result;
}

fn run_source_rescale_from_maximum
  (maximum:ObF32.t) (output:gpu_ref ObF32.t) (#old:erased ObF32.t)
  preserves cpu
  requires on gpu_loc (output |-> old)
  ensures exists* (result:ObF32.t).
    on gpu_loc (output |-> result) ** pure (source_scaled_rescale_contract maximum result)
{
  launch_kernel_1 (fun () -> store_source_rescale_from_maximum maximum output #old);
}

module ObSeq = FStar.Seq
module ObSZ = Kuiper.SizeT

inline_for_extraction noextract
let source_score_row (tile:lseq ObF32.t 128) (row:ObShared.index 16)
  : lseq ObF32.t 8 =
  ObSeq.slice tile (row * 8) (row * 8 + 8)

inline_for_extraction noextract
let source_score_row_at
  (tile:lseq ObF32.t 128) (row:ObShared.index 16) (col:ObShared.index 8)
  : Lemma
      (ObSeq.index (source_score_row tile row) col ==
        ObSeq.index tile (row * 8 + col))
  = ()

inline_for_extraction noextract
let source_part_maximum_step
  (row:ObShared.index 16) (scores:lseq ObF32.t 8)
  (part:ObShared.index 4) (step:ObShared.index 16)
  : Lemma
      (source_part_maximum row scores part (step + 1) ==
        ObF.fmax (source_part_maximum row scores part step)
          (source_row_score row scores (source_column part step)))
  = ()

inline_for_extraction noextract
fn source_read_score_tile
  (tile:larray ObF32.t 128) (row:szlt 16) (col:szlt 64)
  (#values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f values
  returns result:ObF32.t
  ensures pure (result ==
    source_row_score row (source_score_row values row) col)
{
  open Pulse.Lib.Array;
  if (row <^ 14sz && col <=^ row /^ 2sz) {
    source_score_row_at values row col;
    tile.(row *^ 8sz +^ col)
  } else {
    negative_infinity ()
  }
}

inline_for_extraction noextract
fn source_part_maximum_from_tile
  (tile:larray ObF32.t 128) (row:szlt 16) (part:szlt 4)
  (#values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f values
  returns result:ObF32.t
  ensures pure (result ==
    source_part_maximum row (source_score_row values row) part 16)
{
  let mut j = 0sz;
  let mut maximum = negative_infinity ();
  while (!j <^ 16sz)
    invariant exists* (step:sz{step <= 16}).
      j |-> step **
      maximum |-> source_part_maximum row (source_score_row values row) part step
    decreases (16 - !j)
  {
    let step = !j;
    let col = part *^ 2sz +^ (step /^ 2sz) *^ 8sz +^ step %^ 2sz;
    let value = source_read_score_tile tile row col;
    let previous = !maximum;
    source_part_maximum_step row (source_score_row values row) part step;
    maximum := ObF.fmax previous value;
    j := step +^ 1sz;
  };
  !maximum
}

inline_for_extraction noextract
fn source_maximum_from_tile
  (tile:larray ObF32.t 128) (row:szlt 16)
  (#values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f values
  returns result:ObF32.t
  ensures pure (
    result == source_maximum row (source_score_row values row) /\
    not (NaN? (kind result)))
{
  let m0 = source_part_maximum_from_tile tile row 0sz;
  let m1 = source_part_maximum_from_tile tile row 1sz;
  let m2 = source_part_maximum_from_tile tile row 2sz;
  let m3 = source_part_maximum_from_tile tile row 3sz;
  source_maximum_non_nan row (source_score_row values row);
  ObF.fmax (ObF.fmax m0 m1) (ObF.fmax m2 m3)
}

inline_for_extraction noextract
let source_exponent_contract (value scaled result:ObF32.t) : prop =
  forall (vr sr ar:real).
    v_approximates value vr /\
    v_approximates scaled sr /\
    v_approximates (source_attention_literal ()) ar ==>
    v_approximates result (ObPow.exp2 (vr *. ar -. sr))

inline_for_extraction noextract
let source_exponent_chain
  (value scaled argument result:ObF32.t)
  : Lemma
      (requires
        (forall (vr ar nr:real).
          v_approximates value vr /\
          v_approximates (source_attention_literal ()) ar /\
          v_approximates (ObF.sub ObF.zero scaled) nr ==>
          v_approximates argument (vr *. ar +. nr)) /\
        (forall (r:real).
          v_approximates argument r ==> v_approximates result (ObPow.exp2 r)))
      (ensures source_exponent_contract value scaled result)
  =
  let at (vr sr ar:real)
    : Lemma
        (requires v_approximates value vr /\ v_approximates scaled sr /\
          v_approximates (source_attention_literal ()) ar)
        (ensures v_approximates result (ObPow.exp2 (vr *. ar -. sr)))
    =
    let zero_ok = a0 #ObF32.t in
    sub_approx (ObF.zero #ObF32.t) scaled 0.0R sr;
    assert (v_approximates argument (vr *. ar +. (0.0R -. sr)))
  in
  FStar.Classical.forall_intro_3 (FStar.Classical.move_requires_3 at)

inline_for_extraction noextract
let source_branch_exponent_contract (value scaled result:ObF32.t) : prop =
  (ieee_eq value (negative_infinity ()) ==>
    result == (ObF.zero #ObF32.t) /\ v_approximates result 0.0R) /\
  (not (ieee_eq value (negative_infinity ())) ==>
    source_exponent_contract value scaled result)

inline_for_extraction noextract
fn source_exponent_from_scaled (value scaled:ObF32.t)
  preserves gpu
  returns result:ObF32.t
  ensures pure (source_branch_exponent_contract value scaled result)
{
  if ieee_eq value (negative_infinity ()) {
    let zero_ok = a0 #ObF32.t;
    ObF.zero
  } else {
    let argument = ObF32.fma_rn_ftz value (source_attention_literal ())
      (ObF.sub ObF.zero scaled);
    let result = ObF32.exp2_approx_ftz argument;
    source_exponent_chain value scaled argument result;
    result
  }
}

inline_for_extraction noextract
let source_tile_entry_contract
  (values:lseq ObF32.t 128) (row:ObShared.index 16) (col:ObShared.index 64)
  (exponent rescale:ObF32.t) : prop =
  let maximum = source_maximum row (source_score_row values row) in
  let value = source_row_score row (source_score_row values row) col in
  exists (scaled:ObF32.t).
    (forall (mr ar:real).
      v_approximates maximum mr /\ v_approximates (source_attention_literal ()) ar ==>
      v_approximates scaled (mr *. ar)) /\
    source_branch_exponent_contract value scaled exponent /\
    (ieee_eq maximum (negative_infinity ()) ==>
      rescale == (ObF.one #ObF32.t) /\ v_approximates rescale 1.0R) /\
    (not (ieee_eq maximum (negative_infinity ())) ==>
      source_rescale_tail_contract (ObF.sub ObF.zero scaled) rescale /\
      rescale == (ObF.zero #ObF32.t) /\ v_approximates rescale 0.0R)

(* One entry and the rescale share the actual loaded row and scaled maximum.
   The complete probability/sum loops and score-MMA publication are not here. *)
inline_for_extraction noextract
fn source_score_tile_entry
  (tile:larray ObF32.t 128) (row:szlt 16) (col:szlt 64)
  (probability:gpu_ref bf16) (rescale_output:gpu_ref ObF32.t)
  (#values:erased (lseq ObF32.t 128)) (#f:perm)
  (#old_probability:erased bf16) (#old_rescale:erased ObF32.t)
  preserves gpu ** tile |-> Frac f values
  requires probability |-> old_probability ** rescale_output |-> old_rescale
  ensures exists* (exponent rescale:ObF32.t).
    probability |-> ObCasts.fcast #ObF32.t #bf16 exponent **
    rescale_output |-> rescale **
    pure (source_tile_entry_contract values row col exponent rescale)
{
  let maximum = source_maximum_from_tile tile row;
  source_attention_literal_is_positive_normal ();
  source_scale_upper_bound ();
  source_maximum_cases maximum;
  negative_infinity_compares_equal ();
  let scaled = ObF32.mul_rn_ftz maximum (source_attention_literal ());
  ObF.neg_kind scaled;
  assert pure (not (ieee_eq maximum (negative_infinity ())) ==>
    source_addend_guard (ObF.sub ObF.zero scaled));
  let value = source_read_score_tile tile row col;
  let exponent = source_exponent_from_scaled value scaled;
  probability := ObCasts.fcast #ObF32.t #bf16 exponent;
  let one_ok = a1 #ObF32.t;
  let one : ObF32.t = ObF.one;
  let rescale = if ieee_eq maximum (negative_infinity ()) {
    one
  } else {
    source_rescale_tail (ObF.sub ObF.zero scaled)
  };
  assert pure (source_tile_entry_contract values row col exponent rescale);
  rescale_output := rescale;
}

inline_for_extraction noextract
instance source_send_score_array
  (#a:Type0) (arr:array a{is_global_array arr}) (#f:perm) (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn run_source_score_tile_entry
  (tile:larray ObF32.t 128{is_global_array tile}) (row:szlt 16) (col:szlt 64)
  (probability:gpu_ref bf16) (rescale_output:gpu_ref ObF32.t)
  (#values:erased (lseq ObF32.t 128)) (#f:perm)
  (#old_probability:erased bf16) (#old_rescale:erased ObF32.t)
  preserves cpu ** on gpu_loc (tile |-> Frac f values)
  requires on gpu_loc (probability |-> old_probability ** rescale_output |-> old_rescale)
  ensures exists* (exponent rescale:ObF32.t).
    on gpu_loc (
      probability |-> ObCasts.fcast #ObF32.t #bf16 exponent **
      rescale_output |-> rescale) **
    pure (source_tile_entry_contract values row col exponent rescale)
{
  launch_kernel_1 (fun () ->
    source_score_tile_entry tile row col probability rescale_output
      #values #f #old_probability #old_rescale);
}

open Kuiper.Shape

module ObChest = Kuiper.Chest
module ObMMA = Kuiper.TensorCore.MMA
module ObGemm = Kuiper.Spec.GEMM
module ObRoot = FStar.Math.Sqrt

inline_for_extraction noextract
let source_query_register_binding
  (rotated:lseq bf16 (8 * 4096)) (head:ObShared.index 8)
  (step:ObShared.index 8) (thread:ObShared.index 32) (reg:ObShared.index 8)
  : Lemma
      (let row = thread / 4 + ((reg / 2) % 2) * 8 in
       let lane = (thread % 4) * 2 + reg % 2 + (reg / 4) * 8 in
       ObChest.acc2 (source_query_operand rotated head step) row lane ==
         (if row < 14 then ObSeq.index rotated
           ((row / 2) * 4096 + head * 512 + (row % 2) * 128 + step * 16 + lane)
          else ObF.zero))
  = ()

inline_for_extraction noextract
let source_key_register_binding
  (rotated:lseq bf16 (8 * 4096)) (head:ObShared.index 8)
  (step:ObShared.index 8) (thread:ObShared.index 32) (reg:ObShared.index 4)
  : Lemma
      (let key = thread / 4 in
       let lane = (thread % 4) * 2 + reg % 2 + (reg / 2) * 8 in
       ObChest.acc2 (source_key_operand rotated head step) lane key ==
         (if key < 7 then
            ObSeq.index rotated (key * 4096 + head * 512 + 256 + step * 16 + lane)
          else ObF.zero))
  = ()

inline_for_extraction noextract
let source_mma_score_tile_at
  (rotated:lseq bf16 (8 * 4096)) (head:ObShared.index 8)
  (row:ObShared.index 16) (col:ObShared.index 8)
  : Lemma
      (ObSeq.index (source_mma_score_tile rotated head) (row * 8 + col) ==
        ObChest.acc2 (source_mma_scores rotated head 8) row col)
  = ()

inline_for_extraction noextract
let source_qk_representation
  (rotated:lseq bf16 (8 * 4096)) (query:ObShared.heads 16) (key:ObShared.heads 8)
  : prop =
  (forall (token:ObShared.index 7) (head:ObShared.index 16) (lane:ObShared.index 128).
    v_approximates
      (ObSeq.index rotated
        (token * 4096 + (head / 2) * 512 + (head % 2) * 128 + lane))
      (query token head lane)) /\
  (forall (token:ObShared.index 7) (head:ObShared.index 8) (lane:ObShared.index 128).
    v_approximates (ObSeq.index rotated (token * 4096 + head * 512 + 256 + lane))
      (key token head lane))

inline_for_extraction noextract
let rec source_sum_n (n:nat) (f:nat -> GTot real)
  : GTot real (decreases n) =
  if n = 0 then 0.0R else source_sum_n (n - 1) f +. f (n - 1)

inline_for_extraction noextract
let rec source_sum_ext (n:nat) (f g:nat -> GTot real)
  : Lemma
      (requires forall (i:ObShared.index n). f i == g i)
      (ensures source_sum_n n f == source_sum_n n g)
      (decreases n) =
  if n > 0 then source_sum_ext (n - 1) f g

inline_for_extraction noextract
let rec source_sum_append (n m:nat) (f:nat -> GTot real)
  : Lemma
      (source_sum_n (n + m) f ==
        source_sum_n n f +. source_sum_n m (fun i -> f (n + i)))
      (decreases m) =
  if m > 0 then source_sum_append n (m - 1) f

inline_for_extraction noextract
let rec source_sum_shared (n:nat) (f:nat -> GTot real)
  : Lemma (source_sum_n n f == ObShared.sum n f)
      (decreases n) =
  if n > 0 then (
    source_sum_shared (n - 1) f;
    ObSum.sum_pop_right 0 n f
  )

inline_for_extraction noextract
let source_matrix_term
  (a:ObChest.chest2 real 16 16) (b:ObChest.chest2 real 16 8)
  (row:ObShared.index 16) (col:ObShared.index 8) (k:nat) : GTot real =
  if k < 16 then ObChest.acc2 a row k *. ObChest.acc2 b k col else 0.0R

inline_for_extraction noextract
let rec source_matmul_sum
  (a:ObChest.chest2 real 16 16) (b:ObChest.chest2 real 16 8)
  (row:ObShared.index 16) (col:ObShared.index 8) (count:nat{count <= 16})
  : Lemma
      (ObGemm.__matmul_single a b row col count ==
        source_sum_n count (source_matrix_term a b row col))
      (decreases count) =
  if count = 0 then ObGemm.matmul_zero_lemma a b row col
  else (
    source_matmul_sum a b row col (count - 1);
    ObGemm.matmul_single_lemma a b row col count
  )

inline_for_extraction noextract
let source_dot16 (a b:lseq real 16) (i:nat) : Tot real =
  if i < 16 then ObSeq.index a i *. ObSeq.index b i else 0.0R

(* Cellwise use of the reaccepted base's hardware approximation proof. *)
inline_for_extraction noextract
let source_mma_cell_approx
  (a:ObChest.chest2 bf16 16 16) (b:ObChest.chest2 bf16 16 8)
  (c:ObChest.chest2 ObF32.t 16 8)
  (row:ObShared.index 16) (col:ObShared.index 8)
  (ar br:lseq real 16) (cr:real)
  : Lemma
      (requires
        (forall (k:ObShared.index 16).
          ObChest.acc2 a row k %~ ObSeq.index ar k) /\
        (forall (k:ObShared.index 16).
          ObChest.acc2 b k col %~ ObSeq.index br k) /\
        ObChest.acc2 c row col %~ cr)
      (ensures ObChest.acc2 (ObMMA.emma c a b) row col %~
        (cr +. source_sum_n 16 (source_dot16 ar br))) =
  let ra = ObChest.mk2 (fun i k ->
    if i = row then ObSeq.index ar k else to_real (ObChest.acc2 a i k)) in
  let rb = ObChest.mk2 (fun k j ->
    if j = col then ObSeq.index br k else to_real (ObChest.acc2 b k j)) in
  let rc = ObChest.mk2 (fun i j ->
    if i = row && j = col then cr else to_real (ObChest.acc2 c i j)) in
  ObChest.lemma_to_real_chest_approximates a;
  ObChest.lemma_to_real_chest_approximates b;
  ObChest.lemma_to_real_chest_approximates c;
  assert (a %~ ra /\ b %~ rb /\ c %~ rc);
  ObMMA.emma_approx_lemma c a b rc ra rb;
  ObGemm.lemma_matmul_index ra rb row col;
  source_matmul_sum ra rb row col 16;
  source_sum_ext 16 (source_matrix_term ra rb row col) (source_dot16 ar br);
  assert (ObChest.acc2 (ObGemm.matplus rc (ObGemm.matmul ra rb)) row col ==
    cr +. source_sum_n 16 (source_dot16 ar br))

inline_for_extraction noextract
let source_score_term
  (query:ObShared.heads 16) (key:ObShared.heads 8)
  (head:ObShared.index 8) (row:ObShared.index 14) (col:ObShared.index 7)
  (k:nat) : GTot real =
  if k < 128 then query (row / 2) (2 * head + row % 2) k *. key col head k
  else 0.0R

inline_for_extraction noextract
let rec source_mma_scores_at
  (rotated:lseq bf16 (8 * 4096)) (query:ObShared.heads 16) (key:ObShared.heads 8)
  (head:ObShared.index 8) (row:ObShared.index 14) (col:ObShared.index 7)
  (steps:nat{steps <= 8})
  : Lemma
      (requires source_qk_representation rotated query key)
      (ensures ObChest.acc2 (source_mma_scores rotated head steps) row col %~
        source_sum_n (16 * steps) (source_score_term query key head row col))
      (decreases steps) =
  if steps = 0 then (
    let zero_ok = a0 #ObF32.t in ()
  ) else (
    let previous = steps - 1 in
    let ar = ObSeq.init_ghost 16 (fun i ->
      query (row / 2) (2 * head + row % 2) (16 * previous + i)) in
    let br = ObSeq.init_ghost 16 (fun i -> key col head (16 * previous + i)) in
    source_mma_scores_at rotated query key head row col previous;
    source_mma_cell_approx
      (source_query_operand rotated head previous)
      (source_key_operand rotated head previous)
      (source_mma_scores rotated head previous)
      row col ar br
      (source_sum_n (16 * previous) (source_score_term query key head row col));
    source_sum_ext 16 (source_dot16 ar br)
      (fun i -> source_score_term query key head row col (16 * previous + i));
    source_sum_append (16 * previous) 16 (source_score_term query key head row col)
  )

#push-options "--fuel 4 --ifuel 1 --z3rlimit 30"
inline_for_extraction noextract
let source_mma_tile_represents_dot
  (rotated:lseq bf16 (8 * 4096)) (query:ObShared.heads 16) (key:ObShared.heads 8)
  (head:ObShared.index 8) (row:ObShared.index 14) (col:ObShared.index 7)
  : Lemma
      (requires source_qk_representation rotated query key)
      (ensures
        v_approximates (ObSeq.index (source_mma_score_tile rotated head) (row * 8 + col))
          (ObShared.dot (query (row / 2) (2 * head + row % 2)) (key col head))) =
  source_mma_scores_at rotated query key head row col 8;
  source_mma_score_tile_at rotated head row col;
  source_sum_shared 128 (source_score_term query key head row col);
  sum_extensional 128 (source_score_term query key head row col)
    (fun i -> query (row / 2) (2 * head + row % 2) i *. key col head i);
  assert_norm (ObShared.dot (query (row / 2) (2 * head + row % 2)) (key col head) ==
    ObShared.sum 128
      (fun i -> query (row / 2) (2 * head + row % 2) i *. key col head i))
#pop-options

inline_for_extraction noextract
let source_mma_row_guard
  (rotated:lseq bf16 (8 * 4096)) (head:ObShared.index 8) (row:ObShared.index 16)
  : Lemma
      (not (NaN? (kind
        (source_maximum row (source_score_row (source_mma_score_tile rotated head) row))))) =
  source_maximum_non_nan row (source_score_row (source_mma_score_tile rotated head) row)

inline_for_extraction noextract
let source_log_two_nonzero ()
  : Lemma (ObExp.log 2.0R =!= 0.0R) =
  ObExp.exp_log 2.0R;
  ObExp.exp_base ()

inline_for_extraction noextract
let intended_attention_exp2_scale () : real =
  source_log_two_nonzero ();
  ObRoot.rsqrt 128.0R /. ObExp.log 2.0R

inline_for_extraction noextract
let source_exp2_shared_at_zero ()
  : Lemma
      (ObPow.exp2 (0.0R *. intended_attention_exp2_scale ()) ==
        ObExp.exp (0.0R *. ObRoot.rsqrt 128.0R)) =
  ObPow.pow_zero 2.0R;
  ObExp.exp_base ()

(* Historical rational arithmetic is a positive control, not a replacement
   for the source literal or for its required shared mathematical meaning. *)
inline_for_extraction noextract
let historical_rational_attention_value () : ObF32.t =
  ObF.div (ObF.of_int 4278775L) (ObF.of_int 33554432L)

inline_for_extraction noextract
let historical_rational_attention_represents ()
  : Lemma
      (v_approximates (historical_rational_attention_value ())
        (4278775.0R /. 33554432.0R)) =
  of_int_approx #ObF32.t 4278775L;
  of_int_approx #ObF32.t 33554432L;
  div_approx (ObF.of_int #ObF32.t 4278775L) (ObF.of_int #ObF32.t 33554432L)
    4278775.0R 33554432.0R

#push-options "--fuel 2 --ifuel 0 --z3rlimit 120"
inline_for_extraction noextract
let source_attention_ideal_interval ()
  : Lemma (
      1275174308245986.0R /. 10000000000000000.0R <. intended_attention_exp2_scale () /\
      intended_attention_exp2_scale () <. 1275174308245988.0R /. 10000000000000000.0R) =
  ObRoot.sqrt_positive 128.0R;
  ObRoot.sqrt_square 128.0R;
  ObExp.log_two_decimal_bounds ();
  ObExp.log_two_positive ();
  let root = ObRoot.sqrt 128.0R in
  let logarithm = ObExp.log 2.0R in
  assert (11313708498984760.0R /. 1000000000000000.0R <. root);
  assert (root <. 11313708498984762.0R /. 1000000000000000.0R);
  assert (
    (11313708498984760.0R /. 1000000000000000.0R) *.
    (69314718055994530.0R /. 100000000000000000.0R) <. root *. logarithm);
  assert (
    root *. logarithm <.
    (11313708498984762.0R /. 1000000000000000.0R) *.
    (69314718055994532.0R /. 100000000000000000.0R))

inline_for_extraction noextract
let source_attention_literal_represents_ideal ()
  : Lemma (v_approximates (source_attention_literal ()) (intended_attention_exp2_scale ())) =
  source_attention_ideal_interval ();
  assert_norm (
    Kuiper.Float32.Literals.common_normal_cell "0.12751743197441101074"
      1275174308245986 10000000000000000
      1275174308245988 10000000000000000
      8557550 (-26));
  ObF32.literal_interval_approx "0.12751743197441101074"
    1275174308245986 10000000000000000
    1275174308245988 10000000000000000
    8557550 (-26) (intended_attention_exp2_scale ())

inline_for_extraction noextract
let source_attention_literal_represents_rational ()
  : Lemma (v_approximates (source_attention_literal ()) (4278775.0R /. 33554432.0R)) =
  assert_norm (
    Kuiper.Float32.Literals.common_normal_cell "0.12751743197441101074"
      4278775 33554432 4278775 33554432 8557550 (-26));
  ObF32.literal_interval_approx "0.12751743197441101074"
    4278775 33554432 4278775 33554432 8557550 (-26)
    (4278775.0R /. 33554432.0R)

inline_for_extraction noextract
let source_attention_change_of_base (difference:real)
  : Lemma (ObPow.exp2 (difference *. intended_attention_exp2_scale ()) ==
      ObExp.exp (difference *. ObRoot.rsqrt 128.0R)) =
  ObExp.log_two_positive ();
  ObExp.exp2_as_exp (difference *. intended_attention_exp2_scale ())

inline_for_extraction noextract
let source_linear_exponent
  (value scaled result:ObF32.t) (score maximum coefficient:real)
  : Lemma
      (requires
        v_approximates value score /\
        v_approximates scaled (maximum *. coefficient) /\
        v_approximates (source_attention_literal ()) coefficient /\
        source_exponent_contract value scaled result)
      (ensures v_approximates result (ObPow.exp2 ((score -. maximum) *. coefficient))) =
  assert (v_approximates result
    (ObPow.exp2 (score *. coefficient -. maximum *. coefficient)))

inline_for_extraction noextract
let source_active_numerator_refines
  (value scaled result:ObF32.t) (score maximum:real)
  : Lemma
      (requires
        v_approximates value score /\
        v_approximates scaled (maximum *. intended_attention_exp2_scale ()) /\
        source_exponent_contract value scaled result)
      (ensures v_approximates result (ObExp.exp ((score -. maximum) *. ObRoot.rsqrt 128.0R))) =
  source_attention_literal_represents_ideal ();
  source_linear_exponent value scaled result score maximum (intended_attention_exp2_scale ());
  source_attention_change_of_base (score -. maximum)
#pop-options

inline_for_extraction noextract
let source_sentinel_comparison_is_equality (value:ObF32.t)
  : Lemma
      (requires ieee_eq value (negative_infinity ()))
      (ensures value == negative_infinity ()) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  ObF.eq_spec value (negative_infinity ());
  if ObF.is_zero (negative_infinity ()) then
    ObF.is_zero_spec (negative_infinity ())

(* A positive control for the literal-zero branch, not the scaled numerator. *)
inline_for_extraction noextract
let source_sentinel_unscaled_exponential (value:ObF32.t) (score:real)
  : Lemma
      (requires ieee_eq value (negative_infinity ()) /\ v_approximates value score)
      (ensures v_approximates (ObF.zero #ObF32.t) (ObExp.exp score)) =
  source_sentinel_comparison_is_equality value;
  exp_approx value score;
  ObF32.exp_special_values value

inline_for_extraction noextract
let source_shared_shifted_numerator
  (query:ObShared.heads 16) (key:ObShared.heads 8)
  (head:ObShared.index 8) (row:ObShared.index 14) (col:ObShared.index 7)
  (maximum:real) : GTot real =
  ObExp.exp
    ((ObShared.dot (query (row / 2) (2 * head + row % 2)) (key col head) -.
      maximum) *. ObRoot.rsqrt 128.0R)

inline_for_extraction noextract
let source_mma_live_numerator_contract
  (rotated:lseq bf16 (8 * 4096)) (query:ObShared.heads 16) (key:ObShared.heads 8)
  (head:ObShared.index 8) (row:ObShared.index 14)
  (col:ObShared.index 7{col <= row / 2}) (result:ObF32.t) : prop =
  let tile = source_mma_score_tile rotated head in
  let value = ObSeq.index tile (row * 8 + col) in
  let maximum = source_maximum row (source_score_row tile row) in
  (ieee_eq value (negative_infinity ()) ==>
    result == (ObF.zero #ObF32.t) /\
    v_approximates result 0.0R /\
    v_approximates result
      (ObExp.exp (ObShared.dot (query (row / 2) (2 * head + row % 2)) (key col head)))) /\
  (not (ieee_eq value (negative_infinity ())) ==>
    v_approximates result
      (source_shared_shifted_numerator query key head row col (to_real maximum)))

(* This consumes the modeled score tile; it does not create MMA ownership. *)
inline_for_extraction noextract
fn source_mma_live_numerator
  (tile:larray ObF32.t 128) (row:szlt 14) (col:szlt 7{col <= row / 2})
  (#rotated:erased (lseq bf16 (8 * 4096)))
  (#query:erased (ObShared.heads 16)) (#key:erased (ObShared.heads 8))
  (#head:erased (ObShared.index 8)) (#f:perm)
  preserves gpu ** tile |-> Frac f (source_mma_score_tile rotated head)
  requires pure (source_qk_representation rotated query key)
  returns result:ObF32.t
  ensures pure (source_mma_live_numerator_contract rotated query key head row col result)
{
  let maximum = source_maximum_from_tile tile row
    #(hide (source_mma_score_tile rotated head)) #f;
  source_attention_literal_represents_ideal ();
  to_real_ok maximum;
  let scaled = ObF32.mul_rn_ftz maximum (source_attention_literal ());
  assert pure (v_approximates scaled
    (to_real maximum *. intended_attention_exp2_scale ()));
  let value = source_read_score_tile tile row col
    #(hide (source_mma_score_tile rotated head)) #f;
  source_score_row_at (source_mma_score_tile rotated head) row col;
  source_mma_tile_represents_dot rotated query key head row col;
  let result = source_exponent_from_scaled value scaled;
  if ieee_eq value (negative_infinity ()) {
    source_sentinel_unscaled_exponential value
      (ObShared.dot ((reveal query) (row / 2) (2 * (reveal head) + row % 2))
        ((reveal key) col (reveal head)));
    ()
  } else {
    source_active_numerator_refines value scaled result
      (ObShared.dot ((reveal query) (row / 2) (2 * (reveal head) + row % 2))
        ((reveal key) col (reveal head)))
      (to_real maximum);
    ()
  };
  result
}

inline_for_extraction noextract
let source_rescale_tail_is_exponent (scaled rescale:ObF32.t)
  : Lemma
      (requires source_rescale_tail_contract (ObF.sub ObF.zero scaled) rescale)
      (ensures source_exponent_contract (negative_infinity ()) scaled rescale) =
  let from_argument (argument:ObF32.t)
    : Lemma
        (requires
          (forall (vr ar nr:real).
            v_approximates (negative_infinity ()) vr /\
            v_approximates (source_attention_literal ()) ar /\
            v_approximates (ObF.sub ObF.zero scaled) nr ==>
            v_approximates argument (vr *. ar +. nr)) /\
          (forall (r:real).
            v_approximates argument r ==> v_approximates rescale (ObPow.exp2 r)))
        (ensures source_exponent_contract (negative_infinity ()) scaled rescale) =
    source_exponent_chain (negative_infinity ()) scaled argument rescale
  in
  FStar.Classical.forall_intro (FStar.Classical.move_requires from_argument)

(* Reuse the rescale actually computed later by this same source row. *)
inline_for_extraction noextract
let source_branch_numerator_after_rescale
  (value scaled exponent rescale:ObF32.t) (score maximum:real)
  : Lemma
      (requires
        v_approximates value score /\
        v_approximates scaled (maximum *. intended_attention_exp2_scale ()) /\
        source_branch_exponent_contract value scaled exponent /\
        source_rescale_tail_contract (ObF.sub ObF.zero scaled) rescale /\
        rescale == (ObF.zero #ObF32.t))
      (ensures v_approximates exponent
        (ObExp.exp ((score -. maximum) *. ObRoot.rsqrt 128.0R))) =
  if ieee_eq value (negative_infinity ()) then (
    source_sentinel_comparison_is_equality value;
    source_rescale_tail_is_exponent scaled rescale;
    source_active_numerator_refines (negative_infinity ()) scaled rescale score maximum
  ) else
    source_active_numerator_refines value scaled exponent score maximum

inline_for_extraction noextract
let source_live_entry_after_rescale
  (values:lseq ObF32.t 128) (row:ObShared.index 14)
  (col:ObShared.index 7{col <= row / 2})
  (exponent rescale:ObF32.t) (score maximum:real)
  : Lemma
      (requires
        source_tile_entry_contract values row col exponent rescale /\
        v_approximates (ObSeq.index values (row * 8 + col)) score /\
        v_approximates (source_maximum row (source_score_row values row)) maximum /\
        not (ieee_eq (source_maximum row (source_score_row values row)) (negative_infinity ())))
      (ensures v_approximates exponent
        (ObExp.exp ((score -. maximum) *. ObRoot.rsqrt 128.0R))) =
  source_score_row_at values row col;
  source_attention_literal_represents_ideal ();
  let from_scaled (scaled:ObF32.t)
    : Lemma
        (requires
          v_approximates scaled (maximum *. intended_attention_exp2_scale ()) /\
          source_branch_exponent_contract (ObSeq.index values (row * 8 + col)) scaled exponent /\
          source_rescale_tail_contract (ObF.sub ObF.zero scaled) rescale /\
          rescale == (ObF.zero #ObF32.t))
        (ensures v_approximates exponent
          (ObExp.exp ((score -. maximum) *. ObRoot.rsqrt 128.0R))) =
    source_branch_numerator_after_rescale
      (ObSeq.index values (row * 8 + col)) scaled exponent rescale score maximum
  in
  FStar.Classical.forall_intro (FStar.Classical.move_requires from_scaled)

inline_for_extraction noextract
let source_mma_entry_after_rescale
  (rotated:lseq bf16 (8 * 4096)) (query:ObShared.heads 16) (key:ObShared.heads 8)
  (head:ObShared.index 8) (row:ObShared.index 14)
  (col:ObShared.index 7{col <= row / 2}) (exponent rescale:ObF32.t)
  : Lemma
      (requires
        source_qk_representation rotated query key /\
        source_tile_entry_contract (source_mma_score_tile rotated head) row col exponent rescale)
      (ensures
        (let maximum = source_maximum row (source_score_row (source_mma_score_tile rotated head) row) in
        not (ieee_eq maximum (negative_infinity ())) ==>
        v_approximates exponent (source_shared_shifted_numerator query key head row col (to_real maximum)))) =
  let tile = source_mma_score_tile rotated head in
  let maximum = source_maximum row (source_score_row tile row) in
  if not (ieee_eq maximum (negative_infinity ())) then (
    source_mma_tile_represents_dot rotated query key head row col;
    to_real_ok maximum;
    source_live_entry_after_rescale tile row col exponent rescale
      (ObShared.dot (query (row / 2) (2 * head + row % 2)) (key col head))
      (to_real maximum)
  )

inline_for_extraction noextract
let rec source_all_sentinel_part_maximum
  (row:ObShared.index 16) (scores:lseq ObF32.t 8)
  (part:ObShared.index 4) (steps:nat{steps <= 16})
  : Lemma
      (requires forall (col:ObShared.index 64).
        source_row_score row scores col == negative_infinity ())
      (ensures source_part_maximum row scores part steps == negative_infinity ())
      (decreases steps) =
  if steps > 0 then (
    source_all_sentinel_part_maximum row scores part (steps - 1);
    fmax_negative_infinity ()
  )

inline_for_extraction noextract
let source_live_singleton_sentinel_maximum
  (values:lseq ObF32.t 128) (row:ObShared.index 2)
  : Lemma
      (requires ieee_eq (ObSeq.index values (row * 8)) (negative_infinity ()))
      (ensures source_maximum row (source_score_row values row) == negative_infinity ()) =
  source_sentinel_comparison_is_equality (ObSeq.index values (row * 8));
  source_score_row_at values row 0;
  assert (forall (col:ObShared.index 64).
    source_row_score row (source_score_row values row) col == negative_infinity ());
  source_all_sentinel_part_maximum row (source_score_row values row) 0 16;
  source_all_sentinel_part_maximum row (source_score_row values row) 1 16;
  source_all_sentinel_part_maximum row (source_score_row values row) 2 16;
  source_all_sentinel_part_maximum row (source_score_row values row) 3 16;
  fmax_negative_infinity ()

inline_for_extraction noextract
let source_shared_singleton_probability (scores:ObShared.vector 1)
  : Lemma (ObShared.prefix_softmax 0 scores 0 == 1.0R) =
  ObSum.sum_pop_right 0 1 (fun i -> ObExp.exp (scores i));
  shared_softmax_at 1 scores 0

inline_for_extraction noextract
let source_unit_rescale_changes_singleton (numerator:real{numerator >. 0.0R})
  : Lemma (numerator /. (1.0R +. numerator) =!= 1.0R) =
  ()

inline_for_extraction noextract
let real_mul_div (a b:real) (denominator:real{denominator >. 0.0R})
  : Lemma ((a *. b) /. denominator == (a /. denominator) *. b) =
  ()

inline_for_extraction noextract
let source_shifted_prefix_numerators
  (token:ObShared.index 7) (scores:ObShared.vector (token + 1)) (shift:real)
  : GTot (ObShared.vector 7) =
  fun key -> if key <= token then ObExp.exp (scores key -. shift) else 0.0R

#push-options "--fuel 4 --ifuel 1 --z3rlimit 30"
inline_for_extraction noextract
let normalized_dense_prefix
  (token:ObShared.index 7) (scores:ObShared.vector (token + 1))
  (values:ObShared.vector 7) (shift:real)
  : Lemma
      (let numerators = source_shifted_prefix_numerators token scores shift in
       let denominator = ObShared.sum (token + 1) numerators in
       denominator >. 0.0R /\
       ObShared.sum 64 (fun key ->
         padded_real numerators key *. padded_real values key) /. denominator ==
       ObShared.sum (token + 1) (fun key ->
         ObShared.prefix_softmax token scores key *. values key)) =
  let numerators = source_shifted_prefix_numerators token scores shift in
  let denominator = ObShared.sum (token + 1) numerators in
  let terms : ObShared.vector (token + 1) = fun key -> numerators key *. values key in
  ObShared.sum_positive (token + 1) numerators;
  dense_value_equals_prefix token numerators values;
  sum_extensional (token + 1) (fun key -> numerators key *. values key) terms;
  sum_divide (token + 1) terms denominator;
  let at (key:ObShared.index (token + 1))
    : Lemma (terms key /. denominator ==
        ObShared.prefix_softmax token scores key *. values key) =
    prefix_normalize_shifted token scores shift key;
    sum_extensional (token + 1) numerators (fun i -> ObExp.exp (scores i -. shift));
    assert (numerators key /. denominator == ObShared.prefix_softmax token scores key);
    real_mul_div (numerators key) (values key) denominator
  in
  FStar.Classical.forall_intro at;
  sum_extensional (token + 1)
    (fun key -> terms key /. denominator)
    (fun key -> ObShared.prefix_softmax token scores key *. values key);
  sum_extensional (token + 1)
    (fun i -> terms i /. denominator)
    (fun key -> terms key /. denominator);
  assert (ObShared.sum (token + 1) terms /. denominator ==
    ObShared.sum (token + 1) (fun key ->
      ObShared.prefix_softmax token scores key *. values key));
  sum_extensional 64
    (fun key -> padded_real numerators key *. padded_real values key)
    (fun key ->
      padded_real (source_shifted_prefix_numerators token scores shift) key *.
      padded_real values key)
#pop-options

inline_for_extraction noextract
fn source_denominator_combine (rescale s0 s1 s2 s3:ObF32.t)
  preserves gpu
  returns result:ObF32.t
  ensures pure (
    forall (r a b c d:real).
      v_approximates rescale r /\
      v_approximates s0 a /\ v_approximates s1 b /\
      v_approximates s2 c /\ v_approximates s3 d ==>
      v_approximates result (r +. ((a +. b) +. (c +. d))))
{
  let s01 = ObF32.add_rn_ftz s0 s1;
  let s23 = ObF32.add_rn_ftz s2 s3;
  let sum = ObF32.add_rn_ftz s01 s23;
  ObF32.add_rn_ftz rescale sum
}

inline_for_extraction noextract
let source_normalization_chain
  (numerator denominator inverse product:ObF32.t)
  : Lemma
      (requires
        (forall (d:real{d =!= 0.0R}).
          v_approximates denominator d ==> v_approximates inverse (1.0R /. d)) /\
        (forall (n i:real).
          v_approximates numerator n /\ v_approximates inverse i ==>
          v_approximates product (n *. i)))
      (ensures
        forall (n:real) (d:real{d =!= 0.0R}).
          v_approximates numerator n /\ v_approximates denominator d ==>
          v_approximates (ObCasts.fcast #ObF32.t #bf16 product) (n /. d)) =
  let at (n:real) (d:real{d =!= 0.0R})
    : Lemma
        (requires v_approximates numerator n /\ v_approximates denominator d)
        (ensures v_approximates (ObCasts.fcast #ObF32.t #bf16 product) (n /. d)) =
    assert (v_approximates inverse (1.0R /. d));
    assert (v_approximates product (n *. (1.0R /. d)));
    assert (n *. (1.0R /. d) == n /. d);
    assert (v_approximates product (n /. d));
    ObCasts.fcast_approx #ObF32.t #bf16 product (n /. d)
  in
  FStar.Classical.forall_intro_2 (FStar.Classical.move_requires_2 at)

inline_for_extraction noextract
fn source_normalize_value (numerator denominator:ObF32.t)
  preserves gpu
  returns result:bf16
  ensures pure (
    forall (n:real) (d:real{d =!= 0.0R}).
      v_approximates numerator n /\ v_approximates denominator d ==>
      v_approximates result (n /. d))
{
  let inverse = ObF32.rcp_approx_ftz denominator;
  let product = ObF32.mul_rn_ftz numerator inverse;
  source_normalization_chain numerator denominator inverse product;
  ObCasts.fcast #ObF32.t #bf16 product
}

inline_for_extraction noextract
let source_attention_tail_contract
  (numerator rescale s0 s1 s2 s3:ObF32.t) (result:bf16) : prop =
  exists (denominator:ObF32.t).
    (forall (r a b c d:real).
      v_approximates rescale r /\
      v_approximates s0 a /\ v_approximates s1 b /\
      v_approximates s2 c /\ v_approximates s3 d ==>
      v_approximates denominator (r +. ((a +. b) +. (c +. d)))) /\
    (forall (n:real) (d:real{d =!= 0.0R}).
      v_approximates numerator n /\ v_approximates denominator d ==>
      v_approximates result (n /. d))

inline_for_extraction noextract
fn store_source_attention_tail
  (numerator rescale s0 s1 s2 s3:ObF32.t)
  (output:gpu_ref bf16) (#old:erased bf16)
  preserves gpu
  requires output |-> old
  ensures exists* (result:bf16).
    output |-> result **
    pure (source_attention_tail_contract numerator rescale s0 s1 s2 s3 result)
{
  let denominator = source_denominator_combine rescale s0 s1 s2 s3;
  let result = source_normalize_value numerator denominator;
  output := result;
}

fn run_source_attention_tail
  (numerator rescale s0 s1 s2 s3:ObF32.t)
  (output:gpu_ref bf16) (#old:erased bf16)
  preserves cpu
  requires on gpu_loc (output |-> old)
  ensures exists* (result:bf16).
    on gpu_loc (output |-> result) **
    pure (source_attention_tail_contract numerator rescale s0 s1 s2 s3 result)
{
  launch_kernel_1 (fun () ->
    store_source_attention_tail numerator rescale s0 s1 s2 s3 output #old);
}

inline_for_extraction noextract
let bilateral_finite_not_sentinel (value:ObF32.t)
  : Lemma
      (requires Finite? (kind value))
      (ensures not (ieee_eq value (negative_infinity ()))) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  if ieee_eq value (negative_infinity ()) then
    source_sentinel_comparison_is_equality value

inline_for_extraction noextract
let bilateral_negative_infinity_lt_finite (value:ObF32.t)
  : Lemma
      (requires Finite? (kind value))
      (ensures ObF.lt (negative_infinity ()) value) =
  let infinity : ObF32.t = ObF.infinity in
  let negative = negative_infinity () in
  let double_negative = ObF.sub ObF.zero negative in
  let negated = ObF.sub ObF.zero value in
  ObF.neg_kind infinity;
  ObF.neg_kind negative;
  ObF.neg_kind value;
  ObF.neg_neg infinity;
  ObF.eq_spec double_negative infinity;
  if ObF.is_zero infinity then ObF.is_zero_spec infinity;
  assert (double_negative == infinity);
  ObF.infinity_val_spec negated;
  ObF.negate_lt_is_lte infinity negated;
  ObF.lt_neg_flip value negative;
  assert (not (ObF.lt value negative));
  bilateral_finite_not_sentinel value;
  ObF.lte_is_lt_or_eq value negative;
  ObF.negate_lt_is_lte negative value

inline_for_extraction noextract
let bilateral_equal_finite (left right:ObF32.t)
  : Lemma
      (requires ieee_eq left right /\ Finite? (kind right))
      (ensures Finite? (kind left)) =
  ObF.eq_spec left right;
  if ObF.is_zero left then ObF.is_zero_spec left

inline_for_extraction noextract
let bilateral_finite_or_sentinel (value:ObF32.t) : prop =
  Finite? (kind value) \/ value == negative_infinity ()

inline_for_extraction noextract
let bilateral_fmax_classification (left right:ObF32.t)
  : Lemma
      (requires bilateral_finite_or_sentinel left /\ bilateral_finite_or_sentinel right)
      (ensures
        bilateral_finite_or_sentinel (ObF.fmax left right) /\
        ((Finite? (kind left) \/ Finite? (kind right)) ==>
          Finite? (kind (ObF.fmax left right)))) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  if Finite? (kind left) then (
    bilateral_negative_infinity_lt_finite left;
    ObF.negate_lt_is_lte left (negative_infinity ());
    ObF.lte_is_lt_or_eq (negative_infinity ()) left
  );
  if Finite? (kind right) then bilateral_negative_infinity_lt_finite right;
  ObF.fmax_spec left right;
  let selected = if ObF.lt left right then right else left in
  if Finite? (kind selected) then
    bilateral_equal_finite (ObF.fmax left right) selected
  else
    source_sentinel_comparison_is_equality (ObF.fmax left right)

inline_for_extraction noextract
let bilateral_live_row_finite
  (row:ObShared.index 14) (scores:lseq ObF32.t 8) : prop =
  forall (col:ObShared.index 7{col <= row / 2}).
    Finite? (kind (ObSeq.index scores col))

inline_for_extraction noextract
let bilateral_row_score_classification
  (row:ObShared.index 14) (scores:lseq ObF32.t 8) (col:ObShared.index 64)
  : Lemma
      (requires bilateral_live_row_finite row scores)
      (ensures bilateral_finite_or_sentinel (source_row_score row scores col)) =
  ()

inline_for_extraction noextract
let rec bilateral_part_maximum_classification
  (row:ObShared.index 14) (scores:lseq ObF32.t 8)
  (part:ObShared.index 4) (steps:nat{steps <= 16})
  : Lemma
      (requires bilateral_live_row_finite row scores)
      (ensures bilateral_finite_or_sentinel (source_part_maximum row scores part steps))
      (decreases steps) =
  if steps > 0 then (
    bilateral_part_maximum_classification row scores part (steps - 1);
    bilateral_row_score_classification row scores (source_column part (steps - 1));
    bilateral_fmax_classification (source_part_maximum row scores part (steps - 1))
      (source_row_score row scores (source_column part (steps - 1)))
  )

inline_for_extraction noextract
let rec bilateral_part_zero_finite
  (row:ObShared.index 14) (scores:lseq ObF32.t 8) (steps:pos{steps <= 16})
  : Lemma
      (requires bilateral_live_row_finite row scores)
      (ensures Finite? (kind (source_part_maximum row scores 0 steps)))
      (decreases steps) =
  if steps = 1 then (
    assert (source_live row 0);
    bilateral_fmax_classification (negative_infinity ()) (ObSeq.index scores 0)
  ) else (
    bilateral_part_zero_finite row scores (steps - 1);
    bilateral_row_score_classification row scores (source_column 0 (steps - 1));
    bilateral_fmax_classification (source_part_maximum row scores 0 (steps - 1))
      (source_row_score row scores (source_column 0 (steps - 1)))
  )

inline_for_extraction noextract
let bilateral_live_maximum_finite
  (row:ObShared.index 14) (scores:lseq ObF32.t 8)
  : Lemma
      (requires bilateral_live_row_finite row scores)
      (ensures
        Finite? (kind (source_maximum row scores)) /\
        not (ieee_eq (source_maximum row scores) (negative_infinity ()))) =
  bilateral_part_zero_finite row scores 16;
  bilateral_part_maximum_classification row scores 1 16;
  bilateral_part_maximum_classification row scores 2 16;
  bilateral_part_maximum_classification row scores 3 16;
  let m0 = source_part_maximum row scores 0 16 in
  let m1 = source_part_maximum row scores 1 16 in
  let m2 = source_part_maximum row scores 2 16 in
  let m3 = source_part_maximum row scores 3 16 in
  bilateral_fmax_classification m0 m1;
  bilateral_fmax_classification m2 m3;
  bilateral_fmax_classification (ObF.fmax m0 m1) (ObF.fmax m2 m3);
  bilateral_finite_not_sentinel (source_maximum row scores)

inline_for_extraction noextract
let bilateral_mma_maximum_exclusion
  (rotated:lseq bf16 (8 * 4096)) (head:ObShared.index 8) (row:ObShared.index 14)
  : Lemma
      (requires bilateral_mma_live_scores_finite rotated head)
      (ensures
        Finite? (kind (source_maximum row (source_score_row (source_mma_score_tile rotated head) row))) /\
        not (ieee_eq
          (source_maximum row (source_score_row (source_mma_score_tile rotated head) row))
          (negative_infinity ()))) =
  let at (col:ObShared.index 7{col <= row / 2})
    : Lemma
        (Finite? (kind (ObSeq.index
          (source_score_row (source_mma_score_tile rotated head) row) col))) =
    source_score_row_at (source_mma_score_tile rotated head) row col;
    source_mma_score_tile_at rotated head row col
  in
  FStar.Classical.forall_intro at;
  bilateral_live_maximum_finite row (source_score_row (source_mma_score_tile rotated head) row)

inline_for_extraction noextract
let bilateral_mma_entry_refines
  (rotated:lseq bf16 (8 * 4096)) (query:ObShared.heads 16) (key:ObShared.heads 8)
  (head:ObShared.index 8) (row:ObShared.index 14)
  (col:ObShared.index 7{col <= row / 2}) (exponent rescale:ObF32.t)
  : Lemma
      (requires
        source_qk_representation rotated query key /\
        bilateral_mma_live_scores_finite rotated head /\
        source_tile_entry_contract (source_mma_score_tile rotated head) row col exponent rescale)
      (ensures
        rescale == (ObF.zero #ObF32.t) /\ v_approximates rescale 0.0R /\
        v_approximates exponent
          (source_shared_shifted_numerator query key head row col
            (to_real (source_maximum row (source_score_row (source_mma_score_tile rotated head) row))))) =
  bilateral_mma_maximum_exclusion rotated head row;
  source_mma_entry_after_rescale rotated query key head row col exponent rescale

#push-options "--fuel 2 --ifuel 1 --z3rlimit 240"
inline_for_extraction noextract
fn bilateral_source_mma_entry
  (tile:larray ObF32.t 128) (row:szlt 14) (col:szlt 7{col <= row / 2})
  (probability:gpu_ref bf16) (rescale_output:gpu_ref ObF32.t)
  (#rotated:erased (lseq bf16 (8 * 4096)))
  (#query:erased (ObShared.heads 16)) (#key:erased (ObShared.heads 8))
  (#head:erased (ObShared.index 8)) (#f:perm)
  (#old_probability:erased bf16) (#old_rescale:erased ObF32.t)
  preserves gpu ** tile |-> Frac f (source_mma_score_tile rotated head)
  requires probability |-> old_probability ** rescale_output |-> old_rescale **
    pure (source_qk_representation rotated query key /\ bilateral_mma_live_scores_finite rotated head)
  ensures exists* (exponent:ObF32.t).
    probability |-> ObCasts.fcast #ObF32.t #bf16 exponent **
    rescale_output |-> (ObF.zero #ObF32.t) **
    pure (v_approximates exponent
      (source_shared_shifted_numerator query key head row col
        (to_real (source_maximum row (source_score_row (source_mma_score_tile rotated head) row)))))
{
  source_score_tile_entry tile row col probability rescale_output
    #(hide (source_mma_score_tile rotated head)) #f #old_probability #old_rescale;
  with exponent rescale.
  assert probability |-> ObCasts.fcast #ObF32.t #bf16 exponent ** rescale_output |-> rescale;
  bilateral_mma_entry_refines rotated query key head row col exponent rescale;
  assert pure (rescale == (ObF.zero #ObF32.t));
  assert pure (v_approximates exponent
    (source_shared_shifted_numerator query key head row col
      (to_real (source_maximum row (source_score_row (source_mma_score_tile rotated head) row)))));
}
#pop-options

inline_for_extraction noextract
let bilateral_finite_control (row:ObShared.index 14)
  : Lemma
      (let scores = ObSeq.init_ghost 8 (fun _ -> (ObF.zero #ObF32.t)) in
       bilateral_live_row_finite row scores /\
       Finite? (kind (source_maximum row scores))) =
  bilateral_live_maximum_finite row (ObSeq.init_ghost 8 (fun _ -> (ObF.zero #ObF32.t)))

inline_for_extraction noextract
let bilateral_overflow_control_rejected
  (row:ObShared.index 14) (scores:lseq ObF32.t 8)
  : Lemma
      (requires ieee_eq (ObSeq.index scores 0) (negative_infinity ()))
      (ensures not (bilateral_live_row_finite row scores)) =
  source_sentinel_comparison_is_equality (ObSeq.index scores 0);
  ObF.neg_kind (ObF.infinity #ObF32.t)

inline_for_extraction noextract
let bilateral_empty_rows_unchanged
  (row:ObShared.index 16{14 <= row}) (scores:lseq ObF32.t 8)
  : Lemma (source_maximum row scores == negative_infinity ()) =
  source_maximum_empty row scores

module Execution = Pulse.Lib.MonotonicGhostRef

[@@erasable]
noeq
type execution_state = {
  native: invocation_inputs;
  phase: nat;
  boundary: option cache_observation;
  result: option (bf16_values hidden_elements)
}

inline_for_extraction noextract
let execution_advances (a b:execution_state) : prop =
  a.native == b.native /\ a.phase <= b.phase /\
  (Some? a.boundary ==> a.boundary == b.boundary) /\
  (Some? a.result ==> a.result == b.result)

inline_for_extraction noextract
let execution_order : FStar.Preorder.preorder execution_state = execution_advances

inline_for_extraction noextract
let invocation (_:invocation_inputs) = Execution.mref execution_order

inline_for_extraction noextract
let started_state (inputs:invocation_inputs) : execution_state =
  { native = inputs; phase = 0; boundary = None; result = None }

inline_for_extraction noextract
let produced_state (inputs:invocation_inputs) (observation:cache_observation)
  : execution_state =
  { native = inputs; phase = 1; boundary = Some observation; result = None }

inline_for_extraction noextract
let consumed_state (inputs:invocation_inputs) (observation:cache_observation)
  : execution_state =
  { native = inputs; phase = 2; boundary = Some observation; result = None }

inline_for_extraction noextract
let finished_state (inputs:invocation_inputs) (observation:cache_observation)
  (result:bf16_values hidden_elements) : execution_state =
  { native = inputs; phase = 3; boundary = Some observation; result = Some result }

let invocation_pending (#inputs:invocation_inputs) ([@@@mkey] run:invocation inputs) : slprop =
  Execution.pts_to run #1.0R (started_state inputs)

let invocation_consumed (#inputs:invocation_inputs)
  ([@@@mkey] run:invocation inputs) (observation:cache_observation) : slprop =
  Execution.snapshot run (consumed_state inputs observation)

let invocation_result (#inputs:invocation_inputs)
  ([@@@mkey] run:invocation inputs) (observation:cache_observation)
  (result:bf16_values hidden_elements) : slprop =
  Execution.snapshot run (produced_state inputs observation) **
  Execution.snapshot run (finished_state inputs observation result)

inline_for_extraction noextract
let producer_handoff (#inputs:invocation_inputs)
  ([@@@mkey] run:invocation inputs) (observation:cache_observation) : slprop =
  Execution.pts_to run #1.0R (produced_state inputs observation) **
  Execution.snapshot run (produced_state inputs observation)

inline_for_extraction noextract
let consumer_handoff (#inputs:invocation_inputs)
  ([@@@mkey] run:invocation inputs) (observation:cache_observation) : slprop =
  Execution.pts_to run #1.0R (consumed_state inputs observation) **
  Execution.snapshot run (produced_state inputs observation)

ghost
fn begin_invocation (#inputs:erased invocation_inputs)
  returns run:erased (invocation inputs)
  ensures invocation_pending #inputs run
{
  let run = Execution.alloc #execution_state #execution_order (started_state inputs);
  fold invocation_pending #inputs run;
  hide run
}

ghost
fn invocation_observations_agree
  (#inputs:erased invocation_inputs) (#run:erased (invocation inputs))
  (#actual #observed:erased cache_observation)
  (#result:erased (bf16_values hidden_elements))
  preserves invocation_consumed #inputs run actual ** invocation_result #inputs run observed result
  ensures pure (actual == observed)
{
  unfold invocation_consumed #inputs run actual;
  unfold invocation_result #inputs run observed result;
  Execution.snapshots_related run
    #(consumed_state inputs actual) #(finished_state inputs observed result);
  fold invocation_consumed #inputs run actual;
  fold invocation_result #inputs run observed result;
}

ghost
fn execution_no_overflow_refines_shared
  (#inputs:erased invocation_inputs) (#run:erased (invocation inputs))
  (#input:erased ObShared.prefill_input) (#padding:erased (ObShared.vector 1024))
  (#result:erased (bf16_values hidden_elements))
  (#actual #observed:erased cache_observation)
  preserves
    completed_execution #inputs run input padding result observed **
    numerical_premises #inputs run actual
  ensures pure (required_postcondition input padding result observed)
{
  unfold completed_execution #inputs run input padding result observed;
  unfold numerical_premises #inputs run actual;
  invocation_observations_agree #inputs #run #actual #observed #result;
  assert pure (required_postcondition input padding result observed);
  fold completed_execution #inputs run input padding result observed;
  fold numerical_premises #inputs run actual;
}

inline_for_extraction noextract
ghost
fn snapshot_computed_cache
  (rotated:bf16_array (8 * 4096))
  (#values:erased (bf16_values (8 * 4096))) (#f:perm)
  preserves on gpu_loc (rotated |-> Frac f values)
  returns observation:erased cache_observation
  ensures pure ((reveal observation).rotated_buffer == rotated /\
    (reveal observation).rotated_values == reveal values)
{
  hide { rotated_buffer = rotated; rotated_values = reveal values }
}

module SZ = Kuiper.SizeT
module Shared = KPlay.SharedSpec
module F32 = Kuiper.Float32
module BF16 = Kuiper.BFloat16
module Casts = Kuiper.Float.Casts
module Seq = FStar.Seq
module I64 = FStar.Int64
module U64 = FStar.UInt64

#push-options "--fuel 4 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let sum_halves (n:nat) (f:nat -> GTot real)
  : Lemma
      (source_sum_n (2 * n) f ==
        source_sum_n n (fun i -> f i +. f (n + i)))
  =
  source_sum_append n n f;
  let rec combine (k:nat{k <= n})
    : Lemma
        (source_sum_n k f +. source_sum_n k (fun i -> f (n + i)) ==
          source_sum_n k (fun i -> f i +. f (n + i)))
        (decreases k) =
    if k > 0 then combine (k - 1)
  in combine n

inline_for_extraction noextract
let rec source_sum_zero (n:nat)
  : Lemma (source_sum_n n (fun _ -> 0.0R) == 0.0R) (decreases n) =
  if n > 0 then source_sum_zero (n - 1)

inline_for_extraction noextract
let rec sum_swap (n m:nat) (f:nat -> nat -> GTot real)
  : Lemma
      (source_sum_n n (fun i -> source_sum_n m (f i)) ==
        source_sum_n m (fun j -> source_sum_n n (fun i -> f i j)))
      (decreases n) =
  let rec add (k:nat) (left right:nat -> GTot real)
    : Lemma
        (source_sum_n k (fun i -> left i +. right i) ==
          source_sum_n k left +. source_sum_n k right)
        (decreases k) =
    if k > 0 then add (k - 1) left right
  in
  if n > 0 then (
    sum_swap (n - 1) m f;
    add m (fun j -> source_sum_n (n - 1) (fun i -> f i j)) (f (n - 1));
    source_sum_ext m
      (fun j -> source_sum_n n (fun i -> f i j))
      (fun j -> source_sum_n (n - 1) (fun i -> f i j) +. f (n - 1) j)
  ) else (
    source_sum_ext m (fun j -> source_sum_n n (fun i -> f i j)) (fun _ -> 0.0R);
    source_sum_zero m
  )

inline_for_extraction noextract
let source_sum_step (n:pos) (f:nat -> GTot real)
  : Lemma (source_sum_n n f == source_sum_n (n - 1) f +. f (n - 1)) = ()

inline_for_extraction noextract
let rec sum_blocks (n m:nat) (f:nat -> GTot real)
  : Lemma
      (source_sum_n n (fun i -> source_sum_n m (fun j -> f (i * m + j))) ==
        source_sum_n (n * m) f)
      (decreases n) =
  if n > 0 then (
    source_sum_step n (fun i -> source_sum_n m (fun j -> f (i * m + j)));
    sum_blocks (n - 1) m f;
    source_sum_append ((n - 1) * m) m f;
    assert ((n - 1) * m + m == n * m)
  )
#pop-options

inline_for_extraction noextract
let reduce_width (n:nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32

inline_for_extraction noextract
let segment_rep (#n:nat) (xs:lseq F32.t n)
  (start count:nat{start + count <= n}) (r:nat -> GTot real) : prop =
  forall (i:nat{i < count}). Seq.index xs (start + i) %~ r i

inline_for_extraction noextract
let segment_frame (#a:Type0) (#n:nat) (before after:lseq a n)
  (start count:nat) : prop =
  forall (i:nat{i < n}). (i < start \/ start + count <= i) ==>
    Seq.index after i == Seq.index before i

(* The proof-only guard threads a representation implication. It never
   controls execution or replaces the public source-bound numerical premise. *)
[@@CPrologue "__device__"]
fn rec reduce_source_tree
  (buffer:larray F32.t 288) (start:sz)
  (count:szp{reduce_width count /\ start + count <= 288})
  (#original:erased (lseq F32.t 288))
  (#model:erased (nat -> GTot real)) (#guard:erased prop)
  preserves gpu
  requires buffer |-> original **
    pure (guard ==> segment_rep original start count model)
  ensures exists* (after:lseq F32.t 288).
    buffer |-> after **
    pure (segment_frame original after start count /\
      (guard ==> Seq.index after start %~ source_sum_n count model))
  decreases (SZ.v count)
{
  if (1sz <^ count) {
    let half = count /^ 2sz;
    assert pure (0 < half /\ half < count /\ SZ.v count == 2 * half /\ reduce_width half);
    let mut lane = 0sz;
    while (!lane <^ half)
      invariant exists* (j:sz{j <= half}) (xs:lseq F32.t 288).
        lane |-> j ** buffer |-> xs **
        pure (
          segment_frame original xs start count /\
          (forall (i:nat{i < count}).
            guard ==> Seq.index xs (start + i) %~
              (if i < j then (reveal model) i +. (reveal model) (half + i)
               else (reveal model) i)))
      decreases (half - !lane)
    {
      let j = !lane;
      let left = buffer.(start +^ j);
      let right = buffer.(start +^ j +^ half);
      let added = F32.add_rn_ftz left right;
      buffer.(start +^ j) <- added;
      lane := j +^ 1sz;
    };
    let finished = !lane;
    assert pure (SZ.v finished == SZ.v half);
    with xs. assert buffer |-> xs;
    let next:erased (nat -> GTot real) =
      hide (fun i -> (reveal model) i +. (reveal model) (half + i));
    assert pure (guard ==> segment_rep xs start half next);
    reduce_source_tree buffer start half
      #(hide (reveal xs <: lseq F32.t 288)) #next #guard;
    sum_halves half model;
    source_sum_ext half next (fun i -> (reveal model) i +. (reveal model) (half + i));
    assert pure (source_sum_n count model == source_sum_n half next);
    ()
  } else {
    source_sum_step 1 model;
    assert pure (SZ.v count == 1);
    ()
  };
}

inline_for_extraction noextract
let source_epsilon () : F32.t = ObF.of_literal "0.000001"

inline_for_extraction noextract
let source_epsilon_approx ()
  : Lemma (source_epsilon () %~ 0.000001R) =
  assert_norm (Kuiper.Float32.Literals.common_normal_cell
    "0.000001" 1 1000000 1 1000000 8796093 (-43));
  F32.literal_interval_approx "0.000001" 1 1000000 1 1000000
    8796093 (-43) 0.000001R

inline_for_extraction noextract
let rms_width (n:nat) : prop = n = 128 \/ n = 1024

inline_for_extraction noextract
let rms_lanes (n:nat{rms_width n}) : pos = if n = 1024 then 256 else 128

inline_for_extraction noextract
let rms_outer_width (n:nat{rms_width n}) : pos = if n = 1024 then 8 else 32

inline_for_extraction noextract
let rms_square (n:nat{rms_width n}) (r:Shared.vector n) (i:nat) : GTot real =
  if i < n then r i *. r i else 0.0R

inline_for_extraction noextract
let rms_lane_model (n:nat{rms_width n}) (r:Shared.vector n) (lane:nat) : GTot real =
  source_sum_n (n / rms_lanes n)
    (fun step -> rms_square n r (lane + step * rms_lanes n))

inline_for_extraction noextract
let rms_warp_model (n:nat{rms_width n}) (r:Shared.vector n) (warp:nat) : GTot real =
  if warp < rms_lanes n / 32 then
    source_sum_n 32 (fun lane -> rms_lane_model n r (warp * 32 + lane))
  else 0.0R

inline_for_extraction noextract
let rms_warp_at (n:nat{rms_width n}) (r:Shared.vector n)
  (warp:nat{warp < rms_lanes n / 32})
  : Lemma
      (rms_warp_model n r warp ==
        source_sum_n 32 (fun lane -> rms_lane_model n r (warp * 32 + lane))) = ()

inline_for_extraction noextract
let rms_warp_terms (n:nat{rms_width n}) (r:Shared.vector n)
  (warp lane:nat) : GTot real =
  rms_lane_model n r (warp * 32 + lane)

inline_for_extraction noextract
let rms_warp_related (n:nat{rms_width n}) (r:Shared.vector n)
  (warp:nat{warp < rms_lanes n / 32}) (value:F32.t) (guard:prop)
  : Lemma
      (requires guard ==> value %~
        source_sum_n 32 (rms_warp_terms n r warp))
      (ensures guard ==> value %~ rms_warp_model n r warp) =
  rms_warp_at n r warp;
  source_sum_ext 32 (rms_warp_terms n r warp)
    (fun lane -> rms_lane_model n r (warp * 32 + lane))

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let rms_reduction_total (n:nat{rms_width n}) (r:Shared.vector n)
  : Lemma
      (source_sum_n (rms_outer_width n) (rms_warp_model n r) ==
        Shared.sum n (fun i -> r i *. r i)) =
  let lanes = rms_lanes n in
  let warps = lanes / 32 in
  let outer = rms_outer_width n in
  let rec zero_tail (k:nat{warps <= k /\ k <= outer})
    : Lemma
        (source_sum_n k (rms_warp_model n r) ==
          source_sum_n warps (rms_warp_model n r))
        (decreases k) =
    if k > warps then zero_tail (k - 1)
  in
  zero_tail outer;
  source_sum_ext warps (rms_warp_model n r)
    (fun warp -> source_sum_n 32 (fun lane -> rms_lane_model n r (warp * 32 + lane)));
  sum_blocks warps 32 (rms_lane_model n r);
  assert (warps * 32 == lanes /\ (n / lanes) * lanes == n);
  assert (source_sum_n outer (rms_warp_model n r) ==
    source_sum_n lanes (rms_lane_model n r));
  let unfold_lane (lane:nat)
    : Lemma (rms_lane_model n r lane ==
        source_sum_n (n / lanes) (fun step -> rms_square n r (lane + step * lanes))) =
    source_sum_ext (n / lanes)
      (fun step -> rms_square n r (lane + step * rms_lanes n))
      (fun step -> rms_square n r (lane + step * lanes))
  in FStar.Classical.forall_intro unfold_lane;
  source_sum_ext lanes (rms_lane_model n r)
    (fun lane -> source_sum_n (n / lanes)
      (fun step -> rms_square n r (lane + step * lanes)));
  sum_swap lanes (n / lanes)
    (fun lane step -> rms_square n r (lane + step * lanes));
  let commute (step:nat)
    : Lemma
        (source_sum_n lanes (fun lane -> rms_square n r (lane + step * lanes)) ==
          source_sum_n lanes (fun lane -> rms_square n r (step * lanes + lane))) =
    source_sum_ext lanes
      (fun lane -> rms_square n r (lane + step * lanes))
      (fun lane -> rms_square n r (step * lanes + lane))
  in FStar.Classical.forall_intro commute;
  source_sum_ext (n / lanes)
    (fun step -> source_sum_n lanes (fun lane -> rms_square n r (lane + step * lanes)))
    (fun step -> source_sum_n lanes (fun lane -> rms_square n r (step * lanes + lane)));
  sum_blocks (n / lanes) lanes (rms_square n r);
  source_sum_shared n (rms_square n r);
  sum_extensional n (rms_square n r) (fun i -> r i *. r i)
#pop-options

inline_for_extraction noextract
let bf16_to_f32_all (value:bf16)
  : Lemma (forall (r:real).
      value %~ r ==> Casts.fcast #bf16 #F32.t value %~ r) =
  FStar.Classical.forall_intro
    (FStar.Classical.move_requires (Casts.fcast_approx #bf16 #F32.t value))

inline_for_extraction noextract
let f32_to_bf16_all (value:F32.t)
  : Lemma (forall (r:real).
      value %~ r ==> Casts.fcast #F32.t #bf16 value %~ r) =
  FStar.Classical.forall_intro
    (FStar.Classical.move_requires (Casts.fcast_approx #F32.t #bf16 value))

inline_for_extraction noextract
let real_div_reciprocal (a:real) (b:real{b >. 0.0R})
  : Lemma (a /. b == a *. (1.0R /. b)) = ()

inline_for_extraction noextract
let rms_radicand (n:pos) (r:Shared.vector n) : GTot ObRoot.rpos =
  Shared.sum_nonnegative n (fun j -> r j *. r j);
  Shared.sum n (fun j -> r j *. r j) /. of_int n +. 0.000001R

inline_for_extraction noextract
let rms_radicand_at (n:pos) (r:Shared.vector n)
  : Lemma
      (rms_radicand n r ==
        Shared.sum n (fun j -> r j *. r j) *. (1.0R /. of_int n) +. 0.000001R) =
  Shared.sum_nonnegative n (fun j -> r j *. r j);
  real_div_reciprocal (Shared.sum n (fun j -> r j *. r j)) (of_int n)

inline_for_extraction noextract
let rms_point (n:pos) (r gamma:Shared.vector n) (i:Shared.index n)
  : Lemma
      (requires Shared.sum n (fun j -> r j *. r j) >=. 0.0R)
      (ensures Shared.rms_norm gamma r i ==
        gamma i *. (r i *. ObRoot.rsqrt (rms_radicand n r))) =
  Shared.sum_nonnegative n (fun j -> r j *. r j)

inline_for_extraction noextract
let real_product_reorder (a b c:real)
  : Lemma (a *. (b *. c) == c *. (a *. b)) = ()

inline_for_extraction noextract
let rms_cast_refines (n:pos) (r gamma:Shared.vector n) (i:Shared.index n)
  (product:F32.t) (guard:prop)
  : Lemma
      (requires guard ==> product %~
        (r i *. (ObRoot.rsqrt (rms_radicand n r) *. gamma i)))
      (ensures guard ==> Casts.fcast #F32.t #bf16 product %~ Shared.rms_norm gamma r i) =
  Shared.sum_nonnegative n (fun j -> r j *. r j);
  rms_point n r gamma i;
  real_product_reorder (r i) (ObRoot.rsqrt (rms_radicand n r)) (gamma i);
  assert (guard ==> product %~ Shared.rms_norm gamma r i);
  f32_to_bf16_all product

inline_for_extraction noextract
fn rms_source_squares
  (#n:erased nat) (input:larray bf16 n) (buffer:larray F32.t 288)
  (start:sz) (width:szp{rms_width width /\ start + width <= n})
  (lanes:szp{SZ.v lanes == rms_lanes width})
  (#x:erased (lseq bf16 n)) (#old:erased (lseq F32.t 288)) (#f:perm)
  (#r:erased (Shared.vector width)) (#guard:erased prop)
  preserves gpu ** input |-> Frac f x
  requires buffer |-> old **
    pure (guard ==> (forall (i:Shared.index width).
      Seq.index x (start + i) %~ (reveal r) i))
  returns result:F32.t
  ensures exists* (scratch:lseq F32.t 288).
    buffer |-> scratch **
    pure (guard ==> result %~ Shared.sum width (fun i -> (reveal r) i *. (reveal r) i))
{
  let mut thread = 0sz;
  while (!thread <^ lanes)
    invariant exists* (t:sz{t <= lanes}) (xs:lseq F32.t 288).
      thread |-> t ** buffer |-> xs **
      pure (forall (i:nat{i < t}).
        guard ==> Seq.index xs i %~ rms_lane_model width r i)
    decreases (lanes - !thread)
  {
    let t = !thread;
    let term:erased (nat -> GTot real) =
      hide (fun (j:nat) -> rms_square width r (t + j * lanes));
    let mut sum:F32.t = ObF.zero;
    let mut step = 0sz;
    while (!step <^ width /^ lanes)
      invariant exists* (s:sz{s <= width / lanes}) (acc:F32.t).
        step |-> s ** sum |-> acc **
        pure (guard ==> acc %~ source_sum_n s term)
      decreases (width / lanes - !step)
    {
      let s = !step;
      let value = input.(start +^ t +^ s *^ lanes);
      let v = Casts.fcast #bf16 #F32.t value;
      bf16_to_f32_all value;
      assert pure (guard ==> v %~ (reveal r) (t + s * lanes));
      let previous = !sum;
      let next = F32.fma_rn_ftz v v previous;
      assert pure (guard ==> next %~
        ((reveal r) (t + s * lanes) *. (reveal r) (t + s * lanes) +.
          source_sum_n s term));
      source_sum_step (s + 1) term;
      assert pure (guard ==> next %~ source_sum_n (s + 1) term);
      sum := next;
      step := s +^ 1sz;
    };
    let finished = !step;
    assert pure (SZ.v finished == width / lanes);
    let reduced = !sum;
    source_sum_ext (width / lanes)
      term
      (fun j -> rms_square width r (t + j * rms_lanes width));
    assert pure (guard ==> reduced %~ rms_lane_model width r t);
    buffer.(t) <- reduced;
    thread := t +^ 1sz;
  };
  let threads_done = !thread;
  assert pure (SZ.v threads_done == SZ.v lanes);
  let mut warp = 0sz;
  while (!warp <^ lanes /^ 32sz)
    invariant exists* (w:sz{w <= lanes / 32}) (xs:lseq F32.t 288).
      warp |-> w ** buffer |-> xs **
      pure (
        (forall (i:nat{w * 32 <= i /\ i < lanes}).
          guard ==> Seq.index xs i %~ rms_lane_model width r i) /\
        (forall (i:nat{i < w}).
          guard ==> Seq.index xs (256 + i) %~ rms_warp_model width r i))
    decreases (lanes / 32 - !warp)
  {
    let w = !warp;
    with xs. assert buffer |-> xs;
    reduce_source_tree buffer (w *^ 32sz) 32sz
      #(hide (reveal xs <: lseq F32.t 288))
      #(hide (rms_warp_terms width r w)) #guard;
    let reduced = buffer.(w *^ 32sz);
    rms_warp_related width r w reduced guard;
    assert pure (guard ==> reduced %~ rms_warp_model width r w);
    buffer.(256sz +^ w) <- reduced;
    warp := w +^ 1sz;
  };
  let warps_done = !warp;
  assert pure (SZ.v warps_done == lanes / 32);
  let outer = if (width = 1024sz) { 8sz } else { 32sz };
  let mut empty = lanes /^ 32sz;
  while (!empty <^ outer)
    invariant exists* (i:sz{lanes / 32 <= i /\ i <= outer}) (xs:lseq F32.t 288).
      empty |-> i ** buffer |-> xs **
      pure (forall (j:nat{j < i}).
        guard ==> Seq.index xs (256 + j) %~ rms_warp_model width r j)
    decreases (outer - !empty)
  {
    let i = !empty;
    buffer.(256sz +^ i) <- ObF.zero #F32.t;
    empty := i +^ 1sz;
  };
  let empty_done = !empty;
  assert pure (SZ.v empty_done == SZ.v outer);
  with xs. assert buffer |-> xs;
  reduce_source_tree buffer 256sz outer
    #(hide (reveal xs <: lseq F32.t 288))
    #(hide (rms_warp_model width r)) #guard;
  rms_reduction_total width r;
  buffer.(256sz)
}

inline_for_extraction noextract
let bf16_segment_rep (#n:nat) (xs:lseq bf16 n)
  (start count:nat{start + count <= n}) (r:Shared.vector count) : prop =
  forall (i:Shared.index count). Seq.index xs (start + i) %~ r i

inline_for_extraction noextract
let source_inverse_width (width:szp{rms_width width})
  : Tot (value:F32.t{value %~ (1.0R /. of_int (SZ.v width))}) =
  if width = 1024sz then (
    of_int_approx #F32.t 1024L;
    div_approx (ObF.one #F32.t) (ObF.of_int #F32.t 1024L) 1.0R 1024.0R;
    ObF.div #F32.t ObF.one (ObF.of_int 1024L)
  ) else (
    of_int_approx #F32.t 128L;
    div_approx (ObF.one #F32.t) (ObF.of_int #F32.t 128L) 1.0R 128.0R;
    ObF.div #F32.t ObF.one (ObF.of_int 128L)
  )

inline_for_extraction noextract
let bf16_prefix_write (#n:nat) (before after:lseq bf16 n)
  (start count:nat{start + count <= n}) (step:nat{step < count})
  (model:Shared.vector count) (value:bf16) (guard:prop)
  : Lemma
      (requires
        after == Seq.upd before (start + step) value /\
        (guard ==> value %~ model step) /\
        (forall (i:Shared.index count{i < step}).
          guard ==> Seq.index before (start + i) %~ model i))
      (ensures forall (i:Shared.index count{i < step + 1}).
        guard ==> Seq.index after (start + i) %~ model i) =
  let at (i:Shared.index count{i < step + 1})
    : Lemma (guard ==> Seq.index after (start + i) %~ model i) =
    if i = step then assert (Seq.index after (start + i) == value)
    else assert (Seq.index after (start + i) == Seq.index before (start + i))
  in FStar.Classical.forall_intro at

inline_for_extraction noextract
let source_loop_end (i bound:sz)
  : Lemma
      (requires i <= bound /\ not (i <^ bound))
      (ensures SZ.v i == SZ.v bound) = ()

inline_for_extraction noextract
let conditional_fma
  (x y z result:F32.t) (xr yr zr:real) (guard:prop)
  : Lemma
      (requires
        (guard ==> x %~ xr /\ y %~ yr /\ z %~ zr) /\
        (forall (a b c:real).
          x %~ a /\ y %~ b /\ z %~ c ==> result %~ (a *. b +. c)))
      (ensures guard ==> result %~ (xr *. yr +. zr)) = ()

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn rms_source_scale
  (reduced:F32.t) (width:szp{rms_width width})
  (#r:erased (Shared.vector width)) (#guard:erased prop)
  preserves gpu
  requires pure (guard ==> reduced %~
    Shared.sum width (fun i -> (reveal r) i *. (reveal r) i))
  returns scale:F32.t
  ensures pure (guard ==> v_approximates scale (ObRoot.rsqrt (rms_radicand width r)))
{
  let inverse_width = source_inverse_width width;
  assert pure (inverse_width %~ (1.0R /. of_int (SZ.v width)));
  source_epsilon_approx ();
  let radicand = F32.fma_rn_ftz reduced inverse_width (source_epsilon ());
  conditional_fma reduced inverse_width (source_epsilon ()) radicand
    (Shared.sum width (fun i -> (reveal r) i *. (reveal r) i))
    (1.0R /. of_int (SZ.v width)) 0.000001R guard;
  rms_radicand_at width r;
  assert pure (guard ==> v_approximates radicand (rms_radicand width r));
  let scale = F32.rsqrt_approx_ftz radicand;
  scale
}
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn rms_source_row
  (#n #ny:erased nat)
  (input:larray bf16 n) (weights:bf16_array weight_elements)
  (output:larray bf16 ny) (buffer:larray F32.t 288)
  (start:sz) (width:szp{rms_width width /\ start + width <= n})
  (weight_start:sz{weight_start + width <= weight_elements})
  (output_start:sz{output_start + width <= ny})
  (#x:erased (lseq bf16 n)) (#w:erased (bf16_values weight_elements))
  (#old:erased (lseq bf16 ny)) (#tmp:erased (lseq F32.t 288))
  (#fx #fw:perm)
  (#r #gamma:erased (Shared.vector width)) (#guard:erased prop)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w
  requires output |-> old ** buffer |-> tmp **
    pure (guard ==> bf16_segment_rep x start width r /\
      bf16_segment_rep w weight_start width gamma)
  ensures exists* (after:lseq bf16 ny) (scratch:lseq F32.t 288).
    output |-> after ** buffer |-> scratch **
    pure (segment_frame old after output_start width /\
      (guard ==> bf16_segment_rep after output_start width (Shared.rms_norm gamma r)))
{
  let lanes = if (width = 1024sz) { 256sz } else { 128sz };
  let reduced = rms_source_squares input buffer start width lanes #x #tmp #fx #r #guard;
  let scale = rms_source_scale reduced width #r #guard;
  let rr = hide (rms_radicand width r);
  let mut col = 0sz;
  while (!col <^ width)
    invariant exists* (i:sz{i <= width}) (ys:lseq bf16 ny).
      col |-> i ** output |-> ys **
      pure (
        segment_frame old ys output_start i /\
        (forall (j:Shared.index width{j < i}).
          guard ==> Seq.index ys (output_start + j) %~ Shared.rms_norm gamma r j))
    decreases (width - !col)
  {
    let i = !col;
    let value = input.(start +^ i);
    let weight = weights.(weight_start +^ i);
    let xf = Casts.fcast #bf16 #F32.t value;
    let gf = Casts.fcast #bf16 #F32.t weight;
    bf16_to_f32_all value;
    bf16_to_f32_all weight;
    assert pure (guard ==> xf %~ (reveal r) i /\ gf %~ (reveal gamma) i);
    let weighted = F32.mul_rn_ftz scale gf;
    assert pure (guard ==> weighted %~ (ObRoot.rsqrt (reveal rr) *. (reveal gamma) i));
    let product = F32.mul_rn_ftz xf weighted;
    assert pure (guard ==> product %~
      ((reveal r) i *. (ObRoot.rsqrt (reveal rr) *. (reveal gamma) i)));
    rms_cast_refines width r gamma i product guard;
    let result = Casts.fcast #F32.t #bf16 product;
    assert pure (guard ==> result %~ Shared.rms_norm gamma r i);
    with before. assert output |-> before;
    output.(output_start +^ i) <- result;
    with after. assert output |-> after;
    bf16_prefix_write before after output_start width i (Shared.rms_norm gamma r) result guard;
    col := i +^ 1sz;
  };
  let finished = !col;
  source_loop_end finished width;
}
#pop-options

inline_for_extraction noextract
let matrix_prefix (#rows #cols:pos) (xs:lseq bf16 (rows * cols))
  (model:Shared.matrix rows cols) (done:nat) : prop =
  forall (row:Shared.index rows{row < done}) (col:Shared.index cols).
    Seq.index xs (row * cols + col) %~ model row col

inline_for_extraction noextract
let matrix_rep (#rows #cols:pos) (xs:lseq bf16 (rows * cols))
  (model:Shared.matrix rows cols) : prop =
  matrix_prefix xs model rows

inline_for_extraction noextract
let matrix_row_progress (#rows #cols:pos)
  (before after:lseq bf16 (rows * cols))
  (model:Shared.matrix rows cols) (row:Shared.index rows) (guard:prop)
  : Lemma
      (requires
        segment_frame before after (row * cols) cols /\
        (guard ==> matrix_prefix before model row /\
          bf16_segment_rep after (row * cols) cols (model row)))
      (ensures guard ==> matrix_prefix after model (row + 1)) =
  let at (t:Shared.index rows{t < row + 1}) (c:Shared.index cols)
    : Lemma (guard ==> Seq.index after (t * cols + c) %~ model t c) =
    if t <> row then
      assert (Seq.index after (t * cols + c) == Seq.index before (t * cols + c))
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let normalized_hidden8 (gamma:Shared.vector 1024) (x:stored_hidden)
  : GTot stored_hidden =
  fun row -> Shared.rms_norm gamma (x row)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn rms_hidden_kernel
  (input:bf16_array hidden_elements) (weights:bf16_array weight_elements)
  (output:bf16_array hidden_elements) (buffer:larray F32.t 288)
  (weight_start:sz{weight_start + 1024 <= weight_elements})
  (#x:erased (bf16_values hidden_elements)) (#w:erased (bf16_values weight_elements))
  (#old:erased (bf16_values hidden_elements)) (#tmp:erased (lseq F32.t 288))
  (#fx #fw:perm) (#r:erased stored_hidden) (#gamma:erased (Shared.vector 1024))
  (#guard:erased prop)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w
  requires output |-> old ** buffer |-> tmp **
    pure (guard ==> matrix_rep x r /\ bf16_segment_rep w weight_start 1024 gamma)
  ensures exists* (ys:bf16_values hidden_elements) (scratch:lseq F32.t 288).
    output |-> ys ** buffer |-> scratch **
    pure (guard ==> matrix_rep ys (normalized_hidden8 gamma r))
{
  let mut row = 0sz;
  while (!row <^ 8sz)
    invariant exists* (t:sz{t <= 8}) (ys:bf16_values hidden_elements)
      (scratch:lseq F32.t 288).
      row |-> t ** output |-> ys ** buffer |-> scratch **
      pure (guard ==> matrix_prefix ys (normalized_hidden8 gamma r) t)
    decreases (8 - !row)
  {
    let t = !row;
    with before. assert output |-> before;
    with scratch. assert buffer |-> scratch;
    rms_source_row input weights output buffer (t *^ 1024sz) 1024sz weight_start (t *^ 1024sz)
      #x #w #(hide (reveal before <: bf16_values hidden_elements))
      #(hide (reveal scratch <: lseq F32.t 288)) #fx #fw
      #(hide ((reveal r) t)) #gamma #guard;
    with after. assert output |-> after;
    matrix_row_progress before after (normalized_hidden8 gamma r) t guard;
    row := t +^ 1sz;
  };
  let finished = !row;
  source_loop_end finished 8sz;
}
#pop-options

inline_for_extraction noextract
let qkv_weight (w:Shared.parameters) (channel:Shared.index 4096)
  : GTot (Shared.vector 1024) =
  let group = channel / 512 in
  let inner = channel % 512 in
  if inner < 256 then w.q_proj (group * 256 + inner)
  else if inner < 384 then w.k_proj (group * 128 + inner - 256)
  else w.v_proj (group * 128 + inner - 384)

inline_for_extraction noextract
let qkv_query_weight_at (w:Shared.parameters) (head:Shared.index 16)
  (lane:Shared.index 128) (col:Shared.index 1024)
  : Lemma
      (qkv_weight w (head / 2 * 512 + head % 2 * 128 + lane) col ==
        w.q_proj (head * 128 + lane) col) = ()

inline_for_extraction noextract
let qkv_kv_weight_at (w:Shared.parameters) (head:Shared.index 8)
  (lane:Shared.index 128) (col:Shared.index 1024)
  : Lemma
      (qkv_weight w (head * 512 + 256 + lane) col == w.k_proj (head * 128 + lane) col /\
       qkv_weight w (head * 512 + 384 + lane) col == w.v_proj (head * 128 + lane) col) = ()

inline_for_extraction noextract
let gateup_weight (w:Shared.parameters) (channel:Shared.index 6144)
  : GTot (Shared.vector 1024) =
  let index = channel / 128 * 64 + channel % 64 in
  if channel % 128 < 64 then w.gate_proj index else w.up_proj index

inline_for_extraction noextract
let gateup_weight_at (w:Shared.parameters) (col:Shared.index 3072) (k:Shared.index 1024)
  : Lemma
      (gateup_weight w (col / 64 * 128 + col % 64) k == w.gate_proj col k /\
       gateup_weight w (col / 64 * 128 + col % 64 + 64) k == w.up_proj col k) = ()

inline_for_extraction noextract
let source_head (x:Shared.matrix 8 4096)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  : GTot (Shared.vector 128) =
  fun lane -> x row (group * 512 + head * 128 + lane)

[@@"opaque_to_smt"]
inline_for_extraction noextract
let normalized_source_head (w:Shared.parameters) (x:Shared.matrix 8 4096)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  : GTot (Shared.vector 128) =
  Shared.rms_norm (if head = 2 then w.k_norm else w.q_norm) (source_head x row group head)

inline_for_extraction noextract
let normalized_source_head_unfold (w:Shared.parameters) (x:Shared.matrix 8 4096)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  : Lemma (normalized_source_head w x row group head ==
      Shared.rms_norm (if head = 2 then w.k_norm else w.q_norm) (source_head x row group head)) =
  reveal_opaque (`%normalized_source_head) (normalized_source_head w x row group head)

inline_for_extraction noextract
let rotate_source_head (head:Shared.vector 128) (data:Shared.attention_data)
  (row:Shared.index 7) (lane:Shared.index 128) : GTot real =
  head lane *. data.cosine row lane +.
    (if lane < 64 then 0.0R -. head (lane + 64) else head (lane - 64)) *. data.sine row lane

[@@"opaque_to_smt"]
inline_for_extraction noextract
let rotated_model (w:Shared.parameters) (x:Shared.matrix 8 4096)
  (data:Shared.attention_data) : GTot (Shared.matrix 8 4096) =
  fun row col ->
    if row < 7 && col % 512 < 384 then
      rotate_source_head
        (normalized_source_head w x row (col / 512) ((col % 512) / 128))
        data row (col % 128)
    else x row col

inline_for_extraction noextract
let rotated_model_head (w:Shared.parameters) (x:Shared.matrix 8 4096)
  (data:Shared.attention_data)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3) (lane:Shared.index 128)
  : Lemma (rotated_model w x data row (group * 512 + head * 128 + lane) ==
      rotate_source_head (normalized_source_head w x row group head) data row lane) =
  reveal_opaque (`%rotated_model) (rotated_model w x data)

inline_for_extraction noextract
let rotated_model_unchanged (w:Shared.parameters) (x:Shared.matrix 8 4096)
  (data:Shared.attention_data) (row:Shared.index 8) (col:Shared.index 4096)
  : Lemma
      (requires 7 <= row \/ 384 <= col % 512)
      (ensures rotated_model w x data row col == x row col) =
  reveal_opaque (`%rotated_model) (rotated_model w x data)

inline_for_extraction noextract
let rotary_result_refines (head:Shared.vector 128) (data:Shared.attention_data)
  (row:Shared.index 7) (lane:Shared.index 128)
  (partner:Shared.index 128{partner == (lane + 64) % 128})
  (lower:bool{lower == (lane < 64)}) (value:bf16)
  : Lemma
      (requires value %~ (head lane *. data.cosine row lane +.
        (if lower then 0.0R -. head partner else head partner) *. data.sine row lane))
      (ensures value %~ rotate_source_head head data row lane) =
  if lane < 64 then assert (partner == lane + 64)
  else assert (partner == lane - 64)

inline_for_extraction noextract
let source_negative_zero () : Tot (z:F32.t{z %~ 0.0R}) =
  let minus_one = ObF.of_int #F32.t (-1L) in
  of_int_approx #F32.t (-1L);
  a_mul minus_one (ObF.zero #F32.t) (0.0R -. 1.0R) 0.0R;
  ObF.mul minus_one ObF.zero

inline_for_extraction noextract
let source_negate (value:F32.t) : F32.t =
  ObF.sub (source_negative_zero ()) value

inline_for_extraction noextract
let source_negate_approx (value:F32.t) (r:real)
  : Lemma (requires value %~ r)
      (ensures source_negate value %~ (0.0R -. r)) =
  sub_approx (source_negative_zero ()) value 0.0R r

inline_for_extraction noextract
fn rotary_source_value (value partner cosine sine:bf16) (lower:bool)
  (#vr #pr #cr #sr:erased real)
  preserves gpu
  requires pure (value %~ reveal vr /\ partner %~ reveal pr /\
    cosine %~ reveal cr /\ sine %~ reveal sr)
  returns result:bf16
  ensures pure (result %~
    (reveal vr *. reveal cr +. (if lower then 0.0R -. reveal pr else reveal pr) *. reveal sr))
{
  let v = Casts.fcast #bf16 #F32.t value;
  let p = Casts.fcast #bf16 #F32.t partner;
  let c = Casts.fcast #bf16 #F32.t cosine;
  let s = Casts.fcast #bf16 #F32.t sine;
  bf16_to_f32_all value;
  bf16_to_f32_all partner;
  bf16_to_f32_all cosine;
  bf16_to_f32_all sine;
  source_negate_approx p (reveal pr);
  let signed = if lower { source_negate p } else { p };
  let shifted = F32.mul_rn_ftz signed s;
  let rotated = F32.fma_rn_ftz v c shifted;
  f32_to_bf16_all rotated;
  Casts.fcast #F32.t #bf16 rotated
}

inline_for_extraction noextract
fn copy_bf16
  (#n:erased nat) (input output:larray bf16 n) (count:sz{SZ.v count == n})
  (#x #old:erased (lseq bf16 n)) (#f:perm)
  preserves gpu ** input |-> Frac f x
  requires output |-> old
  ensures output |-> x
{
  let mut index = 0sz;
  while (!index <^ count)
    invariant exists* (i:sz{i <= count}) (ys:lseq bf16 n).
      index |-> i ** output |-> ys **
      pure (forall (j:nat{j < i}). Seq.index ys j == Seq.index x j)
    decreases (count - !index)
  {
    let i = !index;
    let value = input.(i);
    output.(i) <- value;
    index := i +^ 1sz;
  };
  let finished = !index;
  source_loop_end finished count;
  with ys. assert output |-> ys;
  Seq.lemma_eq_intro ys x;
  Seq.lemma_eq_elim ys x;
  assert pure (ys == x);
}

inline_for_extraction noextract
let qk_head_number (row:Shared.index 8) (col:Shared.index 4096) : nat =
  row * 24 + col / 512 * 3 + (col % 512) / 128

inline_for_extraction noextract
let rotated_prefix
  (input output:bf16_values (8 * 4096)) (w:Shared.parameters)
  (x:Shared.matrix 8 4096) (data:Shared.attention_data) (done:nat) : prop =
  forall (row:Shared.index 8) (col:Shared.index 4096).
    if row < 7 && col % 512 < 384 && qk_head_number row col < done then
      Seq.index output (row * 4096 + col) %~ rotated_model w x data row col
    else Seq.index output (row * 4096 + col) == Seq.index input (row * 4096 + col)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let qk_weight_offset : sz = SZ.uint32_to_sizet 6293504ul

inline_for_extraction noextract
let rotary_plane_stride : sz = SZ.uint32_to_sizet 524288ul

inline_for_extraction noextract
let rotary_values_at (rt:bf16_values rotary_elements)
  (position:nat{position <= 4089}) (data:Shared.attention_data)
  (row:Shared.index 7) (lane:Shared.index 128)
  : Lemma
      (requires rotary_representation rt position data)
      (ensures
        Seq.index rt ((position + row) * 128 + lane) %~ data.cosine row lane /\
        Seq.index rt (524288 + (position + row) * 128 + lane) %~ data.sine row lane) = ()

inline_for_extraction noextract
let rotated_head_progress
  (input before after:bf16_values (8 * 4096)) (w:Shared.parameters)
  (x:Shared.matrix 8 4096) (data:Shared.attention_data)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  : Lemma
      (requires
        rotated_prefix input before w x data (row * 24 + group * 3 + head) /\
        segment_frame before after (row * 4096 + group * 512 + head * 128) 128 /\
        bf16_segment_rep after (row * 4096 + group * 512 + head * 128) 128
          (rotate_source_head (normalized_source_head w x row group head) data row))
      (ensures rotated_prefix input after w x data (row * 24 + group * 3 + head + 1)) =
  let at (t:Shared.index 8) (c:Shared.index 4096)
    : Lemma
        (if t < 7 && c % 512 < 384 && qk_head_number t c < row * 24 + group * 3 + head + 1
         then Seq.index after (t * 4096 + c) %~ rotated_model w x data t c
         else Seq.index after (t * 4096 + c) == Seq.index input (t * 4096 + c)) =
    if t = row && c / 512 = group && (c % 512) / 128 = head then (
      assert (c == group * 512 + head * 128 + c % 128);
      assert (t < 7 /\ c % 512 < 384);
      rotated_model_head w x data row group head (c % 128);
      assert (rotated_model w x data t c ==
        rotate_source_head (normalized_source_head w x row group head) data row (c % 128))
    ) else (
      assert (t * 4096 + c < row * 4096 + group * 512 + head * 128 \/
        row * 4096 + group * 512 + head * 128 + 128 <= t * 4096 + c);
      assert (Seq.index after (t * 4096 + c) == Seq.index before (t * 4096 + c))
    )
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
fn normalize_rotate_head
  (input output:bf16_array (8 * 4096)) (weights:bf16_array weight_elements)
  (table:bf16_array rotary_elements) (normalized:larray bf16 128) (buffer:larray F32.t 288)
  (position:sz{position <= 4089}) (row:szlt 7) (group:szlt 8) (head:szlt 3)
  (#x #old:erased (bf16_values (8 * 4096))) (#w:erased (bf16_values weight_elements))
  (#rt:erased (bf16_values rotary_elements)) (#old_norm:erased (lseq bf16 128))
  (#tmp:erased (lseq F32.t 288)) (#fx #fw #ft:perm)
  (#xr:erased (Shared.matrix 8 4096)) (#wr:erased Shared.parameters)
  (#data:erased Shared.attention_data)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w ** table |-> Frac ft rt
  requires output |-> old ** normalized |-> old_norm ** buffer |-> tmp **
    pure (matrix_rep x xr /\ weights_representation w wr /\
      rotary_representation rt position data /\
      rotated_prefix x old wr xr data (row * 24 + group * 3 + head))
  ensures exists* (ys:bf16_values (8 * 4096)) (ns:lseq bf16 128) (scratch:lseq F32.t 288).
    output |-> ys ** normalized |-> ns ** buffer |-> scratch **
    pure (rotated_prefix x ys wr xr data (row * 24 + group * 3 + head + 1))
{
  let base = row *^ 4096sz +^ group *^ 512sz +^ head *^ 128sz;
  let weight_start = qk_weight_offset +^ (if head = 2sz then 128sz else 0sz);
  let gamma = hide (if head = 2sz then (reveal wr).k_norm else (reveal wr).q_norm);
  let rh = hide (source_head xr row group head);
  normalized_source_head_unfold wr xr row group head;
  let nh = hide (normalized_source_head wr xr row group head);
  rms_source_row input weights normalized buffer base 128sz weight_start 0sz
    #x #w #old_norm #tmp #fx #fw #rh #gamma #(hide True);
  let rotated = hide (rotate_source_head nh data row);
  let mut lane = 0sz;
  while (!lane <^ 128sz)
    invariant exists* (i:sz{i <= 128}) (ys:bf16_values (8 * 4096)) (ns:lseq bf16 128).
      lane |-> i ** output |-> ys ** normalized |-> ns **
      pure (
        bf16_segment_rep ns 0 128 nh /\
        segment_frame old ys base i /\
        (forall (j:Shared.index 128{j < i}).
          Seq.index ys (base + j) %~ (reveal rotated) j))
    decreases (128 - !lane)
  {
    let i = !lane;
    let partner = (i +^ 64sz) %^ 128sz;
    let v = normalized.(i);
    let p = normalized.(partner);
    rotary_values_at rt position data row i;
    let c = table.((position +^ row) *^ 128sz +^ i);
    let s = table.(rotary_plane_stride +^ (position +^ row) *^ 128sz +^ i);
    let value = rotary_source_value v p c s (i <^ 64sz)
      #(hide ((reveal nh) i)) #(hide ((reveal nh) partner))
      #(hide ((reveal data).cosine row i)) #(hide ((reveal data).sine row i));
    rotary_result_refines nh data row i partner (i <^ 64sz) value;
    assert pure (value %~ (reveal rotated) i);
    with before. assert output |-> before;
    output.(base +^ i) <- value;
    with after. assert output |-> after;
    bf16_prefix_write before after base 128 i rotated value True;
    lane := i +^ 1sz;
  };
  let finished = !lane;
  source_loop_end finished 128sz;
  with after. assert output |-> after;
  rotated_head_progress x old after wr xr data row group head;
  ()
}

inline_for_extraction noextract
let rotated_finish
  (x ys:bf16_values (8 * 4096)) (w:Shared.parameters)
  (xr:Shared.matrix 8 4096) (data:Shared.attention_data)
  : Lemma
      (requires matrix_rep x xr /\ rotated_prefix x ys w xr data 168)
      (ensures
        matrix_rep ys (rotated_model w xr data) /\
        (forall (row:Shared.index 8) (group:Shared.index 8) (lane:Shared.index 128).
          Seq.index ys (row * 4096 + group * 512 + 384 + lane) ==
            Seq.index x (row * 4096 + group * 512 + 384 + lane)) /\
        (forall (col:Shared.index 4096).
          Seq.index ys (7 * 4096 + col) == Seq.index x (7 * 4096 + col))) =
  let unchanged (row:Shared.index 8) (col:Shared.index 4096)
    : Lemma
          (requires 7 <= row \/ 384 <= col % 512)
          (ensures Seq.index ys (row * 4096 + col) == Seq.index x (row * 4096 + col)) =
    assert (not (row < 7 && col % 512 < 384 && qk_head_number row col < 168))
  in
  let value (row:Shared.index 8) (group:Shared.index 8) (lane:Shared.index 128)
    : Lemma
          (Seq.index ys (row * 4096 + group * 512 + 384 + lane) ==
            Seq.index x (row * 4096 + group * 512 + 384 + lane)) =
    unchanged row (group * 512 + 384 + lane)
  in
  FStar.Classical.forall_intro_3 value;
  let padding (col:Shared.index 4096)
    : Lemma (Seq.index ys (7 * 4096 + col) == Seq.index x (7 * 4096 + col)) =
    unchanged 7 col
  in FStar.Classical.forall_intro padding;
  let represented (row:Shared.index 8) (col:Shared.index 4096)
    : Lemma (Seq.index ys (row * 4096 + col) %~ rotated_model w xr data row col) =
    if row >= 7 || col % 512 >= 384 then (
      unchanged row col;
      rotated_model_unchanged w xr data row col
    )
  in FStar.Classical.forall_intro_2 represented

inline_for_extraction noextract
fn norm_rope_kernel
  (input output:bf16_array (8 * 4096)) (weights:bf16_array weight_elements)
  (table:bf16_array rotary_elements) (normalized:larray bf16 128) (buffer:larray F32.t 288)
  (position:sz{position <= 4089})
  (#x #old:erased (bf16_values (8 * 4096))) (#w:erased (bf16_values weight_elements))
  (#rt:erased (bf16_values rotary_elements)) (#old_norm:erased (lseq bf16 128))
  (#tmp:erased (lseq F32.t 288)) (#fx #fw #ft:perm)
  (#xr:erased (Shared.matrix 8 4096)) (#wr:erased Shared.parameters)
  (#data:erased Shared.attention_data)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w ** table |-> Frac ft rt
  requires output |-> old ** normalized |-> old_norm ** buffer |-> tmp **
    pure (matrix_rep x xr /\ weights_representation w wr /\ rotary_representation rt position data)
  ensures exists* (ys:bf16_values (8 * 4096)) (ns:lseq bf16 128) (scratch:lseq F32.t 288).
    output |-> ys ** normalized |-> ns ** buffer |-> scratch **
    pure (matrix_rep ys (rotated_model wr xr data) /\
      (forall (row:Shared.index 8) (group:Shared.index 8) (lane:Shared.index 128).
        Seq.index ys (row * 4096 + group * 512 + 384 + lane) ==
          Seq.index x (row * 4096 + group * 512 + 384 + lane)) /\
      (forall (col:Shared.index 4096).
        Seq.index ys (7 * 4096 + col) == Seq.index x (7 * 4096 + col)))
{
  copy_bf16 input output 32768sz #x #old #fx;
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (t:sz{t <= 7}) (ys:bf16_values (8 * 4096))
      (ns:lseq bf16 128) (scratch:lseq F32.t 288).
      row |-> t ** output |-> ys ** normalized |-> ns ** buffer |-> scratch **
      pure (rotated_prefix x ys wr xr data (t * 24))
    decreases (7 - !row)
  {
    let t = !row;
    let mut group = 0sz;
    while (!group <^ 8sz)
      invariant exists* (g:sz{g <= 8}) (ys:bf16_values (8 * 4096))
        (ns:lseq bf16 128) (scratch:lseq F32.t 288).
        group |-> g ** output |-> ys ** normalized |-> ns ** buffer |-> scratch **
        pure (rotated_prefix x ys wr xr data (t * 24 + g * 3))
      decreases (8 - !group)
    {
      let g = !group;
      let mut head = 0sz;
      while (!head <^ 3sz)
        invariant exists* (h:sz{h <= 3}) (ys:bf16_values (8 * 4096))
          (ns:lseq bf16 128) (scratch:lseq F32.t 288).
          head |-> h ** output |-> ys ** normalized |-> ns ** buffer |-> scratch **
          pure (rotated_prefix x ys wr xr data (t * 24 + g * 3 + h))
        decreases (3 - !head)
      {
        let h = !head;
        with ys. assert output |-> ys;
        with ns. assert normalized |-> ns;
        with scratch. assert buffer |-> scratch;
        normalize_rotate_head input output weights table normalized buffer position t g h
          #x #(hide (reveal ys <: bf16_values (8 * 4096))) #w #rt
          #(hide (reveal ns <: lseq bf16 128)) #(hide (reveal scratch <: lseq F32.t 288))
          #fx #fw #ft #xr #wr #data;
        head := h +^ 1sz;
      };
      let finished = !head;
      source_loop_end finished 3sz;
      group := g +^ 1sz;
    };
    let finished = !group;
    source_loop_end finished 8sz;
    row := t +^ 1sz;
  };
  let finished = !row;
  source_loop_end finished 7sz;
  with ys. assert output |-> ys;
  rotated_finish x ys wr xr data;
  ()
}
#pop-options

open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }
open Pulse.Lib.Trade

module Tensor = Kuiper.Tensor
module Chest = Kuiper.Chest
module MMA = Kuiper.TensorCore.MMA

inline_for_extraction noextract
instance source_c8 : concrete_sz 8 = { x = 8sz }

inline_for_extraction noextract
let dense_width (width:nat) : prop =
  0 < width /\ width <= 3072 /\ width % 16 = 0

inline_for_extraction noextract
let rec dense_mma_fold (width:nat{dense_width width})
  (a:lseq bf16 (16 * width)) (b:lseq bf16 (8 * width))
  (steps:nat{steps <= width / 16})
  : GTot (Chest.chest2 F32.t 16 8) (decreases steps) =
  if steps = 0 then Chest.const (16 @| 8 @| INil) (ObF.zero #F32.t)
  else MMA.emma (dense_mma_fold width a b (steps - 1))
    (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 (steps - 1))
    (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 (steps - 1) 0)

inline_for_extraction noextract
let dense_mma_step (width:nat{dense_width width})
  (a:lseq bf16 (16 * width)) (b:lseq bf16 (8 * width))
  (step:nat{step < width / 16})
  : Lemma
      (dense_mma_fold width a b (step + 1) ==
        MMA.emma (dense_mma_fold width a b step)
          (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 step)
          (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 step 0)) = ()

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn dense_source_mma
  (width:szp{dense_width width})
  (a:larray bf16 (16 * width)) (b:larray bf16 (8 * width))
  (c:larray F32.t 128)
  (#av:erased (lseq bf16 (16 * width))) (#bv:erased (lseq bf16 (8 * width)))
  (#cv:erased (lseq F32.t 128)) (#fa #fb:perm)
  preserves gpu ** a |-> Frac fa av ** b |-> Frac fb bv
  requires c |-> Frac (1.0R /. 32.0R) cv
  ensures c |-> Frac (1.0R /. 32.0R)
    (Tensor.to_seq (rm 16 8) (dense_mma_fold width av bv (width / 16)))
{
  Tensor.tensor_abs' (rm 16 width) a;
  Tensor.tensor_abs' (cm width 8) b;
  Tensor.tensor_abs' (rm 16 8) c;
  let am = Tensor.from_array (rm 16 width) a;
  let bm = Tensor.from_array (cm width 8) b;
  let cmatrix = Tensor.from_array (rm 16 8) c;
  assert rewrites_to am (Tensor.from_array (rm 16 width) a);
  assert rewrites_to bm (Tensor.from_array (cm width 8) b);
  assert rewrites_to cmatrix (Tensor.from_array (rm 16 8) c);
  let fragment = MMA.alloc_fragment ();
  MMA.fill fragment (ObF.zero #F32.t);
  let mut step = 0sz;
  while (!step <^ width /^ 16sz)
    invariant exists* (k:sz{k <= width / 16}).
      step |-> k ** fragment |-> dense_mma_fold width av bv k
    decreases (width / 16 - !step)
  {
    let k = !step;
    let at = array2_extract_tile_ro' am 16 16 0 (SZ.v k);
    let bt = array2_extract_tile_ro' bm 16 8 (SZ.v k) 0;
    MMA.mma_sync at bt fragment;
    elim_trade
      (at |-> Frac fa (ematrix_subtile (Tensor.from_seq (rm 16 width) av) 16 16 0 k))
      (am |-> Frac fa (Tensor.from_seq (rm 16 width) av));
    elim_trade
      (bt |-> Frac fb (ematrix_subtile (Tensor.from_seq (cm width 8) bv) 16 8 k 0))
      (bm |-> Frac fb (Tensor.from_seq (cm width 8) bv));
    dense_mma_step width av bv k;
    step := k +^ 1sz;
  };
  let finished = !step;
  source_loop_end finished (width /^ 16sz);
  MMA.store fragment cmatrix;
  drop_ (fragment |-> dense_mma_fold width av bv (width / 16));
  Tensor.tensor_concr am;
  Tensor.tensor_concr bm;
  Tensor.tensor_concr cmatrix;
  rewrite each Tensor.core am as a;
  rewrite each Tensor.core bm as b;
  rewrite each Tensor.core cmatrix as c;
}
#pop-options

open Kuiper

inline_for_extraction noextract
let slice_outside (start count i:nat) : prop =
  i < start \/ start + count <= i

inline_for_extraction noextract
let slice_remainder (#n:nat) (arr:larray bf16 n) (f:perm)
  (start count:nat) (original:lseq bf16 n) : slprop =
  exists* (contents:lseq (option bf16) n).
    Pulse.Lib.Array.pts_to_mask arr #f contents (slice_outside start count) **
    pure (forall (i:Shared.index n). Seq.index contents i == Some (Seq.index original i))

inline_for_extraction noextract
fn borrow_source_slice
  (#n:erased nat) (arr:larray bf16 n)
  (start:sz) (count:sz{start + count <= n})
  (#original:erased (lseq bf16 n)) (#f:perm)
  requires arr |-> Frac f original
  returns sub:larray bf16 count
  ensures pure (sub == Pulse.Lib.Array.gsub arr start (start + count)) **
    sub |-> Frac f (Seq.slice original start (start + count)) **
    slice_remainder arr f start count original
{
  Pulse.Lib.Array.to_mask arr;
  let sub = Pulse.Lib.Array.sub arr #f start (hide (start + count));
  Pulse.Lib.Array.from_mask sub;
  with values. assert sub |-> Frac f values;
  assert pure (Seq.equal values (Seq.slice original start (start + count)));
  rewrite each (fun (i:nat) -> ~(SZ.v start <= i /\ i < SZ.v start + SZ.v count))
    as slice_outside start count;
  fold slice_remainder arr f start count original;
  sub
}

inline_for_extraction noextract
ghost
fn restore_source_slice
  (#n:erased nat) (arr:larray bf16 n)
  (start:sz) (count:sz{start + count <= n})
  (sub:larray bf16 count{sub == Pulse.Lib.Array.gsub arr start (start + count)})
  (#original:erased (lseq bf16 n)) (#f:perm)
  requires slice_remainder arr f start count original **
    sub |-> Frac f (Seq.slice original start (start + count))
  ensures arr |-> Frac f original
{
  unfold slice_remainder arr f start count original;
  Pulse.Lib.Array.to_mask sub;
  rewrite each sub as Pulse.Lib.Array.gsub arr start (start + count);
  Pulse.Lib.Array.return_sub arr;
  Pulse.Lib.Array.from_mask arr;
  with result. assert arr |-> Frac f result;
  assert pure (Seq.equal result original);
}

inline_for_extraction noextract
let dense_real_term (width:pos) (a b:Shared.vector width) (i:nat) : GTot real =
  if i < width then a i *. b i else 0.0R

inline_for_extraction noextract
let dense_left_operand_at (width:nat{dense_width width}) (a:lseq bf16 (16 * width))
  (step:nat{step < width / 16}) (row:Shared.index 16) (lane:Shared.index 16)
  : Lemma
      (Chest.acc2 (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 step) row lane ==
        Seq.index a (row * width + step * 16 + lane)) = ()

inline_for_extraction noextract
let dense_right_operand_at (width:nat{dense_width width}) (b:lseq bf16 (8 * width))
  (step:nat{step < width / 16}) (lane:Shared.index 16) (col:Shared.index 8)
  : Lemma
      (Chest.acc2 (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 step 0) lane col ==
        Seq.index b (col * width + step * 16 + lane)) = ()

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let rec dense_mma_cell_approx
  (width:nat{dense_width width})
  (a:lseq bf16 (16 * width)) (b:lseq bf16 (8 * width))
  (row:Shared.index 16) (col:Shared.index 8)
  (ar br:Shared.vector width) (steps:nat{steps <= width / 16})
  : Lemma
      (requires
        (forall (i:Shared.index width). Seq.index a (row * width + i) %~ ar i) /\
        (forall (i:Shared.index width). Seq.index b (col * width + i) %~ br i))
      (ensures Chest.acc2 (dense_mma_fold width a b steps) row col %~
        source_sum_n (16 * steps) (dense_real_term width ar br))
      (decreases steps) =
  if steps = 0 then ()
  else (
    let k = steps - 1 in
    let ra = Seq.init_ghost 16 (fun i -> ar (16 * k + i)) in
    let rb = Seq.init_ghost 16 (fun i -> br (16 * k + i)) in
    dense_mma_cell_approx width a b row col ar br k;
    let left_at (i:Shared.index 16)
      : Lemma
          (Chest.acc2 (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 k) row i %~
            Seq.index ra i) =
      dense_left_operand_at width a k row i
    in FStar.Classical.forall_intro left_at;
    let right_at (i:Shared.index 16)
      : Lemma
          (Chest.acc2 (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 k 0) i col %~
            Seq.index rb i) =
      dense_right_operand_at width b k i col
    in FStar.Classical.forall_intro right_at;
    source_mma_cell_approx
      (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 k)
      (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 k 0)
      (dense_mma_fold width a b k)
      row col ra rb
      (source_sum_n (16 * k) (dense_real_term width ar br));
    source_sum_ext 16 (source_dot16 ra rb)
      (fun i -> dense_real_term width ar br (16 * k + i));
    source_sum_append (16 * k) 16 (dense_real_term width ar br);
    dense_mma_step width a b k
  )

inline_for_extraction noextract
let dense_mma_dot
  (width:nat{dense_width width})
  (a:lseq bf16 (16 * width)) (b:lseq bf16 (8 * width))
  (row:Shared.index 16) (col:Shared.index 8) (ar br:Shared.vector width)
  : Lemma
      (requires
        (forall (i:Shared.index width). Seq.index a (row * width + i) %~ ar i) /\
        (forall (i:Shared.index width). Seq.index b (col * width + i) %~ br i))
      (ensures Chest.acc2 (dense_mma_fold width a b (width / 16)) row col %~ Shared.dot ar br) =
  dense_mma_cell_approx width a b row col ar br (width / 16);
  source_sum_shared width (dense_real_term width ar br);
  sum_extensional width (dense_real_term width ar br) (fun i -> ar i *. br i)
#pop-options

inline_for_extraction noextract
let projection_layout (cols layout:nat) : prop =
  (layout = 0 /\ (cols = 1024)) \/ (layout = 1 /\ cols = 4096) \/
  (layout = 2 /\ cols = 6144)

inline_for_extraction noextract
let projection_row (cols:nat) (layout:nat{projection_layout cols layout})
  (col:Shared.index cols) : Shared.index cols =
  if layout = 1 then
    let group = col / 512 in
    let inner = col % 512 in
    if inner < 256 then group * 256 + inner
    else if inner < 384 then 2048 + group * 128 + inner - 256
    else 3072 + group * 128 + inner - 384
  else if layout = 2 then
    col / 128 * 64 + col % 64 + (if col % 128 >= 64 then 3072 else 0)
  else col

inline_for_extraction noextract
let projection_row_source
  (cols:szp) (layout:sz{projection_layout cols layout}) (col:szlt cols)
  : Tot (result:szlt cols{SZ.v result == projection_row cols layout col}) =
  if layout = 1sz then
    let group = col /^ 512sz in
    let inner = col %^ 512sz in
    if inner <^ 256sz then group *^ 256sz +^ inner
    else if inner <^ 384sz then 2048sz +^ group *^ 128sz +^ inner -^ 256sz
    else 3072sz +^ group *^ 128sz +^ inner -^ 384sz
  else if layout = 2sz then
    let group = col /^ 128sz in
    let base = group *^ 64sz in
    let lane = col %^ 64sz in
    let inner = col %^ 128sz in
    if inner <^ 64sz then base +^ lane +^ 0sz else base +^ lane +^ 3072sz
  else col

inline_for_extraction noextract
let projection_block_contiguous
  (cols:nat) (layout:nat{projection_layout cols layout})
  (block:nat{block < cols /\ block % 16 = 0}) (row:Shared.index 16)
  : Lemma
      (projection_row cols layout block + 16 <= cols /\
        projection_row cols layout (block + row) == projection_row cols layout block + row) =
  ()

inline_for_extraction noextract
let projection_scores
  (width:nat{dense_width width}) (cols:nat)
  (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (x:lseq bf16 (8 * width)) (weights:bf16_values weight_elements)
  (block:nat{block < cols /\ block % 16 = 0})
  : GTot (Chest.chest2 F32.t 16 8) =
  projection_block_contiguous cols layout block 0;
  let start = offset + projection_row cols layout block * width in
  dense_mma_fold width (Seq.slice weights start (start + 16 * width)) x (width / 16)

inline_for_extraction noextract
let projection_cell
  (#nr:nat) (width:nat{dense_width width}) (cols:nat)
  (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:bf16_values weight_elements) (residual:lseq bf16 nr)
  (row:Shared.index 8) (col:Shared.index cols) : GTot bf16 =
  let scores = projection_scores width cols layout offset x weights (col / 16 * 16) in
  let rounded = Casts.fcast #F32.t #bf16 (Chest.acc2 scores (col % 16) row) in
  if add_residual then ObF.add rounded (Seq.index residual (row * cols + col)) else rounded

inline_for_extraction noextract
let indexed_matrix (#a:Type0) (rows cols:pos)
  (f:Shared.index rows -> Shared.index cols -> GTot a) : GTot (lseq a (rows * cols)) =
  Seq.init_ghost (rows * cols) (fun i -> f (i / cols) (i % cols))

inline_for_extraction noextract
let indexed_matrix_at (#a:Type0) (rows cols:pos)
  (f:Shared.index rows -> Shared.index cols -> GTot a)
  (row:Shared.index rows) (col:Shared.index cols)
  : Lemma (Seq.index (indexed_matrix rows cols f) (row * cols + col) == f row col) =
  assert ((row * cols + col) / cols == row /\ (row * cols + col) % cols == col)

inline_for_extraction noextract
let projection_values
  (#nr:nat) (width:nat{dense_width width}) (cols:nat)
  (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:bf16_values weight_elements) (residual:lseq bf16 nr)
  : GTot (lseq bf16 (8 * cols)) =
  indexed_matrix 8 cols (projection_cell width cols layout offset add_residual x weights residual)

inline_for_extraction noextract
let projection_weights_rep
  (width:nat{dense_width width}) (cols:nat)
  (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (weights:bf16_values weight_elements) (logical:Shared.matrix cols width) : prop =
  forall (col:Shared.index cols) (k:Shared.index width).
    Seq.index weights (offset + projection_row cols layout col * width + k) %~ logical col k

inline_for_extraction noextract
let projection_math (width cols:pos) (weights:Shared.matrix cols width)
  (x:Shared.matrix 8 width) (residual:Shared.matrix 8 cols) (add_residual:bool)
  : GTot (Shared.matrix 8 cols) =
  fun row col ->
    if add_residual then Shared.linear weights (x row) col +. residual row col
    else Shared.linear weights (x row) col

inline_for_extraction noextract
let shared_dot_commute (#n:pos) (x y:Shared.vector n)
  : Lemma (Shared.dot x y == Shared.dot y x) =
  sum_extensional n (fun i -> x i *. y i) (fun i -> y i *. x i)

inline_for_extraction noextract
let projection_math_at (width cols:pos) (weights:Shared.matrix cols width)
  (x:Shared.matrix 8 width) (residual:Shared.matrix 8 cols) (add_residual:bool)
  (row:Shared.index 8) (col:Shared.index cols)
  : Lemma
      (projection_math width cols weights x residual add_residual row col ==
        (if add_residual then Shared.dot (x row) (weights col) +. residual row col
         else Shared.dot (x row) (weights col))) = ()

inline_for_extraction noextract
let projection_values_at
  (#nr:nat) (width:nat{dense_width width}) (cols:pos)
  (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:bf16_values weight_elements) (residual:lseq bf16 nr)
  (row:Shared.index 8) (col:Shared.index cols)
  : Lemma
      (Seq.index (projection_values width cols layout offset add_residual x weights residual)
          (row * cols + col) ==
        projection_cell width cols layout offset add_residual x weights residual row col) =
  indexed_matrix_at 8 cols (projection_cell width cols layout offset add_residual x weights residual) row col

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let projection_refines
  (#nr:nat) (width:nat{dense_width width}) (cols:pos)
  (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:bf16_values weight_elements) (residual:lseq bf16 nr)
  (xr:Shared.matrix 8 width) (wr:Shared.matrix cols width) (rr:Shared.matrix 8 cols)
  : Lemma
      (requires
        matrix_rep x xr /\ projection_weights_rep width cols layout offset weights wr /\
        (add_residual ==> (forall (row:Shared.index 8) (col:Shared.index cols).
          Seq.index residual (row * cols + col) %~ rr row col)))
      (ensures matrix_rep
        (projection_values width cols layout offset add_residual x weights residual)
        (projection_math width cols wr xr rr add_residual)) =
  let at (row:Shared.index 8) (col:Shared.index cols)
    : Lemma (Seq.index (projection_values width cols layout offset add_residual x weights residual)
        (row * cols + col) %~
        projection_math width cols wr xr rr add_residual row col) =
    let block = col / 16 * 16 in
    projection_block_contiguous cols layout block (col % 16);
    let start = offset + projection_row cols layout block * width in
    dense_mma_dot width (Seq.slice weights start (start + 16 * width)) x
      (col % 16) row (wr col) (xr row);
    let score = Chest.acc2 (projection_scores width cols layout offset x weights block) (col % 16) row in
    Casts.fcast_approx #F32.t #bf16 score (Shared.dot (wr col) (xr row));
    shared_dot_commute (wr col) (xr row);
    assert (Casts.fcast #F32.t #bf16 score %~ Shared.dot (xr row) (wr col));
    projection_math_at width cols wr xr rr add_residual row col;
    projection_values_at width cols layout offset add_residual x weights residual row col;
    if add_residual then (
      a_add (Casts.fcast #F32.t #bf16 score) (Seq.index residual (row * cols + col))
        (Shared.dot (xr row) (wr col)) (rr row col)
    )
  in
  FStar.Classical.forall_intro_2 at
#pop-options

inline_for_extraction noextract
ghost
fn warp_owner_intro (p:slprop)
  requires p
  ensures forall+ (tid:natlt 32). when_ (tid == 0) p
{
  forevery_singleton_intro' #(tid:natlt 32{tid == 0}) (fun _ -> p) 0;
  forevery_unrefine_pred #(natlt 32) (fun _ -> p) (fun tid -> tid == 0);
}

inline_for_extraction noextract
ghost
fn warp_owner_elim (p:slprop)
  requires forall+ (tid:natlt 32). when_ (tid == 0) p
  ensures p
{
  forevery_refine_pred #(natlt 32) (fun _ -> p) (fun tid -> tid == 0);
  forevery_singleton_elim' #(tid:natlt 32{tid == 0}) (fun _ -> p) 0;
}

(* Higher-order ghost arguments retain their typed heads during erasure. *)
ghost
fn warp_distribute_proof (#a:Type0) (buffer:array a)
  (#values:erased (seq a)) (#f:perm) ()
  requires forall+ (tid:natlt 32). when_ (tid == 0) (buffer |-> Frac f values)
  ensures forall+ (_:natlt 32). buffer |-> Frac (f /. 32.0R) values
{
  warp_owner_elim (buffer |-> Frac f values);
  Kuiper.Array.Extra.array_share buffer #values #f 32;
}

ghost
fn warp_gather_proof (#a:Type0) (buffer:array a)
  (#values:erased (seq a)) (#f:perm) ()
  requires forall+ (_:natlt 32). buffer |-> Frac (f /. 32.0R) values
  ensures forall+ (tid:natlt 32). when_ (tid == 0) (buffer |-> Frac f values)
{
  Kuiper.Array.Extra.array_gather buffer #values #f 32;
  warp_owner_intro (buffer |-> Frac f values);
}

inline_for_extraction noextract
let warp_distribute (#a:Type0) (buffer:array a)
  (#values:erased (seq a)) (#f:perm) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** when_ (reveal tid == 0) (buffer |-> Frac f values))
      (ensures fun _ -> thread_id 32 tid ** buffer |-> Frac (f /. 32.0R) values) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun i -> when_ (i == 0) (buffer |-> Frac f values))
    (fun _ -> buffer |-> Frac (f /. 32.0R) values)
    (warp_distribute_proof buffer #values #f ()) #32 #tid

inline_for_extraction noextract
let warp_gather (#a:Type0) (buffer:array a)
  (#values:erased (seq a)) (#f:perm) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** buffer |-> Frac (f /. 32.0R) values)
      (ensures fun _ -> thread_id 32 tid ** when_ (reveal tid == 0) (buffer |-> Frac f values)) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun _ -> buffer |-> Frac (f /. 32.0R) values)
    (fun i -> when_ (i == 0) (buffer |-> Frac f values))
    (warp_gather_proof buffer #values #f ()) #32 #tid

inline_for_extraction noextract
let native_projection_prefix (cols:pos) (ys:lseq (option bf16) (8 * cols))
  (expected:lseq bf16 (8 * cols))
  (block steps:nat) : prop =
  forall (row:Shared.index 8) (col:Shared.index cols).
    (col < block \/ (block <= col /\ col < block + 16 /\ row * 16 + col - block < steps)) ==>
    Seq.index ys (row * cols + col) == Some (Seq.index expected (row * cols + col))

inline_for_extraction noextract
let native_projection_write (cols:pos) (ys after:lseq (option bf16) (8 * cols))
  (expected:lseq bf16 (8 * cols))
  (block:nat{block + 16 <= cols}) (row:Shared.index 8) (col:Shared.index 16) (value:bf16)
  : Lemma
      (requires
        native_projection_prefix cols ys expected block (row * 16 + col) /\
        value == Seq.index expected (row * cols + block + col) /\
        after == Seq.upd ys (row * cols + block + col) (Some value))
      (ensures native_projection_prefix cols after expected block (row * 16 + col + 1)) =
  let at (r:Shared.index 8) (c:Shared.index cols)
    : Lemma
        ((c < block \/ (block <= c /\ c < block + 16 /\ r * 16 + c - block < row * 16 + col + 1)) ==>
          Seq.index after (r * cols + c) == Some (Seq.index expected (r * cols + c))) =
    if r = row && c = block + col then assert (Seq.index after (r * cols + c) == Some value)
    else assert (Seq.index after (r * cols + c) == Seq.index ys (r * cols + c))
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let native_projection_advance (cols:pos) (ys:lseq (option bf16) (8 * cols))
  (expected:lseq bf16 (8 * cols))
  (block:nat{block + 16 <= cols})
  : Lemma
      (requires native_projection_prefix cols ys expected block 128)
      (ensures native_projection_prefix cols ys expected (block + 16) 0) = ()

inline_for_extraction noextract
let native_projection_finished (cols:pos) (ys:lseq (option bf16) (8 * cols))
  (expected:lseq bf16 (8 * cols))
  : Lemma
      (requires native_projection_prefix cols ys expected cols 0)
      (ensures forall (i:Shared.index (8 * cols)).
        Seq.index ys i == Some (Seq.index expected i)) =
  let at (i:Shared.index (8 * cols))
    : Lemma (Seq.index ys i == Some (Seq.index expected i)) =
    assert (i == i / cols * cols + i % cols);
    assert (Seq.index ys (i / cols * cols + i % cols) ==
      Some (Seq.index expected (i / cols * cols + i % cols)))
  in
  FStar.Classical.forall_intro at

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn write_projection_tile
  (#nr:erased nat) (cols:szp) (block:sz{block + 16 <= cols})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (tile:larray F32.t 128) (residual:larray bf16 nr) (output:larray bf16 (8 * cols))
  (#tc:erased (lseq F32.t 128)) (#rs:erased (lseq bf16 nr))
  (#old:erased (lseq (option bf16) (8 * cols)))
  (#expected:erased (lseq bf16 (8 * cols))) (#fr:perm)
  preserves gpu ** tile |-> tc ** residual |-> Frac fr rs
  requires Pulse.Lib.Array.pts_to_mask output old (fun _ -> True) **
    pure (native_projection_prefix cols old expected block 0 /\
      (forall (row:Shared.index 8) (col:Shared.index 16).
        Seq.index expected (row * cols + block + col) ==
          (let rounded = Casts.fcast #F32.t #bf16 (Seq.index tc (col * 8 + row)) in
           if add_residual then ObF.add rounded (Seq.index rs (row * cols + block + col))
           else rounded)))
  ensures exists* (ys:lseq (option bf16) (8 * cols)).
    Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True) **
    pure (native_projection_prefix cols ys expected (block + 16) 0)
{
  let mut row = 0sz;
  while (!row <^ 8sz)
    invariant exists* (r:sz{r <= 8}) (ys:lseq (option bf16) (8 * cols)).
      row |-> r ** Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True) **
      pure (native_projection_prefix cols ys expected block (r * 16))
    decreases (8 - !row)
  {
    let r = !row;
    let mut col = 0sz;
    while (!col <^ 16sz)
      invariant exists* (c:sz{c <= 16}) (ys:lseq (option bf16) (8 * cols)).
        col |-> c ** Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True) **
        pure (native_projection_prefix cols ys expected block (r * 16 + c))
      decreases (16 - !col)
    {
      let c = !col;
      let acc = tile.(c *^ 8sz +^ r);
      let rounded = Casts.fcast #F32.t #bf16 acc;
      let value:bf16 = if add_residual {
        let incoming = residual.(r *^ cols +^ block +^ c);
        (ObF.add rounded incoming <: bf16)
      } else { (rounded <: bf16) };
      assert pure (value == Seq.index expected (r * cols + block + c));
      with (before:seq (option bf16)).
        assert Pulse.Lib.Array.pts_to_mask output before (fun _ -> True);
      Pulse.Lib.Array.mask_write output (r *^ cols +^ block +^ c) value;
      with (after:seq (option bf16)).
        assert Pulse.Lib.Array.pts_to_mask output after (fun _ -> True);
      native_projection_write cols before after expected block r c value;
      col := c +^ 1sz;
    };
    let finished = !col;
    source_loop_end finished 16sz;
    row := r +^ 1sz;
  };
  let finished = !row;
  source_loop_end finished 8sz;
  with ys. assert Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True);
  native_projection_advance cols ys expected block;
  ()
}
#pop-options

inline_for_extraction noextract
let source_tile_sequence_at (tile:Chest.chest2 F32.t 16 8)
  (row:Shared.index 16) (col:Shared.index 8)
  : Lemma (Seq.index (Tensor.to_seq (rm 16 8) tile) (row * 8 + col) ==
      Chest.acc2 tile row col) = ()

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let projection_tile_expected
  (#nr:nat) (width:nat{dense_width width}) (cols:pos)
  (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:bf16_values weight_elements) (residual:lseq bf16 nr)
  (block:nat{block < cols /\ block % 16 = 0})
  : Lemma
      (let tc = Tensor.to_seq (rm 16 8) (projection_scores width cols layout offset x weights block) in
       let expected = projection_values width cols layout offset add_residual x weights residual in
       forall (row:Shared.index 8) (col:Shared.index 16).
         Seq.index expected (row * cols + block + col) ==
           (let rounded = Casts.fcast #F32.t #bf16 (Seq.index tc (col * 8 + row)) in
            if add_residual then ObF.add rounded (Seq.index residual (row * cols + block + col))
            else rounded)) =
  let tc = Tensor.to_seq (rm 16 8) (projection_scores width cols layout offset x weights block) in
  let expected = projection_values width cols layout offset add_residual x weights residual in
  let at (row:Shared.index 8) (col:Shared.index 16)
    : Lemma
        (Seq.index expected (row * cols + block + col) ==
          (let rounded = Casts.fcast #F32.t #bf16 (Seq.index tc (col * 8 + row)) in
           if add_residual then ObF.add rounded (Seq.index residual (row * cols + block + col))
           else rounded)) =
    assert ((block + col) / 16 * 16 == block /\ (block + col) % 16 == col);
    projection_values_at width cols layout offset add_residual x weights residual row (block + col);
    source_tile_sequence_at (projection_scores width cols layout offset x weights block) col row
  in FStar.Classical.forall_intro_2 at
#pop-options

inline_for_extraction noextract
let previous_projection_tile
  (width:nat{dense_width width}) (cols:pos) (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (x:lseq bf16 (8 * width)) (weights:bf16_values weight_elements) (initial:lseq F32.t 128)
  (block:nat{block <= cols /\ block % 16 = 0}) : GTot (lseq F32.t 128) =
  if block = 0 then initial
  else Tensor.to_seq (rm 16 8) (projection_scores width cols layout offset x weights (block - 16))

inline_for_extraction noextract
let projection_output_owned
  (cols:pos) (output:larray bf16 (8 * cols)) (expected:lseq bf16 (8 * cols)) (block:nat) : slprop =
  exists* (ys:lseq (option bf16) (8 * cols)).
    Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True) **
    pure (native_projection_prefix cols ys expected block 0)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn projection_owner_commit
  (#nr:erased nat) (cols:szp) (block:sz{block + 16 <= cols})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (tile:larray F32.t 128) (residual:larray bf16 nr) (output:larray bf16 (8 * cols))
  (tid:szlt 32) (#tc:erased (lseq F32.t 128)) (#rs:erased (lseq bf16 nr))
  (#expected:erased (lseq bf16 (8 * cols))) (#fr:perm)
  preserves gpu ** residual |-> Frac fr rs **
    when_ (SZ.v tid == 0) (tile |-> tc)
  requires when_ (SZ.v tid == 0) (projection_output_owned cols output expected block) **
    pure (forall (row:Shared.index 8) (col:Shared.index 16).
      Seq.index expected (row * cols + block + col) ==
        (let rounded = Casts.fcast #F32.t #bf16 (Seq.index tc (col * 8 + row)) in
         if add_residual then ObF.add rounded (Seq.index rs (row * cols + block + col))
         else rounded))
  ensures when_ (SZ.v tid == 0) (projection_output_owned cols output expected (block + 16))
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (tile |-> tc) as tile |-> tc;
    rewrite when_ (SZ.v tid == 0) (projection_output_owned cols output expected block)
      as projection_output_owned cols output expected block;
    unfold projection_output_owned cols output expected block;
    with ys. assert Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True);
    write_projection_tile cols block add_residual tile residual output
      #tc #rs #(hide (reveal ys <: lseq (option bf16) (8 * cols))) #expected #fr;
    fold projection_output_owned cols output expected (block + 16);
    rewrite tile |-> tc as when_ (SZ.v tid == 0) (tile |-> tc);
    rewrite projection_output_owned cols output expected (block + 16) as
      when_ (SZ.v tid == 0) (projection_output_owned cols output expected (block + 16));
  } else {
    rewrite when_ (SZ.v tid == 0) (projection_output_owned cols output expected block) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (projection_output_owned cols output expected (block + 16));
  };
}

inline_for_extraction noextract
fn projection_kernel_body
  (#nr:erased nat) (width:szp{dense_width width}) (cols:szp)
  (layout:sz{projection_layout cols layout})
  (offset:sz{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (input:larray bf16 (8 * width)) (weights:bf16_array weight_elements)
  (residual:larray bf16 nr) (output:larray bf16 (8 * cols)) (tile:larray F32.t 128)
  (tid:szlt 32)
  (#x:erased (lseq bf16 (8 * width))) (#w:erased (bf16_values weight_elements))
  (#rs:erased (lseq bf16 nr))
  (#cv:erased (lseq F32.t 128)) (#fx #fw #fr:perm)
  preserves gpu ** thread_id 32 tid **
    input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs
  requires tile |-> Frac (1.0R /. 32.0R) cv **
    when_ (SZ.v tid == 0) (Pulse.Lib.Array.pts_to_uninit output (8 * cols))
  ensures tile |-> Frac (1.0R /. 32.0R)
    (previous_projection_tile width cols layout offset x w cv cols) **
    when_ (SZ.v tid == 0)
      (output |-> projection_values width cols layout offset add_residual x w rs)
{
  let expected = hide (projection_values width cols layout offset add_residual x w rs);
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (Pulse.Lib.Array.pts_to_uninit output (8 * cols))
      as Pulse.Lib.Array.pts_to_uninit output (8 * cols);
    unfold Pulse.Lib.Array.pts_to_uninit output (8 * cols);
    fold projection_output_owned cols output expected 0;
    rewrite projection_output_owned cols output expected 0 as
      when_ (SZ.v tid == 0) (projection_output_owned cols output expected 0);
  } else {
    rewrite when_ (SZ.v tid == 0) (Pulse.Lib.Array.pts_to_uninit output (8 * cols)) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (projection_output_owned cols output expected 0);
  };
  let mut block = 0sz;
  while (!block <^ cols)
    invariant exists* (b:sz{b <= cols /\ b % 16 = 0}).
      block |-> b **
      tile |-> Frac (1.0R /. 32.0R) (previous_projection_tile width cols layout offset x w cv b) **
      when_ (SZ.v tid == 0) (projection_output_owned cols output expected b)
    decreases (cols - !block)
  {
    let b = !block;
    projection_block_contiguous cols layout b 0;
    let weight_row = projection_row_source cols layout b;
    let start = offset +^ weight_row *^ width;
    let count = 16sz *^ width;
    let a = borrow_source_slice weights start count #w #fw;
    dense_source_mma width a input tile
      #(hide (Seq.slice w start (start + count))) #x
      #(hide (previous_projection_tile width cols layout offset x w cv b)) #fw #fx;
    restore_source_slice weights start count a #w #fw;
    let tc = hide (Tensor.to_seq (rm 16 8) (projection_scores width cols layout offset x w b));
    assert tile |-> Frac (1.0R /. 32.0R) tc;
    warp_gather tile #(hide (reveal tc <: seq F32.t)) #1.0R #(hide (SZ.v tid));
    projection_tile_expected width cols layout offset add_residual x w rs b;
    projection_owner_commit cols b add_residual tile residual output tid #tc #rs #expected #fr;
    warp_distribute tile #(hide (reveal tc <: seq F32.t)) #1.0R #(hide (SZ.v tid));
    let next = b +^ 16sz;
    assert pure (SZ.v next == SZ.v b + 16);
    rewrite when_ (SZ.v tid == 0) (projection_output_owned cols output expected (b + 16))
      as when_ (SZ.v tid == 0) (projection_output_owned cols output expected next);
    block := next;
  };
  let finished = !block;
  source_loop_end finished cols;
  assert pure (SZ.v finished == SZ.v cols);
  rewrite when_ (SZ.v tid == 0) (projection_output_owned cols output expected finished)
    as when_ (SZ.v tid == 0) (projection_output_owned cols output expected cols);
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (projection_output_owned cols output expected cols)
      as projection_output_owned cols output expected cols;
    unfold projection_output_owned cols output expected cols;
    with masked. assert Pulse.Lib.Array.pts_to_mask output masked (fun _ -> True);
    native_projection_finished cols masked expected;
    Pulse.Lib.Array.from_mask output;
    with ys. assert output |-> ys;
    assert pure (Seq.equal ys expected);
    assert output |-> expected;
    rewrite output |-> expected as when_ (SZ.v tid == 0) (output |-> expected);
  } else {
    rewrite when_ (SZ.v tid == 0) (projection_output_owned cols output expected cols) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (output |-> expected);
  };
}

ghost
fn projection_setup
  (#ni #nr #no:erased nat)
  (input:larray bf16 ni) (weights:bf16_array weight_elements)
  (residual:larray bf16 nr) (output:larray bf16 no) (tile:larray F32.t 128)
  (#x:erased (lseq bf16 ni)) (#w:erased (bf16_values weight_elements))
  (#rs:erased (lseq bf16 nr)) (#cv:erased (lseq F32.t 128))
  (#fx #fw #fr:perm) ()
  norewrite
  requires input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs **
    tile |-> cv ** Pulse.Lib.Array.pts_to_uninit output no
  ensures (forall+ (tid:natlt 32).
    (input |-> Frac (fx /. 32.0R) x ** weights |-> Frac (fw /. 32.0R) w **
      residual |-> Frac (fr /. 32.0R) rs) **
    tile |-> Frac (1.0R /. 32.0R) cv **
    when_ (tid == 0) (Pulse.Lib.Array.pts_to_uninit output no)) ** emp
{
  Kuiper.Array.Extra.array_share input #x #fx 32;
  Kuiper.Array.Extra.array_share weights #w #fw 32;
  Kuiper.Array.Extra.array_share residual #rs #fr 32;
  Kuiper.Array.Extra.array_share tile #cv #1.0R 32;
  warp_owner_intro (Pulse.Lib.Array.pts_to_uninit output no);
  forevery_zip3
    (fun (_:natlt 32) -> input |-> Frac (fx /. 32.0R) x)
    (fun (_:natlt 32) -> weights |-> Frac (fw /. 32.0R) w)
    (fun (_:natlt 32) -> residual |-> Frac (fr /. 32.0R) rs);
  forevery_zip3
    (fun (_:natlt 32) ->
      input |-> Frac (fx /. 32.0R) x ** weights |-> Frac (fw /. 32.0R) w **
        residual |-> Frac (fr /. 32.0R) rs)
    (fun (_:natlt 32) -> tile |-> Frac (1.0R /. 32.0R) cv)
    (fun (tid:natlt 32) -> when_ (tid == 0) (Pulse.Lib.Array.pts_to_uninit output no));
}

ghost
fn projection_teardown
  (#ni #nr #no:erased nat)
  (input:larray bf16 ni) (weights:bf16_array weight_elements)
  (residual:larray bf16 nr) (output:larray bf16 no) (tile:larray F32.t 128)
  (#x:erased (lseq bf16 ni)) (#w:erased (bf16_values weight_elements))
  (#rs:erased (lseq bf16 nr)) (#ys:erased (lseq bf16 no)) (#cv:erased (lseq F32.t 128))
  (#fx #fw #fr:perm) ()
  norewrite
  requires (forall+ (tid:natlt 32).
    (input |-> Frac (fx /. 32.0R) x ** weights |-> Frac (fw /. 32.0R) w **
      residual |-> Frac (fr /. 32.0R) rs) **
    tile |-> Frac (1.0R /. 32.0R) cv ** when_ (tid == 0) (output |-> ys)) ** emp
  ensures input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs **
    tile |-> cv ** output |-> ys
{
  forevery_unzip3
    (fun (_:natlt 32) ->
      input |-> Frac (fx /. 32.0R) x ** weights |-> Frac (fw /. 32.0R) w **
        residual |-> Frac (fr /. 32.0R) rs)
    (fun (_:natlt 32) -> tile |-> Frac (1.0R /. 32.0R) cv)
    (fun (tid:natlt 32) -> when_ (tid == 0) (output |-> ys));
  forevery_unzip3
    (fun (_:natlt 32) -> input |-> Frac (fx /. 32.0R) x)
    (fun (_:natlt 32) -> weights |-> Frac (fw /. 32.0R) w)
    (fun (_:natlt 32) -> residual |-> Frac (fr /. 32.0R) rs);
  Kuiper.Array.Extra.array_gather input #x #fx 32;
  Kuiper.Array.Extra.array_gather weights #w #fw 32;
  Kuiper.Array.Extra.array_gather residual #rs #fr 32;
  Kuiper.Array.Extra.array_gather tile #cv #1.0R 32;
  warp_owner_elim (output |-> ys);
}

inline_for_extraction noextract
fn projection_thread
  (#nr:erased nat) (width:szp{dense_width width}) (cols:szp)
  (layout:sz{projection_layout cols layout})
  (offset:sz{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (input:larray bf16 (8 * width)) (weights:bf16_array weight_elements)
  (residual:larray bf16 nr) (output:larray bf16 (8 * cols)) (tile:larray F32.t 128)
  (#x:erased (lseq bf16 (8 * width))) (#w:erased (bf16_values weight_elements))
  (#rs:erased (lseq bf16 nr))
  (#cv:erased (lseq F32.t 128)) (#fx #fw #fr:perm)
  (tid:szlt 32) ()
  norewrite
  requires gpu **
    ((input |-> Frac (fx /. 32.0R) x ** weights |-> Frac (fw /. 32.0R) w **
      residual |-> Frac (fr /. 32.0R) rs) **
    tile |-> Frac (1.0R /. 32.0R) cv **
    when_ (SZ.v tid == 0) (Pulse.Lib.Array.pts_to_uninit output (8 * cols))) **
    thread_id 32 tid
  ensures gpu **
    ((input |-> Frac (fx /. 32.0R) x ** weights |-> Frac (fw /. 32.0R) w **
      residual |-> Frac (fr /. 32.0R) rs) **
    tile |-> Frac (1.0R /. 32.0R) (previous_projection_tile width cols layout offset x w cv cols) **
    when_ (SZ.v tid == 0) (output |-> projection_values width cols layout offset add_residual x w rs)) **
    thread_id 32 tid
{
  projection_kernel_body width cols layout offset add_residual input weights residual output tile tid
    #x #w #rs #cv #(fx /. 32.0R) #(fw /. 32.0R) #(fr /. 32.0R);
}

inline_for_extraction noextract
let projection_descriptor
  (#nr:erased nat) (width:szp{dense_width width}) (cols:szp)
  (layout:sz{projection_layout cols layout})
  (offset:sz{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (input:larray bf16 (8 * width){is_global_array input})
  (weights:bf16_array weight_elements{is_global_array weights})
  (residual:larray bf16 nr{is_global_array residual})
  (output:larray bf16 (8 * cols){is_global_array output})
  (tile:larray F32.t 128{is_global_array tile})
  (#x:erased (lseq bf16 (8 * width))) (#w:erased (bf16_values weight_elements))
  (#rs:erased (lseq bf16 nr))
  (#cv:erased (lseq F32.t 128)) (#fx #fw #fr:perm)
  : kernel_desc_1_n
      (input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs **
        tile |-> cv ** Pulse.Lib.Array.pts_to_uninit output (8 * cols))
      (input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs **
        tile |-> previous_projection_tile width cols layout offset x w cv cols **
        output |-> projection_values width cols layout offset add_residual x w rs) =
  {
    nthr = 32sz;
    frame = emp;
    kpre = (fun tid ->
      (input |-> Frac (fx /. 32.0R) x ** weights |-> Frac (fw /. 32.0R) w **
        residual |-> Frac (fr /. 32.0R) rs) **
      tile |-> Frac (1.0R /. 32.0R) cv **
      when_ (tid == 0) (Pulse.Lib.Array.pts_to_uninit output (8 * cols)));
    kpost = (fun tid ->
      (input |-> Frac (fx /. 32.0R) x ** weights |-> Frac (fw /. 32.0R) w **
        residual |-> Frac (fr /. 32.0R) rs) **
      tile |-> Frac (1.0R /. 32.0R) (hide (previous_projection_tile width cols layout offset x w cv cols)) **
      when_ (tid == 0) (output |-> hide (projection_values width cols layout offset add_residual x w rs)));
    block_setup = projection_setup input weights residual output tile #x #w #rs #cv #fx #fw #fr;
    block_teardown = projection_teardown input weights residual output tile #x #w #rs
      #(hide (projection_values width cols layout offset add_residual x w rs))
      #(hide (previous_projection_tile width cols layout offset x w cv cols)) #fx #fw #fr;
    f = projection_thread width cols layout offset add_residual input weights residual output tile
      #x #w #rs #cv #fx #fw #fr;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve
  }

inline_for_extraction noextract
fn run_projection
  (#nr:erased nat) (width:szp{dense_width width}) (cols:szp)
  (layout:sz{projection_layout cols layout})
  (offset:sz{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (input:larray bf16 (8 * width){is_global_array input})
  (weights:bf16_array weight_elements{is_global_array weights})
  (residual:larray bf16 nr{is_global_array residual})
  (output:larray bf16 (8 * cols){is_global_array output})
  (tile:larray F32.t 128{is_global_array tile})
  (#x:erased (lseq bf16 (8 * width))) (#w:erased (bf16_values weight_elements))
  (#rs:erased (lseq bf16 nr))
  (#cv:erased (lseq F32.t 128)) (#fx #fw #fr:perm)
  preserves cpu ** on gpu_loc (
    input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs)
  requires on gpu_loc (tile |-> cv ** Pulse.Lib.Array.pts_to_uninit output (8 * cols))
  ensures on gpu_loc (
    tile |-> previous_projection_tile width cols layout offset x w cv cols **
    output |-> projection_values width cols layout offset add_residual x w rs)
{
  launch_sync (projection_descriptor width cols layout offset add_residual input weights residual output tile
    #x #w #rs #cv #fx #fw #fr);
}
#pop-options

inline_for_extraction noextract
ghost
fn initialized_as_uninit
  (#n:erased nat) (output:larray bf16 n) (#values:erased (lseq bf16 n))
  requires on gpu_loc (output |-> values)
  ensures on gpu_loc (Pulse.Lib.Array.pts_to_uninit output n)
{
  ghost_impersonate gpu_loc
    (on gpu_loc (output |-> values))
    (on gpu_loc (Pulse.Lib.Array.pts_to_uninit output n))
    fn _ {
      on_elim (output |-> values);
      Pulse.Lib.Array.to_mask output;
      fold Pulse.Lib.Array.pts_to_uninit output n;
      on_intro (Pulse.Lib.Array.pts_to_uninit output n);
    };
}

inline_for_extraction noextract
let source_silu_coefficient () : F32.t = ObF.of_literal "1.44269504088896340736"

inline_for_extraction noextract
let intended_silu_exp2_scale () : real =
  source_log_two_nonzero ();
  1.0R /. ObExp.log 2.0R

#push-options "--fuel 2 --ifuel 0 --z3rlimit 120"
inline_for_extraction noextract
let source_silu_coefficient_approx ()
  : Lemma (source_silu_coefficient () %~ intended_silu_exp2_scale ()) =
  ObExp.log_two_positive ();
  ObExp.log_two_decimal_bounds ();
  assert (14426950408889632.0R /. 10000000000000000.0R <. intended_silu_exp2_scale ());
  assert (intended_silu_exp2_scale () <. 14426950408889635.0R /. 10000000000000000.0R);
  assert_norm (Kuiper.Float32.Literals.common_normal_cell "1.44269504088896340736"
    14426950408889632 10000000000000000 14426950408889635 10000000000000000
    12102203 (-23));
  F32.literal_interval_approx "1.44269504088896340736"
    14426950408889632 10000000000000000 14426950408889635 10000000000000000
    12102203 (-23) (intended_silu_exp2_scale ())

inline_for_extraction noextract
let silu_change_of_base (x:real)
  : Lemma (ObPow.exp2 ((0.0R -. x) *. intended_silu_exp2_scale ()) ==
      ObExp.exp (0.0R -. x)) =
  ObExp.log_two_positive ();
  ObExp.exp2_as_exp ((0.0R -. x) *. intended_silu_exp2_scale ())
#pop-options

inline_for_extraction noextract
let source_negate_all (value:F32.t)
  : Lemma (forall (r:real). value %~ r ==> source_negate value %~ (0.0R -. r)) =
  FStar.Classical.forall_intro (FStar.Classical.move_requires (source_negate_approx value))

inline_for_extraction noextract
let bf16_multiply_all (left right:bf16)
  : Lemma (forall (lr rr:real). left %~ lr /\ right %~ rr ==> ObF.mul left right %~ (lr *. rr)) =
  let at (lr rr:real)
    : Lemma (requires left %~ lr /\ right %~ rr)
        (ensures ObF.mul left right %~ (lr *. rr)) = a_mul left right lr rr
  in FStar.Classical.forall_intro_2 (FStar.Classical.move_requires_2 at)

inline_for_extraction noextract
let real_reciprocal_product (x:real) (d:real{d >. 0.0R})
  : Lemma (x *. (1.0R /. d) == x /. d) = ()

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn silu_source_value (gate up:bf16) (#gr #ur:erased real) (#guard:erased prop)
  preserves gpu
  requires pure (guard ==> gate %~ reveal gr /\ up %~ reveal ur)
  returns result:bf16
  ensures pure (guard ==> result %~ (Shared.silu (reveal gr) *. reveal ur))
{
  let v = Casts.fcast #bf16 #F32.t gate;
  bf16_to_f32_all gate;
  source_negate_all v;
  source_silu_coefficient_approx ();
  let argument = F32.mul_rn_ftz (source_negate v) (source_silu_coefficient ());
  assert pure (guard ==> argument %~ ((0.0R -. reveal gr) *. intended_silu_exp2_scale ()));
  let e = F32.exp2_approx_ftz argument;
  silu_change_of_base (reveal gr);
  assert pure (guard ==> v_approximates e (ObExp.exp (0.0R -. reveal gr)));
  let denominator = F32.add_rn_ftz (ObF.one #F32.t) e;
  assert pure (guard ==> denominator %~ (1.0R +. ObExp.exp (0.0R -. reveal gr)));
  let inverse = F32.rcp_approx_ftz denominator;
  assert pure (guard ==> inverse %~ (1.0R /. (1.0R +. ObExp.exp (0.0R -. reveal gr))));
  let value = F32.mul_rn_ftz v inverse;
  assert pure (guard ==> value %~
    (reveal gr *. (1.0R /. (1.0R +. ObExp.exp (0.0R -. reveal gr)))));
  real_reciprocal_product (reveal gr) (1.0R +. ObExp.exp (0.0R -. reveal gr));
  assert pure (guard ==> value %~ Shared.silu (reveal gr));
  f32_to_bf16_all value;
  let rounded = Casts.fcast #F32.t #bf16 value;
  bf16_multiply_all rounded up;
  ObF.mul rounded up
}
#pop-options

inline_for_extraction noextract
let activated_model (x:Shared.matrix 8 6144) : GTot (Shared.matrix 8 3072) =
  fun row col ->
    let i = col / 64 * 128 + col % 64 in
    Shared.silu (x row i) *. x row (i + 64)

inline_for_extraction noextract
let silu_source_index (col:szlt 3072)
  : Tot (result:sz{SZ.v result == SZ.v col / 64 * 128 + SZ.v col % 64}) =
  col +^ col /^ 64sz *^ 64sz

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn silu_kernel (input:larray bf16 49152) (output:larray bf16 24576)
  (#x:erased (lseq bf16 49152)) (#old:erased (lseq bf16 24576)) (#f:perm)
  (#xr:erased (Shared.matrix 8 6144)) (#guard:erased prop)
  preserves gpu ** input |-> Frac f x
  requires output |-> old ** pure (guard ==> matrix_rep x xr)
  ensures exists* (ys:lseq bf16 24576).
    output |-> ys ** pure (guard ==> matrix_rep ys (activated_model xr))
{
  let mut row = 0sz;
  while (!row <^ 8sz)
    invariant exists* (r:sz{r <= 8}) (ys:lseq bf16 24576).
      row |-> r ** output |-> ys **
      pure (guard ==> matrix_prefix ys (activated_model xr) r)
    decreases (8 - !row)
  {
    let r = !row;
    with previous. assert output |-> previous;
    let mut col = 0sz;
    while (!col <^ 3072sz)
      invariant exists* (c:sz{c <= 3072}) (ys:lseq bf16 24576).
        col |-> c ** output |-> ys **
        pure (segment_frame previous ys (r * 3072) c /\
          (forall (j:Shared.index 3072{j < c}).
            guard ==> Seq.index ys (r * 3072 + j) %~ activated_model xr r j))
      decreases (3072 - !col)
    {
      let c = !col;
      let i = silu_source_index c;
      let gate = input.(r *^ 6144sz +^ i);
      let up = input.(r *^ 6144sz +^ i +^ 64sz);
      let value = silu_source_value gate up
        #(hide ((reveal xr) r i)) #(hide ((reveal xr) r (i + 64))) #guard;
      with before. assert output |-> before;
      output.(r *^ 3072sz +^ c) <- value;
      with after. assert output |-> after;
      bf16_prefix_write before after (r * 3072) 3072 c (activated_model xr r) value guard;
      col := c +^ 1sz;
    };
    let finished = !col;
    source_loop_end finished 3072sz;
    with after. assert output |-> after;
    matrix_row_progress previous after (activated_model xr) r guard;
    row := r +^ 1sz;
  };
  let finished = !row;
  source_loop_end finished 8sz;
  ()
}
#pop-options

inline_for_extraction noextract
let attention_numerator
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 16) (col:Shared.index 64) : GTot real =
  if source_live row col then
    source_shared_shifted_numerator query key head row col
      (to_real (source_maximum row (source_score_row (source_mma_score_tile rotated head) row)))
  else 0.0R

inline_for_extraction noextract
let attention_denominator
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 16) : GTot real =
  if row < 14 then Shared.sum (row / 2 + 1) (attention_numerator rotated query key head row)
  else 1.0R

inline_for_extraction noextract
let attention_denominator_positive
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 16)
  : Lemma (attention_denominator rotated query key head row >. 0.0R) =
  if row < 14 then Shared.sum_positive (row / 2 + 1) (attention_numerator rotated query key head row)

inline_for_extraction noextract
let partition_entry (f:Shared.vector 64) (part:Shared.index 4) (step:nat) : GTot real =
  if step < 16 then f (source_column part step) else 0.0R

inline_for_extraction noextract
let rec source_sum_zero_tail (n:nat) (cut:nat{cut <= n}) (f:nat -> GTot real)
  : Lemma
      (requires forall (i:Shared.index n). cut <= i ==> f i == 0.0R)
      (ensures source_sum_n n f == source_sum_n cut f)
      (decreases n) =
  if cut < n then source_sum_zero_tail (n - 1) cut f

inline_for_extraction noextract
let partition_total (f:Shared.vector 64) (part:Shared.index 4)
  : Lemma
      (requires forall (i:Shared.index 64). 7 <= i ==> f i == 0.0R)
      (ensures source_sum_n 16 (partition_entry f part) == f (part * 2) +. f (part * 2 + 1)) =
  source_sum_zero_tail 16 2 (partition_entry f part);
  source_sum_step 2 (partition_entry f part);
  source_sum_step 1 (partition_entry f part)

inline_for_extraction noextract
let partitions_sum (f:Shared.vector 64) (cut:nat{cut <= 7})
  : Lemma
      (requires forall (i:Shared.index 64). cut <= i ==> f i == 0.0R)
      (ensures
        (source_sum_n 16 (partition_entry f 0) +. source_sum_n 16 (partition_entry f 1)) +.
        (source_sum_n 16 (partition_entry f 2) +. source_sum_n 16 (partition_entry f 3)) ==
          Shared.sum cut f) =
  partition_total f 0;
  partition_total f 1;
  partition_total f 2;
  partition_total f 3;
  let extended (i:nat) : GTot real = if i < 64 then f i else 0.0R in
  source_sum_zero_tail 8 cut extended;
  source_sum_shared cut extended;
  sum_extensional cut extended f;
  source_sum_step 8 extended;
  source_sum_step 7 extended;
  source_sum_step 6 extended;
  source_sum_step 5 extended;
  source_sum_step 4 extended;
  source_sum_step 3 extended;
  source_sum_step 2 extended;
  source_sum_step 1 extended

inline_for_extraction noextract
let attention_exponent_refines
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 16) (col:Shared.index 64)
  (scaled exponent:F32.t)
  : Lemma
      (requires
        source_qk_representation rotated query key /\
        (forall (mr ar:real).
          source_maximum row (source_score_row (source_mma_score_tile rotated head) row) %~ mr /\
          source_attention_literal () %~ ar ==> scaled %~ (mr *. ar)) /\
        source_branch_exponent_contract
          (source_row_score row (source_score_row (source_mma_score_tile rotated head) row) col)
          scaled exponent)
      (ensures bilateral_mma_live_scores_finite rotated head ==>
        exponent %~ attention_numerator rotated query key head row col) =
  let under_domain ()
    : Lemma
        (requires bilateral_mma_live_scores_finite rotated head)
        (ensures exponent %~ attention_numerator rotated query key head row col) =
    if source_live row col then (
      source_mma_score_tile_at rotated head row col;
      source_score_row_at (source_mma_score_tile rotated head) row col;
      bilateral_finite_not_sentinel (Seq.index (source_mma_score_tile rotated head) (row * 8 + col));
      source_mma_tile_represents_dot rotated query key head row col;
      source_attention_literal_represents_ideal ();
      let maximum = source_maximum row (source_score_row (source_mma_score_tile rotated head) row) in
      to_real_ok maximum;
      source_active_numerator_refines
        (Seq.index (source_mma_score_tile rotated head) (row * 8 + col)) scaled exponent
        (Shared.dot (query (row / 2) (2 * head + row % 2)) (key col head)) (to_real maximum)
    ) else (
      negative_infinity_compares_equal ()
    )
  in
  FStar.Classical.forall_intro (FStar.Classical.move_requires under_domain)

inline_for_extraction noextract
let probability_prefix (ys:lseq bf16 1024) (row:Shared.index 16)
  (f:Shared.vector 64) (part step:nat) (guard:prop) : prop =
  forall (p:Shared.index 4) (j:Shared.index 16).
    (p < part \/ (p = part /\ j < step)) ==>
      (guard ==> Seq.index ys (row * 64 + source_column p j) %~ f (source_column p j))

inline_for_extraction noextract
let probability_write
  (before after:lseq bf16 1024) (row:Shared.index 16) (f:Shared.vector 64)
  (part:Shared.index 4) (step:Shared.index 16) (value:bf16) (guard:prop)
  : Lemma
      (requires probability_prefix before row f part step guard /\
        after == Seq.upd before (row * 64 + source_column part step) value /\
        (guard ==> value %~ f (source_column part step)))
      (ensures probability_prefix after row f part (step + 1) guard) =
  let at (p:Shared.index 4) (j:Shared.index 16)
    : Lemma
        ((p < part \/ (p = part /\ j < step + 1)) ==>
          (guard ==> Seq.index after (row * 64 + source_column p j) %~ f (source_column p j))) =
    source_column_injective p part j step;
    if p = part && j = step then ()
    else assert (Seq.index after (row * 64 + source_column p j) ==
      Seq.index before (row * 64 + source_column p j))
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let probability_prefix_done (ys:lseq bf16 1024) (row:Shared.index 16)
  (f:Shared.vector 64) (guard:prop)
  : Lemma
      (requires probability_prefix ys row f 4 0 guard)
      (ensures guard ==> bf16_segment_rep ys (row * 64) 64 f) =
  let at (col:Shared.index 64)
    : Lemma (guard ==> Seq.index ys (row * 64 + col) %~ f col) =
    source_column_inverse col
  in FStar.Classical.forall_intro at

inline_for_extraction noextract
let attention_baseline (row:Shared.index 16) : real = if row < 14 then 0.0R else 1.0R

inline_for_extraction noextract
let attention_partitions_denominator
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 16)
  : Lemma
      (let f = attention_numerator rotated query key head row in
       attention_baseline row +.
         ((source_sum_n 16 (partition_entry f 0) +. source_sum_n 16 (partition_entry f 1)) +.
          (source_sum_n 16 (partition_entry f 2) +. source_sum_n 16 (partition_entry f 3))) ==
         attention_denominator rotated query key head row) =
  let f = attention_numerator rotated query key head row in
  let cut = if row < 14 then row / 2 + 1 else 0 in
  partitions_sum f cut

inline_for_extraction noextract
let attention_rescale_refines
  (rotated:bf16_values 32768) (head:Shared.index 8) (row:Shared.index 16) (rescale:F32.t)
  : Lemma
      (requires
        (ieee_eq (source_maximum row (source_score_row (source_mma_score_tile rotated head) row))
          (negative_infinity ()) ==> rescale %~ 1.0R) /\
        (not (ieee_eq (source_maximum row (source_score_row (source_mma_score_tile rotated head) row))
          (negative_infinity ())) ==> rescale %~ 0.0R))
      (ensures bilateral_mma_live_scores_finite rotated head ==> rescale %~ attention_baseline row) =
  let under_domain ()
    : Lemma
        (requires bilateral_mma_live_scores_finite rotated head)
        (ensures rescale %~ attention_baseline row) =
    if row < 14 then bilateral_mma_maximum_exclusion rotated head row
    else (
      source_maximum_empty row (source_score_row (source_mma_score_tile rotated head) row);
      negative_infinity_compares_equal ()
    )
  in FStar.Classical.move_requires under_domain ()

inline_for_extraction noextract
let attention_combined_refines
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 16) (rescale s0 s1 s2 s3 result:F32.t)
  : Lemma
      (requires
        (bilateral_mma_live_scores_finite rotated head ==>
          rescale %~ attention_baseline row /\
          s0 %~ source_sum_n 16 (partition_entry (attention_numerator rotated query key head row) 0) /\
          s1 %~ source_sum_n 16 (partition_entry (attention_numerator rotated query key head row) 1) /\
          s2 %~ source_sum_n 16 (partition_entry (attention_numerator rotated query key head row) 2) /\
          s3 %~ source_sum_n 16 (partition_entry (attention_numerator rotated query key head row) 3)) /\
        (forall (b a0 a1 a2 a3:real).
          rescale %~ b /\ s0 %~ a0 /\ s1 %~ a1 /\ s2 %~ a2 /\ s3 %~ a3 ==>
            result %~ (b +. ((a0 +. a1) +. (a2 +. a3)))))
      (ensures bilateral_mma_live_scores_finite rotated head ==>
        result %~ attention_denominator rotated query key head row) =
  attention_partitions_denominator rotated query key head row

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn attention_probability_row
  (tile:larray F32.t 128) (probability:larray bf16 1024)
  (sums:larray F32.t 4) (denominator:larray F32.t 16) (row:szlt 16)
  (#rotated:erased (bf16_values 32768)) (#query:erased (Shared.heads 16))
  (#key:erased (Shared.heads 8)) (#head:erased (Shared.index 8))
  (#old:erased (lseq bf16 1024)) (#old_sums:erased (lseq F32.t 4))
  (#old_den:erased (lseq F32.t 16)) (#ft:perm)
  preserves gpu ** tile |-> Frac ft (source_mma_score_tile rotated head)
  requires probability |-> old ** sums |-> old_sums ** denominator |-> old_den **
    pure (source_qk_representation rotated query key)
  ensures exists* (ps:lseq bf16 1024) (ss:lseq F32.t 4) (ds:lseq F32.t 16).
    probability |-> ps ** sums |-> ss ** denominator |-> ds **
    pure (
      segment_frame old ps (row * 64) 64 /\ segment_frame old_den ds row 1 /\
      (bilateral_mma_live_scores_finite rotated head ==>
        bf16_segment_rep ps (row * 64) 64 (attention_numerator rotated query key head row) /\
        Seq.index ds row %~ attention_denominator rotated query key head row))
{
  let maximum = source_maximum_from_tile tile row;
  source_attention_literal_is_positive_normal ();
  source_scale_upper_bound ();
  source_maximum_cases maximum;
  negative_infinity_compares_equal ();
  let scaled = F32.mul_rn_ftz maximum (source_attention_literal ());
  ObF.neg_kind scaled;
  assert pure (not (ieee_eq maximum (negative_infinity ())) ==>
    source_addend_guard (ObF.sub ObF.zero scaled));
  let guard = hide (bilateral_mma_live_scores_finite rotated head);
  let numerators = hide (attention_numerator rotated query key head row);
  let mut part = 0sz;
  while (!part <^ 4sz)
    invariant exists* (p:sz{p <= 4}) (ps:lseq bf16 1024) (ss:lseq F32.t 4).
      part |-> p ** probability |-> ps ** sums |-> ss **
      pure (segment_frame old ps (row * 64) 64 /\
        probability_prefix ps row numerators p 0 guard /\
        (forall (i:Shared.index 4{i < p}).
          guard ==> Seq.index ss i %~ source_sum_n 16 (partition_entry numerators i)))
    decreases (4 - !part)
  {
    let p = !part;
    let mut sum:F32.t = ObF.zero;
    let mut step = 0sz;
    while (!step <^ 16sz)
      invariant exists* (j:sz{j <= 16}) (ps:lseq bf16 1024) (acc:F32.t).
        step |-> j ** sum |-> acc ** probability |-> ps **
        pure (segment_frame old ps (row * 64) 64 /\
          probability_prefix ps row numerators p j guard /\
          (guard ==> acc %~ source_sum_n j (partition_entry numerators p)))
      decreases (16 - !step)
    {
      let j = !step;
      let col = p *^ 2sz +^ (j /^ 2sz) *^ 8sz +^ j %^ 2sz;
      let value = source_read_score_tile tile row col;
      let exponent = source_exponent_from_scaled value scaled;
      attention_exponent_refines rotated query key head row col scaled exponent;
      f32_to_bf16_all exponent;
      let rounded = Casts.fcast #F32.t #bf16 exponent;
      with before. assert probability |-> before;
      probability.(row *^ 64sz +^ col) <- rounded;
      with after. assert probability |-> after;
      probability_write before after row numerators p j rounded guard;
      let previous = !sum;
      let added = F32.add_rn_ftz previous exponent;
      source_sum_step (j + 1) (partition_entry numerators p);
      sum := added;
      step := j +^ 1sz;
    };
    let finished = !step;
    source_loop_end finished 16sz;
    let accumulated = !sum;
    sums.(p) <- accumulated;
    part := p +^ 1sz;
  };
  let finished = !part;
  source_loop_end finished 4sz;
  let one:F32.t = ObF.one;
  let rescale = if ieee_eq maximum (negative_infinity ()) {
    one
  } else {
    source_rescale_tail (ObF.sub ObF.zero scaled)
  };
  attention_rescale_refines rotated head row rescale;
  let s0 = sums.(0sz);
  let s1 = sums.(1sz);
  let s2 = sums.(2sz);
  let s3 = sums.(3sz);
  let result = source_denominator_combine rescale s0 s1 s2 s3;
  attention_combined_refines rotated query key head row rescale s0 s1 s2 s3 result;
  denominator.(row) <- result;
  with ps. assert probability |-> ps;
  probability_prefix_done ps row numerators guard;
  ()
}
#pop-options

inline_for_extraction noextract
let attention_probability_contract
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key:Shared.heads 8)
  (head:Shared.index 8) (ps:lseq bf16 1024) (ds:lseq F32.t 16) : prop =
  bilateral_mma_live_scores_finite rotated head ==>
    matrix_rep ps (attention_numerator rotated query key head) /\
    segment_rep ds 0 16 (fun row -> if row < 16 then attention_denominator rotated query key head row else 0.0R)

inline_for_extraction noextract
let denominator_row_progress (before after:lseq F32.t 16)
  (row:Shared.index 16) (model:Shared.vector 16) (guard:prop)
  : Lemma
      (requires
        segment_frame before after row 1 /\
        (guard ==> Seq.index after row %~ model row) /\
        (forall (r:Shared.index 16{r < row}). guard ==> Seq.index before r %~ model r))
      (ensures forall (r:Shared.index 16{r < row + 1}). guard ==> Seq.index after r %~ model r) =
  let at (r:Shared.index 16{r < row + 1})
    : Lemma (guard ==> Seq.index after r %~ model r) =
    if r <> row then assert (Seq.index after r == Seq.index before r)
  in FStar.Classical.forall_intro at

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn attention_probability_head
  (tile:larray F32.t 128) (probability:larray bf16 1024)
  (sums:larray F32.t 4) (denominator:larray F32.t 16)
  (#rotated:erased (bf16_values 32768)) (#query:erased (Shared.heads 16))
  (#key:erased (Shared.heads 8)) (#head:erased (Shared.index 8))
  (#old:erased (lseq bf16 1024)) (#old_sums:erased (lseq F32.t 4))
  (#old_den:erased (lseq F32.t 16)) (#ft:perm)
  preserves gpu ** tile |-> Frac ft (source_mma_score_tile rotated head)
  requires probability |-> old ** sums |-> old_sums ** denominator |-> old_den **
    pure (source_qk_representation rotated query key)
  ensures exists* (ps:lseq bf16 1024) (ss:lseq F32.t 4) (ds:lseq F32.t 16).
    probability |-> ps ** sums |-> ss ** denominator |-> ds **
    pure (attention_probability_contract rotated query key head ps ds)
{
  let guard = hide (bilateral_mma_live_scores_finite rotated head);
  let numerators = hide (attention_numerator rotated query key head);
  let denominators = hide (attention_denominator rotated query key head);
  let mut row = 0sz;
  while (!row <^ 16sz)
    invariant exists* (r:sz{r <= 16}) (ps:lseq bf16 1024) (ss:lseq F32.t 4) (ds:lseq F32.t 16).
      row |-> r ** probability |-> ps ** sums |-> ss ** denominator |-> ds **
      pure (
        (guard ==> matrix_prefix ps numerators r) /\
        (forall (i:Shared.index 16{i < r}). guard ==> Seq.index ds i %~ (reveal denominators) i))
    decreases (16 - !row)
  {
    let r = !row;
    with previous. assert probability |-> previous;
    with ds. assert denominator |-> ds;
    with ss. assert sums |-> ss;
    attention_probability_row tile probability sums denominator r
      #rotated #query #key #head
      #(hide (reveal previous <: lseq bf16 1024)) #(hide (reveal ss <: lseq F32.t 4))
      #(hide (reveal ds <: lseq F32.t 16)) #ft;
    with after. assert probability |-> after;
    with next_den. assert denominator |-> next_den;
    matrix_row_progress previous after numerators r guard;
    denominator_row_progress ds next_den r denominators guard;
    row := r +^ 1sz;
  };
  let finished = !row;
  source_loop_end finished 16sz;
  ()
}
#pop-options

inline_for_extraction noextract
let packed_query (rotated:bf16_values 32768) (head:Shared.index 8) : GTot (lseq bf16 2048) =
  indexed_matrix 16 128 (fun row lane ->
    if row < 14 then Seq.index rotated ((row / 2) * 4096 + head * 512 + (row % 2) * 128 + lane)
    else ObF.zero)

inline_for_extraction noextract
let packed_key (rotated:bf16_values 32768) (head:Shared.index 8) : GTot (lseq bf16 1024) =
  indexed_matrix 8 128 (fun key lane ->
    if key < 7 then Seq.index rotated (key * 4096 + head * 512 + 256 + lane)
    else ObF.zero)

inline_for_extraction noextract
let packed_values (rotated:bf16_values 32768) (head:Shared.index 8)
  (block:nat{block < 128 /\ block % 8 = 0}) : GTot (lseq bf16 512) =
  indexed_matrix 8 64 (fun lane key ->
    if key < 7 then Seq.index rotated (key * 4096 + head * 512 + 384 + block + lane)
    else ObF.zero)

#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let packed_score_operands (rotated:bf16_values 32768) (head:Shared.index 8) (step:Shared.index 8)
  : Lemma
      (ematrix_subtile (Tensor.from_seq (rm 16 128) (packed_query rotated head)) 16 16 0 step ==
        source_query_operand rotated head step /\
       ematrix_subtile (Tensor.from_seq (cm 128 8) (packed_key rotated head)) 16 8 step 0 ==
        source_key_operand rotated head step) =
  let qa = ematrix_subtile (Tensor.from_seq (rm 16 128) (packed_query rotated head)) 16 16 0 step in
  let kb = ematrix_subtile (Tensor.from_seq (cm 128 8) (packed_key rotated head)) 16 8 step 0 in
  Kuiper.EMatrix.lemma_equal_intro qa (source_query_operand rotated head step);
  Kuiper.EMatrix.lemma_equal_intro kb (source_key_operand rotated head step)

inline_for_extraction noextract
let rec packed_score_accumulator
  (rotated:bf16_values 32768) (head:Shared.index 8) (steps:nat{steps <= 8})
  : Lemma
      (dense_mma_fold 128 (packed_query rotated head) (packed_key rotated head) steps ==
        source_mma_scores rotated head steps)
      (decreases steps) =
  if steps > 0 then (
    packed_score_accumulator rotated head (steps - 1);
    packed_score_operands rotated head (steps - 1);
    dense_mma_step 128 (packed_query rotated head) (packed_key rotated head) (steps - 1)
  )

inline_for_extraction noextract
let packed_score_tile (rotated:bf16_values 32768) (head:Shared.index 8)
  : Lemma
      (Tensor.to_seq (rm 16 8)
        (dense_mma_fold 128 (packed_query rotated head) (packed_key rotated head) 8) ==
        source_mma_score_tile rotated head) =
  packed_score_accumulator rotated head 8;
  assert (Seq.equal
    (Tensor.to_seq (rm 16 8) (source_mma_scores rotated head 8))
    (source_mma_score_tile rotated head))
#pop-options

inline_for_extraction noextract
let gathered_source (#n #m:nat) (x:lseq bf16 n) (layout:Shared.index m -> GTot nat)
  : GTot (lseq bf16 m) =
  Seq.init_ghost m (fun i -> if layout i < n then Seq.index x (layout i) else ObF.zero)

inline_for_extraction noextract
let gathered_source_at (#n #m:nat) (x:lseq bf16 n)
  (layout:Shared.index m -> GTot nat) (i:Shared.index m)
  : Lemma (Seq.index (gathered_source x layout) i ==
      (if layout i < n then Seq.index x (layout i) else ObF.zero)) = ()

inline_for_extraction noextract
fn gather_source
  (#n #m:erased nat) (input:larray bf16 n) (output:larray bf16 m)
  (length:sz{SZ.v length == n}) (count:sz{SZ.v count == m})
  (#layout:erased (Shared.index m -> GTot nat))
  (index:szlt count -> Tot (sz))
  (#x:erased (lseq bf16 n)) (#old:erased (lseq bf16 m)) (#f:perm)
  preserves gpu ** input |-> Frac f x
  requires output |-> old **
    pure (forall (i:szlt count). SZ.v (index i) == (reveal layout) i)
  ensures output |-> gathered_source x layout
{
  let mut j = 0sz;
  while (!j <^ count)
    invariant exists* (i:sz{i <= count}) (ys:lseq bf16 m).
      j |-> i ** output |-> ys **
      pure (forall (k:Shared.index m{k < i}).
        Seq.index ys k == Seq.index (gathered_source x layout) k)
    decreases (count - !j)
  {
    let i = !j;
    let at = index i;
    let zero:bf16 = ObF.zero;
    let value:bf16 = if (at <^ length) {
      input.(at)
    } else { zero };
    output.(i) <- value;
    j := i +^ 1sz;
  };
  let finished = !j;
  source_loop_end finished count;
  with ys. assert output |-> ys;
  Seq.lemma_eq_intro ys (gathered_source x layout);
  Seq.lemma_eq_elim ys (gathered_source x layout);
  ()
}

inline_for_extraction noextract
let query_gather_index (head:Shared.index 8) (i:Shared.index 2048) : nat =
  let row = i / 128 in
  if row < 14 then (row / 2) * 4096 + head * 512 + (row % 2) * 128 + i % 128 else 32768

inline_for_extraction noextract
let key_gather_index (head:Shared.index 8) (i:Shared.index 1024) : nat =
  let row = i / 128 in
  if row < 7 then row * 4096 + head * 512 + 256 + i % 128 else 32768

inline_for_extraction noextract
let value_gather_index (head:Shared.index 8) (block:nat{block < 128 /\ block % 8 = 0})
  (i:Shared.index 512) : nat =
  if i % 64 < 7 then i % 64 * 4096 + head * 512 + 384 + block + i / 64 else 32768

inline_for_extraction noextract
let query_gather_source (head:szlt 8) (i:szlt 2048)
  : Tot (value:sz{SZ.v value == query_gather_index head i}) =
  let row = i /^ 128sz in
  if row <^ 14sz then row /^ 2sz *^ 4096sz +^ head *^ 512sz +^ row %^ 2sz *^ 128sz +^ i %^ 128sz
  else 32768sz

inline_for_extraction noextract
let key_gather_source (head:szlt 8) (i:szlt 1024)
  : Tot (value:sz{SZ.v value == key_gather_index head i}) =
  let row = i /^ 128sz in
  if row <^ 7sz then row *^ 4096sz +^ head *^ 512sz +^ 256sz +^ i %^ 128sz
  else 32768sz

inline_for_extraction noextract
let value_gather_source (head:szlt 8) (block:szlt 128{block % 8 = 0}) (i:szlt 512)
  : Tot (value:sz{SZ.v value == value_gather_index head block i}) =
  if i %^ 64sz <^ 7sz then i %^ 64sz *^ 4096sz +^ head *^ 512sz +^ 384sz +^ block +^ i /^ 64sz
  else 32768sz

inline_for_extraction noextract
let packed_query_at (rotated:bf16_values 32768) (head:Shared.index 8)
  (row:Shared.index 16) (lane:Shared.index 128)
  : Lemma (Seq.index (packed_query rotated head) (row * 128 + lane) ==
      (if row < 14 then Seq.index rotated ((row / 2) * 4096 + head * 512 + row % 2 * 128 + lane)
       else ObF.zero)) =
  indexed_matrix_at 16 128
    (fun r d -> if r < 14 then Seq.index rotated ((r / 2) * 4096 + head * 512 + r % 2 * 128 + d)
      else ObF.zero) row lane

inline_for_extraction noextract
let packed_key_at (rotated:bf16_values 32768) (head:Shared.index 8)
  (row:Shared.index 8) (lane:Shared.index 128)
  : Lemma (Seq.index (packed_key rotated head) (row * 128 + lane) ==
      (if row < 7 then Seq.index rotated (row * 4096 + head * 512 + 256 + lane)
       else ObF.zero)) =
  indexed_matrix_at 8 128
    (fun r d -> if r < 7 then Seq.index rotated (r * 4096 + head * 512 + 256 + d)
      else ObF.zero) row lane

inline_for_extraction noextract
let packed_values_at (rotated:bf16_values 32768) (head:Shared.index 8)
  (block:nat{block < 128 /\ block % 8 = 0}) (lane:Shared.index 8) (key:Shared.index 64)
  : Lemma (Seq.index (packed_values rotated head block) (lane * 64 + key) ==
      (if key < 7 then Seq.index rotated (key * 4096 + head * 512 + 384 + block + lane)
       else ObF.zero)) =
  indexed_matrix_at 8 64
    (fun d k -> if k < 7 then Seq.index rotated (k * 4096 + head * 512 + 384 + block + d)
      else ObF.zero) lane key

#push-options "--fuel 2 --ifuel 1 --z3rlimit 30"
inline_for_extraction noextract
let gathered_attention_operands
  (rotated:bf16_values 32768) (head:Shared.index 8) (block:nat{block < 128 /\ block % 8 = 0})
  : Lemma
      (gathered_source rotated (query_gather_index head) == packed_query rotated head /\
       gathered_source rotated (key_gather_index head) == packed_key rotated head /\
       gathered_source rotated (value_gather_index head block) == packed_values rotated head block) =
  let query_at (i:Shared.index 2048)
    : Lemma (Seq.index (gathered_source rotated (query_gather_index head)) i ==
        Seq.index (packed_query rotated head) i) =
    gathered_source_at rotated (query_gather_index head) i;
    packed_query_at rotated head (i / 128) (i % 128);
    assert (i / 128 * 128 + i % 128 == i);
    if i / 128 < 14 then assert (query_gather_index head i < 32768)
    else assert (query_gather_index head i == 32768)
  in FStar.Classical.forall_intro query_at;
  Seq.lemma_eq_intro (gathered_source rotated (query_gather_index head)) (packed_query rotated head);
  Seq.lemma_eq_elim (gathered_source rotated (query_gather_index head)) (packed_query rotated head);
  let key_at (i:Shared.index 1024)
    : Lemma (Seq.index (gathered_source rotated (key_gather_index head)) i ==
        Seq.index (packed_key rotated head) i) =
    gathered_source_at rotated (key_gather_index head) i;
    packed_key_at rotated head (i / 128) (i % 128);
    assert (i / 128 * 128 + i % 128 == i);
    if i / 128 < 7 then assert (key_gather_index head i < 32768)
    else assert (key_gather_index head i == 32768)
  in FStar.Classical.forall_intro key_at;
  Seq.lemma_eq_intro (gathered_source rotated (key_gather_index head)) (packed_key rotated head);
  Seq.lemma_eq_elim (gathered_source rotated (key_gather_index head)) (packed_key rotated head);
  let value_at (i:Shared.index 512)
    : Lemma (Seq.index (gathered_source rotated (value_gather_index head block)) i ==
        Seq.index (packed_values rotated head block) i) =
    gathered_source_at rotated (value_gather_index head block) i;
    packed_values_at rotated head block (i / 64) (i % 64);
    assert (i / 64 * 64 + i % 64 == i);
    if i % 64 < 7 then assert (value_gather_index head block i < 32768)
    else assert (value_gather_index head block i == 32768)
  in FStar.Classical.forall_intro value_at;
  Seq.lemma_eq_intro (gathered_source rotated (value_gather_index head block)) (packed_values rotated head block);
  Seq.lemma_eq_elim (gathered_source rotated (value_gather_index head block)) (packed_values rotated head block)
#pop-options

inline_for_extraction noextract
let seq_length_view (#a:Type0) (n:nat) (values:seq a)
  : Pure (lseq a n)
      (requires Seq.length values == n)
      (ensures fun result -> result == values) =
  values

inline_for_extraction noextract
let array_live (#a:Type0) (arr:array a) : slprop =
  exists* (values:seq a). arr |-> values

inline_for_extraction noextract
ghost
fn take_owned_array (#a:Type0) (#n:erased nat) (arr:larray a n) (tid:szlt 32) (default:a)
  requires when_ (SZ.v tid == 0) (array_live arr)
  returns values:erased (lseq a n)
  ensures when_ (SZ.v tid == 0) (arr |-> values)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (array_live arr) as array_live arr;
    unfold array_live arr;
    with values. assert arr |-> values;
    pts_to_len arr;
    let result = hide (seq_length_view n (reveal values));
    rewrite arr |-> values as when_ (SZ.v tid == 0) (arr |-> result);
    result
  } else {
    rewrite when_ (SZ.v tid == 0) (array_live arr) as emp;
    let result = hide (seq_length_view n (Seq.create n default));
    rewrite emp as when_ (SZ.v tid == 0) (arr |-> result);
    result
  }
}

inline_for_extraction noextract
let probability_pair
  (p:larray bf16 1024) (d:larray F32.t 16)
  (phi:lseq bf16 1024 -> lseq F32.t 16 -> prop) : slprop =
  exists* (ps:lseq bf16 1024) (ds:lseq F32.t 16).
    p |-> ps ** d |-> ds ** pure (phi ps ds)

inline_for_extraction noextract
let shared_probability_pair
  (p:larray bf16 1024) (d:larray F32.t 16)
  (phi:lseq bf16 1024 -> lseq F32.t 16 -> prop) : slprop =
  exists* (ps:lseq bf16 1024) (ds:lseq F32.t 16).
    p |-> Frac (1.0R /. 32.0R) ps ** d |-> Frac (1.0R /. 32.0R) ds ** pure (phi ps ds)

ghost
fn share_probability_proof (p:larray bf16 1024) (d:larray F32.t 16)
  (phi:lseq bf16 1024 -> lseq F32.t 16 -> prop) ()
  requires forall+ (tid:natlt 32). when_ (tid == 0) (probability_pair p d phi)
  ensures forall+ (_:natlt 32). shared_probability_pair p d phi
{
  warp_owner_elim (probability_pair p d phi);
  unfold probability_pair p d phi;
  with ps. assert p |-> ps;
  with ds. assert d |-> ds;
  Kuiper.Array.Extra.array_share p 32;
  Kuiper.Array.Extra.array_share d 32;
  forevery_intro_pure #(natlt 32) (fun _ -> phi ps ds);
  forevery_zip3 #(natlt 32)
    (fun _ -> p |-> Frac (1.0R /. 32.0R) ps)
    (fun _ -> d |-> Frac (1.0R /. 32.0R) ds)
    (fun _ -> pure (phi ps ds));
  forevery_map #(natlt 32)
    (fun _ -> p |-> Frac (1.0R /. 32.0R) ps ** d |-> Frac (1.0R /. 32.0R) ds ** pure (phi ps ds))
    (fun _ -> shared_probability_pair p d phi)
    fn _ { fold shared_probability_pair p d phi; };
}

inline_for_extraction noextract
let publish_probability (p:larray bf16 1024) (d:larray F32.t 16)
  (phi:lseq bf16 1024 -> lseq F32.t 16 -> prop) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** when_ (reveal tid == 0) (probability_pair p d phi))
      (ensures fun _ -> thread_id 32 tid ** shared_probability_pair p d phi) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun i -> when_ (i == 0) (probability_pair p d phi))
    (fun _ -> shared_probability_pair p d phi)
    (share_probability_proof p d phi ()) #32 #tid

ghost
fn gather_unknown_proof (#a:Type0) (arr:array a) ()
  requires forall+ (_:natlt 32). exists* (values:seq a). arr |-> Frac (1.0R /. 32.0R) values
  ensures forall+ (tid:natlt 32). when_ (tid == 0) (array_live arr)
{
  Kuiper.Array.Core.array_gather_underspec arr 32;
  fold array_live arr;
  warp_owner_intro (array_live arr);
}

inline_for_extraction noextract
let gather_unknown (#a:Type0) (arr:array a) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** (exists* (values:seq a). arr |-> Frac (1.0R /. 32.0R) values))
      (ensures fun _ -> thread_id 32 tid ** when_ (reveal tid == 0) (array_live arr)) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun _ -> exists* (values:seq a). arr |-> Frac (1.0R /. 32.0R) values)
    (fun i -> when_ (i == 0) (array_live arr))
    (gather_unknown_proof arr ()) #32 #tid

inline_for_extraction noextract
let source_values_representation (rotated:bf16_values 32768) (values:Shared.heads 8) : prop =
  forall (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128).
    v_approximates (Seq.index rotated (row * 4096 + head * 512 + 384 + lane)) (values row head lane)

inline_for_extraction noextract
let attention_value_vector (values:Shared.heads 8) (head:Shared.index 8) (lane:Shared.index 128)
  (key:Shared.index 64) : GTot real =
  if key < 7 then values key head lane else 0.0R

inline_for_extraction noextract
let attention_value_sum
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key:Shared.heads 8) (values:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 14) (lane:Shared.index 128) : GTot real =
  Shared.dot (attention_numerator rotated query key head row) (attention_value_vector values head lane)

[@@"opaque_to_smt"]
inline_for_extraction noextract
let attention_model (data:Shared.attention_data)
  (query:Shared.heads 16) (key values:Shared.heads 8) : GTot (Shared.matrix 8 2048) =
  fun row channel ->
    if row < 7 then
      Shared.attention_values
        (Shared.attention_probabilities data query (Shared.repeat_kv key))
        (Shared.repeat_kv values) row (channel / 128) (channel % 128)
    else 0.0R

inline_for_extraction noextract
let attention_model_padding (data:Shared.attention_data)
  (query:Shared.heads 16) (key values:Shared.heads 8)
  : Lemma (forall (col:Shared.index 2048). attention_model data query key values 7 col == 0.0R) =
  reveal_opaque (`%attention_model) (attention_model data query key values)

inline_for_extraction noextract
let attention_model_live (data:Shared.attention_data)
  (query:Shared.heads 16) (key values:Shared.heads 8)
  (row:Shared.index 7) (col:Shared.index 2048)
  : Lemma
      (attention_model data query key values row col ==
        Shared.attention_values
          (Shared.attention_probabilities data query (Shared.repeat_kv key))
          (Shared.repeat_kv values) row (col / 128) (col % 128)) =
  reveal_opaque (`%attention_model) (attention_model data query key values)

inline_for_extraction noextract
let real_shift_distribute (x y z:real)
  : Lemma ((x -. y) *. z == x *. z -. y *. z) = ()

inline_for_extraction noextract
let dot_pointwise_equal (#n:pos) (x y u v:Shared.vector n)
  : Lemma
      (requires forall (i:Shared.index n). x i == u i /\ y i == v i)
      (ensures Shared.dot x y == Shared.dot u v) =
  sum_extensional n (fun i -> x i *. y i) (fun i -> u i *. v i)

inline_for_extraction noextract
let rms_pointwise_equal (#n:pos) (g0 g1 x0 x1:Shared.vector n) (i:Shared.index n)
  : Lemma
      (requires forall (j:Shared.index n). g0 j == g1 j /\ x0 j == x1 j)
      (ensures Shared.rms_norm g0 x0 i == Shared.rms_norm g1 x1 i) =
  Shared.sum_nonnegative n (fun j -> x0 j *. x0 j);
  Shared.sum_nonnegative n (fun j -> x1 j *. x1 j);
  sum_extensional n (fun j -> x0 j *. x0 j) (fun j -> x1 j *. x1 j);
  assert_norm (Shared.rms_norm g0 x0 i ==
    g0 i *. (x0 i *. ObRoot.rsqrt
      (Shared.sum n (fun j -> x0 j *. x0 j) /. of_int n +. 0.000001R)));
  assert_norm (Shared.rms_norm g1 x1 i ==
    g1 i *. (x1 i *. ObRoot.rsqrt
      (Shared.sum n (fun j -> x1 j *. x1 j) /. of_int n +. 0.000001R)))

inline_for_extraction noextract
let softmax_pointwise_equal (#n:pos) (x y:Shared.vector n) (i:Shared.index n)
  : Lemma
      (requires forall (j:Shared.index n). x j == y j)
      (ensures Shared.softmax x i == Shared.softmax y i) =
  Shared.sum_positive n (fun j -> ObExp.exp (x j));
  Shared.sum_positive n (fun j -> ObExp.exp (y j));
  sum_extensional n (fun j -> ObExp.exp (x j)) (fun j -> ObExp.exp (y j));
  shared_softmax_at n x i;
  shared_softmax_at n y i

inline_for_extraction noextract
let prefix_softmax_pointwise_equal
  (row:Shared.index 7) (x y:Shared.vector (row + 1)) (i:Shared.index 7)
  : Lemma
      (requires forall (j:Shared.index (row + 1)). x j == y j)
      (ensures Shared.prefix_softmax row x i == Shared.prefix_softmax row y i) =
  if i <= row then softmax_pointwise_equal x y i

inline_for_extraction noextract
let attention_model_sum_at (data:Shared.attention_data)
  (query:Shared.heads 16) (key values:Shared.heads 8)
  (token:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 128)
  : Lemma
      (attention_model data query key values token (head * 128 + lane) ==
        Shared.sum (token + 1) (fun k ->
          Shared.attention_probabilities data query (Shared.repeat_kv key) token head k *.
            Shared.repeat_kv values k head lane)) =
  reveal_opaque (`%attention_model) (attention_model data query key values);
  assert ((head * 128 + lane) / 128 == head /\ (head * 128 + lane) % 128 == lane);
  assert (Shared.attention_values
    (Shared.attention_probabilities data query (Shared.repeat_kv key))
    (Shared.repeat_kv values) token head lane ==
      Shared.sum (token + 1) (fun k ->
        Shared.attention_probabilities data query (Shared.repeat_kv key) token head k *.
          Shared.repeat_kv values k head lane)) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.attention_values]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let attention_value_sum_expand
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key values:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 14) (lane:Shared.index 128)
  : Lemma
      (attention_value_sum rotated query key values head row lane ==
        Shared.sum 64 (fun k ->
          attention_numerator rotated query key head row k *. attention_value_vector values head lane k)) =
  assert_norm (attention_value_sum rotated query key values head row lane ==
    Shared.sum 64 (fun k ->
      attention_numerator rotated query key head row k *. attention_value_vector values head lane k))

#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let attention_normalization_refines
  (rotated:bf16_values 32768) (data:Shared.attention_data)
  (query:Shared.heads 16) (key values:Shared.heads 8)
  (head:Shared.index 8) (row:Shared.index 14) (lane:Shared.index 128)
  : Lemma
      (attention_denominator rotated query key head row >. 0.0R /\
       attention_value_sum rotated query key values head row lane /.
         attention_denominator rotated query key head row ==
       attention_model data query key values (row / 2) (head * 256 + row % 2 * 128 + lane)) =
  let token = row / 2 in
  let query_head = 2 * head + row % 2 in
  let maximum = to_real (source_maximum row (source_score_row (source_mma_score_tile rotated head) row)) in
  let scale = ObRoot.rsqrt 128.0R in
  let scores:Shared.vector (token + 1) =
    fun k -> Shared.dot (query token query_head) (key k head) *. scale in
  let shift = maximum *. scale in
  let v:Shared.vector 7 = fun k -> values k head lane in
  let p = source_shifted_prefix_numerators token scores shift in
  let shared_scores:Shared.vector (token + 1) = fun k ->
    Shared.dot (query token query_head) (Shared.repeat_kv key k query_head) *. ObRoot.rsqrt 128.0R in
  assert (query_head / 2 == head);
  let score_at (k:Shared.index (token + 1))
    : Lemma (shared_scores k == scores k) =
    dot_pointwise_equal (query token query_head) (Shared.repeat_kv key k query_head)
      (query token query_head) (key k head)
  in
  FStar.Classical.forall_intro score_at;
  let probability_at (k:Shared.index (token + 1))
    : Lemma
        (Shared.attention_probabilities data query (Shared.repeat_kv key) token query_head k ==
          Shared.prefix_softmax token scores k) =
    assert (Shared.attention_probabilities data query (Shared.repeat_kv key) token query_head k ==
      Shared.prefix_softmax token
        (fun source -> Shared.dot (query token query_head)
          (Shared.repeat_kv key source query_head) *. ObRoot.rsqrt 128.0R) k) by
      (FStar.Tactics.V2.norm [delta_only [`%Shared.attention_probabilities]];
       FStar.Tactics.V2.trivial ());
    prefix_softmax_pointwise_equal token
      (fun source -> Shared.dot (query token query_head)
        (Shared.repeat_kv key source query_head) *. ObRoot.rsqrt 128.0R)
      scores k
  in FStar.Classical.forall_intro probability_at;
  let numerator_at (k:Shared.index 64)
    : Lemma (attention_numerator rotated query key head row k == padded_real p k) =
    if k <= token then
      real_shift_distribute (Shared.dot (query token query_head) (key k head)) maximum scale
  in
  FStar.Classical.forall_intro numerator_at;
  normalized_dense_prefix token scores v shift;
  sum_extensional 64
    (fun k -> padded_real p k *. padded_real v k)
    (fun k -> padded_real (source_shifted_prefix_numerators token scores shift) k *. padded_real v k);
  assert (Shared.sum 64 (fun k -> padded_real p k *. padded_real v k) /.
    Shared.sum (token + 1) p ==
      Shared.sum (token + 1) (fun k -> Shared.prefix_softmax token scores k *. v k));
  sum_extensional (token + 1) (attention_numerator rotated query key head row) p;
  sum_extensional 64
    (fun k -> attention_numerator rotated query key head row k *. attention_value_vector values head lane k)
    (fun k -> padded_real p k *. padded_real v k);
  sum_extensional (token + 1)
    (fun k -> Shared.prefix_softmax token scores k *. v k)
    (fun k -> Shared.attention_probabilities data query (Shared.repeat_kv key) token query_head k *.
      Shared.repeat_kv values k query_head lane);
  attention_model_sum_at data query key values token query_head lane;
  assert (query_head * 128 + lane == head * 256 + row % 2 * 128 + lane);
  assert (Shared.sum (token + 1) (fun k -> Shared.prefix_softmax token scores k *. v k) ==
    attention_model data query key values token (head * 256 + row % 2 * 128 + lane));
  attention_value_sum_expand rotated query key values head row lane;
  assert (attention_denominator rotated query key head row == Shared.sum (token + 1) p);
  assert (attention_value_sum rotated query key values head row lane ==
    Shared.sum 64 (fun k -> padded_real p k *. padded_real v k))

inline_for_extraction noextract
let attention_value_mma_refines
  (rotated:bf16_values 32768) (query:Shared.heads 16) (key values:Shared.heads 8)
  (head:Shared.index 8) (ps:lseq bf16 1024) (ds:lseq F32.t 16)
  (block:nat{block < 128 /\ block % 8 = 0})
  (row:Shared.index 14) (col:Shared.index 8)
  : Lemma
      (requires attention_probability_contract rotated query key head ps ds /\
        source_values_representation rotated values)
      (ensures bilateral_mma_live_scores_finite rotated head ==>
        Chest.acc2 (dense_mma_fold 64 ps (packed_values rotated head block) 4) row col %~
          attention_value_sum rotated query key values head row (block + col)) =
  let under_domain ()
    : Lemma
        (requires bilateral_mma_live_scores_finite rotated head)
        (ensures Chest.acc2 (dense_mma_fold 64 ps (packed_values rotated head block) 4) row col %~
          attention_value_sum rotated query key values head row (block + col)) =
    dense_mma_dot 64 ps (packed_values rotated head block) row col
      (attention_numerator rotated query key head row)
      (attention_value_vector values head (block + col))
  in FStar.Classical.move_requires under_domain ()

inline_for_extraction noextract
let attention_output_refines
  (rotated:bf16_values 32768) (data:Shared.attention_data)
  (query:Shared.heads 16) (key values:Shared.heads 8)
  (head:Shared.index 8) (ps:lseq bf16 1024) (ds:lseq F32.t 16)
  (block:nat{block < 128 /\ block % 8 = 0})
  (row:Shared.index 14) (col:Shared.index 8) (result:bf16)
  : Lemma
      (requires attention_probability_contract rotated query key head ps ds /\
        source_values_representation rotated values /\
        (forall (n:real) (d:real{d =!= 0.0R}).
          Chest.acc2 (dense_mma_fold 64 ps (packed_values rotated head block) 4) row col %~ n /\
          v_approximates (Seq.index ds row) d ==> result %~ (n /. d)))
      (ensures bilateral_mma_live_scores_finite rotated head ==>
        result %~ attention_model data query key values (row / 2) (head * 256 + row % 2 * 128 + block + col)) =
  attention_value_mma_refines rotated query key values head ps ds block row col;
  attention_normalization_refines rotated data query key values head row (block + col)
#pop-options

inline_for_extraction noextract
let attention_domain (rotated:bf16_values 32768) : prop =
  forall (head:Shared.index 8). bilateral_mma_live_scores_finite rotated head

inline_for_extraction noextract
let attention_output_prefix
  (ys:lseq bf16 16384) (model:Shared.matrix 8 2048) (head block:nat) : prop =
  forall (row:Shared.index 7) (channel:Shared.index 2048).
    (channel / 256 < head \/ (channel / 256 = head /\ channel % 128 < block)) ==>
      Seq.index ys (row * 2048 + channel) %~ model row channel

inline_for_extraction noextract
let attention_tile_prefix
  (ys:lseq bf16 16384) (model:Shared.matrix 8 2048)
  (head:Shared.index 8) (block:nat{block < 128 /\ block % 8 = 0}) (done:nat) : prop =
  forall (row:Shared.index 14) (col:Shared.index 8).
    row * 8 + col < done ==>
      Seq.index ys ((row / 2) * 2048 + head * 256 + row % 2 * 128 + block + col) %~
        model (row / 2) (head * 256 + row % 2 * 128 + block + col)

inline_for_extraction noextract
let attention_tile_write
  (before after:lseq bf16 16384) (model:Shared.matrix 8 2048)
  (head:Shared.index 8) (block:nat{block < 128 /\ block % 8 = 0})
  (row:Shared.index 14) (col:Shared.index 8) (value:bf16) (guard:prop)
  : Lemma
      (requires
        (guard ==> attention_output_prefix before model head block /\
          attention_tile_prefix before model head block (row * 8 + col) /\
          value %~ model (row / 2) (head * 256 + row % 2 * 128 + block + col)) /\
        after == Seq.upd before
          ((row / 2) * 2048 + head * 256 + row % 2 * 128 + block + col) value)
      (ensures guard ==> attention_output_prefix after model head block /\
        attention_tile_prefix after model head block (row * 8 + col + 1)) =
  let tile_at (r:Shared.index 14) (c:Shared.index 8)
    : Lemma
        (r * 8 + c < row * 8 + col + 1 ==>
          (guard ==> Seq.index after ((r / 2) * 2048 + head * 256 + r % 2 * 128 + block + c) %~
            model (r / 2) (head * 256 + r % 2 * 128 + block + c))) =
    if r = row && c = col then ()
    else assert ((r / 2) * 2048 + head * 256 + r % 2 * 128 + block + c <>
      (row / 2) * 2048 + head * 256 + row % 2 * 128 + block + col)
  in FStar.Classical.forall_intro_2 tile_at

inline_for_extraction noextract
let attention_tile_done
  (ys:lseq bf16 16384) (model:Shared.matrix 8 2048)
  (head:Shared.index 8) (block:nat{block < 128 /\ block % 8 = 0}) (guard:prop)
  : Lemma
      (requires guard ==> attention_output_prefix ys model head block /\
        attention_tile_prefix ys model head block 112)
      (ensures guard ==> attention_output_prefix ys model head (block + 8)) =
  let at (row:Shared.index 7) (channel:Shared.index 2048)
    : Lemma
        ((channel / 256 < head \/ (channel / 256 = head /\ channel % 128 < block + 8)) ==>
          (guard ==> Seq.index ys (row * 2048 + channel) %~ model row channel)) =
    if channel / 256 = head && block <= channel % 128 && channel % 128 < block + 8 then (
      let r = row * 2 + (channel % 256) / 128 in
      let c = channel % 128 - block in
      assert (r < 14 /\ c < 8);
      assert ((r / 2) * 2048 + head * 256 + r % 2 * 128 + block + c == row * 2048 + channel)
    )
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let attention_output_owned (output:larray bf16 16384)
  (rotated:bf16_values 32768) (model:Shared.matrix 8 2048) (head block:nat) : slprop =
  exists* (ys:lseq bf16 16384). output |-> ys **
    pure (attention_domain rotated ==> attention_output_prefix ys model head block)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn write_attention_tile
  (tile:larray F32.t 128) (denominator:larray F32.t 16) (output:larray bf16 16384)
  (head:szlt 8) (block:szlt 128{block % 8 = 0})
  (#rotated:erased (bf16_values 32768)) (#data:erased Shared.attention_data)
  (#query:erased (Shared.heads 16)) (#key #values:erased (Shared.heads 8))
  (#ps:erased (lseq bf16 1024)) (#ds:erased (lseq F32.t 16)) (#fd:perm)
  preserves gpu **
    tile |-> Tensor.to_seq (rm 16 8) (dense_mma_fold 64 ps (packed_values rotated head block) 4) **
    denominator |-> Frac fd ds
  requires attention_output_owned output rotated (attention_model data query key values) head block **
    pure (attention_probability_contract rotated query key head ps ds /\ source_values_representation rotated values)
  ensures attention_output_owned output rotated (attention_model data query key values) head (block + 8)
{
  unfold attention_output_owned output rotated (attention_model data query key values) head block;
  let guard = hide (attention_domain rotated);
  let model = hide (attention_model data query key values);
  let mut row = 0sz;
  while (!row <^ 14sz)
    invariant exists* (r:sz{r <= 14}) (ys:lseq bf16 16384).
      row |-> r ** output |-> ys **
      pure (guard ==> attention_output_prefix ys model head block /\
        attention_tile_prefix ys model head block (r * 8))
    decreases (14 - !row)
  {
    let r = !row;
    let mut col = 0sz;
    while (!col <^ 8sz)
      invariant exists* (c:sz{c <= 8}) (ys:lseq bf16 16384).
        col |-> c ** output |-> ys **
        pure (guard ==> attention_output_prefix ys model head block /\
          attention_tile_prefix ys model head block (r * 8 + c))
      decreases (8 - !col)
    {
      let c = !col;
      let numerator = tile.(r *^ 8sz +^ c);
      let den = denominator.(r);
      let value = source_normalize_value numerator den;
      source_tile_sequence_at (dense_mma_fold 64 ps (packed_values rotated head block) 4) r c;
      attention_output_refines rotated data query key values head ps ds block r c value;
      with (before:seq bf16). assert output |-> before;
      output.(r /^ 2sz *^ 2048sz +^ head *^ 256sz +^ r %^ 2sz *^ 128sz +^ block +^ c) <- value;
      with (after:seq bf16). assert output |-> after;
      attention_tile_write before after model head block r c value guard;
      col := c +^ 1sz;
    };
    let finished = !col;
    source_loop_end finished 8sz;
    row := r +^ 1sz;
  };
  let finished = !row;
  source_loop_end finished 14sz;
  with ys. assert output |-> ys;
  attention_tile_done ys model head block guard;
  fold attention_output_owned output rotated (attention_model data query key values) head (block + 8);
}
#pop-options

ghost
fn publish_unknown_proof (#a:Type0) (arr:array a) ()
  requires forall+ (tid:natlt 32). when_ (tid == 0) (array_live arr)
  ensures forall+ (_:natlt 32). exists* (values:seq a). arr |-> Frac (1.0R /. 32.0R) values
{
  warp_owner_elim (array_live arr);
  unfold array_live arr;
  with values. assert arr |-> values;
  Kuiper.Array.Extra.array_share arr 32;
  forevery_map #(natlt 32)
    (fun _ -> arr |-> Frac (1.0R /. 32.0R) values)
    (fun _ -> exists* (v:seq a). arr |-> Frac (1.0R /. 32.0R) v)
    fn _ { () };
}

inline_for_extraction noextract
let publish_unknown (#a:Type0) (arr:array a) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** when_ (reveal tid == 0) (array_live arr))
      (ensures fun _ -> thread_id 32 tid ** (exists* (values:seq a). arr |-> Frac (1.0R /. 32.0R) values)) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun i -> when_ (i == 0) (array_live arr))
    (fun _ -> exists* (values:seq a). arr |-> Frac (1.0R /. 32.0R) values)
    (publish_unknown_proof arr ()) #32 #tid

inline_for_extraction noextract
let matrix_rep_if_transport (#rows #cols:pos) (xs:lseq bf16 (rows * cols))
  (left right:Shared.matrix rows cols) (guard:prop)
  : Lemma
      (requires
        (guard ==> matrix_rep xs left) /\
        (forall (row:Shared.index rows) (col:Shared.index cols). left row col == right row col))
      (ensures guard ==> matrix_rep xs right) = ()

inline_for_extraction noextract
let hidden_full (input:Shared.prefill_input) (padding:Shared.vector 1024) : GTot stored_hidden =
  fun row col -> if row < 7 then input.hidden row col else padding col

[@@"opaque_to_smt"]
inline_for_extraction noextract
let canonical_query (input:Shared.prefill_input) : GTot (Shared.heads 16) =
  Shared.rotary input.data
    (Shared.normalize_heads input.weight.q_norm
      (Shared.split_heads #16 (Shared.linear_rows input.weight.q_proj
        (Shared.normalize_hidden input.weight.input_layernorm input.hidden))))

[@@"opaque_to_smt"]
inline_for_extraction noextract
let canonical_key (input:Shared.prefill_input) : GTot (Shared.heads 8) =
  Shared.rotary input.data
    (Shared.normalize_heads input.weight.k_norm
      (Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj
        (Shared.normalize_hidden input.weight.input_layernorm input.hidden))))

[@@"opaque_to_smt"]
inline_for_extraction noextract
let canonical_value (input:Shared.prefill_input) : GTot (Shared.heads 8) =
  Shared.split_heads #8 (Shared.linear_rows input.weight.v_proj
    (Shared.normalize_hidden input.weight.input_layernorm input.hidden))

inline_for_extraction noextract
let canonical_query_unfold (input:Shared.prefill_input)
  : Lemma
      (canonical_query input == Shared.rotary input.data
        (Shared.normalize_heads input.weight.q_norm
          (Shared.split_heads #16 (Shared.linear_rows input.weight.q_proj
            (Shared.normalize_hidden input.weight.input_layernorm input.hidden))))) =
  reveal_opaque (`%canonical_query) (canonical_query input)

inline_for_extraction noextract
let canonical_key_unfold (input:Shared.prefill_input)
  : Lemma
      (canonical_key input == Shared.rotary input.data
        (Shared.normalize_heads input.weight.k_norm
          (Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj
            (Shared.normalize_hidden input.weight.input_layernorm input.hidden))))) =
  reveal_opaque (`%canonical_key) (canonical_key input)

inline_for_extraction noextract
let canonical_value_unfold (input:Shared.prefill_input)
  : Lemma
      (canonical_value input == Shared.split_heads #8 (Shared.linear_rows input.weight.v_proj
        (Shared.normalize_hidden input.weight.input_layernorm input.hidden))) =
  reveal_opaque (`%canonical_value) (canonical_value input)

[@@"opaque_to_smt"]
inline_for_extraction noextract
let full_qkv (input:Shared.prefill_input) (padding:Shared.vector 1024) : GTot (Shared.matrix 8 4096) =
  Shared.linear_rows (qkv_weight input.weight)
    (normalized_hidden8 input.weight.input_layernorm (hidden_full input padding))

inline_for_extraction noextract
let full_qkv_unfold (input:Shared.prefill_input) (padding:Shared.vector 1024)
  : Lemma
      (full_qkv input padding ==
        Shared.linear_rows (qkv_weight input.weight)
          (normalized_hidden8 input.weight.input_layernorm (hidden_full input padding))) =
  reveal_opaque (`%full_qkv) (full_qkv input padding)

#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let source_query_head_bound (group:Shared.index 8) (head:Shared.index 3)
  : Lemma (head = 2 \/ (head < 2 /\ 2 * group + head < 16)) = ()

inline_for_extraction noextract
let rotary_pointwise_equal (#count:pos) (data:Shared.attention_data)
  (vector:Shared.vector 128) (heads:Shared.heads count)
  (row:Shared.index 7) (head:Shared.index count) (lane:Shared.index 128)
  : Lemma
      (requires forall (d:Shared.index 128). vector d == heads row head d)
      (ensures rotate_source_head vector data row lane == Shared.rotary data heads row head lane) = ()

inline_for_extraction noextract
let normalized_hidden_live (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 7) (col:Shared.index 1024)
  : Lemma
      (normalized_hidden8 input.weight.input_layernorm (hidden_full input padding) row col ==
        Shared.normalize_hidden input.weight.input_layernorm input.hidden row col) =
  rms_pointwise_equal input.weight.input_layernorm input.weight.input_layernorm
    (hidden_full input padding row) (input.hidden row) col

inline_for_extraction noextract
let source_head_canonical (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  : Lemma
      (forall (lane:Shared.index 128).
        source_head (full_qkv input padding) row group head lane ==
          (if head = 2 then
            Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj
              (Shared.normalize_hidden input.weight.input_layernorm input.hidden)) row group lane
           else
            Shared.split_heads #16 (Shared.linear_rows input.weight.q_proj
              (Shared.normalize_hidden input.weight.input_layernorm input.hidden)) row (2 * group + head) lane)) =
  full_qkv_unfold input padding;
  source_query_head_bound group head;
  FStar.Classical.forall_intro (normalized_hidden_live input padding row);
  let at (lane:Shared.index 128)
    : Lemma
        (source_head (full_qkv input padding) row group head lane ==
          (if head = 2 then
            Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj
              (Shared.normalize_hidden input.weight.input_layernorm input.hidden)) row group lane
           else
            Shared.split_heads #16 (Shared.linear_rows input.weight.q_proj
              (Shared.normalize_hidden input.weight.input_layernorm input.hidden)) row (2 * group + head) lane)) =
    let physical = group * 512 + head * 128 + lane in
    let original = normalized_hidden8 input.weight.input_layernorm (hidden_full input padding) row in
    let normalized = Shared.normalize_hidden input.weight.input_layernorm input.hidden row in
    if head = 2 then (
      FStar.Classical.forall_intro (qkv_kv_weight_at input.weight group lane);
      dot_pointwise_equal original (qkv_weight input.weight physical)
        normalized (input.weight.k_proj (group * 128 + lane))
    ) else (
      FStar.Classical.forall_intro (qkv_query_weight_at input.weight (2 * group + head) lane);
      dot_pointwise_equal original (qkv_weight input.weight physical)
        normalized (input.weight.q_proj ((2 * group + head) * 128 + lane))
    )
  in
  FStar.Classical.forall_intro at

inline_for_extraction noextract
let rotated_canonical_at (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3) (lane:Shared.index 128)
  : Lemma
      (rotated_model input.weight (full_qkv input padding) input.data row (group * 512 + head * 128 + lane) ==
        (if head = 2 then canonical_key input row group lane
         else canonical_query input row (2 * group + head) lane)) =
  source_query_head_bound group head;
  rotated_model_head input.weight (full_qkv input padding) input.data row group head lane;
  normalized_source_head_unfold input.weight (full_qkv input padding) row group head;
  source_head_canonical input padding row group head;
  let normalized = Shared.normalize_hidden input.weight.input_layernorm input.hidden in
  let at (d:Shared.index 128)
    : Lemma
        (normalized_source_head input.weight (full_qkv input padding) row group head d ==
          (if head = 2 then
            Shared.normalize_heads input.weight.k_norm
              (Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj normalized)) row group d
           else
            Shared.normalize_heads input.weight.q_norm
              (Shared.split_heads #16 (Shared.linear_rows input.weight.q_proj normalized)) row (2 * group + head) d)) =
    if head = 2 then
      rms_pointwise_equal input.weight.k_norm input.weight.k_norm
        (source_head (full_qkv input padding) row group head)
        (Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj normalized) row group) d
    else
      rms_pointwise_equal input.weight.q_norm input.weight.q_norm
        (source_head (full_qkv input padding) row group head)
        (Shared.split_heads #16 (Shared.linear_rows input.weight.q_proj normalized) row (2 * group + head)) d
  in
  FStar.Classical.forall_intro at;
  if head = 2 then
    (canonical_key_unfold input;
     rotary_pointwise_equal input.data
      (normalized_source_head input.weight (full_qkv input padding) row group head)
      (Shared.normalize_heads input.weight.k_norm
        (Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj normalized))) row group lane)
  else
    (canonical_query_unfold input;
     rotary_pointwise_equal input.data
      (normalized_source_head input.weight (full_qkv input padding) row group head)
      (Shared.normalize_heads input.weight.q_norm
        (Shared.split_heads #16 (Shared.linear_rows input.weight.q_proj normalized))) row (2 * group + head) lane)

inline_for_extraction noextract
let raw_value_canonical_at (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : Lemma
      (rotated_model input.weight (full_qkv input padding) input.data row (head * 512 + 384 + lane) ==
        canonical_value input row head lane) =
  full_qkv_unfold input padding;
  canonical_value_unfold input;
  rotated_model_unchanged input.weight (full_qkv input padding) input.data row (head * 512 + 384 + lane);
  FStar.Classical.forall_intro (normalized_hidden_live input padding row);
  FStar.Classical.forall_intro (qkv_kv_weight_at input.weight head lane);
  dot_pointwise_equal
    (normalized_hidden8 input.weight.input_layernorm (hidden_full input padding) row)
    (qkv_weight input.weight (head * 512 + 384 + lane))
    (Shared.normalize_hidden input.weight.input_layernorm input.hidden row)
    (input.weight.v_proj (head * 128 + lane))

inline_for_extraction noextract
let rotated_represents_canonical (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (buffer:bf16_array 32768)
  (rotated:bf16_values 32768)
  : Lemma
      (requires matrix_rep rotated (rotated_model input.weight (full_qkv input padding) input.data))
      (ensures
        source_qk_representation rotated (canonical_query input) (canonical_key input) /\
        source_values_representation rotated (canonical_value input) /\
        computed_cache_representation
          { rotated_buffer = buffer; rotated_values = rotated } (decoder_parts input padding)) =
  let q_at (row:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 128)
    : Lemma
        (v_approximates (Seq.index rotated (row * 4096 + head / 2 * 512 + head % 2 * 128 + lane))
          (canonical_query input row head lane)) =
    rotated_canonical_at input padding row (head / 2) (head % 2) lane
  in FStar.Classical.forall_intro_3 q_at;
  let kv_at (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
    : Lemma
        (v_approximates (Seq.index rotated (row * 4096 + head * 512 + 256 + lane))
           (canonical_key input row head lane) /\
         v_approximates (Seq.index rotated (row * 4096 + head * 512 + 384 + lane))
           (canonical_value input row head lane)) =
    rotated_canonical_at input padding row head 2 lane;
    raw_value_canonical_at input padding row head lane
  in FStar.Classical.forall_intro_3 kv_at;
  let cache_at (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
    : Lemma
        (v_approximates (Seq.index rotated (key_index row head lane))
           ((decoder_parts input padding).keys row head lane) /\
         v_approximates (Seq.index rotated (value_index row head lane))
           ((decoder_parts input padding).values row head lane)) =
    canonical_key_unfold input;
    canonical_value_unfold input;
    decoder_parts_cache_at input padding row head lane;
    kv_at row head lane
  in FStar.Classical.forall_intro_3 cache_at

inline_for_extraction noextract
let real_linear_zero (#n #m:pos) (weights:Shared.matrix m n) (col:Shared.index m)
  : Lemma (Shared.linear weights (fun _ -> 0.0R) col == 0.0R) =
  let zero:Shared.vector n = fun _ -> 0.0R in
  let terms:Shared.vector n = fun i -> zero i *. weights col i in
  sum_extensional n terms zero;
  sum_zero_tail n 0 zero;
  assert_norm (Shared.linear weights zero col == Shared.sum n terms);
  assert (Shared.sum 0 zero == 0.0R)

inline_for_extraction noextract
let full_residual (input:Shared.prefill_input) (padding:Shared.vector 1024) : GTot stored_hidden =
  projection_math 2048 1024 input.weight.o_proj
    (attention_model input.data (canonical_query input) (canonical_key input) (canonical_value input))
    (hidden_full input padding) true

inline_for_extraction noextract
let full_residual_padding (input:Shared.prefill_input) (padding:Shared.vector 1024)
  : Lemma (forall (col:Shared.index 1024). full_residual input padding 7 col == padding col) =
  let zero:Shared.vector 2048 = fun _ -> 0.0R in
  attention_model_padding input.data (canonical_query input) (canonical_key input) (canonical_value input);
  let at (col:Shared.index 1024)
    : Lemma (full_residual input padding 7 col == padding col) =
    dot_pointwise_equal
      (attention_model input.data (canonical_query input) (canonical_key input) (canonical_value input) 7)
      (input.weight.o_proj col) zero (input.weight.o_proj col);
    real_linear_zero input.weight.o_proj col
  in
  FStar.Classical.forall_intro at

inline_for_extraction noextract
let full_postnorm (input:Shared.prefill_input) (padding:Shared.vector 1024) : GTot stored_hidden =
  normalized_hidden8 input.weight.post_attention_layernorm (full_residual input padding)

inline_for_extraction noextract
let full_gateup (input:Shared.prefill_input) (padding:Shared.vector 1024) : GTot (Shared.matrix 8 6144) =
  Shared.linear_rows (gateup_weight input.weight) (full_postnorm input padding)

inline_for_extraction noextract
let full_activated_math (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 8) (col:Shared.index 3072)
  : Lemma
      (activated_model (full_gateup input padding) row col ==
        Shared.silu (Shared.linear input.weight.gate_proj (full_postnorm input padding row) col) *.
          Shared.linear input.weight.up_proj (full_postnorm input padding row) col) =
  FStar.Classical.forall_intro (gateup_weight_at input.weight col);
  let i = col / 64 * 128 + col % 64 in
  dot_pointwise_equal (full_postnorm input padding row) (gateup_weight input.weight i)
    (full_postnorm input padding row) (input.weight.gate_proj col);
  dot_pointwise_equal (full_postnorm input padding row) (gateup_weight input.weight (i + 64))
    (full_postnorm input padding row) (input.weight.up_proj col)

inline_for_extraction noextract
let reference_mixed (input:Shared.prefill_input) : GTot (Shared.matrix 7 2048) =
  Shared.concatenate_heads
    (Shared.attention_values
      (Shared.attention_probabilities input.data (canonical_query input) (Shared.repeat_kv (canonical_key input)))
      (Shared.repeat_kv (canonical_value input)))

inline_for_extraction noextract
let reference_residual (input:Shared.prefill_input) : GTot Shared.hidden_states =
  Shared.residual_add input.hidden (Shared.linear_rows input.weight.o_proj (reference_mixed input))

inline_for_extraction noextract
let reference_postnorm (input:Shared.prefill_input) : GTot Shared.hidden_states =
  Shared.normalize_hidden input.weight.post_attention_layernorm (reference_residual input)

inline_for_extraction noextract
let full_residual_live (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 7) (col:Shared.index 1024)
  : Lemma (full_residual input padding row col == reference_residual input row col) =
  FStar.Classical.forall_intro
    (attention_model_live input.data (canonical_query input) (canonical_key input) (canonical_value input) row);
  dot_pointwise_equal
    (attention_model input.data (canonical_query input) (canonical_key input) (canonical_value input) row)
    (input.weight.o_proj col) (reference_mixed input row) (input.weight.o_proj col)

inline_for_extraction noextract
let full_postnorm_live (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 7) (col:Shared.index 1024)
  : Lemma (full_postnorm input padding row col == reference_postnorm input row col) =
  FStar.Classical.forall_intro (full_residual_live input padding row);
  rms_pointwise_equal input.weight.post_attention_layernorm input.weight.post_attention_layernorm
    (full_residual input padding row) (reference_residual input row) col

inline_for_extraction noextract
let full_postnorm_padding (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (col:Shared.index 1024)
  : Lemma (full_postnorm input padding 7 col == Shared.rms_norm input.weight.post_attention_layernorm padding col) =
  full_residual_padding input padding;
  rms_pointwise_equal input.weight.post_attention_layernorm input.weight.post_attention_layernorm
    (full_residual input padding 7) padding col

inline_for_extraction noextract
let reference_activation (w:Shared.parameters) (x:Shared.vector 1024) : GTot (Shared.vector 3072) =
  fun col -> Shared.silu (Shared.linear w.gate_proj x col) *. Shared.linear w.up_proj x col

inline_for_extraction noextract
let full_activation_equal (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (row:Shared.index 8) (x:Shared.vector 1024) (col:Shared.index 3072)
  : Lemma
      (requires forall (k:Shared.index 1024). full_postnorm input padding row k == x k)
      (ensures activated_model (full_gateup input padding) row col ==
        reference_activation input.weight x col) =
  full_activated_math input padding row col;
  dot_pointwise_equal (full_postnorm input padding row) (input.weight.gate_proj col) x (input.weight.gate_proj col);
  dot_pointwise_equal (full_postnorm input padding row) (input.weight.up_proj col) x (input.weight.up_proj col)

inline_for_extraction noextract
let shared_linear_rows_at (#rows #src #dst:pos)
  (w:Shared.matrix dst src) (x:Shared.matrix rows src)
  (row:Shared.index rows) (col:Shared.index dst)
  : Lemma (Shared.linear_rows w x row col == Shared.dot (x row) (w col)) = ()

inline_for_extraction noextract
let shared_linear_at (#src #dst:pos) (w:Shared.matrix dst src) (x:Shared.vector src)
  (col:Shared.index dst)
  : Lemma (Shared.linear w x col == Shared.dot x (w col)) = ()

inline_for_extraction noextract
let shared_feed_forward_expand (w:Shared.parameters) (x:Shared.hidden_states)
  (row:Shared.index 7) (col:Shared.index 1024)
  : Lemma
      (Shared.feed_forward w x row col ==
        Shared.linear_rows w.down_proj
          (fun r mid -> Shared.silu (Shared.linear_rows w.gate_proj x r mid) *.
            Shared.linear_rows w.up_proj x r mid) row col) = ()

inline_for_extraction noextract
let shared_feed_forward_dot (w:Shared.parameters) (x:Shared.hidden_states)
  (row:Shared.index 7) (col:Shared.index 1024)
  : Lemma
      (Shared.feed_forward w x row col ==
        Shared.dot
          (fun mid -> Shared.silu (Shared.linear_rows w.gate_proj x row mid) *.
            Shared.linear_rows w.up_proj x row mid)
          (w.down_proj col)) =
  shared_feed_forward_expand w x row col;
  shared_linear_rows_at w.down_proj
    (fun r mid -> Shared.silu (Shared.linear_rows w.gate_proj x r mid) *.
      Shared.linear_rows w.up_proj x r mid) row col

inline_for_extraction noextract
let shared_feed_forward_at (input:Shared.prefill_input) (row:Shared.index 7) (col:Shared.index 1024)
  : Lemma
      (Shared.linear input.weight.down_proj (reference_activation input.weight (reference_postnorm input row)) col ==
        Shared.feed_forward input.weight (reference_postnorm input) row col) =
  shared_feed_forward_dot input.weight (reference_postnorm input) row col;
  dot_pointwise_equal (reference_activation input.weight (reference_postnorm input row)) (input.weight.down_proj col)
    (fun mid -> Shared.silu (Shared.linear_rows input.weight.gate_proj (reference_postnorm input) row mid) *.
      Shared.linear_rows input.weight.up_proj (reference_postnorm input) row mid)
    (input.weight.down_proj col)

inline_for_extraction noextract
let shared_hidden_reference_at (input:Shared.prefill_input)
  (row:Shared.index 7) (col:Shared.index 1024)
  : Lemma
      ((Shared.decoder_layer input).hidden_after row col ==
        reference_residual input row col +.
          Shared.feed_forward input.weight (reference_postnorm input) row col) =
  canonical_query_unfold input;
  canonical_key_unfold input;
  canonical_value_unfold input

#push-options "--fuel 4 --ifuel 1 --z3rlimit 30"
inline_for_extraction noextract
let padding_after_expand (w:Shared.parameters) (padding:Shared.vector 1024) (col:Shared.index 1024)
  : Lemma
      (padding_after w padding col == padding col +.
        Shared.linear w.down_proj
          (fun mid ->
            Shared.silu (Shared.linear w.gate_proj (Shared.rms_norm w.post_attention_layernorm padding) mid) *.
              Shared.linear w.up_proj (Shared.rms_norm w.post_attention_layernorm padding) mid)
          col) = ()
#pop-options

inline_for_extraction noextract
let padding_after_at (w:Shared.parameters) (padding:Shared.vector 1024) (col:Shared.index 1024)
  : Lemma
      (padding_after w padding col == padding col +.
        Shared.linear w.down_proj
          (reference_activation w (Shared.rms_norm w.post_attention_layernorm padding)) col) =
  padding_after_expand w padding col;
  dot_pointwise_equal
    (fun mid ->
      Shared.silu (Shared.linear w.gate_proj (Shared.rms_norm w.post_attention_layernorm padding) mid) *.
        Shared.linear w.up_proj (Shared.rms_norm w.post_attention_layernorm padding) mid)
    (w.down_proj col)
    (reference_activation w (Shared.rms_norm w.post_attention_layernorm padding))
    (w.down_proj col);
  shared_linear_at w.down_proj
    (fun mid ->
      Shared.silu (Shared.linear w.gate_proj (Shared.rms_norm w.post_attention_layernorm padding) mid) *.
        Shared.linear w.up_proj (Shared.rms_norm w.post_attention_layernorm padding) mid) col;
  shared_linear_at w.down_proj
    (reference_activation w (Shared.rms_norm w.post_attention_layernorm padding)) col

#push-options "--fuel 0 --ifuel 0 --z3rlimit 60"
inline_for_extraction noextract
let full_output_math (input:Shared.prefill_input) (padding:Shared.vector 1024)
  : Lemma
      (forall (row:Shared.index 8) (col:Shared.index 1024).
        projection_math 3072 1024 input.weight.down_proj
          (activated_model (full_gateup input padding)) (full_residual input padding) true row col ==
          (decoder_parts input padding).hidden row col) =
  let at (row:Shared.index 8) (col:Shared.index 1024)
    : Lemma
        (projection_math 3072 1024 input.weight.down_proj
          (activated_model (full_gateup input padding)) (full_residual input padding) true row col ==
          (decoder_parts input padding).hidden row col) =
    projection_math_at 3072 1024 input.weight.down_proj
      (activated_model (full_gateup input padding)) (full_residual input padding) true row col;
    if row < 7 then (
      FStar.Classical.forall_intro (full_postnorm_live input padding row);
      let live_activation_at (mid:Shared.index 3072)
        : Lemma
            (activated_model (full_gateup input padding) row mid ==
              reference_activation input.weight (reference_postnorm input row) mid) =
        full_activation_equal input padding row (reference_postnorm input row) mid
      in FStar.Classical.forall_intro live_activation_at;
      dot_pointwise_equal
        (activated_model (full_gateup input padding) row) (input.weight.down_proj col)
        (reference_activation input.weight (reference_postnorm input row)) (input.weight.down_proj col);
      shared_linear_at input.weight.down_proj
        (reference_activation input.weight (reference_postnorm input row)) col;
      full_residual_live input padding row col;
      shared_feed_forward_at input row col;
      shared_hidden_reference_at input row col;
      decoder_parts_hidden_at input padding row col
    ) else (
      full_residual_padding input padding;
      FStar.Classical.forall_intro (full_postnorm_padding input padding);
      let normalized = Shared.rms_norm input.weight.post_attention_layernorm padding in
      let padding_activation_at (mid:Shared.index 3072)
        : Lemma
            (activated_model (full_gateup input padding) 7 mid ==
              reference_activation input.weight normalized mid) =
        full_activation_equal input padding 7 normalized mid
      in FStar.Classical.forall_intro padding_activation_at;
      dot_pointwise_equal
        (activated_model (full_gateup input padding) 7) (input.weight.down_proj col)
        (reference_activation input.weight normalized) (input.weight.down_proj col);
      shared_linear_at input.weight.down_proj (reference_activation input.weight normalized) col;
      padding_after_at input.weight padding col;
      decoder_parts_padding_at input padding col
    )
  in
  FStar.Classical.forall_intro_2 at
#pop-options
#pop-options

inline_for_extraction noeq
type attention_buffers = {
  query_buffer: larray bf16 2048;
  key_buffer: larray bf16 1024;
  probability: larray bf16 1024;
  denominator: larray F32.t 16;
  sums: larray F32.t 4;
  value_buffer: larray bf16 512;
  tile: larray F32.t 128
}

inline_for_extraction noextract
let attention_buffer_lengths (s:attention_buffers)
  : Lemma
      (Pulse.Lib.Array.length s.query_buffer == 2048 /\
       Pulse.Lib.Array.length s.key_buffer == 1024 /\
       Pulse.Lib.Array.length s.probability == 1024 /\
       Pulse.Lib.Array.length s.denominator == 16 /\
       Pulse.Lib.Array.length s.sums == 4 /\
       Pulse.Lib.Array.length s.value_buffer == 512 /\
       Pulse.Lib.Array.length s.tile == 128) = ()

inline_for_extraction noextract
let attention_buffers_global (s:attention_buffers) : prop =
  is_global_array s.query_buffer /\ is_global_array s.key_buffer /\ is_global_array s.probability /\
  is_global_array s.denominator /\ is_global_array s.sums /\ is_global_array s.value_buffer /\ is_global_array s.tile

inline_for_extraction noextract
let attention_buffers_allocated (s:attention_buffers) : prop =
  attention_buffers_global s /\
  Pulse.Lib.Array.Core.is_full_array s.query_buffer /\
  Pulse.Lib.Array.Core.is_full_array s.key_buffer /\
  Pulse.Lib.Array.Core.is_full_array s.probability /\
  Pulse.Lib.Array.Core.is_full_array s.denominator /\
  Pulse.Lib.Array.Core.is_full_array s.sums /\
  Pulse.Lib.Array.Core.is_full_array s.value_buffer /\
  Pulse.Lib.Array.Core.is_full_array s.tile

inline_for_extraction noextract
let attention_temporary (s:attention_buffers) : slprop =
  (array_live s.query_buffer ** array_live s.key_buffer ** array_live s.probability) **
  (array_live s.denominator ** array_live s.sums ** array_live s.value_buffer) ** array_live s.tile

inline_for_extraction noextract
let attention_owned (s:attention_buffers) (tid:nat) : slprop =
  (when_ (tid == 0) (array_live s.query_buffer) **
   when_ (tid == 0) (array_live s.key_buffer) **
   when_ (tid == 0) (array_live s.probability)) **
  (when_ (tid == 0) (array_live s.denominator) **
   when_ (tid == 0) (array_live s.sums) **
   when_ (tid == 0) (array_live s.value_buffer)) **
  when_ (tid == 0) (array_live s.tile)

inline_for_extraction noextract
ghost
fn forget_owned_array (#a:Type0) (arr:array a) (tid:szlt 32) (#values:erased (seq a))
  requires when_ (SZ.v tid == 0) (arr |-> values)
  ensures when_ (SZ.v tid == 0) (array_live arr)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (arr |-> values) as arr |-> values;
    fold array_live arr;
    rewrite array_live arr as when_ (SZ.v tid == 0) (array_live arr);
  } else {
    rewrite when_ (SZ.v tid == 0) (arr |-> values) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (array_live arr);
  };
}

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn owner_gather_source
  (#m:erased nat) (input:bf16_array 32768) (output:larray bf16 m)
  (count:sz{SZ.v count == m}) (tid:szlt 32)
  (#layout:erased (Shared.index m -> GTot nat))
  (index:szlt count -> Tot sz)
  (#x:erased (bf16_values 32768)) (#f:perm)
  preserves gpu ** input |-> Frac f x
  requires when_ (SZ.v tid == 0) (array_live output) **
    pure (forall (i:szlt count). SZ.v (index i) == (reveal layout) i)
  ensures when_ (SZ.v tid == 0) (output |-> gathered_source x layout)
{
  let old = take_owned_array output tid (ObF.zero #bf16);
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (output |-> old) as output |-> old;
    gather_source input output 32768sz count #layout index #x #old #f;
    rewrite output |-> gathered_source x layout as
      when_ (SZ.v tid == 0) (output |-> gathered_source x layout);
  } else {
    rewrite when_ (SZ.v tid == 0) (output |-> old) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (output |-> gathered_source x layout);
  };
}

inline_for_extraction noextract
fn owner_probability
  (s:attention_buffers) (tid:szlt 32)
  (#rotated:erased (bf16_values 32768)) (#query:erased (Shared.heads 16))
  (#key:erased (Shared.heads 8)) (#head:erased (Shared.index 8))
  preserves gpu ** when_ (SZ.v tid == 0) (s.tile |-> source_mma_score_tile rotated head)
  requires
    when_ (SZ.v tid == 0) (array_live s.probability) **
    when_ (SZ.v tid == 0) (array_live s.denominator) **
    when_ (SZ.v tid == 0) (array_live s.sums) **
    pure (source_qk_representation rotated query key)
  ensures
    when_ (SZ.v tid == 0)
      (probability_pair s.probability s.denominator (attention_probability_contract rotated query key head)) **
    when_ (SZ.v tid == 0) (array_live s.sums)
{
  let old_p = take_owned_array s.probability tid (ObF.zero #bf16);
  let old_d = take_owned_array s.denominator tid (ObF.zero #F32.t);
  let old_s = take_owned_array s.sums tid (ObF.zero #F32.t);
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (s.tile |-> source_mma_score_tile rotated head)
      as s.tile |-> source_mma_score_tile rotated head;
    rewrite when_ (SZ.v tid == 0) (s.probability |-> old_p) as s.probability |-> old_p;
    rewrite when_ (SZ.v tid == 0) (s.denominator |-> old_d) as s.denominator |-> old_d;
    rewrite when_ (SZ.v tid == 0) (s.sums |-> old_s) as s.sums |-> old_s;
    attention_probability_head s.tile s.probability s.sums s.denominator
      #rotated #query #key #head #old_p #old_s #old_d #1.0R;
    fold probability_pair s.probability s.denominator (attention_probability_contract rotated query key head);
    fold array_live s.sums;
    rewrite s.tile |-> source_mma_score_tile rotated head as
      when_ (SZ.v tid == 0) (s.tile |-> source_mma_score_tile rotated head);
    rewrite probability_pair s.probability s.denominator (attention_probability_contract rotated query key head) as
      when_ (SZ.v tid == 0)
        (probability_pair s.probability s.denominator (attention_probability_contract rotated query key head));
    rewrite array_live s.sums as when_ (SZ.v tid == 0) (array_live s.sums);
  } else {
    rewrite when_ (SZ.v tid == 0) (s.probability |-> old_p) as emp;
    rewrite when_ (SZ.v tid == 0) (s.denominator |-> old_d) as emp;
    rewrite when_ (SZ.v tid == 0) (s.sums |-> old_s) as emp;
    rewrite emp as when_ (SZ.v tid == 0)
      (probability_pair s.probability s.denominator (attention_probability_contract rotated query key head));
    rewrite emp as when_ (SZ.v tid == 0) (array_live s.sums);
  };
}

inline_for_extraction noextract
fn owner_attention_write
  (s:attention_buffers) (output:larray bf16 16384)
  (head:szlt 8) (block:szlt 128{block % 8 = 0}) (tid:szlt 32)
  (#rotated:erased (bf16_values 32768)) (#data:erased Shared.attention_data)
  (#query:erased (Shared.heads 16)) (#key #values:erased (Shared.heads 8))
  (#ps:erased (lseq bf16 1024)) (#ds:erased (lseq F32.t 16))
  preserves gpu ** s.denominator |-> Frac (1.0R /. 32.0R) ds **
    when_ (SZ.v tid == 0)
      (s.tile |-> Tensor.to_seq (rm 16 8) (dense_mma_fold 64 ps (packed_values rotated head block) 4))
  requires when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) head block) **
    pure (attention_probability_contract rotated query key head ps ds /\ source_values_representation rotated values)
  ensures when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) head (block + 8))
{
  let tc = hide (Tensor.to_seq (rm 16 8) (dense_mma_fold 64 ps (packed_values rotated head block) 4));
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (s.tile |-> tc) as s.tile |-> tc;
    rewrite when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) head block)
      as attention_output_owned output rotated (attention_model data query key values) head block;
    write_attention_tile s.tile s.denominator output head block
      #rotated #data #query #key #values #ps #ds #(1.0R /. 32.0R);
    rewrite s.tile |-> tc as when_ (SZ.v tid == 0) (s.tile |-> tc);
    rewrite attention_output_owned output rotated (attention_model data query key values) head (block + 8)
      as when_ (SZ.v tid == 0)
        (attention_output_owned output rotated (attention_model data query key values) head (block + 8));
  } else {
    rewrite when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) head block) as emp;
    rewrite emp as when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) head (block + 8));
  };
}

inline_for_extraction noextract
let attention_head_advance (ys:lseq bf16 16384) (model:Shared.matrix 8 2048)
  (head:Shared.index 8)
  : Lemma
      (attention_output_prefix ys model head 128 <==> attention_output_prefix ys model (head + 1) 0) = ()

inline_for_extraction noextract
ghost
fn owner_attention_advance (output:larray bf16 16384) (tid:szlt 32)
  (rotated:bf16_values 32768) (model:Shared.matrix 8 2048) (head:Shared.index 8)
  requires when_ (SZ.v tid == 0) (attention_output_owned output rotated model head 128)
  ensures when_ (SZ.v tid == 0) (attention_output_owned output rotated model (head + 1) 0)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (attention_output_owned output rotated model head 128)
      as attention_output_owned output rotated model head 128;
    unfold attention_output_owned output rotated model head 128;
    with ys. assert output |-> ys;
    attention_head_advance ys model head;
    fold attention_output_owned output rotated model (head + 1) 0;
    rewrite attention_output_owned output rotated model (head + 1) 0 as
      when_ (SZ.v tid == 0) (attention_output_owned output rotated model (head + 1) 0);
  } else {
    rewrite when_ (SZ.v tid == 0) (attention_output_owned output rotated model head 128) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (attention_output_owned output rotated model (head + 1) 0);
  };
}

inline_for_extraction noextract
fn attention_head
  (input:bf16_array 32768) (output:larray bf16 16384) (s:attention_buffers)
  (head:szlt 8) (tid:szlt 32)
  (#rotated:erased (bf16_values 32768)) (#f:perm) (#data:erased Shared.attention_data)
  (#query:erased (Shared.heads 16)) (#key #values:erased (Shared.heads 8))
  preserves gpu ** thread_id 32 tid ** input |-> Frac f rotated
  requires attention_owned s tid **
    when_ (SZ.v tid == 0) (attention_output_owned output rotated (attention_model data query key values) head 0) **
    pure (source_qk_representation rotated query key /\ source_values_representation rotated values)
  ensures attention_owned s tid **
    when_ (SZ.v tid == 0) (attention_output_owned output rotated (attention_model data query key values) (head + 1) 0)
{
  attention_buffer_lengths s;
  unfold attention_owned s tid;
  owner_gather_source input s.query_buffer 2048sz tid
    #(hide (query_gather_index head)) (query_gather_source head) #rotated #f;
  owner_gather_source input s.key_buffer 1024sz tid
    #(hide (key_gather_index head)) (key_gather_source head) #rotated #f;
  gathered_attention_operands rotated head 0;
  warp_distribute s.query_buffer #(hide (packed_query rotated head <: seq bf16)) #1.0R #(hide (SZ.v tid));
  warp_distribute s.key_buffer #(hide (packed_key rotated head <: seq bf16)) #1.0R #(hide (SZ.v tid));
  publish_unknown s.tile #(hide (SZ.v tid));
  with cv. assert s.tile |-> Frac (1.0R /. 32.0R) cv;
  pts_to_len s.tile;
  let score_old:erased (lseq F32.t 128) = hide (seq_length_view 128 (reveal cv));
  dense_source_mma 128sz s.query_buffer s.key_buffer s.tile
    #(hide (packed_query rotated head)) #(hide (packed_key rotated head))
    #score_old #(1.0R /. 32.0R) #(1.0R /. 32.0R);
  packed_score_tile rotated head;
  warp_gather s.tile #(hide (source_mma_score_tile rotated head <: seq F32.t)) #1.0R #(hide (SZ.v tid));
  gather_unknown s.query_buffer #(hide (SZ.v tid));
  gather_unknown s.key_buffer #(hide (SZ.v tid));
  owner_probability s tid #rotated #query #key #(hide (SZ.v head));
  forget_owned_array s.tile tid #(hide (source_mma_score_tile rotated head <: seq F32.t));
  publish_probability s.probability s.denominator
    (attention_probability_contract rotated query key head) #(hide (SZ.v tid));
  unfold shared_probability_pair s.probability s.denominator
    (attention_probability_contract rotated query key head);
  with ps. assert s.probability |-> Frac (1.0R /. 32.0R) ps;
  with ds. assert s.denominator |-> Frac (1.0R /. 32.0R) ds;
  pts_to_len s.probability;
  pts_to_len s.denominator;
  publish_unknown s.tile #(hide (SZ.v tid));
  pts_to_len s.tile;
  let mut block = 0sz;
  while (!block <^ 128sz)
    invariant exists* (b:sz{b <= 128 /\ b % 8 = 0}) (tc:lseq F32.t 128).
      block |-> b ** s.tile |-> Frac (1.0R /. 32.0R) tc **
      when_ (SZ.v tid == 0) (array_live s.value_buffer) **
      when_ (SZ.v tid == 0)
        (attention_output_owned output rotated (attention_model data query key values) head b)
    decreases (128 - !block)
  {
    let b = !block;
    owner_gather_source input s.value_buffer 512sz tid
      #(hide (value_gather_index head b)) (value_gather_source head b) #rotated #f;
    gathered_attention_operands rotated head b;
    warp_distribute s.value_buffer #(hide (packed_values rotated head b <: seq bf16)) #1.0R #(hide (SZ.v tid));
    with tc. assert s.tile |-> Frac (1.0R /. 32.0R) tc;
    pts_to_len s.tile;
    dense_source_mma 64sz s.probability s.value_buffer s.tile
      #ps #(hide (packed_values rotated head b))
      #tc #(1.0R /. 32.0R) #(1.0R /. 32.0R);
    let next_tile = hide (Tensor.to_seq (rm 16 8)
      (dense_mma_fold 64 ps (packed_values rotated head b) 4));
    warp_gather s.tile #(hide (reveal next_tile <: seq F32.t)) #1.0R #(hide (SZ.v tid));
    owner_attention_write s output head b tid
      #rotated #data #query #key #values
      #ps #ds;
    warp_distribute s.tile #(hide (reveal next_tile <: seq F32.t)) #1.0R #(hide (SZ.v tid));
    gather_unknown s.value_buffer #(hide (SZ.v tid));
    let next = b +^ 8sz;
    assert pure (SZ.v next == SZ.v b + 8);
    rewrite when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) head (b + 8)) as
      when_ (SZ.v tid == 0)
        (attention_output_owned output rotated (attention_model data query key values) head next);
    block := next;
  };
  let finished = !block;
  source_loop_end finished 128sz;
  assert pure (SZ.v finished == 128);
  rewrite when_ (SZ.v tid == 0)
    (attention_output_owned output rotated (attention_model data query key values) head finished) as
    when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) head 128);
  gather_unknown s.tile #(hide (SZ.v tid));
  gather_unknown s.probability #(hide (SZ.v tid));
  gather_unknown s.denominator #(hide (SZ.v tid));
  owner_attention_advance output tid rotated (attention_model data query key values) head;
  fold attention_owned s tid;
}
#pop-options

inline_for_extraction noextract
let attention_output_final (output:larray bf16 16384)
  (rotated:bf16_values 32768) (model:Shared.matrix 8 2048) : slprop =
  exists* (ys:lseq bf16 16384). output |-> ys **
    pure ((attention_domain rotated ==> matrix_rep ys model) /\
      (forall (col:Shared.index 2048). Seq.index ys (7 * 2048 + col) == (ObF.zero #bf16)))

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn pad_attention_output (output:larray bf16 16384)
  (#rotated:erased (bf16_values 32768)) (#model:erased (Shared.matrix 8 2048))
  preserves gpu
  requires attention_output_owned output rotated model 8 0 **
    pure (forall (col:Shared.index 2048). (reveal model) 7 col == 0.0R)
  ensures attention_output_final output rotated model
{
  unfold attention_output_owned output rotated model 8 0;
  with original. assert output |-> original;
  let mut col = 0sz;
  while (!col <^ 2048sz)
    invariant exists* (i:sz{i <= 2048}) (ys:lseq bf16 16384).
      col |-> i ** output |-> ys **
      pure (segment_frame original ys 14336 i /\
        (forall (j:Shared.index 2048{j < i}). Seq.index ys (14336 + j) == (ObF.zero #bf16)))
    decreases (2048 - !col)
  {
    let i = !col;
    output.(14336sz +^ i) <- ObF.zero #bf16;
    col := i +^ 1sz;
  };
  let finished = !col;
  source_loop_end finished 2048sz;
  with ys. assert output |-> ys;
  assert pure (attention_domain rotated ==> matrix_rep ys model);
  fold attention_output_final output rotated model;
}

inline_for_extraction noextract
fn owner_attention_padding (output:larray bf16 16384) (tid:szlt 32)
  (#rotated:erased (bf16_values 32768)) (#model:erased (Shared.matrix 8 2048))
  preserves gpu
  requires when_ (SZ.v tid == 0) (attention_output_owned output rotated model 8 0) **
    pure (forall (col:Shared.index 2048). (reveal model) 7 col == 0.0R)
  ensures when_ (SZ.v tid == 0) (attention_output_final output rotated model)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (attention_output_owned output rotated model 8 0)
      as attention_output_owned output rotated model 8 0;
    pad_attention_output output #rotated #model;
    rewrite attention_output_final output rotated model as
      when_ (SZ.v tid == 0) (attention_output_final output rotated model);
  } else {
    rewrite when_ (SZ.v tid == 0) (attention_output_owned output rotated model 8 0) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (attention_output_final output rotated model);
  };
}

inline_for_extraction noextract
ghost
fn attention_distribute (s:attention_buffers)
  requires attention_temporary s
  ensures forall+ (tid:natlt 32). attention_owned s tid
{
  unfold attention_temporary s;
  warp_owner_intro (array_live s.query_buffer);
  warp_owner_intro (array_live s.key_buffer);
  warp_owner_intro (array_live s.probability);
  warp_owner_intro (array_live s.denominator);
  warp_owner_intro (array_live s.sums);
  warp_owner_intro (array_live s.value_buffer);
  warp_owner_intro (array_live s.tile);
  forevery_zip3 #(natlt 32)
    (fun i -> when_ (i == 0) (array_live s.query_buffer))
    (fun i -> when_ (i == 0) (array_live s.key_buffer))
    (fun i -> when_ (i == 0) (array_live s.probability));
  forevery_zip3 #(natlt 32)
    (fun i -> when_ (i == 0) (array_live s.denominator))
    (fun i -> when_ (i == 0) (array_live s.sums))
    (fun i -> when_ (i == 0) (array_live s.value_buffer));
  forevery_zip3 #(natlt 32)
    (fun i -> when_ (i == 0) (array_live s.query_buffer) **
      when_ (i == 0) (array_live s.key_buffer) ** when_ (i == 0) (array_live s.probability))
    (fun i -> when_ (i == 0) (array_live s.denominator) **
      when_ (i == 0) (array_live s.sums) ** when_ (i == 0) (array_live s.value_buffer))
    (fun i -> when_ (i == 0) (array_live s.tile));
  forevery_map #(natlt 32)
    (fun i ->
      (when_ (i == 0) (array_live s.query_buffer) ** when_ (i == 0) (array_live s.key_buffer) **
        when_ (i == 0) (array_live s.probability)) **
      (when_ (i == 0) (array_live s.denominator) ** when_ (i == 0) (array_live s.sums) **
        when_ (i == 0) (array_live s.value_buffer)) ** when_ (i == 0) (array_live s.tile))
    (attention_owned s)
    fn i { fold attention_owned s i; };
}

inline_for_extraction noextract
ghost
fn attention_gather (s:attention_buffers)
  requires forall+ (tid:natlt 32). attention_owned s tid
  ensures attention_temporary s
{
  forevery_map #(natlt 32) (attention_owned s)
    (fun i ->
      (when_ (i == 0) (array_live s.query_buffer) ** when_ (i == 0) (array_live s.key_buffer) **
        when_ (i == 0) (array_live s.probability)) **
      (when_ (i == 0) (array_live s.denominator) ** when_ (i == 0) (array_live s.sums) **
        when_ (i == 0) (array_live s.value_buffer)) ** when_ (i == 0) (array_live s.tile))
    fn i { unfold attention_owned s i; };
  forevery_unzip3 #(natlt 32)
    (fun i -> when_ (i == 0) (array_live s.query_buffer) **
      when_ (i == 0) (array_live s.key_buffer) ** when_ (i == 0) (array_live s.probability))
    (fun i -> when_ (i == 0) (array_live s.denominator) **
      when_ (i == 0) (array_live s.sums) ** when_ (i == 0) (array_live s.value_buffer))
    (fun i -> when_ (i == 0) (array_live s.tile));
  forevery_unzip3 #(natlt 32)
    (fun i -> when_ (i == 0) (array_live s.query_buffer))
    (fun i -> when_ (i == 0) (array_live s.key_buffer))
    (fun i -> when_ (i == 0) (array_live s.probability));
  forevery_unzip3 #(natlt 32)
    (fun i -> when_ (i == 0) (array_live s.denominator))
    (fun i -> when_ (i == 0) (array_live s.sums))
    (fun i -> when_ (i == 0) (array_live s.value_buffer));
  warp_owner_elim (array_live s.query_buffer);
  warp_owner_elim (array_live s.key_buffer);
  warp_owner_elim (array_live s.probability);
  warp_owner_elim (array_live s.denominator);
  warp_owner_elim (array_live s.sums);
  warp_owner_elim (array_live s.value_buffer);
  warp_owner_elim (array_live s.tile);
  fold attention_temporary s;
}

ghost
fn attention_setup (input:bf16_array 32768) (output:larray bf16 16384) (s:attention_buffers)
  (#rotated:erased (bf16_values 32768)) (#old:erased (lseq bf16 16384)) (#f:perm)
  (#model:erased (Shared.matrix 8 2048)) ()
  norewrite
  requires input |-> Frac f rotated ** attention_temporary s ** output |-> old
  ensures (forall+ (tid:natlt 32).
    input |-> Frac (f /. 32.0R) rotated ** attention_owned s tid **
    when_ (tid == 0) (attention_output_owned output rotated model 0 0)) ** emp
{
  Kuiper.Array.Extra.array_share input #rotated #f 32;
  attention_distribute s;
  fold attention_output_owned output rotated model 0 0;
  warp_owner_intro (attention_output_owned output rotated model 0 0);
  forevery_zip3 #(natlt 32)
    (fun _ -> input |-> Frac (f /. 32.0R) rotated)
    (attention_owned s)
    (fun i -> when_ (i == 0) (attention_output_owned output rotated model 0 0));
}

ghost
fn attention_teardown (input:bf16_array 32768) (output:larray bf16 16384) (s:attention_buffers)
  (#rotated:erased (bf16_values 32768)) (#f:perm) (#model:erased (Shared.matrix 8 2048)) ()
  norewrite
  requires (forall+ (tid:natlt 32).
    input |-> Frac (f /. 32.0R) rotated ** attention_owned s tid **
    when_ (tid == 0) (attention_output_final output rotated model)) ** emp
  ensures input |-> Frac f rotated ** attention_temporary s ** attention_output_final output rotated model
{
  forevery_unzip3 #(natlt 32)
    (fun _ -> input |-> Frac (f /. 32.0R) rotated)
    (attention_owned s)
    (fun i -> when_ (i == 0) (attention_output_final output rotated model));
  Kuiper.Array.Extra.array_gather input #rotated #f 32;
  attention_gather s;
  warp_owner_elim (attention_output_final output rotated model);
}

inline_for_extraction noextract
fn attention_thread (input:bf16_array 32768) (output:larray bf16 16384) (s:attention_buffers)
  (#rotated:erased (bf16_values 32768)) (#f:perm) (#data:erased Shared.attention_data)
  (#query:erased (Shared.heads 16)) (#key #values:erased (Shared.heads 8))
  (#_:squash (source_qk_representation rotated query key /\ source_values_representation rotated values))
  (tid:szlt 32) ()
  norewrite
  requires gpu **
    (input |-> Frac (f /. 32.0R) rotated ** attention_owned s tid **
      when_ (SZ.v tid == 0)
        (attention_output_owned output rotated (attention_model data query key values) 0 0)) **
    thread_id 32 tid
  ensures gpu **
    (input |-> Frac (f /. 32.0R) rotated ** attention_owned s tid **
      when_ (SZ.v tid == 0)
        (attention_output_final output rotated (attention_model data query key values))) **
    thread_id 32 tid
{
  let mut head = 0sz;
  while (!head <^ 8sz)
    invariant exists* (h:sz{h <= 8}).
      head |-> h ** attention_owned s tid **
      when_ (SZ.v tid == 0)
        (attention_output_owned output rotated (attention_model data query key values) h 0)
    decreases (8 - !head)
  {
    let h = !head;
    attention_head input output s h tid #rotated #(f /. 32.0R) #data #query #key #values;
    let next = h +^ 1sz;
    assert pure (SZ.v next == SZ.v h + 1);
    rewrite when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) (h + 1) 0) as
      when_ (SZ.v tid == 0)
        (attention_output_owned output rotated (attention_model data query key values) next 0);
    head := next;
  };
  let finished = !head;
  source_loop_end finished 8sz;
  assert pure (SZ.v finished == 8);
  rewrite when_ (SZ.v tid == 0)
    (attention_output_owned output rotated (attention_model data query key values) finished 0) as
    when_ (SZ.v tid == 0)
      (attention_output_owned output rotated (attention_model data query key values) 8 0);
  attention_model_padding data query key values;
  owner_attention_padding output tid #rotated #(hide (attention_model data query key values));
}

inline_for_extraction noextract
let attention_descriptor
  (input:bf16_array 32768{is_global_array input}) (output:larray bf16 16384{is_global_array output})
  (s:attention_buffers{attention_buffers_global s})
  (#rotated:erased (bf16_values 32768)) (#old:erased (lseq bf16 16384)) (#f:perm)
  (#data:erased Shared.attention_data) (#query:erased (Shared.heads 16))
  (#key #values:erased (Shared.heads 8))
  (#_:squash (source_qk_representation rotated query key /\ source_values_representation rotated values))
  : kernel_desc_1_n
      (input |-> Frac f rotated ** attention_temporary s ** output |-> old)
      (input |-> Frac f rotated ** attention_temporary s **
        attention_output_final output rotated (attention_model data query key values)) =
  {
    nthr = 32sz;
    frame = emp;
    kpre = (fun tid -> input |-> Frac (f /. 32.0R) rotated ** attention_owned s tid **
      when_ (tid == 0) (attention_output_owned output rotated (attention_model data query key values) 0 0));
    kpost = (fun tid -> input |-> Frac (f /. 32.0R) rotated ** attention_owned s tid **
      when_ (tid == 0) (attention_output_final output rotated (attention_model data query key values)));
    block_setup = attention_setup input output s #rotated #old #f #(hide (attention_model data query key values));
    block_teardown = attention_teardown input output s #rotated #f #(hide (attention_model data query key values));
    f = attention_thread input output s #rotated #f #data #query #key #values;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve
  }

inline_for_extraction noextract
fn run_attention
  (input:bf16_array 32768{is_global_array input}) (output:larray bf16 16384{is_global_array output})
  (s:attention_buffers{attention_buffers_global s})
  (#rotated:erased (bf16_values 32768)) (#old:erased (lseq bf16 16384)) (#f:perm)
  (#data:erased Shared.attention_data) (#query:erased (Shared.heads 16))
  (#key #values:erased (Shared.heads 8))
  (#inputs:erased invocation_inputs) (#run:erased (invocation inputs))
  (#observation:erased cache_observation)
  preserves cpu ** on gpu_loc (input |-> Frac f rotated)
  requires on gpu_loc (attention_temporary s ** output |-> old) **
    producer_handoff #inputs run observation **
    pure (source_qk_representation rotated query key /\ source_values_representation rotated values /\
      (reveal observation).rotated_buffer == input /\ (reveal observation).rotated_values == reveal rotated)
  ensures on gpu_loc (attention_temporary s **
    attention_output_final output rotated (attention_model data query key values)) **
    consumer_handoff #inputs run observation ** invocation_consumed #inputs run observation
{
  unfold producer_handoff #inputs run observation;
  launch_sync (attention_descriptor input output s #rotated #old #f #data #query #key #values);
  Execution.update run (consumed_state inputs observation);
  Execution.take_snapshot run (consumed_state inputs observation);
  fold consumer_handoff #inputs run observation;
  fold invocation_consumed #inputs run observation;
}
#pop-options

inline_for_extraction noextract
fn allocate_attention_buffers ()
  preserves cpu
  returns s:attention_buffers
  ensures on gpu_loc (attention_temporary s) ** pure (attention_buffers_allocated s)
{
  let q = gpu_array_alloc #bf16 2048sz;
  let k = gpu_array_alloc #bf16 1024sz;
  let p = gpu_array_alloc #bf16 1024sz;
  let d = gpu_array_alloc #F32.t 16sz;
  let sums = gpu_array_alloc #F32.t 4sz;
  let v = gpu_array_alloc #bf16 512sz;
  let tile = gpu_array_alloc #F32.t 128sz;
  let s:attention_buffers = { query_buffer = q; key_buffer = k; probability = p; denominator = d;
    sums = sums; value_buffer = v; tile = tile };
  assert rewrites_to s ({ query_buffer = q; key_buffer = k; probability = p; denominator = d;
    sums = sums; value_buffer = v; tile = tile } <: attention_buffers);
  with q0. assert on gpu_loc (q |-> q0);
  with k0. assert on gpu_loc (k |-> k0);
  with p0. assert on gpu_loc (p |-> p0);
  with d0. assert on gpu_loc (d |-> d0);
  with s0. assert on gpu_loc (sums |-> s0);
  with v0. assert on gpu_loc (v |-> v0);
  with t0. assert on gpu_loc (tile |-> t0);
  map_loc gpu_loc
    #(q |-> q0 ** k |-> k0 ** p |-> p0 ** d |-> d0 ** sums |-> s0 ** v |-> v0 ** tile |-> t0)
    #(attention_temporary s)
    fn () {
      fold array_live q;
      fold array_live k;
      fold array_live p;
      fold array_live d;
      fold array_live sums;
      fold array_live v;
      fold array_live tile;
      fold attention_temporary s;
    };
  s
}

inline_for_extraction noextract
fn free_attention_buffers (s:attention_buffers{attention_buffers_allocated s})
  preserves cpu
  requires on gpu_loc (attention_temporary s)
  ensures emp
{
  map_loc gpu_loc #(attention_temporary s)
    #(array_live s.query_buffer ** array_live s.key_buffer ** array_live s.probability **
      array_live s.denominator ** array_live s.sums ** array_live s.value_buffer ** array_live s.tile)
    fn () { unfold attention_temporary s; };
  rewrite each array_live s.query_buffer as (exists* values. s.query_buffer |-> values);
  rewrite each array_live s.key_buffer as (exists* values. s.key_buffer |-> values);
  rewrite each array_live s.probability as (exists* values. s.probability |-> values);
  rewrite each array_live s.denominator as (exists* values. s.denominator |-> values);
  rewrite each array_live s.sums as (exists* values. s.sums |-> values);
  rewrite each array_live s.value_buffer as (exists* values. s.value_buffer |-> values);
  rewrite each array_live s.tile as (exists* values. s.tile |-> values);
  gpu_array_free s.query_buffer;
  gpu_array_free s.key_buffer;
  gpu_array_free s.probability;
  gpu_array_free s.denominator;
  gpu_array_free s.sums;
  gpu_array_free s.value_buffer;
  gpu_array_free s.tile;
}

inline_for_extraction noextract
let projection_refines_if
  (#nr:nat) (width:nat{dense_width width}) (cols:pos)
  (layout:nat{projection_layout cols layout})
  (offset:nat{offset + cols * width <= weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:bf16_values weight_elements) (residual:lseq bf16 nr)
  (xr:Shared.matrix 8 width) (wr:Shared.matrix cols width) (rr:Shared.matrix 8 cols) (guard:prop)
  : Lemma
      (requires guard ==>
        matrix_rep x xr /\ projection_weights_rep width cols layout offset weights wr /\
        (add_residual ==> (forall (row:Shared.index 8) (col:Shared.index cols).
          Seq.index residual (row * cols + col) %~ rr row col)))
      (ensures guard ==> matrix_rep
        (projection_values width cols layout offset add_residual x weights residual)
        (projection_math width cols wr xr rr add_residual)) =
  let under_guard ()
    : Lemma
        (requires guard)
        (ensures matrix_rep (projection_values width cols layout offset add_residual x weights residual)
          (projection_math width cols wr xr rr add_residual)) =
    projection_refines width cols layout offset add_residual x weights residual xr wr rr
  in FStar.Classical.move_requires under_guard ()

inline_for_extraction noextract
let conditional_parts_postcondition
  (input:Shared.prefill_input) (padding:Shared.vector 1024)
  (result:bf16_values hidden_elements) (observation:cache_observation)
  : Lemma
      (requires computed_cache_representation observation (decoder_parts input padding) /\
        (observed_numerical_premises observation ==> output_representation result (decoder_parts input padding)))
      (ensures observed_numerical_premises observation ==> required_postcondition input padding result observation) =
  let under_domain ()
    : Lemma (requires observed_numerical_premises observation)
        (ensures required_postcondition input padding result observation) =
    represented_parts_establish_postcondition input padding result observation
  in FStar.Classical.move_requires under_domain ()

#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let source_weight_views (weights:bf16_values weight_elements) (w:Shared.parameters)
  : Lemma
      (requires weights_representation weights w)
      (ensures
        projection_weights_rep 1024 4096 1 2048 weights (qkv_weight w) /\
        projection_weights_rep 2048 1024 0 4196352 weights w.o_proj /\
        projection_weights_rep 1024 6144 2 6293760 weights (gateup_weight w) /\
        projection_weights_rep 3072 1024 0 12585216 weights w.down_proj) =
  let qkv_at (col:Shared.index 4096) (k:Shared.index 1024)
    : Lemma (Seq.index weights (2048 + projection_row 4096 1 col * 1024 + k) %~ qkv_weight w col k) =
    if col % 512 < 256 then () else if col % 512 < 384 then () else ()
  in FStar.Classical.forall_intro_2 qkv_at;
  let gate_at (col:Shared.index 6144) (k:Shared.index 1024)
    : Lemma (Seq.index weights (6293760 + projection_row 6144 2 col * 1024 + k) %~ gateup_weight w col k) =
    if col % 128 < 64 then () else ()
  in FStar.Classical.forall_intro_2 gate_at

inline_for_extraction noextract
let source_input_matrix
  (x:bf16_values hidden_elements) (logical:input_denotation{valid_denotation logical})
  (layer:Shared.index 28) (before:Shared.cache_state{Shared.valid_cache_before layer before})
  : Lemma
      (requires hidden_representation x logical.hidden)
      (ensures matrix_rep x (hidden_full (input_mapping logical layer before) (logical.hidden 7))) =
  ()
#pop-options

module IC = FStar.Int.Cast

inline_for_extraction noextract
let source_position (position:I64.t{0 <= I64.v position /\ I64.v position <= 4089})
  : Tot (p:sz{SZ.v p == I64.v position}) =
  SZ.uint32_to_sizet (IC.int64_to_uint32 position)

inline_for_extraction noextract
let output_weight_offset : sz = SZ.uint32_to_sizet 4196352ul

inline_for_extraction noextract
let gateup_weight_offset : sz = SZ.uint32_to_sizet 6293760ul

inline_for_extraction noextract
let down_weight_offset : sz = SZ.uint32_to_sizet 12585216ul

#push-options "--fuel 2 --ifuel 1 --z3rlimit 90"
inline_for_extraction noextract
fn run_decoder_producer
  (input0:bf16_array hidden_elements{is_global_array input0})
  (input1:bf16_array weight_elements{is_global_array input1})
  (input2:bf16_array rotary_elements{is_global_array input2})
  (position:sz{position <= 4089})
  (normalized:bf16_array 8192{is_global_array normalized})
  (qkv:bf16_array 32768{is_global_array qkv})
  (rotated:bf16_array 32768{is_global_array rotated})
  (rms_buffer:larray F32.t 288{is_global_array rms_buffer})
  (normalized_head:larray bf16 128{is_global_array normalized_head})
  (projection_tile:larray F32.t 128{is_global_array projection_tile})
  (#x:erased (bf16_values hidden_elements))
  (#weights:erased (bf16_values weight_elements))
  (#table:erased (bf16_values rotary_elements))
  (#n0:erased (lseq bf16 8192)) (#q0 #r0:erased (lseq bf16 32768))
  (#rb0:erased (lseq F32.t 288)) (#nh0:erased (lseq bf16 128))
  (#pt0:erased (lseq F32.t 128)) (#fx #fw #ft:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  (#inputs:erased invocation_inputs) (#run:erased (invocation inputs))
  preserves cpu ** on gpu_loc (
    input0 |-> Frac fx x ** input1 |-> Frac fw weights ** input2 |-> Frac ft table)
  requires on gpu_loc (
    normalized |-> n0 ** qkv |-> q0 ** rotated |-> r0 **
    rms_buffer |-> rb0 ** normalized_head |-> nh0 ** projection_tile |-> pt0) **
    invocation_pending #inputs run **
    pure (input_representation x weights table position logical /\
      inputs == native_inputs x weights table position layer before)
  returns observation:erased cache_observation
  ensures exists* (nv:lseq bf16 8192) (qv rv:lseq bf16 32768)
    (rb:lseq F32.t 288) (nh:lseq bf16 128) (pt:lseq F32.t 128).
    on gpu_loc (
      normalized |-> nv ** qkv |-> qv ** rotated |-> rv **
      rms_buffer |-> rb ** normalized_head |-> nh ** projection_tile |-> pt) **
    producer_handoff #inputs run (reveal observation) **
    pure ((reveal observation).rotated_buffer == rotated /\
      (reveal observation).rotated_values == rv /\
      source_qk_representation rv (canonical_query (input_mapping logical layer before))
        (canonical_key (input_mapping logical layer before)) /\
      source_values_representation rv (canonical_value (input_mapping logical layer before)) /\
      computed_cache_representation (reveal observation)
        (decoder_parts (input_mapping logical layer before) ((reveal logical).hidden 7)))
{
  let mapped = hide (input_mapping logical layer before);
  let padding = hide ((reveal logical).hidden 7);
  let producer_weights = hide ((reveal logical).weights);
  let data = hide ((reveal logical).attention);
  let incoming = hide (hidden_full mapped padding);
  let normalized_real = hide (normalized_hidden8 (reveal producer_weights).input_layernorm incoming);
  let qkv_real = hide (full_qkv mapped padding);
  source_input_matrix x logical layer before;
  source_weight_views weights producer_weights;
  full_qkv_unfold mapped padding;
  launch_kernel_1 (fun () ->
    rms_hidden_kernel input0 input1 normalized rms_buffer 0sz
      #x #weights #n0 #rb0 #fx #fw #incoming
      #(hide (reveal producer_weights).input_layernorm) #(hide True));
  with (nv:lseq bf16 8192). assert on gpu_loc (normalized |-> nv);
  initialized_as_uninit qkv #q0;
  run_projection 1024sz 4096sz 1sz 2048sz false normalized input1 input0 qkv projection_tile
    #nv #weights #x #pt0 #1.0R #fw #fx;
  projection_refines 1024 4096 1 2048 false nv weights x normalized_real (qkv_weight producer_weights)
    (fun _ _ -> 0.0R);
  with (qv:lseq bf16 32768). assert on gpu_loc (qkv |-> qv);
  assert pure (matrix_rep qv qkv_real);
  with (rb1:lseq F32.t 288). assert on gpu_loc (rms_buffer |-> rb1);
  launch_kernel_1 (fun () ->
    norm_rope_kernel qkv rotated input1 input2 normalized_head rms_buffer position
      #qv #r0 #weights #table #nh0 #rb1
      #1.0R #fw #ft #qkv_real #producer_weights #data);
  with (rv:lseq bf16 32768). assert on gpu_loc (rotated |-> rv);
  rotated_represents_canonical mapped padding rotated rv;
  let observation = snapshot_computed_cache rotated #rv #1.0R;
  unfold invocation_pending #inputs run;
  Execution.update run (produced_state inputs observation);
  Execution.take_snapshot run (produced_state inputs observation);
  fold producer_handoff #inputs run observation;
  observation
}

fn kplay_run
  (rows:U64.t{U64.v rows == 8})
  (hidden_width:U64.t{U64.v hidden_width == 1024})
  (head_width:U64.t{U64.v head_width == 128})
  (positions:U64.t{U64.v positions == 4096})
  (weights_length:U64.t{U64.v weights_length == weight_elements})
  (rope_length:U64.t{U64.v rope_length == rotary_elements})
  (input0:bf16_array hidden_elements{is_global_array input0})
  (input1:bf16_array weight_elements{is_global_array input1})
  (input2:bf16_array rotary_elements{is_global_array input2})
  (input3:I64.t{0 <= I64.v input3 /\ I64.v input3 <= 4089})
  (output0:bf16_array hidden_elements{is_global_array output0})
  (#x:erased (bf16_values hidden_elements))
  (#weights:erased (bf16_values weight_elements))
  (#table:erased (bf16_values rotary_elements))
  (#fx #fw #ft:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  (#inputs:erased invocation_inputs) (#run:erased (invocation inputs))
  preserves cpu ** on gpu_loc (
    input0 |-> Frac fx x ** input1 |-> Frac fw weights ** input2 |-> Frac ft table)
  requires on gpu_loc (Pulse.Lib.Array.pts_to_uninit output0 hidden_elements) **
    invocation_pending #inputs run **
    pure (input_representation x weights table (I64.v input3) logical /\
      inputs == native_inputs x weights table (I64.v input3) layer before)
  returns observation:erased cache_observation
  ensures exists* (result:bf16_values hidden_elements).
    on gpu_loc (output0 |-> result) **
    pure (computed_cache_representation (reveal observation)
      (decoder_parts (input_mapping logical layer before) ((reveal logical).hidden 7))) **
    invocation_consumed #inputs run (reveal observation) **
    completed_execution #inputs run (input_mapping logical layer before)
      ((reveal logical).hidden 7) result (reveal observation)
{
  let mapped = hide (input_mapping logical layer before);
  let padding = hide ((reveal logical).hidden 7);
  let w = hide ((reveal logical).weights);
  let data = hide ((reveal logical).attention);
  let incoming = hide (hidden_full mapped padding);
  let query = hide (canonical_query mapped);
  let key = hide (canonical_key mapped);
  let values = hide (canonical_value mapped);
  let mixed_real = hide (attention_model data query key values);
  let residual_real = hide (full_residual mapped padding);
  let postnorm_real = hide (full_postnorm mapped padding);
  let gateup_real = hide (full_gateup mapped padding);
  let activated_real = hide (activated_model gateup_real);
  source_weight_views weights w;
  let position = source_position input3;
  let normalized = gpu_array_alloc #bf16 8192sz;
  let qkv = gpu_array_alloc #bf16 32768sz;
  let rotated = gpu_array_alloc #bf16 32768sz;
  let mixed = gpu_array_alloc #bf16 16384sz;
  let residual = gpu_array_alloc #bf16 8192sz;
  let postnorm = gpu_array_alloc #bf16 8192sz;
  let gateup = gpu_array_alloc #bf16 49152sz;
  let activated = gpu_array_alloc #bf16 24576sz;
  let rms_buffer = gpu_array_alloc #F32.t 288sz;
  let normalized_head = gpu_array_alloc #bf16 128sz;
  let projection_tile = gpu_array_alloc #F32.t 128sz;
  with n0. assert on gpu_loc (normalized |-> n0);
  with q0. assert on gpu_loc (qkv |-> q0);
  with r0. assert on gpu_loc (rotated |-> r0);
  with rb0. assert on gpu_loc (rms_buffer |-> rb0);
  with nh0. assert on gpu_loc (normalized_head |-> nh0);
  with pt0. assert on gpu_loc (projection_tile |-> pt0);
  let observation = run_decoder_producer input0 input1 input2 position
    normalized qkv rotated rms_buffer normalized_head projection_tile
    #x #weights #table
    #(hide (seq_length_view 8192 (reveal n0)))
    #(hide (seq_length_view 32768 (reveal q0)))
    #(hide (seq_length_view 32768 (reveal r0)))
    #(hide (seq_length_view 288 (reveal rb0)))
    #(hide (seq_length_view 128 (reveal nh0)))
    #(hide (seq_length_view 128 (reveal pt0)))
    #fx #fw #ft #logical #layer #before #inputs #run;
  with (rv:lseq bf16 32768). assert on gpu_loc (rotated |-> rv);
  let guard = hide (observed_numerical_premises (reveal observation));
  let ab = allocate_attention_buffers ();
  with a0. assert on gpu_loc (mixed |-> a0);
  run_attention rotated mixed ab
    #rv #(hide (seq_length_view 16384 (reveal a0)))
    #1.0R #data #query #key #values #inputs #run #observation;
  map_loc gpu_loc
    #(attention_output_final mixed rv (attention_model data query key values))
    #(exists* (ys:lseq bf16 16384). mixed |-> ys **
      pure ((attention_domain rv ==> matrix_rep ys (attention_model data query key values)) /\
        (forall (col:Shared.index 2048). Seq.index ys (7 * 2048 + col) == (ObF.zero #bf16))))
    fn () { unfold attention_output_final mixed rv (attention_model data query key values); };
  with (av:lseq bf16 16384). assert on gpu_loc (mixed |-> av);
  free_attention_buffers ab;
  with res0. assert on gpu_loc (residual |-> res0);
  with (pt1:lseq F32.t 128). assert on gpu_loc (projection_tile |-> pt1);
  initialized_as_uninit residual #(hide (seq_length_view 8192 (reveal res0)));
  run_projection 2048sz 1024sz 0sz output_weight_offset true mixed input1 input0 residual projection_tile
    #av #weights #x
    #pt1 #1.0R #fw #fx;
  projection_refines_if 2048 1024 0 4196352 true av weights x mixed_real (reveal w).o_proj incoming guard;
  with (resv:lseq bf16 8192). assert on gpu_loc (residual |-> resv);
  assert pure (guard ==> matrix_rep resv residual_real);
  with pn0. assert on gpu_loc (postnorm |-> pn0);
  with (rb2:lseq F32.t 288). assert on gpu_loc (rms_buffer |-> rb2);
  launch_kernel_1 (fun () ->
    rms_hidden_kernel residual input1 postnorm rms_buffer 1024sz
      #resv #weights
      #(hide (seq_length_view hidden_elements (reveal pn0))) #rb2
      #1.0R #fw #residual_real #(hide (reveal w).post_attention_layernorm) #guard);
  with (pnv:lseq bf16 8192). assert on gpu_loc (postnorm |-> pnv);
  with gu0. assert on gpu_loc (gateup |-> gu0);
  with (pt2:lseq F32.t 128). assert on gpu_loc (projection_tile |-> pt2);
  initialized_as_uninit gateup #(hide (seq_length_view 49152 (reveal gu0)));
  run_projection 1024sz 6144sz 2sz gateup_weight_offset false postnorm input1 residual gateup projection_tile
    #pnv #weights #resv
    #pt2 #1.0R #fw #1.0R;
  projection_refines_if 1024 6144 2 6293760 false pnv weights resv
    postnorm_real (gateup_weight w) (fun _ _ -> 0.0R) guard;
  with (guv:lseq bf16 49152). assert on gpu_loc (gateup |-> guv);
  with ac0. assert on gpu_loc (activated |-> ac0);
  launch_kernel_1 (fun () ->
    silu_kernel gateup activated #guv
      #(hide (seq_length_view 24576 (reveal ac0))) #1.0R #gateup_real #guard);
  with (acv:lseq bf16 24576). assert on gpu_loc (activated |-> acv);
  with (pt3:lseq F32.t 128). assert on gpu_loc (projection_tile |-> pt3);
  run_projection 3072sz 1024sz 0sz down_weight_offset true activated input1 residual output0 projection_tile
    #acv #weights #resv
    #pt3 #1.0R #fw #1.0R;
  projection_refines_if 3072 1024 0 12585216 true acv weights resv
    activated_real (reveal w).down_proj residual_real guard;
  with result. assert on gpu_loc (output0 |-> result);
  full_output_math mapped padding;
  matrix_rep_if_transport result
    (projection_math 3072 1024 (reveal w).down_proj activated_real residual_real true)
    (decoder_parts mapped padding).hidden guard;
  assert pure (guard ==> output_representation result (decoder_parts mapped padding));
  assert pure (computed_cache_representation (reveal observation) (decoder_parts mapped padding));
  mapped_cache_shape_and_frame mapped (decoder_parts mapped padding);
  conditional_parts_postcondition mapped padding result (reveal observation);
  unfold consumer_handoff #inputs run observation;
  Execution.update run (finished_state inputs observation result);
  Execution.take_snapshot run (finished_state inputs observation result);
  drop_ (Execution.pts_to run #1.0R (finished_state inputs observation result));
  fold invocation_result #inputs run observation result;
  fold completed_execution #inputs run (input_mapping logical layer before)
    ((reveal logical).hidden 7) result observation;
  gpu_array_free normalized;
  gpu_array_free qkv;
  gpu_array_free rotated;
  gpu_array_free mixed;
  gpu_array_free residual;
  gpu_array_free postnorm;
  gpu_array_free gateup;
  gpu_array_free activated;
  gpu_array_free rms_buffer;
  gpu_array_free normalized_head;
  gpu_array_free projection_tile;
  observation
}
#pop-options

inline_for_extraction noextract
fn kplay_run_safety
  (rows:U64.t{U64.v rows == 8})
  (hidden_width:U64.t{U64.v hidden_width == 1024})
  (head_width:U64.t{U64.v head_width == 128})
  (positions:U64.t{U64.v positions == 4096})
  (weights_length:U64.t{U64.v weights_length == weight_elements})
  (rope_length:U64.t{U64.v rope_length == rotary_elements})
  (input0:bf16_array hidden_elements{is_global_array input0})
  (input1:bf16_array weight_elements{is_global_array input1})
  (input2:bf16_array rotary_elements{is_global_array input2})
  (input3:I64.t{0 <= I64.v input3 /\ I64.v input3 <= 4089})
  (output0:bf16_array hidden_elements{is_global_array output0})
  (#x:erased (bf16_values hidden_elements))
  (#weights:erased (bf16_values weight_elements))
  (#table:erased (bf16_values rotary_elements))
  (#fx #fw #ft:perm)
  (#logical:erased (d:input_denotation{valid_denotation d}))
  (#layer:erased (Shared.index 28))
  (#before:erased (c:Shared.cache_state{Shared.valid_cache_before layer c}))
  preserves cpu ** on gpu_loc (
    input0 |-> Frac fx x ** input1 |-> Frac fw weights ** input2 |-> Frac ft table)
  requires on gpu_loc (Pulse.Lib.Array.pts_to_uninit output0 hidden_elements) **
    pure (input_representation x weights table (I64.v input3) logical)
  ensures exists* (result:bf16_values hidden_elements). on gpu_loc (output0 |-> result)
{
  let inputs = hide (native_inputs x weights table (I64.v input3) layer before);
  let run = begin_invocation #inputs;
  let observation = kplay_run rows hidden_width head_width positions weights_length rope_length
    input0 input1 input2 input3 output0 #x #weights #table #fx #fw #ft #logical #layer #before #inputs #run;
  with result. assert on gpu_loc (output0 |-> result);
  drop_ (invocation_consumed #inputs run observation);
  drop_ (completed_execution #inputs run (input_mapping logical layer before)
    ((reveal logical).hidden 7) result observation);
  ()
}
