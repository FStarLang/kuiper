module KPlay.SharedSpec

open FStar.Real

module Sum = Kuiper.Sum
module Sqrt = FStar.Math.Sqrt

(* Logical real denotations under Kuiper's algebraic approximation model.
   These are not IEEE bit patterns; representation and comparison-preservation
   proofs remain obligations of the two roles. See M.json for the trust boundary. *)

inline_for_extraction noextract
let prompt_length : pos = 7

inline_for_extraction noextract
let hidden_size : pos = 1024

inline_for_extraction noextract
let vocab_size : pos = 151936

inline_for_extraction noextract
let rms_epsilon : Sqrt.rpos = 0.000001R

inline_for_extraction noextract
type prompt_index = i:nat{i < prompt_length}

inline_for_extraction noextract
type hidden_index = i:nat{i < hidden_size}

inline_for_extraction noextract
type vocab_index = i:nat{i < vocab_size}

inline_for_extraction noextract
type hidden_row = hidden_index -> GTot real

inline_for_extraction noextract
type hidden_rows = prompt_index -> GTot hidden_row

inline_for_extraction noextract
type tied_embedding = vocab_index -> GTot hidden_row

inline_for_extraction noextract
type vocabulary_logits = vocab_index -> GTot real

inline_for_extraction noextract
noeq type inputs = {
  hidden_in: hidden_rows;
  norm_weight: hidden_row;
  embedding: tied_embedding
}

inline_for_extraction noextract
noeq type outputs = {
  normalized_hidden: hidden_rows;
  final_logits: vocabulary_logits
}

inline_for_extraction noextract
let sum_hidden (f: hidden_index -> GTot real) : GTot real =
  Sum.sum #real #Sum.real_add_semigroup 0 hidden_size f

inline_for_extraction noextract
let rec sum_nonnegative
  (f: hidden_index -> GTot Sqrt.rnonneg)
  (lo: nat{lo <= hidden_size})
  : Lemma
      (Sum.sum #real #Sum.real_add_semigroup lo hidden_size f >=. 0.0R)
      (decreases (hidden_size - lo))
  =
  if lo < hidden_size then (
    sum_nonnegative f (lo + 1);
    Sum.sum_pop_left #real #Sum.real_add_semigroup lo hidden_size f;
    let tail = Sum.sum #real #Sum.real_add_semigroup (lo + 1) hidden_size f in
    assert_norm (Sum.op #real #Sum.real_add_semigroup (f lo) tail ==
      f lo +. tail)
  ) else (
    assert_norm (Sum.id #real #Sum.real_add_semigroup == 0.0R)
  )

inline_for_extraction noextract
let mean_square (row: hidden_row) : GTot Sqrt.rnonneg =
  let squares: hidden_index -> GTot Sqrt.rnonneg =
    fun j -> row j *. row j in
  sum_nonnegative squares 0;
  sum_hidden squares /. FStar.Real.of_int hidden_size

inline_for_extraction noextract
let normalization_scale (row: hidden_row) : GTot Sqrt.rpos =
  Sqrt.rsqrt (mean_square row +. rms_epsilon)

(* RMSNorm casts the scaled row back before applying gamma. Casts preserve
   the real denotation here; this does not move a concrete rounding boundary. *)
inline_for_extraction noextract
let final_rms_norm (x: inputs) : GTot hidden_rows =
  fun p ->
    let row = x.hidden_in p in
    let scale = normalization_scale row in
    fun d -> x.norm_weight d *. (row d *. scale)

inline_for_extraction noextract
let tied_projection
  (embedding: tied_embedding)
  (last_hidden: hidden_row)
  : GTot vocabulary_logits
  =
  fun v -> sum_hidden (fun d -> last_hidden d *. embedding v d)

inline_for_extraction noextract
let final_norm_projection (x: inputs) : GTot outputs =
  let y = final_rms_norm x in
  {
    normalized_hidden = y;
    final_logits = tied_projection x.embedding (y (prompt_length - 1))
  }

(* The ambient state is the state at entry to this suffix, after the decoder;
   no restriction on its representation or buffer ownership is imposed. *)
inline_for_extraction noextract
let transition (#ambient: Type0) (x: inputs) (before: ambient)
  : GTot (outputs & ambient)
  =
  (final_norm_projection x, before)

inline_for_extraction noextract
let postcondition
  (#ambient: Type0)
  (x: inputs)
  (before: ambient)
  (observed: outputs)
  (after: ambient)
  : GTot prop
  =
  let expected = final_norm_projection x in
  (forall (p: prompt_index) (d: hidden_index).
    observed.normalized_hidden p d == expected.normalized_hidden p d) /\
  (forall (v: vocab_index).
    observed.final_logits v == expected.final_logits v) /\
  after == before

inline_for_extraction noextract
let transition_satisfies_postcondition
  (#ambient: Type0) (x: inputs) (before: ambient)
  : Lemma
      (let (observed, after) = transition x before in
       postcondition x before observed after)
  = ()

(* A strict improvement replaces the incumbent; equal logits keep the
   earlier ID. The refinement proves maximality and the first-index rule. *)
inline_for_extraction noextract
let rec first_argmax_prefix
  (logits: vocabulary_logits)
  (count: pos{count <= vocab_size})
  : GTot (best: vocab_index{
      best < count /\
      (forall (j: nat{j < count}). logits j <=. logits best) /\
      (forall (j: nat{j < best}). logits j <. logits best)})
      (decreases count)
  =
  if count = 1 then 0
  else
    let last = count - 1 in
    let best = first_argmax_prefix logits last in
    if t2b (logits best <. logits last) then last else best

inline_for_extraction noextract
let next_token (observed: outputs)
  : GTot (best: vocab_index{
      (forall (v: vocab_index).
        observed.final_logits v <=. observed.final_logits best) /\
      (forall (v: nat{v < best}).
        observed.final_logits v <. observed.final_logits best)})
  =
  first_argmax_prefix observed.final_logits vocab_size

inline_for_extraction noextract
type prompt_tokens = prompt_index -> Tot vocab_index

inline_for_extraction noextract
type sequence_index = i:nat{i < prompt_length + 1}

inline_for_extraction noextract
type observed_tokens = sequence_index -> GTot vocab_index

(* Derived base.py observation, not another suffix output. *)
inline_for_extraction noextract
let observe_tokens (prompt: prompt_tokens) (observed: outputs)
  : GTot observed_tokens
  =
  let selected = next_token observed in
  fun i -> if i < prompt_length then prompt i else selected
