module KPlay.SharedSpec

open FStar.Real
module S = FStar.Seq.Base

(* Logical boundary: base.py's already-produced logits and original prompt.
   Reals use the paper's algebraic abstraction, not an IEEE error model.
   A role's mapping must preserve finite-score comparisons and numerical ties;
   v_approximates alone is not a comparison theorem. NaN/infinite final-row
   scores are outside this abstraction. Signed zeros have equal real values.
   The authorized argmax-policy.txt revision relaxes only tied-index preference
   in postcondition; the canonical reference functions remain deterministic. *)

inline_for_extraction noextract
let vocab_size : pos = 151936

inline_for_extraction noextract
let prompt_length : pos = 7

inline_for_extraction noextract
let sequence_length : pos = prompt_length + 1

inline_for_extraction noextract
type token_id = i:nat{i < vocab_size}

inline_for_extraction noextract
type prompt_tokens = p:S.seq token_id{S.length p == prompt_length}

inline_for_extraction noextract
type vocabulary_logits = s:S.seq real{S.length s == vocab_size}

inline_for_extraction noextract
type prompt_logits =
  l:S.seq vocabulary_logits{S.length l == prompt_length}

inline_for_extraction noextract
type decode_steps = n:nat{n == 1}

inline_for_extraction noextract
type returned_tokens = q:S.seq token_id{S.length q == sequence_length}

inline_for_extraction noextract
type result = token_id & returned_tokens

inline_for_extraction noextract
let final_logits (logits:prompt_logits) : Tot vocabulary_logits =
  S.index logits (prompt_length - 1)

(* The strict replacement test retains the earlier index on every tie.
   Starting at a real entry, rather than a numeric sentinel, also covers
   an all-negative row. This fold specifies no physical execution order. *)
inline_for_extraction noextract
let rec argmax_prefix
  (scores:vocabulary_logits)
  (count:pos{count <= vocab_size})
  : GTot (winner:nat{winner < count}) (decreases count)
  =
  if count = 1 then 0
  else
    let previous = argmax_prefix scores (count - 1) in
    if t2b (S.index scores (count - 1) >. S.index scores previous)
    then count - 1
    else previous

inline_for_extraction noextract
let first_maximum
  (scores:vocabulary_logits)
  (count:pos{count <= vocab_size})
  (winner:nat{winner < count})
  : prop
  =
  (forall (j:nat{j < count}).
    S.index scores j <=. S.index scores winner) /\
  (forall (j:nat{j < winner}).
    S.index scores j <. S.index scores winner)

inline_for_extraction noextract
let rec argmax_prefix_correct
  (scores:vocabulary_logits)
  (count:pos{count <= vocab_size})
  : Lemma
      (ensures first_maximum scores count (argmax_prefix scores count))
      (decreases count)
  =
  if count = 1 then ()
  else (
    argmax_prefix_correct scores (count - 1);
    let previous = argmax_prefix scores (count - 1) in
    if t2b (S.index scores (count - 1) >. S.index scores previous)
    then ()
    else ())

inline_for_extraction noextract
let first_maximum_unique
  (scores:vocabulary_logits)
  (left right:token_id)
  : Lemma
      (requires first_maximum scores vocab_size left /\
                first_maximum scores vocab_size right)
      (ensures left == right)
  = ()

inline_for_extraction noextract
let maximal_index (scores:vocabulary_logits) (winner:token_id) : prop =
  forall (j:token_id). S.index scores j <=. S.index scores winner

inline_for_extraction noextract
let selected_index (logits:prompt_logits) : GTot token_id =
  argmax_prefix (final_logits logits) vocab_size

inline_for_extraction noextract
let append_token (prompt:prompt_tokens) (selected:token_id)
  : Tot returned_tokens
  =
  S.append prompt (S.create 1 selected)

(* Result positions are logical outputs, not base.py tuple/physical ABI slots.
   The sequence is base.py's first returned component; its hidden-state
   component and upstream cache construction are outside this operation. *)
inline_for_extraction noextract
let argmax_append
  (prompt:prompt_tokens)
  (logits:prompt_logits)
  (_steps:decode_steps)
  : GTot result
  =
  let selected = selected_index logits in
  (selected, append_token prompt selected)

(* Membership in exactly {(w, prompt ++ [w]) | maximal_index (final_logits
   logits) w}. Neither first_maximum nor equality to argmax_append is required. *)
inline_for_extraction noextract
let postcondition
  (prompt:prompt_tokens)
  (logits:prompt_logits)
  (steps:decode_steps)
  (output:result)
  : GTot prop
  =
  maximal_index (final_logits logits) output._1 /\
  output._2 == append_token prompt output._1

inline_for_extraction noextract
let argmax_append_correct
  (prompt:prompt_tokens)
  (logits:prompt_logits)
  (steps:decode_steps)
  : Lemma
      (ensures
        (let output = argmax_append prompt logits steps in
         first_maximum (final_logits logits) vocab_size output._1 /\
         S.length output._2 == sequence_length /\
         (forall (position:nat{position < prompt_length}).
           S.index output._2 position == S.index prompt position) /\
         S.index output._2 prompt_length == output._1))
  =
  argmax_prefix_correct (final_logits logits) vocab_size

inline_for_extraction noextract
let argmax_append_satisfies_postcondition
  (prompt:prompt_tokens)
  (logits:prompt_logits)
  (steps:decode_steps)
  : Lemma
      (ensures
        postcondition prompt logits steps (argmax_append prompt logits steps))
  =
  argmax_append_correct prompt logits steps

inline_for_extraction noextract
let maximal_index_append_allowed
  (prompt:prompt_tokens)
  (logits:prompt_logits)
  (steps:decode_steps)
  (selected:token_id)
  : Lemma
      (requires maximal_index (final_logits logits) selected)
      (ensures
        postcondition prompt logits steps
          (selected, append_token prompt selected))
  = ()

inline_for_extraction noextract
let postcondition_outputs
  (prompt:prompt_tokens)
  (logits:prompt_logits)
  (steps:decode_steps)
  (output:result)
  : Lemma
      (requires postcondition prompt logits steps output)
      (ensures
        maximal_index (final_logits logits) output._1 /\
        S.length output._2 == sequence_length /\
        (forall (position:nat{position < prompt_length}).
          S.index output._2 position == S.index prompt position) /\
        S.index output._2 prompt_length == output._1)
  = ()

inline_for_extraction noextract
let constant_row_selects_zero
  (scores:vocabulary_logits)
  (value:real)
  : Lemma
      (requires forall (j:token_id). S.index scores j == value)
      (ensures argmax_prefix scores vocab_size == 0)
  =
  argmax_prefix_correct scores vocab_size;
  first_maximum_unique scores (argmax_prefix scores vocab_size) 0

(* Nonzero selections on a constant row are allowed even though their complete
   outputs differ from the canonical reference result. *)
inline_for_extraction noextract
let constant_row_all_indices_allowed
  (prompt:prompt_tokens)
  (logits:prompt_logits)
  (steps:decode_steps)
  (value:real)
  (selected:token_id)
  : Lemma
      (requires
        forall (j:token_id). S.index (final_logits logits) j == value)
      (ensures
        (let output = (selected, append_token prompt selected) in
         postcondition prompt logits steps output /\
         (selected <> 0 ==> output =!= argmax_append prompt logits steps)))
  =
  constant_row_selects_zero (final_logits logits) value;
  maximal_index_append_allowed prompt logits steps selected
