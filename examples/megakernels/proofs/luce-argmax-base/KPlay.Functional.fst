module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module SZ = Kuiper.SizeT
module A = Pulse.Lib.Array
module F = Kuiper.Floating
module Shared = KPlay.SharedSpec

// Keep erased contracts folded while the extractor translates function types.
noextract
let selection_result (values:Shared.vec bf16 vocabulary) (winner:sz) : prop =
  SZ.v winner < vocabulary /\
  SZ.v winner == Shared.argmax (logical_scores values)

noextract
let run_result
  (position:position_t) (prompt_length:prompt_length_t)
  (scores:Shared.vec bf16 vocabulary)
  (previous:Shared.vec i64 (previous_count position prompt_length))
  (selected:Shared.vec i64 1)
  (emitted:Shared.vec i64 (output_count position prompt_length))
  : prop =
  functional_post (prompt_size prompt_length) (position_index position)
    scores previous selected emitted

inline_for_extraction noextract
let replace_best (candidate current:f32)
  : result:bool{result == not (priority_le candidate current)} =
  native_order candidate current;
  F.eq_spec candidate candidate;
  F.eq_spec current current;
  F.lt current candidate ||
    (not (F.ieee_eq candidate candidate) && F.ieee_eq current current)

inline_for_extraction noextract
let scan_step
  (#v:Shared.size) (xs:Shared.vec bf16 v)
  (n:Shared.size{n < v}) (winner:nat{winner < n})
  : Lemma
      (requires winner == Shared.argmax_prefix (logical_scores xs) n)
      (ensures (
        let chosen =
          if replace_best (widen (S.index xs n)) (widen (S.index xs winner))
          then n else winner
        in
        Shared.argmax_prefix (logical_scores xs) (n + 1) == chosen /\
        (forall (j:nat{j < n + 1}).
          priority_le (widen (S.index xs j)) (widen (S.index xs chosen))))) =
  score_key_order (widened_scores xs) n winner;
  native_prefix_correct xs (n + 1) (Shared.argmax_prefix (logical_scores xs) (n + 1))

inline_for_extraction noextract
fn select_token
  (#f:perm)
  (#values:erased (Shared.vec bf16 vocabulary))
  (scores:larray bf16 vocabulary)
  preserves gpu ** A.pts_to scores #f (FStar.Ghost.reveal values)
  returns winner:sz
  ensures pure (selection_result (FStar.Ghost.reveal values) winner)
{
  let first = scores.(0sz);
  let mut best = widen first;
  let mut winner = 0sz;
  let mut i = 1sz;
  while (!i <^ SZ.uint_to_t 248320)
    invariant exists* (n w:sz) (b:f32).
      i |-> n ** winner |-> w ** best |-> b **
      pure (
        0 < SZ.v n /\ SZ.v n <= vocabulary /\ SZ.v w < SZ.v n /\
        b == widen (S.index (FStar.Ghost.reveal values) (SZ.v w)) /\
        SZ.v w == Shared.argmax_prefix (logical_scores (FStar.Ghost.reveal values)) (SZ.v n))
    decreases (vocabulary - SZ.v !i)
  {
    let n = !i;
    let w = !winner;
    let b = !best;
    let value = scores.(n);
    let candidate = widen value;
    scan_step (FStar.Ghost.reveal values) (SZ.v n) (SZ.v w);
    if (replace_best candidate b) {
      best := candidate;
      winner := n;
    };
    i := n +^ 1sz;
  };
  !winner
}

inline_for_extraction noextract
let count_to_sz (x:i64{0 <= I64.v x /\ I64.v x <= 2048})
  : result:sz{SZ.v result == I64.v x} =
  SZ.uint32_to_sizet (IC.int64_to_uint32 x)

inline_for_extraction noextract
let previous_extent (position:position_t) (prompt_length:prompt_length_t)
  : result:sz{SZ.v result == previous_count position prompt_length} =
  if I64.gte position prompt_length
  then count_to_sz (I64.add (I64.sub position prompt_length) 1L)
  else 0sz

inline_for_extraction noextract
fn emit_tokens
  (position:position_t)
  (prompt_length:prompt_length_t)
  (token:i64)
  (#previous_fraction:perm)
  (#previous_values:erased (Shared.vec i64 (previous_count position prompt_length)))
  (#old_emitted:erased (Shared.vec i64 (output_count position prompt_length)))
  (previous:array i64)
  (emitted:array i64)
  preserves gpu ** active_contents previous #previous_fraction (FStar.Ghost.reveal previous_values)
  requires active_contents emitted #1.0R (FStar.Ghost.reveal old_emitted)
  ensures active_contents emitted #1.0R
    (accumulated_tokens #(prompt_size prompt_length) (position_index position)
      (FStar.Ghost.reveal previous_values) (S.create 1 token))
{
  let count = previous_extent position prompt_length;
  let mut i = 0sz;
  while (!i <^ count)
    invariant exists* (n:sz)
                      (values:Shared.vec i64 (output_count position prompt_length)).
      i |-> n ** active_contents emitted #1.0R values **
      pure (SZ.v n <= SZ.v count /\
        (forall (j:nat{j < SZ.v n}).
          S.index values j == S.index (FStar.Ghost.reveal previous_values) j))
    decreases (SZ.v count - SZ.v !i)
  {
    let n = !i;
    with values. assert active_contents emitted #1.0R values;
    rewrite (active_contents emitted #1.0R values) as (A.pts_to emitted values);
    rewrite (active_contents previous #previous_fraction (FStar.Ghost.reveal previous_values))
      as (A.pts_to previous #previous_fraction (FStar.Ghost.reveal previous_values));
    let value = previous.(n);
    emitted.(n) <- value;
    with updated. assert A.pts_to emitted updated;
    rewrite (A.pts_to emitted updated) as (active_contents emitted #1.0R updated);
    rewrite (A.pts_to previous #previous_fraction (FStar.Ghost.reveal previous_values))
      as (active_contents previous #previous_fraction (FStar.Ghost.reveal previous_values));
    i := n +^ 1sz;
  };
  let expected:erased (Shared.vec i64 (output_count position prompt_length)) =
    accumulated_tokens #(prompt_size prompt_length) (position_index position)
      (FStar.Ghost.reveal previous_values) (S.create 1 token);
  if (I64.gte position (I64.sub prompt_length 1L)) {
    with values. assert active_contents emitted #1.0R values;
    rewrite (active_contents emitted #1.0R values) as (A.pts_to emitted values);
    emitted.(count) <- token;
    with values. assert A.pts_to emitted values;
    assert pure (S.equal values (FStar.Ghost.reveal expected));
    S.lemma_eq_elim values (FStar.Ghost.reveal expected);
    rewrite (A.pts_to emitted (FStar.Ghost.reveal expected))
      as (active_contents emitted #1.0R (FStar.Ghost.reveal expected));
  } else {
    with values. assert active_contents emitted #1.0R values;
    assert pure (S.equal values (FStar.Ghost.reveal expected));
    S.lemma_eq_elim values (FStar.Ghost.reveal expected);
  }
}

inline_for_extraction noextract
let selection_post
  (#v:Shared.size) (p:Shared.size) (completed:nat)
  (scores:Shared.vec bf16 v)
  (previous:Shared.vec i64 (Shared.retained_count p completed))
  (token:i64)
  : Lemma
      (requires I64.v token == Shared.argmax (logical_scores scores))
      (ensures functional_post p completed scores previous (S.create 1 token)
        (accumulated_tokens #p completed previous (S.create 1 token))) =
  Shared.argmax_correct (logical_scores scores);
  assert (Shared.first_maximum (logical_scores scores) (I64.v token))

inline_for_extraction noextract
fn kernel
  (position:position_t)
  (prompt_length:prompt_length_t)
  (#score_fraction #previous_fraction:perm)
  (#score_values:erased (Shared.vec bf16 vocabulary))
  (#previous_values:erased (Shared.vec i64 (previous_count position prompt_length)))
  (#old_selected:erased (Shared.vec i64 1))
  (#old_emitted:erased (Shared.vec i64 (output_count position prompt_length)))
  (scores:larray bf16 vocabulary)
  (previous:array i64)
  (selected:larray i64 1)
  (emitted:array i64)
  preserves gpu **
    A.pts_to scores #score_fraction (FStar.Ghost.reveal score_values) **
    active_contents previous #previous_fraction (FStar.Ghost.reveal previous_values)
  requires
    A.pts_to selected (FStar.Ghost.reveal old_selected) **
    active_contents emitted #1.0R (FStar.Ghost.reveal old_emitted)
  ensures
    exists* (selected_values:Shared.vec i64 1)
            (emitted_values:Shared.vec i64 (output_count position prompt_length)).
      A.pts_to selected selected_values ** active_contents emitted #1.0R emitted_values **
      pure (run_result position prompt_length
        (FStar.Ghost.reveal score_values) (FStar.Ghost.reveal previous_values)
        selected_values emitted_values)
{
  let winner = select_token #score_fraction scores;
  let token = IC.uint32_to_int64 (SZ.sizet_to_u32 winner);
  assert pure (I64.v token == SZ.v winner);
  selected.(0sz) <- token;
  with selected_values. assert A.pts_to selected selected_values;
  assert pure (S.equal selected_values (S.create 1 token));
  S.lemma_eq_elim selected_values (S.create 1 token);
  emit_tokens position prompt_length token #previous_fraction previous emitted;
  selection_post (prompt_size prompt_length) (position_index position)
    (FStar.Ghost.reveal score_values) (FStar.Ghost.reveal previous_values) token;
  ()
}

inline_for_extraction noextract
instance send_global_array_contents
  (#a:Type0) (arr:array a{is_global_array arr}) (#f:perm) (contents:S.seq a)
  : is_send_across gpu_of (A.pts_to arr #f contents) =
  Kuiper.Array.Core.is_send_pts_to arr #f contents

inline_for_extraction noextract
instance send_active_contents
  (#a:Type0) (arr:array a) (#f:perm)
  (contents:S.seq a{active_global_array (S.length contents) arr})
  : is_send_across gpu_of (active_contents arr #f contents) =
  if S.length contents = 0 then
    let empty_send : is_send_across gpu_of emp = solve in empty_send
  else send_global_array_contents arr #f contents

fn kplay_run
  (position:position_t)
  (prompt_length:prompt_length_t)
  (#score_fraction #previous_fraction:exclusive_input_permission)
  (#score_values:erased (Shared.vec bf16 vocabulary))
  (#previous_values:erased (Shared.vec i64 (previous_count position prompt_length)))
  (scores:larray bf16 vocabulary{is_global_array scores})
  (previous:array i64{active_global_array (previous_count position prompt_length) previous})
  (selected:larray i64 1{is_global_array selected})
  (emitted:array i64{active_global_array (output_count position prompt_length) emitted})
  preserves cpu **
    on gpu_loc (
      A.pts_to scores #score_fraction (FStar.Ghost.reveal score_values) **
      active_contents previous #previous_fraction (FStar.Ghost.reveal previous_values))
  requires
    exists* (old_selected:Shared.vec i64 1)
            (old_emitted:Shared.vec i64 (output_count position prompt_length)).
      on gpu_loc (A.pts_to selected old_selected ** active_contents emitted #1.0R old_emitted)
  ensures
    exists* (selected_values:Shared.vec i64 1)
            (emitted_values:Shared.vec i64 (output_count position prompt_length)).
      on gpu_loc (A.pts_to selected selected_values ** active_contents emitted #1.0R emitted_values) **
      pure (run_result position prompt_length
        (FStar.Ghost.reveal score_values) (FStar.Ghost.reveal previous_values)
        selected_values emitted_values)
{
  with old_selected old_emitted.
    assert on gpu_loc (A.pts_to selected old_selected ** active_contents emitted #1.0R old_emitted);
  launch_kernel_1 (fun () ->
    kernel position prompt_length #score_fraction #previous_fraction
      #score_values #previous_values #old_selected #old_emitted
      scores previous selected emitted);
}
