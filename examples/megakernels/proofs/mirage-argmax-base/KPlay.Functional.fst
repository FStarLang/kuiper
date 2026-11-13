module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq.Base
module C = FStar.Classical
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module F = Kuiper.Floating
module Cast = Kuiper.Float.Casts.Base
module SZ = Kuiper.SizeT
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let score_is_nan (value:F32.t) : b:bool{b == F.is_nan value} =
  F.ieee_eq_refl value;
  not (F.ieee_eq value value)

inline_for_extraction noextract
let row_offset
  (row:I64.t{0 <= I64.v row /\ I64.v row < prompt_length})
  : offset:sz{SZ.v offset == I64.v row * vocab_size} =
  let r = SZ.uint32_to_sizet (IC.int64_to_uint32 row) in
  r *^ SZ.uint32_to_sizet 151936ul

inline_for_extraction noextract
let token_as_int64 (index:szlt vocab_size)
  : token:I64.t{I64.v token == SZ.v index} =
  IC.uint32_to_int64 (SZ.sizet_to_u32 index)

noextract
let scan_state (s:scores vocab_size) (count winner:sz) (best:F32.t) : GTot prop =
  if 1 <= SZ.v count && SZ.v count <= vocab_size && SZ.v winner < SZ.v count
  then
    SZ.v winner == select_prefix s (SZ.v count) /\
    best == S.index s (SZ.v winner)
  else False

(* Avoid expanding the full logical fold while erasing runtime signatures. *)
noextract
let row_selection_result
  (logits:physical_logits) (row:row_index) (winner:token_index) : GTot prop =
  winner == select_prefix (row_scores logits row) vocab_size

inline_for_extraction noextract
fn select_row
  (logits:larray BF16.t logits_size)
  (row:I64.t{0 <= I64.v row /\ I64.v row < prompt_length})
  (#input_logits:erased physical_logits)
  preserves gpu ** logits |-> input_logits
  returns winner:szlt vocab_size
  ensures pure (row_selection_result input_logits (I64.v row) (SZ.v winner))
{
  let s:erased (scores vocab_size) = row_scores input_logits (I64.v row);
  let offset = row_offset row;
  let initial = logits.(offset);
  let mut best = Cast.cast_bf16_to_f32 initial;
  let mut winner = 0sz;
  let mut count = 1sz;
  let limit = SZ.uint32_to_sizet 151936ul;
  while (!count <^ limit)
    invariant exists* (k w:sz) (b:F32.t).
      count |-> k ** winner |-> w ** best |-> b **
      pure (scan_state s k w b)
    decreases (vocab_size - SZ.v !count)
  {
    let k = !count;
    let w = !winner;
    let previous = !best;
    let stored = logits.(offset +^ k);
    let current = Cast.cast_bf16_to_f32 stored;
    let replace =
      not (score_is_nan previous) &&
      (score_is_nan current || lt previous current);
    assert pure (
      select_prefix s (SZ.v k + 1) == (if replace then SZ.v k else SZ.v w));
    if replace {
      best := current;
      winner := k;
    };
    count := k +^ 1sz;
  };
  let selected = !winner;
  selected
}

noextract
let append_selected
  (logits:physical_logits) (prefix:physical_prompt) (row:row_index)
  (copied:physical_output)
  (winner:token_index)
  (token:I64.t{I64.v token == winner})
  : Lemma
      (requires
        (forall (j:nat{j < prompt_length}).
          S.index copied j == S.index prefix j) /\
        row_selection_result logits row winner)
      (ensures
        postcondition logits prefix row (S.upd copied prompt_length token)) =
  select_prefix_correct (row_scores logits row) vocab_size;
  C.exists_intro
    (fun (w:token_index) ->
      I64.v (S.index (S.upd copied prompt_length token) prompt_length) == w /\
      first_winner (row_scores logits row) vocab_size w)
    winner;
  establish_postcondition logits prefix row (S.upd copied prompt_length token)

inline_for_extraction noextract
fn select_token
  (logits:larray BF16.t logits_size)
  (prefix:larray I64.t prompt_length)
  (row:I64.t{0 <= I64.v row /\ I64.v row < prompt_length})
  (output:larray I64.t sequence_length)
  (#input_logits:erased physical_logits)
  (#input_prefix:erased physical_prompt)
  (#old:erased (seq I64.t))
  preserves gpu ** logits |-> input_logits ** prefix |-> input_prefix
  requires output |-> old
  ensures exists* (result:physical_output).
    output |-> result **
    pure (postcondition input_logits input_prefix (I64.v row) result)
{
  pts_to_len output;
  let mut position = 0sz;
  while (!position <^ 7sz)
    invariant exists* (j:sz{SZ.v j <= prompt_length}) (values:physical_output).
      position |-> j ** output |-> values **
      pure (forall (i:nat{i < SZ.v j}).
        S.index values i == S.index input_prefix i)
    decreases (prompt_length - SZ.v !position)
  {
    let j = !position;
    let token = prefix.(j);
    output.(j) <- token;
    position := j +^ 1sz;
  };
  with copied. assert output |-> copied;
  let winner = select_row logits row #input_logits;
  let token = token_as_int64 winner;
  append_selected input_logits input_prefix (I64.v row) copied (SZ.v winner) token;
  output.(7sz) <- token;
}

inline_for_extraction noextract
instance send_global_array_contents
  (#a:Type0)
  (arr:array a{is_global_array arr})
  (#f:perm)
  (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents) =
  Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (logits:larray BF16.t logits_size{is_global_array logits})
  (prefix:larray I64.t prompt_length{is_global_array prefix})
  (score_row:I64.t{0 <= I64.v score_row /\ I64.v score_row < prompt_length})
  (output:larray I64.t sequence_length{is_global_array output})
  (#input_logits:erased physical_logits)
  (#input_prefix:erased physical_prompt)
  preserves cpu **
    on gpu_loc (logits |-> input_logits) **
    on gpu_loc (prefix |-> input_prefix)
  requires exists* old. on gpu_loc (output |-> old)
  ensures exists* (result:physical_output).
    on gpu_loc (output |-> result) **
    pure (postcondition input_logits input_prefix (I64.v score_row) result)
{
  with old. assert on gpu_loc (output |-> old);
  launch_kernel_1 (fun () ->
    select_token logits prefix score_row output #input_logits #input_prefix #old);
}
