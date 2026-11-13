module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq.Base
module I32 = FStar.Int32
module I64 = FStar.Int64
module Cast = FStar.Int.Cast
module SZ = Kuiper.SizeT
module R = FStar.Real
module PA = Pulse.Lib.Array
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
ghost
fn borrow_initial_alias
  (#n:pos)
  (input1:larray i64 n)
  (input2:larray i64 1)
  (#prompt:erased (lseq i64 n))
  requires pure (input2 == PA.gsub input1 0 1)
  requires PA.pts_to input1 #1.0R prompt
  ensures
    PA.pts_to input1 #0.5R prompt **
    PA.pts_to input2 #0.5R (S.slice prompt 0 1) **
    PA.pts_to_range input1 1 n #0.5R (S.slice prompt 1 n)
{
  PA.share input1;
  PA.pts_to_range_intro input1 0.5R prompt;
  PA.pts_to_range_split input1 0 1 n #0.5R #prompt;
  with prefix suffix.
    assert PA.pts_to_range input1 0 1 #0.5R prefix **
      PA.pts_to_range input1 1 n #0.5R suffix;
  PA.unfold_pts_to_range input1 0 1 #0.5R prefix;
  with cells.
    assert PA.pts_to_mask (PA.gsub input1 0 1) #0.5R cells (fun _ -> True);
  rewrite (PA.pts_to_mask (PA.gsub input1 0 1) #0.5R cells (fun _ -> True))
    as (PA.pts_to_mask input2 #0.5R cells (fun _ -> True));
  PA.from_mask input2;
  with values. assert PA.pts_to input2 #0.5R values;
  S.lemma_eq_intro values (S.slice prompt 0 1);
  S.lemma_eq_elim values (S.slice prompt 0 1);
}

inline_for_extraction noextract
ghost
fn return_initial_alias
  (#n:pos)
  (input1:larray i64 n)
  (input2:larray i64 1)
  (#prompt:erased (lseq i64 n))
  requires pure (input2 == PA.gsub input1 0 1)
  requires
    PA.pts_to input1 #0.5R prompt **
    PA.pts_to input2 #0.5R (S.slice prompt 0 1) **
    PA.pts_to_range input1 1 n #0.5R (S.slice prompt 1 n)
  ensures PA.pts_to input1 #1.0R prompt
{
  PA.to_mask input2;
  with cells. assert PA.pts_to_mask input2 #0.5R cells (fun _ -> True);
  rewrite (PA.pts_to_mask input2 #0.5R cells (fun _ -> True))
    as (PA.pts_to_mask (PA.gsub input1 0 1) #0.5R cells (fun _ -> True));
  PA.fold_pts_to_range input1 0 1 #0.5R (S.slice prompt 0 1);
  PA.pts_to_range_join input1 0 1 n #0.5R;
  S.lemma_eq_intro prompt (S.append (S.slice prompt 0 1) (S.slice prompt 1 n));
  S.lemma_eq_elim prompt (S.append (S.slice prompt 0 1) (S.slice prompt 1 n));
  PA.pts_to_range_elim input1 0.5R prompt;
  PA.gather input1;
}

inline_for_extraction noextract
let narrow_i32 (raw:i64) : Tot (token:i32{I32.v token == narrow_token raw})
  = assert_norm (pow2 32 == 4294967296);
    Cast.int64_to_int32 raw

inline_for_extraction noextract
fn select_row
  (prompt_length:i64{
    0 < I64.v prompt_length /\ I64.v prompt_length <= position_limit})
  (position:i64{
    0 <= I64.v position /\ I64.v position < position_limit})
  (input1:larray i64 (I64.v prompt_length))
  (input2:larray i64 1)
  (#prompt:erased (lseq i64 (I64.v prompt_length)))
  (#previous_token:erased (lseq i64 1))
  (#t:erased Shared.token_id)
  (#fprompt #fprevious:perm)
  preserves gpu **
    input1 |-> Frac fprompt prompt **
    input2 |-> Frac fprevious previous_token
  requires pure (selection_represents
    (I64.v prompt_length) (I64.v position) prompt previous_token t)
  returns token:i32
  ensures pure (I32.v token == (FStar.Ghost.reveal t <: int))
{
  selection_refines (I64.v prompt_length) (I64.v position) prompt previous_token t;
  let raw_carried = input2.(0sz);
  let carried = narrow_i32 raw_carried;
  if (I64.lt position prompt_length) {
    let index = SZ.uint32_to_sizet (Cast.int64_to_uint32 position);
    let raw_prompt = input1.(index);
    narrow_i32 raw_prompt
  } else {
    carried
  }
}

inline_for_extraction noextract
let extend_prefix
  (#a:Type0) (#n:nat)
  (s:lseq a n) (f:(i:nat{i < n} -> GTot a)) (j:nat{j < n})
  : Lemma
      (requires forall (i:nat{i < j}). S.index s i == f i)
      (ensures forall (i:nat{i < j + 1}).
        S.index (S.upd s j (f j)) i == f i)
  = let at_index (i:nat{i < j + 1})
      : Lemma (S.index (S.upd s j (f j)) i == f i)
      = if i = j
        then S.lemma_index_upd1 s j (f j)
        else S.lemma_index_upd2 s j (f j) i
    in FStar.Classical.forall_intro at_index

inline_for_extraction noextract
fn embedding_kernel
  (prompt_length:i64{
    0 < I64.v prompt_length /\ I64.v prompt_length <= position_limit})
  (position:i64{
    0 <= I64.v position /\ I64.v position < position_limit})
  (input0:larray bf16 embedding_elements)
  (input1:larray i64 (I64.v prompt_length))
  (input2:larray i64 1)
  (output0:larray bf16 embedding_width)
  (#weights:erased flat_weights)
  (#prompt:erased (lseq i64 (I64.v prompt_length)))
  (#previous_token:erased (lseq i64 1))
  (#old_hidden:erased (lseq bf16 embedding_width))
  (#t:erased Shared.token_id)
  (#model:erased (Shared.embedding_weights R.real))
  (#fweights #fprompt #fprevious:perm)
  preserves gpu **
    input0 |-> Frac fweights weights **
    input1 |-> Frac fprompt prompt **
    input2 |-> Frac fprevious previous_token
  requires pure (
    selection_represents (I64.v prompt_length) (I64.v position)
      prompt previous_token t /\
    weights_model weights model)
  requires output0 |-> old_hidden
  ensures exists* (hidden:Shared.hidden_vector bf16).
    Pulse.Lib.Array.pts_to output0 #1.0R hidden **
    pure (semantic_post weights t model hidden)
{
  let token = select_row prompt_length position input1 input2
    #prompt #previous_token #t #fprompt #fprevious;
  let row = Cast.int32_to_int64 token;
  let offset = I64.mul row 1024L;
  let mut column = 0l;
  while (I32.lt !column 1024l)
    invariant exists* (j:i32{0 <= I32.v j /\ I32.v j <= embedding_width})
      (hidden:lseq bf16 embedding_width).
      column |-> j ** output0 |-> hidden **
      pure (forall (k:nat{k < I32.v j}).
        S.index hidden k ==
          S.index weights (I32.v token * embedding_width + k))
    decreases (embedding_width - I32.v !column)
  {
    let j = !column;
    let address = I64.add offset (Cast.int32_to_int64 j);
    let index = SZ.uint32_to_sizet (Cast.int64_to_uint32 address);
    let value = input0.(index);
    let out_index = SZ.uint32_to_sizet (Cast.int32_to_uint32 j);
    with hidden. assert output0 |-> hidden;
    extend_prefix hidden
      (fun k -> S.index weights (I32.v token * embedding_width + k)) (I32.v j);
    output0.(out_index) <- value;
    column := I32.add j 1l;
  };
  with hidden. assert output0 |-> hidden;
  lookup_transports_model weights t model hidden;
}

inline_for_extraction noextract
instance send_global_array_contents
  (#a:Type0)
  (arr:array a{is_global_array arr})
  (#f:perm)
  (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (prompt_length:i64{
    0 < I64.v prompt_length /\ I64.v prompt_length <= position_limit})
  (position:i64{
    0 <= I64.v position /\ I64.v position < position_limit})
  (input0:larray bf16 embedding_elements{is_global_array input0})
  (input1:larray i64 (I64.v prompt_length){is_global_array input1})
  (input2:larray i64 1{is_global_array input2})
  (output0:larray bf16 embedding_width{is_global_array output0})
  (#weights:erased flat_weights)
  (#prompt:erased (lseq i64 (I64.v prompt_length)))
  (#previous_token:erased (lseq i64 1))
  (#old_hidden:erased (lseq bf16 embedding_width))
  (#t:erased Shared.token_id)
  (#model:erased (Shared.embedding_weights R.real))
  (#fweights #fprompt #fprevious:perm)
  preserves cpu
  preserves on gpu_loc (
    input0 |-> Frac fweights weights **
    input1 |-> Frac fprompt prompt **
    input2 |-> Frac fprevious previous_token)
  requires pure (
    selection_represents (I64.v prompt_length) (I64.v position)
      prompt previous_token t /\
    weights_model weights model)
  requires on gpu_loc (output0 |-> old_hidden)
  ensures exists* (hidden:Shared.hidden_vector bf16).
    on gpu_loc (Pulse.Lib.Array.pts_to output0 #1.0R hidden) **
    pure (semantic_post weights t model hidden)
{
  launch_kernel_1 (fun () ->
    embedding_kernel prompt_length position input0 input1 input2 output0
      #weights #prompt #previous_token #old_hidden #t #model
      #fweights #fprompt #fprevious);
}
