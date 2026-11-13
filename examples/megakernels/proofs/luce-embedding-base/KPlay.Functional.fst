module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Float.Casts

module S = FStar.Seq.Base
module I64 = FStar.Int64
module U32 = FStar.UInt32
module U64 = FStar.UInt64
module IC = FStar.Int.Cast
module SZ = Kuiper.SizeT
module PA = Pulse.Lib.Array
module R = FStar.Real
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let frequency_value (i:u32{U32.v i < rotary_half}) : f32 =
  match i with
  | 0ul -> of_literal "1.0"
  | 1ul -> of_literal "0.604296386241912841796875"
  | 2ul -> of_literal "0.3651741445064544677734375"
  | 3ul -> of_literal "0.22067339718341827392578125"
  | 4ul -> of_literal "0.13335214555263519287109375"
  | 5ul -> of_literal "0.08058421313762664794921875"
  | 6ul -> of_literal "0.0486967526376247406005859375"
  | 7ul -> of_literal "0.02942727319896221160888671875"
  | 8ul -> of_literal "0.01778279431164264678955078125"
  | 9ul -> of_literal "0.01074607856571674346923828125"
  | 10ul -> of_literal "0.0064938166178762912750244140625"
  | 11ul -> of_literal "0.0039241896010935306549072265625"
  | 12ul -> of_literal "0.00237137381918728351593017578125"
  | 13ul -> of_literal "0.001433012657798826694488525390625"
  | 14ul -> of_literal "0.000865964335389435291290283203125"
  | 15ul -> of_literal "0.000523299095220863819122314453125"
  | 16ul -> of_literal "0.000316227786242961883544921875"
  | 17ul -> of_literal "0.000191095299669541418552398681640625"
  | 18ul -> of_literal "0.000115478193038143217563629150390625"
  | 19ul -> of_literal "0.000069783054641447961330413818359375"
  | 20ul -> of_literal "0.0000421696531702764332294464111328125"
  | 21ul -> of_literal "0.000025482966520939953625202178955078125"
  | 22ul -> of_literal "0.000015399264157167635858058929443359375"
  | 23ul -> of_literal "0.00000930572059587575495243072509765625"
  | 24ul -> of_literal "0.00000562341347176698036491870880126953125"
  | 25ul -> of_literal "0.00000339820826411596499383449554443359375"
  | 26ul -> of_literal "0.00000205352489501819945871829986572265625"
  | 27ul -> of_literal "0.00000124093776321387849748134613037109375"
  | 28ul -> of_literal "0.00000074989424092564149759709835052490234375"
  | 29ul -> of_literal "0.000000453158378377338522113859653472900390625"
  | 30ul -> of_literal "0.00000027384197665014653466641902923583984375"
  | _ -> of_literal "0.00000016548170833630138076841831207275390625"

inline_for_extraction noextract
let frequency_refines (i:u32{U32.v i < rotary_half})
  : Lemma (frequency_value i == inverse_frequency (U32.v i))
  = match i with
    | 0ul | 1ul | 2ul | 3ul | 4ul | 5ul | 6ul | 7ul
    | 8ul | 9ul | 10ul | 11ul | 12ul | 13ul | 14ul | 15ul
    | 16ul | 17ul | 18ul | 19ul | 20ul | 21ul | 22ul | 23ul
    | 24ul | 25ul | 26ul | 27ul | 28ul | 29ul | 30ul -> ()
    | _ -> ()

inline_for_extraction noextract
fn frequency_cell (i:u32{U32.v i < rotary_half})
  preserves gpu
  returns result:f32
  ensures pure (result == inverse_frequency (U32.v i))
{
  frequency_refines i;
  frequency_value i
}

inline_for_extraction noextract
fn rotary_cell (p:position) (d:u32{U32.v d < rotary_cells})
  preserves gpu
  returns result:bf16
  ensures pure (result == rotary_component p (U32.v d))
{
  if U32.lt d 32ul {
    let frequency = frequency_cell d;
    fcast (cos (mul frequency (of_int p)))
  } else {
    let i = U32.sub d 32ul;
    let frequency = frequency_cell i;
    fcast (sin (mul frequency (of_int p)))
  }
}

inline_for_extraction noextract
let bounded_index (x:i64{0 <= I64.v x /\ I64.v x < vocab_size})
  : y:sz{SZ.v y == I64.v x}
  = SZ.uint32_to_sizet (IC.int64_to_uint32 x)

inline_for_extraction noextract
let extend_prefix
  (#a:Type0) (#n:nat)
  (s:lseq a n)
  (f:(i:nat{i < n} -> GTot a))
  (j:nat{j < n})
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
fn fill_constant
  (#a:Type0)
  (n:sz)
  (#before:erased (lseq a (SZ.v n)))
  (output:larray a (SZ.v n))
  (value:a)
  preserves gpu
  requires PA.pts_to output #1.0R before
  ensures exists* (after:lseq a (SZ.v n)).
    PA.pts_to output #1.0R after **
    pure (forall (i:nat{i < SZ.v n}). S.index after i == value)
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{SZ.v j <= SZ.v n}) (values:lseq a (SZ.v n)).
      i |-> j ** PA.pts_to output #1.0R values **
      pure (forall (k:nat{k < SZ.v j}). S.index values k == value)
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    with values. assert PA.pts_to output #1.0R values;
    extend_prefix values (fun _ -> value) (SZ.v j);
    output.(j) <- value;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn select_token
  (prompt_length:u64{0 < U64.v prompt_length /\ U64.v prompt_length <= position_limit})
  (p:position)
  (#q:erased prompt_values{S.length q == U64.v prompt_length})
  (#previous_value:erased i64)
  (#t:erased Shared.token_id)
  (prompt:larray i64 (U64.v prompt_length))
  (previous:larray i64 1)
  preserves gpu **
    PA.pts_to prompt #0.5R q **
    previous_permission p previous previous_value
  requires pure (logical_token_value q previous_value p == (FStar.Ghost.reveal t <: int))
  returns token:i64
  ensures pure (I64.v token == (FStar.Ghost.reveal t <: int) /\ token_value_valid token)
{
  let index = bounded_index p;
  let length = FStar.SizeT.uint64_to_sizet prompt_length;
  if (index <^ length) {
    prompt.(index)
  } else {
    assert pure (I64.v p > 0);
    rewrite (previous_permission p previous previous_value)
      as (PA.pts_to previous #0.5R (S.create 1 (FStar.Ghost.reveal previous_value)));
    let token = previous.(0sz);
    rewrite (PA.pts_to previous #0.5R (S.create 1 (FStar.Ghost.reveal previous_value)))
      as (previous_permission p previous previous_value);
    token
  }
}

inline_for_extraction noextract
fn copy_hidden
  (token:i64{token_value_valid token})
  (#w:erased flat_weights)
  (#before:erased (lseq bf16 hidden_size))
  (weights:larray bf16 weight_cells)
  (hidden:larray bf16 hidden_size)
  preserves gpu ** PA.pts_to weights #0.5R w
  requires PA.pts_to hidden #1.0R before
  ensures exists* (h:Shared.hidden_vector bf16).
    PA.pts_to hidden #1.0R h **
    pure (flat_lookup_post w (I64.v token) h)
{
  let row = bounded_index token;
  let offset = row *^ 1024sz;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (j:sz{SZ.v j <= hidden_size}) (values:lseq bf16 hidden_size).
      i |-> j ** PA.pts_to hidden #1.0R values **
      pure (forall (d:nat{d < SZ.v j}).
        S.index values d == S.index w (I64.v token * hidden_size + d))
    decreases (hidden_size - SZ.v !i)
  {
    let j = !i;
    let value = weights.(offset +^ j);
    with values. assert PA.pts_to hidden #1.0R values;
    extend_prefix values
      (fun d -> S.index w (I64.v token * hidden_size + d)) (SZ.v j);
    hidden.(j) <- value;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn write_rotary
  (p:position)
  (#before:erased (lseq bf16 rotary_cells))
  (rotary:larray bf16 rotary_cells)
  preserves gpu
  requires PA.pts_to rotary #1.0R before
  ensures exists* (r:lseq bf16 rotary_cells).
    PA.pts_to rotary #1.0R r ** pure (rotary_post p r)
{
  let mut i = 0sz;
  while (!i <^ 64sz)
    invariant exists* (j:sz{SZ.v j <= rotary_cells}) (values:lseq bf16 rotary_cells).
      i |-> j ** PA.pts_to rotary #1.0R values **
      pure (forall (d:nat{d < SZ.v j}).
        S.index values d == rotary_component p d)
    decreases (rotary_cells - SZ.v !i)
  {
    let j = !i;
    let coordinate = SZ.sizet_to_u32 j;
    let value = rotary_cell p coordinate;
    with values. assert PA.pts_to rotary #1.0R values;
    extend_prefix values (rotary_component p) (SZ.v j);
    rotary.(j) <- value;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn initialize_states
  (p:position)
  (#s0:erased (lseq f32 (initial_recurrent_cells p)))
  (#c0:erased (lseq bf16 (initial_convolution_cells p)))
  (initial_state:larray f32 (initial_recurrent_cells p))
  (initial_conv:larray bf16 (initial_convolution_cells p))
  preserves gpu
  requires
    PA.pts_to initial_state #1.0R s0 **
    PA.pts_to initial_conv #1.0R c0
  ensures exists*
    (s:lseq f32 (initial_recurrent_cells p))
    (c:lseq bf16 (initial_convolution_cells p)).
      PA.pts_to initial_state #1.0R s **
      PA.pts_to initial_conv #1.0R c **
      pure (recurrent_post p s /\ convolution_post p c)
{
  if I64.eq p 0L {
    fill_constant (SZ.uint32_to_sizet 4718592ul) initial_state (zero #f32);
    fill_constant (SZ.uint32_to_sizet 442368ul) initial_conv (zero #bf16);
  } else {
    ()
  }
}

inline_for_extraction noextract
fn embedding_kernel
  (prompt_length:u64{0 < U64.v prompt_length /\ U64.v prompt_length <= position_limit})
  (p:position)
  (#w:erased flat_weights)
  (#q:erased prompt_values{S.length q == U64.v prompt_length})
  (#previous_value:erased i64)
  (#t:erased Shared.token_id)
  (#model:erased (Shared.embedding_weights R.real))
  (#h0:erased (lseq bf16 hidden_size))
  (#r0:erased (lseq bf16 rotary_cells))
  (#s0:erased (lseq f32 (initial_recurrent_cells p)))
  (#c0:erased (lseq bf16 (initial_convolution_cells p)))
  (weights:larray bf16 weight_cells)
  (prompt:larray i64 (U64.v prompt_length))
  (previous:larray i64 1)
  (hidden:larray bf16 hidden_size)
  (rotary:larray bf16 rotary_cells)
  (initial_state:larray f32 (initial_recurrent_cells p))
  (initial_conv:larray bf16 (initial_convolution_cells p))
  preserves gpu **
    PA.pts_to weights #0.5R w **
    PA.pts_to prompt #0.5R q **
    previous_permission p previous previous_value
  requires pure (
    logical_token_value q previous_value p == (FStar.Ghost.reveal t <: int) /\
    weights_model w model)
  requires
    PA.pts_to hidden #1.0R h0 **
    PA.pts_to rotary #1.0R r0 **
    PA.pts_to initial_state #1.0R s0 **
    PA.pts_to initial_conv #1.0R c0
  ensures exists*
    (h:Shared.hidden_vector bf16)
    (r:lseq bf16 rotary_cells)
    (s:lseq f32 (initial_recurrent_cells p))
    (c:lseq bf16 (initial_convolution_cells p)).
      PA.pts_to hidden #1.0R h **
      PA.pts_to rotary #1.0R r **
      PA.pts_to initial_state #1.0R s **
      PA.pts_to initial_conv #1.0R c **
      pure (semantic_post w t model p h r s c)
{
  initialize_states p initial_state initial_conv;
  let token = select_token prompt_length p #q #previous_value #t prompt previous;
  copy_hidden token weights hidden;
  write_rotary p rotary;
  with h r s c.
    assert PA.pts_to hidden #1.0R h **
      PA.pts_to rotary #1.0R r **
      PA.pts_to initial_state #1.0R s **
      PA.pts_to initial_conv #1.0R c;
  lookup_transports_model w t model h;
}

instance send_global_array_contents
  (#a:Type0)
  (arr:array a{is_global_array arr})
  (#f:perm)
  (contents:seq a)
  : is_send_across gpu_of (PA.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

instance send_previous_permission
  (p:position)
  (ptr:array i64{is_global_array ptr})
  (v:i64)
  : is_send_across gpu_of (previous_permission p ptr v)
  = if I64.v p = 0 then solve
    else send_global_array_contents ptr #0.5R (S.create 1 v)

fn kplay_run
  (prompt_length:u64{0 < U64.v prompt_length /\ U64.v prompt_length <= position_limit})
  (p:position)
  (#w:erased flat_weights)
  (#q:erased prompt_values{S.length q == U64.v prompt_length})
  (#previous_value:erased i64)
  (#t:erased Shared.token_id)
  (#model:erased (Shared.embedding_weights R.real))
  (weights:larray bf16 weight_cells{is_global_array weights})
  (prompt:larray i64 (U64.v prompt_length){is_global_array prompt})
  (previous:larray i64 1{is_global_array previous})
  (hidden:larray bf16 hidden_size{is_global_array hidden})
  (rotary:larray bf16 rotary_cells{is_global_array rotary})
  (initial_state:larray f32 (initial_recurrent_cells p){is_global_array initial_state})
  (initial_conv:larray bf16 (initial_convolution_cells p){is_global_array initial_conv})
  preserves cpu **
    on gpu_loc (
      PA.pts_to weights #0.5R (FStar.Ghost.reveal w) **
      PA.pts_to prompt #0.5R (FStar.Ghost.reveal q) **
      previous_permission p previous previous_value)
  requires pure (
    logical_token_value q previous_value p == (FStar.Ghost.reveal t <: int) /\
    weights_model w model)
  requires exists*
    (h0:lseq bf16 hidden_size)
    (r0:lseq bf16 rotary_cells)
    (s0:lseq f32 (initial_recurrent_cells p))
    (c0:lseq bf16 (initial_convolution_cells p)).
      on gpu_loc (
        PA.pts_to hidden #1.0R h0 **
        PA.pts_to rotary #1.0R r0 **
        PA.pts_to initial_state #1.0R s0 **
        PA.pts_to initial_conv #1.0R c0)
  ensures exists*
    (h:Shared.hidden_vector bf16)
    (r:lseq bf16 rotary_cells)
    (s:lseq f32 (initial_recurrent_cells p))
    (c:lseq bf16 (initial_convolution_cells p)).
      on gpu_loc (
        PA.pts_to hidden #1.0R h **
        PA.pts_to rotary #1.0R r **
        PA.pts_to initial_state #1.0R s **
        PA.pts_to initial_conv #1.0R c) **
      pure (semantic_post w t model p h r s c)
{
  with h0 r0 s0 c0.
    assert on gpu_loc (
      PA.pts_to hidden #1.0R h0 **
      PA.pts_to rotary #1.0R r0 **
      PA.pts_to initial_state #1.0R s0 **
      PA.pts_to initial_conv #1.0R c0);
  launch_kernel_1 (fun () ->
    embedding_kernel prompt_length p #w #q #previous_value #t #model
      #h0 #r0 #s0 #c0 weights prompt previous hidden rotary initial_state initial_conv);
}
