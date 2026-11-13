module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Shape
open FStar.Real
open Kuiper.Approximates.Base
module F = Kuiper.Floating.Base
module F32 = Kuiper.Float32
module NativeNeg = Kuiper.Float32.NativeNeg
module NativeNegKind = Kuiper.Float32.NativeNeg.Kind
module Pow = FStar.Math.Pow
module Shared = KPlay.SharedSpec
module S = FStar.Seq.Base
module Chest = Kuiper.Chest
module BF16 = Kuiper.BFloat16
module MMA = Kuiper.TensorCore.MMA
module Gather = Kuiper.TensorCore.MMA.Gather
module RT = Kuiper.Tensor.Runtime
module RuntimeHelpers = Kuiper.Tensor.Runtime.Helpers
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module PC = Pulse.Lib.Core
module KL = Kuiper.Locs
module I64 = FStar.Int64
module List = FStar.List.Tot
module ListProps = FStar.List.Tot.Properties
open Pulse.Lib.Send
module KA = Kuiper.Array.Core
module PA = Pulse.Lib.Array
module RH = Kuiper.Tensor.Runtime.Helpers
module Ghost = FStar.Ghost
module C = FStar.Classical
module KS = Kuiper.Kernel
module SH = Kuiper.SHMem
module Barrier = Kuiper.Barrier
module Observed = Pulse.Lib.MonotonicGhostRef
module E = Kuiper.Epoch
module ST = Kuiper.Kernel.Stream
module P = Pulse.Lib.Pledge
module Tensor = Kuiper.Tensor
module Strided = Kuiper.Array2.Strided
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as source_rm }
module ArrayExtra = Kuiper.Array.Extra
module Exp = FStar.Math.Exp
module Root = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry
module Sum = Kuiper.Sum
module Casts = Kuiper.Float.Casts
module Seq = FStar.Seq
module U64 = FStar.UInt64
module ScoreChest = Kuiper.Chest
module ScoreMMA = Kuiper.TensorCore.MMA
module ScoreF = Kuiper.Floating.Base
module ObExp = FStar.Math.Exp
module ObPow = FStar.Math.Pow
module ObSum = Kuiper.Sum
module ObF = Kuiper.Floating.Base
module ObF32 = Kuiper.Float32
module ObCasts = Kuiper.Float.Casts
module ObI64 = FStar.Int64
module ObTrig = Kuiper.Real.Trigonometry
module ObSeq = FStar.Seq
module ObSZ = Kuiper.SizeT
module ObChest = Kuiper.Chest
module ObMMA = Kuiper.TensorCore.MMA
module ObGemm = Kuiper.Spec.GEMM
module ObRoot = FStar.Math.Sqrt
module Execution = Pulse.Lib.MonotonicGhostRef
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }
open Pulse.Lib.Trade
module IC = FStar.Int.Cast
module EmbS = FStar.Seq
module EmbI32 = FStar.Int32
module EmbI64 = FStar.Int64
module EmbSZ = Kuiper.SizeT
module EmbFC = Kuiper.Float.Casts
module EmbIC = FStar.Int.Cast
module EmbC = FStar.Classical
open Pulse.Lib.Array
module Permission = PulseCore.FractionalPermission
module SP = FStar.Seq.Properties
module HostSize = FStar.SizeT
module Registers = Kuiper.TensorCore.MMA.Registers
module ZeroA = Kuiper.TensorCore.MMA.ZeroA
module Floating = Kuiper.Floating
module Cast = Kuiper.Float.Casts.Base
module SI = FStar.Int
module UI = FStar.UInt
open Kuiper
open Pulse.Lib.Array
#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"

inline_for_extraction noextract
fn source_attention_rescale (maximum scaled:F32.t)
  preserves gpu
  returns result:F32.t
  ensures pure (source_rescale_contract maximum scaled result)
{
  if F.eq maximum (source_negative_infinity ()) {
    F.one
  } else {
    source_attention_literal_bounds ();
    NativeNegKind.neg_kind scaled;
    let negative_scaled = NativeNeg.neg scaled;
    let argument = F32.fma_rn_ftz (source_negative_infinity ())
      (source_attention_literal ()) negative_scaled;
    let result = F32.exp2_approx_ftz argument;
    let zero_related = a0 #F32.t;
    result
  }
}

inline_for_extraction noextract
fn source_attention_score_steps
  (input:larray BF16.t 32768) (head:szlt 8) (fr:MMA.fragment)
  (#values:erased (lseq BF16.t 32768))
  (#fraction:perm) (#before:Chest.chest2 F32.t 16 8)
  preserves gpu ** input |-> Frac fraction values
  requires fr |-> before
  ensures fr |-> source_score_prefix values head 8
{
  MMA.fill fr F.zero;
  let mut step = 0sz;
  while (!step <^ 8sz)
    invariant exists* (s:sz{s <= 8}).
      step |-> s ** fr |-> source_score_prefix values head s
    decreases (8 - !step)
  {
    let s = !step;
    Gather.score input head s fr;
    source_gather_score_operands values head s;
    assert pure (source_score_prefix values head (s + 1) ==
      MMA.emma (source_score_prefix values head s)
        (source_query_operand values head s) (source_key_operand values head s));
    step := s +^ 1sz;
  };
}

inline_for_extraction noextract
fn source_attention_value_steps
  (input:larray BF16.t 32768) (probability:larray BF16.t 1024)
  (head:szlt 8) (block:szlt 16) (fr:MMA.fragment)
  (#values:erased (lseq BF16.t 32768))
  (#probabilities:erased (lseq BF16.t 1024))
  (#fi #fp:perm) (#before:Chest.chest2 F32.t 16 8)
  preserves gpu ** input |-> Frac fi values ** probability |-> Frac fp probabilities
  requires fr |-> before
  ensures fr |-> source_value_prefix values probabilities head block 4
{
  MMA.fill fr F.zero;
  let mut step = 0sz;
  while (!step <^ 4sz)
    invariant exists* (s:sz{s <= 4}).
      step |-> s ** fr |-> source_value_prefix values probabilities head block s
    decreases (4 - !step)
  {
    let s = !step;
    Gather.value input probability head block s fr;
    source_gather_value_operand values head block s;
    assert pure (source_value_prefix values probabilities head block (s + 1) ==
      MMA.emma (source_value_prefix values probabilities head block s)
        (Gather.value_a probabilities s) (source_value_operand values head block s));
    step := s +^ 1sz;
  };
}

inline_for_extraction noextract
fn source_attention_score_tile
  (input:larray BF16.t 32768) (head:szlt 8)
  (tile:larray F32.t 128) (tid:szlt 32)
  (#values:erased (lseq BF16.t 32768))
  (#old:erased (lseq F32.t 128)) (#fraction:perm)
  preserves gpu ** thread_id 32 tid ** input |-> Frac fraction values
  requires tile |-> Frac (1.0R /. 32.0R) old
  ensures tile |-> Frac (1.0R /. 32.0R)
    (source_tile_values (source_score_prefix values head 8))
{
  Tensor.tensor_abs' (source_rm 16 8) tile;
  let matrix = Tensor.from_array (source_rm 16 8) tile;
  assert rewrites_to matrix (Tensor.from_array (source_rm 16 8) tile);
  let fr = MMA.alloc_fragment ();
  source_attention_score_steps input head fr;
  MMA.store fr matrix;
  drop_ (fr |-> source_score_prefix values head 8);
  Tensor.tensor_concr matrix;
  rewrite each Tensor.core matrix as tile;
  source_tile_sequence_binding (source_score_prefix values head 8);
  assert pure (Tensor.to_seq (source_rm 16 8) (source_score_prefix values head 8) ==
    source_tile_values (source_score_prefix values head 8));
  rewrite tile |-> Frac (1.0R /. 32.0R)
      (Tensor.to_seq (source_rm 16 8) (source_score_prefix values head 8)) as
    tile |-> Frac (1.0R /. 32.0R) (source_tile_values (source_score_prefix values head 8));
  ()
}

inline_for_extraction noextract
fn source_attention_value_tile
  (input:larray BF16.t 32768) (probability:larray BF16.t 1024)
  (head:szlt 8) (block:szlt 16) (tile:larray F32.t 128) (tid:szlt 32)
  (#values:erased (lseq BF16.t 32768))
  (#probabilities:erased (lseq BF16.t 1024))
  (#old:erased (lseq F32.t 128)) (#fi #fp:perm)
  preserves gpu ** thread_id 32 tid **
    input |-> Frac fi values ** probability |-> Frac fp probabilities
  requires tile |-> Frac (1.0R /. 32.0R) old
  ensures tile |-> Frac (1.0R /. 32.0R)
    (source_tile_values (source_value_prefix values probabilities head block 4))
{
  Tensor.tensor_abs' (source_rm 16 8) tile;
  let matrix = Tensor.from_array (source_rm 16 8) tile;
  assert rewrites_to matrix (Tensor.from_array (source_rm 16 8) tile);
  let fr = MMA.alloc_fragment ();
  source_attention_value_steps input probability head block fr;
  MMA.store fr matrix;
  drop_ (fr |-> source_value_prefix values probabilities head block 4);
  Tensor.tensor_concr matrix;
  rewrite each Tensor.core matrix as tile;
  source_tile_sequence_binding (source_value_prefix values probabilities head block 4);
  assert pure (Tensor.to_seq (source_rm 16 8)
      (source_value_prefix values probabilities head block 4) ==
    source_tile_values (source_value_prefix values probabilities head block 4));
  rewrite tile |-> Frac (1.0R /. 32.0R)
      (Tensor.to_seq (source_rm 16 8) (source_value_prefix values probabilities head block 4)) as
    tile |-> Frac (1.0R /. 32.0R)
      (source_tile_values (source_value_prefix values probabilities head block 4));
  ()
}

inline_for_extraction noextract
ghost
fn opt_empty_allocation_storage (token:RT.tensor I64.t) (frame:PC.slprop) ()
  requires RT.contents token ** frame
  ensures opt_allocated_storage ({ bf16_buffers = []; token_buffer = token }) frame
{
  fold RuntimeHelpers.all_contents #BF16.t [];
  fold opt_allocated_storage ({ bf16_buffers = []; token_buffer = token }) frame;
}

inline_for_extraction noextract
ghost
fn opt_cons_allocation_storage (value:RT.tensor BF16.t) (buffers:opt_allocated_buffers)
  (frame:PC.slprop) ()
  requires opt_allocated_storage buffers (RT.contents value ** frame)
  ensures opt_allocated_storage
    ({ bf16_buffers = value::buffers.bf16_buffers; token_buffer = buffers.token_buffer }) frame
{
  unfold opt_allocated_storage buffers (RT.contents value ** frame);
  fold RuntimeHelpers.all_contents (value::buffers.bf16_buffers);
  fold opt_allocated_storage
    ({ bf16_buffers = value::buffers.bf16_buffers; token_buffer = buffers.token_buffer }) frame;
}

inline_for_extraction noextract
fn rec opt_allocate_bf16_shapes
  (shapes:list opt_allocation_shape) (stream:Stream.stream_t)
  (#start:Epoch.epoch_t) (frame:PC.slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream start
  requires Pledge.pledge0 (Epoch.epoch_done stream start) frame
  returns outcome:opt_allocation_outcome
  ensures (
    match outcome with
    | OptAllocationFailed _ -> Pledge.pledge0 (Epoch.epoch_done stream start) frame
    | OptAllocated buffers ->
      PC.pure (opt_shapes_match shapes buffers.bf16_buffers /\
        RT.metadata buffers.token_buffer [1; 8]) **
      opt_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream start) (opt_allocated_storage buffers frame))
  decreases shapes
{
  match shapes {
    [] -> {
      let result = RT.empty RT.Int64 stream #start [1; 8] frame ();
      match result {
        RT.Failed error -> { OptAllocationFailed error }
        RT.Allocated token -> {
          let buffers = { bf16_buffers = []; token_buffer = token };
          fold RuntimeHelpers.all_owners #BF16.t stream [];
          rewrite RuntimeHelpers.all_owners #BF16.t stream [] as
            RuntimeHelpers.all_owners stream buffers.bf16_buffers;
          rewrite RT.owner stream token as RT.owner stream buffers.token_buffer;
          fold opt_allocated_owners stream buffers;
          Pledge.rewrite_pledge (RT.contents token ** frame)
            (opt_allocated_storage buffers frame)
            (opt_empty_allocation_storage token frame);
          OptAllocated buffers
        }
      }
    }
    shape::rest -> {
      let result = RT.empty RT.BFloat16 stream #start shape frame ();
      match result {
        RT.Failed error -> { OptAllocationFailed error }
        RT.Allocated value -> {
          let following = opt_allocate_bf16_shapes rest stream #start
            (RT.contents value ** frame) ();
          match following {
            OptAllocationFailed error -> {
              RT.release stream #start value frame ();
              OptAllocationFailed error
            }
            OptAllocated buffers -> {
              let combined = {
                bf16_buffers = value::buffers.bf16_buffers;
                token_buffer = buffers.token_buffer
              };
              Pledge.rewrite_pledge
                (opt_allocated_storage buffers (RT.contents value ** frame))
                (opt_allocated_storage combined frame)
                (opt_cons_allocation_storage value buffers frame);
              unfold opt_allocated_owners stream buffers;
              fold RuntimeHelpers.all_owners stream (value::buffers.bf16_buffers);
              rewrite RuntimeHelpers.all_owners stream (value::buffers.bf16_buffers) as
                RuntimeHelpers.all_owners stream combined.bf16_buffers;
              rewrite RT.owner stream buffers.token_buffer as RT.owner stream combined.token_buffer;
              fold opt_allocated_owners stream combined;
              OptAllocated combined
            }
          }
        }
      }
    }
  }
}

inline_for_extraction noextract
fn opt_prepare_allocations
  (stream:Stream.stream_t) (#start:Epoch.epoch_t)
  (frame:PC.slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream start
  requires Pledge.pledge0 (Epoch.epoch_done stream start) frame
  returns outcome:opt_allocation_outcome
  ensures (
    match outcome with
    | OptAllocationFailed _ -> Pledge.pledge0 (Epoch.epoch_done stream start) frame
    | OptAllocated buffers ->
      PC.pure (opt_allocation_metadata buffers) **
      opt_allocated_owners stream buffers **
      Pledge.pledge0 (Epoch.epoch_done stream start) (opt_allocated_storage buffers frame))
{
  opt_allocation_count ();
  opt_allocate_bf16_shapes opt_bf16_allocation_shapes stream #start frame ()
}
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
ghost
fn opt_single_block_setup (p:slprop) ()
  norewrite
  requires p
  ensures (forall+ (_:natlt 1). p) ** emp
{
  forevery_singleton_intro' #(natlt 1) (fun _ -> p) 0;
}

inline_for_extraction noextract
ghost
fn opt_single_block_teardown (p:slprop) ()
  norewrite
  requires (forall+ (_:natlt 1). p) ** emp
  ensures p
{
  forevery_singleton_elim' #(natlt 1) (fun _ -> p) 0;
}

inline_for_extraction noextract
ghost
fn opt_scalar_block_setup (pre:slprop) (sh:SH.c_shmems opt_scalar_shared_desc)
  (_:natlt 1) ()
  norewrite
  requires SH.live_c_shmems sh ** pre
  ensures (forall+ (_:natlt 1). pre) ** emp
{
  SH.unfold_c_shmems #opt_scalar_shared_desc sh #1.0R (`%opt_scalar_shared_desc);
  forevery_singleton_intro' #(natlt 1) (fun _ -> pre) 0;
}

inline_for_extraction noextract
ghost
fn opt_scalar_block_teardown (post:slprop) (sh:SH.c_shmems opt_scalar_shared_desc)
  (_:natlt 1) ()
  norewrite
  requires (forall+ (_:natlt 1). post) ** emp
  ensures SH.live_c_shmems sh ** post
{
  forevery_singleton_elim' #(natlt 1) (fun _ -> post) 0;
  SH.fold_c_shmems #opt_scalar_shared_desc sh #1.0R (`%opt_scalar_shared_desc);
}

inline_for_extraction noextract
fn opt_scalar_thread (#pre #post:slprop) (code:KS.kernel_desc_1_1 pre post)
  (sh:SH.c_shmems opt_scalar_shared_desc) (bid tid:szlt 1) ()
  requires gpu ** pure (SH.c_shmems_inv sh) ** pre **
    thread_id 1 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 1) ** Barrier.barrier_state 0
  ensures gpu ** post ** thread_id 1 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 1) ** Barrier.barrier_state 0
{
  let device = code.f;
  device ();
}

inline_for_extraction noextract
let opt_scalar_descriptor (#pre #post:slprop) (code:KS.kernel_desc_1_1 pre post)
  : k:KS.kernel_desc pre post{
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)} =
  let send_pre:is_send_across gpu_of pre = code.full_pre_sendable in
  let send_post:is_send_across gpu_of post = code.full_post_sendable in {
    nblk = 1sz;
    nthr = 1sz;
    shmems_desc = opt_scalar_shared_desc;
    kpre = (fun _ _ _ -> pre);
    kpost = (fun _ _ _ -> post);
    barrier_contract = (fun _ _ -> Barrier.empty_contract 1);
    barrier_count = (fun _ -> 0);
    barrier_ok = (fun _ _ -> Barrier.empty_barrier_transform 1);
    f = opt_scalar_thread code;
    frame = emp;
    block_pre = (fun _ -> pre);
    block_post = (fun _ -> post);
    setup = opt_single_block_setup pre;
    teardown = opt_single_block_teardown post;
    block_frame = (fun _ _ -> emp);
    block_setup = opt_scalar_block_setup pre;
    block_teardown = opt_scalar_block_teardown post;
    block_pre_sendable = (fun _ -> send_pre);
    block_post_sendable = (fun _ -> send_post);
    kpre_sendable = (fun _ _ _ _ -> (solve <: is_send_across block_of pre));
    kpost_sendable = (fun _ _ _ _ -> (solve <: is_send_across block_of post))
  }

inline_for_extraction noextract
fn emb_read_token
  (prompt : larray EmbI64.t 7)
  (row : EmbI32.t{0 <= EmbI32.v row /\ EmbI32.v row < 8})
  (#prompt_perm : perm)
  (#t : erased (lseq EmbI64.t 7){emb_valid_tokens t})
  preserves gpu ** prompt |-> Frac prompt_perm (reveal t)
  returns token : EmbI64.t
  ensures pure (0 <= EmbI64.v token /\ EmbI64.v token < 151936 /\
    EmbI64.v token == emb_padded_token t (EmbI32.v row))
{
  if (EmbI32.lt row 7l) {
    let i = emb_index32 row;
    prompt.(i)
  } else {
    0L
  }
}

inline_for_extraction noextract
fn emb_embed
  (weights : larray bf16 emb_embedding_count)
  (prompt : larray EmbI64.t 7)
  (embedding : larray bf16 emb_hidden_count)
  (#weights_perm #prompt_perm : perm)
  (#w : erased emb_embedding_values)
  (#t : erased (lseq EmbI64.t 7){emb_valid_tokens t})
  (#before : erased emb_hidden_values)
  preserves gpu **
    weights |-> Frac weights_perm (reveal w) **
    prompt |-> Frac prompt_perm (reveal t)
  requires embedding |-> before
  ensures exists* (after : emb_hidden_values). embedding |-> after **
    pure (emb_embedding_post 151936 1024 7 8 w t after)
{
  let mut row = 0l;
  while (EmbI32.lt !row 8l)
    invariant exists* (r : EmbI32.t{0 <= EmbI32.v r /\ EmbI32.v r <= 8})
      (values : emb_hidden_values).
      row |-> r ** embedding |-> values **
      pure (emb_embedding_ready 151936 1024 7 8 w t values (EmbI32.v r) 0)
    decreases (8 - EmbI32.v !row)
  {
    let r = !row;
    let token = emb_read_token prompt r #prompt_perm #t;
    let mut column = 0l;
    while (EmbI32.lt !column 1024l)
      invariant exists* (c : EmbI32.t{0 <= EmbI32.v c /\ EmbI32.v c <= 1024})
        (values : emb_hidden_values).
        column |-> c ** embedding |-> values **
        pure (emb_embedding_ready 151936 1024 7 8 w t values (EmbI32.v r) (EmbI32.v c))
      decreases (1024 - EmbI32.v !column)
    {
      let c = !column;
      let source = EmbI64.add (EmbI64.mul token 1024L) (EmbIC.int32_to_int64 c);
      let destination = EmbI32.add (EmbI32.mul r 1024l) c;
      let source_index = emb_index64 source;
      let destination_index = emb_index32 destination;
      let value = weights.(source_index);
      with old. assert embedding |-> old;
      embedding.(destination_index) <- value;
      emb_embedding_step 151936 1024 7 8 w t old (EmbI32.v r) (EmbI32.v c) value;
      column := EmbI32.add !column 1l;
    };
    row := EmbI32.add !row 1l;
  };
  with after. assert embedding |-> after;
  emb_embedding_complete 151936 1024 7 8 w t after;
  assert pure (emb_embedding_post 151936 1024 7 8 w t after);
}

inline_for_extraction noextract
fn emb_write_rotary_pair
  (frequency : EmbI32.t{0 <= EmbI32.v frequency /\ EmbI32.v frequency < 64})
  (position : EmbI32.t{0 <= EmbI32.v position /\ EmbI32.v position < 4096})
  (cosine sine : bf16)
  (table : larray bf16 emb_table_count)
  (#before : erased emb_table_values)
  preserves gpu
  requires table |-> before **
    pure (emb_rope_ready 4096 64 before (EmbI32.v frequency) (EmbI32.v position) /\
      cosine == emb_rotary_value 64 (EmbI32.v position) 0 (EmbI32.v frequency) /\
      sine == emb_rotary_value 64 (EmbI32.v position) 1 (EmbI32.v frequency))
  ensures exists* (after : emb_table_values). table |-> after **
    pure (emb_rope_ready 4096 64 after (EmbI32.v frequency) (EmbI32.v position + 1))
{
  let a = EmbI32.add (EmbI32.mul position 128l) frequency;
  let b = EmbI32.add (EmbI32.add 524288l (EmbI32.mul position 128l)) frequency;
  let a0 = emb_index32 a;
  let a1 = emb_index32 (EmbI32.add a 64l);
  let b0 = emb_index32 b;
  let b1 = emb_index32 (EmbI32.add b 64l);
  table.(a0) <- cosine;
  pts_to_len table;
  table.(a1) <- cosine;
  pts_to_len table;
  table.(b0) <- sine;
  pts_to_len table;
  table.(b1) <- sine;
  pts_to_len table;
  with after. assert table |-> after;
  assert pure (after == EmbS.upd (EmbS.upd (EmbS.upd (EmbS.upd before
    (EmbI32.v a) cosine) (EmbI32.v a + 64) cosine)
    (EmbI32.v b) sine) (EmbI32.v b + 64) sine);
  emb_rope_step 4096 64 before (EmbI32.v frequency) (EmbI32.v position) cosine sine;
  assert pure (emb_rope_ready 4096 64 after (EmbI32.v frequency) (EmbI32.v position + 1));
}

inline_for_extraction noextract
fn emb_make_rope
  (table : larray bf16 emb_table_count)
  (#before : erased emb_table_values)
  preserves gpu
  requires table |-> before
  ensures exists* (after : emb_table_values). table |-> after **
    pure (emb_rope_post 4096 64 after)
{
  let denominator = emb_coordinate 64l;
  let mut frequency = 0l;
  while (EmbI32.lt !frequency 64l)
    invariant exists* (f : EmbI32.t{0 <= EmbI32.v f /\ EmbI32.v f <= 64})
      (values : emb_table_values).
      frequency |-> f ** table |-> values **
      pure (emb_rope_ready 4096 64 values (EmbI32.v f) 0)
    decreases (64 - EmbI32.v !frequency)
  {
    let f = !frequency;
    let inverse = div (one #f32)
      (pow (of_int #f32 1000000L) (div (emb_coordinate f) denominator));
    assert pure (inverse == emb_inverse_frequency 64 (EmbI32.v f));
    let mut position = 0l;
    while (EmbI32.lt !position 4096l)
      invariant exists* (p : EmbI32.t{0 <= EmbI32.v p /\ EmbI32.v p <= 4096})
        (values : emb_table_values).
        position |-> p ** table |-> values **
        pure (emb_rope_ready 4096 64 values (EmbI32.v f) (EmbI32.v p))
      decreases (4096 - EmbI32.v !position)
    {
      let p = !position;
      let phase = mul inverse (emb_coordinate p);
      let cosine = EmbFC.fcast #f32 #bf16 (cos phase);
      let sine = EmbFC.fcast #f32 #bf16 (sin phase);
      Kuiper.Math.div_mod_of_mul_add 64 0 (EmbI32.v f);
      assert pure (cosine == emb_rotary_value 64 (EmbI32.v p) 0 (EmbI32.v f) /\
        sine == emb_rotary_value 64 (EmbI32.v p) 1 (EmbI32.v f));
      emb_write_rotary_pair f p cosine sine table;
      position := EmbI32.add !position 1l;
    };
    with current. assert table |-> current;
    emb_rope_next_frequency 4096 64 current (EmbI32.v f);
    assert pure (emb_rope_ready 4096 64 current (EmbI32.v f + 1) 0);
    frequency := EmbI32.add !frequency 1l;
  };
  with after. assert table |-> after;
  emb_rope_complete 4096 64 after;
  assert pure (emb_rope_post 4096 64 after);
}

inline_for_extraction noextract
fn emb_family_kernel
  (weights : larray bf16 emb_embedding_count)
  (prompt : larray EmbI64.t 7)
  (embedding : larray bf16 emb_hidden_count)
  (table : larray bf16 emb_table_count)
  (#weights_perm #prompt_perm : perm)
  (#w : erased emb_embedding_values)
  (#t : erased (lseq EmbI64.t 7){emb_valid_tokens t})
  (#embedding_before : erased emb_hidden_values)
  (#table_before : erased emb_table_values)
  preserves gpu **
    weights |-> Frac weights_perm (reveal w) **
    prompt |-> Frac prompt_perm (reveal t)
  requires embedding |-> embedding_before ** table |-> table_before
  ensures exists*
    (embedding_after : emb_hidden_values)
    (table_after : emb_table_values).
    embedding |-> embedding_after ** table |-> table_after **
    pure (emb_embedding_post 151936 1024 7 8 w t embedding_after /\
      emb_rope_post 4096 64 table_after)
{
  emb_embed weights prompt embedding;
  emb_make_rope table;
}

inline_for_extraction noextract
ghost
fn u_take_weight_window (weights:larray BF16.t opt_weight_count)
  (#contents:erased (lseq BF16.t opt_weight_count))
  (lo:nat) (hi:nat{lo <= hi /\ hi <= opt_weight_count}) (#fraction:perm)
  requires PA.pts_to weights #fraction contents
  ensures PA.pts_to (u_weight_window weights lo hi) #fraction (S.slice contents lo hi) **
    u_weight_window_frame weights contents lo hi fraction
{
  u_weight_partition contents lo hi;
  PA.pts_to_range_intro weights fraction contents;
  PA.pts_to_range_split weights 0 lo opt_weight_count;
  PA.pts_to_range_split weights lo hi opt_weight_count;
  with middle. assert PA.pts_to_range weights lo hi #fraction middle;
  PA.unfold_pts_to_range weights lo hi #fraction middle;
  PA.from_mask (PA.gsub weights lo hi);
  with window_values. assert PA.pts_to (PA.gsub weights lo hi) #fraction window_values;
  assert pure (S.equal window_values (S.slice contents lo hi));
  S.lemma_eq_elim window_values (S.slice contents lo hi);
  fold u_weight_window_frame weights contents lo hi fraction;
}

inline_for_extraction noextract
ghost
fn u_return_weight_window (weights:larray BF16.t opt_weight_count)
  (#contents:erased (lseq BF16.t opt_weight_count))
  (lo:nat) (hi:nat{lo <= hi /\ hi <= opt_weight_count}) (#fraction:perm)
  requires PA.pts_to (u_weight_window weights lo hi) #fraction (S.slice contents lo hi) **
    u_weight_window_frame weights contents lo hi fraction
  ensures PA.pts_to weights #fraction contents
{
  u_weight_partition contents lo hi;
  unfold u_weight_window_frame weights contents lo hi fraction;
  PA.to_mask (PA.gsub weights lo hi);
  PA.fold_pts_to_range weights lo hi #fraction (S.slice contents lo hi);
  PA.pts_to_range_join weights 0 lo hi;
  PA.pts_to_range_join weights 0 hi opt_weight_count;
  PA.pts_to_range_elim weights fraction contents;
}

inline_for_extraction noextract
fn u_embedding_stage_body
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses)
  (embedding_weights:larray BF16.t emb_embedding_count{
    embedding_weights == u_weight_window weights_address 0 emb_embedding_count})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) ()
  preserves gpu
  requires u_stage_resources weights_address prompt_address a weights prompt v 0
  ensures u_pending_stage weights_address prompt_address a weights prompt 1
{
  unfold u_stage_resources weights_address prompt_address a weights prompt v 0;
  unfold opt_block_storage a v;
  u_take_weight_window weights_address #weights 0 emb_embedding_count #1.0R;
  rewrite PA.pts_to (u_weight_window weights_address 0 emb_embedding_count)
      (S.slice weights 0 emb_embedding_count) as
    embedding_weights |-> S.slice weights 0 emb_embedding_count;
  let raw_prompt = FStar.Ghost.hide #(lseq I64.t 7)
    (FStar.Ghost.reveal #opt_prompt_values prompt);
  assert pure (emb_valid_tokens raw_prompt);
  rewrite PA.pts_to prompt_address prompt as PA.pts_to prompt_address raw_prompt;
  emb_family_kernel embedding_weights prompt_address a.p_embedding a.p_rope
    #1.0R #1.0R #(hide (S.slice weights 0 emb_embedding_count)) #raw_prompt
    #(hide (reveal v).v_embedding) #(hide (reveal v).v_rope);
  with hidden. assert a.p_embedding |-> hidden;
  with table. assert a.p_rope |-> table;
  rewrite embedding_weights |-> S.slice weights 0 emb_embedding_count as
    PA.pts_to (u_weight_window weights_address 0 emb_embedding_count) (S.slice weights 0 emb_embedding_count);
  u_return_weight_window weights_address #weights 0 emb_embedding_count #1.0R;
  rewrite PA.pts_to prompt_address raw_prompt as PA.pts_to prompt_address prompt;
  let next = hide {(reveal v) with v_embedding = reveal hidden; v_rope = reveal table};
  assert pure (u_block_stage weights prompt next 1);
  fold opt_block_storage a next;
  fold u_stage_resources weights_address prompt_address a weights prompt next 1;
  fold u_pending_stage weights_address prompt_address a weights prompt 1;
}

inline_for_extraction noextract
fn dec_copy_bf16
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
  dec_source_loop_end finished count;
  with ys. assert output |-> ys;
  Seq.lemma_eq_intro ys x;
  Seq.lemma_eq_elim ys x;
  assert pure (ys == x);
}

inline_for_extraction noextract
fn u_rotary_source_value (value partner cosine sine:BF16.t) (lower:bool)
  preserves gpu
  returns result:BF16.t
  ensures pure (forall (vr pr cr sr:real).
    value %~ vr /\ partner %~ pr /\ cosine %~ cr /\ sine %~ sr ==>
      result %~ (vr *. cr +. (if lower then 0.0R -. pr else pr) *. sr))
{
  let v = Casts.fcast #BF16.t #F32.t value;
  let p = Casts.fcast #BF16.t #F32.t partner;
  let c = Casts.fcast #BF16.t #F32.t cosine;
  let s = Casts.fcast #BF16.t #F32.t sine;
  dec_bf16_to_f32_all value;
  dec_bf16_to_f32_all partner;
  dec_bf16_to_f32_all cosine;
  dec_bf16_to_f32_all sine;
  dec_source_negate_all p;
  let signed = if lower { dec_source_negate p } else { p };
  let shifted = F32.mul_rn_ftz signed s;
  let rotated = F32.fma_rn_ftz v c shifted;
  dec_f32_to_bf16_all rotated;
  Casts.fcast #F32.t #BF16.t rotated
}
#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
#pop-options

inline_for_extraction noextract
fn u_rms_source_scale (reduced:F32.t) (width:szp{dec_rms_width width})
  preserves gpu
  returns scale:F32.t
  ensures pure (forall (r:dec_spec_vector width).
    reduced %~ dec_spec_sum width (fun i -> r i *. r i) ==>
      v_approximates scale (ObRoot.rsqrt (dec_rms_radicand width r)))
{
  let inverse_width = dec_source_inverse_width width;
  dec_source_epsilon_approx ();
  let radicand = F32.fma_rn_ftz reduced inverse_width (dec_source_epsilon ());
  u_rms_radicand_all width;
  assert pure (forall (r:dec_spec_vector width).
    reduced %~ dec_spec_sum width (fun i -> r i *. r i) ==>
      v_approximates radicand (dec_rms_radicand width r));
  F32.rsqrt_approx_ftz radicand
}

inline_for_extraction noextract
fn rec u_reduce_array (#n:erased pos) (buffer:larray F32.t n) (start:sz)
  (count:szp{dec_reduce_width count /\ start + count <= n})
  (#original:erased (lseq F32.t n))
  preserves gpu
  requires buffer |-> original
  ensures exists* (after:lseq F32.t n).
    buffer |-> after ** pure (u_reduce_result original after start count)
  decreases (SZ.v count)
{
  if (1sz <^ count) {
    let half = count /^ 2sz;
    assert pure (0 < half /\ half < count /\ SZ.v count == 2 * half /\ dec_reduce_width half);
    let mut lane = 0sz;
    while (!lane <^ half)
      invariant exists* (j:sz{j <= half}) (xs:lseq F32.t n).
        lane |-> j ** buffer |-> xs ** pure (u_reduce_prefix original xs start count j)
      decreases (half - !lane)
    {
      let j = !lane;
      with before. assert buffer |-> before;
      let left = buffer.(start +^ j);
      let right = buffer.(start +^ j +^ half);
      let added = F32.add_rn_ftz left right;
      u_reduce_step original before start count j added;
      buffer.(start +^ j) <- added;
      lane := j +^ 1sz;
    };
    dec_source_loop_end !lane half;
    with middle. assert buffer |-> middle;
    u_reduce_array buffer start half #middle;
    with after. assert buffer |-> after;
    u_reduce_finish original middle after start count;
    ()
  } else {
    assert pure (SZ.v count == 1);
  };
}

inline_for_extraction noextract
fn u_rms_squares (#n:erased pos) (input:larray BF16.t n)
  (start:sz) (width:szp{dec_rms_width width /\ start + width <= n})
  (lanes:szp{SZ.v lanes == dec_rms_lanes width})
  (outer:szp{SZ.v outer == dec_rms_outer_width width})
  (partial:larray F32.t lanes) (warp_sum:larray F32.t outer)
  (#x:erased (lseq BF16.t n)) (#fraction:perm)
  preserves gpu ** input |-> Frac fraction x
  requires PA.pts_to_uninit partial lanes ** PA.pts_to_uninit warp_sum outer
  returns result:F32.t
  ensures PA.pts_to_uninit partial lanes ** PA.pts_to_uninit warp_sum outer **
    pure (forall (r:dec_spec_vector width).
      dec_bf16_segment_rep x start width r ==>
        result %~ dec_spec_sum width (fun i -> r i *. r i))
{
  let zero_ok = a0 #F32.t;
  let guards = hide (fun r -> dec_bf16_segment_rep x start width r);
  unfold PA.pts_to_uninit partial lanes;
  let mut thread = 0sz;
  while (!thread <^ lanes)
    invariant exists* (t:sz{t <= lanes}) (mask:lseq (option F32.t) lanes).
      thread |-> t ** PA.pts_to_mask partial mask (fun _ -> True) **
      pure (forall (r:dec_spec_vector width).
        source_mask_rep mask t (dec_rms_lane_model width r) ((reveal guards) r))
    decreases (lanes - !thread)
  {
    let t = !thread;
    let mut sum:F32.t = F.zero;
    let mut step = 0sz;
    while (!step <^ width /^ lanes)
      invariant exists* (s:sz{s <= width / lanes}) (acc:F32.t).
        step |-> s ** sum |-> acc **
        pure (forall (r:dec_spec_vector width).
          (reveal guards) r ==> acc %~ u_rms_lane_partial width t s r)
      decreases (width / lanes - !step)
    {
      let s = !step;
      let value = input.(start +^ t +^ s *^ lanes);
      let v = Casts.fcast #BF16.t #F32.t value;
      dec_bf16_to_f32_all value;
      let previous = !sum;
      let next = F32.fma_rn_ftz v v previous;
      u_rms_acc_step x start width t s previous next;
      assert pure (forall (r:dec_spec_vector width).
        (reveal guards) r ==> next %~ u_rms_lane_partial width t (s + 1) r);
      sum := next;
      step := s +^ 1sz;
    };
    dec_source_loop_end !step (width /^ lanes);
    let reduced = !sum;
    assert pure (forall (r:dec_spec_vector width).
      (reveal guards) r ==> reduced %~ dec_rms_lane_model width r t);
    with before. assert PA.pts_to_mask partial before (fun _ -> True);
    PA.mask_write partial t reduced;
    u_mask_step before t (dec_rms_lane_model width) guards reduced;
    thread := t +^ 1sz;
  };
  dec_source_loop_end !thread lanes;
  with pm. assert PA.pts_to_mask partial pm (fun _ -> True);
  assert pure (source_mask_rep pm lanes (dec_rms_lane_model width (fun _ -> 0.0R))
    ((reveal guards) (fun _ -> 0.0R)));
  PA.from_mask partial;
  let mut warp = 0sz;
  unfold PA.pts_to_uninit warp_sum outer;
  while (!warp <^ lanes /^ 32sz)
    invariant exists* (w:sz{w <= lanes / 32}) (ps:lseq F32.t lanes)
      (ws:lseq (option F32.t) outer).
      warp |-> w ** partial |-> ps ** PA.pts_to_mask warp_sum ws (fun _ -> True) **
      pure (forall (r:dec_spec_vector width).
        ((reveal guards) r ==>
          (forall (i:nat{w * 32 <= i /\ i < lanes}).
            S.index ps i %~ dec_rms_lane_model width r i)) /\
        source_mask_rep ws w (dec_rms_warp_model width r) ((reveal guards) r))
    decreases (lanes / 32 - !warp)
  {
    let w = !warp;
    with ps. assert partial |-> ps;
    u_reduce_array partial (w *^ 32sz) 32sz #ps;
    with after. assert partial |-> after;
    u_rms_warp_finish x start width w ps after;
    let reduced = partial.(w *^ 32sz);
    with ws. assert PA.pts_to_mask warp_sum ws (fun _ -> True);
    PA.mask_write warp_sum w reduced;
    u_mask_step ws w (dec_rms_warp_model width) guards reduced;
    warp := w +^ 1sz;
  };
  dec_source_loop_end !warp (lanes /^ 32sz);
  let mut empty = lanes /^ 32sz;
  while (!empty <^ outer)
    invariant exists* (i:sz{lanes / 32 <= i /\ i <= outer})
      (ws:lseq (option F32.t) outer).
      empty |-> i ** PA.pts_to_mask warp_sum ws (fun _ -> True) **
      pure (forall (r:dec_spec_vector width).
        source_mask_rep ws i (dec_rms_warp_model width r) ((reveal guards) r))
    decreases (outer - !empty)
  {
    let i = !empty;
    with ws. assert PA.pts_to_mask warp_sum ws (fun _ -> True);
    PA.mask_write warp_sum i (F.zero #F32.t);
    u_mask_step ws i (dec_rms_warp_model width) guards (F.zero #F32.t);
    empty := i +^ 1sz;
  };
  dec_source_loop_end !empty outer;
  with mask. assert PA.pts_to_mask warp_sum mask (fun _ -> True);
  assert pure (source_mask_rep mask outer (dec_rms_warp_model width (fun _ -> 0.0R))
    ((reveal guards) (fun _ -> 0.0R)));
  PA.from_mask warp_sum;
  PA.pts_to_len warp_sum;
  with raw_ws. assert warp_sum |-> raw_ws;
  let ws = FStar.Ghost.hide #(lseq F32.t outer)
    (FStar.Ghost.reveal #(S.seq F32.t) raw_ws);
  rewrite warp_sum |-> raw_ws as warp_sum |-> ws;
  u_reduce_array warp_sum 0sz outer #ws;
  with after. assert warp_sum |-> after;
  u_rms_total_finish x start width ws after;
  let result = warp_sum.(0sz);
  PA.to_mask partial;
  fold PA.pts_to_uninit partial lanes;
  PA.to_mask warp_sum;
  fold PA.pts_to_uninit warp_sum outer;
  result
}

inline_for_extraction noextract
fn u_rms_row_masked (#n #nw #ny:erased pos)
  (input:larray BF16.t n) (weights:larray BF16.t nw) (output:larray BF16.t ny)
  (start:sz) (width:szp{dec_rms_width width /\ start + width <= n})
  (weight_start:sz{weight_start + width <= nw})
  (output_start:sz{output_start + width <= ny})
  (lanes:szp{SZ.v lanes == dec_rms_lanes width})
  (outer:szp{SZ.v outer == dec_rms_outer_width width})
  (partial:larray F32.t lanes) (warp_sum:larray F32.t outer)
  (#x:erased (lseq BF16.t n)) (#w:erased (lseq BF16.t nw))
  (#old:erased (lseq (option BF16.t) ny)) (#fx #fw:perm)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w
  requires PA.pts_to_mask output old (fun _ -> True) **
    PA.pts_to_uninit partial lanes ** PA.pts_to_uninit warp_sum outer
  returns after:erased (lseq (option BF16.t) ny)
  ensures PA.pts_to_mask output after (fun _ -> True) **
    PA.pts_to_uninit partial lanes ** PA.pts_to_uninit warp_sum outer **
    pure (forall (r gamma:dec_spec_vector width).
      source_bf16_region old after output_start width width (dec_spec_rms_norm gamma r)
        (u_rms_guard x w start weight_start width r gamma))
{
  let reduced = u_rms_squares input start width lanes outer partial warp_sum #x #fx;
  let scale = u_rms_source_scale reduced width;
  let mut col = 0sz;
  while (!col <^ width)
    invariant exists* (i:sz{i <= width}) (mask:lseq (option BF16.t) ny).
      col |-> i ** PA.pts_to_mask output mask (fun _ -> True) **
      pure (forall (r gamma:dec_spec_vector width).
        source_bf16_region old mask output_start width i (dec_spec_rms_norm gamma r)
          (u_rms_guard x w start weight_start width r gamma))
    decreases (width - !col)
  {
    let i = !col;
    let value = input.(start +^ i);
    let weight = weights.(weight_start +^ i);
    let xf = Casts.fcast #BF16.t #F32.t value;
    let gf = Casts.fcast #BF16.t #F32.t weight;
    dec_bf16_to_f32_all value;
    dec_bf16_to_f32_all weight;
    let weighted = F32.mul_rn_ftz scale gf;
    let product = F32.mul_rn_ftz xf weighted;
    assert pure (forall (r gamma:dec_spec_vector width).
      u_rms_guard x w start weight_start width r gamma ==>
        product %~ (r i *. (ObRoot.rsqrt (dec_rms_radicand width r) *. gamma i)));
    u_rms_cast_all width i product;
    let result = Casts.fcast #F32.t #BF16.t product;
    with before. assert PA.pts_to_mask output before (fun _ -> True);
    PA.mask_write output (output_start +^ i) result;
    u_rms_region_step x w old before start weight_start output_start width i result;
    col := i +^ 1sz;
  };
  dec_source_loop_end !col width;
  PA.pts_to_mask_len output;
  with mask. assert PA.pts_to_mask output mask (fun _ -> True);
  mask
}

inline_for_extraction noextract
fn u_normalize_rotate_head
  (input output:larray BF16.t 32768) (weights:larray BF16.t dec_weight_elements)
  (table:larray BF16.t 1048576)
  (normalized:larray BF16.t 128) (partial:larray F32.t 128) (warp_sum:larray F32.t 32)
  (row:szlt 7) (group:szlt 8) (head:szlt 3)
  (#x #old:erased (lseq BF16.t 32768)) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rt:erased (lseq BF16.t 1048576)) (#fx #fw #ft:perm)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w ** table |-> Frac ft rt
  requires output |-> old **
    PA.pts_to_uninit normalized 128 ** PA.pts_to_uninit partial 128 ** PA.pts_to_uninit warp_sum 32 **
    pure (u_rope_prefix x w rt old (row * 24 + group * 3 + head))
  ensures exists* (after:lseq BF16.t 32768).
    output |-> after **
    PA.pts_to_uninit normalized 128 ** PA.pts_to_uninit partial 128 ** PA.pts_to_uninit warp_sum 32 **
    pure (u_rope_prefix x w rt after (row * 24 + group * 3 + head + 1))
{
  let base = row *^ 4096sz +^ group *^ 512sz +^ head *^ 128sz;
  let weight_start = SZ.uint32_to_sizet 6293504ul +^ (if head = 2sz then 128sz else 0sz);
  unfold PA.pts_to_uninit normalized 128;
  with before_raw. assert PA.pts_to_mask normalized before_raw (fun _ -> True);
  PA.pts_to_mask_len normalized;
  let before = FStar.Ghost.hide #(lseq (option BF16.t) 128)
    (FStar.Ghost.reveal #(S.seq (option BF16.t)) before_raw);
  rewrite PA.pts_to_mask normalized before_raw (fun _ -> True) as
    PA.pts_to_mask normalized before (fun _ -> True);
  let mask = u_rms_row_masked input weights normalized base 128sz weight_start 0sz
    128sz 32sz partial warp_sum #x #w #before #fx #fw;
  let zero = hide (fun (_:Shared.index 128) -> 0.0R);
  source_bf16_full_write_initialized before mask (dec_spec_rms_norm zero zero)
    (u_rms_guard x w base weight_start 128 zero zero);
  PA.from_mask normalized;
  PA.pts_to_len normalized;
  with nv_raw. assert normalized |-> nv_raw;
  let nv = FStar.Ghost.hide #(lseq BF16.t 128)
    (FStar.Ghost.reveal #(S.seq BF16.t) nv_raw);
  rewrite normalized |-> nv_raw as normalized |-> nv;
  u_rms_full_transfer x w base weight_start 128 before mask nv;
  let mut lane = 0sz;
  while (!lane <^ 128sz)
    invariant exists* (i:sz{i <= 128}) (ys:lseq BF16.t 32768).
      lane |-> i ** output |-> ys ** normalized |-> nv **
      pure (dec_segment_frame old ys base i /\
        (forall (j:Shared.index 128{j < i}).
          u_rope_cell x w rt row group head j (S.index ys (base + j))))
    decreases (128 - !lane)
  {
    let i = !lane;
    let partner = (i +^ 64sz) %^ 128sz;
    let v = normalized.(i);
    let p = normalized.(partner);
    let c = table.(row *^ 128sz +^ i);
    let s = table.(SZ.uint32_to_sizet 524288ul +^ row *^ 128sz +^ i);
    let value = u_rotary_source_value v p c s (i <^ 64sz);
    u_rope_from_normalized x w rt nv row group head i value;
    with before_output. assert output |-> before_output;
    u_rope_write x w rt old before_output row group head i value;
    output.(base +^ i) <- value;
    lane := i +^ 1sz;
  };
  dec_source_loop_end !lane 128sz;
  with after. assert output |-> after;
  u_rope_head_advance x w rt old after row group head;
  PA.to_mask normalized;
  fold PA.pts_to_uninit normalized 128;
}

inline_for_extraction noextract
fn u_norm_rope_body (input output:larray BF16.t 32768)
  (weights:larray BF16.t dec_weight_elements) (table:larray BF16.t 1048576)
  (#x #old:erased (lseq BF16.t 32768)) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rt:erased (lseq BF16.t 1048576)) (#fx #fw #ft:perm)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w ** table |-> Frac ft rt
  requires output |-> old
  ensures exists* (ys:lseq BF16.t 32768).
    output |-> ys ** pure (u_rope_prefix x w rt ys 168)
{
  let mut partial : PA.array F32.t = [| 128sz |];
  let mut warp_sum : PA.array F32.t = [| 32sz |];
  let mut normalized : PA.array BF16.t = [| 128sz |];
  dec_copy_bf16 input output 32768sz #x #old #fx;
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (ys:lseq BF16.t 32768).
      row |-> r ** output |-> ys ** PA.pts_to_uninit partial 128 **
      PA.pts_to_uninit warp_sum 32 ** PA.pts_to_uninit normalized 128 **
      pure (u_rope_prefix x w rt ys (r * 24))
    decreases (7 - !row)
  {
    let r = !row;
    let mut group = 0sz;
    while (!group <^ 8sz)
      invariant exists* (g:sz{g <= 8}) (ys:lseq BF16.t 32768).
        group |-> g ** output |-> ys ** PA.pts_to_uninit partial 128 **
        PA.pts_to_uninit warp_sum 32 ** PA.pts_to_uninit normalized 128 **
        pure (u_rope_prefix x w rt ys (r * 24 + g * 3))
      decreases (8 - !group)
    {
      let g = !group;
      let mut head = 0sz;
      while (!head <^ 3sz)
        invariant exists* (h:sz{h <= 3}) (ys:lseq BF16.t 32768).
          head |-> h ** output |-> ys ** PA.pts_to_uninit partial 128 **
          PA.pts_to_uninit warp_sum 32 ** PA.pts_to_uninit normalized 128 **
          pure (u_rope_prefix x w rt ys (r * 24 + g * 3 + h))
        decreases (3 - !head)
      {
        let h = !head;
        with ys. assert output |-> ys;
        u_normalize_rotate_head input output weights table normalized partial warp_sum r g h
          #x #ys #w #rt #fx #fw #ft;
        head := h +^ 1sz;
      };
      dec_source_loop_end !head 3sz;
      group := g +^ 1sz;
    };
    dec_source_loop_end !group 8sz;
    row := r +^ 1sz;
  };
  dec_source_loop_end !row 7sz;
  ()
}

inline_for_extraction noextract
fn u_rope_stage_body
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses)
  (layer_weights:larray BF16.t dec_weight_elements{
    layer_weights == u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) ()
  preserves gpu
  requires u_stage_resources weights_address prompt_address a weights prompt v 3
  ensures u_pending_stage weights_address prompt_address a weights prompt 4
{
  unfold u_stage_resources weights_address prompt_address a weights prompt v 3;
  unfold opt_block_storage a v;
  u_take_weight_window weights_address #weights opt_layer0_start (opt_layer0_start + dec_weight_elements) #1.0R;
  rewrite PA.pts_to (u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements))
      (S.slice weights opt_layer0_start (opt_layer0_start + dec_weight_elements)) as
    layer_weights |-> opt_layer0_values weights;
  u_norm_rope_body a.p_qkv a.p_rotated layer_weights a.p_rope
    #(hide (reveal v).v_qkv) #(hide (reveal v).v_rotated) #(hide (opt_layer0_values weights))
    #(hide (reveal v).v_rope) #1.0R #1.0R #1.0R;
  with ys. assert a.p_rotated |-> ys;
  rewrite layer_weights |-> opt_layer0_values weights as
    PA.pts_to (u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements))
      (S.slice weights opt_layer0_start (opt_layer0_start + dec_weight_elements));
  u_return_weight_window weights_address #weights opt_layer0_start (opt_layer0_start + dec_weight_elements) #1.0R;
  let next = hide {(reveal v) with v_rotated = reveal ys};
  assert pure (u_block_stage weights prompt next 4);
  fold opt_block_storage a next;
  fold u_stage_resources weights_address prompt_address a weights prompt next 4;
  fold u_pending_stage weights_address prompt_address a weights prompt 4;
}

inline_for_extraction noextract
fn u_silu_source_value (gate up:BF16.t)
  preserves gpu
  returns result:BF16.t
  ensures pure (forall (gr ur:real).
    gate %~ gr /\ up %~ ur ==> result %~ (dec_spec_silu gr *. ur))
{
  let v = Casts.fcast #BF16.t #F32.t gate;
  dec_bf16_to_f32_all gate;
  dec_source_negate_all v;
  dec_source_silu_coefficient_approx ();
  let argument = F32.mul_rn_ftz (dec_source_negate v) (dec_source_silu_coefficient ());
  let e = F32.exp2_approx_ftz argument;
  u_silu_change_all ();
  assert pure (forall (r:real). gate %~ r ==> e %~ ObExp.exp (0.0R -. r));
  let denominator = F32.add_rn_ftz (F.one #F32.t) e;
  let inverse = F32.rcp_approx_ftz denominator;
  let value = F32.mul_rn_ftz v inverse;
  u_silu_divide_all ();
  assert pure (forall (r:real). gate %~ r ==> value %~ dec_spec_silu r);
  dec_f32_to_bf16_all value;
  let rounded = Casts.fcast #F32.t #BF16.t value;
  dec_bf16_multiply_all rounded up;
  F.mul rounded up
}

inline_for_extraction noextract
fn u_silu_body (input:larray BF16.t 49152) (output:larray BF16.t 24576)
  (#x:erased (lseq BF16.t 49152)) (#old:erased (lseq BF16.t 24576)) (#fraction:perm)
  preserves gpu ** input |-> Frac fraction x
  requires output |-> old
  ensures exists* (ys:lseq BF16.t 24576). output |-> ys ** pure (u_silu_prefix x ys 24576)
{
  let mut row = 0sz;
  while (!row <^ 8sz)
    invariant exists* (r:sz{r <= 8}) (ys:lseq BF16.t 24576).
      row |-> r ** output |-> ys ** pure (u_silu_prefix x ys (r * 3072))
    decreases (8 - !row)
  {
    let r = !row;
    let mut col = 0sz;
    while (!col <^ 3072sz)
      invariant exists* (c:sz{c <= 3072}) (ys:lseq BF16.t 24576).
        col |-> c ** output |-> ys ** pure (u_silu_prefix x ys (r * 3072 + c))
      decreases (3072 - !col)
    {
      let c = !col;
      let i = dec_silu_source_index c;
      let gate = input.(r *^ 6144sz +^ i);
      let up = input.(r *^ 6144sz +^ i +^ 64sz);
      let value = u_silu_source_value gate up;
      with before. assert output |-> before;
      u_silu_step x before r c value;
      output.(r *^ 3072sz +^ c) <- value;
      col := c +^ 1sz;
    };
    dec_source_loop_end !col 3072sz;
    row := r +^ 1sz;
  };
  dec_source_loop_end !row 8sz;
  ()
}

inline_for_extraction noextract
fn u_silu_stage_body
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) ()
  preserves gpu
  requires u_stage_resources weights_address prompt_address a weights prompt v 8
  ensures u_pending_stage weights_address prompt_address a weights prompt 9
{
  unfold u_stage_resources weights_address prompt_address a weights prompt v 8;
  unfold opt_block_storage a v;
  u_silu_body a.p_gateup a.p_activated #(hide (reveal v).v_gateup) #(hide (reveal v).v_activated) #1.0R;
  with ys. assert a.p_activated |-> ys;
  let next = hide {(reveal v) with v_activated = reveal ys};
  assert pure (u_block_stage weights prompt next 9);
  fold opt_block_storage a next;
  fold u_stage_resources weights_address prompt_address a weights prompt next 9;
  fold u_pending_stage weights_address prompt_address a weights prompt 9;
}

inline_for_extraction noextract
ghost
fn opt_descriptor_frame_setup (#pre #post:slprop)
  (code:KS.kernel_desc pre post) (extra:slprop) ()
  norewrite
  requires pre ** extra
  ensures (forall+ (bid:natlt code.nblk). code.block_pre bid) ** (code.frame ** extra)
{
  let setup = code.setup;
  setup ();
}

inline_for_extraction noextract
ghost
fn opt_descriptor_frame_teardown (#pre #post:slprop)
  (code:KS.kernel_desc pre post) (extra:slprop) ()
  norewrite
  requires (forall+ (bid:natlt code.nblk). code.block_post bid) ** (code.frame ** extra)
  ensures post ** extra
{
  let teardown = code.teardown;
  teardown ();
}

inline_for_extraction noextract
let opt_descriptor_with_frame (#pre #post:slprop)
  (code:KS.kernel_desc pre post) (extra:slprop)
  : k:KS.kernel_desc (pre ** extra) (post ** extra){
    k.nblk == code.nblk /\ k.nthr == code.nthr /\ k.shmems_desc == code.shmems_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == code.barrier_contract bid sh /\
      k.barrier_count bid == code.barrier_count bid)} = {
  code with
  frame = code.frame ** extra;
  setup = opt_descriptor_frame_setup code extra;
  teardown = opt_descriptor_frame_teardown code extra
}

inline_for_extraction noextract
ghost
fn opt_descriptor_refine_setup (#pre #post #pre':slprop)
  (code:KS.kernel_desc pre post)
  (before:ghost fn () requires pre' ensures pre) ()
  norewrite
  requires pre'
  ensures (forall+ (bid:natlt code.nblk). code.block_pre bid) ** code.frame
{
  before ();
  let setup = code.setup;
  setup ();
}

inline_for_extraction noextract
ghost
fn opt_descriptor_refine_teardown (#pre #post #post':slprop)
  (code:KS.kernel_desc pre post)
  (after:ghost fn () requires post ensures post') ()
  norewrite
  requires (forall+ (bid:natlt code.nblk). code.block_post bid) ** code.frame
  ensures post'
{
  let teardown = code.teardown;
  teardown ();
  after ();
}

inline_for_extraction noextract
let opt_descriptor_refine (#pre #post #pre' #post':slprop)
  (code:KS.kernel_desc pre post)
  (before:ghost fn () requires pre' ensures pre)
  (after:ghost fn () requires post ensures post')
  : k:KS.kernel_desc pre' post'{
    k.nblk == code.nblk /\ k.nthr == code.nthr /\ k.shmems_desc == code.shmems_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == code.barrier_contract bid sh /\
      k.barrier_count bid == code.barrier_count bid)} = {
  code with
  setup = opt_descriptor_refine_setup code before;
  teardown = opt_descriptor_refine_teardown code after
}

inline_for_extraction noextract
ghost
fn u_linear_storage_open (a:opt_block_addresses) (#v:erased opt_block_values) (slot:nat{slot < 4})
  requires opt_block_storage a v
  ensures u_linear_input a slot |-> u_linear_x v slot **
    u_linear_residual a slot |-> u_linear_r v slot **
    u_linear_output a slot |-> u_linear_old v slot ** u_linear_frame a v slot
{
  unfold opt_block_storage a v;
  if (slot = 0) {
    fold u_linear_frame a v 0;
    rewrite u_linear_frame a v 0 as u_linear_frame a v slot;
    rewrite a.p_normalized0 |-> (reveal v).v_normalized0 as u_linear_input a slot |-> u_linear_x v slot;
    rewrite a.p_embedding |-> (reveal v).v_embedding as u_linear_residual a slot |-> u_linear_r v slot;
    rewrite a.p_qkv |-> (reveal v).v_qkv as u_linear_output a slot |-> u_linear_old v slot;
  } else if (slot = 1) {
    fold u_linear_frame a v 1;
    rewrite u_linear_frame a v 1 as u_linear_frame a v slot;
    rewrite a.p_attention |-> (reveal v).v_attention as u_linear_input a slot |-> u_linear_x v slot;
    rewrite a.p_embedding |-> (reveal v).v_embedding as u_linear_residual a slot |-> u_linear_r v slot;
    rewrite a.p_residual |-> (reveal v).v_residual as u_linear_output a slot |-> u_linear_old v slot;
  } else if (slot = 2) {
    fold u_linear_frame a v 2;
    rewrite u_linear_frame a v 2 as u_linear_frame a v slot;
    rewrite a.p_postnorm |-> (reveal v).v_postnorm as u_linear_input a slot |-> u_linear_x v slot;
    rewrite a.p_residual |-> (reveal v).v_residual as u_linear_residual a slot |-> u_linear_r v slot;
    rewrite a.p_gateup |-> (reveal v).v_gateup as u_linear_output a slot |-> u_linear_old v slot;
  } else {
    assert pure (slot == 3);
    fold u_linear_frame a v 3;
    rewrite u_linear_frame a v 3 as u_linear_frame a v slot;
    rewrite a.p_activated |-> (reveal v).v_activated as u_linear_input a slot |-> u_linear_x v slot;
    rewrite a.p_residual |-> (reveal v).v_residual as u_linear_residual a slot |-> u_linear_r v slot;
    rewrite a.p_hidden |-> (reveal v).v_hidden as u_linear_output a slot |-> u_linear_old v slot;
  };
}

inline_for_extraction noextract
ghost
fn u_linear_before
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses)
  (layer_weights:larray BF16.t dec_weight_elements{
    layer_weights == u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) (slot:nat{slot < 4}) ()
  requires u_stage_resources weights_address prompt_address a weights prompt v (u_linear_step slot)
  ensures opt_linear_pre (u_linear_input a slot) layer_weights (u_linear_residual a slot) (u_linear_output a slot)
      (u_linear_x v slot) (opt_layer0_values weights) (u_linear_r v slot) 1.0R 1.0R 1.0R **
    u_linear_stage_frame weights_address prompt_address a weights prompt v slot
{
  unfold u_stage_resources weights_address prompt_address a weights prompt v (u_linear_step slot);
  u_linear_storage_open a #v slot;
  u_take_weight_window weights_address #weights opt_layer0_start (opt_layer0_start + dec_weight_elements) #1.0R;
  rewrite PA.pts_to (u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements))
      (S.slice weights opt_layer0_start (opt_layer0_start + dec_weight_elements)) as
    layer_weights |-> opt_layer0_values weights;
  PA.to_mask (u_linear_output a slot);
  fold PA.pts_to_uninit (u_linear_output a slot) (8 * u_linear_cols slot);
  fold opt_linear_read (u_linear_input a slot) layer_weights (u_linear_residual a slot)
    (u_linear_x v slot) (opt_layer0_values weights) (u_linear_r v slot) 1.0R 1.0R 1.0R;
  fold opt_linear_pre (u_linear_input a slot) layer_weights (u_linear_residual a slot) (u_linear_output a slot)
    (u_linear_x v slot) (opt_layer0_values weights) (u_linear_r v slot) 1.0R 1.0R 1.0R;
  fold u_linear_stage_frame weights_address prompt_address a weights prompt v slot;
}

inline_for_extraction noextract
ghost
fn u_linear_storage_close (a:opt_block_addresses) (#v:erased opt_block_values)
  (slot:nat{slot < 4}) (#ys:erased (lseq BF16.t (8 * u_linear_cols slot)))
  requires u_linear_input a slot |-> u_linear_x v slot **
    u_linear_residual a slot |-> u_linear_r v slot **
    u_linear_output a slot |-> ys ** u_linear_frame a v slot
  ensures opt_block_storage a (u_linear_updated v slot ys)
{
  if (slot = 0) {
    rewrite each slot as 0;
    unfold u_linear_frame a v 0;
    fold opt_block_storage a (u_linear_updated v 0 ys);
    rewrite opt_block_storage a (u_linear_updated v 0 ys) as opt_block_storage a (u_linear_updated v slot ys);
  } else if (slot = 1) {
    rewrite each slot as 1;
    unfold u_linear_frame a v 1;
    fold opt_block_storage a (u_linear_updated v 1 ys);
    rewrite opt_block_storage a (u_linear_updated v 1 ys) as opt_block_storage a (u_linear_updated v slot ys);
  } else if (slot = 2) {
    rewrite each slot as 2;
    unfold u_linear_frame a v 2;
    fold opt_block_storage a (u_linear_updated v 2 ys);
    rewrite opt_block_storage a (u_linear_updated v 2 ys) as opt_block_storage a (u_linear_updated v slot ys);
  } else {
    assert pure (slot == 3);
    rewrite each slot as 3;
    unfold u_linear_frame a v 3;
    fold opt_block_storage a (u_linear_updated v 3 ys);
    rewrite opt_block_storage a (u_linear_updated v 3 ys) as opt_block_storage a (u_linear_updated v slot ys);
  };
}

inline_for_extraction noextract
ghost
fn u_linear_after
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses)
  (layer_weights:larray BF16.t dec_weight_elements{
    layer_weights == u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) (slot:nat{slot < 4}) ()
  requires opt_linear_post (u_linear_input a slot) layer_weights (u_linear_residual a slot) (u_linear_output a slot)
      (u_linear_x v slot) (opt_layer0_values weights) (u_linear_r v slot) (u_linear_result weights v slot)
      1.0R 1.0R 1.0R **
    u_linear_stage_frame weights_address prompt_address a weights prompt v slot
  ensures u_pending_stage weights_address prompt_address a weights prompt (u_linear_step slot + 1)
{
  unfold opt_linear_post (u_linear_input a slot) layer_weights (u_linear_residual a slot) (u_linear_output a slot)
    (u_linear_x v slot) (opt_layer0_values weights) (u_linear_r v slot) (u_linear_result weights v slot)
    1.0R 1.0R 1.0R;
  unfold opt_linear_read (u_linear_input a slot) layer_weights (u_linear_residual a slot)
    (u_linear_x v slot) (opt_layer0_values weights) (u_linear_r v slot) 1.0R 1.0R 1.0R;
  unfold u_linear_stage_frame weights_address prompt_address a weights prompt v slot;
  rewrite layer_weights |-> opt_layer0_values weights as
    PA.pts_to (u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements))
      (S.slice weights opt_layer0_start (opt_layer0_start + dec_weight_elements));
  u_return_weight_window weights_address #weights opt_layer0_start (opt_layer0_start + dec_weight_elements) #1.0R;
  u_linear_storage_close a #v slot #(hide (u_linear_result weights v slot));
  u_linear_stage_advance weights prompt v slot;
  fold u_stage_resources weights_address prompt_address a weights prompt
    (u_linear_updated v slot (u_linear_result weights v slot)) (u_linear_step slot + 1);
  fold u_pending_stage weights_address prompt_address a weights prompt (u_linear_step slot + 1);
}

inline_for_extraction noextract
ghost
fn source_warp_owner_intro (p:slprop)
  requires p
  ensures forall+ (tid:natlt 32). when_ (tid == 0) p
{
  forevery_singleton_intro' #(tid:natlt 32{tid == 0}) (fun _ -> p) 0;
  forevery_unrefine_pred #(natlt 32) (fun _ -> p) (fun tid -> tid == 0);
}

inline_for_extraction noextract
ghost
fn opt_linear_block_setup
  (#ni #nr #no:erased nat)
  (input:larray BF16.t ni) (weights:dec_bf16_array dec_weight_elements)
  (residual:larray BF16.t nr) (output:larray BF16.t no)
  (#x:erased (lseq BF16.t ni)) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rs:erased (lseq BF16.t nr)) (#fx #fw #fr:perm)
  (sh:SH.c_shmems opt_linear_shared_desc) (_:natlt 1) ()
  norewrite
  requires SH.live_c_shmems sh ** opt_linear_pre input weights residual output x w rs fx fw fr
  ensures (forall+ (tid:natlt 32).
    opt_linear_kpre input weights residual output (opt_linear_tile sh) x w rs fx fw fr tid) ** emp
{
  SH.unfold_c_shmems sh (`%opt_linear_shared_desc);
  unfold opt_linear_pre input weights residual output x w rs fx fw fr;
  unfold opt_linear_read input weights residual x w rs fx fw fr;
  let tile = opt_linear_tile sh;
  rewrite each fst sh as tile;
  with old. assert tile |-> old;
  PA.pts_to_len tile;
  let tv = FStar.Ghost.hide #(lseq F32.t 128) (FStar.Ghost.reveal #(S.seq F32.t) old);
  rewrite tile |-> old as tile |-> tv;
  ArrayExtra.array_share input #x #fx 32;
  ArrayExtra.array_share weights #w #fw 32;
  ArrayExtra.array_share residual #rs #fr 32;
  ArrayExtra.array_share tile #tv #1.0R 32;
  source_warp_owner_intro (PA.pts_to_uninit output no);
  forevery_zip3 #(natlt 32)
    (fun _ -> input |-> Frac (fx /. 32.0R) x)
    (fun _ -> weights |-> Frac (fw /. 32.0R) w)
    (fun _ -> residual |-> Frac (fr /. 32.0R) rs);
  forevery_ext #(natlt 32)
    (fun _ -> input |-> Frac (fx /. 32.0R) x **
      weights |-> Frac (fw /. 32.0R) w ** residual |-> Frac (fr /. 32.0R) rs)
    (fun _ -> opt_linear_read input weights residual x w rs
      (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R));
  forevery_map #(natlt 32)
    (fun _ -> tile |-> Frac (1.0R /. 32.0R) tv)
    (fun _ -> exists* (v:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) v)
    fn _ { () };
  forevery_zip3 #(natlt 32)
    (fun _ -> opt_linear_read input weights residual x w rs
      (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R))
    (fun _ -> exists* (v:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) v)
    (fun i -> when_ (i == 0) (PA.pts_to_uninit output no));
  forevery_map #(natlt 32)
    (fun i -> opt_linear_read input weights residual x w rs
      (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R) **
      (exists* (v:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) v) **
      when_ (i == 0) (PA.pts_to_uninit output no))
    (fun i -> opt_linear_kpre input weights residual output tile x w rs fx fw fr i)
    fn i { fold opt_linear_kpre input weights residual output tile x w rs fx fw fr i; };
  rewrite each tile as opt_linear_tile sh;
}

inline_for_extraction noextract
ghost
fn source_warp_owner_elim (p:slprop)
  requires forall+ (tid:natlt 32). when_ (tid == 0) p
  ensures p
{
  forevery_refine_pred #(natlt 32) (fun _ -> p) (fun tid -> tid == 0);
  forevery_singleton_elim' #(tid:natlt 32{tid == 0}) (fun _ -> p) 0;
}

inline_for_extraction noextract
ghost
fn opt_linear_block_teardown
  (#ni #nr #no:erased nat)
  (input:larray BF16.t ni) (weights:dec_bf16_array dec_weight_elements)
  (residual:larray BF16.t nr) (output:larray BF16.t no)
  (#x:erased (lseq BF16.t ni)) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rs:erased (lseq BF16.t nr)) (#ys:erased (lseq BF16.t no))
  (#tv:erased (lseq F32.t 128)) (#fx #fw #fr:perm)
  (sh:SH.c_shmems opt_linear_shared_desc) (_:natlt 1) ()
  norewrite
  requires (forall+ (tid:natlt 32).
    opt_linear_kpost input weights residual output (opt_linear_tile sh)
      x w rs ys tv fx fw fr tid) ** emp
  ensures SH.live_c_shmems sh ** opt_linear_post input weights residual output x w rs ys fx fw fr
{
  let tile = opt_linear_tile sh;
  rewrite each opt_linear_tile sh as tile;
  forevery_ext #(natlt 32)
    (fun i -> opt_linear_kpost input weights residual output tile x w rs ys tv fx fw fr i)
    (fun i -> opt_linear_read input weights residual x w rs
      (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R) **
      tile |-> Frac (1.0R /. 32.0R) tv ** when_ (i == 0) (output |-> ys));
  forevery_unzip3 #(natlt 32)
    (fun _ -> opt_linear_read input weights residual x w rs
      (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R))
    (fun _ -> tile |-> Frac (1.0R /. 32.0R) tv)
    (fun i -> when_ (i == 0) (output |-> ys));
  source_warp_owner_elim (output |-> ys);
  ArrayExtra.array_gather tile #tv #1.0R 32;
  forevery_ext #(natlt 32)
    (fun _ -> opt_linear_read input weights residual x w rs
      (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R))
    (fun _ -> input |-> Frac (fx /. 32.0R) x **
      weights |-> Frac (fw /. 32.0R) w ** residual |-> Frac (fr /. 32.0R) rs);
  forevery_unzip3 #(natlt 32)
    (fun _ -> input |-> Frac (fx /. 32.0R) x)
    (fun _ -> weights |-> Frac (fw /. 32.0R) w)
    (fun _ -> residual |-> Frac (fr /. 32.0R) rs);
  ArrayExtra.array_gather input #x #fx 32;
  ArrayExtra.array_gather weights #w #fw 32;
  ArrayExtra.array_gather residual #rs #fr 32;
  fold opt_linear_read input weights residual x w rs fx fw fr;
  fold opt_linear_post input weights residual output x w rs ys fx fw fr;
  rewrite each tile as fst sh;
  SH.fold_c_shmems sh (`%opt_linear_shared_desc);
}

inline_for_extraction noextract
fn dec_dense_source_mma
  (width:szp{dec_dense_width width})
  (a:larray bf16 (16 * width)) (b:larray bf16 (8 * width))
  (c:larray F32.t 128)
  (#av:erased (lseq bf16 (16 * width))) (#bv:erased (lseq bf16 (8 * width)))
  (#cv:erased (lseq F32.t 128)) (#fa #fb:perm)
  preserves gpu ** a |-> Frac fa av ** b |-> Frac fb bv
  requires c |-> Frac (1.0R /. 32.0R) cv
  ensures c |-> Frac (1.0R /. 32.0R)
    (Tensor.to_seq (rm 16 8) (dec_dense_mma_fold width av bv (width / 16)))
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
      step |-> k ** fragment |-> dec_dense_mma_fold width av bv k
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
    dec_dense_mma_step width av bv k;
    step := k +^ 1sz;
  };
  let finished = !step;
  dec_source_loop_end finished (width /^ 16sz);
  MMA.store fragment cmatrix;
  drop_ (fragment |-> dec_dense_mma_fold width av bv (width / 16));
  Tensor.tensor_concr am;
  Tensor.tensor_concr bm;
  Tensor.tensor_concr cmatrix;
  rewrite each Tensor.core am as a;
  rewrite each Tensor.core bm as b;
  rewrite each Tensor.core cmatrix as c;
}

inline_for_extraction noextract
fn dec_borrow_source_slice
  (#n:erased nat) (arr:larray bf16 n)
  (start:sz) (count:sz{start + count <= n})
  (#original:erased (lseq bf16 n)) (#f:perm)
  requires arr |-> Frac f original
  returns sub:larray bf16 count
  ensures pure (sub == Pulse.Lib.Array.gsub arr start (start + count)) **
    sub |-> Frac f (Seq.slice original start (start + count)) **
    dec_slice_remainder arr f start count original
{
  Pulse.Lib.Array.to_mask arr;
  let sub = Pulse.Lib.Array.sub arr #f start (hide (start + count));
  Pulse.Lib.Array.from_mask sub;
  with dec_spec_values. assert sub |-> Frac f dec_spec_values;
  assert pure (Seq.equal dec_spec_values (Seq.slice original start (start + count)));
  rewrite each (fun (i:nat) -> ~(SZ.v start <= i /\ i < SZ.v start + SZ.v count))
    as dec_slice_outside start count;
  fold dec_slice_remainder arr f start count original;
  sub
}

inline_for_extraction noextract
ghost
fn dec_restore_source_slice
  (#n:erased nat) (arr:larray bf16 n)
  (start:sz) (count:sz{start + count <= n})
  (sub:larray bf16 count{sub == Pulse.Lib.Array.gsub arr start (start + count)})
  (#original:erased (lseq bf16 n)) (#f:perm)
  requires dec_slice_remainder arr f start count original **
    sub |-> Frac f (Seq.slice original start (start + count))
  ensures arr |-> Frac f original
{
  unfold dec_slice_remainder arr f start count original;
  Pulse.Lib.Array.to_mask sub;
  rewrite each sub as Pulse.Lib.Array.gsub arr start (start + count);
  Pulse.Lib.Array.return_sub arr;
  Pulse.Lib.Array.from_mask arr;
  with result. assert arr |-> Frac f result;
  assert pure (Seq.equal result original);
}

inline_for_extraction noextract
ghost
fn dec_warp_owner_elim (p:slprop)
  requires forall+ (tid:natlt 32). when_ (tid == 0) p
  ensures p
{
  forevery_refine_pred #(natlt 32) (fun _ -> p) (fun tid -> tid == 0);
  forevery_singleton_elim' #(tid:natlt 32{tid == 0}) (fun _ -> p) 0;
}

ghost
fn dec_warp_distribute_proof (#a:Type0) (buffer:array a)
  (#dec_spec_values:erased (seq a)) (#f:perm) ()
  requires forall+ (tid:natlt 32). when_ (tid == 0) (buffer |-> Frac f dec_spec_values)
  ensures forall+ (_:natlt 32). buffer |-> Frac (f /. 32.0R) dec_spec_values
{
  dec_warp_owner_elim (buffer |-> Frac f dec_spec_values);
  Kuiper.Array.Extra.array_share buffer #dec_spec_values #f 32;
}

inline_for_extraction noextract
ghost
fn dec_warp_owner_intro (p:slprop)
  requires p
  ensures forall+ (tid:natlt 32). when_ (tid == 0) p
{
  forevery_singleton_intro' #(tid:natlt 32{tid == 0}) (fun _ -> p) 0;
  forevery_unrefine_pred #(natlt 32) (fun _ -> p) (fun tid -> tid == 0);
}

ghost
fn dec_warp_gather_proof (#a:Type0) (buffer:array a)
  (#dec_spec_values:erased (seq a)) (#f:perm) ()
  requires forall+ (_:natlt 32). buffer |-> Frac (f /. 32.0R) dec_spec_values
  ensures forall+ (tid:natlt 32). when_ (tid == 0) (buffer |-> Frac f dec_spec_values)
{
  Kuiper.Array.Extra.array_gather buffer #dec_spec_values #f 32;
  dec_warp_owner_intro (buffer |-> Frac f dec_spec_values);
}

inline_for_extraction noextract
fn dec_write_projection_tile
  (#nr:erased nat) (cols:szp) (block:sz{block + 16 <= cols})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (tile:larray F32.t 128) (residual:larray bf16 nr) (output:larray bf16 (8 * cols))
  (#tc:erased (lseq F32.t 128)) (#rs:erased (lseq bf16 nr))
  (#old:erased (lseq (option bf16) (8 * cols)))
  (#expected:erased (lseq bf16 (8 * cols))) (#fr:perm)
  preserves gpu ** tile |-> tc ** residual |-> Frac fr rs
  requires Pulse.Lib.Array.pts_to_mask output old (fun _ -> True) **
    pure (dec_native_projection_prefix cols old expected block 0 /\
      (forall (row:dec_spec_index 8) (col:dec_spec_index 16).
        Seq.index expected (row * cols + block + col) ==
          (let rounded = Casts.fcast #F32.t #bf16 (Seq.index tc (col * 8 + row)) in
           if add_residual then ObF.add rounded (Seq.index rs (row * cols + block + col))
           else rounded)))
  ensures exists* (ys:lseq (option bf16) (8 * cols)).
    Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True) **
    pure (dec_native_projection_prefix cols ys expected (block + 16) 0)
{
  let mut row = 0sz;
  while (!row <^ 8sz)
    invariant exists* (r:sz{r <= 8}) (ys:lseq (option bf16) (8 * cols)).
      row |-> r ** Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True) **
      pure (dec_native_projection_prefix cols ys expected block (r * 16))
    decreases (8 - !row)
  {
    let r = !row;
    let mut col = 0sz;
    while (!col <^ 16sz)
      invariant exists* (c:sz{c <= 16}) (ys:lseq (option bf16) (8 * cols)).
        col |-> c ** Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True) **
        pure (dec_native_projection_prefix cols ys expected block (r * 16 + c))
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
      dec_native_projection_write cols before after expected block r c value;
      col := c +^ 1sz;
    };
    let finished = !col;
    dec_source_loop_end finished 16sz;
    row := r +^ 1sz;
  };
  let finished = !row;
  dec_source_loop_end finished 8sz;
  with ys. assert Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True);
  dec_native_projection_advance cols ys expected block;
  ()
}

inline_for_extraction noextract
fn dec_projection_owner_commit
  (#nr:erased nat) (cols:szp) (block:sz{block + 16 <= cols})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (tile:larray F32.t 128) (residual:larray bf16 nr) (output:larray bf16 (8 * cols))
  (tid:szlt 32) (#tc:erased (lseq F32.t 128)) (#rs:erased (lseq bf16 nr))
  (#expected:erased (lseq bf16 (8 * cols))) (#fr:perm)
  preserves gpu ** residual |-> Frac fr rs **
    when_ (SZ.v tid == 0) (tile |-> tc)
  requires when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected block) **
    pure (forall (row:dec_spec_index 8) (col:dec_spec_index 16).
      Seq.index expected (row * cols + block + col) ==
        (let rounded = Casts.fcast #F32.t #bf16 (Seq.index tc (col * 8 + row)) in
         if add_residual then ObF.add rounded (Seq.index rs (row * cols + block + col))
         else rounded))
  ensures when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected (block + 16))
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (tile |-> tc) as tile |-> tc;
    rewrite when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected block)
      as dec_projection_output_owned cols output expected block;
    unfold dec_projection_output_owned cols output expected block;
    with ys. assert Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True);
    dec_write_projection_tile cols block add_residual tile residual output
      #tc #rs #(hide (reveal ys <: lseq (option bf16) (8 * cols))) #expected #fr;
    fold dec_projection_output_owned cols output expected (block + 16);
    rewrite tile |-> tc as when_ (SZ.v tid == 0) (tile |-> tc);
    rewrite dec_projection_output_owned cols output expected (block + 16) as
      when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected (block + 16));
  } else {
    rewrite when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected block) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected (block + 16));
  };
}

inline_for_extraction noextract
fn dec_projection_kernel_body
  (#nr:erased nat) (width:szp{dec_dense_width width}) (cols:szp)
  (layout:sz{dec_projection_layout cols layout})
  (offset:sz{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (input:larray bf16 (8 * width)) (weights:dec_bf16_array dec_weight_elements)
  (residual:larray bf16 nr) (output:larray bf16 (8 * cols)) (tile:larray F32.t 128)
  (tid:szlt 32)
  (#x:erased (lseq bf16 (8 * width))) (#w:erased (dec_bf16_values dec_weight_elements))
  (#rs:erased (lseq bf16 nr))
  (#cv:erased (lseq F32.t 128)) (#fx #fw #fr:perm)
  preserves gpu ** thread_id 32 tid **
    input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs
  requires tile |-> Frac (1.0R /. 32.0R) cv **
    when_ (SZ.v tid == 0) (Pulse.Lib.Array.pts_to_uninit output (8 * cols))
  ensures tile |-> Frac (1.0R /. 32.0R)
    (dec_previous_projection_tile width cols layout offset x w cv cols) **
    when_ (SZ.v tid == 0)
      (output |-> dec_projection_values width cols layout offset add_residual x w rs)
{
  let expected = hide (dec_projection_values width cols layout offset add_residual x w rs);
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (Pulse.Lib.Array.pts_to_uninit output (8 * cols))
      as Pulse.Lib.Array.pts_to_uninit output (8 * cols);
    unfold Pulse.Lib.Array.pts_to_uninit output (8 * cols);
    fold dec_projection_output_owned cols output expected 0;
    rewrite dec_projection_output_owned cols output expected 0 as
      when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected 0);
  } else {
    rewrite when_ (SZ.v tid == 0) (Pulse.Lib.Array.pts_to_uninit output (8 * cols)) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected 0);
  };
  let mut block = 0sz;
  while (!block <^ cols)
    invariant exists* (b:sz{b <= cols /\ b % 16 = 0}).
      block |-> b **
      tile |-> Frac (1.0R /. 32.0R) (dec_previous_projection_tile width cols layout offset x w cv b) **
      when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected b)
    decreases (cols - !block)
  {
    let b = !block;
    dec_projection_block_contiguous cols layout b 0;
    let weight_row = dec_projection_row_source cols layout b;
    let start = offset +^ weight_row *^ width;
    let count = 16sz *^ width;
    let a = dec_borrow_source_slice weights start count #w #fw;
    dec_dense_source_mma width a input tile
      #(hide (Seq.slice w start (start + count))) #x
      #(hide (dec_previous_projection_tile width cols layout offset x w cv b)) #fw #fx;
    dec_restore_source_slice weights start count a #w #fw;
    let tc = hide (Tensor.to_seq (rm 16 8) (dec_projection_scores width cols layout offset x w b));
    assert tile |-> Frac (1.0R /. 32.0R) tc;
    dec_warp_gather tile #(hide (reveal tc <: seq F32.t)) #1.0R #(hide (SZ.v tid));
    dec_projection_tile_expected width cols layout offset add_residual x w rs b;
    dec_projection_owner_commit cols b add_residual tile residual output tid #tc #rs #expected #fr;
    dec_warp_distribute tile #(hide (reveal tc <: seq F32.t)) #1.0R #(hide (SZ.v tid));
    let next = b +^ 16sz;
    assert pure (SZ.v next == SZ.v b + 16);
    rewrite when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected (b + 16))
      as when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected next);
    block := next;
  };
  let finished = !block;
  dec_source_loop_end finished cols;
  assert pure (SZ.v finished == SZ.v cols);
  rewrite when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected finished)
    as when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected cols);
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected cols)
      as dec_projection_output_owned cols output expected cols;
    unfold dec_projection_output_owned cols output expected cols;
    with masked. assert Pulse.Lib.Array.pts_to_mask output masked (fun _ -> True);
    dec_native_projection_finished cols masked expected;
    Pulse.Lib.Array.from_mask output;
    with ys. assert output |-> ys;
    assert pure (Seq.equal ys expected);
    assert output |-> expected;
    rewrite output |-> expected as when_ (SZ.v tid == 0) (output |-> expected);
  } else {
    rewrite when_ (SZ.v tid == 0) (dec_projection_output_owned cols output expected cols) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (output |-> expected);
  };
}

inline_for_extraction noextract
fn source_linear_body
  (#nr:erased nat) (width:szp{dec_dense_width width}) (cols:szp)
  (layout:sz{dec_projection_layout cols layout})
  (offset:sz{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (input:larray BF16.t (8 * width)) (weights:dec_bf16_array dec_weight_elements)
  (residual:larray BF16.t nr) (output:larray BF16.t (8 * cols))
  (tile:larray F32.t 128) (tid:szlt 32)
  (#x:erased (lseq BF16.t (8 * width))) (#w:erased (dec_bf16_values dec_weight_elements))
  (#rs:erased (lseq BF16.t nr)) (#cv:erased (lseq F32.t 128))
  (#fx #fw #fr:perm)
  preserves gpu ** thread_id 32 tid **
    input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs
  requires tile |-> Frac (1.0R /. 32.0R) cv **
    when_ (SZ.v tid == 0) (PA.pts_to_uninit output (8 * cols))
  ensures tile |-> Frac (1.0R /. 32.0R)
    (dec_previous_projection_tile width cols layout offset x w cv cols) **
    when_ (SZ.v tid == 0)
      (output |-> dec_projection_values width cols layout offset add_residual x w rs)
{
  dec_projection_kernel_body width cols layout offset add_residual
    input weights residual output tile tid #x #w #rs #cv #fx #fw #fr;
}

inline_for_extraction noextract
fn opt_linear_shared_thread
  (#nr:erased nat) (width:szp{dec_dense_width width}) (cols:szp)
  (layout:sz{dec_projection_layout cols layout})
  (offset:sz{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (input:larray BF16.t (8 * width)) (weights:dec_bf16_array dec_weight_elements)
  (residual:larray BF16.t nr) (output:larray BF16.t (8 * cols))
  (#x:erased (lseq BF16.t (8 * width))) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rs:erased (lseq BF16.t nr)) (#fx #fw #fr:perm)
  (sh:SH.c_shmems opt_linear_shared_desc) (bid:szlt 1) (tid:szlt 32) ()
  requires gpu ** pure (SH.c_shmems_inv sh) **
    opt_linear_kpre input weights residual output (opt_linear_tile sh) x w rs fx fw fr tid **
    thread_id 32 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 32) ** Barrier.barrier_state 0
  ensures gpu **
    opt_linear_kpost input weights residual output (opt_linear_tile sh) x w rs
      (dec_projection_values width cols layout offset add_residual x w rs)
      (opt_linear_final_tile width cols layout offset x w) fx fw fr tid **
    thread_id 32 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 32) ** Barrier.barrier_state 0
{
  unfold opt_linear_kpre input weights residual output (opt_linear_tile sh) x w rs fx fw fr tid;
  unfold opt_linear_read input weights residual x w rs
    (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R);
  let tile = opt_linear_tile sh;
  rewrite each opt_linear_tile sh as tile;
  with old. assert tile |-> Frac (1.0R /. 32.0R) old;
  source_linear_body width cols layout offset add_residual input weights residual output tile tid
    #x #w #rs #old #(fx /. 32.0R) #(fw /. 32.0R) #(fr /. 32.0R);
  opt_linear_old_tile_irrelevant width cols layout offset x w old;
  rewrite tile |-> Frac (1.0R /. 32.0R)
    (dec_previous_projection_tile width cols layout offset x w old cols) as
    tile |-> Frac (1.0R /. 32.0R) (opt_linear_final_tile width cols layout offset x w);
  fold opt_linear_read input weights residual x w rs
    (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R);
  fold opt_linear_kpost input weights residual output tile x w rs
    (dec_projection_values width cols layout offset add_residual x w rs)
    (opt_linear_final_tile width cols layout offset x w) fx fw fr tid;
  rewrite each tile as opt_linear_tile sh;
}

inline_for_extraction noextract
let u_linear_source_descriptor (width:szp{dec_dense_width width}) (cols:szp)
  (layout:sz{dec_projection_layout cols layout})
  (offset:sz{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= 8192})
  (input:larray BF16.t (8 * width){KA.is_global_array input})
  (weights:larray BF16.t dec_weight_elements{KA.is_global_array weights})
  (residual:larray BF16.t 8192{KA.is_global_array residual})
  (output:larray BF16.t (8 * cols){KA.is_global_array output})
  (#x:erased (lseq BF16.t (8 * width))) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rs:erased (lseq BF16.t 8192))
  : k:KS.kernel_desc
      (opt_linear_pre input weights residual output x w rs 1.0R 1.0R 1.0R)
      (opt_linear_post input weights residual output x w rs
        (dec_projection_values width cols layout offset add_residual x w rs) 1.0R 1.0R 1.0R){
    SZ.v k.nblk == 1 /\ k.nthr == 32sz /\ k.shmems_desc == opt_linear_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 32sz /\ k.barrier_count bid == 0)} =
  opt_linear_shared_descriptor #8192 width cols layout offset add_residual
    input weights residual output #x #w #rs #1.0R #1.0R #1.0R

inline_for_extraction noextract
ghost
fn u_attention_block_setup (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (#x:erased (lseq BF16.t 32768)) (#old:erased (lseq BF16.t 16384))
  (#fraction:perm) (sh:SH.c_shmems opt_attention_shared_desc) (_:natlt 1) ()
  norewrite
  requires SH.live_c_shmems sh ** u_attention_pre input output x old fraction
  ensures (forall+ (tid:natlt 32).
    u_attention_kpre input output (opt_attention_shared sh) x old fraction tid) ** emp
{
  SH.unfold_c_shmems sh (`%opt_attention_shared_desc);
  unfold u_attention_pre input output x old fraction;
  let shared = opt_attention_shared sh;
  rewrite each fst sh as shared.score_tile;
  rewrite each fst (snd sh) as shared.probability;
  rewrite each fst (snd (snd sh)) as shared.denominator;
  with old_tile. assert shared.score_tile |-> old_tile;
  PA.pts_to_len shared.score_tile;
  let tv = FStar.Ghost.hide #(lseq F32.t 128) (FStar.Ghost.reveal #(S.seq F32.t) old_tile);
  rewrite shared.score_tile |-> old_tile as shared.score_tile |-> tv;
  fold dec_array_live shared.probability;
  fold dec_array_live shared.denominator;
  source_warp_owner_intro (dec_array_live shared.probability);
  source_warp_owner_intro (dec_array_live shared.denominator);
  ArrayExtra.array_share shared.score_tile #tv #1.0R 32;
  forevery_map #(natlt 32)
    (fun _ -> shared.score_tile |-> Frac (1.0R /. 32.0R) tv)
    (fun _ -> exists* (v:lseq F32.t 128). shared.score_tile |-> Frac (1.0R /. 32.0R) v)
    fn _ { () };
  forevery_zip3 #(natlt 32)
    (fun _ -> exists* (v:lseq F32.t 128). shared.score_tile |-> Frac (1.0R /. 32.0R) v)
    (fun i -> when_ (i == 0) (dec_array_live shared.probability))
    (fun i -> when_ (i == 0) (dec_array_live shared.denominator));
  forevery_map #(natlt 32)
    (fun i -> (exists* (v:lseq F32.t 128). shared.score_tile |-> Frac (1.0R /. 32.0R) v) **
      when_ (i == 0) (dec_array_live shared.probability) **
      when_ (i == 0) (dec_array_live shared.denominator))
    (fun i -> source_attention_available shared i)
    fn i { fold source_attention_available shared i; };
  ArrayExtra.array_share input #x #fraction 32;
  source_warp_owner_intro (output |-> old);
  forevery_zip3 #(natlt 32)
    (fun _ -> input |-> Frac (fraction /. 32.0R) x)
    (fun i -> source_attention_available shared i)
    (fun i -> when_ (i == 0) (output |-> old));
  forevery_map #(natlt 32)
    (fun i -> input |-> Frac (fraction /. 32.0R) x **
      source_attention_available shared i ** when_ (i == 0) (output |-> old))
    (fun i -> u_attention_kpre input output shared x old fraction i)
    fn i { fold u_attention_kpre input output shared x old fraction i; };
  rewrite each shared as opt_attention_shared sh;
}

inline_for_extraction noextract
ghost
fn u_attention_block_teardown (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (#x:erased (lseq BF16.t 32768)) (#fraction:perm)
  (sh:SH.c_shmems opt_attention_shared_desc) (_:natlt 1) ()
  norewrite
  requires (forall+ (tid:natlt 32).
    u_attention_kpost input output (opt_attention_shared sh) x fraction tid) ** emp
  ensures SH.live_c_shmems sh ** u_attention_post input output x fraction
{
  let shared = opt_attention_shared sh;
  rewrite each opt_attention_shared sh as shared;
  forevery_map #(natlt 32)
    (fun i -> u_attention_kpost input output shared x fraction i)
    (fun i -> input |-> Frac (fraction /. 32.0R) x ** source_attention_available shared i **
      when_ (i == 0) (u_attention_final output x))
    fn i { unfold u_attention_kpost input output shared x fraction i; };
  forevery_unzip3 #(natlt 32)
    (fun _ -> input |-> Frac (fraction /. 32.0R) x)
    (fun i -> source_attention_available shared i)
    (fun i -> when_ (i == 0) (u_attention_final output x));
  ArrayExtra.array_gather input #x #fraction 32;
  source_warp_owner_elim (u_attention_final output x);
  fold u_attention_post input output x fraction;
  forevery_map #(natlt 32)
    (fun i -> source_attention_available shared i)
    (fun i -> (exists* (v:lseq F32.t 128). shared.score_tile |-> Frac (1.0R /. 32.0R) v) **
      when_ (i == 0) (dec_array_live shared.probability) **
      when_ (i == 0) (dec_array_live shared.denominator))
    fn i { unfold source_attention_available shared i; };
  forevery_unzip3 #(natlt 32)
    (fun _ -> exists* (v:lseq F32.t 128). shared.score_tile |-> Frac (1.0R /. 32.0R) v)
    (fun i -> when_ (i == 0) (dec_array_live shared.probability))
    (fun i -> when_ (i == 0) (dec_array_live shared.denominator));
  forevery_map #(natlt 32)
    (fun _ -> exists* (v:lseq F32.t 128). shared.score_tile |-> Frac (1.0R /. 32.0R) v)
    (fun _ -> exists* (v:S.seq F32.t). shared.score_tile |-> Frac (1.0R /. 32.0R) v)
    fn _ { with v. assert shared.score_tile |-> Frac (1.0R /. 32.0R) v; () };
  KA.array_gather_underspec shared.score_tile 32;
  source_warp_owner_elim (dec_array_live shared.probability);
  source_warp_owner_elim (dec_array_live shared.denominator);
  unfold dec_array_live shared.probability;
  unfold dec_array_live shared.denominator;
  rewrite each shared.score_tile as fst sh;
  rewrite each shared.probability as fst (snd sh);
  rewrite each shared.denominator as fst (snd (snd sh));
  SH.fold_c_shmems sh (`%opt_attention_shared_desc);
}

inline_for_extraction noextract
ghost
fn source_tile_gather_proof (tile:larray F32.t 128)
  (#values:erased (lseq F32.t 128)) ()
  requires forall+ (_:natlt 32). tile |-> Frac (1.0R /. 32.0R) values
  ensures forall+ (tid:natlt 32). when_ (tid == 0) (tile |-> values)
{
  ArrayExtra.array_gather tile #values #1.0R 32;
  source_warp_owner_intro (tile |-> values);
}

inline_for_extraction noextract
ghost
fn source_publish_observed_proof (shared:source_attention_shared)
  (tile:lseq F32.t 128)
  (fact:lseq BF16.t 1024 -> lseq F32.t 16 -> prop) ()
  requires forall+ (tid:natlt 32).
    when_ (tid == 0) (source_probability_owned shared tile fact)
  ensures forall+ (_:natlt 32). source_probability_shared shared tile fact
{
  source_warp_owner_elim (source_probability_owned shared tile fact);
  unfold source_probability_owned shared tile fact;
  with pv. assert shared.probability |-> pv;
  with dv. assert shared.denominator |-> dv;
  ArrayExtra.array_share shared.score_tile #tile #1.0R 32;
  ArrayExtra.array_share shared.probability #pv #1.0R 32;
  ArrayExtra.array_share shared.denominator #dv #1.0R 32;
  forevery_zip3 #(natlt 32)
    (fun _ -> shared.score_tile |-> Frac (1.0R /. 32.0R) tile)
    (fun _ -> shared.probability |-> Frac (1.0R /. 32.0R) pv)
    (fun _ -> shared.denominator |-> Frac (1.0R /. 32.0R) dv);
  forevery_intro_pure #(natlt 32) (fun _ -> fact pv dv);
  forevery_zip #(natlt 32)
    (fun _ -> shared.score_tile |-> Frac (1.0R /. 32.0R) tile **
      shared.probability |-> Frac (1.0R /. 32.0R) pv **
      shared.denominator |-> Frac (1.0R /. 32.0R) dv)
    (fun _ -> pure (fact pv dv));
  forevery_map #(natlt 32)
    (fun _ -> (shared.score_tile |-> Frac (1.0R /. 32.0R) tile **
      shared.probability |-> Frac (1.0R /. 32.0R) pv **
      shared.denominator |-> Frac (1.0R /. 32.0R) dv) ** pure (fact pv dv))
    (fun _ -> source_probability_shared shared tile fact)
    fn _ { fold source_probability_shared shared tile fact; };
}

inline_for_extraction noextract
ghost
fn source_forget_probability_owner (shared:source_attention_shared) (tid:szlt 32)
  (#pv:erased (lseq BF16.t 1024)) (#dv:erased (lseq F32.t 16))
  requires when_ (SZ.v tid == 0) (shared.probability |-> pv ** shared.denominator |-> dv)
  ensures when_ (SZ.v tid == 0) (dec_array_live shared.probability) **
    when_ (SZ.v tid == 0) (dec_array_live shared.denominator)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (shared.probability |-> pv ** shared.denominator |-> dv)
      as shared.probability |-> pv ** shared.denominator |-> dv;
    fold dec_array_live shared.probability;
    fold dec_array_live shared.denominator;
    rewrite dec_array_live shared.probability as
      when_ (SZ.v tid == 0) (dec_array_live shared.probability);
    rewrite dec_array_live shared.denominator as
      when_ (SZ.v tid == 0) (dec_array_live shared.denominator);
  } else {
    rewrite when_ (SZ.v tid == 0) (shared.probability |-> pv ** shared.denominator |-> dv) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (dec_array_live shared.probability);
    rewrite emp as when_ (SZ.v tid == 0) (dec_array_live shared.denominator);
  };
}

inline_for_extraction noextract
ghost
fn source_tile_distribute_proof (tile:larray F32.t 128)
  (#values:erased (lseq F32.t 128)) ()
  requires forall+ (tid:natlt 32). when_ (tid == 0) (tile |-> values)
  ensures forall+ (_:natlt 32). tile |-> Frac (1.0R /. 32.0R) values
{
  source_warp_owner_elim (tile |-> values);
  ArrayExtra.array_share tile #values #1.0R 32;
}

inline_for_extraction noextract
ghost
fn source_head_retire_proof (shared:source_attention_shared)
  (#tv:erased (lseq F32.t 128)) (#pv:erased (lseq BF16.t 1024))
  (#dv:erased (lseq F32.t 16)) ()
  requires forall+ (tid:natlt 32).
    when_ (tid == 0) (shared.score_tile |-> tv) **
    shared.probability |-> Frac (1.0R /. 32.0R) pv **
    shared.denominator |-> Frac (1.0R /. 32.0R) dv
  ensures forall+ (tid:natlt 32).
    shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
    when_ (tid == 0) (shared.probability |-> pv ** shared.denominator |-> dv)
{
  forevery_unzip3 #(natlt 32)
    (fun i -> when_ (i == 0) (shared.score_tile |-> tv))
    (fun _ -> shared.probability |-> Frac (1.0R /. 32.0R) pv)
    (fun _ -> shared.denominator |-> Frac (1.0R /. 32.0R) dv);
  source_warp_owner_elim (shared.score_tile |-> tv);
  ArrayExtra.array_share shared.score_tile #tv #1.0R 32;
  ArrayExtra.array_gather shared.probability #pv #1.0R 32;
  ArrayExtra.array_gather shared.denominator #dv #1.0R 32;
  source_warp_owner_intro (shared.probability |-> pv ** shared.denominator |-> dv);
  forevery_zip #(natlt 32)
    (fun _ -> shared.score_tile |-> Frac (1.0R /. 32.0R) tv)
    (fun i -> when_ (i == 0) (shared.probability |-> pv ** shared.denominator |-> dv));
}

inline_for_extraction noextract
ghost
fn source_value_end_proof (shared:source_attention_shared) (block:natlt 16)
  (#tv:erased (lseq F32.t 128)) (#pv:erased (lseq BF16.t 1024))
  (#dv:erased (lseq F32.t 16)) ()
  requires forall+ (tid:natlt 32).
    when_ (tid == 0) (shared.score_tile |-> tv) **
    shared.probability |-> Frac (1.0R /. 32.0R) pv **
    shared.denominator |-> Frac (1.0R /. 32.0R) dv
  ensures forall+ (tid:natlt 32).
    shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
    source_value_read_phase shared tid pv dv (block + 1)
{
  if (block = 15) {
    source_head_retire_proof shared #tv #pv #dv ();
    forevery_ext #(natlt 32)
      (fun i -> shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
        when_ (i == 0) (shared.probability |-> pv ** shared.denominator |-> dv))
      (fun i -> shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
        source_value_read_phase shared i pv dv (block + 1));
  } else {
    forevery_unzip3 #(natlt 32)
      (fun i -> when_ (i == 0) (shared.score_tile |-> tv))
      (fun _ -> shared.probability |-> Frac (1.0R /. 32.0R) pv)
      (fun _ -> shared.denominator |-> Frac (1.0R /. 32.0R) dv);
    source_tile_distribute_proof shared.score_tile #tv ();
    forevery_zip3 #(natlt 32)
      (fun _ -> shared.score_tile |-> Frac (1.0R /. 32.0R) tv)
      (fun _ -> shared.probability |-> Frac (1.0R /. 32.0R) pv)
      (fun _ -> shared.denominator |-> Frac (1.0R /. 32.0R) dv);
    forevery_ext #(natlt 32)
      (fun _ -> shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
        shared.probability |-> Frac (1.0R /. 32.0R) pv **
        shared.denominator |-> Frac (1.0R /. 32.0R) dv)
      (fun i -> shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
        source_value_read_phase shared i pv dv (block + 1));
  };
}

inline_for_extraction noextract
ghost
fn dec_take_owned_array (#a:Type0) (#n:erased nat) (arr:larray a n) (tid:szlt 32) (default:a)
  requires when_ (SZ.v tid == 0) (dec_array_live arr)
  returns dec_spec_values:erased (lseq a n)
  ensures when_ (SZ.v tid == 0) (arr |-> dec_spec_values)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (dec_array_live arr) as dec_array_live arr;
    unfold dec_array_live arr;
    with dec_spec_values. assert arr |-> dec_spec_values;
    pts_to_len arr;
    let result = hide (dec_seq_length_view n (reveal dec_spec_values));
    rewrite arr |-> dec_spec_values as when_ (SZ.v tid == 0) (arr |-> result);
    result
  } else {
    rewrite when_ (SZ.v tid == 0) (dec_array_live arr) as emp;
    let result = hide (dec_seq_length_view n (Seq.create n default));
    rewrite emp as when_ (SZ.v tid == 0) (arr |-> result);
    result
  }
}

inline_for_extraction noextract
fn dec_source_read_score_tile
  (tile:larray ObF32.t 128) (row:szlt 16) (col:szlt 64)
  (#dec_spec_values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f dec_spec_values
  returns result:ObF32.t
  ensures pure (result ==
    dec_source_row_score row (dec_source_score_row dec_spec_values row) col)
{
  open Pulse.Lib.Array;
  if (row <^ 14sz && col <=^ row /^ 2sz) {
    dec_source_score_row_at dec_spec_values row col;
    tile.(row *^ 8sz +^ col)
  } else {
    dec_negative_infinity ()
  }
}

inline_for_extraction noextract
fn dec_source_part_maximum_from_tile
  (tile:larray ObF32.t 128) (row:szlt 16) (part:szlt 4)
  (#dec_spec_values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f dec_spec_values
  returns result:ObF32.t
  ensures pure (result ==
    dec_source_part_maximum row (dec_source_score_row dec_spec_values row) part 16)
{
  let mut j = 0sz;
  let mut maximum = dec_negative_infinity ();
  while (!j <^ 16sz)
    invariant exists* (step:sz{step <= 16}).
      j |-> step **
      maximum |-> dec_source_part_maximum row (dec_source_score_row dec_spec_values row) part step
    decreases (16 - !j)
  {
    let step = !j;
    let col = part *^ 2sz +^ (step /^ 2sz) *^ 8sz +^ step %^ 2sz;
    let value = dec_source_read_score_tile tile row col;
    let previous = !maximum;
    dec_source_part_maximum_step row (dec_source_score_row dec_spec_values row) part step;
    maximum := ObF.fmax previous value;
    j := step +^ 1sz;
  };
  !maximum
}

inline_for_extraction noextract
fn dec_source_maximum_from_tile
  (tile:larray ObF32.t 128) (row:szlt 16)
  (#dec_spec_values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f dec_spec_values
  returns result:ObF32.t
  ensures pure (
    result == dec_source_maximum row (dec_source_score_row dec_spec_values row) /\
    not (NaN? (kind result)))
{
  let m0 = dec_source_part_maximum_from_tile tile row 0sz;
  let m1 = dec_source_part_maximum_from_tile tile row 1sz;
  let m2 = dec_source_part_maximum_from_tile tile row 2sz;
  let m3 = dec_source_part_maximum_from_tile tile row 3sz;
  dec_source_maximum_non_nan row (dec_source_score_row dec_spec_values row);
  ObF.fmax (ObF.fmax m0 m1) (ObF.fmax m2 m3)
}

inline_for_extraction noextract
fn dec_source_exponent_from_scaled (value scaled:ObF32.t)
  preserves gpu
  returns result:ObF32.t
  ensures pure (dec_source_branch_exponent_contract value scaled result)
{
  if ieee_eq value (dec_negative_infinity ()) {
    let zero_ok = a0 #ObF32.t;
    ObF.zero
  } else {
    let argument = ObF32.fma_rn_ftz value (dec_source_attention_literal ())
      (NativeNeg.neg scaled);
    let result = ObF32.exp2_approx_ftz argument;
    dec_source_exponent_chain value scaled argument result;
    result
  }
}

inline_for_extraction noextract
fn dec_source_denominator_combine (rescale s0 s1 s2 s3:ObF32.t)
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
fn u_probability_row
  (tile:larray F32.t 128) (probability:larray BF16.t 1024)
  (sums:larray F32.t 4) (denominator:larray F32.t 16) (row:szlt 16)
  (#x:erased (lseq BF16.t 32768)) (#head:erased (Shared.index 8))
  (#old:erased (lseq BF16.t 1024)) (#old_sums:erased (lseq F32.t 4))
  (#old_den:erased (lseq F32.t 16)) (#ft:perm)
  preserves gpu ** tile |-> Frac ft (dec_source_mma_score_tile x head)
  requires probability |-> old ** sums |-> old_sums ** denominator |-> old_den
  ensures exists* (ps:lseq BF16.t 1024) (ss:lseq F32.t 4) (ds:lseq F32.t 16).
    probability |-> ps ** sums |-> ss ** denominator |-> ds **
    pure (dec_segment_frame old ps (row * 64) 64 /\ dec_segment_frame old_den ds row 1 /\
      (forall (m:u_attention_meaning). u_probability_guard x head m ==>
        dec_bf16_segment_rep ps (row * 64) 64 (dec_attention_numerator x m.u_query m.u_key head row) /\
        S.index ds row %~ dec_attention_denominator x m.u_query m.u_key head row))
{
  let maximum = dec_source_maximum_from_tile tile row;
  dec_source_attention_literal_is_positive_normal ();
  dec_source_scale_upper_bound ();
  dec_source_maximum_cases maximum;
  dec_negative_infinity_compares_equal ();
  let scaled = F32.mul_rn_ftz maximum (dec_source_attention_literal ());
  let mut part = 0sz;
  while (!part <^ 4sz)
    invariant exists* (p:sz{p <= 4}) (ps:lseq BF16.t 1024) (ss:lseq F32.t 4).
      part |-> p ** probability |-> ps ** sums |-> ss **
      pure (dec_segment_frame old ps (row * 64) 64 /\
        (forall (m:u_attention_meaning).
          dec_probability_prefix ps row (dec_attention_numerator x m.u_query m.u_key head row)
            p 0 (u_probability_guard x head m) /\
          (forall (i:Shared.index 4{i < p}).
            u_probability_guard x head m ==> S.index ss i %~ dec_source_sum_n 16
              (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) i))))
    decreases (4 - !part)
  {
    let p = !part;
    let mut sum:F32.t = F.zero;
    let mut step = 0sz;
    while (!step <^ 16sz)
      invariant exists* (j:sz{j <= 16}) (ps:lseq BF16.t 1024) (acc:F32.t).
        step |-> j ** sum |-> acc ** probability |-> ps **
        pure (dec_segment_frame old ps (row * 64) 64 /\
          (forall (m:u_attention_meaning).
            dec_probability_prefix ps row (dec_attention_numerator x m.u_query m.u_key head row)
              p j (u_probability_guard x head m) /\
            (u_probability_guard x head m ==> acc %~ dec_source_sum_n j
              (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) p))))
      decreases (16 - !step)
    {
      let j = !step;
      let col = p *^ 2sz +^ (j /^ 2sz) *^ 8sz +^ j %^ 2sz;
      let value = dec_source_read_score_tile tile row col;
      let exponent = dec_source_exponent_from_scaled value scaled;
      u_exponent_related x head row col scaled exponent;
      dec_f32_to_bf16_all exponent;
      let rounded = Casts.fcast #F32.t #BF16.t exponent;
      with before. assert probability |-> before;
      probability.(row *^ 64sz +^ col) <- rounded;
      with after. assert probability |-> after;
      u_probability_write x head before after row p j rounded;
      let previous = !sum;
      let added = F32.add_rn_ftz previous exponent;
      u_probability_sum_step x head row p j previous exponent added;
      sum := added;
      step := j +^ 1sz;
    };
    dec_source_loop_end !step 16sz;
    let accumulated = !sum;
    sums.(p) <- accumulated;
    part := p +^ 1sz;
  };
  dec_source_loop_end !part 4sz;
  let rescale = source_attention_rescale maximum scaled;
  dec_attention_rescale_refines x head row rescale;
  let s0 = sums.(0sz);
  let s1 = sums.(1sz);
  let s2 = sums.(2sz);
  let s3 = sums.(3sz);
  let result = dec_source_denominator_combine rescale s0 s1 s2 s3;
  u_probability_combine x head row rescale s0 s1 s2 s3 result;
  denominator.(row) <- result;
  with ps. assert probability |-> ps;
  u_probability_row_finish x head ps row;
  ()
}

inline_for_extraction noextract
fn u_probability_rows (tile:larray F32.t 128) (probability:larray BF16.t 1024)
  (denominator:larray F32.t 16)
  (#x:erased (lseq BF16.t 32768)) (#head:erased (Shared.index 8))
  (#old_p:erased (lseq BF16.t 1024)) (#old_d:erased (lseq F32.t 16))
  preserves gpu ** tile |-> dec_source_mma_score_tile x head
  requires probability |-> old_p ** denominator |-> old_d
  ensures exists* (pv:lseq BF16.t 1024) (dv:lseq F32.t 16).
    probability |-> pv ** denominator |-> dv ** pure (u_probability_contract x head pv dv)
{
  let mut row = 0sz;
  while (!row <^ 16sz)
    invariant exists* (r:sz{r <= 16}) (ps:lseq BF16.t 1024) (ds:lseq F32.t 16).
      row |-> r ** probability |-> ps ** denominator |-> ds **
      pure (forall (m:u_attention_meaning). u_probability_guard x head m ==>
        dec_matrix_prefix ps (dec_attention_numerator x m.u_query m.u_key head) r /\
        (forall (i:Shared.index 16{i < r}).
          S.index ds i %~ dec_attention_denominator x m.u_query m.u_key head i))
    decreases (16 - !row)
  {
    let r = !row;
    let initial:F32.t = F.zero;
    let mut sums = [| initial; 4sz |];
    with before. assert probability |-> before;
    with before_d. assert denominator |-> before_d;
    u_probability_row tile probability sums denominator r
      #x #head #before #(hide (S.create 4 initial)) #before_d #1.0R;
    with after. assert probability |-> after;
    with after_d. assert denominator |-> after_d;
    u_probability_rows_advance x head before after before_d after_d r;
    row := r +^ 1sz;
  };
  dec_source_loop_end !row 16sz;
  ()
}

inline_for_extraction noextract
fn u_owner_probability (shared:source_attention_shared) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768)) (#head:erased (Shared.index 8))
  preserves gpu
  requires
    when_ (SZ.v tid == 0) (shared.score_tile |-> dec_source_mma_score_tile x head) **
    when_ (SZ.v tid == 0) (dec_array_live shared.probability) **
    when_ (SZ.v tid == 0) (dec_array_live shared.denominator)
  ensures when_ (SZ.v tid == 0)
    (source_probability_owned shared (dec_source_mma_score_tile x head) (u_probability_contract x head))
{
  let old_p = dec_take_owned_array shared.probability tid (F.zero #BF16.t);
  let old_d = dec_take_owned_array shared.denominator tid (F.zero #F32.t);
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (shared.score_tile |-> dec_source_mma_score_tile x head)
      as shared.score_tile |-> dec_source_mma_score_tile x head;
    rewrite when_ (SZ.v tid == 0) (shared.probability |-> old_p) as shared.probability |-> old_p;
    rewrite when_ (SZ.v tid == 0) (shared.denominator |-> old_d) as shared.denominator |-> old_d;
    u_probability_rows shared.score_tile shared.probability shared.denominator #x #head #old_p #old_d;
    fold source_probability_owned shared (dec_source_mma_score_tile x head) (u_probability_contract x head);
    rewrite source_probability_owned shared (dec_source_mma_score_tile x head)
      (u_probability_contract x head) as when_ (SZ.v tid == 0)
        (source_probability_owned shared (dec_source_mma_score_tile x head) (u_probability_contract x head));
  } else {
    rewrite when_ (SZ.v tid == 0) (shared.score_tile |-> dec_source_mma_score_tile x head) as emp;
    rewrite when_ (SZ.v tid == 0) (shared.probability |-> old_p) as emp;
    rewrite when_ (SZ.v tid == 0) (shared.denominator |-> old_d) as emp;
    rewrite emp as when_ (SZ.v tid == 0)
      (source_probability_owned shared (dec_source_mma_score_tile x head) (u_probability_contract x head));
  };
}

inline_for_extraction noextract
ghost
fn u_advance_attention_output (output:larray BF16.t 16384) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768)) (head:szlt 8)
  requires when_ (SZ.v tid == 0) (u_attention_owned output x head 128)
  ensures when_ (SZ.v tid == 0) (u_attention_owned output x (head + 1) 0)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (u_attention_owned output x head 128)
      as u_attention_owned output x head 128;
    unfold u_attention_owned output x head 128;
    with ys. assert output |-> ys;
    u_attention_head_advance x ys head;
    fold u_attention_owned output x (head + 1) 0;
    rewrite u_attention_owned output x (head + 1) 0 as
      when_ (SZ.v tid == 0) (u_attention_owned output x (head + 1) 0);
  } else {
    rewrite when_ (SZ.v tid == 0) (u_attention_owned output x head 128) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (u_attention_owned output x (head + 1) 0);
  };
}

inline_for_extraction noextract
fn dec_source_normalize_value (numerator denominator:ObF32.t)
  preserves gpu
  returns result:bf16
  ensures pure (
    forall (n:real) (d:real{d =!= 0.0R}).
      v_approximates numerator n /\ v_approximates denominator d ==>
      v_approximates result (n /. d))
{
  let inverse = ObF32.rcp_approx_ftz denominator;
  let product = ObF32.mul_rn_ftz numerator inverse;
  dec_source_normalization_chain numerator denominator inverse product;
  ObCasts.fcast #ObF32.t #bf16 product
}

inline_for_extraction noextract
fn u_write_attention_tile
  (tile:larray F32.t 128) (denominator:larray F32.t 16) (output:larray BF16.t 16384)
  (head:szlt 8) (block:szlt 128{block % 8 = 0})
  (#x:erased (lseq BF16.t 32768))
  (#ps:erased (lseq BF16.t 1024)) (#ds:erased (lseq F32.t 16)) (#fd:perm)
  preserves gpu **
    tile |-> Tensor.to_seq (rm 16 8) (dec_dense_mma_fold 64 ps (dec_packed_values x head block) 4) **
    denominator |-> Frac fd ds
  requires u_attention_owned output x head block ** pure (u_probability_contract x head ps ds)
  ensures u_attention_owned output x head (block + 8)
{
  unfold u_attention_owned output x head block;
  let mut row = 0sz;
  while (!row <^ 14sz)
    invariant exists* (r:sz{r <= 14}) (ys:lseq BF16.t 16384).
      row |-> r ** output |-> ys ** pure (u_attention_prefix x ys head block /\
        (forall (m:u_attention_meaning). u_attention_guard x m ==>
          dec_attention_tile_prefix ys (u_attention_model m) head block (r * 8)))
    decreases (14 - !row)
  {
    let r = !row;
    let mut col = 0sz;
    while (!col <^ 8sz)
      invariant exists* (c:sz{c <= 8}) (ys:lseq BF16.t 16384).
        col |-> c ** output |-> ys ** pure (u_attention_prefix x ys head block /\
          (forall (m:u_attention_meaning). u_attention_guard x m ==>
            dec_attention_tile_prefix ys (u_attention_model m) head block (r * 8 + c)))
      decreases (8 - !col)
    {
      let c = !col;
      let numerator = tile.(r *^ 8sz +^ c);
      let den = denominator.(r);
      let value = dec_source_normalize_value numerator den;
      dec_source_tile_sequence_at (dec_dense_mma_fold 64 ps (dec_packed_values x head block) 4) r c;
      u_attention_value_related x head ps ds block r c value;
      with before. assert output |-> before;
      output.(r /^ 2sz *^ 2048sz +^ head *^ 256sz +^ r %^ 2sz *^ 128sz +^ block +^ c) <- value;
      with after. assert output |-> after;
      u_attention_write x before after head block r c value;
      col := c +^ 1sz;
    };
    dec_source_loop_end !col 8sz;
    row := r +^ 1sz;
  };
  dec_source_loop_end !row 14sz;
  with ys. assert output |-> ys;
  u_attention_tile_done x ys head block;
  fold u_attention_owned output x head (block + 8);
}

inline_for_extraction noextract
fn u_owner_value_commit (shared:source_attention_shared) (output:larray BF16.t 16384)
  (head:szlt 8) (block:szlt 16) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768))
  (#pv:erased (lseq BF16.t 1024)) (#dv:erased (lseq F32.t 16))
  preserves gpu ** shared.denominator |-> Frac (1.0R /. 32.0R) dv **
    when_ (SZ.v tid == 0) (shared.score_tile |-> Tensor.to_seq (rm 16 8)
      (dec_dense_mma_fold 64 pv (dec_packed_values x head (block * 8)) 4))
  requires when_ (SZ.v tid == 0) (u_attention_owned output x head (block * 8)) **
    pure (u_probability_contract x head pv dv)
  ensures when_ (SZ.v tid == 0) (u_attention_owned output x head ((block + 1) * 8))
{
  let tile = hide (Tensor.to_seq (rm 16 8)
    (dec_dense_mma_fold 64 pv (dec_packed_values x head (block * 8)) 4));
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (shared.score_tile |-> tile) as shared.score_tile |-> tile;
    rewrite when_ (SZ.v tid == 0) (u_attention_owned output x head (block * 8))
      as u_attention_owned output x head (block * 8);
    u_write_attention_tile shared.score_tile shared.denominator output head (block *^ 8sz)
      #x #pv #dv #(1.0R /. 32.0R);
    rewrite shared.score_tile |-> tile as when_ (SZ.v tid == 0) (shared.score_tile |-> tile);
    rewrite u_attention_owned output x head (block * 8 + 8) as
      when_ (SZ.v tid == 0) (u_attention_owned output x head ((block + 1) * 8));
  } else {
    rewrite when_ (SZ.v tid == 0) (u_attention_owned output x head (block * 8)) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (u_attention_owned output x head ((block + 1) * 8));
  };
}

inline_for_extraction noextract
fn u_attention_one_head (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (shared:source_attention_shared) (head:szlt 8) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768)) (#fraction:perm)
  preserves gpu ** thread_id 32 tid ** input |-> Frac fraction x
  requires source_attention_available shared tid **
    when_ (SZ.v tid == 0) (u_attention_owned output x head 0)
  ensures source_attention_available shared tid **
    when_ (SZ.v tid == 0) (u_attention_owned output x (head + 1) 0)
{
  unfold source_attention_available shared tid;
  with old_tile. assert shared.score_tile |-> Frac (1.0R /. 32.0R) old_tile;
  source_attention_score_tile input head shared.score_tile tid #x #old_tile #fraction;
  source_score_tile_related x head;
  rewrite shared.score_tile |-> Frac (1.0R /. 32.0R)
      (source_tile_values (source_score_prefix x head 8)) as
    shared.score_tile |-> Frac (1.0R /. 32.0R) (dec_source_mma_score_tile x head);
  source_tile_gather shared.score_tile #(hide (dec_source_mma_score_tile x head)) #(hide (SZ.v tid));
  u_owner_probability shared tid #x #(hide (SZ.v head));
  source_publish_observed shared #(hide (dec_source_mma_score_tile x head))
    (u_probability_contract x head) #(hide (SZ.v tid));
  unfold source_probability_shared shared (dec_source_mma_score_tile x head) (u_probability_contract x head);
  with pv. assert shared.probability |-> Frac (1.0R /. 32.0R) pv;
  with dv. assert shared.denominator |-> Frac (1.0R /. 32.0R) dv;
  let mut block = 0sz;
  fold source_value_read_phase shared tid pv dv 0;
  while (!block <^ 16sz)
    invariant exists* (b:sz{b <= 16}) (tv:lseq F32.t 128).
      block |-> b ** shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
      source_value_read_phase shared tid pv dv b **
      when_ (SZ.v tid == 0) (u_attention_owned output x head (b * 8))
    decreases (16 - !block)
  {
    let b = !block;
    rewrite source_value_read_phase shared tid pv dv b as
      shared.probability |-> Frac (1.0R /. 32.0R) pv **
      shared.denominator |-> Frac (1.0R /. 32.0R) dv;
    with before. assert shared.score_tile |-> Frac (1.0R /. 32.0R) before;
    source_attention_value_tile input shared.probability head b shared.score_tile tid
      #x #pv #before #fraction #(1.0R /. 32.0R);
    source_value_tile_related x pv head b;
    let tile = hide (Tensor.to_seq (rm 16 8)
      (dec_dense_mma_fold 64 pv (dec_packed_values x head (b * 8)) 4));
    rewrite shared.score_tile |-> Frac (1.0R /. 32.0R)
        (source_tile_values (source_value_prefix x pv head b 4)) as
      shared.score_tile |-> Frac (1.0R /. 32.0R) tile;
    source_tile_gather shared.score_tile #tile #(hide (SZ.v tid));
    u_owner_value_commit shared output head b tid #x #pv #dv;
    source_value_end shared #(hide (SZ.v b)) #tile #pv #dv #(hide (SZ.v tid));
    let next = b +^ 1sz;
    assert pure (SZ.v next == SZ.v b + 1);
    rewrite source_value_read_phase shared tid pv dv (b + 1) as source_value_read_phase shared tid pv dv next;
    rewrite when_ (SZ.v tid == 0) (u_attention_owned output x head ((b + 1) * 8)) as
      when_ (SZ.v tid == 0) (u_attention_owned output x head (next * 8));
    block := next;
  };
  let finished = !block;
  dec_source_loop_end finished 16sz;
  rewrite source_value_read_phase shared tid pv dv finished as source_value_read_phase shared tid pv dv 16;
  rewrite when_ (SZ.v tid == 0) (u_attention_owned output x head (finished * 8)) as
    when_ (SZ.v tid == 0) (u_attention_owned output x head 128);
  unfold source_value_read_phase shared tid pv dv 16;
  source_forget_probability_owner shared tid #pv #dv;
  u_advance_attention_output output tid #x head;
  fold source_attention_available shared tid;
}

inline_for_extraction noextract
ghost
fn u_attention_output_begin (output:larray BF16.t 16384) (tid:szlt 32)
  (#old:erased (lseq BF16.t 16384)) (#x:erased (lseq BF16.t 32768))
  requires when_ (SZ.v tid == 0) (output |-> old)
  ensures when_ (SZ.v tid == 0) (u_attention_owned output x 0 0)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (output |-> old) as output |-> old;
    fold u_attention_owned output x 0 0;
    rewrite u_attention_owned output x 0 0 as when_ (SZ.v tid == 0) (u_attention_owned output x 0 0);
  } else {
    rewrite when_ (SZ.v tid == 0) (output |-> old) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (u_attention_owned output x 0 0);
  };
}

inline_for_extraction noextract
fn u_pad_attention_output (output:larray BF16.t 16384) (#x:erased (lseq BF16.t 32768))
  preserves gpu
  requires u_attention_owned output x 8 0
  ensures u_attention_final output x
{
  unfold u_attention_owned output x 8 0;
  with original. assert output |-> original;
  let mut col = 0sz;
  while (!col <^ 2048sz)
    invariant exists* (i:sz{i <= 2048}) (ys:lseq BF16.t 16384).
      col |-> i ** output |-> ys **
      pure (dec_segment_frame original ys 14336 i /\
        (forall (j:Shared.index 2048{j < i}). S.index ys (14336 + j) == (F.zero #BF16.t)))
    decreases (2048 - !col)
  {
    let i = !col;
    output.(14336sz +^ i) <- F.zero #BF16.t;
    col := i +^ 1sz;
  };
  dec_source_loop_end !col 2048sz;
  with ys. assert output |-> ys;
  u_attention_padding_related x original ys;
  fold u_attention_final output x;
}

inline_for_extraction noextract
fn u_owner_attention_padding (output:larray BF16.t 16384) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768))
  preserves gpu
  requires when_ (SZ.v tid == 0) (u_attention_owned output x 8 0)
  ensures when_ (SZ.v tid == 0) (u_attention_final output x)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (u_attention_owned output x 8 0) as u_attention_owned output x 8 0;
    u_pad_attention_output output #x;
    rewrite u_attention_final output x as when_ (SZ.v tid == 0) (u_attention_final output x);
  } else {
    rewrite when_ (SZ.v tid == 0) (u_attention_owned output x 8 0) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (u_attention_final output x);
  };
}

inline_for_extraction noextract
fn u_attention_body (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (shared:source_attention_shared) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768)) (#old:erased (lseq BF16.t 16384)) (#fraction:perm)
  preserves gpu ** thread_id 32 tid ** input |-> Frac fraction x
  requires source_attention_available shared tid ** when_ (SZ.v tid == 0) (output |-> old)
  ensures source_attention_available shared tid ** when_ (SZ.v tid == 0) (u_attention_final output x)
{
  u_attention_output_begin output tid #old #x;
  let mut head = 0sz;
  while (!head <^ 8sz)
    invariant exists* (h:sz{h <= 8}).
      head |-> h ** source_attention_available shared tid **
      when_ (SZ.v tid == 0) (u_attention_owned output x h 0)
    decreases (8 - !head)
  {
    let h = !head;
    u_attention_one_head input output shared h tid #x #fraction;
    let next = h +^ 1sz;
    assert pure (SZ.v next == SZ.v h + 1);
    rewrite when_ (SZ.v tid == 0) (u_attention_owned output x (h + 1) 0) as
      when_ (SZ.v tid == 0) (u_attention_owned output x next 0);
    head := next;
  };
  let finished = !head;
  dec_source_loop_end finished 8sz;
  rewrite when_ (SZ.v tid == 0) (u_attention_owned output x finished 0) as
    when_ (SZ.v tid == 0) (u_attention_owned output x 8 0);
  u_owner_attention_padding output tid #x;
}

inline_for_extraction noextract
fn u_attention_shared_thread (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (#x:erased (lseq BF16.t 32768)) (#old:erased (lseq BF16.t 16384)) (#fraction:perm)
  (sh:SH.c_shmems opt_attention_shared_desc) (bid:szlt 1) (tid:szlt 32) ()
  requires gpu ** pure (SH.c_shmems_inv sh) **
    u_attention_kpre input output (opt_attention_shared sh) x old fraction tid **
    thread_id 32 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 32) ** Barrier.barrier_state 0
  ensures gpu ** u_attention_kpost input output (opt_attention_shared sh) x fraction tid **
    thread_id 32 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 32) ** Barrier.barrier_state 0
{
  unfold u_attention_kpre input output (opt_attention_shared sh) x old fraction tid;
  u_attention_body input output (opt_attention_shared sh) tid #x #old #(fraction /. 32.0R);
  fold u_attention_kpost input output (opt_attention_shared sh) x fraction tid;
}

inline_for_extraction noextract
let u_attention_descriptor
  (input:larray BF16.t 32768{KA.is_global_array input})
  (output:larray BF16.t 16384{KA.is_global_array output})
  (#x:erased (lseq BF16.t 32768)) (#old:erased (lseq BF16.t 16384)) (#fraction:perm)
  : k:KS.kernel_desc (u_attention_pre input output x old fraction) (u_attention_post input output x fraction){
    SZ.v k.nblk == 1 /\ k.nthr == 32sz /\ k.shmems_desc == opt_attention_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 32sz /\ k.barrier_count bid == 0)} =
  let pre = u_attention_pre input output x old fraction in
  let post = u_attention_post input output x fraction in {
    nblk = 1sz; nthr = 32sz; shmems_desc = opt_attention_shared_desc;
    kpre = (fun sh _ tid -> u_attention_kpre input output (opt_attention_shared sh) x old fraction tid);
    kpost = (fun sh _ tid -> u_attention_kpost input output (opt_attention_shared sh) x fraction tid);
    barrier_contract = (fun _ _ -> Barrier.empty_contract 32);
    barrier_count = (fun _ -> 0);
    barrier_ok = (fun _ _ -> Barrier.empty_barrier_transform 32);
    f = u_attention_shared_thread input output #x #old #fraction;
    frame = emp; block_pre = (fun _ -> pre); block_post = (fun _ -> post);
    setup = opt_single_block_setup pre; teardown = opt_single_block_teardown post;
    block_frame = (fun _ _ -> emp);
    block_setup = u_attention_block_setup input output #x #old #fraction;
    block_teardown = u_attention_block_teardown input output #x #fraction;
    block_pre_sendable = (fun _ -> (solve <: is_send_across gpu_of (u_attention_pre input output x old fraction)));
    block_post_sendable = (fun _ -> (solve <: is_send_across gpu_of (u_attention_post input output x fraction)));
    kpre_sendable = (fun sh pf _ tid -> u_attention_kpre_send input output #x #old #fraction sh pf tid);
    kpost_sendable = (fun sh pf _ tid -> u_attention_kpost_send input output #x #fraction sh pf tid)
  }

inline_for_extraction noextract
ghost
fn u_attention_before
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) ()
  requires u_stage_resources weights_address prompt_address a weights prompt v 4
  ensures u_attention_pre a.p_rotated a.p_attention (reveal v).v_rotated (reveal v).v_attention 1.0R **
    u_attention_stage_frame weights_address prompt_address a weights prompt v
{
  unfold u_stage_resources weights_address prompt_address a weights prompt v 4;
  unfold opt_block_storage a v;
  fold u_attention_pre a.p_rotated a.p_attention (reveal v).v_rotated (reveal v).v_attention 1.0R;
  fold u_attention_stage_frame weights_address prompt_address a weights prompt v;
}

inline_for_extraction noextract
ghost
fn u_attention_after
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) ()
  requires u_attention_post a.p_rotated a.p_attention (reveal v).v_rotated 1.0R **
    u_attention_stage_frame weights_address prompt_address a weights prompt v
  ensures u_pending_stage weights_address prompt_address a weights prompt 5
{
  unfold u_attention_post a.p_rotated a.p_attention (reveal v).v_rotated 1.0R;
  unfold u_attention_final a.p_attention (reveal v).v_rotated;
  with ys. assert a.p_attention |-> ys;
  unfold u_attention_stage_frame weights_address prompt_address a weights prompt v;
  let next = hide {(reveal v) with v_attention = reveal ys};
  assert pure (u_block_stage weights prompt next 5);
  fold opt_block_storage a next;
  fold u_stage_resources weights_address prompt_address a weights prompt next 5;
  fold u_pending_stage weights_address prompt_address a weights prompt 5;
}

inline_for_extraction noextract
ghost
fn opt_pending_setup
  (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (#run:erased (opt_observation a)) ()
  norewrite
  requires opt_pending_pre run pre
  ensures (forall+ (_:natlt 1). opt_pending_block_pre family run) ** opt_pending_frame family run
{
  unfold opt_pending_pre run pre;
  with x. assert pre x;
  let code = family.pending_code x;
  Observed.update run (Some x);
  Observed.take_snapshot run (Some x);
  fold opt_seen #a run x;
  let setup = code.setup;
  setup ();
  forevery_singleton_elim' #(natlt code.nblk) code.block_pre 0;
  rewrite code.frame as (family.pending_code x).frame;
  rewrite code.block_pre 0 as (family.pending_code x).block_pre 0;
  fold opt_pending_frame family run;
  fold opt_pending_block_pre family run;
  forevery_singleton_intro' #(natlt 1) (fun _ -> opt_pending_block_pre family run) 0;
}

inline_for_extraction noextract
ghost
fn opt_pending_teardown
  (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (#run:erased (opt_observation a)) ()
  norewrite
  requires (forall+ (_:natlt 1). opt_pending_block_post family run) ** opt_pending_frame family run
  ensures exists* (x:a). post x
{
  forevery_singleton_elim' #(natlt 1) (fun _ -> opt_pending_block_post family run) 0;
  unfold opt_pending_block_post family run;
  unfold opt_pending_frame family run;
  with x. assert Observed.pts_to run #1.0R (Some x);
  with y. assert opt_seen #a run y;
  unfold opt_seen #a run y;
  Observed.recall_snapshot #(option a) #(opt_once_order #a) run #1.0R
    #(Some (FStar.Ghost.reveal #a x)) #(Some (FStar.Ghost.reveal #a y));
  assert pure (x == y);
  let code = family.pending_code x;
  rewrite (family.pending_code y).block_post 0 as code.block_post 0;
  rewrite (family.pending_code x).frame as code.frame;
  forevery_singleton_intro' #(natlt code.nblk) code.block_post 0;
  let teardown = code.teardown;
  teardown ();
  drop_ (Observed.pts_to run #1.0R (Some x));
  drop_ (Observed.snapshot #(option a) #(opt_once_order #a) run (Some (FStar.Ghost.reveal #a y)));
}

inline_for_extraction noextract
ghost
fn opt_snapshot_copies (#a:Type0) (run:opt_observation a) (#observed:erased a)
  (count:nat)
  norewrite
  preserves opt_seen #a run observed
  ensures opt_copies #a run observed count
{
  forevery_emp_intro (natlt count);
  forevery_map_extra #(natlt count) (opt_seen #a run observed)
    (fun _ -> emp) (fun _ -> opt_seen #a run observed)
    fn _ { dup (opt_seen #a run observed) (); };
  fold opt_copies #a run observed count;
}

inline_for_extraction noextract
ghost
fn opt_pending_block_setup
  (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (#run:erased (opt_observation a))
  (sh:SH.c_shmems family.pending_shared) (_:natlt 1) ()
  norewrite
  requires SH.live_c_shmems sh ** opt_pending_block_pre family run
  ensures
    (forall+ (tid:natlt family.pending_threads).
      opt_observed_thread run (fun x -> opt_pending_source_pre family x sh tid)) **
    opt_pending_block_frame family run sh
{
  unfold opt_pending_block_pre family run;
  with x. assert opt_seen #a run x ** (family.pending_code x).block_pre 0;
  let code = family.pending_code x;
  rewrite (family.pending_code x).block_pre 0 as code.block_pre 0;
  rewrite SH.live_c_shmems #family.pending_shared sh as
    SH.live_c_shmems #code.shmems_desc sh;
  let setup = code.block_setup sh 0;
  setup ();
  opt_snapshot_copies run #x family.pending_threads;
  unfold opt_copies run x family.pending_threads;
  rewrite code.block_frame sh 0 as (family.pending_code x).block_frame sh 0;
  fold opt_pending_block_frame family run sh;
  forevery_rw_type (natlt code.nthr) (natlt family.pending_threads)
    (fun tid -> code.kpre sh 0 tid);
  forevery_zip #(natlt family.pending_threads)
    (fun _ -> opt_seen #a run x)
    (fun tid -> code.kpre sh 0 tid);
  forevery_map #(natlt family.pending_threads)
    (fun tid -> opt_seen #a run x ** code.kpre sh 0 tid)
    (fun tid -> opt_observed_thread run (fun y -> opt_pending_source_pre family y sh tid))
    fn tid {
      rewrite code.kpre sh 0 tid as opt_pending_source_pre family x sh tid;
      fold opt_observed_thread run (fun y -> opt_pending_source_pre family y sh tid);
    };
}

inline_for_extraction noextract
ghost
fn opt_observed_align (#a:Type0) (run:opt_observation a) (pred:a -> slprop)
  (#x:erased a)
  requires opt_seen #a run x ** opt_observed_thread run pred
  ensures pred x
{
  unfold opt_observed_thread run pred;
  with y. assert opt_seen #a run y ** pred y;
  unfold opt_seen #a run x;
  unfold opt_seen #a run y;
  Observed.snapshots_related #(option a) #(opt_once_order #a) run
    #(Some (FStar.Ghost.reveal #a x)) #(Some (FStar.Ghost.reveal #a y));
  assert pure (x == y);
  drop_ (Observed.snapshot #(option a) #(opt_once_order #a) run (Some (FStar.Ghost.reveal #a x)));
  drop_ (Observed.snapshot #(option a) #(opt_once_order #a) run (Some (FStar.Ghost.reveal #a y)));
  rewrite pred y as pred x;
}

inline_for_extraction noextract
ghost
fn opt_pending_block_teardown
  (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (#run:erased (opt_observation a))
  (sh:SH.c_shmems family.pending_shared) (_:natlt 1) ()
  norewrite
  requires
    (forall+ (tid:natlt family.pending_threads).
      opt_observed_thread run (fun x -> opt_pending_source_post family x sh tid)) **
    opt_pending_block_frame family run sh
  ensures SH.live_c_shmems sh ** opt_pending_block_post family run
{
  unfold opt_pending_block_frame family run sh;
  with x. assert opt_seen #a run x ** (family.pending_code x).block_frame sh 0;
  let code = family.pending_code x;
  opt_snapshot_copies run #x family.pending_threads;
  unfold opt_copies run x family.pending_threads;
  forevery_zip #(natlt family.pending_threads)
    (fun _ -> opt_seen #a run x)
    (fun tid -> opt_observed_thread run (fun y -> opt_pending_source_post family y sh tid));
  forevery_map #(natlt family.pending_threads)
    (fun tid -> opt_seen #a run x **
      opt_observed_thread run (fun y -> opt_pending_source_post family y sh tid))
    (fun tid -> opt_pending_source_post family x sh tid)
    (fun tid -> opt_observed_align run (fun y -> opt_pending_source_post family y sh tid) #x);
  forevery_ext #(natlt family.pending_threads)
    (fun tid -> opt_pending_source_post family x sh tid)
    (fun tid -> code.kpost sh 0 tid);
  forevery_rw_type (natlt family.pending_threads) (natlt code.nthr)
    (fun tid -> code.kpost sh 0 tid);
  rewrite (family.pending_code x).block_frame sh 0 as code.block_frame sh 0;
  let teardown = code.block_teardown sh 0;
  teardown ();
  rewrite SH.live_c_shmems #code.shmems_desc sh as
    SH.live_c_shmems #family.pending_shared sh;
  rewrite code.block_post 0 as (family.pending_code x).block_post 0;
  fold opt_pending_block_post family run;
}

inline_for_extraction noextract
fn opt_pending_device
  (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (#run:erased (opt_observation a))
  (sh:SH.c_shmems family.pending_shared) (bid:szlt 1) (tid:szlt family.pending_threads) ()
  requires gpu ** pure (SH.c_shmems_inv sh) **
    opt_observed_thread run (fun x -> opt_pending_source_pre family x sh tid) **
    thread_id family.pending_threads tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract family.pending_threads) ** Barrier.barrier_state 0
  ensures gpu **
    opt_observed_thread run (fun x -> opt_pending_source_post family x sh tid) **
    thread_id family.pending_threads tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract family.pending_threads) ** Barrier.barrier_state 0
{
  unfold opt_observed_thread run (fun x -> opt_pending_source_pre family x sh tid);
  with x. assert opt_seen #a run x ** opt_pending_source_pre family x sh tid;
  let code = family.pending_code x;
  assert pure (SZ.v bid == 0);
  rewrite opt_pending_source_pre family x sh tid as code.kpre sh 0 tid;
  rewrite each family.pending_threads as code.nthr;
  rewrite block_id 1 bid as block_id code.nblk 0;
  rewrite Barrier.barrier_tok (Barrier.empty_contract code.nthr) as
    Barrier.barrier_tok (code.barrier_contract 0 sh);
  let execute = code.f sh 0sz tid;
  execute ();
  rewrite code.kpost sh 0 tid as opt_pending_source_post family x sh tid;
  rewrite each code.nthr as family.pending_threads;
  rewrite block_id code.nblk 0 as block_id 1 bid;
  assert pure (code.barrier_contract 0 sh == Barrier.empty_contract family.pending_threads /\
    code.barrier_count 0 == 0);
  rewrite Barrier.barrier_tok (code.barrier_contract 0 sh) as
    Barrier.barrier_tok (Barrier.empty_contract family.pending_threads);
  rewrite Barrier.barrier_state (code.barrier_count 0) as Barrier.barrier_state 0;
  fold opt_observed_thread run (fun x -> opt_pending_source_post family x sh tid);
}

inline_for_extraction noextract
ghost
fn opt_pending_join_input (#a:Type0) (#run:erased (opt_observation a))
  (pre:a -> slprop) ()
  requires on gpu_loc (Observed.pts_to run #1.0R None) **
    on gpu_loc (exists* (x:a). pre x)
  ensures on gpu_loc (opt_pending_pre run pre)
{
  on_star_intro (Observed.pts_to run #1.0R None) (exists* (x:a). pre x);
  rewrite on gpu_loc (Observed.pts_to run #1.0R None ** (exists* (x:a). pre x))
    as on gpu_loc (opt_pending_pre run pre);
}

inline_for_extraction noextract
fn opt_launch_pending (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (stream:ST.stream_t) (#epoch:E.epoch_t)
  preserves cpu ** ST.stream_live stream
  requires E.epoch_live stream epoch **
    P.pledge0 (E.epoch_done stream epoch) (on gpu_loc (exists* (x:a). pre x))
  ensures E.epoch_live stream (E.epoch_next epoch) **
    P.pledge0 (E.epoch_done stream (E.epoch_next epoch)) (on gpu_loc (exists* (x:a). post x))
{
  let observation = Observed.alloc #(option a) #(opt_once_order #a) None;
  let run = hide observation;
  rewrite Observed.pts_to observation #1.0R None as Observed.pts_to run #1.0R None;
  placeless_on_intro (Observed.pts_to run #1.0R None) gpu_loc;
  P.return_pledge (E.epoch_done stream epoch) (on gpu_loc (Observed.pts_to run #1.0R None)) #solve;
  P.join_pledge (on gpu_loc (Observed.pts_to run #1.0R None)) (on gpu_loc (exists* (x:a). pre x));
  P.rewrite_pledge
    (on gpu_loc (Observed.pts_to run #1.0R None) ** on gpu_loc (exists* (x:a). pre x))
    (on gpu_loc (opt_pending_pre run pre))
    (opt_pending_join_input #a #run pre);
  KS.launch_kernel_full (opt_pending_descriptor family #run) stream #epoch;
}

inline_for_extraction noextract
ghost
fn u_on_exists_drop (#a:Type0) (post:slprop) ()
  requires on gpu_loc (exists* (_:a). post)
  ensures on gpu_loc post
{
  ()
}

inline_for_extraction noextract
fn u_launch_const_post (#a:Type0) (#pre:a -> slprop) (#post:slprop)
  (family:opt_pending_family a pre (fun _ -> post))
  (stream:ST.stream_t) (#epoch:E.epoch_t)
  preserves cpu ** ST.stream_live stream
  requires E.epoch_live stream epoch **
    P.pledge0 (E.epoch_done stream epoch) (on gpu_loc (exists* (x:a). pre x))
  ensures E.epoch_live stream (E.epoch_next epoch) **
    P.pledge0 (E.epoch_done stream (E.epoch_next epoch)) (on gpu_loc post)
{
  opt_launch_pending family stream #epoch;
  P.rewrite_pledge (on gpu_loc (exists* (_:a). post)) (on gpu_loc post)
    (u_on_exists_drop #a post);
}

inline_for_extraction noextract
ghost
fn u_stage_expand (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7) (a:opt_block_addresses)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stage:nat{stage <= 11}) (extra:slprop) ()
  requires on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt stage ** extra)
  ensures on gpu_loc (exists* (v:opt_block_values).
    u_stage_resources weights_address prompt_address a weights prompt v stage ** extra)
{
  ghost_impersonate gpu_loc
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt stage ** extra))
    (on gpu_loc (exists* (v:opt_block_values).
      u_stage_resources weights_address prompt_address a weights prompt v stage ** extra))
    fn () {
      on_elim (u_pending_stage weights_address prompt_address a weights prompt stage ** extra);
      unfold u_pending_stage weights_address prompt_address a weights prompt stage;
      with v. assert u_stage_resources weights_address prompt_address a weights prompt v stage;
      intro_exists_erased #opt_block_values
        (fun v -> u_stage_resources weights_address prompt_address a weights prompt v stage ** extra) v;
      on_intro (exists* (v:opt_block_values).
        u_stage_resources weights_address prompt_address a weights prompt v stage ** extra);
    };
}

inline_for_extraction noextract
fn u_launch_stage (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7) (a:opt_block_addresses)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stage:nat{stage < 11}) (extra:slprop)
  (family:opt_pending_family opt_block_values
    (fun v -> u_stage_resources weights_address prompt_address a weights prompt v stage)
    (fun _ -> u_pending_stage weights_address prompt_address a weights prompt (stage + 1)))
  (stream:ST.stream_t) (#epoch:E.epoch_t)
  preserves cpu ** ST.stream_live stream
  requires E.epoch_live stream epoch **
    P.pledge0 (E.epoch_done stream epoch)
      (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt stage ** extra))
  ensures E.epoch_live stream (E.epoch_next epoch) **
    P.pledge0 (E.epoch_done stream (E.epoch_next epoch))
      (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt (stage + 1) ** extra))
{
  P.rewrite_pledge
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt stage ** extra))
    (on gpu_loc (exists* (v:opt_block_values).
      u_stage_resources weights_address prompt_address a weights prompt v stage ** extra))
    (u_stage_expand weights_address prompt_address a #weights #prompt stage extra);
  u_launch_const_post (u_frame_family family extra) stream #epoch;
}

inline_for_extraction noextract
fn u_rms_row_initialized (#n #nw #ny:erased pos)
  (input:larray BF16.t n) (weights:larray BF16.t nw) (output:larray BF16.t ny)
  (start:sz) (width:szp{dec_rms_width width /\ start + width <= n})
  (weight_start:sz{weight_start + width <= nw})
  (output_start:sz{output_start + width <= ny})
  (lanes:szp{SZ.v lanes == dec_rms_lanes width})
  (outer:szp{SZ.v outer == dec_rms_outer_width width})
  (partial:larray F32.t lanes) (warp_sum:larray F32.t outer)
  (#x:erased (lseq BF16.t n)) (#w:erased (lseq BF16.t nw))
  (#old:erased (lseq BF16.t ny)) (#fx #fw:perm)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w
  requires output |-> old ** PA.pts_to_uninit partial lanes ** PA.pts_to_uninit warp_sum outer
  ensures exists* (after:lseq BF16.t ny).
    output |-> after ** PA.pts_to_uninit partial lanes ** PA.pts_to_uninit warp_sum outer **
    pure (u_rms_row_result x w old after start weight_start output_start width)
{
  PA.to_mask output;
  with before_raw. assert PA.pts_to_mask output before_raw (fun _ -> True);
  PA.pts_to_mask_len output;
  let before = FStar.Ghost.hide #(lseq (option BF16.t) ny)
    (FStar.Ghost.reveal #(S.seq (option BF16.t)) before_raw);
  rewrite PA.pts_to_mask output before_raw (fun _ -> True) as
    PA.pts_to_mask output before (fun _ -> True);
  let after_mask = u_rms_row_masked input weights output start width weight_start output_start
    lanes outer partial warp_sum #x #w #before #fx #fw;
  let zero = hide (fun (_:Shared.index width) -> 0.0R);
  source_bf16_region_initialized before after_mask output_start width
    (dec_spec_rms_norm zero zero) (u_rms_guard x w start weight_start width zero zero);
  PA.from_mask output;
  PA.pts_to_len output;
  with after_raw. assert output |-> after_raw;
  let after = FStar.Ghost.hide #(lseq BF16.t ny)
    (FStar.Ghost.reveal #(S.seq BF16.t) after_raw);
  rewrite output |-> after_raw as output |-> after;
  u_rms_transfer x w old after before after_mask start weight_start output_start width;
  ()
}

inline_for_extraction noextract
fn u_rmsnorm_body (#nw:erased pos) (rows:szp{rows <= 8})
  (input output:larray BF16.t (rows * 1024))
  (weights:larray BF16.t nw) (offset:sz{offset + 1024 <= nw})
  (#x #old:erased (lseq BF16.t (rows * 1024))) (#w:erased (lseq BF16.t nw))
  (#fx #fw:perm)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w
  requires output |-> old
  ensures exists* (ys:lseq BF16.t (rows * 1024)).
    output |-> ys ** pure (u_rms_prefix x w ys offset rows)
{
  let mut partial : PA.array F32.t = [| 256sz |];
  let mut warp_sum : PA.array F32.t = [| 8sz |];
  let mut row = 0sz;
  while (!row <^ rows)
    invariant exists* (r:sz{r <= rows}) (ys:lseq BF16.t (rows * 1024)).
      row |-> r ** output |-> ys **
      PA.pts_to_uninit partial 256 ** PA.pts_to_uninit warp_sum 8 **
      pure (u_rms_prefix x w ys offset r)
    decreases (rows - !row)
  {
    let r = !row;
    with before. assert output |-> before;
    u_rms_row_initialized input weights output (r *^ 1024sz) 1024sz offset (r *^ 1024sz)
      256sz 8sz partial warp_sum #x #w #before #fx #fw;
    with after. assert output |-> after;
    u_rms_row_advance x w before after offset r;
    row := r +^ 1sz;
  };
  dec_source_loop_end !row rows;
  ()
}

inline_for_extraction noextract
ghost
fn u_rms_storage_open (a:opt_block_addresses) (#v:erased opt_block_values) (slot:nat{slot < 3})
  requires opt_block_storage a v
  ensures u_rms_input a slot |-> u_rms_input_values v slot **
    u_rms_output a slot |-> u_rms_output_values v slot ** u_rms_frame a v slot
{
  unfold opt_block_storage a v;
  if (slot = 0) {
    fold u_rms_frame a v 0;
    rewrite u_rms_frame a v 0 as u_rms_frame a v slot;
    rewrite a.p_embedding |-> (reveal v).v_embedding as u_rms_input a slot |-> u_rms_input_values v slot;
    rewrite a.p_normalized0 |-> (reveal v).v_normalized0 as u_rms_output a slot |-> u_rms_output_values v slot;
  } else if (slot = 1) {
    fold u_rms_frame a v 1;
    rewrite u_rms_frame a v 1 as u_rms_frame a v slot;
    rewrite a.p_residual |-> (reveal v).v_residual as u_rms_input a slot |-> u_rms_input_values v slot;
    rewrite a.p_postnorm |-> (reveal v).v_postnorm as u_rms_output a slot |-> u_rms_output_values v slot;
  } else {
    assert pure (slot == 2);
    fold u_rms_frame a v 2;
    rewrite u_rms_frame a v 2 as u_rms_frame a v slot;
    rewrite a.p_hidden |-> (reveal v).v_hidden as u_rms_input a slot |-> u_rms_input_values v slot;
    rewrite a.p_normalized1 |-> (reveal v).v_normalized1 as u_rms_output a slot |-> u_rms_output_values v slot;
  };
}

inline_for_extraction noextract
ghost
fn u_rms_storage_close (a:opt_block_addresses) (#v:erased opt_block_values)
  (slot:nat{slot < 3}) (#ys:erased (lseq BF16.t 8192))
  requires u_rms_input a slot |-> u_rms_input_values v slot **
    u_rms_output a slot |-> ys ** u_rms_frame a v slot
  ensures opt_block_storage a (u_rms_updated v slot ys)
{
  if (slot = 0) {
    rewrite each slot as 0;
    unfold u_rms_frame a v 0;
    fold opt_block_storage a (u_rms_updated v 0 ys);
    rewrite opt_block_storage a (u_rms_updated v 0 ys) as opt_block_storage a (u_rms_updated v slot ys);
  } else if (slot = 1) {
    rewrite each slot as 1;
    unfold u_rms_frame a v 1;
    fold opt_block_storage a (u_rms_updated v 1 ys);
    rewrite opt_block_storage a (u_rms_updated v 1 ys) as opt_block_storage a (u_rms_updated v slot ys);
  } else {
    assert pure (slot == 2);
    rewrite each slot as 2;
    unfold u_rms_frame a v 2;
    fold opt_block_storage a (u_rms_updated v 2 ys);
    rewrite opt_block_storage a (u_rms_updated v 2 ys) as opt_block_storage a (u_rms_updated v slot ys);
  };
}

inline_for_extraction noextract
fn u_rms_stage_body (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7) (a:opt_block_addresses)
  (slot:nat{slot < 3}) (offset:sz{SZ.v offset == u_rms_offset slot})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) ()
  preserves gpu
  requires u_stage_resources weights_address prompt_address a weights prompt v (u_rms_step slot)
  ensures u_pending_stage weights_address prompt_address a weights prompt (u_rms_step slot + 1)
{
  unfold u_stage_resources weights_address prompt_address a weights prompt v (u_rms_step slot);
  u_rms_storage_open a #v slot;
  let input = u_rms_input a slot;
  let output = u_rms_output a slot;
  rewrite u_rms_input a slot |-> u_rms_input_values v slot as input |-> u_rms_input_values v slot;
  rewrite u_rms_output a slot |-> u_rms_output_values v slot as output |-> u_rms_output_values v slot;
  u_rmsnorm_body 8sz input output weights_address offset
    #(hide (u_rms_input_values v slot)) #(hide (u_rms_output_values v slot)) #weights #1.0R #1.0R;
  with ys. assert output |-> ys;
  u_rms_stage_advance weights prompt v slot ys;
  rewrite input |-> u_rms_input_values v slot as
    u_rms_input a slot |-> u_rms_input_values v slot;
  rewrite output |-> ys as u_rms_output a slot |-> ys;
  u_rms_storage_close a #v slot #ys;
  fold u_stage_resources weights_address prompt_address a weights prompt
    (u_rms_updated v slot ys) (u_rms_step slot + 1);
  fold u_pending_stage weights_address prompt_address a weights prompt (u_rms_step slot + 1);
}

inline_for_extraction noextract
ghost
fn u_queue_step (stream:ST.stream_t) (#start:E.epoch_t) (index:nat) (post:slprop)
  requires E.epoch_live stream (E.epoch_next (start + index)) **
    P.pledge0 (E.epoch_done stream (E.epoch_next (start + index))) post
  ensures E.epoch_live stream (start + (index + 1)) **
    P.pledge0 (E.epoch_done stream (start + (index + 1))) post
{
  rewrite E.epoch_live stream (E.epoch_next (start + index)) as
    E.epoch_live stream (start + (index + 1));
  rewrite P.pledge0 (E.epoch_done stream (E.epoch_next (start + index))) post as
    P.pledge0 (E.epoch_done stream (start + (index + 1))) post;
}

inline_for_extraction noextract
fn u_first_block_launches
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (embedding_weights:larray BF16.t emb_embedding_count{
    embedding_weights == u_weight_window weights_address 0 emb_embedding_count})
  (layer_weights:larray BF16.t dec_weight_elements{
    layer_weights == u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (extra:slprop) (stream:ST.stream_t) (#start:E.epoch_t)
  preserves cpu ** ST.stream_live stream
  requires E.epoch_live stream start **
    P.pledge0 (E.epoch_done stream start)
      (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 0 ** extra))
  ensures E.epoch_live stream (start + 11) **
    P.pledge0 (E.epoch_done stream (start + 11))
      (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 11 ** extra))
{
  u_source_offsets ();
  let embedding = u_embedding_stage weights_address prompt_address a embedding_weights #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 0 extra embedding stream #start;

  let input_norm = u_rms_stage weights_address prompt_address a 0 (SZ.uint32_to_sizet 155583488ul) #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 1 extra input_norm stream #(start + 1);
  u_queue_step stream #start 1
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 2 ** extra));

  let qkv = u_linear_stage weights_address prompt_address a layer_weights
    0 1024sz 4096sz 1sz 2048sz false #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 2 extra qkv stream #(start + 2);
  u_queue_step stream #start 2
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 3 ** extra));

  let rotary = u_rope_stage weights_address prompt_address a layer_weights #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 3 extra rotary stream #(start + 3);
  u_queue_step stream #start 3
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 4 ** extra));

  let attention = u_attention_stage weights_address prompt_address a #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 4 extra attention stream #(start + 4);
  u_queue_step stream #start 4
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 5 ** extra));

  let residual = u_linear_stage weights_address prompt_address a layer_weights
    1 2048sz 1024sz 0sz (SZ.uint32_to_sizet 4196352ul) true #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 5 extra residual stream #(start + 5);
  u_queue_step stream #start 5
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 6 ** extra));

  let post_norm = u_rms_stage weights_address prompt_address a 1 (SZ.uint32_to_sizet 155584512ul) #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 6 extra post_norm stream #(start + 6);
  u_queue_step stream #start 6
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 7 ** extra));

  let gate_up = u_linear_stage weights_address prompt_address a layer_weights
    2 1024sz 6144sz 2sz (SZ.uint32_to_sizet 6293760ul) false #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 7 extra gate_up stream #(start + 7);
  u_queue_step stream #start 7
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 8 ** extra));

  let silu = u_silu_stage weights_address prompt_address a #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 8 extra silu stream #(start + 8);
  u_queue_step stream #start 8
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 9 ** extra));

  let down = u_linear_stage weights_address prompt_address a layer_weights
    3 3072sz 1024sz 0sz (SZ.uint32_to_sizet 12585216ul) true #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 9 extra down stream #(start + 9);
  u_queue_step stream #start 9
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 10 ** extra));

  let next_norm = u_rms_stage weights_address prompt_address a 2 (SZ.uint32_to_sizet 171314432ul) #weights #prompt;
  u_launch_stage weights_address prompt_address a #weights #prompt 10 extra next_norm stream #(start + 10);
  u_queue_step stream #start 10
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 11 ** extra));
}

inline_for_extraction noextract
fn rec opt_buffer_data_ptr
  (stream:Stream.stream_t) (buffers:list (RT.tensor BF16.t))
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
        opt_buffer_data_ptr stream rest (index - 1)
      };
      fold RuntimeHelpers.all_owners stream buffers;
      address
    }
  }
}

inline_for_extraction noextract
fn opt_prefix_data_ptr (stream:Stream.stream_t)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (index:nat{index < 12 /\ index < List.length buffers.bf16_buffers})
  preserves KL.cpu ** opt_allocated_owners stream buffers
  returns address:PA.array BF16.t
  ensures pure (
    address == RT.data (List.index buffers.bf16_buffers index) /\
    PA.length address == RT.numel (opt_prefix_shape index) /\
    Kuiper.Array.Core.is_global_array address)
{
  opt_prefix_metadata buffers index;
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  unfold opt_allocated_owners stream buffers;
  let address = opt_buffer_data_ptr stream buffers.bf16_buffers index;
  fold opt_allocated_owners stream buffers;
  address
}
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn u_read_block_addresses (stream:ST.stream_t)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  preserves cpu ** opt_allocated_owners stream buffers
  returns a:opt_block_addresses
  ensures pure (a == u_allocated_addresses buffers /\ opt_block_global a)
{
  u_prefix_list_partition buffers;
  let embedding = opt_prefix_data_ptr stream buffers 0;
  let rope = opt_prefix_data_ptr stream buffers 1;
  let normalized0 = opt_prefix_data_ptr stream buffers 2;
  let qkv = opt_prefix_data_ptr stream buffers 3;
  let rotated = opt_prefix_data_ptr stream buffers 4;
  let attention = opt_prefix_data_ptr stream buffers 5;
  let residual = opt_prefix_data_ptr stream buffers 6;
  let postnorm = opt_prefix_data_ptr stream buffers 7;
  let gateup = opt_prefix_data_ptr stream buffers 8;
  let activated = opt_prefix_data_ptr stream buffers 9;
  let hidden = opt_prefix_data_ptr stream buffers 10;
  let normalized1 = opt_prefix_data_ptr stream buffers 11;
  {
    p_embedding = embedding; p_rope = rope; p_normalized0 = normalized0;
    p_qkv = qkv; p_rotated = rotated; p_attention = attention;
    p_residual = residual; p_postnorm = postnorm; p_gateup = gateup;
    p_activated = activated; p_hidden = hidden; p_normalized1 = normalized1
  }
}
#pop-options
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
ghost
fn u_prefix_storage_open
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (a:opt_block_addresses{a == u_allocated_addresses buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  requires opt_allocated_storage buffers (u_input_storage weights_address prompt_address weights prompt)
  ensures on gpu_loc
    (u_pending_stage weights_address prompt_address a weights prompt 0 ** u_remaining_storage buffers)
{
  u_prefix_list_partition buffers;
  let b0 = List.index buffers.bf16_buffers 0;
  let b1 = List.index buffers.bf16_buffers 1;
  let b2 = List.index buffers.bf16_buffers 2;
  let b3 = List.index buffers.bf16_buffers 3;
  let b4 = List.index buffers.bf16_buffers 4;
  let b5 = List.index buffers.bf16_buffers 5;
  let b6 = List.index buffers.bf16_buffers 6;
  let b7 = List.index buffers.bf16_buffers 7;
  let b8 = List.index buffers.bf16_buffers 8;
  let b9 = List.index buffers.bf16_buffers 9;
  let b10 = List.index buffers.bf16_buffers 10;
  let b11 = List.index buffers.bf16_buffers 11;
  let rest = u_remaining_tensors buffers;
  unfold opt_allocated_storage buffers (u_input_storage weights_address prompt_address weights prompt);
  rewrite RH.all_contents buffers.bf16_buffers as
    RH.all_contents (b0::b1::b2::b3::b4::b5::b6::b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b0::b1::b2::b3::b4::b5::b6::b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b1::b2::b3::b4::b5::b6::b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b2::b3::b4::b5::b6::b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b3::b4::b5::b6::b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b4::b5::b6::b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b5::b6::b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b6::b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b7::b8::b9::b10::b11::rest);
  unfold RH.all_contents (b8::b9::b10::b11::rest);
  unfold RH.all_contents (b9::b10::b11::rest);
  unfold RH.all_contents (b10::b11::rest);
  unfold RH.all_contents (b11::rest);
  rewrite RH.all_contents rest as RH.all_contents (u_remaining_tensors buffers);
  fold u_remaining_storage buffers;
  ghost_impersonate gpu_loc
    (RT.contents b0 ** RT.contents b1 ** RT.contents b2 ** RT.contents b3 **
      RT.contents b4 ** RT.contents b5 ** RT.contents b6 ** RT.contents b7 **
      RT.contents b8 ** RT.contents b9 ** RT.contents b10 ** RT.contents b11 **
      u_input_storage weights_address prompt_address weights prompt)
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 0))
    fn () {
      unfold RT.contents b0; unfold RT.contents b1; unfold RT.contents b2; unfold RT.contents b3;
      unfold RT.contents b4; unfold RT.contents b5; unfold RT.contents b6; unfold RT.contents b7;
      unfold RT.contents b8; unfold RT.contents b9; unfold RT.contents b10; unfold RT.contents b11;
      with e0. assert on gpu_loc (PA.pts_to (RT.data b0) e0);
      with e1. assert on gpu_loc (PA.pts_to (RT.data b1) e1);
      with e2. assert on gpu_loc (PA.pts_to (RT.data b2) e2);
      with e3. assert on gpu_loc (PA.pts_to (RT.data b3) e3);
      with e4. assert on gpu_loc (PA.pts_to (RT.data b4) e4);
      with e5. assert on gpu_loc (PA.pts_to (RT.data b5) e5);
      with e6. assert on gpu_loc (PA.pts_to (RT.data b6) e6);
      with e7. assert on gpu_loc (PA.pts_to (RT.data b7) e7);
      with e8. assert on gpu_loc (PA.pts_to (RT.data b8) e8);
      with e9. assert on gpu_loc (PA.pts_to (RT.data b9) e9);
      with e10. assert on gpu_loc (PA.pts_to (RT.data b10) e10);
      with e11. assert on gpu_loc (PA.pts_to (RT.data b11) e11);
      on_elim (PA.pts_to (RT.data b0) e0); on_elim (PA.pts_to (RT.data b1) e1);
      on_elim (PA.pts_to (RT.data b2) e2); on_elim (PA.pts_to (RT.data b3) e3);
      on_elim (PA.pts_to (RT.data b4) e4); on_elim (PA.pts_to (RT.data b5) e5);
      on_elim (PA.pts_to (RT.data b6) e6); on_elim (PA.pts_to (RT.data b7) e7);
      on_elim (PA.pts_to (RT.data b8) e8); on_elim (PA.pts_to (RT.data b9) e9);
      on_elim (PA.pts_to (RT.data b10) e10); on_elim (PA.pts_to (RT.data b11) e11);
      PA.pts_to_len (RT.data b0); PA.pts_to_len (RT.data b1);
      PA.pts_to_len (RT.data b2); PA.pts_to_len (RT.data b3);
      PA.pts_to_len (RT.data b4); PA.pts_to_len (RT.data b5);
      PA.pts_to_len (RT.data b6); PA.pts_to_len (RT.data b7);
      PA.pts_to_len (RT.data b8); PA.pts_to_len (RT.data b9);
      PA.pts_to_len (RT.data b10); PA.pts_to_len (RT.data b11);
      let v = Ghost.hide #opt_block_values {
        v_embedding = Ghost.reveal #(S.seq BF16.t) e0;
        v_rope = Ghost.reveal #(S.seq BF16.t) e1;
        v_normalized0 = Ghost.reveal #(S.seq BF16.t) e2;
        v_qkv = Ghost.reveal #(S.seq BF16.t) e3;
        v_rotated = Ghost.reveal #(S.seq BF16.t) e4;
        v_attention = Ghost.reveal #(S.seq BF16.t) e5;
        v_residual = Ghost.reveal #(S.seq BF16.t) e6;
        v_postnorm = Ghost.reveal #(S.seq BF16.t) e7;
        v_gateup = Ghost.reveal #(S.seq BF16.t) e8;
        v_activated = Ghost.reveal #(S.seq BF16.t) e9;
        v_hidden = Ghost.reveal #(S.seq BF16.t) e10;
        v_normalized1 = Ghost.reveal #(S.seq BF16.t) e11
      };
      rewrite each RT.data b0 as a.p_embedding; rewrite each RT.data b1 as a.p_rope;
      rewrite each RT.data b2 as a.p_normalized0; rewrite each RT.data b3 as a.p_qkv;
      rewrite each RT.data b4 as a.p_rotated; rewrite each RT.data b5 as a.p_attention;
      rewrite each RT.data b6 as a.p_residual; rewrite each RT.data b7 as a.p_postnorm;
      rewrite each RT.data b8 as a.p_gateup; rewrite each RT.data b9 as a.p_activated;
      rewrite each RT.data b10 as a.p_hidden; rewrite each RT.data b11 as a.p_normalized1;
      fold opt_block_storage a v;
      unfold u_input_storage weights_address prompt_address weights prompt;
      on_elim (PA.pts_to weights_address weights);
      on_elim (PA.pts_to prompt_address prompt);
      fold u_stage_resources weights_address prompt_address a weights prompt v 0;
      fold u_pending_stage weights_address prompt_address a weights prompt 0;
      on_intro (u_pending_stage weights_address prompt_address a weights prompt 0);
    };
  placeless_on_intro (u_remaining_storage buffers) #(u_remaining_placeless buffers) gpu_loc;
  on_star_intro (u_pending_stage weights_address prompt_address a weights prompt 0) (u_remaining_storage buffers);
}
#pop-options

inline_for_extraction noextract
ghost
fn u_publish_first_block (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  requires u_after_first_block weights_address prompt_address buffers weights prompt
  ensures source_after_first_block weights_address prompt_address buffers weights prompt
{
  unfold u_after_first_block weights_address prompt_address buffers weights prompt;
  ghost_impersonate gpu_loc
    (on gpu_loc (
      PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
      (exists* (v:opt_block_values).
        opt_block_storage (u_allocated_addresses buffers) v ** pure (u_first_block_observation weights prompt v))))
    (on gpu_loc (
      PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
      (exists* (v:opt_block_values).
        opt_block_storage (u_allocated_addresses buffers) v ** pure (source_first_block_observation weights prompt v))))
    fn () {
      on_elim (
        PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
        (exists* (v:opt_block_values).
          opt_block_storage (u_allocated_addresses buffers) v ** pure (u_first_block_observation weights prompt v)));
      with v. assert opt_block_storage (u_allocated_addresses buffers) v;
      u_public_block_observation weights prompt v;
      intro_exists_erased #opt_block_values
        (fun v -> opt_block_storage (u_allocated_addresses buffers) v **
          pure (source_first_block_observation weights prompt v)) v;
      on_intro (
        PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
        (exists* (v:opt_block_values).
          opt_block_storage (u_allocated_addresses buffers) v ** pure (source_first_block_observation weights prompt v)));
    };
  fold source_after_first_block weights_address prompt_address buffers weights prompt;
}

inline_for_extraction noextract
ghost
fn u_empty_mask_localize (weights:larray BF16.t opt_weight_count)
  (#contents:erased (lseq BF16.t opt_weight_count)) (ambient:slprop)
  preserves cpu ** ambient ** on gpu_loc (PA.pts_to weights contents)
  ensures PA.pts_to_mask weights #1.0R (u_metadata_mask contents) (fun _ -> False)
{
  ghost_impersonate gpu_loc (on gpu_loc (PA.pts_to weights contents))
    (on gpu_loc (PA.pts_to weights contents) **
      on gpu_loc (PA.pts_to_mask weights #1.0R (u_metadata_mask contents) (fun _ -> False)))
    fn () {
      on_elim (PA.pts_to weights contents);
      PA.to_mask weights;
      PA.mask_vext weights (u_metadata_mask contents);
      PA.split_mask weights #1.0R #(u_metadata_mask contents) #(fun _ -> True) (fun _ -> False);
      PA.mask_mext weights #1.0R #(u_metadata_mask contents)
        #(PA.mask_isect (fun _ -> True) (fun _ -> False)) (fun _ -> False);
      on_intro (PA.pts_to_mask weights #1.0R (u_metadata_mask contents) (fun _ -> False));
      PA.mask_mext weights #1.0R #(u_metadata_mask contents)
        #(PA.mask_diff (fun _ -> True) (fun _ -> False)) (fun _ -> True);
      PA.from_mask weights;
      with restored. assert PA.pts_to weights restored;
      assert pure (S.equal restored contents);
      S.lemma_eq_elim restored contents;
      on_intro (PA.pts_to weights contents);
    };
  placeless_on_elim (PA.pts_to_mask weights #1.0R (u_metadata_mask contents) (fun _ -> False)) gpu_loc;
}

inline_for_extraction noextract
fn u_materialize_weight_window
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (lo:HostSize.t) (hi:erased nat{HostSize.v lo <= hi /\ hi <= opt_weight_count})
  (#contents:erased (lseq BF16.t opt_weight_count)) (ambient:slprop)
  preserves cpu ** ambient ** on gpu_loc (PA.pts_to weights contents)
  returns window:PA.array BF16.t
  ensures rewrites_to window (u_weight_window weights (HostSize.v lo) hi)
  ensures pure (PA.length window == hi - HostSize.v lo /\ KA.is_global_array window)
{
  u_weight_window_shape weights (HostSize.v lo) hi;
  u_empty_mask_localize weights #contents ambient;
  let window = PA.sub weights #1.0R #(fun _ -> False) lo hi
    #(Ghost.hide (u_metadata_mask contents <: S.seq (option BF16.t)));
  PA.mask_mext weights (fun _ -> False);
  drop_ (PA.pts_to_mask weights #1.0R (u_metadata_mask contents) (fun _ -> False));
  PA.mask_mext window (fun _ -> False);
  drop_ (PA.pts_to_mask window #1.0R
    (S.slice (u_metadata_mask contents) (HostSize.v lo) hi) (fun _ -> False));
  window
}
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
ghost
fn u_finish_first_block (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (a:opt_block_addresses{a == u_allocated_addresses buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  requires on gpu_loc
    (u_pending_stage weights_address prompt_address a weights prompt 11 ** u_remaining_storage buffers)
  ensures u_after_first_block weights_address prompt_address buffers weights prompt
{
  on_star_elim (u_pending_stage weights_address prompt_address a weights prompt 11) (u_remaining_storage buffers);
  placeless_on_elim (u_remaining_storage buffers) #(u_remaining_placeless buffers) gpu_loc;
  ghost_impersonate gpu_loc
    (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 11))
    (on gpu_loc (
      PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
      (exists* (v:opt_block_values).
        opt_block_storage (u_allocated_addresses buffers) v **
        pure (u_first_block_observation weights prompt v))))
    fn () {
      on_elim (u_pending_stage weights_address prompt_address a weights prompt 11);
      unfold u_pending_stage weights_address prompt_address a weights prompt 11;
      with v. assert u_stage_resources weights_address prompt_address a weights prompt v 11;
      unfold u_stage_resources weights_address prompt_address a weights prompt v 11;
      u_block_numerical_from_stage weights prompt v;
      u_first_block_cache_binding v;
      rewrite opt_block_storage a v as opt_block_storage (u_allocated_addresses buffers) v;
      intro_exists_erased #opt_block_values
        (fun v -> opt_block_storage (u_allocated_addresses buffers) v **
          pure (u_first_block_observation weights prompt v)) v;
      on_intro (
        PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
        (exists* (v:opt_block_values).
          opt_block_storage (u_allocated_addresses buffers) v **
          pure (u_first_block_observation weights prompt v)));
    };
  fold u_after_first_block weights_address prompt_address buffers weights prompt;
}

inline_for_extraction noextract
fn first_block_refines
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t) (ambient:slprop)
  preserves cpu ** ST.stream_live stream ** RT.runtime stream ** ambient
  requires E.epoch_live stream start ** u_input_storage weights_address prompt_address weights prompt
  returns outcome:u_first_block_outcome
  ensures (
    match outcome with
    | UBlockFailed _ ->
      E.epoch_live stream start **
      P.pledge0 (E.epoch_done stream start) (u_input_storage weights_address prompt_address weights prompt)
    | UBlockEnqueued buffers ->
      E.epoch_live stream (start + 11) ** opt_allocated_owners stream buffers **
      P.pledge0 (E.epoch_done stream (start + 11))
        (source_after_first_block weights_address prompt_address buffers weights prompt))
{
  unfold u_input_storage weights_address prompt_address weights prompt;
  let embedding_weights = u_materialize_weight_window weights_address
    (HostSize.uint32_to_sizet 0ul) (hide emb_embedding_count) #weights ambient;
  let layer_weights = u_materialize_weight_window weights_address
    (HostSize.uint32_to_sizet 155583488ul) (hide (opt_layer0_start + dec_weight_elements)) #weights ambient;
  fold u_input_storage weights_address prompt_address weights prompt;
  P.return_pledge (E.epoch_done stream start)
    (u_input_storage weights_address prompt_address weights prompt) #solve;
  let allocated = opt_prepare_allocations stream #start
    (u_input_storage weights_address prompt_address weights prompt) ();
  match allocated {
    OptAllocationFailed error -> { UBlockFailed error }
    OptAllocated buffers -> {
      let a = u_read_block_addresses stream buffers;
      P.rewrite_pledge
        (opt_allocated_storage buffers (u_input_storage weights_address prompt_address weights prompt))
        (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 0 **
          u_remaining_storage buffers))
        (u_prefix_storage_open weights_address prompt_address buffers a #weights #prompt);
      u_first_block_launches weights_address prompt_address a embedding_weights layer_weights
        #weights #prompt (u_remaining_storage buffers) stream #start;
      P.rewrite_pledge
        (on gpu_loc (u_pending_stage weights_address prompt_address a weights prompt 11 **
          u_remaining_storage buffers))
        (u_after_first_block weights_address prompt_address buffers weights prompt)
        (u_finish_first_block weights_address prompt_address buffers a #weights #prompt);
      P.rewrite_pledge
        (u_after_first_block weights_address prompt_address buffers weights prompt)
        (source_after_first_block weights_address prompt_address buffers weights prompt)
        (u_publish_first_block weights_address prompt_address buffers #weights #prompt);
      UBlockEnqueued buffers
    }
  }
}
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
ghost
fn pg_take (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (#fraction:perm) (mask:Shared.index 256 -> prop)
  (index:Shared.index 256{mask index})
  requires pg_masked_storage buffers v fraction mask
  ensures PA.pts_to (pg_address buffers index) #fraction (pg_slot v index) **
    pg_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 256) i index))
{
  unfold pg_masked_storage buffers v fraction mask;
  forevery_remove' #(Shared.index 256) mask
    (fun i -> PA.pts_to (pg_address buffers i) #fraction (pg_slot v i)) index;
  rewrite (forall+ (i:Shared.index 256{mask i /\ ~(eq2 #(Shared.index 256) i index)}).
    PA.pts_to (pg_address buffers i) #fraction (pg_slot v i)) as
    pg_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 256) i index));
}

inline_for_extraction noextract
ghost
fn pg_put (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (#fraction:perm) (mask:Shared.index 256 -> prop)
  (index:Shared.index 256{mask index}) (#value:lseq BF16.t (pg_size index))
  requires PA.pts_to (pg_address buffers index) #fraction value **
    pg_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 256) i index))
  ensures pg_masked_storage buffers (pg_replace v index value) fraction mask
{
  let after = hide (pg_replace v index value);
  unfold pg_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 256) i index));
  forevery_ext
    (fun (i:Shared.index 256{mask i /\ ~(eq2 #(Shared.index 256) i index)}) ->
      PA.pts_to (pg_address buffers i) #fraction (pg_slot v i))
    (fun (i:Shared.index 256{mask i /\ ~(eq2 #(Shared.index 256) i index)}) ->
      PA.pts_to (pg_address buffers i) #fraction (pg_slot after i));
  rewrite PA.pts_to (pg_address buffers index) #fraction value as
    PA.pts_to (pg_address buffers index) #fraction (pg_slot after index);
  forevery_insert #(Shared.index 256) #(fun i -> mask i /\ ~(eq2 #(Shared.index 256) i index))
    (fun i -> PA.pts_to (pg_address buffers i) #fraction (pg_slot after i)) index;
  forevery_refine_ext #(Shared.index 256)
    #(fun i -> (mask i /\ ~(eq2 #(Shared.index 256) i index)) \/ eq2 #(Shared.index 256) index i) mask
    (fun i -> PA.pts_to (pg_address buffers i) #fraction (pg_slot after i));
  fold pg_masked_storage buffers after fraction mask;
  rewrite pg_masked_storage buffers after fraction mask as
    pg_masked_storage buffers (pg_replace v index value) fraction mask;
}

inline_for_extraction noextract
ghost
fn pg_split3 (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (a:Shared.index 256) (b:Shared.index 256{a <> b})
  (c:Shared.index 256{a <> c /\ b <> c})
  requires pg_storage buffers v 1.0R
  ensures PA.pts_to (pg_address buffers a) (pg_slot v a) **
    PA.pts_to (pg_address buffers b) (pg_slot v b) **
    PA.pts_to (pg_address buffers c) (pg_slot v c) **
    pg_masked_storage buffers v 1.0R (pg_mask3 a b c)
{
  rewrite pg_storage buffers v 1.0R as pg_masked_storage buffers v 1.0R (fun _ -> True);
  pg_take buffers #v #1.0R (fun _ -> True) a;
  rewrite pg_masked_storage buffers v 1.0R (fun i -> True /\ ~(eq2 #(Shared.index 256) i a)) as
    pg_masked_storage buffers v 1.0R (fun i -> i <> a);
  pg_take buffers #v #1.0R (fun i -> i <> a) b;
  rewrite pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ ~(eq2 #(Shared.index 256) i b)) as
    pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ i <> b);
  pg_take buffers #v #1.0R (fun i -> i <> a /\ i <> b) c;
  rewrite pg_masked_storage buffers v 1.0R (fun i -> (i <> a /\ i <> b) /\ ~(eq2 #(Shared.index 256) i c))
    as pg_masked_storage buffers v 1.0R (pg_mask3 a b c);
}

inline_for_extraction noextract
ghost
fn pg_join3 (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (a:Shared.index 256) (b:Shared.index 256{a <> b})
  (c:Shared.index 256{a <> c /\ b <> c})
  (#value:lseq BF16.t (pg_size c))
  requires PA.pts_to (pg_address buffers a) (pg_slot v a) **
    PA.pts_to (pg_address buffers b) (pg_slot v b) **
    PA.pts_to (pg_address buffers c) value **
    pg_masked_storage buffers v 1.0R (pg_mask3 a b c)
  ensures pg_storage buffers (pg_replace v c value) 1.0R
{
  rewrite pg_masked_storage buffers v 1.0R (pg_mask3 a b c) as
    pg_masked_storage buffers v 1.0R (fun i -> (i <> a /\ i <> b) /\ ~(eq2 #(Shared.index 256) i c));
  pg_put buffers #v #1.0R (fun i -> i <> a /\ i <> b) c #value;
  let after = hide (pg_replace v c value);
  rewrite pg_masked_storage buffers (pg_replace v c value) 1.0R (fun i -> i <> a /\ i <> b) as
    pg_masked_storage buffers after 1.0R (fun i -> i <> a /\ i <> b);
  rewrite PA.pts_to (pg_address buffers b) (pg_slot v b) as PA.pts_to (pg_address buffers b) (pg_slot after b);
  rewrite pg_masked_storage buffers after 1.0R (fun i -> i <> a /\ i <> b) as
    pg_masked_storage buffers after 1.0R (fun i -> i <> a /\ ~(eq2 #(Shared.index 256) i b));
  pg_put buffers #after #1.0R (fun i -> i <> a) b #(pg_slot after b);
  pg_replace_same after b;
  rewrite pg_masked_storage buffers (pg_replace after b (pg_slot after b)) 1.0R (fun i -> i <> a)
    as pg_masked_storage buffers after 1.0R (fun i -> i <> a);
  rewrite PA.pts_to (pg_address buffers a) (pg_slot v a) as PA.pts_to (pg_address buffers a) (pg_slot after a);
  rewrite pg_masked_storage buffers after 1.0R (fun i -> i <> a) as
    pg_masked_storage buffers after 1.0R (fun i -> True /\ ~(eq2 #(Shared.index 256) i a));
  pg_put buffers #after #1.0R (fun _ -> True) a #(pg_slot after a);
  pg_replace_same after a;
  rewrite pg_masked_storage buffers (pg_replace after a (pg_slot after a)) 1.0R (fun _ -> True)
    as pg_masked_storage buffers after 1.0R (fun _ -> True);
  unfold pg_masked_storage buffers after 1.0R (fun _ -> True);
  forevery_unrefine (fun (i:Shared.index 256) -> PA.pts_to (pg_address buffers i) (pg_slot after i));
  fold pg_storage buffers after 1.0R;
  rewrite pg_storage buffers after 1.0R as pg_storage buffers (pg_replace v c value) 1.0R;
}

inline_for_extraction noextract
ghost
fn pg_split2 (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (a:Shared.index 256) (b:Shared.index 256{a <> b})
  requires pg_storage buffers v 1.0R
  ensures PA.pts_to (pg_address buffers a) (pg_slot v a) **
    PA.pts_to (pg_address buffers b) (pg_slot v b) **
    pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ i <> b)
{
  rewrite pg_storage buffers v 1.0R as pg_masked_storage buffers v 1.0R (fun _ -> True);
  pg_take buffers #v #1.0R (fun _ -> True) a;
  rewrite pg_masked_storage buffers v 1.0R (fun i -> True /\ ~(eq2 #(Shared.index 256) i a)) as
    pg_masked_storage buffers v 1.0R (fun i -> i <> a);
  pg_take buffers #v #1.0R (fun i -> i <> a) b;
  rewrite pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ ~(eq2 #(Shared.index 256) i b)) as
    pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ i <> b);
}

inline_for_extraction noextract
ghost
fn pg_join2 (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (a:Shared.index 256) (b:Shared.index 256{a <> b}) (#value:lseq BF16.t (pg_size b))
  requires PA.pts_to (pg_address buffers a) (pg_slot v a) **
    PA.pts_to (pg_address buffers b) value **
    pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ i <> b)
  ensures pg_storage buffers (pg_replace v b value) 1.0R
{
  rewrite pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ i <> b) as
    pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ ~(eq2 #(Shared.index 256) i b));
  pg_put buffers #v #1.0R (fun i -> i <> a) b #value;
  let after = hide (pg_replace v b value);
  rewrite pg_masked_storage buffers (pg_replace v b value) 1.0R (fun i -> i <> a) as
    pg_masked_storage buffers after 1.0R (fun i -> i <> a);
  rewrite PA.pts_to (pg_address buffers a) (pg_slot v a) as PA.pts_to (pg_address buffers a) (pg_slot after a);
  rewrite pg_masked_storage buffers after 1.0R (fun i -> i <> a) as
    pg_masked_storage buffers after 1.0R (fun i -> True /\ ~(eq2 #(Shared.index 256) i a));
  pg_put buffers #after #1.0R (fun _ -> True) a #(pg_slot after a);
  pg_replace_same after a;
  rewrite pg_masked_storage buffers (pg_replace after a (pg_slot after a)) 1.0R (fun _ -> True)
    as pg_masked_storage buffers after 1.0R (fun _ -> True);
  unfold pg_masked_storage buffers after 1.0R (fun _ -> True);
  forevery_unrefine (fun (i:Shared.index 256) -> PA.pts_to (pg_address buffers i) (pg_slot after i));
  fold pg_storage buffers after 1.0R;
  rewrite pg_storage buffers after 1.0R as pg_storage buffers (pg_replace v b value) 1.0R;
}

inline_for_extraction noextract
ghost
fn rec pg_collect_tail (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (start:nat{start <= 256})
  preserves loc gpu_loc
  requires RH.all_contents (pg_tail buffers start)
  returns observed:erased (pg_partial_values start)
  ensures pg_collected buffers start observed
  decreases (256 - start)
{
  if (start = 256) {
    assert pure (pg_tail buffers start == []);
    rewrite RH.all_contents (pg_tail buffers start) as RH.all_contents #BF16.t [];
    unfold RH.all_contents #BF16.t [];
    let values = hide #(pg_partial_values start) (S.create #(S.seq BF16.t) 256 (S.empty #BF16.t));
    forevery_intro_empty (fun (i:Shared.index 256{start <= i}) ->
      PA.pts_to (pg_address buffers i) (S.index values i));
    fold pg_collected buffers start values;
    values
  } else {
    opt_allocation_count ();
    opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
    pg_list_drop_step buffers.bf16_buffers start;
    let handle = List.index buffers.bf16_buffers start;
    rewrite RH.all_contents (pg_tail buffers start) as RT.contents handle ** RH.all_contents (pg_tail buffers (start + 1));
    unfold RT.contents handle;
    with head_values. assert on gpu_loc (PA.pts_to (RT.data handle) head_values);
    on_elim (PA.pts_to (RT.data handle) head_values);
    rewrite each RT.data handle as pg_address buffers start;
    PA.pts_to_len (pg_address buffers start);
    let tail_values = pg_collect_tail buffers (start + 1);
    let values = hide #(pg_partial_values start) (S.upd #(S.seq BF16.t) tail_values start head_values);
    unfold pg_collected buffers (start + 1) tail_values;
    forevery_ext
      (fun (i:Shared.index 256{start + 1 <= i}) -> PA.pts_to (pg_address buffers i) (S.index tail_values i))
      (fun (i:Shared.index 256{start + 1 <= i}) -> PA.pts_to (pg_address buffers i) (S.index values i));
    rewrite PA.pts_to (pg_address buffers start) head_values as PA.pts_to (pg_address buffers start) (S.index values start);
    forevery_insert #(Shared.index 256) #(fun i -> start + 1 <= i)
      (fun i -> PA.pts_to (pg_address buffers i) (S.index values i)) start;
    forevery_refine_ext #(Shared.index 256) #(fun i -> start + 1 <= i \/ eq2 #(Shared.index 256) start i)
      (fun i -> start <= i) (fun i -> PA.pts_to (pg_address buffers i) (S.index values i));
    fold pg_collected buffers start values;
    values
  }
}
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
ghost
fn pg_insert_collected (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (index:Shared.index 256)
  requires PA.pts_to (pg_address buffers index) (pg_slot v index) ** pg_collected buffers (index + 1) v
  ensures pg_collected buffers index v
{
  unfold pg_collected buffers (index + 1) v;
  forevery_insert #(Shared.index 256) #(fun i -> index + 1 <= i)
    (fun i -> PA.pts_to (pg_address buffers i) (pg_slot v i)) index;
  forevery_refine_ext #(Shared.index 256) #(fun i -> index + 1 <= i \/ eq2 #(Shared.index 256) index i)
    (fun i -> index <= i) (fun i -> PA.pts_to (pg_address buffers i) (pg_slot v i));
  fold pg_collected buffers index v;
}
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
ghost
fn pg_join_first_block (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#first:opt_block_values) (#tail:pg_partial_values 12)
  requires opt_block_storage (u_allocated_addresses buffers) first ** pg_collected buffers 12 tail
  ensures pg_storage buffers (pg_initial_values first tail) 1.0R
{
  let v = hide (pg_initial_values first tail);
  unfold pg_collected buffers 12 tail;
  forevery_ext
    (fun (i:Shared.index 256{12 <= i}) -> PA.pts_to (pg_address buffers i) (S.index tail i))
    (fun (i:Shared.index 256{12 <= i}) -> PA.pts_to (pg_address buffers i) (S.index v i));
  fold pg_collected buffers 12 v;
  unfold opt_block_storage (u_allocated_addresses buffers) first;
  pg_insert_collected buffers #v 11;
  pg_insert_collected buffers #v 10;
  pg_insert_collected buffers #v 9;
  pg_insert_collected buffers #v 8;
  pg_insert_collected buffers #v 7;
  pg_insert_collected buffers #v 6;
  pg_insert_collected buffers #v 5;
  pg_insert_collected buffers #v 4;
  pg_insert_collected buffers #v 3;
  pg_insert_collected buffers #v 2;
  pg_insert_collected buffers #v 1;
  pg_insert_collected buffers #v 0;
  unfold pg_collected buffers 0 v;
  forevery_unrefine (fun (i:Shared.index 256) -> PA.pts_to (pg_address buffers i) (pg_slot v i));
  fold pg_storage buffers v 1.0R;
  rewrite pg_storage buffers v 1.0R as pg_storage buffers (pg_initial_values first tail) 1.0R;
}
#pop-options

inline_for_extraction noextract
ghost
fn pg_from_first_block (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  requires source_after_first_block weights_address prompt_address buffers weights prompt
  ensures pg_after_stage weights_address prompt_address buffers weights prompt 11
{
  let remaining_placeless = u_remaining_placeless buffers;
  ghost_impersonate gpu_loc
    (source_after_first_block weights_address prompt_address buffers weights prompt)
    (pg_after_stage weights_address prompt_address buffers weights prompt 11)
    fn () {
      unfold source_after_first_block weights_address prompt_address buffers weights prompt;
      unfold u_remaining_storage buffers;
      on_elim (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
        (exists* (v:opt_block_values). opt_block_storage (u_allocated_addresses buffers) v **
          pure (source_first_block_observation weights prompt v)));
      with first. assert opt_block_storage (u_allocated_addresses buffers) first;
      rewrite RH.all_contents (u_remaining_tensors buffers) as RH.all_contents (pg_tail buffers 12);
      let tail = pg_collect_tail buffers 12;
      pg_initial_progress weights prompt first tail;
      pg_join_first_block buffers #first #tail;
      fold pg_core weights_address buffers weights prompt (pg_initial_values first tail) 11;
      fold pg_pending weights_address buffers weights prompt 11;
      on_intro (pg_pending weights_address buffers weights prompt 11 **
        (PA.pts_to prompt_address prompt ** RT.contents buffers.token_buffer));
      fold pg_after_stage weights_address prompt_address buffers weights prompt 11;
    };
}

inline_for_extraction noextract
fn pg_rms_stage_body
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (layer:pg_remaining_layer) (step:nat{step = 4 \/ (step = 8 /\ layer < 27)})
  (input:larray BF16.t 8192{input == pg_address buffers (pg_rms_input_index layer step)})
  (output:larray BF16.t 8192{output == pg_address buffers (pg_target layer step)})
  (offset:sz{SZ.v offset == pg_rms_weight_offset layer step})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) ()
  norewrite
  preserves gpu
  requires pg_core weights_address buffers weights prompt v (2 + 9 * layer + step)
  ensures pg_pending weights_address buffers weights prompt (3 + 9 * layer + step)
{
  pg_rms_step_binding layer step;
  unfold pg_core weights_address buffers weights prompt v (2 + 9 * layer + step);
  let source = pg_rms_input_index layer step;
  let target = pg_target layer step;
  pg_split2 buffers #v source target;
  rewrite PA.pts_to (pg_address buffers source) (pg_slot v source) as input |-> pg_slot v source;
  rewrite PA.pts_to (pg_address buffers target) (pg_slot v target) as output |-> pg_slot v target;
  u_rmsnorm_body 8sz input output weights_address offset
    #(hide (pg_slot v source)) #(hide (pg_slot v target)) #weights #1.0R #1.0R;
  with ys. assert output |-> ys;
  rewrite input |-> pg_slot v source as PA.pts_to (pg_address buffers source) (pg_slot v source);
  rewrite output |-> ys as PA.pts_to (pg_address buffers target) ys;
  pg_join2 buffers #v source target #ys;
  assert pure (pg_step_fact weights (pg_replace v target ys) layer step);
  pg_stage_advance weights prompt v layer step ys;
  fold pg_core weights_address buffers weights prompt (pg_replace v target ys) (3 + 9 * layer + step);
  fold pg_pending weights_address buffers weights prompt (3 + 9 * layer + step);
}

inline_for_extraction noextract
fn pg_silu_stage_body
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 49152{input == pg_address buffers (pg_index layer 8)})
  (output:larray BF16.t 24576{output == pg_address buffers (pg_index layer 9)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) ()
  norewrite
  preserves gpu
  requires pg_core weights_address buffers weights prompt v (8 + 9 * layer)
  ensures pg_pending weights_address buffers weights prompt (9 + 9 * layer)
{
  unfold pg_core weights_address buffers weights prompt v (8 + 9 * layer);
  let source = pg_index layer 8;
  let target = pg_index layer 9;
  pg_split2 buffers #v source target;
  rewrite PA.pts_to (pg_address buffers source) (pg_slot v source) as input |-> pg_slot v source;
  rewrite PA.pts_to (pg_address buffers target) (pg_slot v target) as output |-> pg_slot v target;
  u_silu_body input output #(hide (pg_slot v source)) #(hide (pg_slot v target)) #1.0R;
  with ys. assert output |-> ys;
  rewrite input |-> pg_slot v source as PA.pts_to (pg_address buffers source) (pg_slot v source);
  rewrite output |-> ys as PA.pts_to (pg_address buffers target) ys;
  pg_join2 buffers #v source target #ys;
  assert pure (pg_step_fact weights (pg_replace v target ys) layer 6);
  pg_stage_advance weights prompt v layer 6 ys;
  fold pg_core weights_address buffers weights prompt (pg_replace v target ys) (9 + 9 * layer);
  fold pg_pending weights_address buffers weights prompt (9 + 9 * layer);
}

inline_for_extraction noextract
fn pg_rope_stage_body
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 3)})
  (output:larray BF16.t 32768{output == pg_address buffers (pg_index layer 4)})
  (table:larray BF16.t 1048576{table == pg_address buffers 1})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) ()
  norewrite
  preserves gpu
  requires pg_core weights_address buffers weights prompt v (3 + 9 * layer)
  ensures pg_pending weights_address buffers weights prompt (4 + 9 * layer)
{
  unfold pg_core weights_address buffers weights prompt v (3 + 9 * layer);
  let source = pg_index layer 3;
  let target = pg_index layer 4;
  pg_split3 buffers #v source 1 target;
  rewrite PA.pts_to (pg_address buffers source) (pg_slot v source) as input |-> pg_slot v source;
  rewrite PA.pts_to (pg_address buffers target) (pg_slot v target) as output |-> pg_slot v target;
  rewrite PA.pts_to (pg_address buffers 1) (pg_slot v 1) as table |-> pg_slot v 1;
  u_take_weight_window weights_address #weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements) #1.0R;
  rewrite PA.pts_to (u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements))
      (S.slice weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements))
    as window |-> pg_layer_weights weights layer;
  u_norm_rope_body input output window table #(hide (pg_slot v source)) #(hide (pg_slot v target))
    #(hide (pg_layer_weights weights layer)) #(hide (pg_slot v 1)) #1.0R #1.0R #1.0R;
  with ys. assert output |-> ys;
  rewrite window |-> pg_layer_weights weights layer as
    PA.pts_to (u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements))
      (S.slice weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements));
  u_return_weight_window weights_address #weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements) #1.0R;
  rewrite input |-> pg_slot v source as PA.pts_to (pg_address buffers source) (pg_slot v source);
  rewrite output |-> ys as PA.pts_to (pg_address buffers target) ys;
  rewrite table |-> pg_slot v 1 as PA.pts_to (pg_address buffers 1) (pg_slot v 1);
  pg_join3 buffers #v source 1 target #ys;
  assert pure (pg_step_fact weights (pg_replace v target ys) layer 1);
  pg_stage_advance weights prompt v layer 1 ys;
  fold pg_core weights_address buffers weights prompt (pg_replace v target ys) (4 + 9 * layer);
  fold pg_pending weights_address buffers weights prompt (4 + 9 * layer);
}

inline_for_extraction noextract
ghost
fn pg_linear_before
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer) (slot:Shared.index 4)
  (input:larray BF16.t (8 * u_linear_width slot){input == pg_address buffers (pg_linear_input_index layer slot)})
  (residual:larray BF16.t 8192{residual == pg_address buffers (pg_linear_residual_index layer slot)})
  (output:larray BF16.t (8 * u_linear_cols slot){output == pg_address buffers (pg_linear_target_index layer slot)})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (#v:erased pg_values) ()
  requires pg_core weights_address buffers weights prompt v (2 + 9 * layer + pg_linear_step slot)
  ensures opt_linear_pre input window residual output (pg_slot v (pg_linear_input_index layer slot))
      (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot)) 1.0R 1.0R 1.0R **
    pg_linear_frame weights_address buffers weights prompt v layer slot
{
  pg_linear_binding layer slot;
  unfold pg_core weights_address buffers weights prompt v (2 + 9 * layer + pg_linear_step slot);
  pg_split3 buffers #v (pg_linear_input_index layer slot) (pg_linear_residual_index layer slot) (pg_linear_target_index layer slot);
  rewrite PA.pts_to (pg_address buffers (pg_linear_input_index layer slot)) (pg_slot v (pg_linear_input_index layer slot))
    as input |-> pg_slot v (pg_linear_input_index layer slot);
  rewrite PA.pts_to (pg_address buffers (pg_linear_residual_index layer slot)) (pg_slot v (pg_linear_residual_index layer slot))
    as residual |-> pg_slot v (pg_linear_residual_index layer slot);
  rewrite PA.pts_to (pg_address buffers (pg_linear_target_index layer slot)) (pg_slot v (pg_linear_target_index layer slot))
    as output |-> pg_slot v (pg_linear_target_index layer slot);
  u_take_weight_window weights_address #weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements) #1.0R;
  rewrite PA.pts_to (u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements))
      (S.slice weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)) as
    window |-> pg_layer_weights weights layer;
  PA.to_mask output;
  fold PA.pts_to_uninit output (8 * u_linear_cols slot);
  fold opt_linear_read input window residual (pg_slot v (pg_linear_input_index layer slot))
    (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot)) 1.0R 1.0R 1.0R;
  fold opt_linear_pre input window residual output (pg_slot v (pg_linear_input_index layer slot))
    (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot)) 1.0R 1.0R 1.0R;
  fold pg_linear_frame weights_address buffers weights prompt v layer slot;
}

inline_for_extraction noextract
ghost
fn pg_linear_after
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer) (slot:Shared.index 4)
  (input:larray BF16.t (8 * u_linear_width slot){input == pg_address buffers (pg_linear_input_index layer slot)})
  (residual:larray BF16.t 8192{residual == pg_address buffers (pg_linear_residual_index layer slot)})
  (output:larray BF16.t (8 * u_linear_cols slot){output == pg_address buffers (pg_linear_target_index layer slot)})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (#v:erased pg_values) ()
  requires opt_linear_post input window residual output (pg_slot v (pg_linear_input_index layer slot))
      (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot))
      (pg_linear_result weights v layer slot) 1.0R 1.0R 1.0R **
    pg_linear_frame weights_address buffers weights prompt v layer slot
  ensures pg_pending weights_address buffers weights prompt (3 + 9 * layer + pg_linear_step slot)
{
  pg_linear_binding layer slot;
  unfold opt_linear_post input window residual output (pg_slot v (pg_linear_input_index layer slot))
    (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot))
    (pg_linear_result weights v layer slot) 1.0R 1.0R 1.0R;
  unfold opt_linear_read input window residual (pg_slot v (pg_linear_input_index layer slot))
    (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot)) 1.0R 1.0R 1.0R;
  unfold pg_linear_frame weights_address buffers weights prompt v layer slot;
  rewrite window |-> pg_layer_weights weights layer as
    PA.pts_to (u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements))
      (S.slice weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements));
  u_return_weight_window weights_address #weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements) #1.0R;
  rewrite input |-> pg_slot v (pg_linear_input_index layer slot) as
    PA.pts_to (pg_address buffers (pg_linear_input_index layer slot)) (pg_slot v (pg_linear_input_index layer slot));
  rewrite residual |-> pg_slot v (pg_linear_residual_index layer slot) as
    PA.pts_to (pg_address buffers (pg_linear_residual_index layer slot)) (pg_slot v (pg_linear_residual_index layer slot));
  rewrite output |-> pg_linear_result weights v layer slot as
    PA.pts_to (pg_address buffers (pg_linear_target_index layer slot)) (pg_linear_result weights v layer slot);
  pg_join3 buffers #v (pg_linear_input_index layer slot) (pg_linear_residual_index layer slot)
    (pg_linear_target_index layer slot) #(hide (pg_linear_result weights v layer slot));
  pg_linear_progress weights prompt v layer slot;
  fold pg_core weights_address buffers weights prompt
    (pg_replace v (pg_linear_target_index layer slot) (pg_linear_result weights v layer slot))
    (3 + 9 * layer + pg_linear_step slot);
  fold pg_pending weights_address buffers weights prompt (3 + 9 * layer + pg_linear_step slot);
}

inline_for_extraction noextract
ghost
fn pg_attention_before (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 4)})
  (output:larray BF16.t 16384{output == pg_address buffers (pg_index layer 5)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (#v:erased pg_values) ()
  requires pg_core weights_address buffers weights prompt v (4 + 9 * layer)
  ensures u_attention_pre input output (pg_slot v (pg_index layer 4)) (pg_slot v (pg_index layer 5)) 1.0R **
    pg_attention_frame weights_address buffers weights prompt v layer
{
  unfold pg_core weights_address buffers weights prompt v (4 + 9 * layer);
  pg_split2 buffers #v (pg_index layer 4) (pg_index layer 5);
  rewrite PA.pts_to (pg_address buffers (pg_index layer 4)) (pg_slot v (pg_index layer 4)) as
    input |-> pg_slot v (pg_index layer 4);
  rewrite PA.pts_to (pg_address buffers (pg_index layer 5)) (pg_slot v (pg_index layer 5)) as
    output |-> pg_slot v (pg_index layer 5);
  fold u_attention_pre input output (pg_slot v (pg_index layer 4)) (pg_slot v (pg_index layer 5)) 1.0R;
  fold pg_attention_frame weights_address buffers weights prompt v layer;
}

inline_for_extraction noextract
ghost
fn pg_attention_after (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 4)})
  (output:larray BF16.t 16384{output == pg_address buffers (pg_index layer 5)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (#v:erased pg_values) ()
  requires u_attention_post input output (pg_slot v (pg_index layer 4)) 1.0R **
    pg_attention_frame weights_address buffers weights prompt v layer
  ensures pg_pending weights_address buffers weights prompt (5 + 9 * layer)
{
  unfold u_attention_post input output (pg_slot v (pg_index layer 4)) 1.0R;
  unfold u_attention_final output (pg_slot v (pg_index layer 4));
  with ys. assert output |-> ys;
  unfold pg_attention_frame weights_address buffers weights prompt v layer;
  rewrite input |-> pg_slot v (pg_index layer 4) as
    PA.pts_to (pg_address buffers (pg_index layer 4)) (pg_slot v (pg_index layer 4));
  rewrite output |-> ys as PA.pts_to (pg_address buffers (pg_index layer 5)) ys;
  pg_join2 buffers #v (pg_index layer 4) (pg_index layer 5) #ys;
  assert pure (pg_step_fact weights (pg_replace v (pg_index layer 5) ys) layer 2);
  pg_stage_advance weights prompt v layer 2 ys;
  fold pg_core weights_address buffers weights prompt (pg_replace v (pg_index layer 5) ys) (5 + 9 * layer);
  fold pg_pending weights_address buffers weights prompt (5 + 9 * layer);
}

inline_for_extraction noextract
fn pg_data_ptr (stream:ST.stream_t) (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (index:Shared.index 256)
  preserves cpu ** opt_allocated_owners stream buffers
  returns address:larray BF16.t (pg_size index)
  ensures pure (address == pg_address buffers index /\ KA.is_global_array address)
{
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  pg_shape_at index;
  opt_matching_shape_at opt_bf16_allocation_shapes buffers.bf16_buffers index;
  unfold opt_allocated_owners stream buffers;
  let address = opt_buffer_data_ptr stream buffers.bf16_buffers index;
  fold opt_allocated_owners stream buffers;
  address
}

inline_for_extraction noextract
fn pg_pending_weight_window
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (lo:HostSize.t) (hi:erased nat{HostSize.v lo <= hi /\ hi <= opt_weight_count})
  (#contents:erased (lseq BF16.t opt_weight_count))
  preserves cpu ** PA.pts_to_mask weights #1.0R (u_metadata_mask contents) (fun _ -> False)
  returns window:PA.array BF16.t
  ensures rewrites_to window (u_weight_window weights (HostSize.v lo) hi)
  ensures pure (PA.length window == hi - HostSize.v lo /\ KA.is_global_array window)
{
  u_weight_window_shape weights (HostSize.v lo) hi;
  let window = PA.sub weights #1.0R #(fun _ -> False) lo hi
    #(hide (u_metadata_mask contents <: S.seq (option BF16.t)));
  PA.mask_mext weights (fun _ -> False);
  PA.mask_mext window (fun _ -> False);
  drop_ (PA.pts_to_mask window #1.0R (S.slice (u_metadata_mask contents) (HostSize.v lo) hi) (fun _ -> False));
  window
}

inline_for_extraction noextract
ghost
fn pg_expand_pending (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (calls:nat{11 <= calls /\ calls <= 253}) (extra:slprop) ()
  requires on gpu_loc (pg_pending weights_address buffers weights prompt calls ** extra)
  ensures on gpu_loc (exists* (v:pg_values). pg_core weights_address buffers weights prompt v calls ** extra)
{
  ghost_impersonate gpu_loc
    (on gpu_loc (pg_pending weights_address buffers weights prompt calls ** extra))
    (on gpu_loc (exists* (v:pg_values). pg_core weights_address buffers weights prompt v calls ** extra))
    fn () {
      on_elim (pg_pending weights_address buffers weights prompt calls ** extra);
      unfold pg_pending weights_address buffers weights prompt calls;
      with v. assert pg_core weights_address buffers weights prompt v calls;
      intro_exists_erased #pg_values
        (fun v -> pg_core weights_address buffers weights prompt v calls ** extra) v;
      on_intro (exists* (v:pg_values). pg_core weights_address buffers weights prompt v calls ** extra);
    };
}

inline_for_extraction noextract
fn pg_launch_step (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (calls:nat{11 <= calls /\ calls < 253})
  (after:nat{after == calls + 1})
  (family:opt_pending_family pg_values
    (fun v -> pg_core weights_address buffers weights prompt v calls)
    (fun _ -> pg_pending weights_address buffers weights prompt after))
  (stream:ST.stream_t) (#start:E.epoch_t)
  preserves cpu ** ST.stream_live stream
  requires E.epoch_live stream (start + calls) **
    P.pledge0 (E.epoch_done stream (start + calls))
      (pg_after_stage weights_address prompt_address buffers weights prompt calls)
  ensures E.epoch_live stream (start + after) **
    P.pledge0 (E.epoch_done stream (start + after))
      (pg_after_stage weights_address prompt_address buffers weights prompt after)
{
  let extra = PA.pts_to prompt_address prompt ** RT.contents buffers.token_buffer;
  rewrite P.pledge0 (E.epoch_done stream (start + calls))
      (pg_after_stage weights_address prompt_address buffers weights prompt calls) as
    P.pledge0 (E.epoch_done stream (start + calls))
      (on gpu_loc (pg_pending weights_address buffers weights prompt calls ** extra));
  P.rewrite_pledge (on gpu_loc (pg_pending weights_address buffers weights prompt calls ** extra))
    (on gpu_loc (exists* (v:pg_values). pg_core weights_address buffers weights prompt v calls ** extra))
    (pg_expand_pending weights_address buffers #weights #prompt calls extra);
  u_launch_const_post (u_frame_family family extra) stream #(start + calls);
  u_queue_step stream #start calls (on gpu_loc (pg_pending weights_address buffers weights prompt after ** extra));
  rewrite E.epoch_live stream (start + (calls + 1)) as E.epoch_live stream (start + after);
  rewrite P.pledge0 (E.epoch_done stream (start + (calls + 1)))
      (on gpu_loc (pg_pending weights_address buffers weights prompt after ** extra)) as
    P.pledge0 (E.epoch_done stream (start + after))
      (pg_after_stage weights_address prompt_address buffers weights prompt after);
}

inline_for_extraction noextract
fn pg_enqueue_layer
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (layer:pg_remaining_layer)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t)
  preserves cpu ** ST.stream_live stream ** opt_allocated_owners stream buffers **
    PA.pts_to_mask weights_address #1.0R (u_metadata_mask weights) (fun _ -> False)
  requires E.epoch_live stream (start + (2 + 9 * layer)) **
    P.pledge0 (E.epoch_done stream (start + (2 + 9 * layer)))
      (pg_after_stage weights_address prompt_address buffers weights prompt (2 + 9 * layer))
  ensures E.epoch_live stream (start + pg_layer_end layer) **
    P.pledge0 (E.epoch_done stream (start + pg_layer_end layer))
      (pg_after_stage weights_address prompt_address buffers weights prompt (pg_layer_end layer))
{
  pg_layer_step_calls layer;
  let base = HostSize.uint_to_t (pg_layer_base layer);
  let window = pg_pending_weight_window weights_address base (hide (pg_layer_base layer + dec_weight_elements)) #weights;
  let input = pg_data_ptr stream buffers (pg_index layer 0);
  let table = pg_data_ptr stream buffers 1;
  let norm = pg_data_ptr stream buffers (pg_index layer 2);
  let qkv = pg_data_ptr stream buffers (pg_index layer 3);
  let rotated = pg_data_ptr stream buffers (pg_index layer 4);
  let attention = pg_data_ptr stream buffers (pg_index layer 5);
  let residual = pg_data_ptr stream buffers (pg_index layer 6);
  let postnorm = pg_data_ptr stream buffers (pg_index layer 7);
  let gateup = pg_data_ptr stream buffers (pg_index layer 8);
  let activated = pg_data_ptr stream buffers (pg_index layer 9);
  let hidden = pg_data_ptr stream buffers (pg_index layer 10);
  let qkv_code = pg_linear_family weights_address buffers layer 0 norm input qkv window 1024sz 4096sz 1sz 2048sz false #weights #prompt;
  pg_launch_step weights_address prompt_address buffers #weights #prompt (2 + 9 * layer) (3 + 9 * layer) qkv_code stream #start;
  let rope_code = pg_rope_family weights_address buffers layer qkv rotated table window #weights #prompt;
  pg_launch_step weights_address prompt_address buffers #weights #prompt (3 + 9 * layer) (4 + 9 * layer) rope_code stream #start;
  let attention_code = pg_attention_family weights_address buffers layer rotated attention #weights #prompt;
  pg_launch_step weights_address prompt_address buffers #weights #prompt (4 + 9 * layer) (5 + 9 * layer) attention_code stream #start;
  let projection_code = pg_linear_family weights_address buffers layer 1 attention input residual window
    2048sz 1024sz 0sz (SZ.uint32_to_sizet 4196352ul) true #weights #prompt;
  pg_launch_step weights_address prompt_address buffers #weights #prompt (5 + 9 * layer) (6 + 9 * layer) projection_code stream #start;
  let rms_code = pg_rms_family weights_address buffers layer 4 residual postnorm (base +^ 1024sz) #weights #prompt;
  pg_launch_step weights_address prompt_address buffers #weights #prompt (6 + 9 * layer) (7 + 9 * layer) rms_code stream #start;
  let gate_code = pg_linear_family weights_address buffers layer 2 postnorm residual gateup window
    1024sz 6144sz 2sz (SZ.uint32_to_sizet 6293760ul) false #weights #prompt;
  pg_launch_step weights_address prompt_address buffers #weights #prompt (7 + 9 * layer) (8 + 9 * layer) gate_code stream #start;
  let silu_code = pg_silu_family weights_address buffers layer gateup activated #weights #prompt;
  pg_launch_step weights_address prompt_address buffers #weights #prompt (8 + 9 * layer) (9 + 9 * layer) silu_code stream #start;
  let down_code = pg_linear_family weights_address buffers layer 3 activated residual hidden window
    3072sz 1024sz 0sz (SZ.uint32_to_sizet 12585216ul) true #weights #prompt;
  pg_launch_step weights_address prompt_address buffers #weights #prompt (9 + 9 * layer) (10 + 9 * layer) down_code stream #start;
  if (layer < 27) {
    let nextnorm = pg_data_ptr stream buffers (pg_target layer 8);
    let next_code = pg_rms_family weights_address buffers layer 8 hidden nextnorm
      (base +^ SZ.uint32_to_sizet 15730944ul) #weights #prompt;
    pg_launch_step weights_address prompt_address buffers #weights #prompt (10 + 9 * layer) (pg_layer_end layer) next_code stream #start;
  } else {
    rewrite E.epoch_live stream (start + (10 + 9 * layer)) as E.epoch_live stream (start + pg_layer_end layer);
    rewrite P.pledge0 (E.epoch_done stream (start + (10 + 9 * layer)))
        (pg_after_stage weights_address prompt_address buffers weights prompt (10 + 9 * layer)) as
      P.pledge0 (E.epoch_done stream (start + pg_layer_end layer))
        (pg_after_stage weights_address prompt_address buffers weights prompt (pg_layer_end layer));
  };
}

inline_for_extraction noextract
fn pg_enqueue_remaining_layers
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t)
  preserves cpu ** ST.stream_live stream ** opt_allocated_owners stream buffers **
    PA.pts_to_mask weights_address #1.0R (u_metadata_mask weights) (fun _ -> False)
  requires E.epoch_live stream (start + 11) **
    P.pledge0 (E.epoch_done stream (start + 11)) (pg_after_stage weights_address prompt_address buffers weights prompt 11)
  ensures E.epoch_live stream (start + 253) **
    P.pledge0 (E.epoch_done stream (start + 253)) (pg_after_stage weights_address prompt_address buffers weights prompt 253)
{
  let mut layer:nat = 1;
  while (!layer < 28)
    invariant exists* (l:nat{1 <= l /\ l <= 28}).
      layer |-> l ** E.epoch_live stream (start + pg_loop_epoch l) **
      P.pledge0 (E.epoch_done stream (start + pg_loop_epoch l))
        (pg_after_stage weights_address prompt_address buffers weights prompt (pg_loop_epoch l))
    decreases (28 - !layer)
  {
    let l = !layer;
    rewrite E.epoch_live stream (start + pg_loop_epoch l) as E.epoch_live stream (start + (2 + 9 * l));
    rewrite P.pledge0 (E.epoch_done stream (start + pg_loop_epoch l))
        (pg_after_stage weights_address prompt_address buffers weights prompt (pg_loop_epoch l)) as
      P.pledge0 (E.epoch_done stream (start + (2 + 9 * l)))
        (pg_after_stage weights_address prompt_address buffers weights prompt (2 + 9 * l));
    pg_enqueue_layer weights_address prompt_address buffers l #weights #prompt stream #start;
    pg_layer_next l;
    rewrite E.epoch_live stream (start + pg_layer_end l) as E.epoch_live stream (start + pg_loop_epoch (l + 1));
    rewrite P.pledge0 (E.epoch_done stream (start + pg_layer_end l))
        (pg_after_stage weights_address prompt_address buffers weights prompt (pg_layer_end l)) as
      P.pledge0 (E.epoch_done stream (start + pg_loop_epoch (l + 1)))
        (pg_after_stage weights_address prompt_address buffers weights prompt (pg_loop_epoch (l + 1)));
    layer := l + 1;
  };
  with finished. assert layer |-> finished;
  assert pure (finished == 28);
  rewrite E.epoch_live stream (start + pg_loop_epoch finished) as E.epoch_live stream (start + 253);
  rewrite P.pledge0 (E.epoch_done stream (start + pg_loop_epoch finished))
      (pg_after_stage weights_address prompt_address buffers weights prompt (pg_loop_epoch finished)) as
    P.pledge0 (E.epoch_done stream (start + 253))
      (pg_after_stage weights_address prompt_address buffers weights prompt 253);
}

inline_for_extraction noextract
fn pg_live_projection_instruction
  (weight_tile:larray BF16.t 16384) (hidden:larray BF16.t 7168)
  (step:szlt 64) (fragment:MMA.fragment) (tid:szlt 32)
  (#wv:erased (lseq BF16.t 16384)) (#hv:erased (lseq BF16.t 7168))
  (#before:Chest.chest2 F32.t 16 8) (#fw #fh:perm)
  preserves gpu ** thread_id 32 tid **
    weight_tile |-> Frac fw wv ** hidden |-> Frac fh hv
  requires fragment |-> before
  ensures fragment |-> MMA.emma before
    (ematrix_subtile (Tensor.from_seq (rm 16 1024) wv) 16 16 0 step)
    (pg_projection_b hv step)
{
  Tensor.tensor_abs' (rm 16 1024) weight_tile;
  Tensor.tensor_abs' (cm 1024 7) hidden;
  let am = Tensor.from_array (rm 16 1024) weight_tile;
  let bm = Tensor.from_array (cm 1024 7) hidden;
  assert rewrites_to am (Tensor.from_array (rm 16 1024) weight_tile);
  assert rewrites_to bm (Tensor.from_array (cm 1024 7) hidden);
  let at = array2_extract_tile_ro' am 16 16 0 (SZ.v step);
  let bt = array2_extract_tile_ro' bm 16 7 (SZ.v step) 0;
  Registers.mma_sync7 at bt fragment;
  elim_trade
    (at |-> Frac fw (ematrix_subtile (Tensor.from_seq (rm 16 1024) wv) 16 16 0 step))
    (am |-> Frac fw (Tensor.from_seq (rm 16 1024) wv));
  elim_trade
    (bt |-> Frac fh (ematrix_subtile (Tensor.from_seq (cm 1024 7) hv) 16 7 step 0))
    (bm |-> Frac fh (Tensor.from_seq (cm 1024 7) hv));
  pg_projection_b_matches_registers hv step;
  Tensor.tensor_concr am;
  Tensor.tensor_concr bm;
  rewrite each Tensor.core am as weight_tile;
  rewrite each Tensor.core bm as hidden;
}

inline_for_extraction noextract
ghost
fn pg_projection_bind_operands
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (block:nat{block < 153600 /\ block % 16 == 0}) (step:Shared.index 64)
  ensures pure (
    (block < 151936 ==>
      block + 16 <= 151936 /\
      pg_projection_a weights block step ==
        ematrix_subtile (Tensor.from_seq (rm 16 1024)
          (S.slice weights (block * 1024) ((block + 16) * 1024))) 16 16 0 step) /\
    (151936 <= block ==>
      pg_projection_a weights block step == ZeroA.operand_a /\
      pg_projection_b hidden step == Registers.pad_b7 (ZeroA.operand_b7 hidden step)))
{
  if (block < 151936) {
    pg_projection_live_block block;
    pg_projection_live_a weights block step;
    ()
  } else {
    pg_zero_a_binding weights hidden block step;
    ()
  };
}

inline_for_extraction noextract
fn pg_projection_tile_mma
  (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168)
  (tile:larray F32.t 128) (block:sz{block < 153600 /\ block % 16 == 0}) (tid:szlt 32)
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq F32.t 128)) (#fw #fh:perm)
  preserves gpu ** thread_id 32 tid ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires tile |-> Frac (1.0R /. 32.0R) old
  ensures tile |-> Frac (1.0R /. 32.0R) (pg_projection_tile_values wv hv block)
{
  let fragment = MMA.alloc_fragment ();
  MMA.fill fragment (F.zero #F32.t);
  let mut step = 0sz;
  while (!step <^ 64sz)
    invariant exists* (s:sz{s <= 64}).
      step |-> s ** fragment |-> pg_projection_acc wv hv block s
    decreases (64 - !step)
  {
    let s = !step;
    pg_projection_bind_operands #wv #hv block s;
    if (block <^ SZ.uint32_to_sizet 151936ul) {
      let weight_tile = dec_borrow_source_slice weights (block *^ 1024sz) 16384sz #wv #fw;
      pg_live_projection_instruction weight_tile hidden s fragment tid
        #(hide (S.slice wv (block * 1024) ((block + 16) * 1024))) #hv
        #(pg_projection_acc wv hv block s) #fw #fh;
      dec_restore_source_slice weights (block *^ 1024sz) 16384sz weight_tile #wv #fw;
      rewrite fragment |-> MMA.emma (pg_projection_acc wv hv block s)
        (ematrix_subtile (Tensor.from_seq (rm 16 1024)
          (S.slice wv (block * 1024) ((block + 16) * 1024))) 16 16 0 s)
        (pg_projection_b hv s) as fragment |-> pg_projection_acc wv hv block (s + 1);
      ()
    } else {
      ZeroA.mma_sync7 hidden s fragment #hv #fh;
      rewrite fragment |-> MMA.emma (pg_projection_acc wv hv block s) ZeroA.operand_a
        (Registers.pad_b7 (ZeroA.operand_b7 hv s)) as fragment |-> pg_projection_acc wv hv block (s + 1);
      ()
    };
    step := s +^ 1sz;
  };
  dec_source_loop_end !step 64sz;
  Tensor.tensor_abs' (rm 16 8) tile;
  let matrix = Tensor.from_array (rm 16 8) tile;
  assert rewrites_to matrix (Tensor.from_array (rm 16 8) tile);
  MMA.store fragment matrix;
  drop_ (fragment |-> pg_projection_acc wv hv block 64);
  Tensor.tensor_concr matrix;
  rewrite each Tensor.core matrix as tile;
}

inline_for_extraction noextract
fn pg_projection_write_tile (output:larray BF16.t 1228800) (tile:larray F32.t 128)
  (block:sz{block < 153600 /\ block % 16 == 0})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  preserves gpu ** tile |-> pg_projection_tile_values weights hidden block
  requires pg_projection_owned output old (pg_projection_output weights hidden old) block
  ensures pg_projection_owned output old (pg_projection_output weights hidden old) (block + 16)
{
  unfold pg_projection_owned output old (pg_projection_output weights hidden old) block;
  let mut row = 0sz;
  while (!row <^ 7sz)
    invariant exists* (r:sz{r <= 7}) (ys:lseq BF16.t 1228800).
      row |-> r ** output |-> ys **
      pure (pg_projection_written ys (pg_projection_output weights hidden old) block (r * 16) /\ pg_projection_outside old ys)
    decreases (7 - !row)
  {
    let r = !row;
    let mut col = 0sz;
    while (!col <^ 16sz)
      invariant exists* (c:sz{c <= 16}) (ys:lseq BF16.t 1228800).
        col |-> c ** output |-> ys **
        pure (pg_projection_written ys (pg_projection_output weights hidden old) block (r * 16 + c) /\ pg_projection_outside old ys)
      decreases (16 - !col)
    {
      let c = !col;
      let value = tile.(c *^ 8sz +^ r);
      dec_source_tile_sequence_at (pg_projection_acc weights hidden block 64) c r;
      let rounded = Casts.fcast #F32.t #BF16.t value;
      assert pure (rounded == S.index (pg_projection_output weights hidden old) (r * 153600 + block + c));
      with ys. assert output |-> ys;
      pg_projection_write ys (pg_projection_output weights hidden old) block r c;
      output.(r *^ SZ.uint32_to_sizet 153600ul +^ block +^ c) <- rounded;
      col := c +^ 1sz;
    };
    dec_source_loop_end !col 16sz;
    row := r +^ 1sz;
  };
  dec_source_loop_end !row 7sz;
  with ys. assert output |-> ys;
  assert pure (pg_projection_written ys (pg_projection_output weights hidden old) (block + 16) 0);
  fold pg_projection_owned output old (pg_projection_output weights hidden old) (block + 16);
}

inline_for_extraction noextract
fn pg_projection_owner_write (output:larray BF16.t 1228800) (tile:larray F32.t 128)
  (block:sz{block < 153600 /\ block % 16 == 0}) (tid:szlt 32)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  preserves gpu ** when_ (SZ.v tid == 0) (tile |-> pg_projection_tile_values weights hidden block)
  requires when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) block)
  ensures when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) (block + 16))
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (tile |-> pg_projection_tile_values weights hidden block)
      as tile |-> pg_projection_tile_values weights hidden block;
    rewrite when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) block)
      as pg_projection_owned output old (pg_projection_output weights hidden old) block;
    pg_projection_write_tile output tile block #weights #hidden #old;
    rewrite tile |-> pg_projection_tile_values weights hidden block as
      when_ (SZ.v tid == 0) (tile |-> pg_projection_tile_values weights hidden block);
    rewrite pg_projection_owned output old (pg_projection_output weights hidden old) (block + 16) as
      when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) (block + 16));
  } else {
    rewrite when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) block) as emp;
    rewrite emp as when_ (SZ.v tid == 0)
      (pg_projection_owned output old (pg_projection_output weights hidden old) (block + 16));
  };
}

inline_for_extraction noextract
ghost
fn pg_projection_begin (output:larray BF16.t 1228800) (tid:szlt 32)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  requires when_ (SZ.v tid == 0) (output |-> old)
  ensures when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) 0)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (output |-> old) as output |-> old;
    fold pg_projection_owned output old (pg_projection_output weights hidden old) 0;
    rewrite pg_projection_owned output old (pg_projection_output weights hidden old) 0 as
      when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) 0);
  } else {
    rewrite when_ (SZ.v tid == 0) (output |-> old) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) 0);
  };
}

inline_for_extraction noextract
ghost
fn pg_projection_finish (output:larray BF16.t 1228800) (tid:szlt 32)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  requires when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) 153600)
  ensures when_ (SZ.v tid == 0) (output |-> pg_projection_output weights hidden old)
{
  if (tid = 0sz) {
    rewrite when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) 153600)
      as pg_projection_owned output old (pg_projection_output weights hidden old) 153600;
    unfold pg_projection_owned output old (pg_projection_output weights hidden old) 153600;
    with ys. assert output |-> ys;
    pg_projection_finished weights hidden old ys;
    rewrite output |-> ys as output |-> pg_projection_output weights hidden old;
    rewrite output |-> pg_projection_output weights hidden old as
      when_ (SZ.v tid == 0) (output |-> pg_projection_output weights hidden old);
  } else {
    rewrite when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) 153600) as emp;
    rewrite emp as when_ (SZ.v tid == 0) (output |-> pg_projection_output weights hidden old);
  };
}

inline_for_extraction noextract
fn pg_projection_body
  (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168)
  (output:larray BF16.t 1228800) (tile:larray F32.t 128) (tid:szlt 32)
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800)) (#tv:erased (lseq F32.t 128)) (#fw #fh:perm)
  preserves gpu ** thread_id 32 tid ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires tile |-> Frac (1.0R /. 32.0R) tv ** when_ (SZ.v tid == 0) (output |-> old)
  ensures tile |-> Frac (1.0R /. 32.0R) (pg_projection_tile_values wv hv 153584) **
    when_ (SZ.v tid == 0) (output |-> pg_projection_output wv hv old)
{
  pg_projection_begin output tid #wv #hv #old;
  let mut block = 0sz;
  while (!block <^ SZ.uint32_to_sizet 153600ul)
    invariant exists* (b:sz{b <= 153600 /\ b % 16 == 0}).
      block |-> b ** tile |-> Frac (1.0R /. 32.0R) (pg_projection_previous wv hv tv b) **
      when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output wv hv old) b)
    decreases (153600 - !block)
  {
    let b = !block;
    pg_projection_tile_mma weights hidden tile b tid #wv #hv #(hide (pg_projection_previous wv hv tv b)) #fw #fh;
    source_tile_gather tile #(hide (pg_projection_tile_values wv hv b)) #(hide (SZ.v tid));
    pg_projection_owner_write output tile b tid #wv #hv #old;
    source_tile_distribute tile #(hide (pg_projection_tile_values wv hv b)) #(hide (SZ.v tid));
    let next = b +^ 16sz;
    rewrite when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output wv hv old) (b + 16)) as
      when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output wv hv old) next);
    block := next;
  };
  let finished = !block;
  dec_source_loop_end finished (SZ.uint32_to_sizet 153600ul);
  rewrite when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output wv hv old) finished) as
    when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output wv hv old) 153600);
  rewrite tile |-> Frac (1.0R /. 32.0R) (pg_projection_previous wv hv tv finished) as
    tile |-> Frac (1.0R /. 32.0R) (pg_projection_tile_values wv hv 153584);
  pg_projection_finish output tid #wv #hv #old;
}

inline_for_extraction noextract
ghost
fn pg_projection_block_setup
  (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168) (output:larray BF16.t 1228800)
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  (sh:SH.c_shmems opt_linear_shared_desc) (_:natlt 1) ()
  norewrite
  requires SH.live_c_shmems sh ** pg_projection_pre weights hidden output wv hv old
  ensures (forall+ (tid:natlt 32).
    pg_projection_kpre weights hidden output (opt_linear_tile sh) wv hv old tid) ** emp
{
  SH.unfold_c_shmems sh (`%opt_linear_shared_desc);
  unfold pg_projection_pre weights hidden output wv hv old;
  let tile = opt_linear_tile sh;
  rewrite each fst sh as tile;
  with raw. assert tile |-> raw;
  PA.pts_to_len tile;
  let tv = hide #(lseq F32.t 128) (Ghost.reveal #(S.seq F32.t) raw);
  rewrite tile |-> raw as tile |-> tv;
  ArrayExtra.array_share weights #wv #1.0R 32;
  ArrayExtra.array_share hidden #hv #1.0R 32;
  ArrayExtra.array_share tile #tv #1.0R 32;
  source_warp_owner_intro (output |-> old);
  forevery_zip #(natlt 32)
    (fun _ -> weights |-> Frac (1.0R /. 32.0R) wv)
    (fun _ -> hidden |-> Frac (1.0R /. 32.0R) hv);
  forevery_map #(natlt 32)
    (fun _ -> tile |-> Frac (1.0R /. 32.0R) tv)
    (fun _ -> exists* (v:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) v)
    fn _ { () };
  forevery_zip3 #(natlt 32)
    (fun _ -> weights |-> Frac (1.0R /. 32.0R) wv ** hidden |-> Frac (1.0R /. 32.0R) hv)
    (fun _ -> exists* (v:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) v)
    (fun i -> when_ (i == 0) (output |-> old));
  forevery_ext #(natlt 32)
    (fun i -> (weights |-> Frac (1.0R /. 32.0R) wv ** hidden |-> Frac (1.0R /. 32.0R) hv) **
      (exists* (v:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) v) ** when_ (i == 0) (output |-> old))
    (fun i -> pg_projection_kpre weights hidden output tile wv hv old i);
  rewrite each tile as opt_linear_tile sh;
}

inline_for_extraction noextract
ghost
fn pg_projection_block_teardown
  (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168) (output:larray BF16.t 1228800)
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  (sh:SH.c_shmems opt_linear_shared_desc) (_:natlt 1) ()
  norewrite
  requires (forall+ (tid:natlt 32).
    pg_projection_kpost weights hidden output (opt_linear_tile sh) wv hv old tid) ** emp
  ensures SH.live_c_shmems sh ** pg_projection_post weights hidden output wv hv old
{
  let tile = opt_linear_tile sh;
  rewrite each opt_linear_tile sh as tile;
  forevery_ext #(natlt 32)
    (fun i -> pg_projection_kpost weights hidden output tile wv hv old i)
    (fun i -> (weights |-> Frac (1.0R /. 32.0R) wv ** hidden |-> Frac (1.0R /. 32.0R) hv) **
      tile |-> Frac (1.0R /. 32.0R) (pg_projection_tile_values wv hv 153584) **
      when_ (i == 0) (output |-> pg_projection_output wv hv old));
  forevery_unzip3 #(natlt 32)
    (fun _ -> weights |-> Frac (1.0R /. 32.0R) wv ** hidden |-> Frac (1.0R /. 32.0R) hv)
    (fun _ -> tile |-> Frac (1.0R /. 32.0R) (pg_projection_tile_values wv hv 153584))
    (fun i -> when_ (i == 0) (output |-> pg_projection_output wv hv old));
  source_warp_owner_elim (output |-> pg_projection_output wv hv old);
  ArrayExtra.array_gather tile #(hide (pg_projection_tile_values wv hv 153584)) #1.0R 32;
  forevery_unzip #(natlt 32)
    (fun _ -> weights |-> Frac (1.0R /. 32.0R) wv)
    (fun _ -> hidden |-> Frac (1.0R /. 32.0R) hv);
  ArrayExtra.array_gather weights #wv #1.0R 32;
  ArrayExtra.array_gather hidden #hv #1.0R 32;
  fold pg_projection_post weights hidden output wv hv old;
  rewrite each tile as fst sh;
  SH.fold_c_shmems sh (`%opt_linear_shared_desc);
}

inline_for_extraction noextract
fn pg_projection_thread
  (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168) (output:larray BF16.t 1228800)
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  (sh:SH.c_shmems opt_linear_shared_desc) (bid:szlt 1) (tid:szlt 32) ()
  norewrite
  requires gpu ** pure (SH.c_shmems_inv sh) **
    pg_projection_kpre weights hidden output (opt_linear_tile sh) wv hv old tid **
    thread_id 32 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 32) ** Barrier.barrier_state 0
  ensures gpu ** pg_projection_kpost weights hidden output (opt_linear_tile sh) wv hv old tid **
    thread_id 32 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 32) ** Barrier.barrier_state 0
{
  unfold pg_projection_kpre weights hidden output (opt_linear_tile sh) wv hv old tid;
  let tile = opt_linear_tile sh;
  rewrite each opt_linear_tile sh as tile;
  with tv. assert tile |-> Frac (1.0R /. 32.0R) tv;
  pg_projection_body weights hidden output tile tid #wv #hv #old #tv #(1.0R /. 32.0R) #(1.0R /. 32.0R);
  fold pg_projection_kpost weights hidden output tile wv hv old tid;
  rewrite each tile as opt_linear_tile sh;
}

inline_for_extraction noextract
let pg_projection_descriptor
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (hidden:larray BF16.t 7168{KA.is_global_array hidden})
  (output:larray BF16.t 1228800{KA.is_global_array output})
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  : k:KS.kernel_desc (pg_projection_pre weights hidden output wv hv old)
      (pg_projection_post weights hidden output wv hv old){
    SZ.v k.nblk == 1 /\ k.nthr == 32sz /\ k.shmems_desc == opt_linear_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 32sz /\ k.barrier_count bid == 0)} =
  let pre = pg_projection_pre weights hidden output wv hv old in
  let post = pg_projection_post weights hidden output wv hv old in {
    nblk = 1sz; nthr = 32sz; shmems_desc = opt_linear_shared_desc;
    kpre = (fun sh _ tid -> pg_projection_kpre weights hidden output (opt_linear_tile sh) wv hv old tid);
    kpost = (fun sh _ tid -> pg_projection_kpost weights hidden output (opt_linear_tile sh) wv hv old tid);
    barrier_contract = (fun _ _ -> Barrier.empty_contract 32);
    barrier_count = (fun _ -> 0);
    barrier_ok = (fun _ _ -> Barrier.empty_barrier_transform 32);
    f = pg_projection_thread weights hidden output #wv #hv #old;
    frame = emp; block_pre = (fun _ -> pre); block_post = (fun _ -> post);
    setup = opt_single_block_setup pre; teardown = opt_single_block_teardown post;
    block_frame = (fun _ _ -> emp);
    block_setup = pg_projection_block_setup weights hidden output #wv #hv #old;
    block_teardown = pg_projection_block_teardown weights hidden output #wv #hv #old;
    block_pre_sendable = (fun _ ->
      (solve <: is_send_across gpu_of (pg_projection_pre weights hidden output wv hv old)));
    block_post_sendable = (fun _ ->
      (solve <: is_send_across gpu_of (pg_projection_post weights hidden output wv hv old)));
    kpre_sendable = (fun sh pf _ tid -> pg_projection_pre_send weights hidden output #wv #hv #old sh pf tid);
    kpost_sendable = (fun sh pf _ tid -> pg_projection_post_send weights hidden output #wv #hv #old sh pf tid)
  }
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
ghost
fn pg_take_hidden_prefix (input:larray BF16.t 8192)
  (#x:erased (lseq BF16.t 8192)) (#fraction:perm)
  requires input |-> Frac fraction x
  ensures pg_hidden_prefix input |-> Frac fraction (S.slice x 0 7168) **
    PA.pts_to_range input 7168 8192 #fraction (S.slice x 7168 8192)
{
  PA.pts_to_range_intro input fraction x;
  PA.pts_to_range_split input 0 7168 8192;
  with first. assert PA.pts_to_range input 0 7168 #fraction first;
  PA.unfold_pts_to_range input 0 7168 #fraction first;
  PA.from_mask (PA.gsub input 0 7168);
  with values. assert PA.pts_to (PA.gsub input 0 7168) #fraction values;
  assert pure (S.equal values (S.slice x 0 7168));
  S.lemma_eq_elim values (S.slice x 0 7168);
}

inline_for_extraction noextract
ghost
fn pg_return_hidden_prefix (input:larray BF16.t 8192)
  (#x:erased (lseq BF16.t 8192)) (#fraction:perm)
  requires pg_hidden_prefix input |-> Frac fraction (S.slice x 0 7168) **
    PA.pts_to_range input 7168 8192 #fraction (S.slice x 7168 8192)
  ensures input |-> Frac fraction x
{
  rewrite pg_hidden_prefix input |-> Frac fraction (S.slice x 0 7168) as
    PA.pts_to (PA.gsub input 0 7168) #fraction (S.slice x 0 7168);
  PA.to_mask (PA.gsub input 0 7168);
  PA.fold_pts_to_range input 0 7168 #fraction (S.slice x 0 7168);
  PA.pts_to_range_join input 0 7168 8192;
  assert pure (S.equal (S.append (S.slice x 0 7168) (S.slice x 7168 8192)) x);
  S.lemma_eq_elim (S.append (S.slice x 0 7168) (S.slice x 7168 8192)) x;
  PA.pts_to_range_elim input fraction x;
}

inline_for_extraction noextract
fn pg_final_rms_body (weights:larray BF16.t opt_weight_count)
  (input:larray BF16.t 8192) (prefix:larray BF16.t 7168{prefix == pg_hidden_prefix input})
  (output:larray BF16.t 7168)
  (#w:erased (lseq BF16.t opt_weight_count)) (#x:erased (lseq BF16.t 8192))
  (#old:erased (lseq BF16.t 7168)) (#fx #fw:perm)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w
  requires output |-> old
  ensures exists* (ys:lseq BF16.t 7168).
    output |-> ys ** pure (u_rms_prefix #7 #opt_weight_count (S.slice x 0 7168) w ys 155582464 7)
{
  pg_take_hidden_prefix input #x #fx;
  rewrite pg_hidden_prefix input |-> Frac fx (S.slice x 0 7168) as
    prefix |-> Frac fx (S.slice x 0 7168);
  u_rmsnorm_body 7sz prefix output weights (SZ.uint32_to_sizet 155582464ul)
    #(hide (S.slice x 0 7168)) #old #w #fx #fw;
  rewrite prefix |-> Frac fx (S.slice x 0 7168) as
    pg_hidden_prefix input |-> Frac fx (S.slice x 0 7168);
  pg_return_hidden_prefix input #x #fx;
}
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
fn pg_sel_maybe_store (#n:erased pos)
  (values:larray F32.t n) (indices:larray I64.t n)
  (target source:szlt n) (take:bool) (right:F32.t)
  (#vs:erased (lseq (option F32.t) n)) (#is:erased (lseq (option I64.t) n))
  (#right_index:erased I64.t)
  preserves gpu
  requires PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
    pure (S.index is source == Some (Ghost.reveal #I64.t right_index))
  ensures PA.pts_to_mask values (if take then S.upd vs target (Some right) else vs) (fun _ -> True) **
    PA.pts_to_mask indices
      (if take then S.upd is target (Some (Ghost.reveal #I64.t right_index)) else is) (fun _ -> True)
{
  if (take) {
    let ri = PA.mask_read indices source;
    assert pure (ri == right_index);
    PA.mask_write values target right;
    PA.mask_write indices target ri;
    rewrite PA.pts_to_mask indices (S.upd is target (Some ri)) (fun _ -> True) as
      PA.pts_to_mask indices (S.upd is target (Some (Ghost.reveal #I64.t right_index))) (fun _ -> True);
  };
}

inline_for_extraction noextract
fn pg_selector_reduce (#n:erased pos)
  (values:larray F32.t n) (indices:larray I64.t n)
  (width:szp{pg_sel_reduce_width width}) (base:sz{base + width <= n})
  (#vs:erased (lseq (option F32.t) n)) (#is:erased (lseq (option I64.t) n))
  (#scores:erased (Shared.vector F32.t 151936)) (#group:erased (Shared.token_id -> nat))
  (#origin:erased nat) (#decode:erased (int -> int)) (#valid:erased (int -> prop))
  preserves gpu
  requires PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
    pure (pg_sel_forest vs is base width width scores group origin decode valid)
  ensures exists* (av:lseq (option F32.t) n) (ai:lseq (option I64.t) n).
    PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
    pure (pg_sel_forest av ai base width 1 scores group origin decode valid /\
      pg_sel_outside vs av base width /\ pg_sel_outside is ai base width)
{
  let mut active = width;
  while (!active >^ 1sz)
    invariant exists* (a:szp{pg_sel_active width a})
      (av:lseq (option F32.t) n) (ai:lseq (option I64.t) n).
      active |-> a ** PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
      pure (pg_sel_forest av ai base width a scores group origin decode valid /\
        pg_sel_outside vs av base width /\ pg_sel_outside is ai base width)
    decreases (SZ.v !active)
  {
    let a = !active;
    let offset = a /^ 2sz;
    assert pure (0 < SZ.v offset /\ SZ.v a == 2 * SZ.v offset /\ pg_sel_active width offset);
    let mut lane = 0sz;
    while (!lane <^ offset)
      invariant exists* (l:sz{l <= offset}) (av:lseq (option F32.t) n) (ai:lseq (option I64.t) n).
        lane |-> l ** PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
        pure (pg_sel_partial av ai base width a l scores group origin decode valid /\
          pg_sel_outside vs av base width /\ pg_sel_outside is ai base width)
      decreases (offset - !lane)
    {
      let l = !lane;
      with before_v. assert PA.pts_to_mask values before_v (fun _ -> True);
      with before_i. assert PA.pts_to_mask indices before_i (fun _ -> True);
      pg_sel_partial_cell before_v before_i base width a l scores group origin decode valid l;
      pg_sel_partial_cell before_v before_i base width a l scores group origin decode valid (l + offset);
      let left = PA.mask_read values (base +^ l);
      let right = PA.mask_read values (base +^ l +^ offset);
      let right_index = hide #I64.t (Some?.v (S.index before_i (base + l + offset)));
      let take = F.lt left right;
      pg_sel_maybe_store values indices (base +^ l) (base +^ l +^ offset) take right
        #before_v #before_i #right_index;
      with after_v. assert PA.pts_to_mask values after_v (fun _ -> True);
      with after_i. assert PA.pts_to_mask indices after_i (fun _ -> True);
      let li = hide (Some?.v (S.index before_i (base + l)));
      let ri = hide (Some?.v (S.index before_i (base + l + offset)));
      pg_sel_reduce_step before_v before_i base width a l scores group origin decode valid left right li ri;
      lane := l +^ 1sz;
    };
    dec_source_loop_end !lane offset;
    active := offset;
  };
  assert pure (SZ.v !active == 1);
}

inline_for_extraction noextract
fn pg_scan_maybe_store
  (values:larray F32.t 256) (indices:larray I64.t 256)
  (thread:szlt 128) (at:szlt 1600) (take:bool) (candidate:F32.t)
  (#vs:erased (lseq (option F32.t) 256)) (#is:erased (lseq (option I64.t) 256))
  preserves gpu
  requires PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True)
  ensures PA.pts_to_mask values (if take then S.upd vs thread (Some candidate) else vs) (fun _ -> True) **
    PA.pts_to_mask indices
      (if take then S.upd is thread (Some (FStar.Int.Cast.uint32_to_int64 (SZ.sizet_to_u32 at))) else is) (fun _ -> True)
{
  if (take) {
    let index64 = FStar.Int.Cast.uint32_to_int64 (SZ.sizet_to_u32 at);
    PA.mask_write values thread candidate;
    PA.mask_write indices thread index64;
  };
}

inline_for_extraction noextract
fn pg_scan_thread
  (logits:larray BF16.t 1228800) (values:larray F32.t 256) (indices:larray I64.t 256)
  (chunk:szlt 96) (thread:szlt 128)
  (#xs:erased (lseq BF16.t 1228800)) (#fv:perm)
  (#old_v:erased (lseq (option F32.t) 256)) (#old_i:erased (lseq (option I64.t) 256))
  preserves gpu ** logits |-> Frac fv xs
  requires PA.pts_to_mask values old_v (fun _ -> True) ** PA.pts_to_mask indices old_i (fun _ -> True)
  ensures exists* (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256).
    PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
    pure (pg_sel_cell #256 vs is thread (pg_selector_scores xs)
      (pg_thread_leaf chunk thread) (pg_local_decode chunk) pg_local_valid /\
      pg_sel_outside old_v vs thread 1 /\ pg_sel_outside old_i is thread 1)
{
  PA.mask_write values thread pg_selector_initial;
  PA.mask_write indices thread (-1L);
  let mut cursor:sz = thread;
  while (!cursor <^ 1600sz && chunk *^ 1600sz +^ !cursor <^ SZ.uint32_to_sizet 151936ul)
    invariant exists* (at:sz{thread <= at /\ at <= thread + 1664 /\ (at - thread) % 128 == 0})
      (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256).
      cursor |-> at ** PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
      pure (pg_sel_cell #256 vs is thread (pg_selector_scores xs) (pg_thread_mask chunk thread at)
        (pg_local_decode chunk) pg_local_valid /\
        pg_sel_outside old_v vs thread 1 /\ pg_sel_outside old_i is thread 1)
    decreases (1728 - !cursor)
  {
    let at = !cursor;
    with before_v. assert PA.pts_to_mask values before_v (fun _ -> True);
    with before_i. assert PA.pts_to_mask indices before_i (fun _ -> True);
    let current = PA.mask_read values thread;
    let previous_index = hide (Some?.v (S.index before_i thread));
    let raw = logits.(SZ.uint32_to_sizet 921600ul +^ chunk *^ 1600sz +^ at);
    let candidate = Cast.cast_bf16_to_f32 raw;
    pg_selector_scan (pg_selector_scores xs) (pg_thread_mask chunk thread at) current
      (pg_local_decode chunk (I64.v previous_index)) (chunk * 1600 + at);
    pg_scan_mask_step chunk thread at;
    let take = F.lt current candidate;
    pg_scan_maybe_store values indices thread at take candidate #before_v #before_i;
    cursor := at +^ 128sz;
  };
  let finished = !cursor;
  pg_scan_mask_end chunk thread finished;
  ()
}
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
fn pg_selector_chunk (logits:larray BF16.t 1228800)
  (values:larray F32.t 256) (indices:larray I64.t 256) (chunk:szlt 96)
  (#xs:erased (lseq BF16.t 1228800)) (#fraction:perm)
  (#old_v:erased (lseq (option F32.t) 256)) (#old_i:erased (lseq (option I64.t) 256))
  preserves gpu ** logits |-> Frac fraction xs
  requires PA.pts_to_mask values old_v (fun _ -> True) ** PA.pts_to_mask indices old_i (fun _ -> True)
  returns result:(F32.t & I64.t)
  ensures (exists* (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256).
    PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True)) **
    pure (pg_local_valid (I64.v result._2) /\
      pg_selector_best (pg_selector_scores xs) (pg_vocabulary_chunk chunk)
        result._1 (pg_local_decode chunk (I64.v result._2)))
{
  let mut thread = 0sz;
  while (!thread <^ 128sz)
    invariant exists* (t:sz{t <= 128}) (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256).
      thread |-> t ** PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
      pure (pg_chunk_scanned vs is (pg_selector_scores xs) chunk t)
    decreases (128 - !thread)
  {
    let t = !thread;
    with vs. assert PA.pts_to_mask values vs (fun _ -> True);
    with is. assert PA.pts_to_mask indices is (fun _ -> True);
    pg_scan_thread logits values indices chunk t #xs #fraction #vs #is;
    thread := t +^ 1sz;
  };
  dec_source_loop_end !thread 128sz;
  let mut warp = 0sz;
  while (!warp <^ 4sz)
    invariant exists* (w:sz{w <= 4}) (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256).
      warp |-> w ** PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
      pure (pg_chunk_warps vs is (pg_selector_scores xs) chunk w)
    decreases (4 - !warp)
  {
    let w = !warp;
    with before_v. assert PA.pts_to_mask values before_v (fun _ -> True);
    with before_i. assert PA.pts_to_mask indices before_i (fun _ -> True);
    pg_chunk_warp_start before_v before_i (pg_selector_scores xs) chunk w;
    pg_selector_reduce values indices 32sz (w *^ 32sz) #before_v #before_i
      #(hide (pg_selector_scores xs)) #(hide (pg_thread_group chunk)) #(hide (w * 32))
      #(hide (pg_local_decode chunk)) #(hide pg_local_valid);
    with after_v. assert PA.pts_to_mask values after_v (fun _ -> True);
    with after_i. assert PA.pts_to_mask indices after_i (fun _ -> True);
    assert pure (pg_sel_cell #256 after_v after_i (w * 32) (pg_selector_scores xs)
      (pg_sel_cover (pg_thread_group chunk) (w * 32) 32 1 0) (pg_local_decode chunk) pg_local_valid);
    let value = PA.mask_read values (w *^ 32sz);
    let index = PA.mask_read indices (w *^ 32sz);
    pg_chunk_compress before_v after_v before_i after_i (pg_selector_scores xs) chunk w value index;
    PA.mask_write values w value;
    PA.mask_write indices w index;
    warp := w +^ 1sz;
  };
  dec_source_loop_end !warp 4sz;
  with vs. assert PA.pts_to_mask values vs (fun _ -> True);
  with is. assert PA.pts_to_mask indices is (fun _ -> True);
  pg_chunk_merge_start vs is (pg_selector_scores xs) chunk;
  pg_selector_reduce values indices 4sz 0sz #vs #is
    #(hide (pg_selector_scores xs)) #(hide (pg_chunk_group chunk)) #(hide 0)
    #(hide (pg_local_decode chunk)) #(hide pg_local_valid);
  with final_v. assert PA.pts_to_mask values final_v (fun _ -> True);
  with final_i. assert PA.pts_to_mask indices final_i (fun _ -> True);
  assert pure (pg_sel_cell #256 final_v final_i 0 (pg_selector_scores xs)
    (pg_sel_cover (pg_chunk_group chunk) 0 4 1 0) (pg_local_decode chunk) pg_local_valid);
  let value = PA.mask_read values 0sz;
  let index = PA.mask_read indices 0sz;
  pg_selector_mask_equal (pg_selector_scores xs) (pg_sel_cover (pg_chunk_group chunk) 0 4 1 0)
    (pg_vocabulary_chunk chunk) value (pg_local_decode chunk (I64.v index));
  (value,index)
}

inline_for_extraction noextract
fn pg_selector_chunks (logits:larray BF16.t 1228800)
  (values:larray F32.t 256) (indices:larray I64.t 256)
  (chunk_values:larray F32.t 96) (chunk_indices:larray I64.t 96)
  (#xs:erased (lseq BF16.t 1228800)) (#fraction:perm)
  (#vv:erased (lseq (option F32.t) 256)) (#ii:erased (lseq (option I64.t) 256))
  (#cv:erased (lseq (option F32.t) 96)) (#ci:erased (lseq (option I64.t) 96))
  preserves gpu ** logits |-> Frac fraction xs
  requires PA.pts_to_mask values vv (fun _ -> True) ** PA.pts_to_mask indices ii (fun _ -> True) **
    PA.pts_to_mask chunk_values cv (fun _ -> True) ** PA.pts_to_mask chunk_indices ci (fun _ -> True)
  ensures exists* (av:lseq (option F32.t) 256) (ai:lseq (option I64.t) 256)
    (bv:lseq (option F32.t) 96) (bi:lseq (option I64.t) 96).
    PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
    PA.pts_to_mask chunk_values bv (fun _ -> True) ** PA.pts_to_mask chunk_indices bi (fun _ -> True) **
    pure (pg_chunks_ready bv bi (pg_selector_scores xs) 96)
{
  let mut chunk = 0sz;
  while (!chunk <^ 96sz)
    invariant exists* (c:sz{c <= 96})
      (av:lseq (option F32.t) 256) (ai:lseq (option I64.t) 256)
      (bv:lseq (option F32.t) 96) (bi:lseq (option I64.t) 96).
      chunk |-> c ** PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
      PA.pts_to_mask chunk_values bv (fun _ -> True) ** PA.pts_to_mask chunk_indices bi (fun _ -> True) **
      pure (pg_chunks_ready bv bi (pg_selector_scores xs) c)
    decreases (96 - !chunk)
  {
    let c = !chunk;
    with av. assert PA.pts_to_mask values av (fun _ -> True);
    with ai. assert PA.pts_to_mask indices ai (fun _ -> True);
    let result = pg_selector_chunk logits values indices c #xs #fraction #av #ai;
    PA.mask_write chunk_values c result._1;
    PA.mask_write chunk_indices c result._2;
    chunk := c +^ 1sz;
  };
  dec_source_loop_end !chunk 96sz;
}

inline_for_extraction noextract
fn pg_pack_maybe_store
  (values:larray F32.t 256) (indices:larray I64.t 256)
  (chunk_indices:larray I64.t 96) (thread:szlt 96) (take:bool) (candidate:F32.t)
  (#vs:erased (lseq (option F32.t) 256)) (#is:erased (lseq (option I64.t) 256))
  (#ci:erased (lseq (option I64.t) 96)) (#index:erased pg_local_index)
  preserves gpu ** PA.pts_to_mask chunk_indices ci (fun _ -> True)
  requires PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
    pure (S.index ci thread == Some (Ghost.reveal #pg_local_index index) /\
      (take ==> 0 <= I64.v index /\ I64.v index < 1600))
  ensures PA.pts_to_mask values (if take then S.upd vs thread (Some candidate) else vs) (fun _ -> True) **
    PA.pts_to_mask indices
      (if take then S.upd is thread (Some (I64.int_to_t (SZ.v thread * 4294967296 + I64.v index)))
       else is) (fun _ -> True)
{
  if (take) {
    PA.mask_write values thread candidate;
    let actual_index = PA.mask_read chunk_indices thread;
    let packed = pg_pack_index thread actual_index;
    PA.mask_write indices thread packed;
    rewrite PA.pts_to_mask indices (S.upd is thread (Some packed)) (fun _ -> True) as
      PA.pts_to_mask indices
        (S.upd is thread (Some (I64.int_to_t (SZ.v thread * 4294967296 + I64.v index)))) (fun _ -> True);
  };
}

inline_for_extraction noextract
fn pg_initialize_leaf
  (values:larray F32.t 256) (indices:larray I64.t 256)
  (chunk_values:larray F32.t 96) (chunk_indices:larray I64.t 96)
  (thread:szlt 256)
  (#vs:erased (lseq (option F32.t) 256)) (#is:erased (lseq (option I64.t) 256))
  (#cv:erased (lseq (option F32.t) 96)) (#ci:erased (lseq (option I64.t) 96))
  (#scores:erased (Shared.vector F32.t 151936))
  preserves gpu ** PA.pts_to_mask chunk_values cv (fun _ -> True) ** PA.pts_to_mask chunk_indices ci (fun _ -> True)
  requires PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
    pure (pg_chunks_ready cv ci scores 96)
  ensures exists* (av:lseq (option F32.t) 256) (ai:lseq (option I64.t) 256).
    PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
    pure (pg_sel_cell #256 av ai thread scores (pg_vocabulary_chunk thread) pg_packed_decode pg_packed_valid /\
      pg_sel_outside vs av thread 1 /\ pg_sel_outside is ai thread 1)
{
  PA.mask_write values thread pg_selector_initial;
  PA.mask_write indices thread (-1L);
  if (thread <^ 96sz) {
    assert pure (pg_sel_cell #96 cv ci thread scores (pg_vocabulary_chunk thread)
      (pg_local_decode thread) pg_local_valid);
    let candidate = PA.mask_read chunk_values thread;
    let index = hide #pg_local_index (Some?.v (S.index ci thread));
    pg_chunk_pack_related scores thread candidate index;
    let take = F.lt pg_selector_initial candidate;
    with av. assert PA.pts_to_mask values av (fun _ -> True);
    with ai. assert PA.pts_to_mask indices ai (fun _ -> True);
    pg_pack_maybe_store values indices chunk_indices thread take candidate #av #ai #ci #index;
  };
}

inline_for_extraction noextract
fn pg_initialize_leaves
  (values:larray F32.t 256) (indices:larray I64.t 256)
  (chunk_values:larray F32.t 96) (chunk_indices:larray I64.t 96)
  (#vs:erased (lseq (option F32.t) 256)) (#is:erased (lseq (option I64.t) 256))
  (#cv:erased (lseq (option F32.t) 96)) (#ci:erased (lseq (option I64.t) 96))
  (#scores:erased (Shared.vector F32.t 151936))
  preserves gpu ** PA.pts_to_mask chunk_values cv (fun _ -> True) ** PA.pts_to_mask chunk_indices ci (fun _ -> True)
  requires PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
    pure (pg_chunks_ready cv ci scores 96)
  ensures exists* (av:lseq (option F32.t) 256) (ai:lseq (option I64.t) 256).
    PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
    pure (pg_final_leaves av ai scores 256)
{
  let mut thread = 0sz;
  while (!thread <^ 256sz)
    invariant exists* (t:sz{t <= 256}) (av:lseq (option F32.t) 256) (ai:lseq (option I64.t) 256).
      thread |-> t ** PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
      pure (pg_final_leaves av ai scores t)
    decreases (256 - !thread)
  {
    let t = !thread;
    with av. assert PA.pts_to_mask values av (fun _ -> True);
    with ai. assert PA.pts_to_mask indices ai (fun _ -> True);
    pg_initialize_leaf values indices chunk_values chunk_indices t #av #ai #cv #ci #scores;
    thread := t +^ 1sz;
  };
  dec_source_loop_end !thread 256sz;
}
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
fn pg_selector_final_reduce (values:larray F32.t 256) (indices:larray I64.t 256)
  (#vs:erased (lseq (option F32.t) 256)) (#is:erased (lseq (option I64.t) 256))
  (#scores:erased (Shared.vector F32.t 151936))
  preserves gpu
  requires PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
    pure (pg_final_leaves vs is scores 256)
  ensures exists* (av:lseq (option F32.t) 256) (ai:lseq (option I64.t) 256).
    PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
    pure (pg_sel_cell #256 av ai 0 scores (fun _ -> True) pg_packed_decode pg_packed_valid)
{
  let mut warp = 0sz;
  while (!warp <^ 8sz)
    invariant exists* (w:sz{w <= 8}) (av:lseq (option F32.t) 256) (ai:lseq (option I64.t) 256).
      warp |-> w ** PA.pts_to_mask values av (fun _ -> True) ** PA.pts_to_mask indices ai (fun _ -> True) **
      pure (pg_final_warps av ai scores w)
    decreases (8 - !warp)
  {
    let w = !warp;
    with before_v. assert PA.pts_to_mask values before_v (fun _ -> True);
    with before_i. assert PA.pts_to_mask indices before_i (fun _ -> True);
    pg_final_warp_start before_v before_i scores w;
    pg_selector_reduce values indices 32sz (w *^ 32sz) #before_v #before_i
      #scores #(hide pg_final_chunk_group) #(hide (SZ.v w * 32))
      #(hide pg_packed_decode) #(hide pg_packed_valid);
    with after_v. assert PA.pts_to_mask values after_v (fun _ -> True);
    with after_i. assert PA.pts_to_mask indices after_i (fun _ -> True);
    assert pure (pg_sel_cell #256 after_v after_i (w * 32) scores
      (pg_sel_cover pg_final_chunk_group (w * 32) 32 1 0) pg_packed_decode pg_packed_valid);
    let value = PA.mask_read values (w *^ 32sz);
    let index = PA.mask_read indices (w *^ 32sz);
    pg_final_compress before_v after_v before_i after_i scores w value index;
    PA.mask_write values w value;
    PA.mask_write indices w index;
    warp := w +^ 1sz;
  };
  dec_source_loop_end !warp 8sz;
  with av. assert PA.pts_to_mask values av (fun _ -> True);
  with ai. assert PA.pts_to_mask indices ai (fun _ -> True);
  pg_final_merge_start av ai scores;
  pg_selector_reduce values indices 8sz 0sz #av #ai
    #scores #(hide pg_final_group) #(hide 0)
    #(hide pg_packed_decode) #(hide pg_packed_valid);
  with bv. assert PA.pts_to_mask values bv (fun _ -> True);
  with bi. assert PA.pts_to_mask indices bi (fun _ -> True);
  assert pure (pg_sel_cell #256 bv bi 0 scores
    (pg_sel_cover pg_final_group 0 8 1 0) pg_packed_decode pg_packed_valid);
  pg_sel_cell_mask bv bi 0 scores (pg_sel_cover pg_final_group 0 8 1 0)
    (fun _ -> True) pg_packed_decode pg_packed_valid;
}

inline_for_extraction noextract
fn pg_selector_body (logits:larray BF16.t 1228800) (prompt:larray I64.t 7) (output:larray I64.t 8)
  (#xs:erased (lseq BF16.t 1228800)) (#ps:erased (lseq I64.t 7))
  (#old:erased (lseq I64.t 8)) (#fx #fp:perm)
  preserves gpu ** logits |-> Frac fx xs ** prompt |-> Frac fp ps
  requires output |-> old
  ensures exists* (ys:lseq I64.t 8). output |-> ys ** pure (pg_selector_result xs ps ys)
{
  let mut values:PA.array F32.t = [| 256sz |];
  let mut chunk_values:PA.array F32.t = [| 96sz |];
  let mut indices:PA.array I64.t = [| 256sz |];
  let mut chunk_indices:PA.array I64.t = [| 96sz |];
  unfold PA.pts_to_uninit values 256;
  unfold PA.pts_to_uninit chunk_values 96;
  unfold PA.pts_to_uninit indices 256;
  unfold PA.pts_to_uninit chunk_indices 96;
  with vv. assert PA.pts_to_mask values vv (fun _ -> True);
  with ii. assert PA.pts_to_mask indices ii (fun _ -> True);
  with cv. assert PA.pts_to_mask chunk_values cv (fun _ -> True);
  with ci. assert PA.pts_to_mask chunk_indices ci (fun _ -> True);
  PA.pts_to_mask_len values; PA.pts_to_mask_len indices;
  PA.pts_to_mask_len chunk_values; PA.pts_to_mask_len chunk_indices;
  let vv0 = hide #(lseq (option F32.t) 256) (Ghost.reveal #(S.seq (option F32.t)) vv);
  let ii0 = hide #(lseq (option I64.t) 256) (Ghost.reveal #(S.seq (option I64.t)) ii);
  let cv0 = hide #(lseq (option F32.t) 96) (Ghost.reveal #(S.seq (option F32.t)) cv);
  let ci0 = hide #(lseq (option I64.t) 96) (Ghost.reveal #(S.seq (option I64.t)) ci);
  rewrite PA.pts_to_mask values vv (fun _ -> True) as PA.pts_to_mask values vv0 (fun _ -> True);
  rewrite PA.pts_to_mask indices ii (fun _ -> True) as PA.pts_to_mask indices ii0 (fun _ -> True);
  rewrite PA.pts_to_mask chunk_values cv (fun _ -> True) as PA.pts_to_mask chunk_values cv0 (fun _ -> True);
  rewrite PA.pts_to_mask chunk_indices ci (fun _ -> True) as PA.pts_to_mask chunk_indices ci0 (fun _ -> True);
  pg_selector_chunks logits values indices chunk_values chunk_indices #xs #fx #vv0 #ii0 #cv0 #ci0;
  with vv1. assert PA.pts_to_mask values vv1 (fun _ -> True);
  with ii1. assert PA.pts_to_mask indices ii1 (fun _ -> True);
  with cv1. assert PA.pts_to_mask chunk_values cv1 (fun _ -> True);
  with ci1. assert PA.pts_to_mask chunk_indices ci1 (fun _ -> True);
  pg_initialize_leaves values indices chunk_values chunk_indices #vv1 #ii1 #cv1 #ci1 #(hide (pg_selector_scores xs));
  with vv2. assert PA.pts_to_mask values vv2 (fun _ -> True);
  with ii2. assert PA.pts_to_mask indices ii2 (fun _ -> True);
  pg_selector_final_reduce values indices #vv2 #ii2 #(hide (pg_selector_scores xs));
  let mut i = 0sz;
  while (!i <^ 7sz)
    invariant exists* (n:sz{n <= 7}) (ys:lseq I64.t 8).
      i |-> n ** output |-> ys ** pure (forall (j:Shared.index 7{j < n}). S.index ys j == S.index ps j)
    decreases (7 - !i)
  {
    let at = !i;
    let value = prompt.(at);
    output.(at) <- value;
    i := at +^ 1sz;
  };
  dec_source_loop_end !i 7sz;
  with vv3. assert PA.pts_to_mask values vv3 (fun _ -> True);
  with ii3. assert PA.pts_to_mask indices ii3 (fun _ -> True);
  let packed = PA.mask_read indices 0sz;
  let value = hide #F32.t (Some?.v (S.index vv3 0));
  let selected = pg_decode_index packed;
  assert pure (pg_selector_best (pg_selector_scores xs) (fun _ -> True) value (I64.v selected));
  output.(7sz) <- selected;
  fold PA.pts_to_uninit values 256;
  fold PA.pts_to_uninit chunk_values 96;
  fold PA.pts_to_uninit indices 256;
  fold PA.pts_to_uninit chunk_indices 96;
}

inline_for_extraction noextract
fn pg_selector_call (logits:larray BF16.t 1228800) (prompt:larray I64.t 7) (output:larray I64.t 8)
  (#xs:erased (lseq BF16.t 1228800)) (#ps:erased (lseq I64.t 7)) (#old:erased (lseq I64.t 8)) ()
  norewrite
  preserves gpu
  requires pg_selector_pre logits prompt output xs ps old
  ensures pg_selector_post logits prompt output xs ps
{
  unfold pg_selector_pre logits prompt output xs ps old;
  pg_selector_body logits prompt output #xs #ps #old #1.0R #1.0R;
  fold pg_selector_post logits prompt output xs ps;
}

inline_for_extraction noextract
let pg_selector_descriptor
  (logits:larray BF16.t 1228800{KA.is_global_array logits})
  (prompt:larray I64.t 7{KA.is_global_array prompt}) (output:larray I64.t 8{KA.is_global_array output})
  (#xs:erased (lseq BF16.t 1228800)) (#ps:erased (lseq I64.t 7)) (#old:erased (lseq I64.t 8))
  : k:KS.kernel_desc (pg_selector_pre logits prompt output xs ps old) (pg_selector_post logits prompt output xs ps){
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)} =
  let raw:KS.kernel_desc_1_1 (pg_selector_pre logits prompt output xs ps old)
      (pg_selector_post logits prompt output xs ps) = {
    f = pg_selector_call logits prompt output #xs #ps #old;
    full_pre_sendable = solve; full_post_sendable = solve
  } in opt_scalar_descriptor #(pg_selector_pre logits prompt output xs ps old)
    #(pg_selector_post logits prompt output xs ps) raw

inline_for_extraction noextract
ghost
fn pg_adopt_suffix (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  requires pg_after_stage weights_address prompt_address buffers weights prompt 253
  ensures on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 0)
{
  ghost_impersonate gpu_loc
    (pg_after_stage weights_address prompt_address buffers weights prompt 253)
    (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 0))
    fn () {
      unfold pg_after_stage weights_address prompt_address buffers weights prompt 253;
      on_elim (pg_pending weights_address buffers weights prompt 253 **
        (PA.pts_to prompt_address prompt ** RT.contents buffers.token_buffer));
      unfold pg_pending weights_address buffers weights prompt 253;
      with v. assert pg_core weights_address buffers weights prompt v 253;
      unfold pg_core weights_address buffers weights prompt v 253;
      unfold RT.contents buffers.token_buffer;
      with raw. assert on gpu_loc (PA.pts_to (RT.data buffers.token_buffer) raw);
      on_elim (PA.pts_to (RT.data buffers.token_buffer) raw);
      PA.pts_to_len (RT.data buffers.token_buffer);
      let tokens = hide #(lseq I64.t 8) (Ghost.reveal #(S.seq I64.t) raw);
      rewrite PA.pts_to (RT.data buffers.token_buffer) raw as PA.pts_to (RT.data buffers.token_buffer) tokens;
      fold pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 0;
      intro_exists_erased #pg_suffix_observation
        (fun observed -> pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 0)
        (hide #pg_suffix_observation (Ghost.reveal #pg_values v,Ghost.reveal #(lseq I64.t 8) tokens));
      fold pg_suffix_pending weights_address prompt_address buffers weights prompt 0;
      on_intro (pg_suffix_pending weights_address prompt_address buffers weights prompt 0);
    };
}

inline_for_extraction noextract
fn pg_materialize_hidden_prefix (input:larray BF16.t 8192)
  (#x:erased (lseq BF16.t 8192))
  preserves gpu ** input |-> x
  returns prefix:PA.array BF16.t
  ensures rewrites_to prefix (pg_hidden_prefix input)
  ensures pure (PA.length prefix == 7168)
{
  PA.to_mask input;
  with mask. assert PA.pts_to_mask input mask (fun _ -> True);
  PA.split_mask input #1.0R #mask #(fun _ -> True) (fun _ -> False);
  PA.mask_mext input #1.0R #mask #(PA.mask_isect (fun _ -> True) (fun _ -> False)) (fun _ -> False);
  let prefix = PA.sub input #1.0R #(fun _ -> False) 0sz (hide 7168) #mask;
  PA.mask_mext prefix (fun _ -> False);
  drop_ (PA.pts_to_mask prefix #1.0R (S.slice mask 0 7168) (fun _ -> False));
  PA.mask_mext input #1.0R #mask
    #(fun k -> False /\ ~(0 <= k /\ k < 7168)) (fun _ -> False);
  drop_ (PA.pts_to_mask input #1.0R mask (fun _ -> False));
  PA.mask_mext input #1.0R #mask #(PA.mask_diff (fun _ -> True) (fun _ -> False)) (fun _ -> True);
  PA.from_mask input;
  with restored. assert input |-> restored;
  assert pure (S.equal restored x);
  S.lemma_eq_elim restored x;
  prefix
}

inline_for_extraction noextract
fn pg_final_rms_call (weights:larray BF16.t opt_weight_count) (input:larray BF16.t 8192)
  (output:larray BF16.t 7168)
  (#wv:erased (lseq BF16.t opt_weight_count)) (#x:erased (lseq BF16.t 8192))
  (#old:erased (lseq BF16.t 7168)) ()
  norewrite
  preserves gpu
  requires pg_final_rms_pre weights input output wv x old
  ensures pg_final_rms_post weights input output wv x
{
  unfold pg_final_rms_pre weights input output wv x old;
  let prefix = pg_materialize_hidden_prefix input #x;
  pg_final_rms_body weights input prefix output #wv #x #old #1.0R #1.0R;
  fold pg_final_rms_post weights input output wv x;
}

inline_for_extraction noextract
let pg_final_rms_descriptor
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (input:larray BF16.t 8192{KA.is_global_array input})
  (output:larray BF16.t 7168{KA.is_global_array output})
  (#wv:erased (lseq BF16.t opt_weight_count)) (#x:erased (lseq BF16.t 8192))
  (#old:erased (lseq BF16.t 7168))
  : k:KS.kernel_desc (pg_final_rms_pre weights input output wv x old) (pg_final_rms_post weights input output wv x){
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)} =
  let raw:KS.kernel_desc_1_1 (pg_final_rms_pre weights input output wv x old)
      (pg_final_rms_post weights input output wv x) = {
    f = pg_final_rms_call weights input output #wv #x #old;
    full_pre_sendable = solve; full_post_sendable = solve
  } in opt_scalar_descriptor #(pg_final_rms_pre weights input output wv x old)
    #(pg_final_rms_post weights input output wv x) raw

inline_for_extraction noextract
ghost
fn pg_suffix_pair_open (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) (step:nat{step <= 2})
  (source:Shared.index 256) (target:Shared.index 256{source <> target}) ()
  norewrite
  requires pg_suffix_core weights_address prompt_address buffers weights prompt v tokens step
  ensures PA.pts_to weights_address weights **
    PA.pts_to (pg_address buffers source) (pg_slot v source) **
    PA.pts_to (pg_address buffers target) (pg_slot v target) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens step source target
{
  unfold pg_suffix_core weights_address prompt_address buffers weights prompt v tokens step;
  pg_split2 buffers #v source target;
  fold pg_suffix_pair_frame prompt_address buffers weights prompt v tokens step source target;
}

inline_for_extraction noextract
ghost
fn pg_suffix_pair_close (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) (step:nat{step < 2})
  (source:Shared.index 256) (target:Shared.index 256{source <> target})
  (#ys:erased (lseq BF16.t (pg_size target))) ()
  norewrite
  requires PA.pts_to weights_address weights **
    PA.pts_to (pg_address buffers source) (pg_slot v source) ** PA.pts_to (pg_address buffers target) ys **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens step source target **
    pure (pg_suffix_arithmetic weights prompt (pg_replace v target ys) tokens (step + 1))
  ensures pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1)
{
  unfold pg_suffix_pair_frame prompt_address buffers weights prompt v tokens step source target;
  pg_join2 buffers #v source target #ys;
  fold pg_suffix_core weights_address prompt_address buffers weights prompt (pg_replace v target ys) tokens (step + 1);
  intro_exists_erased #pg_suffix_observation
    (fun observed -> pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) (step + 1))
    (hide #pg_suffix_observation (pg_replace v target ys,Ghost.reveal #(lseq I64.t 8) tokens));
  fold pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1);
}

inline_for_extraction noextract
ghost
fn pg_final_rms_before (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 8192{input == pg_address buffers 253})
  (output:larray BF16.t 7168{output == pg_address buffers 254})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) ()
  norewrite
  requires pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 0
  ensures pg_final_rms_pre weights_address input output weights (pg_slot v 253) (pg_slot v 254) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 0 253 254
{
  pg_suffix_pair_open weights_address prompt_address buffers #weights #prompt #v #tokens 0 253 254 ();
  rewrite PA.pts_to (pg_address buffers 253) (pg_slot v 253) as input |-> pg_slot v 253;
  rewrite PA.pts_to (pg_address buffers 254) (pg_slot v 254) as output |-> pg_slot v 254;
  fold pg_final_rms_pre weights_address input output weights (pg_slot v 253) (pg_slot v 254);
}

inline_for_extraction noextract
ghost
fn pg_final_rms_after (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 8192{input == pg_address buffers 253})
  (output:larray BF16.t 7168{output == pg_address buffers 254})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) ()
  norewrite
  requires pg_final_rms_post weights_address input output weights (pg_slot v 253) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 0 253 254
  ensures pg_suffix_pending weights_address prompt_address buffers weights prompt 1
{
  unfold pg_final_rms_post weights_address input output weights (pg_slot v 253);
  with ys. assert output |-> ys;
  unfold pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 0 253 254;
  pg_suffix_rms_advance weights prompt v tokens ys;
  fold pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 0 253 254;
  rewrite input |-> pg_slot v 253 as PA.pts_to (pg_address buffers 253) (pg_slot v 253);
  rewrite output |-> ys as PA.pts_to (pg_address buffers 254) ys;
  pg_suffix_pair_close weights_address prompt_address buffers #weights #prompt #v #tokens 0 253 254 #ys ();
}

inline_for_extraction noextract
ghost
fn pg_final_projection_before (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 7168{input == pg_address buffers 254})
  (output:larray BF16.t 1228800{output == pg_address buffers 255})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) ()
  norewrite
  requires pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 1
  ensures pg_projection_pre weights_address input output weights (pg_slot v 254) (pg_slot v 255) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 1 254 255
{
  pg_suffix_pair_open weights_address prompt_address buffers #weights #prompt #v #tokens 1 254 255 ();
  rewrite PA.pts_to (pg_address buffers 254) (pg_slot v 254) as input |-> pg_slot v 254;
  rewrite PA.pts_to (pg_address buffers 255) (pg_slot v 255) as output |-> pg_slot v 255;
  fold pg_projection_pre weights_address input output weights (pg_slot v 254) (pg_slot v 255);
}

inline_for_extraction noextract
ghost
fn pg_final_projection_after (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 7168{input == pg_address buffers 254})
  (output:larray BF16.t 1228800{output == pg_address buffers 255})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) ()
  norewrite
  requires pg_projection_post weights_address input output weights (pg_slot v 254) (pg_slot v 255) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 1 254 255
  ensures pg_suffix_pending weights_address prompt_address buffers weights prompt 2
{
  unfold pg_projection_post weights_address input output weights (pg_slot v 254) (pg_slot v 255);
  unfold pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 1 254 255;
  pg_suffix_projection_advance weights prompt v tokens;
  fold pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 1 254 255;
  rewrite input |-> pg_slot v 254 as PA.pts_to (pg_address buffers 254) (pg_slot v 254);
  rewrite output |-> pg_projection_output weights (pg_slot v 254) (pg_slot v 255) as
    PA.pts_to (pg_address buffers 255) (pg_projection_output weights (pg_slot v 254) (pg_slot v 255));
  pg_suffix_pair_close weights_address prompt_address buffers #weights #prompt #v #tokens 1 254 255
    #(hide (pg_projection_output weights (pg_slot v 254) (pg_slot v 255))) ();
}

inline_for_extraction noextract
ghost
fn pg_final_selector_before (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (logits:larray BF16.t 1228800{logits == pg_address buffers 255})
  (output:larray I64.t 8{output == RT.data buffers.token_buffer})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) ()
  norewrite
  requires pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 2
  ensures pg_selector_pre logits prompt_address output (pg_slot v 255) prompt tokens **
    pg_selector_stage_frame weights_address buffers weights prompt v tokens
{
  unfold pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 2;
  rewrite pg_storage buffers v 1.0R as pg_masked_storage buffers v 1.0R (fun _ -> True);
  pg_take buffers #v #1.0R (fun _ -> True) 255;
  rewrite pg_masked_storage buffers v 1.0R (fun i -> True /\ ~(eq2 #(Shared.index 256) i 255))
    as pg_masked_storage buffers v 1.0R (fun i -> i <> 255);
  rewrite PA.pts_to (pg_address buffers 255) (pg_slot v 255) as logits |-> pg_slot v 255;
  rewrite PA.pts_to (RT.data buffers.token_buffer) tokens as output |-> tokens;
  fold pg_selector_pre logits prompt_address output (pg_slot v 255) prompt tokens;
  fold pg_selector_stage_frame weights_address buffers weights prompt v tokens;
}

inline_for_extraction noextract
ghost
fn pg_final_selector_after (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (logits:larray BF16.t 1228800{logits == pg_address buffers 255})
  (output:larray I64.t 8{output == RT.data buffers.token_buffer})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) ()
  norewrite
  requires pg_selector_post logits prompt_address output (pg_slot v 255) prompt **
    pg_selector_stage_frame weights_address buffers weights prompt v tokens
  ensures pg_suffix_pending weights_address prompt_address buffers weights prompt 3
{
  unfold pg_selector_post logits prompt_address output (pg_slot v 255) prompt;
  with ys. assert output |-> ys;
  unfold pg_selector_stage_frame weights_address buffers weights prompt v tokens;
  rewrite logits |-> pg_slot v 255 as PA.pts_to (pg_address buffers 255) (pg_slot v 255);
  rewrite output |-> ys as PA.pts_to (RT.data buffers.token_buffer) ys;
  rewrite pg_masked_storage buffers v 1.0R (fun i -> i <> 255) as
    pg_masked_storage buffers v 1.0R (fun i -> True /\ ~(eq2 #(Shared.index 256) i 255));
  pg_put buffers #v #1.0R (fun _ -> True) 255 #(pg_slot v 255);
  pg_replace_same v 255;
  rewrite pg_masked_storage buffers (pg_replace v 255 (pg_slot v 255)) 1.0R (fun _ -> True) as
    pg_masked_storage buffers v 1.0R (fun _ -> True);
  unfold pg_masked_storage buffers v 1.0R (fun _ -> True);
  forevery_unrefine (fun (i:Shared.index 256) -> PA.pts_to (pg_address buffers i) (pg_slot v i));
  fold pg_storage buffers v 1.0R;
  pg_suffix_selector_advance weights prompt v tokens ys;
  fold pg_suffix_core weights_address prompt_address buffers weights prompt v ys 3;
  intro_exists_erased #pg_suffix_observation
    (fun observed -> pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 3)
    (hide #pg_suffix_observation (Ghost.reveal #pg_values v,Ghost.reveal #(lseq I64.t 8) ys));
  fold pg_suffix_pending weights_address prompt_address buffers weights prompt 3;
}

inline_for_extraction noextract
ghost
fn pg_expand_suffix_pending (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (step:nat{step <= 3}) ()
  requires on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt step)
  ensures on gpu_loc (exists* (observed:pg_suffix_observation).
    pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) step)
{
  ghost_impersonate gpu_loc
    (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt step))
    (on gpu_loc (exists* (observed:pg_suffix_observation).
      pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) step))
    fn () {
      on_elim (pg_suffix_pending weights_address prompt_address buffers weights prompt step);
      unfold pg_suffix_pending weights_address prompt_address buffers weights prompt step;
      on_intro (exists* (observed:pg_suffix_observation).
        pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) step);
    };
}

inline_for_extraction noextract
fn pg_launch_suffix_step (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (step:nat{step < 3})
  (family:opt_pending_family pg_suffix_observation
    (fun observed -> pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) step)
    (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1)))
  (stream:ST.stream_t) (#start:E.epoch_t)
  preserves cpu ** ST.stream_live stream
  requires E.epoch_live stream (start + (253 + step)) **
    P.pledge0 (E.epoch_done stream (start + (253 + step)))
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt step))
  ensures E.epoch_live stream (start + (253 + (step + 1))) **
    P.pledge0 (E.epoch_done stream (start + (253 + (step + 1))))
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1)))
{
  P.rewrite_pledge (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt step))
    (on gpu_loc (exists* (observed:pg_suffix_observation).
      pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) step))
    (pg_expand_suffix_pending weights_address prompt_address buffers #weights #prompt step);
  u_launch_const_post family stream #(start + (253 + step));
  u_queue_step stream #start (253 + step)
    (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1)));
  rewrite E.epoch_live stream (start + (253 + step + 1)) as
    E.epoch_live stream (start + (253 + (step + 1)));
  rewrite P.pledge0 (E.epoch_done stream (start + (253 + step + 1)))
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1))) as
    P.pledge0 (E.epoch_done stream (start + (253 + (step + 1))))
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1)));
}

inline_for_extraction noextract
fn pg_enqueue_suffix
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t)
  preserves cpu ** ST.stream_live stream ** opt_allocated_owners stream buffers
  requires E.epoch_live stream (start + 253) **
    P.pledge0 (E.epoch_done stream (start + 253)) (pg_after_stage weights_address prompt_address buffers weights prompt 253)
  ensures E.epoch_live stream (start + 256) **
    P.pledge0 (E.epoch_done stream (start + 256))
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 3))
{
  P.rewrite_pledge (pg_after_stage weights_address prompt_address buffers weights prompt 253)
    (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 0))
    (pg_adopt_suffix weights_address prompt_address buffers #weights #prompt);
  let input = pg_data_ptr stream buffers 253;
  let hidden = pg_data_ptr stream buffers 254;
  let logits = pg_data_ptr stream buffers 255;
  unfold opt_allocated_owners stream buffers;
  let tokens = RT.data_ptr stream buffers.token_buffer;
  fold opt_allocated_owners stream buffers;
  let norm_code = pg_final_rms_family weights_address prompt_address buffers input hidden #weights #prompt;
  pg_launch_suffix_step weights_address prompt_address buffers #weights #prompt 0 norm_code stream #start;
  let projection_code = pg_final_projection_family weights_address prompt_address buffers hidden logits #weights #prompt;
  pg_launch_suffix_step weights_address prompt_address buffers #weights #prompt 1 projection_code stream #start;
  let selector_code = pg_final_selector_family weights_address prompt_address buffers logits tokens #weights #prompt;
  pg_launch_suffix_step weights_address prompt_address buffers #weights #prompt 2 selector_code stream #start;
}

inline_for_extraction noextract
ghost
fn rec pg_owners_append (stream:ST.stream_t) (left right:list (RT.tensor BF16.t))
  requires RH.all_owners stream left ** RH.all_owners stream right
  ensures RH.all_owners stream (List.append left right)
  decreases left
{
  match left {
    [] -> {
      unfold RH.all_owners #BF16.t stream [];
      rewrite RH.all_owners stream right as RH.all_owners stream (List.append left right);
    }
    head::tail -> {
      unfold RH.all_owners stream left;
      pg_owners_append stream tail right;
      fold RH.all_owners stream (head :: List.append tail right);
      rewrite RH.all_owners stream (head :: List.append tail right) as RH.all_owners stream (List.append left right);
    }
  }
}

inline_for_extraction noextract
ghost
fn rec pg_owners_split (stream:ST.stream_t) (left right:list (RT.tensor BF16.t))
  requires RH.all_owners stream (List.append left right)
  ensures RH.all_owners stream left ** RH.all_owners stream right
  decreases left
{
  match left {
    [] -> {
      rewrite RH.all_owners stream (List.append left right) as RH.all_owners stream right;
      fold RH.all_owners #BF16.t stream [];
    }
    head::tail -> {
      rewrite RH.all_owners stream (List.append left right) as RH.all_owners stream (head :: List.append tail right);
      unfold RH.all_owners stream (head :: List.append tail right);
      pg_owners_split stream tail right;
      fold RH.all_owners stream left;
    }
  }
}

inline_for_extraction noextract
ghost
fn rec pg_owners_reverse (stream:ST.stream_t) (items:list (RT.tensor BF16.t))
  requires RH.all_owners stream items
  ensures RH.all_owners stream (List.rev' items)
  decreases items
{
  match items {
    [] -> { rewrite RH.all_owners stream items as RH.all_owners stream (List.rev' items); }
    head::tail -> {
      unfold RH.all_owners stream items;
      pg_owners_reverse stream tail;
      fold RH.all_owners #BF16.t stream [];
      fold RH.all_owners stream [head];
      pg_owners_append stream (List.rev' tail) [head];
      rewrite RH.all_owners stream (List.append (List.rev' tail) [head]) as RH.all_owners stream (List.rev' items);
    }
  }
}

inline_for_extraction noextract
ghost
fn rec pg_contents_append (left right:list (RT.tensor BF16.t))
  requires RH.all_contents left ** RH.all_contents right
  ensures RH.all_contents (List.append left right)
  decreases left
{
  match left {
    [] -> {
      unfold RH.all_contents #BF16.t [];
      rewrite RH.all_contents right as RH.all_contents (List.append left right);
    }
    head::tail -> {
      unfold RH.all_contents left;
      pg_contents_append tail right;
      fold RH.all_contents (head :: List.append tail right);
      rewrite RH.all_contents (head :: List.append tail right) as RH.all_contents (List.append left right);
    }
  }
}

inline_for_extraction noextract
ghost
fn rec pg_contents_reverse (items:list (RT.tensor BF16.t))
  requires RH.all_contents items
  ensures RH.all_contents (List.rev' items)
  decreases items
{
  match items {
    [] -> { rewrite RH.all_contents items as RH.all_contents (List.rev' items); }
    head::tail -> {
      unfold RH.all_contents items;
      pg_contents_reverse tail;
      fold RH.all_contents #BF16.t [];
      fold RH.all_contents [head];
      pg_contents_append (List.rev' tail) [head];
      rewrite RH.all_contents (List.append (List.rev' tail) [head]) as RH.all_contents (List.rev' items);
    }
  }
}

inline_for_extraction noextract
ghost
fn pg_retirement_owners (stream:ST.stream_t) (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  requires opt_allocated_owners stream buffers
  ensures RH.all_owners stream (pg_retirement buffers) **
    RT.owner stream (pg_returned buffers).returned_tokens ** RT.owner stream (pg_returned buffers).returned_hidden
{
  pg_retirement_binding buffers;
  unfold opt_allocated_owners stream buffers;
  let prefix = (List.splitAt 254 buffers.bf16_buffers)._1;
  let logits = List.index buffers.bf16_buffers 255;
  let hidden = (pg_returned buffers).returned_hidden;
  rewrite RH.all_owners stream buffers.bf16_buffers as RH.all_owners stream (List.append prefix [hidden;logits]);
  pg_owners_split stream prefix [hidden;logits];
  unfold RH.all_owners stream [hidden;logits];
  unfold RH.all_owners stream [logits];
  unfold RH.all_owners #BF16.t stream [];
  pg_owners_reverse stream prefix;
  fold RH.all_owners stream (logits :: List.rev' prefix);
  rewrite RH.all_owners stream (logits :: List.rev' prefix) as RH.all_owners stream (pg_retirement buffers);
  rewrite RT.owner stream buffers.token_buffer as RT.owner stream (pg_returned buffers).returned_tokens;
  rewrite RT.owner stream hidden as RT.owner stream (pg_returned buffers).returned_hidden;
}

inline_for_extraction noextract
ghost
fn rec pg_prefix_to_runtime (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (count:nat{count <= 254})
  preserves loc gpu_loc
  requires forall+ (i:Shared.index 256{i < count}). PA.pts_to (pg_address buffers i) (pg_slot v i)
  ensures RH.all_contents (List.splitAt count buffers.bf16_buffers)._1
  decreases count
{
  if (count = 0) {
    forevery_elim_empty (fun (i:Shared.index 256{i < count}) -> PA.pts_to (pg_address buffers i) (pg_slot v i));
    fold RH.all_contents #BF16.t [];
    rewrite RH.all_contents #BF16.t [] as RH.all_contents (List.splitAt count buffers.bf16_buffers)._1;
  } else {
    let last = count - 1;
    forevery_remove' #(Shared.index 256) (fun i -> i < count)
      (fun i -> PA.pts_to (pg_address buffers i) (pg_slot v i)) last;
    forevery_refine_ext #(Shared.index 256) #(fun i -> i < count /\ ~(eq2 #(Shared.index 256) i last))
      (fun i -> i < last) (fun i -> PA.pts_to (pg_address buffers i) (pg_slot v i));
    pg_prefix_to_runtime buffers #v last;
    on_intro (PA.pts_to (pg_address buffers last) (pg_slot v last));
    opt_allocation_count ();
    opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
    opt_matching_shape_at opt_bf16_allocation_shapes buffers.bf16_buffers last;
    pg_shape_at last;
    let handle = List.index buffers.bf16_buffers last;
    rewrite each pg_address buffers last as RT.data handle;
    fold RT.contents handle;
    fold RH.all_contents #BF16.t [];
    fold RH.all_contents [handle];
    pg_contents_append (List.splitAt last buffers.bf16_buffers)._1 [handle];
    pg_list_prefix_step buffers.bf16_buffers count;
    rewrite RH.all_contents (List.append (List.splitAt last buffers.bf16_buffers)._1 [handle]) as
      RH.all_contents (List.splitAt count buffers.bf16_buffers)._1;
  }
}

inline_for_extraction noextract
ghost
fn pg_pack_source_return (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (output:pg_returned_buffers{pg_returned_metadata output})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:pg_values) (#tokens:lseq I64.t 8)
  preserves loc gpu_loc
  requires PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
    PA.pts_to (RT.data output.returned_tokens) tokens ** pure (pg_suffix_arithmetic weights prompt v tokens 3)
  ensures pg_source_return_frame weights_address prompt_address weights prompt output
{
  intro_exists
    (fun (ts:lseq I64.t 8) ->
      PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
      PA.pts_to (RT.data output.returned_tokens) ts ** pure (pg_suffix_arithmetic weights prompt v ts 3)) tokens;
  intro_exists
    (fun (snapshot:pg_values) ->
      exists* (ts:lseq I64.t 8).
        PA.pts_to (RT.data output.returned_hidden) (pg_slot snapshot 254) **
        PA.pts_to (RT.data output.returned_tokens) ts ** pure (pg_suffix_arithmetic weights prompt snapshot ts 3)) v;
  on_intro (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    (exists* (snapshot:pg_values) (ts:lseq I64.t 8).
      PA.pts_to (RT.data output.returned_hidden) (pg_slot snapshot 254) **
      PA.pts_to (RT.data output.returned_tokens) ts ** pure (pg_suffix_arithmetic weights prompt snapshot ts 3)));
  fold pg_source_return_frame weights_address prompt_address weights prompt output;
}

inline_for_extraction noextract
ghost
fn pg_retirement_snapshot_at (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) ()
  norewrite
  preserves loc gpu_loc
  requires pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 3
  ensures RH.all_contents (pg_retirement buffers) **
    pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers)
{
      unfold pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 3;
      rewrite pg_storage buffers v 1.0R as pg_masked_storage buffers v 1.0R (fun _ -> True);
      pg_take buffers #v #1.0R (fun _ -> True) 254;
      pg_take buffers #v #1.0R (fun i -> True /\ ~(eq2 #(Shared.index 256) i 254)) 255;
      unfold pg_masked_storage buffers v 1.0R
        (fun i -> (True /\ ~(eq2 #(Shared.index 256) i 254)) /\ ~(eq2 #(Shared.index 256) i 255));
      forevery_refine_ext #(Shared.index 256)
        #(fun i -> (True /\ ~(eq2 #(Shared.index 256) i 254)) /\ ~(eq2 #(Shared.index 256) i 255))
        (fun i -> i < 254) (fun i -> PA.pts_to (pg_address buffers i) (pg_slot v i));
      pg_prefix_to_runtime buffers #v 254;
      pg_contents_reverse (List.splitAt 254 buffers.bf16_buffers)._1;
      on_intro (PA.pts_to (pg_address buffers 255) (pg_slot v 255));
      pg_retirement_binding buffers;
      opt_allocation_count ();
      opt_matching_shape_at opt_bf16_allocation_shapes buffers.bf16_buffers 255;
      pg_shape_at 255;
      let logits = List.index buffers.bf16_buffers 255;
      rewrite each pg_address buffers 255 as RT.data logits;
      fold RT.contents logits;
      fold RH.all_contents (logits :: List.rev' (List.splitAt 254 buffers.bf16_buffers)._1);
      rewrite RH.all_contents (logits :: List.rev' (List.splitAt 254 buffers.bf16_buffers)._1) as RH.all_contents (pg_retirement buffers);
      pg_return_addresses buffers;
      let output = pg_returned buffers;
      rewrite each pg_address buffers 254 as RT.data output.returned_hidden;
      rewrite each RT.data buffers.token_buffer as RT.data output.returned_tokens;
      pg_pack_source_return weights_address prompt_address output #weights #prompt #v #tokens;
      rewrite pg_source_return_frame weights_address prompt_address weights prompt output as
        pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers);
}

inline_for_extraction noextract
ghost
fn pg_retirement_contents_at (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  norewrite
  preserves loc gpu_loc
  requires on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 3)
  ensures RH.all_contents (pg_retirement buffers) **
    pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers)
{
  on_elim (pg_suffix_pending weights_address prompt_address buffers weights prompt 3);
  unfold pg_suffix_pending weights_address prompt_address buffers weights prompt 3;
  with observed. assert pg_suffix_core weights_address prompt_address buffers weights prompt
    (pg_suffix_tensors observed) (pg_suffix_tokens observed) 3;
  pg_retirement_snapshot_at weights_address prompt_address buffers #weights #prompt
    #(hide #pg_values (pg_suffix_tensors observed)) #(hide #(lseq I64.t 8) (pg_suffix_tokens observed)) ();
}

inline_for_extraction noextract
fn pg_source_return (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t)
  preserves cpu ** ST.stream_live stream ** RT.runtime stream ** E.epoch_live stream (start + 256)
  requires opt_allocated_owners stream buffers **
    P.pledge0 (E.epoch_done stream (start + 256))
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 3))
  returns output:pg_returned_result
  ensures RT.owner stream output.returned_tokens ** RT.owner stream output.returned_hidden **
    P.pledge0 (E.epoch_done stream (start + 256))
      (pg_source_return_frame weights_address prompt_address weights prompt output)
{
  pg_retirement_binding buffers;
  let output = pg_returned buffers;
  pg_retirement_owners stream buffers;
  rewrite RT.owner stream (pg_returned buffers).returned_tokens as RT.owner stream output.returned_tokens;
  rewrite RT.owner stream (pg_returned buffers).returned_hidden as RT.owner stream output.returned_hidden;
  P.rewrite_pledge (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 3))
    (RH.all_contents (pg_retirement buffers) ** pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers))
    (pg_retirement_contents weights_address prompt_address buffers #weights #prompt);
  RH.release_ordered stream #(start + 256) (pg_retirement buffers)
    (pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers)) ();
  rewrite P.pledge0 (E.epoch_done stream (start + 256))
      (pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers)) as
    P.pledge0 (E.epoch_done stream (start + 256)) (pg_source_return_frame weights_address prompt_address weights prompt output);
  output
}
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
#pop-options

inline_for_extraction noextract
fn pg_source_program
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t) (ambient:slprop)
  preserves cpu ** ST.stream_live stream ** RT.runtime stream ** ambient
  requires E.epoch_live stream start ** u_input_storage weights_address prompt_address weights prompt
  returns outcome:pg_source_outcome
  ensures (
    match outcome with
    | PGSourceFailed _ ->
      E.epoch_live stream start **
      P.pledge0 (E.epoch_done stream start) (u_input_storage weights_address prompt_address weights prompt)
    | PGSourceReturned output ->
      E.epoch_live stream (start + 256) ** RT.owner stream output.returned_tokens ** RT.owner stream output.returned_hidden **
      P.pledge0 (E.epoch_done stream (start + 256))
        (pg_source_return_frame weights_address prompt_address weights prompt output))
{
  unfold u_input_storage weights_address prompt_address weights prompt;
  u_empty_mask_localize weights_address #weights ambient;
  fold u_input_storage weights_address prompt_address weights prompt;
  let block = first_block_refines weights_address prompt_address #weights #prompt stream #start ambient;
  match block {
    UBlockFailed error -> {
      drop_ (PA.pts_to_mask weights_address #1.0R (u_metadata_mask weights) (fun _ -> False));
      PGSourceFailed error
    }
    UBlockEnqueued buffers -> {
      P.rewrite_pledge (source_after_first_block weights_address prompt_address buffers weights prompt)
        (pg_after_stage weights_address prompt_address buffers weights prompt 11)
        (pg_from_first_block weights_address prompt_address buffers #weights #prompt);
      pg_enqueue_remaining_layers weights_address prompt_address buffers #weights #prompt stream #start;
      drop_ (PA.pts_to_mask weights_address #1.0R (u_metadata_mask weights) (fun _ -> False));
      pg_enqueue_suffix weights_address prompt_address buffers #weights #prompt stream #start;
      let output = pg_source_return weights_address prompt_address buffers #weights #prompt stream #start;
      PGSourceReturned output
    }
  }
}

#pop-options


#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"
open Kuiper
open FStar.Real
module N = Kuiper.Approximates.Base

inline_for_extraction noextract
ghost
fn mg_pack_return (#frame_t:Type0) (#frame:erased frame_t)
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (output:pg_returned_result)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:pg_values) (#tokens:lseq I64.t 8)
  norewrite
  preserves loc gpu_loc
  requires PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
    PA.pts_to (RT.data output.returned_tokens) tokens ** pure (pg_suffix_arithmetic weights prompt v tokens 3)
  ensures program_return_frame weights_address prompt_address weights prompt output frame
{
  mg_program_result_from_snapshot weights prompt v tokens (Ghost.reveal #frame_t frame);
  intro_exists
    (fun (ts:lseq I64.t 8) ->
      PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
      PA.pts_to (RT.data output.returned_tokens) ts **
      pure (pg_suffix_arithmetic weights prompt v ts 3 /\ mg_program_result weights prompt v ts frame)) tokens;
  intro_exists
    (fun (snapshot:pg_values) ->
      exists* (ts:lseq I64.t 8).
        PA.pts_to (RT.data output.returned_hidden) (pg_slot snapshot 254) **
        PA.pts_to (RT.data output.returned_tokens) ts **
        pure (pg_suffix_arithmetic weights prompt snapshot ts 3 /\ mg_program_result weights prompt snapshot ts frame)) v;
  on_intro (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    (exists* (snapshot:pg_values) (ts:lseq I64.t 8).
      PA.pts_to (RT.data output.returned_hidden) (pg_slot snapshot 254) **
      PA.pts_to (RT.data output.returned_tokens) ts **
      pure (pg_suffix_arithmetic weights prompt snapshot ts 3 /\ mg_program_result weights prompt snapshot ts frame)));
  fold program_return_frame weights_address prompt_address weights prompt output frame;
}

inline_for_extraction noextract
ghost
fn mg_strengthen_return_at (#frame_t:Type0) (#frame:erased frame_t)
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (output:pg_returned_result)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  norewrite
  preserves loc gpu_loc
  requires pg_source_return_frame weights_address prompt_address weights prompt output
  ensures program_return_frame weights_address prompt_address weights prompt output frame
{
  unfold pg_source_return_frame weights_address prompt_address weights prompt output;
  on_elim (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    (exists* (v:pg_values) (tokens:lseq I64.t 8).
      PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
      PA.pts_to (RT.data output.returned_tokens) tokens ** pure (pg_suffix_arithmetic weights prompt v tokens 3)));
  with v tokens. assert PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
    PA.pts_to (RT.data output.returned_tokens) tokens ** pure (pg_suffix_arithmetic weights prompt v tokens 3);
  mg_pack_return #frame_t #frame weights_address prompt_address output #weights #prompt #v #tokens;
}

inline_for_extraction noextract
fn program_refines (#frame_t:Type0) (#logical_frame:erased frame_t)
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t) (ambient:slprop)
  preserves cpu ** ST.stream_live stream ** RT.runtime stream ** ambient
  requires E.epoch_live stream start ** u_input_storage weights_address prompt_address weights prompt
  returns outcome:pg_source_outcome
  ensures (
    match outcome with
    | PGSourceFailed _ ->
      E.epoch_live stream start **
      P.pledge0 (E.epoch_done stream start) (u_input_storage weights_address prompt_address weights prompt)
    | PGSourceReturned output ->
      E.epoch_live stream (start + 256) ** RT.owner stream output.returned_tokens ** RT.owner stream output.returned_hidden **
      P.pledge0 (E.epoch_done stream (start + 256))
        (program_return_frame weights_address prompt_address weights prompt output logical_frame))
{
  let outcome = pg_source_program weights_address prompt_address #weights #prompt stream #start ambient;
  match outcome {
    PGSourceFailed error -> { PGSourceFailed error }
    PGSourceReturned output -> {
      P.rewrite_pledge (pg_source_return_frame weights_address prompt_address weights prompt output)
        (program_return_frame weights_address prompt_address weights prompt output logical_frame)
        (mg_strengthen_return #frame_t #logical_frame weights_address prompt_address output #weights #prompt);
      PGSourceReturned output
    }
  }
}

#pop-options
