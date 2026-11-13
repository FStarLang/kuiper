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
instance opt_send_global_array (#a:Type0)
  (buffer:PA.array a{KA.is_global_array buffer}) (#fraction:perm) (values:S.seq a)
  : is_send_across gpu_of (PA.pts_to buffer #fraction values) =
  KA.is_send_pts_to buffer #fraction values

inline_for_extraction noextract
let source_attention_literal () : F32.t =
  F.of_literal "0.12751743197441101074"

inline_for_extraction noextract
let source_negative_infinity () : F32.t =
  F.sub F.zero F.infinity

inline_for_extraction noextract
let source_rescale_contract (maximum scaled result:F32.t) : prop =
  (F.eq maximum (source_negative_infinity ()) ==> result == F.one) /\
  (not (F.eq maximum (source_negative_infinity ())) /\
    F.Finite? (F.kind scaled) ==>
      result == F.zero /\ v_approximates result 0.0R)

inline_for_extraction noextract
let source_attention_literal_bounds ()
  : Lemma (
      F32.positive_normal (source_attention_literal ()) /\
      F.lte (source_attention_literal ()) F.one) =
  assert_norm (
    Kuiper.Float32.Literals.positive_normal_decimal "0.12751743197441101074");
  assert_norm (
    Kuiper.Float32.Literals.at_most_one_decimal "0.12751743197441101074");
  F32.positive_normal_of_literal "0.12751743197441101074";
  F32.positive_literal_at_most_one "0.12751743197441101074"

inline_for_extraction noextract
fn source_attention_rescale (maximum scaled:F32.t)
  preserves gpu
  returns result:F32.t
  ensures pure (source_rescale_contract maximum scaled result)

inline_for_extraction noextract
let source_query_address (head:Shared.index 8) (step:Shared.index 8)
  (row:Shared.index 14) (column:Shared.index 16)
  : n:nat{n < 32768} =
  (row / 2) * 4096 + head * 512 + (row % 2) * 128 + step * 16 + column

inline_for_extraction noextract
let source_value_address (head:Shared.index 8) (block:Shared.index 16)
  (key:Shared.index 7) (column:Shared.index 8)
  : n:nat{n < 32768} =
  key * 4096 + head * 512 + 384 + block * 8 + column

inline_for_extraction noextract
let source_query_operand (rotated:lseq BF16.t 32768)
  (head:Shared.index 8) (step:Shared.index 8)
  : GTot (Chest.chest2 BF16.t 16 16) =
  Chest.mk2 (fun row column ->
    if row < 14 then S.index rotated (source_query_address head step row column)
    else F.zero)

inline_for_extraction noextract
let source_key_operand (rotated:lseq BF16.t 32768)
  (head:Shared.index 8) (step:Shared.index 8)
  : GTot (Chest.chest2 BF16.t 16 8) =
  Chest.mk2 (fun column key ->
    if key < 7 then
      S.index rotated (key * 4096 + head * 512 + 256 + step * 16 + column)
    else F.zero)

inline_for_extraction noextract
let source_value_operand (rotated:lseq BF16.t 32768)
  (head:Shared.index 8) (block:Shared.index 16) (step:Shared.index 4)
  : GTot (Chest.chest2 BF16.t 16 8) =
  Chest.mk2 (fun row column ->
    let key = step * 16 + row in
    if key < 7 then S.index rotated (source_value_address head block key column)
    else F.zero)

inline_for_extraction noextract
let rec source_score_prefix (rotated:lseq BF16.t 32768)
  (head:Shared.index 8) (steps:nat{steps <= 8})
  : GTot (Chest.chest2 F32.t 16 8) (decreases steps) =
  if steps = 0 then Chest.const (16 @| 8 @| INil) F.zero
  else MMA.emma (source_score_prefix rotated head (steps - 1))
    (source_query_operand rotated head (steps - 1))
    (source_key_operand rotated head (steps - 1))

inline_for_extraction noextract
let rec source_value_prefix (rotated:lseq BF16.t 32768)
  (probability:lseq BF16.t 1024) (head:Shared.index 8)
  (block:Shared.index 16) (steps:nat{steps <= 4})
  : GTot (Chest.chest2 F32.t 16 8) (decreases steps) =
  if steps = 0 then Chest.const (16 @| 8 @| INil) F.zero
  else MMA.emma (source_value_prefix rotated probability head block (steps - 1))
    (Gather.value_a probability (steps - 1))
    (source_value_operand rotated head block (steps - 1))

inline_for_extraction noextract
let source_gather_score_operands (values:lseq BF16.t 32768)
  (head:Shared.index 8) (step:Shared.index 8)
  : Lemma (
      Gather.score_a values head step == source_query_operand values head step /\
      Gather.score_b values head step == source_key_operand values head step) =
  Chest.ext (Gather.score_a values head step) (source_query_operand values head step);
  Chest.ext (Gather.score_b values head step) (source_key_operand values head step)

inline_for_extraction noextract
fn source_attention_score_steps
  (input:larray BF16.t 32768) (head:szlt 8) (fr:MMA.fragment)
  (#values:erased (lseq BF16.t 32768))
  (#fraction:perm) (#before:Chest.chest2 F32.t 16 8)
  preserves gpu ** input |-> Frac fraction values
  requires fr |-> before
  ensures fr |-> source_score_prefix values head 8

inline_for_extraction noextract
let source_gather_value_operand (values:lseq BF16.t 32768)
  (head:Shared.index 8) (block:Shared.index 16) (step:Shared.index 4)
  : Lemma (Gather.value_b values head block step ==
      source_value_operand values head block step) =
  Chest.ext (Gather.value_b values head block step)
    (source_value_operand values head block step)

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

inline_for_extraction noextract
let source_tile_values (values:Chest.chest2 F32.t 16 8)
  : GTot (lseq F32.t 128) =
  S.init_ghost 128 (fun i -> Chest.acc2 values (i / 8) (i % 8))

inline_for_extraction noextract
let source_tile_sequence_binding (values:Chest.chest2 F32.t 16 8)
  : Lemma (Tensor.to_seq (source_rm 16 8) values == source_tile_values values) =
  assert (S.equal (Tensor.to_seq (source_rm 16 8) values) (source_tile_values values));
  S.lemma_eq_elim (Tensor.to_seq (source_rm 16 8) values) (source_tile_values values)

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

inline_for_extraction noextract
type opt_allocation_shape = shape:list pos{RT.numel shape <= 2147483647}

inline_for_extraction noextract
let opt_layer_allocation_shapes : list opt_allocation_shape =
  [[8; 1024]; [8; 4096]; [8; 4096]; [8; 2048]; [8; 1024];
   [8; 1024]; [8; 6144]; [8; 3072]; [8; 1024]]

inline_for_extraction noextract
let rec opt_layer_allocations (count:nat{count <= 28})
  : Tot (list opt_allocation_shape) (decreases count) =
  if count = 0 then []
  else List.append opt_layer_allocation_shapes (opt_layer_allocations (count - 1))

inline_for_extraction noextract
let opt_bf16_allocation_shapes : list opt_allocation_shape =
  List.append [[8; 1024]; [2; 4096; 128]]
    (List.append (opt_layer_allocations 28) [[7; 1024]; [8; 153600]])

inline_for_extraction noextract
let rec opt_layer_allocation_count (count:nat{count <= 28})
  : Lemma (List.length (opt_layer_allocations count) == 9 * count)
      (decreases count) =
  if count > 0 then (
    opt_layer_allocation_count (count - 1);
    ListProps.append_length opt_layer_allocation_shapes (opt_layer_allocations (count - 1));
    assert_norm (List.length opt_layer_allocation_shapes == 9))

inline_for_extraction noextract
let opt_allocation_count ()
  : Lemma (List.length opt_bf16_allocation_shapes == 256 /\
      List.length opt_bf16_allocation_shapes + 1 == 257) =
  opt_layer_allocation_count 28;
  ListProps.append_length (opt_layer_allocations 28) [[7; 1024]; [8; 153600]];
  ListProps.append_length [[8; 1024]; [2; 4096; 128]]
    (List.append (opt_layer_allocations 28) [[7; 1024]; [8; 153600]])

inline_for_extraction noextract
let rec opt_shapes_match (shapes:list opt_allocation_shape)
  (buffers:list (RT.tensor BF16.t)) : GTot prop (decreases shapes) =
  match shapes, buffers with
  | [], [] -> True
  | shape::rest, value::tail ->
    RT.metadata value shape /\ opt_shapes_match rest tail
  | _ -> False

inline_for_extraction noextract
noeq type opt_allocated_buffers = {
  bf16_buffers: list (RT.tensor BF16.t);
  token_buffer: RT.tensor I64.t
}

inline_for_extraction noextract
let opt_allocation_metadata (buffers:opt_allocated_buffers) : prop =
  opt_shapes_match opt_bf16_allocation_shapes buffers.bf16_buffers /\
  RT.metadata buffers.token_buffer [1; 8]

inline_for_extraction noextract
let opt_allocated_owners (stream:Stream.stream_t) (buffers:opt_allocated_buffers)
  : PC.slprop =
  RuntimeHelpers.all_owners stream buffers.bf16_buffers ** RT.owner stream buffers.token_buffer

inline_for_extraction noextract
let opt_allocated_storage (buffers:opt_allocated_buffers) (frame:PC.slprop)
  : PC.slprop =
  RT.contents buffers.token_buffer **
  RuntimeHelpers.all_contents buffers.bf16_buffers ** frame

inline_for_extraction noextract
noeq type opt_allocation_outcome =
  | OptAllocationFailed : error:RT.allocation_error -> opt_allocation_outcome
  | OptAllocated : buffers:opt_allocated_buffers -> opt_allocation_outcome

inline_for_extraction noextract
ghost
fn opt_empty_allocation_storage (token:RT.tensor I64.t) (frame:PC.slprop) ()
  requires RT.contents token ** frame
  ensures opt_allocated_storage ({ bf16_buffers = []; token_buffer = token }) frame

inline_for_extraction noextract
ghost
fn opt_cons_allocation_storage (value:RT.tensor BF16.t) (buffers:opt_allocated_buffers)
  (frame:PC.slprop) ()
  requires opt_allocated_storage buffers (RT.contents value ** frame)
  ensures opt_allocated_storage
    ({ bf16_buffers = value::buffers.bf16_buffers; token_buffer = buffers.token_buffer }) frame

inline_for_extraction noextract
fn opt_allocate_bf16_shapes
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

inline_for_extraction noextract
let opt_weight_count : pos = 596049920

inline_for_extraction noextract
let opt_prompt_valid (prompt:lseq I64.t 7) : prop =
  forall (i:Shared.index 7). 0 <= I64.v (S.index prompt i) /\ I64.v (S.index prompt i) < 151936

inline_for_extraction noextract
type opt_prompt_values = prompt:lseq I64.t 7{opt_prompt_valid prompt}

inline_for_extraction noextract
noeq type opt_block_addresses = {
  p_embedding: larray BF16.t 8192;
  p_rope: larray BF16.t 1048576;
  p_normalized0: larray BF16.t 8192;
  p_qkv: larray BF16.t 32768;
  p_rotated: larray BF16.t 32768;
  p_attention: larray BF16.t 16384;
  p_residual: larray BF16.t 8192;
  p_postnorm: larray BF16.t 8192;
  p_gateup: larray BF16.t 49152;
  p_activated: larray BF16.t 24576;
  p_hidden: larray BF16.t 8192;
  p_normalized1: larray BF16.t 8192
}

inline_for_extraction noextract
let opt_block_global (a:opt_block_addresses) : prop =
  KA.is_global_array a.p_embedding /\ KA.is_global_array a.p_rope /\
  KA.is_global_array a.p_normalized0 /\ KA.is_global_array a.p_qkv /\
  KA.is_global_array a.p_rotated /\ KA.is_global_array a.p_attention /\
  KA.is_global_array a.p_residual /\ KA.is_global_array a.p_postnorm /\
  KA.is_global_array a.p_gateup /\ KA.is_global_array a.p_activated /\
  KA.is_global_array a.p_hidden /\ KA.is_global_array a.p_normalized1

inline_for_extraction noextract
noeq type opt_block_values = {
  v_embedding: lseq BF16.t 8192;
  v_rope: lseq BF16.t 1048576;
  v_normalized0: lseq BF16.t 8192;
  v_qkv: lseq BF16.t 32768;
  v_rotated: lseq BF16.t 32768;
  v_attention: lseq BF16.t 16384;
  v_residual: lseq BF16.t 8192;
  v_postnorm: lseq BF16.t 8192;
  v_gateup: lseq BF16.t 49152;
  v_activated: lseq BF16.t 24576;
  v_hidden: lseq BF16.t 8192;
  v_normalized1: lseq BF16.t 8192
}

inline_for_extraction noextract
let opt_block_storage (a:opt_block_addresses) (v:opt_block_values) : slprop =
  a.p_embedding |-> v.v_embedding ** a.p_rope |-> v.v_rope **
  a.p_normalized0 |-> v.v_normalized0 ** a.p_qkv |-> v.v_qkv **
  a.p_rotated |-> v.v_rotated ** a.p_attention |-> v.v_attention **
  a.p_residual |-> v.v_residual ** a.p_postnorm |-> v.v_postnorm **
  a.p_gateup |-> v.v_gateup ** a.p_activated |-> v.v_activated **
  a.p_hidden |-> v.v_hidden ** a.p_normalized1 |-> v.v_normalized1
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let opt_prefix_shape (index:nat{index < 12}) : opt_allocation_shape =
  if index = 1 then [2; 4096; 128]
  else if index = 3 || index = 4 then [8; 4096]
  else if index = 5 then [8; 2048]
  else if index = 8 then [8; 6144]
  else if index = 9 then [8; 3072]
  else [8; 1024]
#pop-options

inline_for_extraction noextract
let rec opt_matching_count (shapes:list opt_allocation_shape)
  (buffers:list (RT.tensor BF16.t))
  : Lemma
      (requires opt_shapes_match shapes buffers)
      (ensures List.length shapes == List.length buffers)
      (decreases shapes) =
  match shapes, buffers with
  | _::rest, _::tail -> opt_matching_count rest tail
  | _ -> ()

inline_for_extraction noextract
let rec opt_matching_shape_at (shapes:list opt_allocation_shape)
  (buffers:list (RT.tensor BF16.t))
  (index:nat{index < List.length shapes /\ index < List.length buffers})
  : Lemma
      (requires opt_shapes_match shapes buffers)
      (ensures RT.metadata (List.index buffers index) (List.index shapes index))
      (decreases index) =
  if index > 0 then
    opt_matching_shape_at (List.tail shapes) (List.tail buffers) (index - 1)
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let opt_prefix_shape_at (index:nat{index < 12})
  : Lemma (List.index opt_bf16_allocation_shapes index == opt_prefix_shape index) =
  opt_allocation_count ();
  if index = 0 then ()
  else if index = 1 then ()
  else if index = 2 then ()
  else if index = 3 then ()
  else if index = 4 then ()
  else if index = 5 then ()
  else if index = 6 then ()
  else if index = 7 then ()
  else if index = 8 then ()
  else if index = 9 then ()
  else if index = 10 then ()
  else assert (index == 11)
#pop-options

inline_for_extraction noextract
let opt_prefix_metadata
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (index:nat{index < 12 /\ index < List.length buffers.bf16_buffers})
  : Lemma (RT.metadata (List.index buffers.bf16_buffers index) (opt_prefix_shape index)) =
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  opt_matching_shape_at opt_bf16_allocation_shapes buffers.bf16_buffers index;
  opt_prefix_shape_at index

inline_for_extraction noextract
let u_remaining_tensors (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  : Tot (list (RT.tensor BF16.t)) =
  (List.splitAt 12 buffers.bf16_buffers)._2

inline_for_extraction noextract
let u_remaining_storage (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) : slprop =
  RT.contents buffers.token_buffer ** RH.all_contents (u_remaining_tensors buffers)
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_prefix_list_partition (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  : Lemma (
      List.length buffers.bf16_buffers == 256 /\
      List.length (u_remaining_tensors buffers) == 244 /\
      buffers.bf16_buffers ==
        List.index buffers.bf16_buffers 0 :: List.index buffers.bf16_buffers 1 ::
        List.index buffers.bf16_buffers 2 :: List.index buffers.bf16_buffers 3 ::
        List.index buffers.bf16_buffers 4 :: List.index buffers.bf16_buffers 5 ::
        List.index buffers.bf16_buffers 6 :: List.index buffers.bf16_buffers 7 ::
        List.index buffers.bf16_buffers 8 :: List.index buffers.bf16_buffers 9 ::
        List.index buffers.bf16_buffers 10 :: List.index buffers.bf16_buffers 11 ::
        u_remaining_tensors buffers) =
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  List.lemma_splitAt_snd_length 12 buffers.bf16_buffers
#pop-options
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_allocated_addresses (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  : GTot (a:opt_block_addresses{opt_block_global a}) =
  u_prefix_list_partition buffers;
  opt_prefix_metadata buffers 0; opt_prefix_metadata buffers 1;
  opt_prefix_metadata buffers 2; opt_prefix_metadata buffers 3;
  opt_prefix_metadata buffers 4; opt_prefix_metadata buffers 5;
  opt_prefix_metadata buffers 6; opt_prefix_metadata buffers 7;
  opt_prefix_metadata buffers 8; opt_prefix_metadata buffers 9;
  opt_prefix_metadata buffers 10; opt_prefix_metadata buffers 11;
  {
    p_embedding = RT.data (List.index buffers.bf16_buffers 0);
    p_rope = RT.data (List.index buffers.bf16_buffers 1);
    p_normalized0 = RT.data (List.index buffers.bf16_buffers 2);
    p_qkv = RT.data (List.index buffers.bf16_buffers 3);
    p_rotated = RT.data (List.index buffers.bf16_buffers 4);
    p_attention = RT.data (List.index buffers.bf16_buffers 5);
    p_residual = RT.data (List.index buffers.bf16_buffers 6);
    p_postnorm = RT.data (List.index buffers.bf16_buffers 7);
    p_gateup = RT.data (List.index buffers.bf16_buffers 8);
    p_activated = RT.data (List.index buffers.bf16_buffers 9);
    p_hidden = RT.data (List.index buffers.bf16_buffers 10);
    p_normalized1 = RT.data (List.index buffers.bf16_buffers 11)
  }
#pop-options

inline_for_extraction noextract
let u_live_matrix (#cols:pos) (x:lseq BF16.t (8 * cols)) : GTot (Shared.matrix BF16.t 7 cols) =
  fun row col -> S.index x (row * cols + col)

inline_for_extraction noextract
let u_actual_rotary (rt:lseq BF16.t 1048576) : GTot (Shared.rotary_tables BF16.t) = {
  cosine = (fun row lane -> S.index rt (row * 128 + lane));
  sine = (fun row lane -> S.index rt (524288 + row * 128 + lane))
}

inline_for_extraction noextract
let u_native_layer (v:opt_block_values) : GTot (Shared.layer_values BF16.t) = {
  hidden_after = u_live_matrix #1024 v.v_hidden;
  keys = (fun row head lane -> S.index v.v_rotated (row * 4096 + head * 512 + 256 + lane));
  values = (fun row head lane -> S.index v.v_rotated (row * 4096 + head * 512 + 384 + lane))
}

inline_for_extraction noextract
let u_next_gamma_values (weights:lseq BF16.t opt_weight_count) : GTot (Shared.vector BF16.t 1024) =
  fun col -> S.index weights (171314432 + col)

inline_for_extraction noextract
let u_block_cache_after (before:Shared.cache_state BF16.t) (v:opt_block_values)
  : GTot (Shared.cache_state BF16.t) = Shared.update_cache before 0 (u_native_layer v)

inline_for_extraction noextract
let u_first_block_cache (v:opt_block_values) : GTot (Shared.cache_state BF16.t) =
  u_block_cache_after (fun _ -> Shared.Empty) v

inline_for_extraction noextract
let source_block_score_domain (rotated:lseq BF16.t 32768) : prop =
  forall (head:Shared.index 8) (row:Shared.index 14) (key:Shared.index 7{key <= row / 2}).
    F.Finite? (F.kind (Chest.acc2 (source_score_prefix rotated head 8) row key))

inline_for_extraction noextract
let source_block_parameters (weights:lseq BF16.t opt_weight_count)
  : GTot (Shared.parameters BF16.t) = {
  input_layernorm = (fun i -> S.index weights (155583488 + i));
  post_attention_layernorm = (fun i -> S.index weights (155584512 + i));
  q_proj = (fun row col -> S.index weights (155585536 + row * 1024 + col));
  k_proj = (fun row col -> S.index weights (155585536 + (2048 + row) * 1024 + col));
  v_proj = (fun row col -> S.index weights (155585536 + (3072 + row) * 1024 + col));
  o_proj = (fun row col -> S.index weights (159779840 + row * 2048 + col));
  q_norm = (fun i -> S.index weights (161876992 + i));
  k_norm = (fun i -> S.index weights (161877120 + i));
  gate_proj = (fun row col -> S.index weights (161877248 + row * 1024 + col));
  up_proj = (fun row col -> S.index weights (161877248 + (3072 + row) * 1024 + col));
  down_proj = (fun row col -> S.index weights (168168704 + row * 3072 + col))
}

inline_for_extraction noextract
let source_block_rotary_cell (component:Shared.index 2) (row:Shared.index 4096) (col:Shared.index 128)
  : BF16.t =
  let frequency = F.div (F.one #F32.t)
    (F.pow (F.of_int #F32.t 1000000L)
      (F.div (F.of_int #F32.t (I64.int_to_t (col % 64))) (F.of_int #F32.t 64L))) in
  let angle = F.mul frequency (F.of_int #F32.t (I64.int_to_t row)) in
  Kuiper.Float.Casts.fcast #F32.t #BF16.t (if component = 0 then F.cos angle else F.sin angle)

inline_for_extraction noextract
let source_first_block_observation (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) : GTot prop =
  (forall (row:Shared.index 8) (col:Shared.index 1024).
    S.index v.v_embedding (row * 1024 + col) ==
      S.index weights ((if row < 7 then I64.v (S.index prompt row) else 0) * 1024 + col)) /\
  (forall (component:Shared.index 2) (row:Shared.index 4096) (col:Shared.index 128).
    S.index v.v_rope ((component * 4096 + row) * 128 + col) == source_block_rotary_cell component row col) /\
  (forall (row:Shared.index 8) (head:Shared.index 8) (lane:Shared.index 128).
    S.index v.v_rotated (row * 4096 + head * 512 + 384 + lane) ==
      S.index v.v_qkv (row * 4096 + head * 512 + 384 + lane)) /\
  (forall (col:Shared.index 4096). S.index v.v_rotated (28672 + col) == S.index v.v_qkv (28672 + col)) /\
  (forall (col:Shared.index 2048). S.index v.v_attention (14336 + col) == (F.zero #BF16.t)) /\
  (source_block_score_domain v.v_rotated ==>
    Shared.decoder_relation (source_block_parameters weights) (u_live_matrix #1024 v.v_embedding)
      (u_actual_rotary v.v_rope) (u_native_layer v)) /\
  Shared.final_norm_relation (u_next_gamma_values weights) (u_live_matrix #1024 v.v_hidden)
    (u_live_matrix #1024 v.v_normalized1) /\
  Shared.cache_length (u_first_block_cache v 0) == 7 /\
  (forall (other:Shared.index 28{other > 0}). u_first_block_cache v other == Shared.Empty)

inline_for_extraction noextract
let u_input_storage (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) : slprop =
  on gpu_loc (PA.pts_to weights_address weights) ** on gpu_loc (PA.pts_to prompt_address prompt)

inline_for_extraction noextract
let source_after_first_block (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) : slprop =
  on gpu_loc (
    PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    (exists* (v:opt_block_values).
      opt_block_storage (u_allocated_addresses buffers) v **
      pure (source_first_block_observation weights prompt v))) **
  u_remaining_storage buffers

inline_for_extraction noextract
noeq type u_first_block_outcome =
  | UBlockFailed : error:RT.allocation_error -> u_first_block_outcome
  | UBlockEnqueued : buffers:opt_allocated_buffers{opt_allocation_metadata buffers} -> u_first_block_outcome

inline_for_extraction noextract
let emb_embedding_count : pos = 151936 * 1024

inline_for_extraction noextract
let dec_weight_elements : pos = 15730944

inline_for_extraction noextract
let opt_layer0_start : nat = 155583488

type dec_spec_index (n:nat) = i:nat{i < n}

type dec_spec_vector (n:nat) = dec_spec_index n -> GTot real

type dec_spec_matrix (rows cols:nat) = dec_spec_index rows -> GTot (dec_spec_vector cols)

inline_for_extraction noextract
let dec_matrix_prefix (#rows #cols:pos) (xs:lseq bf16 (rows * cols))
  (model:dec_spec_matrix rows cols) (done:nat) : prop =
  forall (row:dec_spec_index rows{row < done}) (col:dec_spec_index cols).
    Seq.index xs (row * cols + col) %~ model row col

inline_for_extraction noextract
let dec_matrix_rep (#rows #cols:pos) (xs:lseq bf16 (rows * cols))
  (model:dec_spec_matrix rows cols) : prop =
  dec_matrix_prefix xs model rows

inline_for_extraction noextract
let emb_padded_token (#tokens:nat) (prompt:lseq EmbI64.t tokens) (row:nat) : int =
  if row < tokens then EmbI64.v (EmbS.index prompt row) else 0

inline_for_extraction noextract
let emb_embedding_post
  (vocab width tokens rows:nat)
  (weights:lseq bf16 (vocab * width))
  (prompt:lseq EmbI64.t tokens)
  (embedding:lseq bf16 (rows * width)) : prop =
  forall (row:nat{row < rows}) (column:nat{column < width}).
    let token = emb_padded_token prompt row in
    if 0 <= token && token < vocab then
      EmbS.index embedding (row * width + column) ==
        EmbS.index weights (token * width + column)
    else False

inline_for_extraction noextract
let emb_coordinate_f32 (i:nat{i < 0x100000000}) : f32 =
  of_int #f32 (EmbI64.int_to_t i)

inline_for_extraction noextract
let emb_inverse_frequency
  (half_width:pos{half_width < 0x100000000})
  (frequency:nat{frequency < half_width}) : f32 =
  div (one #f32)
    (pow (of_int #f32 1000000L)
      (div (emb_coordinate_f32 frequency) (emb_coordinate_f32 half_width)))

inline_for_extraction noextract
let emb_rotary_phase
  (half_width:pos{half_width < 0x100000000})
  (position:nat{position < 0x100000000})
  (frequency:nat{frequency < half_width}) : f32 =
  mul (emb_inverse_frequency half_width frequency) (emb_coordinate_f32 position)

inline_for_extraction noextract
let emb_rotary_value
  (half_width:pos{half_width < 0x100000000})
  (position:nat{position < 0x100000000})
  (component:nat{component < 2})
  (column:nat{column < 2 * half_width}) : bf16 =
  let phase = emb_rotary_phase half_width position (column % half_width) in
  EmbFC.fcast #f32 #bf16 (if component = 0 then cos phase else sin phase)

inline_for_extraction noextract
let emb_rope_post
  (positions:nat{positions < 0x100000000})
  (half_width:pos{half_width < 0x100000000})
  (table:lseq bf16 (4 * positions * half_width)) : prop =
  forall
    (component:nat{component < 2})
    (position:nat{position < positions})
    (column:nat{column < 2 * half_width}).
    EmbS.index table ((component * positions + position) * (2 * half_width) + column)
      == emb_rotary_value half_width position component column

inline_for_extraction noextract
type dec_bf16_values (n:nat) = lseq BF16.t n

inline_for_extraction noextract
let dec_dense_width (width:nat) : prop =
  0 < width /\ width <= 3072 /\ width % 16 = 0

inline_for_extraction noextract
let dec_indexed_matrix (#a:Type0) (rows cols:pos)
  (f:dec_spec_index rows -> dec_spec_index cols -> GTot a) : GTot (lseq a (rows * cols)) =
  Seq.init_ghost (rows * cols) (fun i -> f (i / cols) (i % cols))

inline_for_extraction noextract
let dec_projection_layout (cols layout:nat) : prop =
  (layout = 0 /\ (cols = 1024)) \/ (layout = 1 /\ cols = 4096) \/
  (layout = 2 /\ cols = 6144)

inline_for_extraction noextract
let rec dec_dense_mma_fold (width:nat{dec_dense_width width})
  (a:lseq bf16 (16 * width)) (b:lseq bf16 (8 * width))
  (steps:nat{steps <= width / 16})
  : GTot (Chest.chest2 F32.t 16 8) (decreases steps) =
  if steps = 0 then Chest.const (16 @| 8 @| INil) (ObF.zero #F32.t)
  else MMA.emma (dec_dense_mma_fold width a b (steps - 1))
    (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 (steps - 1))
    (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 (steps - 1) 0)

inline_for_extraction noextract
let dec_projection_row (cols:nat) (layout:nat{dec_projection_layout cols layout})
  (col:dec_spec_index cols) : dec_spec_index cols =
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
let dec_projection_block_contiguous
  (cols:nat) (layout:nat{dec_projection_layout cols layout})
  (block:nat{block < cols /\ block % 16 = 0}) (row:dec_spec_index 16)
  : Lemma
      (dec_projection_row cols layout block + 16 <= cols /\
        dec_projection_row cols layout (block + row) == dec_projection_row cols layout block + row) =
  ()

inline_for_extraction noextract
let dec_projection_scores
  (width:nat{dec_dense_width width}) (cols:nat)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (x:lseq bf16 (8 * width)) (weights:dec_bf16_values dec_weight_elements)
  (block:nat{block < cols /\ block % 16 = 0})
  : GTot (Chest.chest2 F32.t 16 8) =
  dec_projection_block_contiguous cols layout block 0;
  let start = offset + dec_projection_row cols layout block * width in
  dec_dense_mma_fold width (Seq.slice weights start (start + 16 * width)) x (width / 16)

inline_for_extraction noextract
let dec_projection_cell
  (#nr:nat) (width:nat{dec_dense_width width}) (cols:nat)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:dec_bf16_values dec_weight_elements) (residual:lseq bf16 nr)
  (row:dec_spec_index 8) (col:dec_spec_index cols) : GTot bf16 =
  let scores = dec_projection_scores width cols layout offset x weights (col / 16 * 16) in
  let rounded = Casts.fcast #F32.t #bf16 (Chest.acc2 scores (col % 16) row) in
  if add_residual then ObF.add rounded (Seq.index residual (row * cols + col)) else rounded

inline_for_extraction noextract
let dec_projection_values
  (#nr:nat) (width:nat{dec_dense_width width}) (cols:nat)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:dec_bf16_values dec_weight_elements) (residual:lseq bf16 nr)
  : GTot (lseq bf16 (8 * cols)) =
  dec_indexed_matrix 8 cols (dec_projection_cell width cols layout offset add_residual x weights residual)

inline_for_extraction noextract
let opt_layer0_values (weights:lseq BF16.t opt_weight_count)
  : GTot (lseq BF16.t dec_weight_elements) =
  S.slice weights opt_layer0_start (opt_layer0_start + dec_weight_elements)

inline_for_extraction noextract
let dec_spec_sum (n:nat) (f:dec_spec_vector n) : GTot real =
  Sum.sum 0 n f

inline_for_extraction noextract
let rec dec_spec_sum_nonnegative (n:nat) (f:dec_spec_vector n)
  : Lemma
      (requires forall (i:dec_spec_index n). f i >=. 0.0R)
      (ensures dec_spec_sum n f >=. 0.0R)
      (decreases n)
  =
  if n > 0 then (
    dec_spec_sum_nonnegative (n - 1) f;
    Sum.sum_pop_right 0 n f
  )

inline_for_extraction noextract
let dec_spec_rms_norm (#n:pos) (dec_spec_weight x:dec_spec_vector n) : GTot (dec_spec_vector n) =
  let squares : dec_spec_vector n = fun i -> x i *. x i in
  dec_spec_sum_nonnegative n squares;
  let mean_square = dec_spec_sum n squares /. of_int n in
  let inverse_rms = Root.rsqrt (mean_square +. 0.000001R) in
  fun i -> dec_spec_weight i *. (x i *. inverse_rms)

inline_for_extraction noextract
let dec_bf16_segment_rep (#n:nat) (xs:lseq bf16 n)
  (start count:nat{start + count <= n}) (r:dec_spec_vector count) : prop =
  forall (i:dec_spec_index count). Seq.index xs (start + i) %~ r i

inline_for_extraction noextract
let u_rms_prefix (#rows #nw:pos) (x:lseq BF16.t (rows * 1024))
  (w:lseq BF16.t nw) (ys:lseq BF16.t (rows * 1024))
  (offset:nat{offset + 1024 <= nw}) (done:nat) : prop =
  forall (xr:dec_spec_matrix rows 1024) (gamma:dec_spec_vector 1024).
    {:pattern (dec_matrix_rep #rows #1024 x xr); (dec_bf16_segment_rep #nw w offset 1024 gamma)}
    dec_matrix_rep #rows #1024 x xr /\ dec_bf16_segment_rep #nw w offset 1024 gamma ==>
      dec_matrix_prefix #rows #1024 ys (fun row -> dec_spec_rms_norm gamma (xr row)) done

inline_for_extraction noextract
let dec_qk_head_number (row:dec_spec_index 8) (col:dec_spec_index 4096) : nat =
  row * 24 + col / 512 * 3 + (col % 512) / 128

inline_for_extraction noextract
let u_rms_guard (#n #nw:pos) (x:lseq BF16.t n) (w:lseq BF16.t nw)
  (start weight_start:nat) (width:pos{start + width <= n /\ weight_start + width <= nw})
  (r gamma:dec_spec_vector width) : prop =
  dec_bf16_segment_rep x start width r /\ dec_bf16_segment_rep w weight_start width gamma

inline_for_extraction noextract
let u_rope_weight_start (head:Shared.index 3) : nat =
  6293504 + (if head = 2 then 128 else 0)

[@@"opaque_to_smt"]
inline_for_extraction noextract
let u_rope_cell (x:lseq BF16.t 32768) (w:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  (lane:Shared.index 128) (value:BF16.t) : prop =
  forall (r gamma:dec_spec_vector 128) (cr sr:real).
    {:pattern (u_rms_guard x w (row * 4096 + group * 512 + head * 128)
      (u_rope_weight_start head) 128 r gamma);
      (S.index rt (row * 128 + lane) %~ cr);
      (S.index rt (524288 + row * 128 + lane) %~ sr)}
    u_rms_guard x w (row * 4096 + group * 512 + head * 128)
      (u_rope_weight_start head) 128 r gamma /\
    S.index rt (row * 128 + lane) %~ cr /\
    S.index rt (524288 + row * 128 + lane) %~ sr ==>
      value %~ (
        dec_spec_rms_norm gamma r lane *. cr +.
        (if lane < 64 then 0.0R -. dec_spec_rms_norm gamma r (lane + 64)
         else dec_spec_rms_norm gamma r (lane - 64)) *. sr)

inline_for_extraction noextract
let u_rope_prefix (x:lseq BF16.t 32768) (w:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (ys:lseq BF16.t 32768) (done:nat) : prop =
  forall (row:Shared.index 8) (col:Shared.index 4096).
    if row < 7 && col % 512 < 384 && dec_qk_head_number row col < done then
      u_rope_cell x w rt row (col / 512) ((col % 512) / 128) (col % 128)
        (S.index ys (row * 4096 + col))
    else S.index ys (row * 4096 + col) == S.index x (row * 4096 + col)

inline_for_extraction noextract
let dec_spec_silu (x:real) : real =
  x /. (1.0R +. Exp.exp (0.0R -. x))

inline_for_extraction noextract
let u_silu_prefix (x:lseq BF16.t 49152) (ys:lseq BF16.t 24576)
  (done:nat) : prop =
  forall (row:Shared.index 8) (col:Shared.index 3072) (gr ur:real).
    row * 3072 + col < done /\
    S.index x (row * 6144 + col / 64 * 128 + col % 64) %~ gr /\
    S.index x (row * 6144 + col / 64 * 128 + col % 64 + 64) %~ ur ==>
      S.index ys (row * 3072 + col) %~ (dec_spec_silu gr *. ur)

type dec_spec_heads (count:nat) = dec_spec_index 7 -> GTot (dec_spec_matrix count 128)

[@@erasable]
noeq
type dec_spec_attention_data = {
  dec_spec_position_ids: dec_spec_index 7 -> int;
  dec_spec_cache_positions: dec_spec_index 7 -> int;
  dec_spec_cosine: dec_spec_matrix 7 128;
  dec_spec_sine: dec_spec_matrix 7 128;
  dec_spec_additive_mask: dec_spec_matrix 7 7
}

inline_for_extraction noextract
noeq type u_attention_meaning = {
  u_query: dec_spec_heads 16;
  u_key: dec_spec_heads 8;
  u_value: dec_spec_heads 8;
  u_data: dec_spec_attention_data
}

inline_for_extraction noextract
let dec_spec_repeat_kv (x:dec_spec_heads 8) : GTot (dec_spec_heads 16) =
  fun token head lane -> x token (head / 2) lane

type dec_spec_attention_rows = dec_spec_index 7 -> GTot (dec_spec_matrix 16 7)

inline_for_extraction noextract
let dec_spec_dot (#n:nat) (x y:dec_spec_vector n) : GTot real =
  dec_spec_sum n (fun i -> x i *. y i)

inline_for_extraction noextract
let dec_spec_sum_positive (n:pos) (f:dec_spec_vector n)
  : Lemma
      (requires forall (i:dec_spec_index n). f i >. 0.0R)
      (ensures dec_spec_sum n f >. 0.0R)
  =
  dec_spec_sum_nonnegative (n - 1) f;
  Sum.sum_pop_right 0 n f

inline_for_extraction noextract
let dec_spec_softmax (#n:pos) (x:dec_spec_vector n) : GTot (dec_spec_vector n) =
  let numerators : dec_spec_vector n = fun i -> Exp.exp (x i) in
  dec_spec_sum_positive n numerators;
  let denominator = dec_spec_sum n numerators in
  fun i -> numerators i /. denominator

inline_for_extraction noextract
let dec_spec_prefix_softmax (token:dec_spec_index 7) (scores:dec_spec_vector (token + 1))
  : GTot (dec_spec_vector 7) =
  let probabilities = dec_spec_softmax scores in
  fun key_token ->
    if key_token <= token then probabilities key_token else 0.0R

inline_for_extraction noextract
let dec_spec_attention_probabilities
  (dec_spec_data:dec_spec_attention_data) (query repeated_key:dec_spec_heads 16)
  : GTot dec_spec_attention_rows =
  let scaling = Root.rsqrt 128.0R in
  fun token head ->
    dec_spec_prefix_softmax token (fun key_token ->
      dec_spec_dot (query token head) (repeated_key key_token head) *. scaling)

inline_for_extraction noextract
let dec_spec_attention_values (probabilities:dec_spec_attention_rows) (repeated_value:dec_spec_heads 16)
  : GTot (dec_spec_heads 16) =
  fun token head lane ->
    dec_spec_sum (token + 1) (fun key_token ->
      probabilities token head key_token *. repeated_value key_token head lane)

[@@"opaque_to_smt"]
inline_for_extraction noextract
let dec_attention_model (dec_spec_data:dec_spec_attention_data)
  (query:dec_spec_heads 16) (key dec_spec_values:dec_spec_heads 8) : GTot (dec_spec_matrix 8 2048) =
  fun row channel ->
    if row < 7 then
      dec_spec_attention_values
        (dec_spec_attention_probabilities dec_spec_data query (dec_spec_repeat_kv key))
        (dec_spec_repeat_kv dec_spec_values) row (channel / 128) (channel % 128)
    else 0.0R

inline_for_extraction noextract
let u_attention_model (m:u_attention_meaning) : GTot (dec_spec_matrix 8 2048) =
  dec_attention_model m.u_data m.u_query m.u_key m.u_value

inline_for_extraction noextract
let dec_source_qk_representation
  (rotated:lseq bf16 (8 * 4096)) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  : prop =
  (forall (token:dec_spec_index 7) (head:dec_spec_index 16) (lane:dec_spec_index 128).
    v_approximates
      (ObSeq.index rotated
        (token * 4096 + (head / 2) * 512 + (head % 2) * 128 + lane))
      (query token head lane)) /\
  (forall (token:dec_spec_index 7) (head:dec_spec_index 8) (lane:dec_spec_index 128).
    v_approximates (ObSeq.index rotated (token * 4096 + head * 512 + 256 + lane))
      (key token head lane))

inline_for_extraction noextract
let dec_source_values_representation (rotated:dec_bf16_values 32768) (dec_spec_values:dec_spec_heads 8) : prop =
  forall (row:dec_spec_index 7) (head:dec_spec_index 8) (lane:dec_spec_index 128).
    v_approximates (Seq.index rotated (row * 4096 + head * 512 + 384 + lane)) (dec_spec_values row head lane)

inline_for_extraction noextract
let dec_source_query_operand
  (rotated:dec_bf16_values (8 * 4096)) (head:dec_spec_index 8) (step:dec_spec_index 8)
  : GTot (ScoreChest.chest2 BF16.t 16 16) =
  ScoreChest.mk2 (fun row lane ->
    if row < 14 then
      Seq.index rotated
        ((row / 2) * 4096 + head * 512 + (row % 2) * 128 + step * 16 + lane)
    else ScoreF.zero)

inline_for_extraction noextract
let dec_source_key_operand
  (rotated:dec_bf16_values (8 * 4096)) (head:dec_spec_index 8) (step:dec_spec_index 8)
  : GTot (ScoreChest.chest2 BF16.t 16 8) =
  ScoreChest.mk2 (fun lane key ->
    if key < 7 then
      Seq.index rotated (key * 4096 + head * 512 + 256 + step * 16 + lane)
    else ScoreF.zero)

inline_for_extraction noextract
let rec dec_source_mma_scores
  (rotated:dec_bf16_values (8 * 4096)) (head:dec_spec_index 8) (steps:nat{steps <= 8})
  : GTot (ScoreChest.chest2 F32.t 16 8) (decreases steps) =
  if steps = 0 then ScoreChest.const (16 @| 8 @| INil) (ScoreF.zero #F32.t)
  else ScoreMMA.emma (dec_source_mma_scores rotated head (steps - 1))
    (dec_source_query_operand rotated head (steps - 1))
    (dec_source_key_operand rotated head (steps - 1))

inline_for_extraction noextract
let dec_bilateral_mma_live_scores_finite
  (rotated:dec_bf16_values (8 * 4096)) (head:dec_spec_index 8) : prop =
  forall (row:dec_spec_index 14) (col:dec_spec_index 7{col <= row / 2}).
    Finite? (kind (ScoreChest.acc2 (dec_source_mma_scores rotated head 8) row col))

inline_for_extraction noextract
let dec_attention_domain (rotated:dec_bf16_values 32768) : prop =
  forall (head:dec_spec_index 8). dec_bilateral_mma_live_scores_finite rotated head

inline_for_extraction noextract
let u_attention_guard (x:lseq BF16.t 32768) (m:u_attention_meaning) : prop =
  dec_attention_domain x /\
  dec_source_qk_representation x m.u_query m.u_key /\
  dec_source_values_representation x m.u_value

inline_for_extraction noextract
let u_block_stage (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) (stage:nat{stage <= 11}) : GTot prop =
  (1 <= stage ==>
    emb_embedding_post 151936 1024 7 8 (S.slice weights 0 emb_embedding_count) prompt v.v_embedding /\
    emb_rope_post 4096 64 v.v_rope) /\
  (2 <= stage ==> u_rms_prefix #8 #opt_weight_count v.v_embedding weights v.v_normalized0 155583488 8) /\
  (3 <= stage ==>
    v.v_qkv == dec_projection_values 1024 4096 1 2048 false
      v.v_normalized0 (opt_layer0_values weights) v.v_embedding) /\
  (4 <= stage ==> u_rope_prefix v.v_qkv (opt_layer0_values weights) v.v_rope v.v_rotated 168) /\
  (5 <= stage ==>
    (forall (m:u_attention_meaning). u_attention_guard v.v_rotated m ==>
      dec_matrix_rep v.v_attention (u_attention_model m)) /\
    (forall (col:Shared.index 2048). S.index v.v_attention (14336 + col) == (F.zero #BF16.t))) /\
  (6 <= stage ==>
    v.v_residual == dec_projection_values 2048 1024 0 4196352 true
      v.v_attention (opt_layer0_values weights) v.v_embedding) /\
  (7 <= stage ==> u_rms_prefix #8 #opt_weight_count v.v_residual weights v.v_postnorm 155584512 8) /\
  (8 <= stage ==>
    v.v_gateup == dec_projection_values 1024 6144 2 6293760 false
      v.v_postnorm (opt_layer0_values weights) v.v_residual) /\
  (9 <= stage ==> u_silu_prefix v.v_gateup v.v_activated 24576) /\
  (10 <= stage ==>
    v.v_hidden == dec_projection_values 3072 1024 0 12585216 true
      v.v_activated (opt_layer0_values weights) v.v_residual) /\
  (11 <= stage ==> u_rms_prefix #8 #opt_weight_count v.v_hidden weights v.v_normalized1 171314432 8)

inline_for_extraction noextract
let u_stage_resources (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7) (a:opt_block_addresses)
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) (stage:nat{stage <= 11}) : slprop =
  PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
  opt_block_storage a v ** pure (u_block_stage weights prompt v stage)

inline_for_extraction noextract
let u_pending_stage (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7) (a:opt_block_addresses)
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (stage:nat{stage <= 11}) : slprop =
  exists* (v:opt_block_values). u_stage_resources weights_address prompt_address a weights prompt v stage

inline_for_extraction noextract
let u_weight_window (weights:larray BF16.t opt_weight_count)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= opt_weight_count}) : GTot (larray BF16.t (hi - lo)) =
  PA.gsub weights lo hi

inline_for_extraction noextract
noeq type opt_pending_family (a:Type0) (pre post:a -> slprop) = {
  pending_threads: (t:szp{t <= max_threads});
  pending_shared: list SH.shmem_desc;
  pending_code: x:erased a -> Tot
    (k:KS.kernel_desc (pre (FStar.Ghost.reveal #a x)) (post (FStar.Ghost.reveal #a x)){
      SZ.v k.nblk == 1 /\ k.nthr == pending_threads /\ k.shmems_desc == pending_shared /\
      (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
        k.barrier_contract bid sh == Barrier.empty_contract pending_threads /\
        k.barrier_count bid == 0)})
}

inline_for_extraction noextract
let opt_scalar_shared_desc : list SH.shmem_desc = []

inline_for_extraction noextract
ghost
fn opt_single_block_setup (p:slprop) ()
  norewrite
  requires p
  ensures (forall+ (_:natlt 1). p) ** emp

inline_for_extraction noextract
ghost
fn opt_single_block_teardown (p:slprop) ()
  norewrite
  requires (forall+ (_:natlt 1). p) ** emp
  ensures p

inline_for_extraction noextract
ghost
fn opt_scalar_block_setup (pre:slprop) (sh:SH.c_shmems opt_scalar_shared_desc)
  (_:natlt 1) ()
  norewrite
  requires SH.live_c_shmems sh ** pre
  ensures (forall+ (_:natlt 1). pre) ** emp

inline_for_extraction noextract
ghost
fn opt_scalar_block_teardown (post:slprop) (sh:SH.c_shmems opt_scalar_shared_desc)
  (_:natlt 1) ()
  norewrite
  requires (forall+ (_:natlt 1). post) ** emp
  ensures SH.live_c_shmems sh ** post

inline_for_extraction noextract
fn opt_scalar_thread (#pre #post:slprop) (code:KS.kernel_desc_1_1 pre post)
  (sh:SH.c_shmems opt_scalar_shared_desc) (bid tid:szlt 1) ()
  requires gpu ** pure (SH.c_shmems_inv sh) ** pre **
    thread_id 1 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 1) ** Barrier.barrier_state 0
  ensures gpu ** post ** thread_id 1 tid ** block_id 1 bid **
    Barrier.barrier_tok (Barrier.empty_contract 1) ** Barrier.barrier_state 0

inline_for_extraction noextract
val opt_scalar_descriptor (#pre #post:slprop) (code:KS.kernel_desc_1_1 pre post)
  : k:KS.kernel_desc pre post{
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)}

inline_for_extraction noextract
let emb_valid_tokens (ts:lseq EmbI64.t 7) : prop =
  forall (i:nat{i < 7}).
    0 <= EmbI64.v (EmbS.index ts i) /\ EmbI64.v (EmbS.index ts i) < 151936

inline_for_extraction noextract
let emb_hidden_count : pos = 8 * 1024

inline_for_extraction noextract
let emb_table_count : pos = 2 * 4096 * 128

inline_for_extraction noextract
type emb_embedding_values = lseq bf16 emb_embedding_count

inline_for_extraction noextract
type emb_hidden_values = lseq bf16 emb_hidden_count

inline_for_extraction noextract
type emb_table_values = lseq bf16 emb_table_count

inline_for_extraction noextract
let emb_embedding_cell
  (vocab width tokens rows : nat)
  (weights : lseq bf16 (vocab * width))
  (prompt : lseq EmbI64.t tokens)
  (embedding : lseq bf16 (rows * width))
  (row : nat{row < rows})
  (column : nat{column < width})
  : prop =
  let token = emb_padded_token prompt row in
  if 0 <= token && token < vocab then
    EmbS.index embedding (row * width + column) ==
      EmbS.index weights (token * width + column)
  else False

inline_for_extraction noextract
let emb_embedding_ready
  (vocab width tokens rows : nat)
  (weights : lseq bf16 (vocab * width))
  (prompt : lseq EmbI64.t tokens)
  (embedding : lseq bf16 (rows * width))
  (done_rows done_columns : nat)
  : prop =
  forall (r : nat{r < rows}) (c : nat{c < width}).
    (r < done_rows \/ (r == done_rows /\ c < done_columns)) ==>
      emb_embedding_cell vocab width tokens rows weights prompt embedding r c

inline_for_extraction noextract
let emb_valid_prompt (vocab tokens:nat) (prompt:lseq EmbI64.t tokens) : prop =
  forall (i:nat{i < tokens}).
    0 <= EmbI64.v (EmbS.index prompt i) /\ EmbI64.v (EmbS.index prompt i) < vocab

inline_for_extraction noextract
let emb_row_major_unique
  (width : pos)
  (row other_row : nat)
  (column other_column : nat{other_column < width})
  : Lemma
    (requires column < width)
    (ensures
      (row * width + column == other_row * width + other_column)
      <==> (row == other_row /\ column == other_column))
  =
  Kuiper.Math.div_mod_of_mul_add width row column;
  Kuiper.Math.div_mod_of_mul_add width other_row other_column

inline_for_extraction noextract
let emb_token_range
  (vocab : pos)
  (tokens : nat)
  (prompt : lseq EmbI64.t tokens)
  (row : nat)
  : Lemma
    (requires emb_valid_prompt vocab tokens prompt)
    (ensures 0 <= emb_padded_token prompt row /\ emb_padded_token prompt row < vocab)
  = if row < tokens then () else ()

inline_for_extraction noextract
let emb_embedding_step
  (vocab : pos) (width : pos) (tokens rows : nat)
  (weights : lseq bf16 (vocab * width))
  (prompt : lseq EmbI64.t tokens)
  (embedding : lseq bf16 (rows * width))
  (row : nat{row < rows})
  (column : nat{column < width})
  (value : bf16)
  : Lemma
    (requires
      emb_valid_prompt vocab tokens prompt /\
      emb_embedding_ready vocab width tokens rows weights prompt embedding row column /\
      (let token = emb_padded_token prompt row in
       if 0 <= token && token < vocab then
         value == EmbS.index weights (token * width + column)
       else False))
    (ensures
      emb_embedding_ready vocab width tokens rows weights prompt
        (EmbS.upd embedding (row * width + column) value) row (column + 1))
  =
  let after = EmbS.upd embedding (row * width + column) value in
  let at_cell (r : nat{r < rows}) (c : nat{c < width})
    : Lemma
      ((r < row \/ (r == row /\ c < column + 1)) ==>
        emb_embedding_cell vocab width tokens rows weights prompt after r c)
    =
    emb_row_major_unique width row r column c;
    emb_token_range vocab tokens prompt r;
    if r < row || (r = row && c < column + 1) then (
      if r = row && c = column then
        EmbS.lemma_index_upd1 embedding (row * width + column) value
      else (
        assert (emb_embedding_cell vocab width tokens rows weights prompt embedding r c);
        EmbS.lemma_index_upd2 embedding (row * width + column) value (r * width + c)
      )
    )
  in
  EmbC.forall_intro_2 at_cell

inline_for_extraction noextract
let emb_embedding_complete
  (vocab : pos) (width tokens rows : nat)
  (weights : lseq bf16 (vocab * width))
  (prompt : lseq EmbI64.t tokens)
  (embedding : lseq bf16 (rows * width))
  : Lemma
    (requires emb_valid_prompt vocab tokens prompt /\
      emb_embedding_ready vocab width tokens rows weights prompt embedding rows 0)
    (ensures emb_embedding_post vocab width tokens rows weights prompt embedding)
  =
  let valid_row (r : nat)
    : Lemma (0 <= emb_padded_token prompt r /\ emb_padded_token prompt r < vocab)
    = emb_token_range vocab tokens prompt r
  in
  EmbC.forall_intro valid_row;
  let at_cell (r : nat{r < rows}) (c : nat{c < width})
    : Lemma (
      let token = emb_padded_token prompt r in
      if 0 <= token && token < vocab then
        EmbS.index embedding (r * width + c) == EmbS.index weights (token * width + c)
      else False)
    = assert (emb_embedding_cell vocab width tokens rows weights prompt embedding r c)
  in
  EmbC.forall_intro_2 at_cell;
  assert (emb_embedding_post vocab width tokens rows weights prompt embedding) by (
    FStar.Tactics.V2.norm [delta_only [`%emb_embedding_post]; iota; zeta];
    FStar.Tactics.V2.smt ()
  )

inline_for_extraction noextract
let emb_index64 (value:EmbI64.t{0 <= EmbI64.v value /\ EmbSZ.fits (EmbI64.v value)})
  : i:sz{EmbSZ.v i == EmbI64.v value} =
  assert_norm (pow2 32 == 0x100000000);
  EmbSZ.uint32_to_sizet (EmbIC.int64_to_uint32 value)

inline_for_extraction noextract
let emb_index32 (value:EmbI32.t{0 <= EmbI32.v value})
  : i:sz{EmbSZ.v i == EmbI32.v value} =
  assert_norm (pow2 32 == 0x100000000);
  EmbSZ.uint32_to_sizet (EmbIC.int32_to_uint32 value)

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

inline_for_extraction noextract
let emb_rope_ready
  (positions : pos{positions < 0x100000000})
  (half_width : pos{half_width < 0x100000000})
  (table : lseq bf16 (4 * positions * half_width))
  (done_frequencies done_positions : nat)
  : prop =
  forall
    (component : nat{component < 2})
    (position : nat{position < positions})
    (column : nat{column < 2 * half_width}).
    (column % half_width < done_frequencies \/
      (column % half_width == done_frequencies /\ position < done_positions)) ==>
    EmbS.index table ((component * positions + position) * (2 * half_width) + column)
      == emb_rotary_value half_width position component column

inline_for_extraction noextract
let emb_rope_next_frequency
  (positions : pos{positions < 0x100000000})
  (half_width : pos{half_width < 0x100000000})
  (table : lseq bf16 (4 * positions * half_width))
  (frequency : nat{frequency < half_width})
  : Lemma
    (requires emb_rope_ready positions half_width table frequency positions)
    (ensures emb_rope_ready positions half_width table (frequency + 1) 0)
  = ()

inline_for_extraction noextract
let emb_rope_complete
  (positions : pos{positions < 0x100000000})
  (half_width : pos{half_width < 0x100000000})
  (table : lseq bf16 (4 * positions * half_width))
  : Lemma
    (requires emb_rope_ready positions half_width table half_width 0)
    (ensures emb_rope_post positions half_width table)
  = ()

inline_for_extraction noextract
let emb_coordinate (i:EmbI32.t{0 <= EmbI32.v i})
  : r:f32{r == emb_coordinate_f32 (EmbI32.v i)} =
  let integer = EmbIC.int32_to_int64 i in
  EmbI64.v_inj integer (EmbI64.int_to_t (EmbI32.v i));
  of_int #f32 integer

inline_for_extraction noextract
let emb_table_index_unique
  (positions half_width : pos)
  (component other_component : nat{other_component < 2})
  (position other_position : nat{other_position < positions})
  (column other_column : nat{other_column < 2 * half_width})
  : Lemma
    (requires component < 2 /\ position < positions /\ column < 2 * half_width)
    (ensures
      (((component * positions + position) * (2 * half_width) + column ==
        (other_component * positions + other_position) * (2 * half_width) + other_column)
       <==>
       (component == other_component /\ position == other_position /\ column == other_column)))
  =
  emb_row_major_unique (2 * half_width)
    (component * positions + position) (other_component * positions + other_position)
    column other_column;
  emb_row_major_unique positions component other_component position other_position

inline_for_extraction noextract
let emb_updated_index
  (values : seq bf16)
  (at : nat{at < EmbS.length values})
  (value : bf16)
  (index : nat{index < EmbS.length values})
  : Lemma
    (EmbS.index (EmbS.upd values at value) index ==
      (if index = at then value else EmbS.index values index))
  =
  if index = at then EmbS.lemma_index_upd1 values at value
  else EmbS.lemma_index_upd2 values at value index

inline_for_extraction noextract
let emb_table_flat_index (positions half_width:pos)
  (k:nat{k < 2}) (p:nat{p < positions}) (c:nat{c < 2 * half_width})
  : Tot (i:nat{i < 4 * positions * half_width /\
      i == (k * positions + p) * (2 * half_width) + c}) =
  if k = 0 then () else assert (k == 1);
  (k * positions + p) * (2 * half_width) + c

inline_for_extraction noextract
let emb_rope_step
  (positions : pos{positions < 0x100000000})
  (half_width : pos{half_width < 0x100000000})
  (table : lseq bf16 (4 * positions * half_width))
  (frequency : nat{frequency < half_width})
  (position : nat{position < positions})
  (cosine sine : bf16)
  : Lemma
    (requires
      emb_rope_ready positions half_width table frequency position /\
      cosine == emb_rotary_value half_width position 0 frequency /\
      sine == emb_rotary_value half_width position 1 frequency)
    (ensures (
      let a = position * (2 * half_width) + frequency in
      let b = positions * (2 * half_width) + a in
      let after = EmbS.upd (EmbS.upd (EmbS.upd (EmbS.upd table
        a cosine) (a + half_width) cosine) b sine) (b + half_width) sine in
      emb_rope_ready positions half_width after frequency (position + 1)))
  =
  let a = position * (2 * half_width) + frequency in
  let b = positions * (2 * half_width) + a in
  let first : lseq bf16 (4 * positions * half_width) = EmbS.upd table a cosine in
  let second : lseq bf16 (4 * positions * half_width) =
    EmbS.upd first (a + half_width) cosine in
  let third : lseq bf16 (4 * positions * half_width) = EmbS.upd second b sine in
  let after : lseq bf16 (4 * positions * half_width) =
    EmbS.upd third (b + half_width) sine in
  Kuiper.Math.div_mod_of_mul_add half_width 0 frequency;
  Kuiper.Math.div_mod_of_mul_add half_width 1 frequency;
  let at_cell
    (k : nat{k < 2})
    (p : nat{p < positions})
    (c : nat{c < 2 * half_width})
    : Lemma
      ((c % half_width < frequency \/
        (c % half_width == frequency /\ p < position + 1)) ==>
       EmbS.index after (emb_table_flat_index positions half_width k p c)
         == emb_rotary_value half_width p k c)
    =
    emb_table_index_unique positions half_width k 0 p position c frequency;
    emb_table_index_unique positions half_width k 0 p position c (frequency + half_width);
    emb_table_index_unique positions half_width k 1 p position c frequency;
    emb_table_index_unique positions half_width k 1 p position c (frequency + half_width);
    (if c < half_width then
      Kuiper.Math.div_mod_of_mul_add half_width 0 c
    else
      Kuiper.Math.div_mod_of_mul_add half_width 1 (c - half_width));
    let index = emb_table_flat_index positions half_width k p c in
    emb_updated_index table a cosine index;
    emb_updated_index first (a + half_width) cosine index;
    emb_updated_index second b sine index;
    emb_updated_index third (b + half_width) sine index;
    if c % half_width < frequency ||
      (c % half_width = frequency && p < position + 1) then (
      if c % half_width = frequency && p = position then (
        assert (c == frequency \/ c == frequency + half_width);
        if k = 0 then (
          assert (index == a \/ index == a + half_width);
          assert (EmbS.index after index == cosine);
          assert (emb_rotary_value half_width p k c == emb_rotary_value half_width position 0 frequency)
        ) else (
          assert (index == b \/ index == b + half_width);
          assert (EmbS.index after index == sine);
          assert (emb_rotary_value half_width p k c == emb_rotary_value half_width position 1 frequency)
        )
      ) else (
        assert (c % half_width < frequency \/
          (c % half_width == frequency /\ p < position));
        assert (EmbS.index table index == emb_rotary_value half_width p k c);
        assert (index <> a /\ index <> a + half_width /\ index <> b /\ index <> b + half_width);
        assert (EmbS.index after index == EmbS.index table index)
      )
    )
  in
  EmbC.forall_intro_3 at_cell

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

inline_for_extraction noextract
fn emb_make_rope
  (table : larray bf16 emb_table_count)
  (#before : erased emb_table_values)
  preserves gpu
  requires table |-> before
  ensures exists* (after : emb_table_values). table |-> after **
    pure (emb_rope_post 4096 64 after)

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

inline_for_extraction noextract
let u_weight_window_frame (weights:larray BF16.t opt_weight_count)
  (contents:lseq BF16.t opt_weight_count)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= opt_weight_count}) (fraction:perm) : slprop =
  PA.pts_to_range weights 0 lo #fraction (S.slice contents 0 lo) **
  PA.pts_to_range weights hi opt_weight_count #fraction (S.slice contents hi opt_weight_count)

inline_for_extraction noextract
let u_weight_partition (contents:lseq BF16.t opt_weight_count)
  (lo:nat) (hi:nat{lo <= hi /\ hi <= opt_weight_count})
  : Lemma (
      S.slice (S.slice contents lo opt_weight_count) 0 (hi - lo) == S.slice contents lo hi /\
      S.slice (S.slice contents lo opt_weight_count) (hi - lo) (opt_weight_count - lo) ==
        S.slice contents hi opt_weight_count /\
      S.append (S.append (S.slice contents 0 lo) (S.slice contents lo hi))
        (S.slice contents hi opt_weight_count) == contents) =
  SP.slice_slice contents lo opt_weight_count 0 (hi - lo);
  SP.slice_slice contents lo opt_weight_count (hi - lo) (opt_weight_count - lo);
  assert (S.equal (S.append (S.append (S.slice contents 0 lo) (S.slice contents lo hi))
    (S.slice contents hi opt_weight_count)) contents);
  S.lemma_eq_elim (S.append (S.append (S.slice contents 0 lo) (S.slice contents lo hi))
    (S.slice contents hi opt_weight_count)) contents

inline_for_extraction noextract
ghost
fn u_take_weight_window (weights:larray BF16.t opt_weight_count)
  (#contents:erased (lseq BF16.t opt_weight_count))
  (lo:nat) (hi:nat{lo <= hi /\ hi <= opt_weight_count}) (#fraction:perm)
  requires PA.pts_to weights #fraction contents
  ensures PA.pts_to (u_weight_window weights lo hi) #fraction (S.slice contents lo hi) **
    u_weight_window_frame weights contents lo hi fraction

inline_for_extraction noextract
ghost
fn u_return_weight_window (weights:larray BF16.t opt_weight_count)
  (#contents:erased (lseq BF16.t opt_weight_count))
  (lo:nat) (hi:nat{lo <= hi /\ hi <= opt_weight_count}) (#fraction:perm)
  requires PA.pts_to (u_weight_window weights lo hi) #fraction (S.slice contents lo hi) **
    u_weight_window_frame weights contents lo hi fraction
  ensures PA.pts_to weights #fraction contents

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

inline_for_extraction noextract
let u_embedding_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (embedding_weights:larray BF16.t emb_embedding_count{
    embedding_weights == u_weight_window weights_address 0 emb_embedding_count})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased opt_block_values)
  : k:KS.kernel_desc
      (u_stage_resources weights_address prompt_address a weights prompt v 0)
      (u_pending_stage weights_address prompt_address a weights prompt 1){
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)} =
  let raw:KS.kernel_desc_1_1
      (u_stage_resources weights_address prompt_address a weights prompt v 0)
      (u_pending_stage weights_address prompt_address a weights prompt 1) = {
    f = u_embedding_stage_body weights_address prompt_address a embedding_weights #weights #prompt #v;
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in opt_scalar_descriptor raw

inline_for_extraction noextract
let u_embedding_stage
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (embedding_weights:larray BF16.t emb_embedding_count{
    embedding_weights == u_weight_window weights_address 0 emb_embedding_count})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family opt_block_values
      (fun v -> u_stage_resources weights_address prompt_address a weights prompt v 0)
      (fun _ -> u_pending_stage weights_address prompt_address a weights prompt 1) = {
    pending_threads = 1sz;
    pending_shared = opt_scalar_shared_desc;
    pending_code = u_embedding_stage_descriptor weights_address prompt_address a embedding_weights #weights #prompt
  }

inline_for_extraction noextract
let dec_source_loop_end (i bound:sz)
  : Lemma
      (requires i <= bound /\ not (i <^ bound))
      (ensures SZ.v i == SZ.v bound) = ()

inline_for_extraction noextract
fn dec_copy_bf16
  (#n:erased nat) (input output:larray bf16 n) (count:sz{SZ.v count == n})
  (#x #old:erased (lseq bf16 n)) (#f:perm)
  preserves gpu ** input |-> Frac f x
  requires output |-> old
  ensures output |-> x

inline_for_extraction noextract
let dec_segment_frame (#a:Type0) (#n:nat) (before after:lseq a n)
  (start count:nat) : prop =
  forall (i:nat{i < n}). (i < start \/ start + count <= i) ==>
    Seq.index after i == Seq.index before i

inline_for_extraction noextract
let source_bf16_region (#n:nat)
  (before after:lseq (option BF16.t) n)
  (start:nat) (width:pos{start + width <= n}) (done:nat{done <= width})
  (model:dec_spec_vector width) (guard:prop) : prop =
  dec_segment_frame #(option BF16.t) #n before after start done /\
  (forall (j:Shared.index width{j < done}).
    Some? (S.index after (start + j)) /\
    (forall (v:BF16.t). S.index after (start + j) == Some v ==>
      (guard ==> v %~ model j)))

inline_for_extraction noextract
let source_bf16_full_write_initialized (#n:pos)
  (before after:lseq (option BF16.t) n) (model:dec_spec_vector n) (guard:prop)
  : Lemma
      (requires source_bf16_region #n before after 0 n n model guard)
      (ensures forall (i:Shared.index n). Some? (S.index after i)) =
  let at (i:Shared.index n) : Lemma (Some? (S.index after i)) =
    assert (Some? (S.index after (0 + i)))
  in C.forall_intro at

inline_for_extraction noextract
let dec_f32_to_bf16_all (value:F32.t)
  : Lemma (forall (r:real).
      value %~ r ==> Casts.fcast #F32.t #bf16 value %~ r) =
  FStar.Classical.forall_intro
    (FStar.Classical.move_requires (Casts.fcast_approx #F32.t #bf16 value))

inline_for_extraction noextract
let dec_bf16_to_f32_all (value:bf16)
  : Lemma (forall (r:real).
      value %~ r ==> Casts.fcast #bf16 #F32.t value %~ r) =
  FStar.Classical.forall_intro
    (FStar.Classical.move_requires (Casts.fcast_approx #bf16 #F32.t value))

inline_for_extraction noextract
let dec_source_negate (value:F32.t) : F32.t = NativeNeg.neg value

inline_for_extraction noextract
let dec_source_negate_approx (value:F32.t) (r:real)
  : Lemma (requires v_approximates value r)
          (ensures v_approximates (dec_source_negate value) (0.0R -. r)) =
  NativeNeg.neg_approx value r

inline_for_extraction noextract
let dec_source_negate_all (value:F32.t)
  : Lemma (forall (r:real). value %~ r ==> dec_source_negate value %~ (0.0R -. r)) =
  FStar.Classical.forall_intro (FStar.Classical.move_requires (dec_source_negate_approx value))

inline_for_extraction noextract
fn u_rotary_source_value (value partner cosine sine:BF16.t) (lower:bool)
  preserves gpu
  returns result:BF16.t
  ensures pure (forall (vr pr cr sr:real).
    value %~ vr /\ partner %~ pr /\ cosine %~ cr /\ sine %~ sr ==>
      result %~ (vr *. cr +. (if lower then 0.0R -. pr else pr) *. sr))

inline_for_extraction noextract
let dec_rms_width (n:nat) : prop = n = 128 \/ n = 1024

inline_for_extraction noextract
let dec_rms_lanes (n:nat{dec_rms_width n}) : pos = if n = 1024 then 256 else 128

inline_for_extraction noextract
let dec_rms_outer_width (n:nat{dec_rms_width n}) : pos = if n = 1024 then 8 else 32

inline_for_extraction noextract
let dec_rms_radicand (n:pos) (r:dec_spec_vector n) : GTot ObRoot.rpos =
  dec_spec_sum_nonnegative n (fun j -> r j *. r j);
  dec_spec_sum n (fun j -> r j *. r j) /. of_int n +. 0.000001R

inline_for_extraction noextract
let dec_source_epsilon () : F32.t = ObF.of_literal "0.000001"

inline_for_extraction noextract
let dec_source_epsilon_approx ()
  : Lemma (dec_source_epsilon () %~ 0.000001R) =
  assert_norm (Kuiper.Float32.Literals.common_normal_cell
    "0.000001" 1 1000000 1 1000000 8796093 (-43));
  F32.literal_interval_approx "0.000001" 1 1000000 1 1000000
    8796093 (-43) 0.000001R

inline_for_extraction noextract
let dec_source_inverse_width (width:szp{dec_rms_width width})
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
#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
inline_for_extraction noextract
let dec_real_divide_product (x:real) (d:real{d =!= 0.0R})
  : Lemma (x /. d == x *. (1.0R /. d)) = ()
#pop-options

inline_for_extraction noextract
let dec_rms_radicand_at (n:pos) (r:dec_spec_vector n)
  : Lemma
      (dec_rms_radicand n r ==
        dec_spec_sum n (fun j -> r j *. r j) *. (1.0R /. of_int n) +. 0.000001R) =
  dec_spec_sum_nonnegative n (fun j -> r j *. r j);
  assert (of_int n >. 0.0R);
  assert (dec_rms_radicand n r ==
    dec_spec_sum n (fun j -> r j *. r j) /. of_int n +. 0.000001R) by
    (FStar.Tactics.V2.norm [delta_only [`%dec_rms_radicand]; iota; zeta];
     FStar.Tactics.V2.trivial ());
  dec_real_divide_product (dec_spec_sum n (fun j -> r j *. r j)) (of_int n)

inline_for_extraction noextract
let u_rms_radicand_all (width:pos)
  : Lemma (forall (r:dec_spec_vector width).
      dec_rms_radicand width r ==
        dec_spec_sum width (fun i -> r i *. r i) *. (1.0R /. of_int width) +. 0.000001R) =
  C.forall_intro (dec_rms_radicand_at width)

inline_for_extraction noextract
fn u_rms_source_scale (reduced:F32.t) (width:szp{dec_rms_width width})
  preserves gpu
  returns scale:F32.t
  ensures pure (forall (r:dec_spec_vector width).
    reduced %~ dec_spec_sum width (fun i -> r i *. r i) ==>
      v_approximates scale (ObRoot.rsqrt (dec_rms_radicand width r)))

inline_for_extraction noextract
let rec dec_source_sum_n (n:nat) (f:nat -> GTot real)
  : GTot real (decreases n) =
  if n = 0 then 0.0R else dec_source_sum_n (n - 1) f +. f (n - 1)

inline_for_extraction noextract
let dec_rms_square (n:nat{dec_rms_width n}) (r:dec_spec_vector n) (i:nat) : GTot real =
  if i < n then r i *. r i else 0.0R

inline_for_extraction noextract
let dec_rms_lane_model (n:nat{dec_rms_width n}) (r:dec_spec_vector n) (lane:nat) : GTot real =
  dec_source_sum_n (n / dec_rms_lanes n)
    (fun step -> dec_rms_square n r (lane + step * dec_rms_lanes n))

inline_for_extraction noextract
let dec_rms_warp_model (n:nat{dec_rms_width n}) (r:dec_spec_vector n) (warp:nat) : GTot real =
  if warp < dec_rms_lanes n / 32 then
    dec_source_sum_n 32 (fun lane -> dec_rms_lane_model n r (warp * 32 + lane))
  else 0.0R

inline_for_extraction noextract
let source_mask_rep (#n:nat) (mask:lseq (option F32.t) n) (count:nat{count <= n})
  (model:nat -> GTot real) (guard:prop) : prop =
  forall (i:nat{i < count}).
    i < n /\ Some? (S.index mask i) /\
    (forall (v:F32.t). S.index mask i == Some v ==> (guard ==> v %~ model i))

inline_for_extraction noextract
let dec_reduce_width (n:nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32

inline_for_extraction noextract
let dec_segment_rep (#n:nat) (xs:lseq F32.t n)
  (start count:nat{start + count <= n}) (r:nat -> GTot real) : prop =
  forall (i:nat{i < count}). Seq.index xs (start + i) %~ r i

inline_for_extraction noextract
let u_reduce_result (#n:pos) (original after:lseq F32.t n)
  (start:nat) (count:pos{start + count <= n}) : prop =
  dec_segment_frame original after start count /\
  (forall (model:nat -> GTot real).
    dec_segment_rep original start count model ==>
      S.index after start %~ dec_source_sum_n count model)

inline_for_extraction noextract
let u_reduce_prefix (#n:pos) (original current:lseq F32.t n)
  (start:nat) (count:pos{start + count <= n /\ count % 2 == 0})
  (done:nat{done <= count / 2}) : prop =
  dec_segment_frame original current start count /\
  (forall (model:nat -> GTot real).
    dec_segment_rep original start count model ==>
      (forall (i:nat{i < count}).
        S.index current (start + i) %~
          (if i < done then model i +. model (count / 2 + i) else model i)))

inline_for_extraction noextract
let u_reduce_step (#n:pos) (original before:lseq F32.t n)
  (start:nat) (count:pos{start + count <= n /\ count % 2 == 0})
  (i:nat{i < count / 2}) (value:F32.t)
  : Lemma
      (requires u_reduce_prefix original before start count i /\
        (forall (a b:real).
          S.index before (start + i) %~ a /\
          S.index before (start + i + count / 2) %~ b ==> value %~ (a +. b)))
      (ensures u_reduce_prefix original (S.upd before (start + i) value) start count (i + 1)) =
  let at (model:nat -> GTot real) (j:nat{j < count})
    : Lemma (dec_segment_rep original start count model ==>
      S.index (S.upd before (start + i) value) (start + j) %~
        (if j < i + 1 then model j +. model (count / 2 + j) else model j)) =
    if i = j then S.lemma_index_upd1 before (start + i) value
    else S.lemma_index_upd2 before (start + i) value (start + j)
  in C.forall_intro_2 at

inline_for_extraction noextract
let rec dec_source_sum_append (n m:nat) (f:nat -> GTot real)
  : Lemma
      (dec_source_sum_n (n + m) f ==
        dec_source_sum_n n f +. dec_source_sum_n m (fun i -> f (n + i)))
      (decreases m) =
  if m > 0 then dec_source_sum_append n (m - 1) f

inline_for_extraction noextract
let dec_sum_halves (n:nat) (f:nat -> GTot real)
  : Lemma
      (dec_source_sum_n (2 * n) f ==
        dec_source_sum_n n (fun i -> f i +. f (n + i)))
  =
  dec_source_sum_append n n f;
  let rec combine (k:nat{k <= n})
    : Lemma
        (dec_source_sum_n k f +. dec_source_sum_n k (fun i -> f (n + i)) ==
          dec_source_sum_n k (fun i -> f i +. f (n + i)))
        (decreases k) =
    if k > 0 then combine (k - 1)
  in combine n

inline_for_extraction noextract
let u_reduce_finish (#n:pos) (original middle after:lseq F32.t n)
  (start:nat) (count:pos{start + count <= n /\ count % 2 == 0})
  : Lemma
      (requires u_reduce_prefix original middle start count (count / 2) /\
        u_reduce_result middle after start (count / 2))
      (ensures u_reduce_result original after start count) =
  let at (model:nat -> GTot real)
    : Lemma (dec_segment_rep original start count model ==>
        S.index after start %~ dec_source_sum_n count model) =
    let next = fun i -> model i +. model (count / 2 + i) in
    dec_sum_halves (count / 2) model;
    assert (dec_segment_rep original start count model ==>
      dec_segment_rep middle start (count / 2) next)
  in C.forall_intro at

inline_for_extraction noextract
fn u_reduce_array (#n:erased pos) (buffer:larray F32.t n) (start:sz)
  (count:szp{dec_reduce_width count /\ start + count <= n})
  (#original:erased (lseq F32.t n))
  preserves gpu
  requires buffer |-> original
  ensures exists* (after:lseq F32.t n).
    buffer |-> after ** pure (u_reduce_result original after start count)


inline_for_extraction noextract
let source_mask_rep_step (#n:pos) (mask:lseq (option F32.t) n)
  (count:nat{count < n}) (model:nat -> GTot real) (guard:prop) (value:F32.t)
  : Lemma
      (requires source_mask_rep #n mask count model guard /\ (guard ==> value %~ model count))
      (ensures source_mask_rep #n (S.upd mask count (Some value)) (count + 1) model guard) =
  let at (i:nat{i < count + 1})
    : Lemma (
        i < n /\ Some? (S.index (S.upd mask count (Some value)) i) /\
        (forall (v:F32.t). S.index (S.upd mask count (Some value)) i == Some v ==>
          (guard ==> v %~ model i))) =
    if i = count then S.lemma_index_upd1 mask count (Some value)
    else S.lemma_index_upd2 mask count (Some value) i
  in C.forall_intro at

inline_for_extraction noextract
let u_mask_step (#a:Type0) (#n:pos) (mask:lseq (option F32.t) n)
  (count:nat{count < n}) (models:a -> nat -> GTot real) (guards:a -> prop) (value:F32.t)
  : Lemma
      (requires
        (forall (r:a). source_mask_rep #n mask count (models r) (guards r)) /\
        (forall (r:a). guards r ==> value %~ models r count))
      (ensures forall (r:a).
        source_mask_rep #n (S.upd mask count (Some value)) (count + 1) (models r) (guards r)) =
  let at (r:a)
    : Lemma (source_mask_rep #n (S.upd mask count (Some value)) (count + 1)
        (models r) (guards r)) =
    source_mask_rep_step #n mask count (models r) (guards r) value
  in C.forall_intro at

inline_for_extraction noextract
let u_rms_lane_partial (width:nat{dec_rms_width width}) (lane steps:nat)
  (r:dec_spec_vector width) : GTot real =
  dec_source_sum_n steps (fun j -> dec_rms_square width r (lane + j * dec_rms_lanes width))

inline_for_extraction noextract
let dec_source_sum_step (n:pos) (f:nat -> GTot real)
  : Lemma (dec_source_sum_n n f == dec_source_sum_n (n - 1) f +. f (n - 1)) = ()

inline_for_extraction noextract
let dec_conditional_fma
  (x y z result:F32.t) (xr yr zr:real) (guard:prop)
  : Lemma
      (requires
        (guard ==> x %~ xr /\ y %~ yr /\ z %~ zr) /\
        (forall (a b c:real).
          x %~ a /\ y %~ b /\ z %~ c ==> result %~ (a *. b +. c)))
      (ensures guard ==> result %~ (xr *. yr +. zr)) = ()

inline_for_extraction noextract
let u_rms_acc_step (#n:pos) (x:lseq BF16.t n) (start:nat)
  (width:nat{dec_rms_width width /\ start + width <= n})
  (lane:nat{lane < dec_rms_lanes width})
  (step:nat{step < width / dec_rms_lanes width}) (previous next:F32.t)
  : Lemma
      (requires
        (forall (r:dec_spec_vector width).
          dec_bf16_segment_rep x start width r ==>
            previous %~ u_rms_lane_partial width lane step r) /\
        (let v = Casts.fcast #BF16.t #F32.t
            (S.index x (start + lane + step * dec_rms_lanes width)) in
         forall (a b c:real).
           v %~ a /\ v %~ b /\ previous %~ c ==> next %~ (a *. b +. c)))
      (ensures forall (r:dec_spec_vector width).
        dec_bf16_segment_rep x start width r ==>
          next %~ u_rms_lane_partial width lane (step + 1) r) =
  let index = lane + step * dec_rms_lanes width in
  let value = S.index x (start + index) in
  let v = Casts.fcast #BF16.t #F32.t value in
  dec_bf16_to_f32_all value;
  let at (r:dec_spec_vector width)
    : Lemma (dec_bf16_segment_rep x start width r ==>
        next %~ u_rms_lane_partial width lane (step + 1) r) =
    dec_source_sum_step (step + 1)
      (fun j -> dec_rms_square width r (lane + j * dec_rms_lanes width));
    assert (dec_bf16_segment_rep x start width r ==> v %~ r index);
    dec_conditional_fma v v previous next (r index) (r index)
      (u_rms_lane_partial width lane step r) (dec_bf16_segment_rep x start width r)
  in C.forall_intro at

inline_for_extraction noextract
let dec_rms_warp_terms (n:nat{dec_rms_width n}) (r:dec_spec_vector n)
  (warp lane:nat) : GTot real =
  dec_rms_lane_model n r (warp * 32 + lane)

inline_for_extraction noextract
let rec dec_source_sum_ext (n:nat) (f g:nat -> GTot real)
  : Lemma
      (requires forall (i:dec_spec_index n). f i == g i)
      (ensures dec_source_sum_n n f == dec_source_sum_n n g)
      (decreases n) =
  if n > 0 then dec_source_sum_ext (n - 1) f g

inline_for_extraction noextract
let dec_rms_warp_at (n:nat{dec_rms_width n}) (r:dec_spec_vector n)
  (warp:nat{warp < dec_rms_lanes n / 32})
  : Lemma
      (dec_rms_warp_model n r warp ==
        dec_source_sum_n 32 (fun lane -> dec_rms_lane_model n r (warp * 32 + lane))) = ()

inline_for_extraction noextract
let dec_rms_warp_related (n:nat{dec_rms_width n}) (r:dec_spec_vector n)
  (warp:nat{warp < dec_rms_lanes n / 32}) (value:F32.t) (guard:prop)
  : Lemma
      (requires guard ==> value %~
        dec_source_sum_n 32 (dec_rms_warp_terms n r warp))
      (ensures guard ==> value %~ dec_rms_warp_model n r warp) =
  dec_rms_warp_at n r warp;
  dec_source_sum_ext 32 (dec_rms_warp_terms n r warp)
    (fun lane -> dec_rms_lane_model n r (warp * 32 + lane))

inline_for_extraction noextract
let u_rms_warp_finish (#n:pos) (x:lseq BF16.t n)
  (start:nat) (width:nat{dec_rms_width width /\ start + width <= n})
  (warp:nat{warp < dec_rms_lanes width / 32})
  (before after:lseq F32.t (dec_rms_lanes width))
  : Lemma
      (requires
        (forall (r:dec_spec_vector width).
          dec_bf16_segment_rep x start width r ==>
            dec_segment_rep before (warp * 32) 32 (dec_rms_warp_terms width r warp)) /\
        u_reduce_result before after (warp * 32) 32)
      (ensures forall (r:dec_spec_vector width).
        dec_bf16_segment_rep x start width r ==>
          S.index after (warp * 32) %~ dec_rms_warp_model width r warp) =
  let at (r:dec_spec_vector width)
    : Lemma (dec_bf16_segment_rep x start width r ==>
        S.index after (warp * 32) %~ dec_rms_warp_model width r warp) =
    dec_rms_warp_related width r warp (S.index after (warp * 32))
      (dec_bf16_segment_rep x start width r)
  in C.forall_intro at

inline_for_extraction noextract
let rec dec_sum_extensional (n:nat) (left right:dec_spec_vector n)
  : Lemma
      (requires forall (i:dec_spec_index n). left i == right i)
      (ensures dec_spec_sum n left == dec_spec_sum n right)
      (decreases n)
  =
  if n > 0 then (
    dec_sum_extensional (n - 1) left right;
    ObSum.sum_pop_right 0 n left;
    ObSum.sum_pop_right 0 n right
  )

inline_for_extraction noextract
let rec dec_source_sum_shared (n:nat) (f:nat -> GTot real)
  : Lemma (dec_source_sum_n n f == dec_spec_sum n f)
      (decreases n) =
  if n > 0 then (
    dec_source_sum_shared (n - 1) f;
    ObSum.sum_pop_right 0 n f
  )

inline_for_extraction noextract
let rec dec_source_sum_zero (n:nat)
  : Lemma (dec_source_sum_n n (fun _ -> 0.0R) == 0.0R) (decreases n) =
  if n > 0 then dec_source_sum_zero (n - 1)

inline_for_extraction noextract
let rec dec_sum_swap (n m:nat) (f:nat -> nat -> GTot real)
  : Lemma
      (dec_source_sum_n n (fun i -> dec_source_sum_n m (f i)) ==
        dec_source_sum_n m (fun j -> dec_source_sum_n n (fun i -> f i j)))
      (decreases n) =
  let rec add (k:nat) (left right:nat -> GTot real)
    : Lemma
        (dec_source_sum_n k (fun i -> left i +. right i) ==
          dec_source_sum_n k left +. dec_source_sum_n k right)
        (decreases k) =
    if k > 0 then add (k - 1) left right
  in
  if n > 0 then (
    dec_sum_swap (n - 1) m f;
    add m (fun j -> dec_source_sum_n (n - 1) (fun i -> f i j)) (f (n - 1));
    dec_source_sum_ext m
      (fun j -> dec_source_sum_n n (fun i -> f i j))
      (fun j -> dec_source_sum_n (n - 1) (fun i -> f i j) +. f (n - 1) j)
  ) else (
    dec_source_sum_ext m (fun j -> dec_source_sum_n n (fun i -> f i j)) (fun _ -> 0.0R);
    dec_source_sum_zero m
  )

inline_for_extraction noextract
let rec dec_sum_blocks (n m:nat) (f:nat -> GTot real)
  : Lemma
      (dec_source_sum_n n (fun i -> dec_source_sum_n m (fun j -> f (i * m + j))) ==
        dec_source_sum_n (n * m) f)
      (decreases n) =
  if n > 0 then (
    dec_source_sum_step n (fun i -> dec_source_sum_n m (fun j -> f (i * m + j)));
    dec_sum_blocks (n - 1) m f;
    dec_source_sum_append ((n - 1) * m) m f;
    assert ((n - 1) * m + m == n * m)
  )

inline_for_extraction noextract
let dec_rms_reduction_total (n:nat{dec_rms_width n}) (r:dec_spec_vector n)
  : Lemma
      (dec_source_sum_n (dec_rms_outer_width n) (dec_rms_warp_model n r) ==
        dec_spec_sum n (fun i -> r i *. r i)) =
  let lanes = dec_rms_lanes n in
  let warps = lanes / 32 in
  let outer = dec_rms_outer_width n in
  let rec zero_tail (k:nat{warps <= k /\ k <= outer})
    : Lemma
        (dec_source_sum_n k (dec_rms_warp_model n r) ==
          dec_source_sum_n warps (dec_rms_warp_model n r))
        (decreases k) =
    if k > warps then zero_tail (k - 1)
  in
  zero_tail outer;
  dec_source_sum_ext warps (dec_rms_warp_model n r)
    (fun warp -> dec_source_sum_n 32 (fun lane -> dec_rms_lane_model n r (warp * 32 + lane)));
  dec_sum_blocks warps 32 (dec_rms_lane_model n r);
  assert (warps * 32 == lanes /\ (n / lanes) * lanes == n);
  assert (dec_source_sum_n outer (dec_rms_warp_model n r) ==
    dec_source_sum_n lanes (dec_rms_lane_model n r));
  let unfold_lane (lane:nat)
    : Lemma (dec_rms_lane_model n r lane ==
        dec_source_sum_n (n / lanes) (fun step -> dec_rms_square n r (lane + step * lanes))) =
    dec_source_sum_ext (n / lanes)
      (fun step -> dec_rms_square n r (lane + step * dec_rms_lanes n))
      (fun step -> dec_rms_square n r (lane + step * lanes))
  in FStar.Classical.forall_intro unfold_lane;
  dec_source_sum_ext lanes (dec_rms_lane_model n r)
    (fun lane -> dec_source_sum_n (n / lanes)
      (fun step -> dec_rms_square n r (lane + step * lanes)));
  dec_sum_swap lanes (n / lanes)
    (fun lane step -> dec_rms_square n r (lane + step * lanes));
  let commute (step:nat)
    : Lemma
        (dec_source_sum_n lanes (fun lane -> dec_rms_square n r (lane + step * lanes)) ==
          dec_source_sum_n lanes (fun lane -> dec_rms_square n r (step * lanes + lane))) =
    dec_source_sum_ext lanes
      (fun lane -> dec_rms_square n r (lane + step * lanes))
      (fun lane -> dec_rms_square n r (step * lanes + lane))
  in FStar.Classical.forall_intro commute;
  dec_source_sum_ext (n / lanes)
    (fun step -> dec_source_sum_n lanes (fun lane -> dec_rms_square n r (lane + step * lanes)))
    (fun step -> dec_source_sum_n lanes (fun lane -> dec_rms_square n r (step * lanes + lane)));
  dec_sum_blocks (n / lanes) lanes (dec_rms_square n r);
  dec_source_sum_shared n (dec_rms_square n r);
  dec_sum_extensional n (dec_rms_square n r) (fun i -> r i *. r i)

inline_for_extraction noextract
let u_rms_total_finish (#n:pos) (x:lseq BF16.t n)
  (start:nat) (width:nat{dec_rms_width width /\ start + width <= n})
  (before after:lseq F32.t (dec_rms_outer_width width))
  : Lemma
      (requires
        (forall (r:dec_spec_vector width).
          dec_bf16_segment_rep x start width r ==>
            dec_segment_rep before 0 (dec_rms_outer_width width) (dec_rms_warp_model width r)) /\
        u_reduce_result before after 0 (dec_rms_outer_width width))
      (ensures forall (r:dec_spec_vector width).
        dec_bf16_segment_rep x start width r ==>
          S.index after 0 %~ dec_spec_sum width (fun i -> r i *. r i)) =
  let at (r:dec_spec_vector width)
    : Lemma (dec_bf16_segment_rep x start width r ==>
        S.index after 0 %~ dec_spec_sum width (fun i -> r i *. r i)) =
    dec_rms_reduction_total width r
  in C.forall_intro at

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

inline_for_extraction noextract
let dec_rms_point (n:pos) (r gamma:dec_spec_vector n) (i:dec_spec_index n)
  : Lemma
      (requires dec_spec_sum n (fun j -> r j *. r j) >=. 0.0R)
      (ensures dec_spec_rms_norm gamma r i ==
        gamma i *. (r i *. ObRoot.rsqrt (dec_rms_radicand n r))) =
  dec_spec_sum_nonnegative n (fun j -> r j *. r j);
  assert (dec_spec_rms_norm gamma r i ==
    gamma i *. (r i *. ObRoot.rsqrt
      (dec_spec_sum n (fun j -> r j *. r j) /. of_int n +. 0.000001R))) by
    (FStar.Tactics.V2.norm [delta_only [`%dec_spec_rms_norm]; iota; zeta];
     FStar.Tactics.V2.trivial ());
  assert (dec_rms_radicand n r ==
    dec_spec_sum n (fun j -> r j *. r j) /. of_int n +. 0.000001R) by
    (FStar.Tactics.V2.norm [delta_only [`%dec_rms_radicand]; iota; zeta];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let dec_real_product_reorder (a b c:real)
  : Lemma (a *. (b *. c) == c *. (a *. b)) = ()

inline_for_extraction noextract
let dec_rms_cast_refines (n:pos) (r gamma:dec_spec_vector n) (i:dec_spec_index n)
  (product:F32.t) (guard:prop)
  : Lemma
      (requires guard ==> product %~
        (r i *. (ObRoot.rsqrt (dec_rms_radicand n r) *. gamma i)))
      (ensures guard ==> Casts.fcast #F32.t #bf16 product %~ dec_spec_rms_norm gamma r i) =
  dec_spec_sum_nonnegative n (fun j -> r j *. r j);
  dec_rms_point n r gamma i;
  dec_real_product_reorder (r i) (ObRoot.rsqrt (dec_rms_radicand n r)) (gamma i);
  assert (guard ==> product %~ dec_spec_rms_norm gamma r i);
  dec_f32_to_bf16_all product

inline_for_extraction noextract
let u_rms_cast_all (width:pos) (i:Shared.index width) (product:F32.t)
  : Lemma (forall (r gamma:dec_spec_vector width).
    product %~ (r i *. (ObRoot.rsqrt (dec_rms_radicand width r) *. gamma i)) ==>
      Casts.fcast #F32.t #BF16.t product %~ dec_spec_rms_norm gamma r i) =
  let at (r gamma:dec_spec_vector width)
    : Lemma (
        product %~ (r i *. (ObRoot.rsqrt (dec_rms_radicand width r) *. gamma i)) ==>
          Casts.fcast #F32.t #BF16.t product %~ dec_spec_rms_norm gamma r i) =
    dec_rms_cast_refines width r gamma i product
      (product %~ (r i *. (ObRoot.rsqrt (dec_rms_radicand width r) *. gamma i)))
  in C.forall_intro_2 at

inline_for_extraction noextract
let source_bf16_region_step (#n:pos)
  (original before:lseq (option BF16.t) n)
  (start:nat) (width:pos{start + width <= n}) (done:nat{done < width})
  (model:dec_spec_vector width) (guard:prop) (value:BF16.t)
  : Lemma
      (requires source_bf16_region #n original before start width done model guard /\
        (guard ==> value %~ model done))
      (ensures source_bf16_region #n original
        (S.upd before (start + done) (Some value)) start width (done + 1) model guard) =
  let after = S.upd before (start + done) (Some value) in
  let frame (i:Shared.index n)
    : Lemma ((i < start \/ start + done + 1 <= i) ==>
        S.index after i == S.index original i) =
    if i < start || start + done + 1 <= i then
      S.lemma_index_upd2 before (start + done) (Some value) i
  in C.forall_intro frame;
  let at (j:Shared.index width{j < done + 1})
    : Lemma (
        Some? (S.index after (start + j)) /\
        (forall (v:BF16.t). S.index after (start + j) == Some v ==>
          (guard ==> v %~ model j))) =
    if j = done then S.lemma_index_upd1 before (start + done) (Some value)
    else S.lemma_index_upd2 before (start + done) (Some value) (start + j)
  in C.forall_intro at

inline_for_extraction noextract
let u_rms_region_step (#n #nw #ny:pos)
  (x:lseq BF16.t n) (w:lseq BF16.t nw)
  (original before:lseq (option BF16.t) ny)
  (start weight_start output_start:nat)
  (width:pos{start + width <= n /\ weight_start + width <= nw /\ output_start + width <= ny})
  (done:nat{done < width}) (value:BF16.t)
  : Lemma
      (requires
        (forall (r gamma:dec_spec_vector width).
          source_bf16_region #ny original before output_start width done (dec_spec_rms_norm gamma r)
            (u_rms_guard x w start weight_start width r gamma)) /\
        (forall (r gamma:dec_spec_vector width).
          u_rms_guard x w start weight_start width r gamma ==>
            value %~ dec_spec_rms_norm gamma r done))
      (ensures forall (r gamma:dec_spec_vector width).
        source_bf16_region #ny original (S.upd before (output_start + done) (Some value))
          output_start width (done + 1) (dec_spec_rms_norm gamma r)
          (u_rms_guard x w start weight_start width r gamma)) =
  let at (r gamma:dec_spec_vector width)
    : Lemma (source_bf16_region #ny original (S.upd before (output_start + done) (Some value))
        output_start width (done + 1) (dec_spec_rms_norm gamma r)
        (u_rms_guard x w start weight_start width r gamma)) =
    source_bf16_region_step #ny original before output_start width done
      (dec_spec_rms_norm gamma r) (u_rms_guard x w start weight_start width r gamma) value
  in C.forall_intro_2 at

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

inline_for_extraction noextract
let u_rope_from_normalized (x:lseq BF16.t 32768) (w:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (nv:lseq BF16.t 128)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  (lane:Shared.index 128) (value:BF16.t)
  : Lemma
      (requires
        (forall (r gamma:dec_spec_vector 128).
          u_rms_guard x w (row * 4096 + group * 512 + head * 128)
            (u_rope_weight_start head) 128 r gamma ==>
              dec_bf16_segment_rep nv 0 128 (dec_spec_rms_norm gamma r)) /\
        (forall (vr pr cr sr:real).
          S.index nv lane %~ vr /\ S.index nv ((lane + 64) % 128) %~ pr /\
          S.index rt (row * 128 + lane) %~ cr /\ S.index rt (524288 + row * 128 + lane) %~ sr ==>
            value %~ (vr *. cr +. (if lane < 64 then 0.0R -. pr else pr) *. sr)))
      (ensures u_rope_cell x w rt row group head lane value) =
  reveal_opaque (`%u_rope_cell) (u_rope_cell x w rt row group head lane value);
  assert ((lane + 64) % 128 == (if lane < 64 then lane + 64 else lane - 64))

inline_for_extraction noextract
let source_bf16_full_write_related (#n:pos)
  (before mask:lseq (option BF16.t) n) (after:lseq BF16.t n)
  (model:dec_spec_vector n) (guard:prop)
  : Lemma
      (requires source_bf16_region #n before mask 0 n n model guard /\
        (forall (i:Shared.index n). S.index mask i == Some (S.index after i)))
      (ensures guard ==> dec_bf16_segment_rep #n after 0 n model) =
  let at (i:Shared.index n) : Lemma (guard ==> S.index after i %~ model i) =
    assert (S.index mask (0 + i) == Some (S.index after i))
  in C.forall_intro at

inline_for_extraction noextract
let u_rms_full_transfer (#n #nw:pos)
  (x:lseq BF16.t n) (w:lseq BF16.t nw)
  (start weight_start:nat) (width:pos{start + width <= n /\ weight_start + width <= nw})
  (before mask:lseq (option BF16.t) width) (after:lseq BF16.t width)
  : Lemma
      (requires
        (forall (r gamma:dec_spec_vector width).
          source_bf16_region before mask 0 width width (dec_spec_rms_norm gamma r)
            (u_rms_guard x w start weight_start width r gamma)) /\
        (forall (i:Shared.index width). S.index mask i == Some (S.index after i)))
      (ensures forall (r gamma:dec_spec_vector width).
        u_rms_guard x w start weight_start width r gamma ==>
          dec_bf16_segment_rep after 0 width (dec_spec_rms_norm gamma r)) =
  let at (r gamma:dec_spec_vector width)
    : Lemma (u_rms_guard x w start weight_start width r gamma ==>
        dec_bf16_segment_rep after 0 width (dec_spec_rms_norm gamma r)) =
    source_bf16_full_write_related before mask after (dec_spec_rms_norm gamma r)
      (u_rms_guard x w start weight_start width r gamma)
  in C.forall_intro_2 at

inline_for_extraction noextract
let u_rope_write (x:lseq BF16.t 32768) (w:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (old before:lseq BF16.t 32768)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  (lane:Shared.index 128) (value:BF16.t)
  : Lemma
      (requires
        (let base = row * 4096 + group * 512 + head * 128 in
         dec_segment_frame old before base lane /\
         (forall (j:Shared.index 128{j < lane}).
           u_rope_cell x w rt row group head j (S.index before (base + j)))) /\
        u_rope_cell x w rt row group head lane value)
      (ensures (
        let base = row * 4096 + group * 512 + head * 128 in
        let after = S.upd before (base + lane) value in
        dec_segment_frame old after base (lane + 1) /\
        (forall (j:Shared.index 128{j < lane + 1}).
          u_rope_cell x w rt row group head j (S.index after (base + j))))) =
  let base = row * 4096 + group * 512 + head * 128 in
  let frame (i:Shared.index 32768)
    : Lemma ((i < base \/ base + lane + 1 <= i) ==>
        S.index (S.upd before (base + lane) value) i == S.index old i) =
    if i < base || base + lane + 1 <= i then
      S.lemma_index_upd2 before (base + lane) value i
  in C.forall_intro frame;
  let at (j:Shared.index 128{j < lane + 1})
    : Lemma (u_rope_cell x w rt row group head j
        (S.index (S.upd before (base + lane) value) (base + j))) =
    if lane = j then S.lemma_index_upd1 before (base + lane) value
    else S.lemma_index_upd2 before (base + lane) value (base + j)
  in C.forall_intro at

inline_for_extraction noextract
let u_rope_head_advance (x:lseq BF16.t 32768) (w:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (before after:lseq BF16.t 32768)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3)
  : Lemma
      (requires u_rope_prefix x w rt before (row * 24 + group * 3 + head) /\
        dec_segment_frame before after (row * 4096 + group * 512 + head * 128) 128 /\
        (forall (lane:Shared.index 128).
          u_rope_cell x w rt row group head lane
            (S.index after (row * 4096 + group * 512 + head * 128 + lane))))
      (ensures u_rope_prefix x w rt after (row * 24 + group * 3 + head + 1)) =
  let at (r:Shared.index 8) (c:Shared.index 4096)
    : Lemma (
      if r < 7 && c % 512 < 384 && dec_qk_head_number r c < row * 24 + group * 3 + head + 1 then
        u_rope_cell x w rt r (c / 512) ((c % 512) / 128) (c % 128)
          (S.index after (r * 4096 + c))
      else S.index after (r * 4096 + c) == S.index x (r * 4096 + c)) =
    if r = row && c / 512 = group && (c % 512) / 128 = head then
      assert (r * 4096 + c == row * 4096 + group * 512 + head * 128 + c % 128)
    else
      assert (r * 4096 + c < row * 4096 + group * 512 + head * 128 \/
        row * 4096 + group * 512 + head * 128 + 128 <= r * 4096 + c)
  in C.forall_intro_2 at

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

inline_for_extraction noextract
fn u_norm_rope_body (input output:larray BF16.t 32768)
  (weights:larray BF16.t dec_weight_elements) (table:larray BF16.t 1048576)
  (#x #old:erased (lseq BF16.t 32768)) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rt:erased (lseq BF16.t 1048576)) (#fx #fw #ft:perm)
  preserves gpu ** input |-> Frac fx x ** weights |-> Frac fw w ** table |-> Frac ft rt
  requires output |-> old
  ensures exists* (ys:lseq BF16.t 32768).
    output |-> ys ** pure (u_rope_prefix x w rt ys 168)

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

inline_for_extraction noextract
let u_rope_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (layer_weights:larray BF16.t dec_weight_elements{
    layer_weights == u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased opt_block_values)
  : k:KS.kernel_desc
      (u_stage_resources weights_address prompt_address a weights prompt v 3)
      (u_pending_stage weights_address prompt_address a weights prompt 4){
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)} =
  let raw:KS.kernel_desc_1_1
      (u_stage_resources weights_address prompt_address a weights prompt v 3)
      (u_pending_stage weights_address prompt_address a weights prompt 4) = {
    f = u_rope_stage_body weights_address prompt_address a layer_weights #weights #prompt #v;
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in opt_scalar_descriptor raw

inline_for_extraction noextract
let u_rope_stage
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (layer_weights:larray BF16.t dec_weight_elements{
    layer_weights == u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family opt_block_values
      (fun v -> u_stage_resources weights_address prompt_address a weights prompt v 3)
      (fun _ -> u_pending_stage weights_address prompt_address a weights prompt 4) = {
    pending_threads = 1sz;
    pending_shared = opt_scalar_shared_desc;
    pending_code = u_rope_stage_descriptor weights_address prompt_address a layer_weights #weights #prompt
  }

inline_for_extraction noextract
let dec_silu_source_index (col:szlt 3072)
  : Tot (result:sz{SZ.v result == SZ.v col / 64 * 128 + SZ.v col % 64}) =
  col +^ col /^ 64sz *^ 64sz

inline_for_extraction noextract
let dec_source_silu_coefficient () : F32.t = ObF.of_literal "1.44269504088896340736"

inline_for_extraction noextract
let dec_source_log_two_nonzero ()
  : Lemma (ObExp.log 2.0R =!= 0.0R) =
  ObExp.exp_log 2.0R;
  ObExp.exp_base ()

inline_for_extraction noextract
let dec_intended_silu_exp2_scale () : real =
  dec_source_log_two_nonzero ();
  1.0R /. ObExp.log 2.0R

inline_for_extraction noextract
let dec_source_silu_coefficient_approx ()
  : Lemma (dec_source_silu_coefficient () %~ dec_intended_silu_exp2_scale ()) =
  ObExp.log_two_positive ();
  ObExp.log_two_decimal_bounds ();
  assert (14426950408889632.0R /. 10000000000000000.0R <. dec_intended_silu_exp2_scale ());
  assert (dec_intended_silu_exp2_scale () <. 14426950408889635.0R /. 10000000000000000.0R);
  assert_norm (Kuiper.Float32.Literals.common_normal_cell "1.44269504088896340736"
    14426950408889632 10000000000000000 14426950408889635 10000000000000000
    12102203 (-23));
  F32.literal_interval_approx "1.44269504088896340736"
    14426950408889632 10000000000000000 14426950408889635 10000000000000000
    12102203 (-23) (dec_intended_silu_exp2_scale ())

inline_for_extraction noextract
let dec_bf16_multiply_all (left right:bf16)
  : Lemma (forall (lr rr:real). left %~ lr /\ right %~ rr ==> ObF.mul left right %~ (lr *. rr)) =
  let at (lr rr:real)
    : Lemma (requires left %~ lr /\ right %~ rr)
        (ensures ObF.mul left right %~ (lr *. rr)) = a_mul left right lr rr
  in FStar.Classical.forall_intro_2 (FStar.Classical.move_requires_2 at)

inline_for_extraction noextract
let dec_silu_change_of_base (x:real)
  : Lemma (ObPow.exp2 ((0.0R -. x) *. dec_intended_silu_exp2_scale ()) ==
      ObExp.exp (0.0R -. x)) =
  ObExp.log_two_positive ();
  ObExp.exp2_as_exp ((0.0R -. x) *. dec_intended_silu_exp2_scale ())

inline_for_extraction noextract
let u_silu_change_all ()
  : Lemma (forall (r:real).
      ObPow.exp2 ((0.0R -. r) *. dec_intended_silu_exp2_scale ()) ==
        ObExp.exp (0.0R -. r)) =
  C.forall_intro dec_silu_change_of_base

inline_for_extraction noextract
let dec_real_reciprocal_product (x:real) (d:real{d >. 0.0R})
  : Lemma (x *. (1.0R /. d) == x /. d) = ()

inline_for_extraction noextract
let u_silu_divide_all ()
  : Lemma (forall (r:real).
      r *. (1.0R /. (1.0R +. ObExp.exp (0.0R -. r))) == dec_spec_silu r) =
  let at (r:real) : Lemma (
      r *. (1.0R /. (1.0R +. ObExp.exp (0.0R -. r))) == dec_spec_silu r) =
    dec_real_reciprocal_product r (1.0R +. ObExp.exp (0.0R -. r))
  in C.forall_intro at

inline_for_extraction noextract
fn u_silu_source_value (gate up:BF16.t)
  preserves gpu
  returns result:BF16.t
  ensures pure (forall (gr ur:real).
    gate %~ gr /\ up %~ ur ==> result %~ (dec_spec_silu gr *. ur))

inline_for_extraction noextract
let u_silu_step (x:lseq BF16.t 49152) (before:lseq BF16.t 24576)
  (row:Shared.index 8) (col:Shared.index 3072) (value:BF16.t)
  : Lemma
      (requires u_silu_prefix x before (row * 3072 + col) /\
        (forall (gr ur:real).
          S.index x (row * 6144 + col / 64 * 128 + col % 64) %~ gr /\
          S.index x (row * 6144 + col / 64 * 128 + col % 64 + 64) %~ ur ==>
            value %~ (dec_spec_silu gr *. ur)))
      (ensures u_silu_prefix x (S.upd before (row * 3072 + col) value) (row * 3072 + col + 1)) =
  let at (r:Shared.index 8) (c:Shared.index 3072)
    : Lemma (
        (r * 3072 + c == row * 3072 + col ==> r == row /\ c == col) /\
        (r * 3072 + c <> row * 3072 + col ==>
          S.index (S.upd before (row * 3072 + col) value) (r * 3072 + c) ==
            S.index before (r * 3072 + c))) =
    if r * 3072 + c <> row * 3072 + col then
      S.lemma_index_upd2 before (row * 3072 + col) value (r * 3072 + c)
  in C.forall_intro_2 at

inline_for_extraction noextract
fn u_silu_body (input:larray BF16.t 49152) (output:larray BF16.t 24576)
  (#x:erased (lseq BF16.t 49152)) (#old:erased (lseq BF16.t 24576)) (#fraction:perm)
  preserves gpu ** input |-> Frac fraction x
  requires output |-> old
  ensures exists* (ys:lseq BF16.t 24576). output |-> ys ** pure (u_silu_prefix x ys 24576)

inline_for_extraction noextract
fn u_silu_stage_body
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) ()
  preserves gpu
  requires u_stage_resources weights_address prompt_address a weights prompt v 8
  ensures u_pending_stage weights_address prompt_address a weights prompt 9

inline_for_extraction noextract
let u_silu_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased opt_block_values)
  : k:KS.kernel_desc
      (u_stage_resources weights_address prompt_address a weights prompt v 8)
      (u_pending_stage weights_address prompt_address a weights prompt 9){
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)} =
  let raw:KS.kernel_desc_1_1
      (u_stage_resources weights_address prompt_address a weights prompt v 8)
      (u_pending_stage weights_address prompt_address a weights prompt 9) = {
    f = u_silu_stage_body weights_address prompt_address a #weights #prompt #v;
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in opt_scalar_descriptor raw

inline_for_extraction noextract
let u_silu_stage
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family opt_block_values
      (fun v -> u_stage_resources weights_address prompt_address a weights prompt v 8)
      (fun _ -> u_pending_stage weights_address prompt_address a weights prompt 9) = {
    pending_threads = 1sz;
    pending_shared = opt_scalar_shared_desc;
    pending_code = u_silu_stage_descriptor weights_address prompt_address a #weights #prompt
  }

inline_for_extraction noextract
let opt_linear_shared_desc = [SH.SHArray F32.t 128sz]

inline_for_extraction noextract
let u_weight_window_shape (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (lo:nat) (hi:nat{lo <= hi /\ hi <= opt_weight_count})
  : Lemma (
      PA.length (u_weight_window weights lo hi) == hi - lo /\
      PA.base_of (u_weight_window weights lo hi) == PA.base_of weights /\
      PA.offset_of (u_weight_window weights lo hi) == PA.offset_of weights + lo /\
      KA.is_global_array (u_weight_window weights lo hi)) =
  PA.length_gsub weights lo hi;
  PA.base_of_gsub weights lo hi;
  PA.offset_of_gsub weights lo hi;
  PA.loc_id_of_array_gsub weights lo hi;
  KA.visibility_of_gsub weights lo hi

inline_for_extraction noextract
let u_linear_step (slot:nat{slot < 4}) : nat =
  if slot = 0 then 2 else if slot = 1 then 5 else if slot = 2 then 7 else 9

inline_for_extraction noextract
let u_linear_width (slot:nat{slot < 4}) : pos =
  if slot = 1 then 2048 else if slot = 3 then 3072 else 1024

inline_for_extraction noextract
let u_linear_cols (slot:nat{slot < 4}) : pos =
  if slot = 0 then 4096 else if slot = 2 then 6144 else 1024

inline_for_extraction noextract
let u_linear_layout (slot:nat{slot < 4}) : nat =
  if slot = 0 then 1 else if slot = 2 then 2 else 0

inline_for_extraction noextract
let u_linear_offset (slot:nat{slot < 4}) : nat =
  if slot = 0 then 2048 else if slot = 1 then 4196352 else if slot = 2 then 6293760 else 12585216

inline_for_extraction noextract
let u_linear_add (slot:nat{slot < 4}) : bool = slot = 1 || slot = 3

inline_for_extraction noextract
let u_linear_shape (slot:nat{slot < 4})
  : Lemma (dec_dense_width (u_linear_width slot) /\
    dec_projection_layout (u_linear_cols slot) (u_linear_layout slot) /\
    u_linear_offset slot + u_linear_cols slot * u_linear_width slot <= dec_weight_elements /\
    (u_linear_add slot ==> 8 * u_linear_cols slot <= 8192)) =
  if slot = 0 then () else if slot = 1 then () else if slot = 2 then () else ()

inline_for_extraction noextract
type dec_bf16_array (n:nat) = larray BF16.t n

inline_for_extraction noextract
let opt_linear_read
  (#ni #nr:nat) (input:larray BF16.t ni)
  (weights:dec_bf16_array dec_weight_elements) (residual:larray BF16.t nr)
  (x:lseq BF16.t ni) (w:lseq BF16.t dec_weight_elements) (rs:lseq BF16.t nr)
  (fx fw fr:perm) : slprop =
  input |-> Frac fx x ** weights |-> Frac fw w ** residual |-> Frac fr rs

inline_for_extraction noextract
let opt_linear_pre
  (#ni #nr #no:nat) (input:larray BF16.t ni)
  (weights:dec_bf16_array dec_weight_elements) (residual:larray BF16.t nr)
  (output:larray BF16.t no)
  (x:lseq BF16.t ni) (w:lseq BF16.t dec_weight_elements) (rs:lseq BF16.t nr)
  (fx fw fr:perm) : slprop =
  opt_linear_read input weights residual x w rs fx fw fr ** PA.pts_to_uninit output no

inline_for_extraction noextract
let opt_linear_post
  (#ni #nr #no:nat) (input:larray BF16.t ni)
  (weights:dec_bf16_array dec_weight_elements) (residual:larray BF16.t nr)
  (output:larray BF16.t no)
  (x:lseq BF16.t ni) (w:lseq BF16.t dec_weight_elements) (rs:lseq BF16.t nr)
  (ys:lseq BF16.t no) (fx fw fr:perm) : slprop =
  opt_linear_read input weights residual x w rs fx fw fr ** output |-> ys

inline_for_extraction noextract
ghost
fn opt_descriptor_frame_setup (#pre #post:slprop)
  (code:KS.kernel_desc pre post) (extra:slprop) ()
  norewrite
  requires pre ** extra
  ensures (forall+ (bid:natlt code.nblk). code.block_pre bid) ** (code.frame ** extra)

inline_for_extraction noextract
ghost
fn opt_descriptor_frame_teardown (#pre #post:slprop)
  (code:KS.kernel_desc pre post) (extra:slprop) ()
  norewrite
  requires (forall+ (bid:natlt code.nblk). code.block_post bid) ** (code.frame ** extra)
  ensures post ** extra

inline_for_extraction noextract
val opt_descriptor_with_frame (#pre #post:slprop)
  (code:KS.kernel_desc pre post) (extra:slprop)
  : k:KS.kernel_desc (pre ** extra) (post ** extra){
    k.nblk == code.nblk /\ k.nthr == code.nthr /\ k.shmems_desc == code.shmems_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == code.barrier_contract bid sh /\
      k.barrier_count bid == code.barrier_count bid)}

inline_for_extraction noextract
ghost
fn opt_descriptor_refine_setup (#pre #post #pre':slprop)
  (code:KS.kernel_desc pre post)
  (before:ghost fn () requires pre' ensures pre) ()
  norewrite
  requires pre'
  ensures (forall+ (bid:natlt code.nblk). code.block_pre bid) ** code.frame

inline_for_extraction noextract
ghost
fn opt_descriptor_refine_teardown (#pre #post #post':slprop)
  (code:KS.kernel_desc pre post)
  (after:ghost fn () requires post ensures post') ()
  norewrite
  requires (forall+ (bid:natlt code.nblk). code.block_post bid) ** code.frame
  ensures post'

inline_for_extraction noextract
val opt_descriptor_refine (#pre #post #pre' #post':slprop)
  (code:KS.kernel_desc pre post)
  (before:ghost fn () requires pre' ensures pre)
  (after:ghost fn () requires post ensures post')
  : k:KS.kernel_desc pre' post'{
    k.nblk == code.nblk /\ k.nthr == code.nthr /\ k.shmems_desc == code.shmems_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == code.barrier_contract bid sh /\
      k.barrier_count bid == code.barrier_count bid)}

inline_for_extraction noextract
let u_linear_input (a:opt_block_addresses) (slot:nat{slot < 4})
  : larray BF16.t (8 * u_linear_width slot) =
  if slot = 0 then a.p_normalized0 else if slot = 1 then a.p_attention
  else if slot = 2 then a.p_postnorm else a.p_activated

inline_for_extraction noextract
let u_linear_residual (a:opt_block_addresses) (slot:nat{slot < 4}) : larray BF16.t 8192 =
  if slot < 2 then a.p_embedding else a.p_residual

inline_for_extraction noextract
let u_linear_output (a:opt_block_addresses) (slot:nat{slot < 4})
  : larray BF16.t (8 * u_linear_cols slot) =
  if slot = 0 then a.p_qkv else if slot = 1 then a.p_residual
  else if slot = 2 then a.p_gateup else a.p_hidden

inline_for_extraction noextract
let u_linear_x (v:opt_block_values) (slot:nat{slot < 4})
  : lseq BF16.t (8 * u_linear_width slot) =
  if slot = 0 then v.v_normalized0 else if slot = 1 then v.v_attention
  else if slot = 2 then v.v_postnorm else v.v_activated

inline_for_extraction noextract
let u_linear_r (v:opt_block_values) (slot:nat{slot < 4}) : lseq BF16.t 8192 =
  if slot < 2 then v.v_embedding else v.v_residual

inline_for_extraction noextract
let u_linear_result (weights:lseq BF16.t opt_weight_count) (v:opt_block_values)
  (slot:nat{slot < 4}) : GTot (lseq BF16.t (8 * u_linear_cols slot)) =
  u_linear_shape slot;
  dec_projection_values (u_linear_width slot) (u_linear_cols slot) (u_linear_layout slot)
    (u_linear_offset slot) (u_linear_add slot) (u_linear_x v slot)
    (opt_layer0_values weights) (u_linear_r v slot)

inline_for_extraction noextract
let u_linear_frame (a:opt_block_addresses) (v:opt_block_values) (slot:nat{slot < 4}) : slprop =
  a.p_rope |-> v.v_rope ** a.p_rotated |-> v.v_rotated ** a.p_normalized1 |-> v.v_normalized1 **
  (if slot = 0 then
    a.p_attention |-> v.v_attention ** a.p_residual |-> v.v_residual **
    a.p_postnorm |-> v.v_postnorm ** a.p_gateup |-> v.v_gateup **
    a.p_activated |-> v.v_activated ** a.p_hidden |-> v.v_hidden
   else if slot = 1 then
    a.p_normalized0 |-> v.v_normalized0 ** a.p_qkv |-> v.v_qkv **
    a.p_postnorm |-> v.v_postnorm ** a.p_gateup |-> v.v_gateup **
    a.p_activated |-> v.v_activated ** a.p_hidden |-> v.v_hidden
   else if slot = 2 then
    a.p_embedding |-> v.v_embedding ** a.p_normalized0 |-> v.v_normalized0 **
    a.p_qkv |-> v.v_qkv ** a.p_attention |-> v.v_attention **
    a.p_activated |-> v.v_activated ** a.p_hidden |-> v.v_hidden
   else
    a.p_embedding |-> v.v_embedding ** a.p_normalized0 |-> v.v_normalized0 **
    a.p_qkv |-> v.v_qkv ** a.p_attention |-> v.v_attention **
    a.p_postnorm |-> v.v_postnorm ** a.p_gateup |-> v.v_gateup)

inline_for_extraction noextract
let u_linear_stage_frame
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses) (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) (slot:nat{slot < 4}) : slprop =
  PA.pts_to prompt_address prompt **
  u_weight_window_frame weights_address weights opt_layer0_start (opt_layer0_start + dec_weight_elements) 1.0R **
  u_linear_frame a v slot ** pure (u_block_stage weights prompt v (u_linear_step slot))

inline_for_extraction noextract
let u_linear_old (v:opt_block_values) (slot:nat{slot < 4})
  : lseq BF16.t (8 * u_linear_cols slot) =
  if slot = 0 then v.v_qkv else if slot = 1 then v.v_residual
  else if slot = 2 then v.v_gateup else v.v_hidden

inline_for_extraction noextract
ghost
fn u_linear_storage_open (a:opt_block_addresses) (#v:erased opt_block_values) (slot:nat{slot < 4})
  requires opt_block_storage a v
  ensures u_linear_input a slot |-> u_linear_x v slot **
    u_linear_residual a slot |-> u_linear_r v slot **
    u_linear_output a slot |-> u_linear_old v slot ** u_linear_frame a v slot

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

inline_for_extraction noextract
let u_linear_updated (v:opt_block_values) (slot:nat{slot < 4})
  (ys:lseq BF16.t (8 * u_linear_cols slot)) : opt_block_values =
  if slot = 0 then {v with v_qkv = ys}
  else if slot = 1 then {v with v_residual = ys}
  else if slot = 2 then {v with v_gateup = ys}
  else {v with v_hidden = ys}

inline_for_extraction noextract
let u_linear_stage_advance (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) (slot:nat{slot < 4})
  : Lemma
      (requires u_block_stage weights prompt v (u_linear_step slot))
      (ensures u_block_stage weights prompt (u_linear_updated v slot (u_linear_result weights v slot))
        (u_linear_step slot + 1)) =
  if slot = 0 then () else if slot = 1 then () else if slot = 2 then () else ()

inline_for_extraction noextract
ghost
fn u_linear_storage_close (a:opt_block_addresses) (#v:erased opt_block_values)
  (slot:nat{slot < 4}) (#ys:erased (lseq BF16.t (8 * u_linear_cols slot)))
  requires u_linear_input a slot |-> u_linear_x v slot **
    u_linear_residual a slot |-> u_linear_r v slot **
    u_linear_output a slot |-> ys ** u_linear_frame a v slot
  ensures opt_block_storage a (u_linear_updated v slot ys)

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

inline_for_extraction noextract
let opt_linear_tile (sh:SH.c_shmems opt_linear_shared_desc) : larray F32.t 128 =
  fst sh

inline_for_extraction noextract
let opt_linear_final_tile
  (width:nat{dec_dense_width width}) (cols:pos)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (x:lseq BF16.t (8 * width)) (w:lseq BF16.t dec_weight_elements)
  : GTot (lseq F32.t 128) =
  Tensor.to_seq (rm 16 8) (dec_projection_scores width cols layout offset x w (cols - 16))

inline_for_extraction noextract
let opt_linear_kpre
  (#ni #nr #no:nat) (input:larray BF16.t ni)
  (weights:dec_bf16_array dec_weight_elements) (residual:larray BF16.t nr)
  (output:larray BF16.t no) (tile:larray F32.t 128)
  (x:lseq BF16.t ni) (w:lseq BF16.t dec_weight_elements) (rs:lseq BF16.t nr)
  (fx fw fr:perm) (tid:nat) : slprop =
  opt_linear_read input weights residual x w rs (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R) **
  (exists* (tv:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) tv) **
  when_ (tid == 0) (PA.pts_to_uninit output no)

inline_for_extraction noextract
let opt_linear_kpost
  (#ni #nr #no:nat) (input:larray BF16.t ni)
  (weights:dec_bf16_array dec_weight_elements) (residual:larray BF16.t nr)
  (output:larray BF16.t no) (tile:larray F32.t 128)
  (x:lseq BF16.t ni) (w:lseq BF16.t dec_weight_elements) (rs:lseq BF16.t nr)
  (ys:lseq BF16.t no) (tv:lseq F32.t 128)
  (fx fw fr:perm) (tid:nat) : slprop =
  opt_linear_read input weights residual x w rs (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R) **
  tile |-> Frac (1.0R /. 32.0R) tv ** when_ (tid == 0) (output |-> ys)

inline_for_extraction noextract
ghost
fn source_warp_owner_intro (p:slprop)
  requires p
  ensures forall+ (tid:natlt 32). when_ (tid == 0) p

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

inline_for_extraction noextract
ghost
fn source_warp_owner_elim (p:slprop)
  requires forall+ (tid:natlt 32). when_ (tid == 0) p
  ensures p

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

inline_for_extraction noextract
let dec_previous_projection_tile
  (width:nat{dec_dense_width width}) (cols:pos) (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (x:lseq bf16 (8 * width)) (weights:dec_bf16_values dec_weight_elements) (initial:lseq F32.t 128)
  (block:nat{block <= cols /\ block % 16 = 0}) : GTot (lseq F32.t 128) =
  if block = 0 then initial
  else Tensor.to_seq (rm 16 8) (dec_projection_scores width cols layout offset x weights (block - 16))

inline_for_extraction noextract
let dec_dense_mma_step (width:nat{dec_dense_width width})
  (a:lseq bf16 (16 * width)) (b:lseq bf16 (8 * width))
  (step:nat{step < width / 16})
  : Lemma
      (dec_dense_mma_fold width a b (step + 1) ==
        MMA.emma (dec_dense_mma_fold width a b step)
          (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 step)
          (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 step 0)) = ()

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

inline_for_extraction noextract
let dec_slice_outside (start count i:nat) : prop =
  i < start \/ start + count <= i

inline_for_extraction noextract
let dec_slice_remainder (#n:nat) (arr:larray bf16 n) (f:perm)
  (start count:nat) (original:lseq bf16 n) : slprop =
  exists* (contents:lseq (option bf16) n).
    Pulse.Lib.Array.pts_to_mask arr #f contents (dec_slice_outside start count) **
    pure (forall (i:dec_spec_index n). Seq.index contents i == Some (Seq.index original i))

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

inline_for_extraction noextract
let dec_projection_row_source
  (cols:szp) (layout:sz{dec_projection_layout cols layout}) (col:szlt cols)
  : Tot (result:szlt cols{SZ.v result == dec_projection_row cols layout col}) =
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
ghost
fn dec_warp_owner_elim (p:slprop)
  requires forall+ (tid:natlt 32). when_ (tid == 0) p
  ensures p

ghost
fn dec_warp_distribute_proof (#a:Type0) (buffer:array a)
  (#dec_spec_values:erased (seq a)) (#f:perm) ()
  requires forall+ (tid:natlt 32). when_ (tid == 0) (buffer |-> Frac f dec_spec_values)
  ensures forall+ (_:natlt 32). buffer |-> Frac (f /. 32.0R) dec_spec_values

inline_for_extraction noextract
let dec_warp_distribute (#a:Type0) (buffer:array a)
  (#dec_spec_values:erased (seq a)) (#f:perm) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** when_ (reveal tid == 0) (buffer |-> Frac f dec_spec_values))
      (ensures fun _ -> thread_id 32 tid ** buffer |-> Frac (f /. 32.0R) dec_spec_values) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun i -> when_ (i == 0) (buffer |-> Frac f dec_spec_values))
    (fun _ -> buffer |-> Frac (f /. 32.0R) dec_spec_values)
    (dec_warp_distribute_proof buffer #dec_spec_values #f ()) #32 #tid

inline_for_extraction noextract
ghost
fn dec_warp_owner_intro (p:slprop)
  requires p
  ensures forall+ (tid:natlt 32). when_ (tid == 0) p

ghost
fn dec_warp_gather_proof (#a:Type0) (buffer:array a)
  (#dec_spec_values:erased (seq a)) (#f:perm) ()
  requires forall+ (_:natlt 32). buffer |-> Frac (f /. 32.0R) dec_spec_values
  ensures forall+ (tid:natlt 32). when_ (tid == 0) (buffer |-> Frac f dec_spec_values)

inline_for_extraction noextract
let dec_warp_gather (#a:Type0) (buffer:array a)
  (#dec_spec_values:erased (seq a)) (#f:perm) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** buffer |-> Frac (f /. 32.0R) dec_spec_values)
      (ensures fun _ -> thread_id 32 tid ** when_ (reveal tid == 0) (buffer |-> Frac f dec_spec_values)) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun _ -> buffer |-> Frac (f /. 32.0R) dec_spec_values)
    (fun i -> when_ (i == 0) (buffer |-> Frac f dec_spec_values))
    (dec_warp_gather_proof buffer #dec_spec_values #f ()) #32 #tid

inline_for_extraction noextract
let dec_native_projection_prefix (cols:pos) (ys:lseq (option bf16) (8 * cols))
  (expected:lseq bf16 (8 * cols))
  (block steps:nat) : prop =
  forall (row:dec_spec_index 8) (col:dec_spec_index cols).
    (col < block \/ (block <= col /\ col < block + 16 /\ row * 16 + col - block < steps)) ==>
    Seq.index ys (row * cols + col) == Some (Seq.index expected (row * cols + col))

inline_for_extraction noextract
let dec_native_projection_finished (cols:pos) (ys:lseq (option bf16) (8 * cols))
  (expected:lseq bf16 (8 * cols))
  : Lemma
      (requires dec_native_projection_prefix cols ys expected cols 0)
      (ensures forall (i:dec_spec_index (8 * cols)).
        Seq.index ys i == Some (Seq.index expected i)) =
  let at (i:dec_spec_index (8 * cols))
    : Lemma (Seq.index ys i == Some (Seq.index expected i)) =
    assert (i == i / cols * cols + i % cols);
    assert (Seq.index ys (i / cols * cols + i % cols) ==
      Some (Seq.index expected (i / cols * cols + i % cols)))
  in
  FStar.Classical.forall_intro at

inline_for_extraction noextract
let dec_source_tile_sequence_at (tile:Chest.chest2 F32.t 16 8)
  (row:dec_spec_index 16) (col:dec_spec_index 8)
  : Lemma (Seq.index (Tensor.to_seq (rm 16 8) tile) (row * 8 + col) ==
      Chest.acc2 tile row col) = ()

inline_for_extraction noextract
let dec_indexed_matrix_at (#a:Type0) (rows cols:pos)
  (f:dec_spec_index rows -> dec_spec_index cols -> GTot a)
  (row:dec_spec_index rows) (col:dec_spec_index cols)
  : Lemma (Seq.index (dec_indexed_matrix rows cols f) (row * cols + col) == f row col) =
  assert ((row * cols + col) / cols == row /\ (row * cols + col) % cols == col)

inline_for_extraction noextract
let dec_projection_values_at
  (#nr:nat) (width:nat{dec_dense_width width}) (cols:pos)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:dec_bf16_values dec_weight_elements) (residual:lseq bf16 nr)
  (row:dec_spec_index 8) (col:dec_spec_index cols)
  : Lemma
      (Seq.index (dec_projection_values width cols layout offset add_residual x weights residual)
          (row * cols + col) ==
        dec_projection_cell width cols layout offset add_residual x weights residual row col) =
  dec_indexed_matrix_at 8 cols (dec_projection_cell width cols layout offset add_residual x weights residual) row col

inline_for_extraction noextract
let dec_projection_tile_expected
  (#nr:nat) (width:nat{dec_dense_width width}) (cols:pos)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:dec_bf16_values dec_weight_elements) (residual:lseq bf16 nr)
  (block:nat{block < cols /\ block % 16 = 0})
  : Lemma
      (let tc = Tensor.to_seq (rm 16 8) (dec_projection_scores width cols layout offset x weights block) in
       let expected = dec_projection_values width cols layout offset add_residual x weights residual in
       forall (row:dec_spec_index 8) (col:dec_spec_index 16).
         Seq.index expected (row * cols + block + col) ==
           (let rounded = Casts.fcast #F32.t #bf16 (Seq.index tc (col * 8 + row)) in
            if add_residual then ObF.add rounded (Seq.index residual (row * cols + block + col))
            else rounded)) =
  let tc = Tensor.to_seq (rm 16 8) (dec_projection_scores width cols layout offset x weights block) in
  let expected = dec_projection_values width cols layout offset add_residual x weights residual in
  let at (row:dec_spec_index 8) (col:dec_spec_index 16)
    : Lemma
        (Seq.index expected (row * cols + block + col) ==
          (let rounded = Casts.fcast #F32.t #bf16 (Seq.index tc (col * 8 + row)) in
           if add_residual then ObF.add rounded (Seq.index residual (row * cols + block + col))
           else rounded)) =
    assert ((block + col) / 16 * 16 == block /\ (block + col) % 16 == col);
    dec_projection_values_at width cols layout offset add_residual x weights residual row (block + col);
    dec_source_tile_sequence_at (dec_projection_scores width cols layout offset x weights block) col row
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let dec_projection_output_owned
  (cols:pos) (output:larray bf16 (8 * cols)) (expected:lseq bf16 (8 * cols)) (block:nat) : slprop =
  exists* (ys:lseq (option bf16) (8 * cols)).
    Pulse.Lib.Array.pts_to_mask output ys (fun _ -> True) **
    pure (dec_native_projection_prefix cols ys expected block 0)

inline_for_extraction noextract
let dec_native_projection_write (cols:pos) (ys after:lseq (option bf16) (8 * cols))
  (expected:lseq bf16 (8 * cols))
  (block:nat{block + 16 <= cols}) (row:dec_spec_index 8) (col:dec_spec_index 16) (value:bf16)
  : Lemma
      (requires
        dec_native_projection_prefix cols ys expected block (row * 16 + col) /\
        value == Seq.index expected (row * cols + block + col) /\
        after == Seq.upd ys (row * cols + block + col) (Some value))
      (ensures dec_native_projection_prefix cols after expected block (row * 16 + col + 1)) =
  let at (r:dec_spec_index 8) (c:dec_spec_index cols)
    : Lemma
        ((c < block \/ (block <= c /\ c < block + 16 /\ r * 16 + c - block < row * 16 + col + 1)) ==>
          Seq.index after (r * cols + c) == Some (Seq.index expected (r * cols + c))) =
    if r = row && c = block + col then assert (Seq.index after (r * cols + c) == Some value)
    else assert (Seq.index after (r * cols + c) == Seq.index ys (r * cols + c))
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let dec_native_projection_advance (cols:pos) (ys:lseq (option bf16) (8 * cols))
  (expected:lseq bf16 (8 * cols))
  (block:nat{block + 16 <= cols})
  : Lemma
      (requires dec_native_projection_prefix cols ys expected block 128)
      (ensures dec_native_projection_prefix cols ys expected (block + 16) 0) = ()

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

inline_for_extraction noextract
let opt_linear_old_tile_irrelevant
  (width:nat{dec_dense_width width}) (cols:pos)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (x:lseq BF16.t (8 * width)) (w:lseq BF16.t dec_weight_elements)
  (old:lseq F32.t 128)
  : Lemma (dec_previous_projection_tile width cols layout offset x w old cols ==
      opt_linear_final_tile width cols layout offset x w) = ()

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

inline_for_extraction noextract
let opt_linear_tile_block_visible (sh:SH.c_shmems opt_linear_shared_desc)
  : Lemma
      (requires SH.c_shmems_inv sh)
      (ensures SH.is_block_array (opt_linear_tile sh)) = ()

inline_for_extraction noextract
let opt_linear_kpre_send
  (#ni #nr #no:erased nat)
  (input:larray BF16.t ni{KA.is_global_array input})
  (weights:dec_bf16_array dec_weight_elements{KA.is_global_array weights})
  (residual:larray BF16.t nr{KA.is_global_array residual})
  (output:larray BF16.t no{KA.is_global_array output})
  (#x:erased (lseq BF16.t ni)) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rs:erased (lseq BF16.t nr)) (#fx #fw #fr:perm)
  (sh:SH.c_shmems opt_linear_shared_desc) (_:squash (SH.c_shmems_inv sh)) (tid:nat)
  : is_send_across block_of
      (opt_linear_kpre input weights residual output (opt_linear_tile sh) x w rs fx fw fr tid) =
  opt_linear_tile_block_visible sh;
  let tile = opt_linear_tile sh in
  let send_value (v:lseq F32.t 128)
    : is_send_across block_of (tile |-> Frac (1.0R /. 32.0R) v) =
    SH.is_send_across_block_array tile #(1.0R /. 32.0R) #v in
  let send_tile : is_send_across block_of
      (exists* (v:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) v) =
    is_send_across_exists _ #send_value in
  let send_read : is_send_across block_of
      (opt_linear_read input weights residual x w rs
        (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R)) = solve in
  let send_output : is_send_across block_of (when_ (tid == 0) (PA.pts_to_uninit output no)) = solve in
  is_send_across_star _ _ #send_read
    #(is_send_across_star _ _ #send_tile #send_output)

inline_for_extraction noextract
let opt_linear_kpost_send
  (#ni #nr #no:erased nat)
  (input:larray BF16.t ni{KA.is_global_array input})
  (weights:dec_bf16_array dec_weight_elements{KA.is_global_array weights})
  (residual:larray BF16.t nr{KA.is_global_array residual})
  (output:larray BF16.t no{KA.is_global_array output})
  (#x:erased (lseq BF16.t ni)) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rs:erased (lseq BF16.t nr)) (#ys:erased (lseq BF16.t no))
  (#tv:erased (lseq F32.t 128)) (#fx #fw #fr:perm)
  (sh:SH.c_shmems opt_linear_shared_desc) (_:squash (SH.c_shmems_inv sh)) (tid:nat)
  : is_send_across block_of
      (opt_linear_kpost input weights residual output (opt_linear_tile sh) x w rs ys tv fx fw fr tid) =
  opt_linear_tile_block_visible sh;
  let tile = opt_linear_tile sh in
  let send_tile : is_send_across block_of (tile |-> Frac (1.0R /. 32.0R) tv) =
    SH.is_send_across_block_array tile #(1.0R /. 32.0R) #tv in
  let send_read : is_send_across block_of
      (opt_linear_read input weights residual x w rs
        (fx /. 32.0R) (fw /. 32.0R) (fr /. 32.0R)) = solve in
  let send_output : is_send_across block_of (when_ (tid == 0) (output |-> ys)) = solve in
  is_send_across_star _ _ #send_read
    #(is_send_across_star _ _ #send_tile #send_output)

inline_for_extraction noextract
let opt_linear_shared_descriptor
  (#nr:erased nat) (width:szp{dec_dense_width width}) (cols:szp)
  (layout:sz{dec_projection_layout cols layout})
  (offset:sz{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (input:larray BF16.t (8 * width){KA.is_global_array input})
  (weights:dec_bf16_array dec_weight_elements{KA.is_global_array weights})
  (residual:larray BF16.t nr{KA.is_global_array residual})
  (output:larray BF16.t (8 * cols){KA.is_global_array output})
  (#x:erased (lseq BF16.t (8 * width))) (#w:erased (lseq BF16.t dec_weight_elements))
  (#rs:erased (lseq BF16.t nr)) (#fx #fw #fr:perm)
  : KS.kernel_desc
      (opt_linear_pre input weights residual output x w rs fx fw fr)
      (opt_linear_post input weights residual output x w rs
        (dec_projection_values width cols layout offset add_residual x w rs) fx fw fr) =
  let pre = opt_linear_pre input weights residual output x w rs fx fw fr in
  let ys = hide (dec_projection_values width cols layout offset add_residual x w rs) in
  let tv = hide (opt_linear_final_tile width cols layout offset x w) in
  let post = opt_linear_post input weights residual output x w rs ys fx fw fr in
  {
    nblk = 1sz;
    nthr = 32sz;
    shmems_desc = opt_linear_shared_desc;
    kpre = (fun sh _ tid -> opt_linear_kpre input weights residual output
      (opt_linear_tile sh) x w rs fx fw fr tid);
    kpost = (fun sh _ tid -> opt_linear_kpost input weights residual output
      (opt_linear_tile sh) x w rs ys tv fx fw fr tid);
    barrier_contract = (fun _ _ -> Barrier.empty_contract 32);
    barrier_count = (fun _ -> 0);
    barrier_ok = (fun _ _ -> Barrier.empty_barrier_transform 32);
    f = opt_linear_shared_thread width cols layout offset add_residual
      input weights residual output #x #w #rs #fx #fw #fr;
    frame = emp;
    block_pre = (fun _ -> pre);
    block_post = (fun _ -> post);
    setup = opt_single_block_setup pre;
    teardown = opt_single_block_teardown post;
    block_frame = (fun _ _ -> emp);
    block_setup = opt_linear_block_setup input weights residual output #x #w #rs #fx #fw #fr;
    block_teardown = opt_linear_block_teardown input weights residual output
      #x #w #rs #ys #tv #fx #fw #fr;
    block_pre_sendable = (fun _ ->
      (solve <: is_send_across gpu_of (opt_linear_pre input weights residual output x w rs fx fw fr)));
    block_post_sendable = (fun _ ->
      (solve <: is_send_across gpu_of (opt_linear_post input weights residual output x w rs ys fx fw fr)));
    kpre_sendable = (fun sh pf _ tid ->
      opt_linear_kpre_send input weights residual output #x #w #rs #fx #fw #fr sh pf tid);
    kpost_sendable = (fun sh pf _ tid ->
      opt_linear_kpost_send input weights residual output #x #w #rs #ys #tv #fx #fw #fr sh pf tid)
  }

inline_for_extraction noextract
val u_linear_source_descriptor (width:szp{dec_dense_width width}) (cols:szp)
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
      k.barrier_contract bid sh == Barrier.empty_contract 32sz /\ k.barrier_count bid == 0)}

inline_for_extraction noextract
let u_linear_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (layer_weights:larray BF16.t dec_weight_elements{
    layer_weights == u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements)})
  (slot:nat{slot < 4})
  (width:szp{SZ.v width == u_linear_width slot}) (cols:szp{SZ.v cols == u_linear_cols slot})
  (layout:sz{SZ.v layout == u_linear_layout slot}) (offset:sz{SZ.v offset == u_linear_offset slot})
  (add_residual:bool{add_residual == u_linear_add slot})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased opt_block_values)
  : k:KS.kernel_desc
      (u_stage_resources weights_address prompt_address a weights prompt v (u_linear_step slot))
      (u_pending_stage weights_address prompt_address a weights prompt (u_linear_step slot + 1)){
    SZ.v k.nblk == 1 /\ k.nthr == 32sz /\ k.shmems_desc == opt_linear_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 32sz /\ k.barrier_count bid == 0)} =
  u_linear_shape slot;
  u_weight_window_shape weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements);
  let pre = opt_linear_pre (u_linear_input a slot) layer_weights
    (u_linear_residual a slot) (u_linear_output a slot) (u_linear_x v slot)
    (opt_layer0_values weights) (u_linear_r v slot) 1.0R 1.0R 1.0R in
  let post = opt_linear_post (u_linear_input a slot) layer_weights
    (u_linear_residual a slot) (u_linear_output a slot) (u_linear_x v slot)
    (opt_layer0_values weights) (u_linear_r v slot) (u_linear_result weights v slot) 1.0R 1.0R 1.0R in
  let extra = u_linear_stage_frame weights_address prompt_address a weights prompt v slot in
  let code = u_linear_source_descriptor width cols layout offset add_residual
    (u_linear_input a slot) layer_weights (u_linear_residual a slot) (u_linear_output a slot)
    #(hide (u_linear_x v slot)) #(hide (opt_layer0_values weights)) #(hide (u_linear_r v slot)) in
  let framed = opt_descriptor_with_frame code extra in
  opt_descriptor_refine #(pre ** extra) #(post ** extra)
    #(u_stage_resources weights_address prompt_address a weights prompt v (u_linear_step slot))
    #(u_pending_stage weights_address prompt_address a weights prompt (u_linear_step slot + 1)) framed
    (u_linear_before weights_address prompt_address a layer_weights #weights #prompt #v slot)
    (u_linear_after weights_address prompt_address a layer_weights #weights #prompt #v slot)

inline_for_extraction noextract
let u_linear_stage
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (layer_weights:larray BF16.t dec_weight_elements{
    layer_weights == u_weight_window weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements)})
  (slot:nat{slot < 4})
  (width:szp{SZ.v width == u_linear_width slot}) (cols:szp{SZ.v cols == u_linear_cols slot})
  (layout:sz{SZ.v layout == u_linear_layout slot}) (offset:sz{SZ.v offset == u_linear_offset slot})
  (add_residual:bool{add_residual == u_linear_add slot})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family opt_block_values
      (fun v -> u_stage_resources weights_address prompt_address a weights prompt v (u_linear_step slot))
      (fun _ -> u_pending_stage weights_address prompt_address a weights prompt (u_linear_step slot + 1)) =
  u_linear_shape slot;
  u_weight_window_shape weights_address opt_layer0_start (opt_layer0_start + dec_weight_elements);
  {
    pending_threads = 32sz;
    pending_shared = opt_linear_shared_desc;
    pending_code = u_linear_stage_descriptor weights_address prompt_address a layer_weights
      slot width cols layout offset add_residual #weights #prompt
  }

inline_for_extraction noextract
let opt_attention_shared_desc =
  [SH.SHArray F32.t 128sz; SH.SHArray BF16.t 1024sz; SH.SHArray F32.t 16sz]

inline_for_extraction noextract
let u_attention_pre (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (x:lseq BF16.t 32768) (old:lseq BF16.t 16384) (fraction:perm) : slprop =
  input |-> Frac fraction x ** output |-> old

inline_for_extraction noextract
let u_attention_final (output:larray BF16.t 16384) (x:lseq BF16.t 32768) : slprop =
  exists* (ys:lseq BF16.t 16384). output |-> ys **
    pure ((forall (m:u_attention_meaning). u_attention_guard x m ==> dec_matrix_rep ys (u_attention_model m)) /\
      (forall (col:Shared.index 2048). S.index ys (7 * 2048 + col) == (F.zero #BF16.t)))

inline_for_extraction noextract
let u_attention_post (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (x:lseq BF16.t 32768) (fraction:perm) : slprop =
  input |-> Frac fraction x ** u_attention_final output x

inline_for_extraction noextract
noeq type source_attention_shared = {
  score_tile: larray F32.t 128;
  probability: larray BF16.t 1024;
  denominator: larray F32.t 16
}

inline_for_extraction noextract
let opt_attention_shared (sh:SH.c_shmems opt_attention_shared_desc) : source_attention_shared = {
  score_tile = fst sh;
  probability = fst (snd sh);
  denominator = fst (snd (snd sh))
}

inline_for_extraction noextract
let dec_array_live (#a:Type0) (arr:array a) : slprop =
  exists* (dec_spec_values:seq a). arr |-> dec_spec_values

inline_for_extraction noextract
let source_attention_available (shared:source_attention_shared) (tid:nat) : slprop =
  (exists* (tv:lseq F32.t 128). shared.score_tile |-> Frac (1.0R /. 32.0R) tv) **
  when_ (tid == 0) (dec_array_live shared.probability) **
  when_ (tid == 0) (dec_array_live shared.denominator)

inline_for_extraction noextract
let u_attention_kpre (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (shared:source_attention_shared) (x:lseq BF16.t 32768) (old:lseq BF16.t 16384)
  (fraction:perm) (tid:nat) : slprop =
  input |-> Frac (fraction /. 32.0R) x ** source_attention_available shared tid **
  when_ (tid == 0) (output |-> old)

inline_for_extraction noextract
let u_attention_kpost (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (shared:source_attention_shared) (x:lseq BF16.t 32768) (fraction:perm) (tid:nat) : slprop =
  input |-> Frac (fraction /. 32.0R) x ** source_attention_available shared tid **
  when_ (tid == 0) (u_attention_final output x)

inline_for_extraction noextract
ghost
fn u_attention_block_setup (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (#x:erased (lseq BF16.t 32768)) (#old:erased (lseq BF16.t 16384))
  (#fraction:perm) (sh:SH.c_shmems opt_attention_shared_desc) (_:natlt 1) ()
  norewrite
  requires SH.live_c_shmems sh ** u_attention_pre input output x old fraction
  ensures (forall+ (tid:natlt 32).
    u_attention_kpre input output (opt_attention_shared sh) x old fraction tid) ** emp

inline_for_extraction noextract
ghost
fn u_attention_block_teardown (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (#x:erased (lseq BF16.t 32768)) (#fraction:perm)
  (sh:SH.c_shmems opt_attention_shared_desc) (_:natlt 1) ()
  norewrite
  requires (forall+ (tid:natlt 32).
    u_attention_kpost input output (opt_attention_shared sh) x fraction tid) ** emp
  ensures SH.live_c_shmems sh ** u_attention_post input output x fraction

inline_for_extraction noextract
let dec_attention_output_prefix
  (ys:lseq bf16 16384) (model:dec_spec_matrix 8 2048) (head block:nat) : prop =
  forall (row:dec_spec_index 7) (channel:dec_spec_index 2048).
    (channel / 256 < head \/ (channel / 256 = head /\ channel % 128 < block)) ==>
      Seq.index ys (row * 2048 + channel) %~ model row channel

inline_for_extraction noextract
let u_attention_prefix (x:lseq BF16.t 32768) (ys:lseq BF16.t 16384)
  (head block:nat) : prop =
  forall (m:u_attention_meaning). u_attention_guard x m ==>
    dec_attention_output_prefix ys (u_attention_model m) head block

inline_for_extraction noextract
let u_attention_owned (output:larray BF16.t 16384) (x:lseq BF16.t 32768)
  (head block:nat) : slprop =
  exists* (ys:lseq BF16.t 16384). output |-> ys ** pure (u_attention_prefix x ys head block)

inline_for_extraction noextract
ghost
fn source_tile_gather_proof (tile:larray F32.t 128)
  (#values:erased (lseq F32.t 128)) ()
  requires forall+ (_:natlt 32). tile |-> Frac (1.0R /. 32.0R) values
  ensures forall+ (tid:natlt 32). when_ (tid == 0) (tile |-> values)

inline_for_extraction noextract
let source_tile_gather (tile:larray F32.t 128)
  (#values:erased (lseq F32.t 128)) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** tile |-> Frac (1.0R /. 32.0R) values)
      (ensures fun _ -> thread_id 32 tid **
        when_ (reveal tid == 0) (tile |-> values)) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun _ -> tile |-> Frac (1.0R /. 32.0R) values)
    (fun i -> when_ (i == 0) (tile |-> values))
    (source_tile_gather_proof tile #values ()) #32 #tid

inline_for_extraction noextract
let dec_source_mma_score_tile
  (rotated:dec_bf16_values (8 * 4096)) (head:dec_spec_index 8)
  : GTot (lseq F32.t 128) =
  Seq.init_ghost 128 (fun i ->
    ScoreChest.acc2 (dec_source_mma_scores rotated head 8) (i / 8) (i % 8))

inline_for_extraction noextract
let dec_packed_values (rotated:dec_bf16_values 32768) (head:dec_spec_index 8)
  (block:nat{block < 128 /\ block % 8 = 0}) : GTot (lseq bf16 512) =
  dec_indexed_matrix 8 64 (fun lane key ->
    if key < 7 then Seq.index rotated (key * 4096 + head * 512 + 384 + block + lane)
    else ObF.zero)

inline_for_extraction noextract
let rec source_score_fold_related (input:lseq BF16.t 32768)
  (head:Shared.index 8) (steps:nat{steps <= 8})
  : Lemma (source_score_prefix input head steps == dec_source_mma_scores input head steps)
      (decreases steps) =
  if steps > 0 then (
    source_score_fold_related input head (steps - 1);
    Chest.ext (source_query_operand input head (steps - 1))
      (dec_source_query_operand input head (steps - 1));
    Chest.ext (source_key_operand input head (steps - 1))
      (dec_source_key_operand input head (steps - 1)))

inline_for_extraction noextract
let source_score_tile_related (input:lseq BF16.t 32768) (head:Shared.index 8)
  : Lemma (source_tile_values (source_score_prefix input head 8) ==
      dec_source_mma_score_tile input head) =
  source_score_fold_related input head 8;
  assert (S.equal (source_tile_values (source_score_prefix input head 8))
    (dec_source_mma_score_tile input head));
  S.lemma_eq_elim (source_tile_values (source_score_prefix input head 8))
    (dec_source_mma_score_tile input head)

inline_for_extraction noextract
let dec_dense_left_operand_at (width:nat{dec_dense_width width}) (a:lseq bf16 (16 * width))
  (step:nat{step < width / 16}) (row:dec_spec_index 16) (lane:dec_spec_index 16)
  : Lemma
      (Chest.acc2 (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 step) row lane ==
        Seq.index a (row * width + step * 16 + lane)) = ()

inline_for_extraction noextract
let dec_dense_right_operand_at (width:nat{dec_dense_width width}) (b:lseq bf16 (8 * width))
  (step:nat{step < width / 16}) (lane:dec_spec_index 16) (col:dec_spec_index 8)
  : Lemma
      (Chest.acc2 (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 step 0) lane col ==
        Seq.index b (col * width + step * 16 + lane)) = ()

inline_for_extraction noextract
let dec_packed_values_at (rotated:dec_bf16_values 32768) (head:dec_spec_index 8)
  (block:nat{block < 128 /\ block % 8 = 0}) (lane:dec_spec_index 8) (key:dec_spec_index 64)
  : Lemma (Seq.index (dec_packed_values rotated head block) (lane * 64 + key) ==
      (if key < 7 then Seq.index rotated (key * 4096 + head * 512 + 384 + block + lane)
       else ObF.zero)) =
  dec_indexed_matrix_at 8 64
    (fun d k -> if k < 7 then Seq.index rotated (k * 4096 + head * 512 + 384 + block + d)
      else ObF.zero) lane key

inline_for_extraction noextract
let source_value_matrices_related (input:lseq BF16.t 32768)
  (probability:lseq BF16.t 1024) (head:Shared.index 8)
  (block:Shared.index 16) (step:Shared.index 4)
  : Lemma (
      Gather.value_a probability step ==
        ematrix_subtile (Tensor.from_seq (rm 16 64) probability) 16 16 0 step /\
      source_value_operand input head block step ==
        ematrix_subtile
          (Tensor.from_seq (cm 64 8) (dec_packed_values input head (block * 8)))
          16 8 step 0) =
  let left = ematrix_subtile (Tensor.from_seq (rm 16 64) probability) 16 16 0 step in
  let right = ematrix_subtile
    (Tensor.from_seq (cm 64 8) (dec_packed_values input head (block * 8)))
    16 8 step 0 in
  let left_at (r:Shared.index 16) (c:Shared.index 16)
    : Lemma (Chest.acc2 (Gather.value_a probability step) r c == Chest.acc2 left r c) =
    dec_dense_left_operand_at 64 probability step r c
  in C.forall_intro_2 left_at;
  let right_at (r:Shared.index 16) (c:Shared.index 8)
    : Lemma (Chest.acc2 (source_value_operand input head block step) r c ==
        Chest.acc2 right r c) =
    dec_dense_right_operand_at 64 (dec_packed_values input head (block * 8)) step r c;
    dec_packed_values_at input head (block * 8) c (step * 16 + r)
  in C.forall_intro_2 right_at;
  Chest.ext (Gather.value_a probability step) left;
  Chest.ext (source_value_operand input head block step) right

inline_for_extraction noextract
let rec source_value_fold_related (input:lseq BF16.t 32768)
  (probability:lseq BF16.t 1024) (head:Shared.index 8)
  (block:Shared.index 16) (steps:nat{steps <= 4})
  : Lemma (
      source_value_prefix input probability head block steps ==
        dec_dense_mma_fold 64 probability (dec_packed_values input head (block * 8)) steps)
      (decreases steps) =
  if steps > 0 then (
    source_value_fold_related input probability head block (steps - 1);
    source_value_matrices_related input probability head block (steps - 1);
    dec_dense_mma_step 64 probability (dec_packed_values input head (block * 8)) (steps - 1))

inline_for_extraction noextract
let source_value_tile_related (input:lseq BF16.t 32768)
  (probability:lseq BF16.t 1024) (head:Shared.index 8) (block:Shared.index 16)
  : Lemma (
      source_tile_values (source_value_prefix input probability head block 4) ==
        Tensor.to_seq (rm 16 8)
          (dec_dense_mma_fold 64 probability (dec_packed_values input head (block * 8)) 4)) =
  source_value_fold_related input probability head block 4;
  source_tile_sequence_binding (source_value_prefix input probability head block 4)

inline_for_extraction noextract
let source_probability_shared (shared:source_attention_shared)
  (tile:lseq F32.t 128)
  (fact:lseq BF16.t 1024 -> lseq F32.t 16 -> prop) : slprop =
  exists* (p:lseq BF16.t 1024) (d:lseq F32.t 16).
    shared.score_tile |-> Frac (1.0R /. 32.0R) tile **
    shared.probability |-> Frac (1.0R /. 32.0R) p **
    shared.denominator |-> Frac (1.0R /. 32.0R) d ** pure (fact p d)

inline_for_extraction noextract
let source_probability_owned (shared:source_attention_shared)
  (tile:lseq F32.t 128)
  (fact:lseq BF16.t 1024 -> lseq F32.t 16 -> prop) : slprop =
  exists* (p:lseq BF16.t 1024) (d:lseq F32.t 16).
    shared.score_tile |-> tile ** shared.probability |-> p **
    shared.denominator |-> d ** pure (fact p d)

inline_for_extraction noextract
ghost
fn source_publish_observed_proof (shared:source_attention_shared)
  (tile:lseq F32.t 128)
  (fact:lseq BF16.t 1024 -> lseq F32.t 16 -> prop) ()
  requires forall+ (tid:natlt 32).
    when_ (tid == 0) (source_probability_owned shared tile fact)
  ensures forall+ (_:natlt 32). source_probability_shared shared tile fact

inline_for_extraction noextract
let source_publish_observed (shared:source_attention_shared)
  (#tile:erased (lseq F32.t 128))
  (fact:lseq BF16.t 1024 -> lseq F32.t 16 -> prop) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid **
        when_ (reveal tid == 0) (source_probability_owned shared tile fact))
      (ensures fun _ -> thread_id 32 tid ** source_probability_shared shared tile fact) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun i -> when_ (i == 0) (source_probability_owned shared tile fact))
    (fun _ -> source_probability_shared shared tile fact)
    (source_publish_observed_proof shared tile fact ()) #32 #tid

inline_for_extraction noextract
let source_value_read_phase (shared:source_attention_shared) (tid:nat)
  (pv:lseq BF16.t 1024) (dv:lseq F32.t 16) (block:nat) : slprop =
  if block < 16 then
    shared.probability |-> Frac (1.0R /. 32.0R) pv **
    shared.denominator |-> Frac (1.0R /. 32.0R) dv
  else when_ (tid == 0) (shared.probability |-> pv ** shared.denominator |-> dv)

inline_for_extraction noextract
ghost
fn source_forget_probability_owner (shared:source_attention_shared) (tid:szlt 32)
  (#pv:erased (lseq BF16.t 1024)) (#dv:erased (lseq F32.t 16))
  requires when_ (SZ.v tid == 0) (shared.probability |-> pv ** shared.denominator |-> dv)
  ensures when_ (SZ.v tid == 0) (dec_array_live shared.probability) **
    when_ (SZ.v tid == 0) (dec_array_live shared.denominator)

inline_for_extraction noextract
ghost
fn source_tile_distribute_proof (tile:larray F32.t 128)
  (#values:erased (lseq F32.t 128)) ()
  requires forall+ (tid:natlt 32). when_ (tid == 0) (tile |-> values)
  ensures forall+ (_:natlt 32). tile |-> Frac (1.0R /. 32.0R) values

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

inline_for_extraction noextract
let source_value_end (shared:source_attention_shared) (#block:enatlt 16)
  (#tv:erased (lseq F32.t 128)) (#pv:erased (lseq BF16.t 1024))
  (#dv:erased (lseq F32.t 16)) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid **
        when_ (reveal tid == 0) (shared.score_tile |-> tv) **
        shared.probability |-> Frac (1.0R /. 32.0R) pv **
        shared.denominator |-> Frac (1.0R /. 32.0R) dv)
      (ensures fun _ -> thread_id 32 tid **
        shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
        source_value_read_phase shared tid pv dv (reveal block + 1)) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun i -> when_ (i == 0) (shared.score_tile |-> tv) **
      shared.probability |-> Frac (1.0R /. 32.0R) pv **
      shared.denominator |-> Frac (1.0R /. 32.0R) dv)
    (fun i -> shared.score_tile |-> Frac (1.0R /. 32.0R) tv **
      source_value_read_phase shared i pv dv (reveal block + 1))
    (source_value_end_proof shared (reveal block) #tv #pv #dv ()) #32 #tid

inline_for_extraction noextract
let dec_source_live (row:dec_spec_index 16) (col:dec_spec_index 64) : bool =
  row < 14 && col <= row / 2

inline_for_extraction noextract
let dec_source_column (part:dec_spec_index 4) (step:dec_spec_index 16)
  : col:nat{col < 64} =
  part * 2 + (step / 2) * 8 + step % 2

inline_for_extraction noextract
let dec_negative_infinity () : ObF32.t =
  ObF.sub ObF.zero ObF.infinity

inline_for_extraction noextract
let dec_source_row_score
  (row:dec_spec_index 16) (scores:lseq ObF32.t 8) (col:dec_spec_index 64)
  : ObF32.t =
  if dec_source_live row col then FStar.Seq.index scores col else dec_negative_infinity ()

inline_for_extraction noextract
let rec dec_source_part_maximum
  (row:dec_spec_index 16) (scores:lseq ObF32.t 8)
  (part:dec_spec_index 4) (steps:nat{steps <= 16})
  : Tot ObF32.t (decreases steps) =
  if steps = 0 then dec_negative_infinity ()
  else
    ObF.fmax (dec_source_part_maximum row scores part (steps - 1))
      (dec_source_row_score row scores (dec_source_column part (steps - 1)))

inline_for_extraction noextract
let dec_source_maximum (row:dec_spec_index 16) (scores:lseq ObF32.t 8) : ObF32.t =
  ObF.fmax
    (ObF.fmax (dec_source_part_maximum row scores 0 16) (dec_source_part_maximum row scores 1 16))
    (ObF.fmax (dec_source_part_maximum row scores 2 16) (dec_source_part_maximum row scores 3 16))

inline_for_extraction noextract
let dec_source_score_row (tile:lseq ObF32.t 128) (row:dec_spec_index 16)
  : lseq ObF32.t 8 =
  ObSeq.slice tile (row * 8) (row * 8 + 8)

inline_for_extraction noextract
let dec_source_shared_shifted_numerator
  (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 14) (col:dec_spec_index 7)
  (maximum:real) : GTot real =
  ObExp.exp
    ((dec_spec_dot (query (row / 2) (2 * head + row % 2)) (key col head) -.
      maximum) *. ObRoot.rsqrt 128.0R)

inline_for_extraction noextract
let dec_attention_numerator
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 16) (col:dec_spec_index 64) : GTot real =
  if dec_source_live row col then
    dec_source_shared_shifted_numerator query key head row col
      (to_real (dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row)))
  else 0.0R

inline_for_extraction noextract
let dec_attention_denominator
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 16) : GTot real =
  if row < 14 then dec_spec_sum (row / 2 + 1) (dec_attention_numerator rotated query key head row)
  else 1.0R

inline_for_extraction noextract
let dec_attention_probability_contract
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (ps:lseq bf16 1024) (ds:lseq F32.t 16) : prop =
  dec_bilateral_mma_live_scores_finite rotated head ==>
    dec_matrix_rep ps (dec_attention_numerator rotated query key head) /\
    dec_segment_rep ds 0 16 (fun row -> if row < 16 then dec_attention_denominator rotated query key head row else 0.0R)

inline_for_extraction noextract
let u_probability_contract (x:lseq BF16.t 32768) (head:Shared.index 8)
  (ps:lseq BF16.t 1024) (ds:lseq F32.t 16) : prop =
  forall (m:u_attention_meaning).
    dec_source_qk_representation x m.u_query m.u_key ==>
      dec_attention_probability_contract x m.u_query m.u_key head ps ds

inline_for_extraction noextract
let dec_seq_length_view (#a:Type0) (n:nat) (dec_spec_values:seq a)
  : Pure (lseq a n)
      (requires Seq.length dec_spec_values == n)
      (ensures fun result -> result == dec_spec_values) =
  dec_spec_values

inline_for_extraction noextract
ghost
fn dec_take_owned_array (#a:Type0) (#n:erased nat) (arr:larray a n) (tid:szlt 32) (default:a)
  requires when_ (SZ.v tid == 0) (dec_array_live arr)
  returns dec_spec_values:erased (lseq a n)
  ensures when_ (SZ.v tid == 0) (arr |-> dec_spec_values)

inline_for_extraction noextract
let u_probability_guard (x:lseq BF16.t 32768) (head:Shared.index 8)
  (m:u_attention_meaning) : prop =
  dec_bilateral_mma_live_scores_finite x head /\
    dec_source_qk_representation x m.u_query m.u_key

inline_for_extraction noextract
let dec_negative_infinity_compares_equal ()
  : Lemma (ieee_eq (dec_negative_infinity ()) (dec_negative_infinity ())) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  ObF.eq_spec (dec_negative_infinity ()) (dec_negative_infinity ())

inline_for_extraction noextract
let dec_source_attention_literal () : ObF32.t =
  ObF.of_literal "0.12751743197441101074"

inline_for_extraction noextract
let dec_source_attention_literal_is_positive_normal ()
  : Lemma (ObF32.positive_normal (dec_source_attention_literal ())) =
  assert_norm (Kuiper.Float32.Literals.positive_normal_decimal "0.12751743197441101074");
  ObF32.positive_normal_of_literal "0.12751743197441101074"

inline_for_extraction noextract
let dec_source_scale_upper_bound ()
  : Lemma (ObF.lte (dec_source_attention_literal ()) ObF.one) =
  assert_norm (Kuiper.Float32.Literals.positive_normal_decimal "0.12751743197441101074");
  assert_norm (Kuiper.Float32.Literals.at_most_one_decimal "0.12751743197441101074");
  ObF32.positive_literal_at_most_one "0.12751743197441101074"

inline_for_extraction noextract
let dec_source_maximum_cases (maximum:ObF32.t)
  : Lemma (not (NaN? (kind maximum)) ==>
      Finite? (kind maximum) \/ maximum == ObF.infinity \/ maximum == dec_negative_infinity ()) =
  if not (NaN? (kind maximum)) then ObF32.non_nan_classification maximum

inline_for_extraction noextract
let dec_source_score_row_at
  (tile:lseq ObF32.t 128) (row:dec_spec_index 16) (col:dec_spec_index 8)
  : Lemma
      (ObSeq.index (dec_source_score_row tile row) col ==
        ObSeq.index tile (row * 8 + col))
  = ()

inline_for_extraction noextract
fn dec_source_read_score_tile
  (tile:larray ObF32.t 128) (row:szlt 16) (col:szlt 64)
  (#dec_spec_values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f dec_spec_values
  returns result:ObF32.t
  ensures pure (result ==
    dec_source_row_score row (dec_source_score_row dec_spec_values row) col)

inline_for_extraction noextract
let dec_fmax_non_nan_when_both_non_nan (left right:ObF32.t)
  : Lemma
      (requires not (NaN? (kind left)) /\ not (NaN? (kind right)))
      (ensures not (NaN? (kind (ObF.fmax left right)))) =
  ObF.fmax_spec left right;
  ObF.eq_spec (ObF.fmax left right) (if ObF.lt left right then right else left)

inline_for_extraction noextract
let dec_fmax_non_nan_left (left right:ObF32.t)
  : Lemma
      (requires not (NaN? (kind left)))
      (ensures not (NaN? (kind (ObF.fmax left right)))) =
  ObF32.fmax_special_values left right;
  if not (NaN? (kind right)) then
    dec_fmax_non_nan_when_both_non_nan left right

inline_for_extraction noextract
let rec dec_source_part_maximum_non_nan
  (row:dec_spec_index 16) (scores:lseq ObF32.t 8)
  (part:dec_spec_index 4) (steps:nat{steps <= 16})
  : Lemma (not (NaN? (kind (dec_source_part_maximum row scores part steps))))
      (decreases steps) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  if steps > 0 then (
    dec_source_part_maximum_non_nan row scores part (steps - 1);
    dec_fmax_non_nan_left
      (dec_source_part_maximum row scores part (steps - 1))
      (dec_source_row_score row scores (dec_source_column part (steps - 1)))
  )

inline_for_extraction noextract
let dec_source_maximum_non_nan (row:dec_spec_index 16) (scores:lseq ObF32.t 8)
  : Lemma (not (NaN? (kind (dec_source_maximum row scores)))) =
  dec_source_part_maximum_non_nan row scores 0 16;
  dec_source_part_maximum_non_nan row scores 1 16;
  dec_source_part_maximum_non_nan row scores 2 16;
  dec_source_part_maximum_non_nan row scores 3 16;
  dec_fmax_non_nan_left
    (dec_source_part_maximum row scores 0 16) (dec_source_part_maximum row scores 1 16);
  dec_fmax_non_nan_left
    (dec_source_part_maximum row scores 2 16) (dec_source_part_maximum row scores 3 16);
  dec_fmax_non_nan_left
    (ObF.fmax (dec_source_part_maximum row scores 0 16) (dec_source_part_maximum row scores 1 16))
    (ObF.fmax (dec_source_part_maximum row scores 2 16) (dec_source_part_maximum row scores 3 16))

inline_for_extraction noextract
let dec_source_part_maximum_step
  (row:dec_spec_index 16) (scores:lseq ObF32.t 8)
  (part:dec_spec_index 4) (step:dec_spec_index 16)
  : Lemma
      (dec_source_part_maximum row scores part (step + 1) ==
        ObF.fmax (dec_source_part_maximum row scores part step)
          (dec_source_row_score row scores (dec_source_column part step)))
  = ()

inline_for_extraction noextract
fn dec_source_part_maximum_from_tile
  (tile:larray ObF32.t 128) (row:szlt 16) (part:szlt 4)
  (#dec_spec_values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f dec_spec_values
  returns result:ObF32.t
  ensures pure (result ==
    dec_source_part_maximum row (dec_source_score_row dec_spec_values row) part 16)

inline_for_extraction noextract
fn dec_source_maximum_from_tile
  (tile:larray ObF32.t 128) (row:szlt 16)
  (#dec_spec_values:erased (lseq ObF32.t 128)) (#f:perm)
  preserves gpu ** tile |-> Frac f dec_spec_values
  returns result:ObF32.t
  ensures pure (
    result == dec_source_maximum row (dec_source_score_row dec_spec_values row) /\
    not (NaN? (kind result)))

inline_for_extraction noextract
let dec_source_exponent_contract (value scaled result:ObF32.t) : prop =
  forall (vr sr ar:real).
    v_approximates value vr /\
    v_approximates scaled sr /\
    v_approximates (dec_source_attention_literal ()) ar ==>
    v_approximates result (ObPow.exp2 (vr *. ar -. sr))

inline_for_extraction noextract
let dec_source_exponent_chain
  (value scaled argument result:ObF32.t)
  : Lemma
      (requires
        (forall (vr ar nr:real).
          v_approximates value vr /\
          v_approximates (dec_source_attention_literal ()) ar /\
          v_approximates (NativeNeg.neg scaled) nr ==>
          v_approximates argument (vr *. ar +. nr)) /\
        (forall (r:real).
          v_approximates argument r ==> v_approximates result (ObPow.exp2 r)))
      (ensures dec_source_exponent_contract value scaled result)
  =
  let at (vr sr ar:real)
    : Lemma
        (requires v_approximates value vr /\ v_approximates scaled sr /\
          v_approximates (dec_source_attention_literal ()) ar)
        (ensures v_approximates result (ObPow.exp2 (vr *. ar -. sr)))
    =
    let zero_ok = a0 #ObF32.t in
    NativeNeg.neg_approx scaled sr;
    assert (v_approximates argument (vr *. ar +. (0.0R -. sr)))
  in
  FStar.Classical.forall_intro_3 (FStar.Classical.move_requires_3 at)

inline_for_extraction noextract
let dec_source_branch_exponent_contract (value scaled result:ObF32.t) : prop =
  (ieee_eq value (dec_negative_infinity ()) ==>
    result == (ObF.zero #ObF32.t) /\ v_approximates result 0.0R) /\
  (not (ieee_eq value (dec_negative_infinity ())) ==>
    dec_source_exponent_contract value scaled result)

inline_for_extraction noextract
fn dec_source_exponent_from_scaled (value scaled:ObF32.t)
  preserves gpu
  returns result:ObF32.t
  ensures pure (dec_source_branch_exponent_contract value scaled result)

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

inline_for_extraction noextract
let dec_partition_entry (f:dec_spec_vector 64) (part:dec_spec_index 4) (step:nat) : GTot real =
  if step < 16 then f (dec_source_column part step) else 0.0R

inline_for_extraction noextract
let dec_probability_prefix (ys:lseq bf16 1024) (row:dec_spec_index 16)
  (f:dec_spec_vector 64) (part step:nat) (guard:prop) : prop =
  forall (p:dec_spec_index 4) (j:dec_spec_index 16).
    (p < part \/ (p = part /\ j < step)) ==>
      (guard ==> Seq.index ys (row * 64 + dec_source_column p j) %~ f (dec_source_column p j))

inline_for_extraction noextract
let dec_fmax_negative_infinity ()
  : Lemma
      (ObF.fmax (dec_negative_infinity ()) (dec_negative_infinity ()) == dec_negative_infinity ()) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  ObF.fmax_spec (dec_negative_infinity ()) (dec_negative_infinity ());
  ObF.eq_spec (ObF.fmax (dec_negative_infinity ()) (dec_negative_infinity ())) (dec_negative_infinity ())

inline_for_extraction noextract
let dec_empty_rows_are_masked (row:dec_spec_index 16{14 <= row}) (col:dec_spec_index 64)
  : Lemma (not (dec_source_live row col)) =
  ()

inline_for_extraction noextract
let rec dec_source_part_maximum_empty
  (row:dec_spec_index 16{14 <= row}) (scores:lseq ObF32.t 8)
  (part:dec_spec_index 4) (steps:nat{steps <= 16})
  : Lemma
      (dec_source_part_maximum row scores part steps == dec_negative_infinity ())
      (decreases steps)
  =
  if steps > 0 then (
    dec_source_part_maximum_empty row scores part (steps - 1);
    dec_empty_rows_are_masked row (dec_source_column part (steps - 1));
    dec_fmax_negative_infinity ()
  )

inline_for_extraction noextract
let dec_source_maximum_empty
  (row:dec_spec_index 16{14 <= row}) (scores:lseq ObF32.t 8)
  : Lemma
      (dec_source_maximum row scores == dec_negative_infinity () /\
       ieee_eq (dec_source_maximum row scores) (dec_negative_infinity ()))
  =
  dec_source_part_maximum_empty row scores 0 16;
  dec_source_part_maximum_empty row scores 1 16;
  dec_source_part_maximum_empty row scores 2 16;
  dec_source_part_maximum_empty row scores 3 16;
  dec_fmax_negative_infinity ();
  dec_negative_infinity_compares_equal ()

inline_for_extraction noextract
let dec_source_mma_score_tile_at
  (rotated:lseq bf16 (8 * 4096)) (head:dec_spec_index 8)
  (row:dec_spec_index 16) (col:dec_spec_index 8)
  : Lemma
      (ObSeq.index (dec_source_mma_score_tile rotated head) (row * 8 + col) ==
        ObChest.acc2 (dec_source_mma_scores rotated head 8) row col)
  = ()

inline_for_extraction noextract
let dec_source_sentinel_comparison_is_equality (value:ObF32.t)
  : Lemma
      (requires ieee_eq value (dec_negative_infinity ()))
      (ensures value == dec_negative_infinity ()) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  ObF.eq_spec value (dec_negative_infinity ());
  if ObF.is_zero (dec_negative_infinity ()) then
    ObF.is_zero_spec (dec_negative_infinity ())

inline_for_extraction noextract
let dec_bilateral_finite_not_sentinel (value:ObF32.t)
  : Lemma
      (requires Finite? (kind value))
      (ensures not (ieee_eq value (dec_negative_infinity ()))) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  if ieee_eq value (dec_negative_infinity ()) then
    dec_source_sentinel_comparison_is_equality value

inline_for_extraction noextract
let dec_bilateral_negative_infinity_lt_finite (value:ObF32.t)
  : Lemma
      (requires Finite? (kind value))
      (ensures ObF.lt (dec_negative_infinity ()) value) =
  let infinity : ObF32.t = ObF.infinity in
  let negative = dec_negative_infinity () in
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
  dec_bilateral_finite_not_sentinel value;
  ObF.lte_is_lt_or_eq value negative;
  ObF.negate_lt_is_lte negative value

inline_for_extraction noextract
let dec_bilateral_equal_finite (left right:ObF32.t)
  : Lemma
      (requires ieee_eq left right /\ Finite? (kind right))
      (ensures Finite? (kind left)) =
  ObF.eq_spec left right;
  if ObF.is_zero left then ObF.is_zero_spec left

inline_for_extraction noextract
let dec_bilateral_finite_or_sentinel (value:ObF32.t) : prop =
  Finite? (kind value) \/ value == dec_negative_infinity ()

inline_for_extraction noextract
let dec_bilateral_fmax_classification (left right:ObF32.t)
  : Lemma
      (requires dec_bilateral_finite_or_sentinel left /\ dec_bilateral_finite_or_sentinel right)
      (ensures
        dec_bilateral_finite_or_sentinel (ObF.fmax left right) /\
        ((Finite? (kind left) \/ Finite? (kind right)) ==>
          Finite? (kind (ObF.fmax left right)))) =
  ObF.neg_kind (ObF.infinity #ObF32.t);
  if Finite? (kind left) then (
    dec_bilateral_negative_infinity_lt_finite left;
    ObF.negate_lt_is_lte left (dec_negative_infinity ());
    ObF.lte_is_lt_or_eq (dec_negative_infinity ()) left
  );
  if Finite? (kind right) then dec_bilateral_negative_infinity_lt_finite right;
  ObF.fmax_spec left right;
  let selected = if ObF.lt left right then right else left in
  if Finite? (kind selected) then
    dec_bilateral_equal_finite (ObF.fmax left right) selected
  else
    dec_source_sentinel_comparison_is_equality (ObF.fmax left right)

inline_for_extraction noextract
let dec_bilateral_live_row_finite
  (row:dec_spec_index 14) (scores:lseq ObF32.t 8) : prop =
  forall (col:dec_spec_index 7{col <= row / 2}).
    Finite? (kind (ObSeq.index scores col))

inline_for_extraction noextract
let dec_bilateral_row_score_classification
  (row:dec_spec_index 14) (scores:lseq ObF32.t 8) (col:dec_spec_index 64)
  : Lemma
      (requires dec_bilateral_live_row_finite row scores)
      (ensures dec_bilateral_finite_or_sentinel (dec_source_row_score row scores col)) =
  ()

inline_for_extraction noextract
let rec dec_bilateral_part_maximum_classification
  (row:dec_spec_index 14) (scores:lseq ObF32.t 8)
  (part:dec_spec_index 4) (steps:nat{steps <= 16})
  : Lemma
      (requires dec_bilateral_live_row_finite row scores)
      (ensures dec_bilateral_finite_or_sentinel (dec_source_part_maximum row scores part steps))
      (decreases steps) =
  if steps > 0 then (
    dec_bilateral_part_maximum_classification row scores part (steps - 1);
    dec_bilateral_row_score_classification row scores (dec_source_column part (steps - 1));
    dec_bilateral_fmax_classification (dec_source_part_maximum row scores part (steps - 1))
      (dec_source_row_score row scores (dec_source_column part (steps - 1)))
  )

inline_for_extraction noextract
let rec dec_bilateral_part_zero_finite
  (row:dec_spec_index 14) (scores:lseq ObF32.t 8) (steps:pos{steps <= 16})
  : Lemma
      (requires dec_bilateral_live_row_finite row scores)
      (ensures Finite? (kind (dec_source_part_maximum row scores 0 steps)))
      (decreases steps) =
  if steps = 1 then (
    assert (dec_source_live row 0);
    dec_bilateral_fmax_classification (dec_negative_infinity ()) (ObSeq.index scores 0)
  ) else (
    dec_bilateral_part_zero_finite row scores (steps - 1);
    dec_bilateral_row_score_classification row scores (dec_source_column 0 (steps - 1));
    dec_bilateral_fmax_classification (dec_source_part_maximum row scores 0 (steps - 1))
      (dec_source_row_score row scores (dec_source_column 0 (steps - 1)))
  )

inline_for_extraction noextract
let dec_bilateral_live_maximum_finite
  (row:dec_spec_index 14) (scores:lseq ObF32.t 8)
  : Lemma
      (requires dec_bilateral_live_row_finite row scores)
      (ensures
        Finite? (kind (dec_source_maximum row scores)) /\
        not (ieee_eq (dec_source_maximum row scores) (dec_negative_infinity ()))) =
  dec_bilateral_part_zero_finite row scores 16;
  dec_bilateral_part_maximum_classification row scores 1 16;
  dec_bilateral_part_maximum_classification row scores 2 16;
  dec_bilateral_part_maximum_classification row scores 3 16;
  let m0 = dec_source_part_maximum row scores 0 16 in
  let m1 = dec_source_part_maximum row scores 1 16 in
  let m2 = dec_source_part_maximum row scores 2 16 in
  let m3 = dec_source_part_maximum row scores 3 16 in
  dec_bilateral_fmax_classification m0 m1;
  dec_bilateral_fmax_classification m2 m3;
  dec_bilateral_fmax_classification (ObF.fmax m0 m1) (ObF.fmax m2 m3);
  dec_bilateral_finite_not_sentinel (dec_source_maximum row scores)

inline_for_extraction noextract
let dec_bilateral_mma_maximum_exclusion
  (rotated:lseq bf16 (8 * 4096)) (head:dec_spec_index 8) (row:dec_spec_index 14)
  : Lemma
      (requires dec_bilateral_mma_live_scores_finite rotated head)
      (ensures
        Finite? (kind (dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row))) /\
        not (ieee_eq
          (dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row))
          (dec_negative_infinity ()))) =
  let at (col:dec_spec_index 7{col <= row / 2})
    : Lemma
        (Finite? (kind (ObSeq.index
          (dec_source_score_row (dec_source_mma_score_tile rotated head) row) col))) =
    dec_source_score_row_at (dec_source_mma_score_tile rotated head) row col;
    dec_source_mma_score_tile_at rotated head row col
  in
  FStar.Classical.forall_intro at;
  dec_bilateral_live_maximum_finite row (dec_source_score_row (dec_source_mma_score_tile rotated head) row)

inline_for_extraction noextract
let dec_attention_baseline (row:dec_spec_index 16) : real = if row < 14 then 0.0R else 1.0R

inline_for_extraction noextract
let dec_attention_rescale_refines
  (rotated:dec_bf16_values 32768) (head:dec_spec_index 8) (row:dec_spec_index 16) (rescale:F32.t)
  : Lemma
      (requires
        (ieee_eq (dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row))
          (dec_negative_infinity ()) ==> rescale %~ 1.0R) /\
        (not (ieee_eq (dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row))
          (dec_negative_infinity ())) /\
         Finite? (kind (dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row))) ==> rescale %~ 0.0R))
      (ensures dec_bilateral_mma_live_scores_finite rotated head ==> rescale %~ dec_attention_baseline row) =
  let under_domain ()
    : Lemma
        (requires dec_bilateral_mma_live_scores_finite rotated head)
        (ensures rescale %~ dec_attention_baseline row) =
    if row < 14 then dec_bilateral_mma_maximum_exclusion rotated head row
    else (
      dec_source_maximum_empty row (dec_source_score_row (dec_source_mma_score_tile rotated head) row);
      dec_negative_infinity_compares_equal ()
    )
  in FStar.Classical.move_requires under_domain ()

inline_for_extraction noextract
let dec_source_score_term
  (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 14) (col:dec_spec_index 7)
  (k:nat) : GTot real =
  if k < 128 then query (row / 2) (2 * head + row % 2) k *. key col head k
  else 0.0R

inline_for_extraction noextract
let dec_source_dot16 (a b:lseq real 16) (i:nat) : Tot real =
  if i < 16 then ObSeq.index a i *. ObSeq.index b i else 0.0R

inline_for_extraction noextract
let dec_source_matrix_term
  (a:ObChest.chest2 real 16 16) (b:ObChest.chest2 real 16 8)
  (row:dec_spec_index 16) (col:dec_spec_index 8) (k:nat) : GTot real =
  if k < 16 then ObChest.acc2 a row k *. ObChest.acc2 b k col else 0.0R

inline_for_extraction noextract
let rec dec_source_matmul_sum
  (a:ObChest.chest2 real 16 16) (b:ObChest.chest2 real 16 8)
  (row:dec_spec_index 16) (col:dec_spec_index 8) (count:nat{count <= 16})
  : Lemma
      (ObGemm.__matmul_single a b row col count ==
        dec_source_sum_n count (dec_source_matrix_term a b row col))
      (decreases count) =
  if count = 0 then ObGemm.matmul_zero_lemma a b row col
  else (
    dec_source_matmul_sum a b row col (count - 1);
    ObGemm.matmul_single_lemma a b row col count
  )

inline_for_extraction noextract
let dec_source_mma_cell_approx
  (a:ObChest.chest2 bf16 16 16) (b:ObChest.chest2 bf16 16 8)
  (c:ObChest.chest2 ObF32.t 16 8)
  (row:dec_spec_index 16) (col:dec_spec_index 8)
  (ar br:lseq real 16) (cr:real)
  : Lemma
      (requires
        (forall (k:dec_spec_index 16).
          ObChest.acc2 a row k %~ ObSeq.index ar k) /\
        (forall (k:dec_spec_index 16).
          ObChest.acc2 b k col %~ ObSeq.index br k) /\
        ObChest.acc2 c row col %~ cr)
      (ensures ObChest.acc2 (ObMMA.emma c a b) row col %~
        (cr +. dec_source_sum_n 16 (dec_source_dot16 ar br))) =
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
  dec_source_matmul_sum ra rb row col 16;
  dec_source_sum_ext 16 (dec_source_matrix_term ra rb row col) (dec_source_dot16 ar br);
  assert (ObChest.acc2 (ObGemm.matplus rc (ObGemm.matmul ra rb)) row col ==
    cr +. dec_source_sum_n 16 (dec_source_dot16 ar br))

inline_for_extraction noextract
let rec dec_source_mma_scores_at
  (rotated:lseq bf16 (8 * 4096)) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 14) (col:dec_spec_index 7)
  (steps:nat{steps <= 8})
  : Lemma
      (requires dec_source_qk_representation rotated query key)
      (ensures ObChest.acc2 (dec_source_mma_scores rotated head steps) row col %~
        dec_source_sum_n (16 * steps) (dec_source_score_term query key head row col))
      (decreases steps) =
  if steps = 0 then (
    let zero_ok = a0 #ObF32.t in ()
  ) else (
    let previous = steps - 1 in
    let ar = ObSeq.init_ghost 16 (fun i ->
      query (row / 2) (2 * head + row % 2) (16 * previous + i)) in
    let br = ObSeq.init_ghost 16 (fun i -> key col head (16 * previous + i)) in
    dec_source_mma_scores_at rotated query key head row col previous;
    dec_source_mma_cell_approx
      (dec_source_query_operand rotated head previous)
      (dec_source_key_operand rotated head previous)
      (dec_source_mma_scores rotated head previous)
      row col ar br
      (dec_source_sum_n (16 * previous) (dec_source_score_term query key head row col));
    dec_source_sum_ext 16 (dec_source_dot16 ar br)
      (fun i -> dec_source_score_term query key head row col (16 * previous + i));
    dec_source_sum_append (16 * previous) 16 (dec_source_score_term query key head row col)
  )

inline_for_extraction noextract
let dec_source_mma_tile_represents_dot
  (rotated:lseq bf16 (8 * 4096)) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 14) (col:dec_spec_index 7)
  : Lemma
      (requires dec_source_qk_representation rotated query key)
      (ensures
        v_approximates (ObSeq.index (dec_source_mma_score_tile rotated head) (row * 8 + col))
          (dec_spec_dot (query (row / 2) (2 * head + row % 2)) (key col head))) =
  dec_source_mma_scores_at rotated query key head row col 8;
  dec_source_mma_score_tile_at rotated head row col;
  dec_source_sum_shared 128 (dec_source_score_term query key head row col);
  dec_sum_extensional 128 (dec_source_score_term query key head row col)
    (fun i -> query (row / 2) (2 * head + row % 2) i *. key col head i);
  assert_norm (dec_spec_dot (query (row / 2) (2 * head + row % 2)) (key col head) ==
    dec_spec_sum 128
      (fun i -> query (row / 2) (2 * head + row % 2) i *. key col head i))

inline_for_extraction noextract
let dec_intended_attention_exp2_scale () : real =
  dec_source_log_two_nonzero ();
  ObRoot.rsqrt 128.0R /. ObExp.log 2.0R

inline_for_extraction noextract
let dec_source_attention_ideal_interval ()
  : Lemma (
      1275174308245986.0R /. 10000000000000000.0R <. dec_intended_attention_exp2_scale () /\
      dec_intended_attention_exp2_scale () <. 1275174308245988.0R /. 10000000000000000.0R) =
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
let dec_source_attention_literal_represents_ideal ()
  : Lemma (v_approximates (dec_source_attention_literal ()) (dec_intended_attention_exp2_scale ())) =
  dec_source_attention_ideal_interval ();
  assert_norm (
    Kuiper.Float32.Literals.common_normal_cell "0.12751743197441101074"
      1275174308245986 10000000000000000
      1275174308245988 10000000000000000
      8557550 (-26));
  ObF32.literal_interval_approx "0.12751743197441101074"
    1275174308245986 10000000000000000
    1275174308245988 10000000000000000
    8557550 (-26) (dec_intended_attention_exp2_scale ())

inline_for_extraction noextract
let dec_source_attention_change_of_base (difference:real)
  : Lemma (ObPow.exp2 (difference *. dec_intended_attention_exp2_scale ()) ==
      ObExp.exp (difference *. ObRoot.rsqrt 128.0R)) =
  ObExp.log_two_positive ();
  ObExp.exp2_as_exp (difference *. dec_intended_attention_exp2_scale ())

inline_for_extraction noextract
let dec_source_linear_exponent
  (value scaled result:ObF32.t) (score maximum coefficient:real)
  : Lemma
      (requires
        v_approximates value score /\
        v_approximates scaled (maximum *. coefficient) /\
        v_approximates (dec_source_attention_literal ()) coefficient /\
        dec_source_exponent_contract value scaled result)
      (ensures v_approximates result (ObPow.exp2 ((score -. maximum) *. coefficient))) =
  assert (v_approximates result
    (ObPow.exp2 (score *. coefficient -. maximum *. coefficient)))

inline_for_extraction noextract
let dec_source_active_numerator_refines
  (value scaled result:ObF32.t) (score maximum:real)
  : Lemma
      (requires
        v_approximates value score /\
        v_approximates scaled (maximum *. dec_intended_attention_exp2_scale ()) /\
        dec_source_exponent_contract value scaled result)
      (ensures v_approximates result (ObExp.exp ((score -. maximum) *. ObRoot.rsqrt 128.0R))) =
  dec_source_attention_literal_represents_ideal ();
  dec_source_linear_exponent value scaled result score maximum (dec_intended_attention_exp2_scale ());
  dec_source_attention_change_of_base (score -. maximum)

inline_for_extraction noextract
let dec_attention_exponent_refines
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 16) (col:dec_spec_index 64)
  (scaled exponent:F32.t)
  : Lemma
      (requires
        dec_source_qk_representation rotated query key /\
        (forall (mr ar:real).
          dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row) %~ mr /\
          dec_source_attention_literal () %~ ar ==> scaled %~ (mr *. ar)) /\
        dec_source_branch_exponent_contract
          (dec_source_row_score row (dec_source_score_row (dec_source_mma_score_tile rotated head) row) col)
          scaled exponent)
      (ensures dec_bilateral_mma_live_scores_finite rotated head ==>
        exponent %~ dec_attention_numerator rotated query key head row col) =
  let under_domain ()
    : Lemma
        (requires dec_bilateral_mma_live_scores_finite rotated head)
        (ensures exponent %~ dec_attention_numerator rotated query key head row col) =
    if dec_source_live row col then (
      dec_source_mma_score_tile_at rotated head row col;
      dec_source_score_row_at (dec_source_mma_score_tile rotated head) row col;
      dec_bilateral_finite_not_sentinel (Seq.index (dec_source_mma_score_tile rotated head) (row * 8 + col));
      dec_source_mma_tile_represents_dot rotated query key head row col;
      dec_source_attention_literal_represents_ideal ();
      let maximum = dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row) in
      to_real_ok maximum;
      dec_source_active_numerator_refines
        (Seq.index (dec_source_mma_score_tile rotated head) (row * 8 + col)) scaled exponent
        (dec_spec_dot (query (row / 2) (2 * head + row % 2)) (key col head)) (to_real maximum)
    ) else (
      dec_negative_infinity_compares_equal ()
    )
  in
  FStar.Classical.forall_intro (FStar.Classical.move_requires under_domain)

inline_for_extraction noextract
let u_exponent_related (x:lseq BF16.t 32768) (head:Shared.index 8)
  (row:Shared.index 16) (col:Shared.index 64) (scaled exponent:F32.t)
  : Lemma
      (requires
        (forall (mr ar:real).
          dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile x head) row) %~ mr /\
          dec_source_attention_literal () %~ ar ==> scaled %~ (mr *. ar)) /\
        dec_source_branch_exponent_contract
          (dec_source_row_score row (dec_source_score_row (dec_source_mma_score_tile x head) row) col)
          scaled exponent)
      (ensures forall (m:u_attention_meaning).
        u_probability_guard x head m ==>
          exponent %~ dec_attention_numerator x m.u_query m.u_key head row col) =
  let at (m:u_attention_meaning)
    : Lemma
        (requires dec_source_qk_representation x m.u_query m.u_key)
        (ensures dec_bilateral_mma_live_scores_finite x head ==>
          exponent %~ dec_attention_numerator x m.u_query m.u_key head row col) =
    dec_attention_exponent_refines x m.u_query m.u_key head row col scaled exponent
  in C.forall_intro (C.move_requires at)

inline_for_extraction noextract
let dec_source_column_injective
  (p q:dec_spec_index 4) (i j:dec_spec_index 16)
  : Lemma (dec_source_column p i == dec_source_column q j ==> p == q /\ i == j) =
  ()

inline_for_extraction noextract
let dec_probability_write
  (before after:lseq bf16 1024) (row:dec_spec_index 16) (f:dec_spec_vector 64)
  (part:dec_spec_index 4) (step:dec_spec_index 16) (value:bf16) (guard:prop)
  : Lemma
      (requires dec_probability_prefix before row f part step guard /\
        after == Seq.upd before (row * 64 + dec_source_column part step) value /\
        (guard ==> value %~ f (dec_source_column part step)))
      (ensures dec_probability_prefix after row f part (step + 1) guard) =
  let at (p:dec_spec_index 4) (j:dec_spec_index 16)
    : Lemma
        ((p < part \/ (p = part /\ j < step + 1)) ==>
          (guard ==> Seq.index after (row * 64 + dec_source_column p j) %~ f (dec_source_column p j))) =
    dec_source_column_injective p part j step;
    if p = part && j = step then ()
    else assert (Seq.index after (row * 64 + dec_source_column p j) ==
      Seq.index before (row * 64 + dec_source_column p j))
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let u_probability_write (x:lseq BF16.t 32768) (head:Shared.index 8)
  (before after:lseq BF16.t 1024) (row:Shared.index 16)
  (part:Shared.index 4) (step:Shared.index 16) (value:BF16.t)
  : Lemma
      (requires
        (forall (m:u_attention_meaning).
          dec_probability_prefix before row (dec_attention_numerator x m.u_query m.u_key head row)
            part step (u_probability_guard x head m) /\
          (u_probability_guard x head m ==>
            value %~ dec_attention_numerator x m.u_query m.u_key head row
              (dec_source_column part step))) /\
        after == S.upd before (row * 64 + dec_source_column part step) value)
      (ensures forall (m:u_attention_meaning).
        dec_probability_prefix after row (dec_attention_numerator x m.u_query m.u_key head row)
          part (step + 1) (u_probability_guard x head m)) =
  let at (m:u_attention_meaning)
    : Lemma (dec_probability_prefix after row (dec_attention_numerator x m.u_query m.u_key head row)
        part (step + 1) (u_probability_guard x head m)) =
    dec_probability_write before after row (dec_attention_numerator x m.u_query m.u_key head row)
      part step value (u_probability_guard x head m)
  in C.forall_intro at

inline_for_extraction noextract
let u_probability_sum_step (x:lseq BF16.t 32768) (head:Shared.index 8)
  (row:Shared.index 16) (part:Shared.index 4) (step:Shared.index 16)
  (previous exponent added:F32.t)
  : Lemma
      (requires
        (forall (m:u_attention_meaning). u_probability_guard x head m ==>
          previous %~ dec_source_sum_n step
            (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) part) /\
          exponent %~ dec_attention_numerator x m.u_query m.u_key head row (dec_source_column part step)) /\
        (forall (a b:real). previous %~ a /\ exponent %~ b ==> added %~ (a +. b)))
      (ensures forall (m:u_attention_meaning). u_probability_guard x head m ==>
        added %~ dec_source_sum_n (step + 1)
          (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) part)) =
  let at (m:u_attention_meaning)
    : Lemma (u_probability_guard x head m ==>
        added %~ dec_source_sum_n (step + 1)
          (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) part)) =
    dec_source_sum_step (step + 1)
      (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) part)
  in C.forall_intro at

inline_for_extraction noextract
let rec dec_source_sum_zero_tail (n:nat) (cut:nat{cut <= n}) (f:nat -> GTot real)
  : Lemma
      (requires forall (i:dec_spec_index n). cut <= i ==> f i == 0.0R)
      (ensures dec_source_sum_n n f == dec_source_sum_n cut f)
      (decreases n) =
  if cut < n then dec_source_sum_zero_tail (n - 1) cut f

inline_for_extraction noextract
let dec_partition_total (f:dec_spec_vector 64) (part:dec_spec_index 4)
  : Lemma
      (requires forall (i:dec_spec_index 64). 7 <= i ==> f i == 0.0R)
      (ensures dec_source_sum_n 16 (dec_partition_entry f part) == f (part * 2) +. f (part * 2 + 1)) =
  dec_source_sum_zero_tail 16 2 (dec_partition_entry f part);
  dec_source_sum_step 2 (dec_partition_entry f part);
  dec_source_sum_step 1 (dec_partition_entry f part)

inline_for_extraction noextract
let dec_partitions_sum (f:dec_spec_vector 64) (cut:nat{cut <= 7})
  : Lemma
      (requires forall (i:dec_spec_index 64). cut <= i ==> f i == 0.0R)
      (ensures
        (dec_source_sum_n 16 (dec_partition_entry f 0) +. dec_source_sum_n 16 (dec_partition_entry f 1)) +.
        (dec_source_sum_n 16 (dec_partition_entry f 2) +. dec_source_sum_n 16 (dec_partition_entry f 3)) ==
          dec_spec_sum cut f) =
  dec_partition_total f 0;
  dec_partition_total f 1;
  dec_partition_total f 2;
  dec_partition_total f 3;
  let extended (i:nat) : GTot real = if i < 64 then f i else 0.0R in
  dec_source_sum_zero_tail 8 cut extended;
  dec_source_sum_shared cut extended;
  dec_sum_extensional cut extended f;
  dec_source_sum_step 8 extended;
  dec_source_sum_step 7 extended;
  dec_source_sum_step 6 extended;
  dec_source_sum_step 5 extended;
  dec_source_sum_step 4 extended;
  dec_source_sum_step 3 extended;
  dec_source_sum_step 2 extended;
  dec_source_sum_step 1 extended

inline_for_extraction noextract
let dec_attention_partitions_denominator
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 16)
  : Lemma
      (let f = dec_attention_numerator rotated query key head row in
       dec_attention_baseline row +.
         ((dec_source_sum_n 16 (dec_partition_entry f 0) +. dec_source_sum_n 16 (dec_partition_entry f 1)) +.
          (dec_source_sum_n 16 (dec_partition_entry f 2) +. dec_source_sum_n 16 (dec_partition_entry f 3))) ==
         dec_attention_denominator rotated query key head row) =
  let f = dec_attention_numerator rotated query key head row in
  let cut = if row < 14 then row / 2 + 1 else 0 in
  dec_partitions_sum f cut

inline_for_extraction noextract
let dec_attention_combined_refines
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 16) (rescale s0 s1 s2 s3 result:F32.t)
  : Lemma
      (requires
        (dec_bilateral_mma_live_scores_finite rotated head ==>
          rescale %~ dec_attention_baseline row /\
          s0 %~ dec_source_sum_n 16 (dec_partition_entry (dec_attention_numerator rotated query key head row) 0) /\
          s1 %~ dec_source_sum_n 16 (dec_partition_entry (dec_attention_numerator rotated query key head row) 1) /\
          s2 %~ dec_source_sum_n 16 (dec_partition_entry (dec_attention_numerator rotated query key head row) 2) /\
          s3 %~ dec_source_sum_n 16 (dec_partition_entry (dec_attention_numerator rotated query key head row) 3)) /\
        (forall (b a0 a1 a2 a3:real).
          rescale %~ b /\ s0 %~ a0 /\ s1 %~ a1 /\ s2 %~ a2 /\ s3 %~ a3 ==>
            result %~ (b +. ((a0 +. a1) +. (a2 +. a3)))))
      (ensures dec_bilateral_mma_live_scores_finite rotated head ==>
        result %~ dec_attention_denominator rotated query key head row) =
  dec_attention_partitions_denominator rotated query key head row

inline_for_extraction noextract
let u_probability_combine (x:lseq BF16.t 32768) (head:Shared.index 8)
  (row:Shared.index 16) (rescale s0 s1 s2 s3 result:F32.t)
  : Lemma
      (requires
        (forall (m:u_attention_meaning). u_probability_guard x head m ==>
          rescale %~ dec_attention_baseline row /\
          s0 %~ dec_source_sum_n 16 (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) 0) /\
          s1 %~ dec_source_sum_n 16 (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) 1) /\
          s2 %~ dec_source_sum_n 16 (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) 2) /\
          s3 %~ dec_source_sum_n 16 (dec_partition_entry (dec_attention_numerator x m.u_query m.u_key head row) 3)) /\
        (forall (b a0 a1 a2 a3:real).
          rescale %~ b /\ s0 %~ a0 /\ s1 %~ a1 /\ s2 %~ a2 /\ s3 %~ a3 ==>
            result %~ (b +. ((a0 +. a1) +. (a2 +. a3)))))
      (ensures forall (m:u_attention_meaning). u_probability_guard x head m ==>
        result %~ dec_attention_denominator x m.u_query m.u_key head row) =
  let at (m:u_attention_meaning)
    : Lemma
        (requires dec_source_qk_representation x m.u_query m.u_key)
        (ensures dec_bilateral_mma_live_scores_finite x head ==>
          result %~ dec_attention_denominator x m.u_query m.u_key head row) =
    dec_attention_combined_refines x m.u_query m.u_key head row rescale s0 s1 s2 s3 result
  in C.forall_intro (C.move_requires at)

inline_for_extraction noextract
let dec_source_column_inverse (col:dec_spec_index 64)
  : Lemma (dec_source_column ((col % 8) / 2) ((col / 8) * 2 + col % 2) == col) =
  ()

inline_for_extraction noextract
let dec_probability_prefix_done (ys:lseq bf16 1024) (row:dec_spec_index 16)
  (f:dec_spec_vector 64) (guard:prop)
  : Lemma
      (requires dec_probability_prefix ys row f 4 0 guard)
      (ensures guard ==> dec_bf16_segment_rep ys (row * 64) 64 f) =
  let at (col:dec_spec_index 64)
    : Lemma (guard ==> Seq.index ys (row * 64 + col) %~ f col) =
    dec_source_column_inverse col
  in FStar.Classical.forall_intro at

inline_for_extraction noextract
let u_probability_row_finish (x:lseq BF16.t 32768) (head:Shared.index 8)
  (ps:lseq BF16.t 1024) (row:Shared.index 16)
  : Lemma
      (requires forall (m:u_attention_meaning).
        dec_probability_prefix ps row (dec_attention_numerator x m.u_query m.u_key head row)
          4 0 (u_probability_guard x head m))
      (ensures forall (m:u_attention_meaning). u_probability_guard x head m ==>
        dec_bf16_segment_rep ps (row * 64) 64 (dec_attention_numerator x m.u_query m.u_key head row)) =
  let at (m:u_attention_meaning)
    : Lemma (u_probability_guard x head m ==>
        dec_bf16_segment_rep ps (row * 64) 64 (dec_attention_numerator x m.u_query m.u_key head row)) =
    dec_probability_prefix_done ps row (dec_attention_numerator x m.u_query m.u_key head row)
      (u_probability_guard x head m)
  in C.forall_intro at

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

inline_for_extraction noextract
let dec_matrix_row_progress (#rows #cols:pos)
  (before after:lseq bf16 (rows * cols))
  (model:dec_spec_matrix rows cols) (row:dec_spec_index rows) (guard:prop)
  : Lemma
      (requires
        dec_segment_frame before after (row * cols) cols /\
        (guard ==> dec_matrix_prefix before model row /\
          dec_bf16_segment_rep after (row * cols) cols (model row)))
      (ensures guard ==> dec_matrix_prefix after model (row + 1)) =
  let at (t:dec_spec_index rows{t < row + 1}) (c:dec_spec_index cols)
    : Lemma (guard ==> Seq.index after (t * cols + c) %~ model t c) =
    if t <> row then
      assert (Seq.index after (t * cols + c) == Seq.index before (t * cols + c))
  in FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let dec_denominator_row_progress (before after:lseq F32.t 16)
  (row:dec_spec_index 16) (model:dec_spec_vector 16) (guard:prop)
  : Lemma
      (requires
        dec_segment_frame before after row 1 /\
        (guard ==> Seq.index after row %~ model row) /\
        (forall (r:dec_spec_index 16{r < row}). guard ==> Seq.index before r %~ model r))
      (ensures forall (r:dec_spec_index 16{r < row + 1}). guard ==> Seq.index after r %~ model r) =
  let at (r:dec_spec_index 16{r < row + 1})
    : Lemma (guard ==> Seq.index after r %~ model r) =
    if r <> row then assert (Seq.index after r == Seq.index before r)
  in FStar.Classical.forall_intro at

inline_for_extraction noextract
let u_probability_rows_advance (x:lseq BF16.t 32768) (head:Shared.index 8)
  (before after:lseq BF16.t 1024) (old_den new_den:lseq F32.t 16) (row:Shared.index 16)
  : Lemma
      (requires dec_segment_frame before after (row * 64) 64 /\ dec_segment_frame old_den new_den row 1 /\
        (forall (m:u_attention_meaning). u_probability_guard x head m ==>
          dec_matrix_prefix before (dec_attention_numerator x m.u_query m.u_key head) row /\
          (forall (i:Shared.index 16{i < row}).
            S.index old_den i %~ dec_attention_denominator x m.u_query m.u_key head i) /\
          dec_bf16_segment_rep after (row * 64) 64 (dec_attention_numerator x m.u_query m.u_key head row) /\
          S.index new_den row %~ dec_attention_denominator x m.u_query m.u_key head row))
      (ensures forall (m:u_attention_meaning). u_probability_guard x head m ==>
        dec_matrix_prefix after (dec_attention_numerator x m.u_query m.u_key head) (row + 1) /\
        (forall (i:Shared.index 16{i < row + 1}).
          S.index new_den i %~ dec_attention_denominator x m.u_query m.u_key head i)) =
  let at (m:u_attention_meaning)
    : Lemma (u_probability_guard x head m ==>
        dec_matrix_prefix after (dec_attention_numerator x m.u_query m.u_key head) (row + 1) /\
        (forall (i:Shared.index 16{i < row + 1}).
          S.index new_den i %~ dec_attention_denominator x m.u_query m.u_key head i)) =
    dec_matrix_row_progress before after (dec_attention_numerator x m.u_query m.u_key head) row
      (u_probability_guard x head m);
    dec_denominator_row_progress old_den new_den row (dec_attention_denominator x m.u_query m.u_key head)
      (u_probability_guard x head m)
  in C.forall_intro at

inline_for_extraction noextract
fn u_probability_rows (tile:larray F32.t 128) (probability:larray BF16.t 1024)
  (denominator:larray F32.t 16)
  (#x:erased (lseq BF16.t 32768)) (#head:erased (Shared.index 8))
  (#old_p:erased (lseq BF16.t 1024)) (#old_d:erased (lseq F32.t 16))
  preserves gpu ** tile |-> dec_source_mma_score_tile x head
  requires probability |-> old_p ** denominator |-> old_d
  ensures exists* (pv:lseq BF16.t 1024) (dv:lseq F32.t 16).
    probability |-> pv ** denominator |-> dv ** pure (u_probability_contract x head pv dv)

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

inline_for_extraction noextract
let dec_attention_head_advance (ys:lseq bf16 16384) (model:dec_spec_matrix 8 2048)
  (head:dec_spec_index 8)
  : Lemma
      (dec_attention_output_prefix ys model head 128 <==> dec_attention_output_prefix ys model (head + 1) 0) = ()

inline_for_extraction noextract
let u_attention_head_advance (x:lseq BF16.t 32768) (ys:lseq BF16.t 16384)
  (head:Shared.index 8)
  : Lemma (u_attention_prefix x ys head 128 <==> u_attention_prefix x ys (head + 1) 0) =
  let at (m:u_attention_meaning)
    : Lemma (dec_attention_output_prefix ys (u_attention_model m) head 128 <==>
        dec_attention_output_prefix ys (u_attention_model m) (head + 1) 0) =
    dec_attention_head_advance ys (u_attention_model m) head
  in C.forall_intro at

inline_for_extraction noextract
ghost
fn u_advance_attention_output (output:larray BF16.t 16384) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768)) (head:szlt 8)
  requires when_ (SZ.v tid == 0) (u_attention_owned output x head 128)
  ensures when_ (SZ.v tid == 0) (u_attention_owned output x (head + 1) 0)

inline_for_extraction noextract
let dec_source_normalization_chain
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
fn dec_source_normalize_value (numerator denominator:ObF32.t)
  preserves gpu
  returns result:bf16
  ensures pure (
    forall (n:real) (d:real{d =!= 0.0R}).
      v_approximates numerator n /\ v_approximates denominator d ==>
      v_approximates result (n /. d))

inline_for_extraction noextract
let dec_attention_tile_prefix
  (ys:lseq bf16 16384) (model:dec_spec_matrix 8 2048)
  (head:dec_spec_index 8) (block:nat{block < 128 /\ block % 8 = 0}) (done:nat) : prop =
  forall (row:dec_spec_index 14) (col:dec_spec_index 8).
    row * 8 + col < done ==>
      Seq.index ys ((row / 2) * 2048 + head * 256 + row % 2 * 128 + block + col) %~
        model (row / 2) (head * 256 + row % 2 * 128 + block + col)

inline_for_extraction noextract
let dec_padded_real (dec_spec_values:dec_spec_vector 7) (key:dec_spec_index 64) : GTot real =
  if key < 7 then dec_spec_values key else 0.0R

inline_for_extraction noextract
let dec_source_shifted_prefix_numerators
  (token:dec_spec_index 7) (scores:dec_spec_vector (token + 1)) (shift:real)
  : GTot (dec_spec_vector 7) =
  fun key -> if key <= token then ObExp.exp (scores key -. shift) else 0.0R

inline_for_extraction noextract
let dec_real_add_div (a b:real) (denominator:real{denominator >. 0.0R})
  : Lemma (a /. denominator +. b /. denominator == (a +. b) /. denominator)
  = ()

inline_for_extraction noextract
let rec dec_sum_divide (n:nat) (dec_spec_values:dec_spec_vector n) (denominator:real{denominator >. 0.0R})
  : Lemma
      (dec_spec_sum n (fun i -> dec_spec_values i /. denominator) ==
        dec_spec_sum n dec_spec_values /. denominator)
      (decreases n)
  =
  let divided : dec_spec_vector n = fun i -> dec_spec_values i /. denominator in
  if n > 0 then (
    dec_sum_divide (n - 1) dec_spec_values denominator;
    dec_sum_extensional (n - 1) divided (fun i -> dec_spec_values i /. denominator);
    assert (dec_spec_sum (n - 1) divided ==
      dec_spec_sum (n - 1) dec_spec_values /. denominator);
    ObSum.sum_pop_right 0 n dec_spec_values;
    ObSum.sum_pop_right 0 n divided;
    dec_real_add_div (dec_spec_sum (n - 1) dec_spec_values) (dec_spec_values (n - 1)) denominator
  )

inline_for_extraction noextract
let dec_real_div_cancel (a:real) (b:real{b >. 0.0R}) (c:real{c >. 0.0R})
  : Lemma ((a /. c) /. (b /. c) == a /. b)
  = ()

inline_for_extraction noextract
let dec_shared_softmax_at (n:pos) (scores:dec_spec_vector n) (i:dec_spec_index n)
  : Lemma
      (requires dec_spec_sum n (fun j -> ObExp.exp (scores j)) >. 0.0R)
      (ensures dec_spec_softmax scores i ==
        ObExp.exp (scores i) /. dec_spec_sum n (fun j -> ObExp.exp (scores j)))
  =
  dec_spec_sum_positive n (fun j -> ObExp.exp (scores j));
  assert_norm (dec_spec_softmax scores i ==
    ObExp.exp (scores i) /. dec_spec_sum n (fun j -> ObExp.exp (scores j)))

inline_for_extraction noextract
let dec_prefix_normalize_shifted
  (token:dec_spec_index 7) (scores:dec_spec_vector (token + 1))
  (shift:real) (key:dec_spec_index 7{key <= token})
  : Lemma
      (let shifted : dec_spec_vector (token + 1) =
         fun i -> ObExp.exp (scores i -. shift) in
       dec_spec_sum (token + 1) shifted >. 0.0R /\
       shifted key /. dec_spec_sum (token + 1) shifted ==
         dec_spec_prefix_softmax token scores key)
  =
  let original : dec_spec_vector (token + 1) = fun i -> ObExp.exp (scores i) in
  let shifted : dec_spec_vector (token + 1) = fun i -> ObExp.exp (scores i -. shift) in
  let divided : dec_spec_vector (token + 1) = fun i -> original i /. ObExp.exp shift in
  let at (i:dec_spec_index (token + 1))
    : Lemma (shifted i == divided i) = ObExp.exp_sub (scores i) shift
  in
  FStar.Classical.forall_intro at;
  dec_sum_extensional (token + 1) shifted divided;
  dec_sum_divide (token + 1) original (ObExp.exp shift);
  dec_spec_sum_positive (token + 1) original;
  dec_spec_sum_positive (token + 1) shifted;
  dec_shared_softmax_at (token + 1) scores key;
  at key;
  assert (dec_spec_sum (token + 1) shifted ==
    dec_spec_sum (token + 1) original /. ObExp.exp shift);
  dec_real_div_cancel (original key) (dec_spec_sum (token + 1) original) (ObExp.exp shift)

inline_for_extraction noextract
let rec dec_sum_zero_tail (n:nat) (cut:nat{cut <= n}) (dec_spec_values:dec_spec_vector n)
  : Lemma
      (requires forall (i:dec_spec_index n). cut <= i ==> dec_spec_values i == 0.0R)
      (ensures dec_spec_sum n dec_spec_values == dec_spec_sum cut dec_spec_values)
      (decreases n)
  =
  if cut < n then (
    dec_sum_zero_tail (n - 1) cut dec_spec_values;
    assert (dec_spec_values (n - 1) == 0.0R);
    ObSum.sum_pop_right 0 n dec_spec_values;
    assert (dec_spec_sum n dec_spec_values ==
      dec_spec_sum (n - 1) dec_spec_values +. dec_spec_values (n - 1))
  ) else assert (cut == n)

inline_for_extraction noextract
let dec_dense_value_equals_prefix
  (token:dec_spec_index 7) (probabilities dec_spec_values:dec_spec_vector 7)
  : Lemma
      (requires forall (key:dec_spec_index 7).
        token < key ==> probabilities key == 0.0R)
      (ensures
        dec_spec_sum 64 (fun key ->
          dec_padded_real probabilities key *. dec_padded_real dec_spec_values key) ==
        dec_spec_sum (token + 1) (fun key -> probabilities key *. dec_spec_values key))
  =
  let term : dec_spec_vector 64 =
    fun key -> dec_padded_real probabilities key *. dec_padded_real dec_spec_values key in
  let tail (key:dec_spec_index 64)
    : Lemma (token + 1 <= key ==> term key == 0.0R) = ()
  in
  FStar.Classical.forall_intro tail;
  dec_sum_zero_tail 64 (token + 1) term;
  dec_sum_extensional (token + 1) term (fun key -> probabilities key *. dec_spec_values key)

inline_for_extraction noextract
let dec_real_mul_div (a b:real) (denominator:real{denominator >. 0.0R})
  : Lemma ((a *. b) /. denominator == (a /. denominator) *. b) =
  ()

inline_for_extraction noextract
let dec_normalized_dense_prefix
  (token:dec_spec_index 7) (scores:dec_spec_vector (token + 1))
  (dec_spec_values:dec_spec_vector 7) (shift:real)
  : Lemma
      (let numerators = dec_source_shifted_prefix_numerators token scores shift in
       let denominator = dec_spec_sum (token + 1) numerators in
       denominator >. 0.0R /\
       dec_spec_sum 64 (fun key ->
         dec_padded_real numerators key *. dec_padded_real dec_spec_values key) /. denominator ==
       dec_spec_sum (token + 1) (fun key ->
         dec_spec_prefix_softmax token scores key *. dec_spec_values key)) =
  let numerators = dec_source_shifted_prefix_numerators token scores shift in
  let denominator = dec_spec_sum (token + 1) numerators in
  let terms : dec_spec_vector (token + 1) = fun key -> numerators key *. dec_spec_values key in
  dec_spec_sum_positive (token + 1) numerators;
  dec_dense_value_equals_prefix token numerators dec_spec_values;
  dec_sum_extensional (token + 1) (fun key -> numerators key *. dec_spec_values key) terms;
  dec_sum_divide (token + 1) terms denominator;
  let at (key:dec_spec_index (token + 1))
    : Lemma (terms key /. denominator ==
        dec_spec_prefix_softmax token scores key *. dec_spec_values key) =
    dec_prefix_normalize_shifted token scores shift key;
    dec_sum_extensional (token + 1) numerators (fun i -> ObExp.exp (scores i -. shift));
    assert (numerators key /. denominator == dec_spec_prefix_softmax token scores key);
    dec_real_mul_div (numerators key) (dec_spec_values key) denominator
  in
  FStar.Classical.forall_intro at;
  dec_sum_extensional (token + 1)
    (fun key -> terms key /. denominator)
    (fun key -> dec_spec_prefix_softmax token scores key *. dec_spec_values key);
  dec_sum_extensional (token + 1)
    (fun i -> terms i /. denominator)
    (fun key -> terms key /. denominator);
  assert (dec_spec_sum (token + 1) terms /. denominator ==
    dec_spec_sum (token + 1) (fun key ->
      dec_spec_prefix_softmax token scores key *. dec_spec_values key));
  dec_sum_extensional 64
    (fun key -> dec_padded_real numerators key *. dec_padded_real dec_spec_values key)
    (fun key ->
      dec_padded_real (dec_source_shifted_prefix_numerators token scores shift) key *.
      dec_padded_real dec_spec_values key)

inline_for_extraction noextract
let dec_attention_value_vector (dec_spec_values:dec_spec_heads 8) (head:dec_spec_index 8) (lane:dec_spec_index 128)
  (key:dec_spec_index 64) : GTot real =
  if key < 7 then dec_spec_values key head lane else 0.0R

inline_for_extraction noextract
let dec_attention_value_sum
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key:dec_spec_heads 8) (dec_spec_values:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 14) (lane:dec_spec_index 128) : GTot real =
  dec_spec_dot (dec_attention_numerator rotated query key head row) (dec_attention_value_vector dec_spec_values head lane)

inline_for_extraction noextract
let dec_real_shift_distribute (x y z:real)
  : Lemma ((x -. y) *. z == x *. z -. y *. z) = ()

inline_for_extraction noextract
let dec_dot_pointwise_equal (#n:pos) (x y u v:dec_spec_vector n)
  : Lemma
      (requires forall (i:dec_spec_index n). x i == u i /\ y i == v i)
      (ensures dec_spec_dot x y == dec_spec_dot u v) =
  dec_sum_extensional n (fun i -> x i *. y i) (fun i -> u i *. v i)

inline_for_extraction noextract
let dec_softmax_pointwise_equal (#n:pos) (x y:dec_spec_vector n) (i:dec_spec_index n)
  : Lemma
      (requires forall (j:dec_spec_index n). x j == y j)
      (ensures dec_spec_softmax x i == dec_spec_softmax y i) =
  dec_spec_sum_positive n (fun j -> ObExp.exp (x j));
  dec_spec_sum_positive n (fun j -> ObExp.exp (y j));
  dec_sum_extensional n (fun j -> ObExp.exp (x j)) (fun j -> ObExp.exp (y j));
  dec_shared_softmax_at n x i;
  dec_shared_softmax_at n y i

inline_for_extraction noextract
let dec_prefix_softmax_pointwise_equal
  (row:dec_spec_index 7) (x y:dec_spec_vector (row + 1)) (i:dec_spec_index 7)
  : Lemma
      (requires forall (j:dec_spec_index (row + 1)). x j == y j)
      (ensures dec_spec_prefix_softmax row x i == dec_spec_prefix_softmax row y i) =
  if i <= row then dec_softmax_pointwise_equal x y i

inline_for_extraction noextract
let dec_attention_model_sum_at (dec_spec_data:dec_spec_attention_data)
  (query:dec_spec_heads 16) (key dec_spec_values:dec_spec_heads 8)
  (token:dec_spec_index 7) (head:dec_spec_index 16) (lane:dec_spec_index 128)
  : Lemma
      (dec_attention_model dec_spec_data query key dec_spec_values token (head * 128 + lane) ==
        dec_spec_sum (token + 1) (fun k ->
          dec_spec_attention_probabilities dec_spec_data query (dec_spec_repeat_kv key) token head k *.
            dec_spec_repeat_kv dec_spec_values k head lane)) =
  reveal_opaque (`%dec_attention_model) (dec_attention_model dec_spec_data query key dec_spec_values);
  assert ((head * 128 + lane) / 128 == head /\ (head * 128 + lane) % 128 == lane);
  assert (dec_spec_attention_values
    (dec_spec_attention_probabilities dec_spec_data query (dec_spec_repeat_kv key))
    (dec_spec_repeat_kv dec_spec_values) token head lane ==
      dec_spec_sum (token + 1) (fun k ->
        dec_spec_attention_probabilities dec_spec_data query (dec_spec_repeat_kv key) token head k *.
          dec_spec_repeat_kv dec_spec_values k head lane)) by
    (FStar.Tactics.V2.norm [delta_only [`%dec_spec_attention_values]];
     FStar.Tactics.V2.trivial ())

inline_for_extraction noextract
let dec_attention_value_sum_expand
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key dec_spec_values:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 14) (lane:dec_spec_index 128)
  : Lemma
      (dec_attention_value_sum rotated query key dec_spec_values head row lane ==
        dec_spec_sum 64 (fun k ->
          dec_attention_numerator rotated query key head row k *. dec_attention_value_vector dec_spec_values head lane k)) =
  assert_norm (dec_attention_value_sum rotated query key dec_spec_values head row lane ==
    dec_spec_sum 64 (fun k ->
      dec_attention_numerator rotated query key head row k *. dec_attention_value_vector dec_spec_values head lane k))

inline_for_extraction noextract
let dec_attention_normalization_refines
  (rotated:dec_bf16_values 32768) (dec_spec_data:dec_spec_attention_data)
  (query:dec_spec_heads 16) (key dec_spec_values:dec_spec_heads 8)
  (head:dec_spec_index 8) (row:dec_spec_index 14) (lane:dec_spec_index 128)
  : Lemma
      (dec_attention_denominator rotated query key head row >. 0.0R /\
       dec_attention_value_sum rotated query key dec_spec_values head row lane /.
         dec_attention_denominator rotated query key head row ==
       dec_attention_model dec_spec_data query key dec_spec_values (row / 2) (head * 256 + row % 2 * 128 + lane)) =
  let token = row / 2 in
  let query_head = 2 * head + row % 2 in
  let maximum = to_real (dec_source_maximum row (dec_source_score_row (dec_source_mma_score_tile rotated head) row)) in
  let scale = ObRoot.rsqrt 128.0R in
  let scores:dec_spec_vector (token + 1) =
    fun k -> dec_spec_dot (query token query_head) (key k head) *. scale in
  let shift = maximum *. scale in
  let v:dec_spec_vector 7 = fun k -> dec_spec_values k head lane in
  let p = dec_source_shifted_prefix_numerators token scores shift in
  let shared_scores:dec_spec_vector (token + 1) = fun k ->
    dec_spec_dot (query token query_head) (dec_spec_repeat_kv key k query_head) *. ObRoot.rsqrt 128.0R in
  assert (query_head / 2 == head);
  let score_at (k:dec_spec_index (token + 1))
    : Lemma (shared_scores k == scores k) =
    dec_dot_pointwise_equal (query token query_head) (dec_spec_repeat_kv key k query_head)
      (query token query_head) (key k head)
  in
  FStar.Classical.forall_intro score_at;
  let probability_at (k:dec_spec_index (token + 1))
    : Lemma
        (dec_spec_attention_probabilities dec_spec_data query (dec_spec_repeat_kv key) token query_head k ==
          dec_spec_prefix_softmax token scores k) =
    assert (dec_spec_attention_probabilities dec_spec_data query (dec_spec_repeat_kv key) token query_head k ==
      dec_spec_prefix_softmax token
        (fun source -> dec_spec_dot (query token query_head)
          (dec_spec_repeat_kv key source query_head) *. ObRoot.rsqrt 128.0R) k) by
      (FStar.Tactics.V2.norm [delta_only [`%dec_spec_attention_probabilities]];
       FStar.Tactics.V2.trivial ());
    dec_prefix_softmax_pointwise_equal token
      (fun source -> dec_spec_dot (query token query_head)
        (dec_spec_repeat_kv key source query_head) *. ObRoot.rsqrt 128.0R)
      scores k
  in FStar.Classical.forall_intro probability_at;
  let numerator_at (k:dec_spec_index 64)
    : Lemma (dec_attention_numerator rotated query key head row k == dec_padded_real p k) =
    if k <= token then
      dec_real_shift_distribute (dec_spec_dot (query token query_head) (key k head)) maximum scale
  in
  FStar.Classical.forall_intro numerator_at;
  dec_normalized_dense_prefix token scores v shift;
  dec_sum_extensional 64
    (fun k -> dec_padded_real p k *. dec_padded_real v k)
    (fun k -> dec_padded_real (dec_source_shifted_prefix_numerators token scores shift) k *. dec_padded_real v k);
  assert (dec_spec_sum 64 (fun k -> dec_padded_real p k *. dec_padded_real v k) /.
    dec_spec_sum (token + 1) p ==
      dec_spec_sum (token + 1) (fun k -> dec_spec_prefix_softmax token scores k *. v k));
  dec_sum_extensional (token + 1) (dec_attention_numerator rotated query key head row) p;
  dec_sum_extensional 64
    (fun k -> dec_attention_numerator rotated query key head row k *. dec_attention_value_vector dec_spec_values head lane k)
    (fun k -> dec_padded_real p k *. dec_padded_real v k);
  dec_sum_extensional (token + 1)
    (fun k -> dec_spec_prefix_softmax token scores k *. v k)
    (fun k -> dec_spec_attention_probabilities dec_spec_data query (dec_spec_repeat_kv key) token query_head k *.
      dec_spec_repeat_kv dec_spec_values k query_head lane);
  dec_attention_model_sum_at dec_spec_data query key dec_spec_values token query_head lane;
  assert (query_head * 128 + lane == head * 256 + row % 2 * 128 + lane);
  assert (dec_spec_sum (token + 1) (fun k -> dec_spec_prefix_softmax token scores k *. v k) ==
    dec_attention_model dec_spec_data query key dec_spec_values token (head * 256 + row % 2 * 128 + lane));
  dec_attention_value_sum_expand rotated query key dec_spec_values head row lane;
  assert (dec_attention_denominator rotated query key head row == dec_spec_sum (token + 1) p);
  assert (dec_attention_value_sum rotated query key dec_spec_values head row lane ==
    dec_spec_sum 64 (fun k -> dec_padded_real p k *. dec_padded_real v k))

inline_for_extraction noextract
let dec_dense_real_term (width:pos) (a b:dec_spec_vector width) (i:nat) : GTot real =
  if i < width then a i *. b i else 0.0R

inline_for_extraction noextract
let rec dec_dense_mma_cell_approx
  (width:nat{dec_dense_width width})
  (a:lseq bf16 (16 * width)) (b:lseq bf16 (8 * width))
  (row:dec_spec_index 16) (col:dec_spec_index 8)
  (ar br:dec_spec_vector width) (steps:nat{steps <= width / 16})
  : Lemma
      (requires
        (forall (i:dec_spec_index width). Seq.index a (row * width + i) %~ ar i) /\
        (forall (i:dec_spec_index width). Seq.index b (col * width + i) %~ br i))
      (ensures Chest.acc2 (dec_dense_mma_fold width a b steps) row col %~
        dec_source_sum_n (16 * steps) (dec_dense_real_term width ar br))
      (decreases steps) =
  if steps = 0 then ()
  else (
    let k = steps - 1 in
    let ra = Seq.init_ghost 16 (fun i -> ar (16 * k + i)) in
    let rb = Seq.init_ghost 16 (fun i -> br (16 * k + i)) in
    dec_dense_mma_cell_approx width a b row col ar br k;
    let left_at (i:dec_spec_index 16)
      : Lemma
          (Chest.acc2 (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 k) row i %~
            Seq.index ra i) =
      dec_dense_left_operand_at width a k row i
    in FStar.Classical.forall_intro left_at;
    let right_at (i:dec_spec_index 16)
      : Lemma
          (Chest.acc2 (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 k 0) i col %~
            Seq.index rb i) =
      dec_dense_right_operand_at width b k i col
    in FStar.Classical.forall_intro right_at;
    dec_source_mma_cell_approx
      (ematrix_subtile (Tensor.from_seq (rm 16 width) a) 16 16 0 k)
      (ematrix_subtile (Tensor.from_seq (cm width 8) b) 16 8 k 0)
      (dec_dense_mma_fold width a b k)
      row col ra rb
      (dec_source_sum_n (16 * k) (dec_dense_real_term width ar br));
    dec_source_sum_ext 16 (dec_source_dot16 ra rb)
      (fun i -> dec_dense_real_term width ar br (16 * k + i));
    dec_source_sum_append (16 * k) 16 (dec_dense_real_term width ar br);
    dec_dense_mma_step width a b k
  )

inline_for_extraction noextract
let dec_dense_mma_dot
  (width:nat{dec_dense_width width})
  (a:lseq bf16 (16 * width)) (b:lseq bf16 (8 * width))
  (row:dec_spec_index 16) (col:dec_spec_index 8) (ar br:dec_spec_vector width)
  : Lemma
      (requires
        (forall (i:dec_spec_index width). Seq.index a (row * width + i) %~ ar i) /\
        (forall (i:dec_spec_index width). Seq.index b (col * width + i) %~ br i))
      (ensures Chest.acc2 (dec_dense_mma_fold width a b (width / 16)) row col %~ dec_spec_dot ar br) =
  dec_dense_mma_cell_approx width a b row col ar br (width / 16);
  dec_source_sum_shared width (dec_dense_real_term width ar br);
  dec_sum_extensional width (dec_dense_real_term width ar br) (fun i -> ar i *. br i)

inline_for_extraction noextract
let dec_attention_value_mma_refines
  (rotated:dec_bf16_values 32768) (query:dec_spec_heads 16) (key dec_spec_values:dec_spec_heads 8)
  (head:dec_spec_index 8) (ps:lseq bf16 1024) (ds:lseq F32.t 16)
  (block:nat{block < 128 /\ block % 8 = 0})
  (row:dec_spec_index 14) (col:dec_spec_index 8)
  : Lemma
      (requires dec_attention_probability_contract rotated query key head ps ds /\
        dec_source_values_representation rotated dec_spec_values)
      (ensures dec_bilateral_mma_live_scores_finite rotated head ==>
        Chest.acc2 (dec_dense_mma_fold 64 ps (dec_packed_values rotated head block) 4) row col %~
          dec_attention_value_sum rotated query key dec_spec_values head row (block + col)) =
  let under_domain ()
    : Lemma
        (requires dec_bilateral_mma_live_scores_finite rotated head)
        (ensures Chest.acc2 (dec_dense_mma_fold 64 ps (dec_packed_values rotated head block) 4) row col %~
          dec_attention_value_sum rotated query key dec_spec_values head row (block + col)) =
    dec_dense_mma_dot 64 ps (dec_packed_values rotated head block) row col
      (dec_attention_numerator rotated query key head row)
      (dec_attention_value_vector dec_spec_values head (block + col))
  in FStar.Classical.move_requires under_domain ()

inline_for_extraction noextract
let dec_attention_output_refines
  (rotated:dec_bf16_values 32768) (dec_spec_data:dec_spec_attention_data)
  (query:dec_spec_heads 16) (key dec_spec_values:dec_spec_heads 8)
  (head:dec_spec_index 8) (ps:lseq bf16 1024) (ds:lseq F32.t 16)
  (block:nat{block < 128 /\ block % 8 = 0})
  (row:dec_spec_index 14) (col:dec_spec_index 8) (result:bf16)
  : Lemma
      (requires dec_attention_probability_contract rotated query key head ps ds /\
        dec_source_values_representation rotated dec_spec_values /\
        (forall (n:real) (d:real{d =!= 0.0R}).
          Chest.acc2 (dec_dense_mma_fold 64 ps (dec_packed_values rotated head block) 4) row col %~ n /\
          v_approximates (Seq.index ds row) d ==> result %~ (n /. d)))
      (ensures dec_bilateral_mma_live_scores_finite rotated head ==>
        result %~ dec_attention_model dec_spec_data query key dec_spec_values (row / 2) (head * 256 + row % 2 * 128 + block + col)) =
  dec_attention_value_mma_refines rotated query key dec_spec_values head ps ds block row col;
  dec_attention_normalization_refines rotated dec_spec_data query key dec_spec_values head row (block + col)

inline_for_extraction noextract
let u_attention_value_related (x:lseq BF16.t 32768) (head:Shared.index 8)
  (ps:lseq BF16.t 1024) (ds:lseq F32.t 16)
  (block:nat{block < 128 /\ block % 8 = 0})
  (row:Shared.index 14) (col:Shared.index 8) (value:BF16.t)
  : Lemma
      (requires u_probability_contract x head ps ds /\
        (forall (nr:real) (dr:real{dr =!= 0.0R}).
          Chest.acc2 (dec_dense_mma_fold 64 ps (dec_packed_values x head block) 4) row col %~ nr /\
          v_approximates (S.index ds row) dr ==> value %~ (nr /. dr)))
      (ensures forall (m:u_attention_meaning). u_attention_guard x m ==>
        value %~ u_attention_model m (row / 2) (head * 256 + row % 2 * 128 + block + col)) =
  let at (m:u_attention_meaning)
    : Lemma
        (requires dec_source_qk_representation x m.u_query m.u_key /\
          dec_source_values_representation x m.u_value)
        (ensures dec_attention_domain x ==>
          value %~ u_attention_model m (row / 2) (head * 256 + row % 2 * 128 + block + col)) =
    dec_attention_output_refines x m.u_data m.u_query m.u_key m.u_value head ps ds block row col value
  in C.forall_intro (C.move_requires at)

inline_for_extraction noextract
let dec_attention_tile_write
  (before after:lseq bf16 16384) (model:dec_spec_matrix 8 2048)
  (head:dec_spec_index 8) (block:nat{block < 128 /\ block % 8 = 0})
  (row:dec_spec_index 14) (col:dec_spec_index 8) (value:bf16) (guard:prop)
  : Lemma
      (requires
        (guard ==> dec_attention_output_prefix before model head block /\
          dec_attention_tile_prefix before model head block (row * 8 + col) /\
          value %~ model (row / 2) (head * 256 + row % 2 * 128 + block + col)) /\
        after == Seq.upd before
          ((row / 2) * 2048 + head * 256 + row % 2 * 128 + block + col) value)
      (ensures guard ==> dec_attention_output_prefix after model head block /\
        dec_attention_tile_prefix after model head block (row * 8 + col + 1)) =
  let tile_at (r:dec_spec_index 14) (c:dec_spec_index 8)
    : Lemma
        (r * 8 + c < row * 8 + col + 1 ==>
          (guard ==> Seq.index after ((r / 2) * 2048 + head * 256 + r % 2 * 128 + block + c) %~
            model (r / 2) (head * 256 + r % 2 * 128 + block + c))) =
    if r = row && c = col then ()
    else assert ((r / 2) * 2048 + head * 256 + r % 2 * 128 + block + c <>
      (row / 2) * 2048 + head * 256 + row % 2 * 128 + block + col)
  in FStar.Classical.forall_intro_2 tile_at

inline_for_extraction noextract
let u_attention_write (x:lseq BF16.t 32768) (before after:lseq BF16.t 16384)
  (head:Shared.index 8) (block:nat{block < 128 /\ block % 8 = 0})
  (row:Shared.index 14) (col:Shared.index 8) (value:BF16.t)
  : Lemma
      (requires u_attention_prefix x before head block /\
        (forall (m:u_attention_meaning). u_attention_guard x m ==>
          dec_attention_tile_prefix before (u_attention_model m) head block (row * 8 + col) /\
          value %~ u_attention_model m (row / 2) (head * 256 + row % 2 * 128 + block + col)) /\
        after == S.upd before ((row / 2) * 2048 + head * 256 + row % 2 * 128 + block + col) value)
      (ensures u_attention_prefix x after head block /\
        (forall (m:u_attention_meaning). u_attention_guard x m ==>
          dec_attention_tile_prefix after (u_attention_model m) head block (row * 8 + col + 1))) =
  let at (m:u_attention_meaning)
    : Lemma (u_attention_guard x m ==>
        dec_attention_output_prefix after (u_attention_model m) head block /\
        dec_attention_tile_prefix after (u_attention_model m) head block (row * 8 + col + 1)) =
    dec_attention_tile_write before after (u_attention_model m) head block row col value
      (u_attention_guard x m)
  in C.forall_intro at

inline_for_extraction noextract
let dec_attention_tile_done
  (ys:lseq bf16 16384) (model:dec_spec_matrix 8 2048)
  (head:dec_spec_index 8) (block:nat{block < 128 /\ block % 8 = 0}) (guard:prop)
  : Lemma
      (requires guard ==> dec_attention_output_prefix ys model head block /\
        dec_attention_tile_prefix ys model head block 112)
      (ensures guard ==> dec_attention_output_prefix ys model head (block + 8)) =
  let at (row:dec_spec_index 7) (channel:dec_spec_index 2048)
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
let u_attention_tile_done (x:lseq BF16.t 32768) (ys:lseq BF16.t 16384)
  (head:Shared.index 8) (block:nat{block < 128 /\ block % 8 = 0})
  : Lemma
      (requires u_attention_prefix x ys head block /\
        (forall (m:u_attention_meaning). u_attention_guard x m ==>
          dec_attention_tile_prefix ys (u_attention_model m) head block 112))
      (ensures u_attention_prefix x ys head (block + 8)) =
  let at (m:u_attention_meaning)
    : Lemma (u_attention_guard x m ==>
        dec_attention_output_prefix ys (u_attention_model m) head (block + 8)) =
    dec_attention_tile_done ys (u_attention_model m) head block (u_attention_guard x m)
  in C.forall_intro at

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

inline_for_extraction noextract
fn u_attention_one_head (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (shared:source_attention_shared) (head:szlt 8) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768)) (#fraction:perm)
  preserves gpu ** thread_id 32 tid ** input |-> Frac fraction x
  requires source_attention_available shared tid **
    when_ (SZ.v tid == 0) (u_attention_owned output x head 0)
  ensures source_attention_available shared tid **
    when_ (SZ.v tid == 0) (u_attention_owned output x (head + 1) 0)

inline_for_extraction noextract
ghost
fn u_attention_output_begin (output:larray BF16.t 16384) (tid:szlt 32)
  (#old:erased (lseq BF16.t 16384)) (#x:erased (lseq BF16.t 32768))
  requires when_ (SZ.v tid == 0) (output |-> old)
  ensures when_ (SZ.v tid == 0) (u_attention_owned output x 0 0)

inline_for_extraction noextract
let source_attention_padding_model (data:dec_spec_attention_data)
  (query:dec_spec_heads 16) (key vals:dec_spec_heads 8)
  : Lemma (forall (c:Shared.index 2048).
      dec_attention_model data query key vals 7 c == 0.0R) =
  reveal_opaque (`%dec_attention_model) (dec_attention_model data query key vals)

inline_for_extraction noextract
let u_attention_padding_related (x:lseq BF16.t 32768)
  (original after:lseq BF16.t 16384)
  : Lemma
      (requires u_attention_prefix x original 8 0 /\
        dec_segment_frame original after 14336 2048 /\
        (forall (i:Shared.index 2048). S.index after (14336 + i) == (F.zero #BF16.t)))
      (ensures forall (m:u_attention_meaning).
        u_attention_guard x m ==> dec_matrix_rep after (u_attention_model m)) =
  let at (m:u_attention_meaning)
    : Lemma (u_attention_guard x m ==> dec_matrix_rep after (u_attention_model m)) =
    source_attention_padding_model m.u_data m.u_query m.u_key m.u_value
  in C.forall_intro at

inline_for_extraction noextract
fn u_pad_attention_output (output:larray BF16.t 16384) (#x:erased (lseq BF16.t 32768))
  preserves gpu
  requires u_attention_owned output x 8 0
  ensures u_attention_final output x

inline_for_extraction noextract
fn u_owner_attention_padding (output:larray BF16.t 16384) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768))
  preserves gpu
  requires when_ (SZ.v tid == 0) (u_attention_owned output x 8 0)
  ensures when_ (SZ.v tid == 0) (u_attention_final output x)

inline_for_extraction noextract
fn u_attention_body (input:larray BF16.t 32768) (output:larray BF16.t 16384)
  (shared:source_attention_shared) (tid:szlt 32)
  (#x:erased (lseq BF16.t 32768)) (#old:erased (lseq BF16.t 16384)) (#fraction:perm)
  preserves gpu ** thread_id 32 tid ** input |-> Frac fraction x
  requires source_attention_available shared tid ** when_ (SZ.v tid == 0) (output |-> old)
  ensures source_attention_available shared tid ** when_ (SZ.v tid == 0) (u_attention_final output x)

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

inline_for_extraction noextract
let opt_send_when (test:prop) (p:slprop) (sender:is_send_across block_of p)
  : is_send_across block_of (when_ test p) =
  if t2b test then sender else solve

inline_for_extraction noextract
let opt_block_array_live_send (#a:Type0)
  (buffer:PA.array a{SH.is_block_array buffer})
  : is_send_across block_of (dec_array_live buffer) =
  let at (v:S.seq a) : is_send_across block_of (buffer |-> v) =
    SH.is_send_across_block_array buffer #1.0R #v in
  is_send_across_exists _ #at

inline_for_extraction noextract
let opt_attention_shared_block_visible (sh:SH.c_shmems opt_attention_shared_desc)
  : Lemma
      (requires SH.c_shmems_inv sh)
      (ensures SH.is_block_array (opt_attention_shared sh).score_tile /\
        SH.is_block_array (opt_attention_shared sh).probability /\
        SH.is_block_array (opt_attention_shared sh).denominator) = ()

inline_for_extraction noextract
let opt_attention_available_send (sh:SH.c_shmems opt_attention_shared_desc)
  (_:squash (SH.c_shmems_inv sh)) (tid:nat)
  : is_send_across block_of (source_attention_available (opt_attention_shared sh) tid) =
  opt_attention_shared_block_visible sh;
  let shared = opt_attention_shared sh in
  let tile_at (v:lseq F32.t 128)
    : is_send_across block_of (shared.score_tile |-> Frac (1.0R /. 32.0R) v) =
    SH.is_send_across_block_array shared.score_tile #(1.0R /. 32.0R) #v in
  let tile_send : is_send_across block_of
      (exists* (v:lseq F32.t 128). shared.score_tile |-> Frac (1.0R /. 32.0R) v) =
    is_send_across_exists _ #tile_at in
  let p_send = opt_send_when (tid == 0) (dec_array_live shared.probability)
    (opt_block_array_live_send shared.probability) in
  let d_send = opt_send_when (tid == 0) (dec_array_live shared.denominator)
    (opt_block_array_live_send shared.denominator) in
  is_send_across_star _ _ #tile_send #(is_send_across_star _ _ #p_send #d_send)

inline_for_extraction noextract
let u_attention_kpre_send
  (input:larray BF16.t 32768{KA.is_global_array input})
  (output:larray BF16.t 16384{KA.is_global_array output})
  (#x:erased (lseq BF16.t 32768)) (#old:erased (lseq BF16.t 16384)) (#fraction:perm)
  (sh:SH.c_shmems opt_attention_shared_desc) (pf:squash (SH.c_shmems_inv sh)) (tid:nat)
  : is_send_across block_of (u_attention_kpre input output (opt_attention_shared sh) x old fraction tid) =
  let input_send:is_send_across block_of (input |-> Frac (fraction /. 32.0R) x) = solve in
  let shared_send = opt_attention_available_send sh pf tid in
  let output_send:is_send_across block_of (when_ (tid == 0) (output |-> old)) = solve in
  is_send_across_star _ _ #input_send #(is_send_across_star _ _ #shared_send #output_send)

inline_for_extraction noextract
let u_attention_kpost_send
  (input:larray BF16.t 32768{KA.is_global_array input})
  (output:larray BF16.t 16384{KA.is_global_array output})
  (#x:erased (lseq BF16.t 32768)) (#fraction:perm)
  (sh:SH.c_shmems opt_attention_shared_desc) (pf:squash (SH.c_shmems_inv sh)) (tid:nat)
  : is_send_across block_of (u_attention_kpost input output (opt_attention_shared sh) x fraction tid) =
  let input_send:is_send_across block_of (input |-> Frac (fraction /. 32.0R) x) = solve in
  let shared_send = opt_attention_available_send sh pf tid in
  let output_send:is_send_across block_of (when_ (tid == 0) (u_attention_final output x)) = solve in
  is_send_across_star _ _ #input_send #(is_send_across_star _ _ #shared_send #output_send)

inline_for_extraction noextract
val u_attention_descriptor
  (input:larray BF16.t 32768{KA.is_global_array input})
  (output:larray BF16.t 16384{KA.is_global_array output})
  (#x:erased (lseq BF16.t 32768)) (#old:erased (lseq BF16.t 16384)) (#fraction:perm)
  : k:KS.kernel_desc (u_attention_pre input output x old fraction) (u_attention_post input output x fraction){
    SZ.v k.nblk == 1 /\ k.nthr == 32sz /\ k.shmems_desc == opt_attention_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 32sz /\ k.barrier_count bid == 0)}

inline_for_extraction noextract
let u_attention_stage_frame
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (a:opt_block_addresses) (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) : slprop =
  PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
  a.p_embedding |-> v.v_embedding ** a.p_rope |-> v.v_rope ** a.p_normalized0 |-> v.v_normalized0 **
  a.p_qkv |-> v.v_qkv ** a.p_residual |-> v.v_residual ** a.p_postnorm |-> v.v_postnorm **
  a.p_gateup |-> v.v_gateup ** a.p_activated |-> v.v_activated **
  a.p_hidden |-> v.v_hidden ** a.p_normalized1 |-> v.v_normalized1 **
  pure (u_block_stage weights prompt v 4)

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

inline_for_extraction noextract
let u_attention_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased opt_block_values)
  : k:KS.kernel_desc
      (u_stage_resources weights_address prompt_address a weights prompt v 4)
      (u_pending_stage weights_address prompt_address a weights prompt 5){
    SZ.v k.nblk == 1 /\ k.nthr == 32sz /\ k.shmems_desc == opt_attention_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 32sz /\ k.barrier_count bid == 0)} =
  let code = u_attention_descriptor a.p_rotated a.p_attention
    #(hide (reveal v).v_rotated) #(hide (reveal v).v_attention) #1.0R in
  let extra = u_attention_stage_frame weights_address prompt_address a weights prompt v in
  let framed = opt_descriptor_with_frame code extra in
  opt_descriptor_refine
    #(u_attention_pre a.p_rotated a.p_attention (reveal v).v_rotated (reveal v).v_attention 1.0R ** extra)
    #(u_attention_post a.p_rotated a.p_attention (reveal v).v_rotated 1.0R ** extra)
    #(u_stage_resources weights_address prompt_address a weights prompt v 4)
    #(u_pending_stage weights_address prompt_address a weights prompt 5) framed
    (u_attention_before weights_address prompt_address a #weights #prompt #v)
    (u_attention_after weights_address prompt_address a #weights #prompt #v)

inline_for_extraction noextract
let u_attention_stage
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family opt_block_values
      (fun v -> u_stage_resources weights_address prompt_address a weights prompt v 4)
      (fun _ -> u_pending_stage weights_address prompt_address a weights prompt 5) = {
    pending_threads = 32sz;
    pending_shared = opt_attention_shared_desc;
    pending_code = u_attention_stage_descriptor weights_address prompt_address a #weights #prompt
  }

inline_for_extraction noextract
let opt_once_order (#a:Type0) : FStar.Preorder.preorder (option a) =
  fun before after -> before == None \/ before == after

inline_for_extraction noextract
let opt_observation (a:Type0) = Observed.mref (opt_once_order #a)

inline_for_extraction noextract
let opt_pending_pre (#a:Type0) (run:opt_observation a) (pre:a -> slprop) : slprop =
  Observed.pts_to run #1.0R None ** (exists* (x:a). pre x)

inline_for_extraction noextract
let opt_seen (#a:Type0) (run:opt_observation a) (value:erased a) : slprop =
  Observed.snapshot #(option a) #(opt_once_order #a) run
    (Some (FStar.Ghost.reveal #a value))

inline_for_extraction noextract
let opt_observed_thread (#a:Type0) (run:opt_observation a) (pred:a -> slprop) : slprop =
  exists* (x:a). opt_seen #a run (FStar.Ghost.hide #a x) ** pred x

inline_for_extraction noextract
let opt_pending_frame (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (run:opt_observation a) : slprop =
  exists* (x:a). Observed.pts_to run #1.0R (Some x) ** (family.pending_code (hide x)).frame

inline_for_extraction noextract
let opt_pending_block_pre (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (run:opt_observation a) : slprop =
  exists* (x:a). opt_seen #a run (hide x) ** (family.pending_code (hide x)).block_pre 0

inline_for_extraction noextract
let opt_pending_block_post (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (run:opt_observation a) : slprop =
  exists* (x:a). opt_seen #a run (hide x) ** (family.pending_code (hide x)).block_post 0

inline_for_extraction noextract
let opt_pending_block_frame (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (run:opt_observation a)
  (sh:SH.c_shmems family.pending_shared) : slprop =
  exists* (x:a). opt_seen #a run (hide x) ** (family.pending_code (hide x)).block_frame sh 0

inline_for_extraction noextract
let opt_pending_source_pre (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (x:a)
  (sh:SH.c_shmems family.pending_shared) (tid:natlt family.pending_threads) : slprop =
  (family.pending_code (hide x)).kpre sh 0 tid

inline_for_extraction noextract
let opt_pending_source_post (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (x:a)
  (sh:SH.c_shmems family.pending_shared) (tid:natlt family.pending_threads) : slprop =
  (family.pending_code (hide x)).kpost sh 0 tid

inline_for_extraction noextract
ghost
fn opt_pending_setup
  (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (#run:erased (opt_observation a)) ()
  norewrite
  requires opt_pending_pre run pre
  ensures (forall+ (_:natlt 1). opt_pending_block_pre family run) ** opt_pending_frame family run

inline_for_extraction noextract
ghost
fn opt_pending_teardown
  (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (#run:erased (opt_observation a)) ()
  norewrite
  requires (forall+ (_:natlt 1). opt_pending_block_post family run) ** opt_pending_frame family run
  ensures exists* (x:a). post x

inline_for_extraction noextract
let opt_copies (#a:Type0) (run:opt_observation a) (value:erased a) (count:nat) : slprop =
  forall+ (_:natlt count). opt_seen #a run value

inline_for_extraction noextract
ghost
fn opt_snapshot_copies (#a:Type0) (run:opt_observation a) (#observed:erased a)
  (count:nat)
  norewrite
  preserves opt_seen #a run observed
  ensures opt_copies #a run observed count

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

inline_for_extraction noextract
ghost
fn opt_observed_align (#a:Type0) (run:opt_observation a) (pred:a -> slprop)
  (#x:erased a)
  requires opt_seen #a run x ** opt_observed_thread run pred
  ensures pred x

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

inline_for_extraction noextract
let opt_pending_descriptor
  (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (#run:erased (opt_observation a))
  : KS.kernel_desc (opt_pending_pre run pre) (exists* (x:a). post x) =
  {
    nblk = 1sz; nthr = family.pending_threads; shmems_desc = family.pending_shared;
    kpre = (fun sh _ tid -> opt_observed_thread run (fun x -> opt_pending_source_pre family x sh tid));
    kpost = (fun sh _ tid -> opt_observed_thread run (fun x -> opt_pending_source_post family x sh tid));
    barrier_contract = (fun _ _ -> Barrier.empty_contract family.pending_threads);
    barrier_count = (fun _ -> 0);
    barrier_ok = (fun _ _ -> Barrier.empty_barrier_transform family.pending_threads);
    f = opt_pending_device family #run;
    frame = opt_pending_frame family run;
    block_pre = (fun _ -> opt_pending_block_pre family run);
    block_post = (fun _ -> opt_pending_block_post family run);
    setup = opt_pending_setup family #run;
    teardown = opt_pending_teardown family #run;
    block_frame = (fun sh _ -> opt_pending_block_frame family run sh);
    block_setup = opt_pending_block_setup family #run;
    block_teardown = opt_pending_block_teardown family #run;
    block_pre_sendable = (fun _ ->
      is_send_across_exists
        (fun x -> opt_seen #a run (hide x) ** (family.pending_code (hide x)).block_pre 0)
        #(fun x -> is_send_across_star _ _ #solve #((family.pending_code (hide x)).block_pre_sendable 0)));
    block_post_sendable = (fun _ ->
      is_send_across_exists
        (fun x -> opt_seen #a run (hide x) ** (family.pending_code (hide x)).block_post 0)
        #(fun x -> is_send_across_star _ _ #solve #((family.pending_code (hide x)).block_post_sendable 0)));
    kpre_sendable = (fun sh pf _ tid ->
      is_send_across_exists
        (fun x -> opt_seen #a run (hide x) ** opt_pending_source_pre family x sh tid)
        #(fun x -> is_send_across_star _ _ #solve
          #((family.pending_code (hide x)).kpre_sendable sh pf 0 tid)));
    kpost_sendable = (fun sh pf _ tid ->
      is_send_across_exists
        (fun x -> opt_seen #a run (hide x) ** opt_pending_source_post family x sh tid)
        #(fun x -> is_send_across_star _ _ #solve
          #((family.pending_code (hide x)).kpost_sendable sh pf 0 tid)))
  }

inline_for_extraction noextract
ghost
fn opt_pending_join_input (#a:Type0) (#run:erased (opt_observation a))
  (pre:a -> slprop) ()
  requires on gpu_loc (Observed.pts_to run #1.0R None) **
    on gpu_loc (exists* (x:a). pre x)
  ensures on gpu_loc (opt_pending_pre run pre)

inline_for_extraction noextract
fn opt_launch_pending (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (stream:ST.stream_t) (#epoch:E.epoch_t)
  preserves cpu ** ST.stream_live stream
  requires E.epoch_live stream epoch **
    P.pledge0 (E.epoch_done stream epoch) (on gpu_loc (exists* (x:a). pre x))
  ensures E.epoch_live stream (E.epoch_next epoch) **
    P.pledge0 (E.epoch_done stream (E.epoch_next epoch)) (on gpu_loc (exists* (x:a). post x))

inline_for_extraction noextract
ghost
fn u_on_exists_drop (#a:Type0) (post:slprop) ()
  requires on gpu_loc (exists* (_:a). post)
  ensures on gpu_loc post

inline_for_extraction noextract
fn u_launch_const_post (#a:Type0) (#pre:a -> slprop) (#post:slprop)
  (family:opt_pending_family a pre (fun _ -> post))
  (stream:ST.stream_t) (#epoch:E.epoch_t)
  preserves cpu ** ST.stream_live stream
  requires E.epoch_live stream epoch **
    P.pledge0 (E.epoch_done stream epoch) (on gpu_loc (exists* (x:a). pre x))
  ensures E.epoch_live stream (E.epoch_next epoch) **
    P.pledge0 (E.epoch_done stream (E.epoch_next epoch)) (on gpu_loc post)

inline_for_extraction noextract
let u_frame_code (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (extra:slprop) (v:erased a)
  : k:KS.kernel_desc (pre v ** extra) (post v ** extra){
    SZ.v k.nblk == 1 /\ k.nthr == family.pending_threads /\ k.shmems_desc == family.pending_shared /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract family.pending_threads /\ k.barrier_count bid == 0)} =
  opt_descriptor_with_frame (family.pending_code v) extra

inline_for_extraction noextract
let u_frame_family (#a:Type0) (#pre #post:a -> slprop)
  (family:opt_pending_family a pre post) (extra:slprop)
  : opt_pending_family a (fun v -> pre v ** extra) (fun v -> post v ** extra) = {
    pending_threads = family.pending_threads;
    pending_shared = family.pending_shared;
    pending_code = u_frame_code family extra
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

inline_for_extraction noextract
let u_rms_step (slot:nat{slot < 3}) : nat =
  if slot = 0 then 1 else if slot = 1 then 6 else 10

inline_for_extraction noextract
let u_rms_offset (slot:nat{slot < 3}) : nat =
  if slot = 0 then 155583488 else if slot = 1 then 155584512 else 171314432

inline_for_extraction noextract
let source_bf16_region_initialized (#n:pos)
  (before after:lseq (option BF16.t) n)
  (start:nat) (width:pos{start + width <= n})
  (model:dec_spec_vector width) (guard:prop)
  : Lemma
      (requires source_bf16_region #n before after start width width model guard /\
        (forall (i:Shared.index n). Some? (S.index before i)))
      (ensures forall (i:Shared.index n). Some? (S.index after i)) =
  let at (i:Shared.index n)
    : Lemma (Some? (S.index after i)) =
    if i < start || start + width <= i then
      assert (S.index after i == S.index before i)
    else (
      assert (0 <= i - start /\ i - start < width);
      let written (j:Shared.index width)
        : Lemma (Some? (S.index after (start + j))) = () in
      written (i - start))
  in C.forall_intro at

inline_for_extraction noextract
let u_rms_row_result (#n #nw #ny:pos) (x:lseq BF16.t n) (w:lseq BF16.t nw)
  (old after:lseq BF16.t ny) (start weight_start output_start:nat)
  (width:pos{start + width <= n /\ weight_start + width <= nw /\ output_start + width <= ny})
  : prop =
  dec_segment_frame old after output_start width /\
  (forall (r gamma:dec_spec_vector width).
    u_rms_guard x w start weight_start width r gamma ==>
      dec_bf16_segment_rep after output_start width (dec_spec_rms_norm gamma r))

inline_for_extraction noextract
let source_bf16_region_transfer (#n:pos)
  (before after:lseq BF16.t n) (bm am:lseq (option BF16.t) n)
  (start:nat) (width:pos{start + width <= n})
  (model:dec_spec_vector width) (guard:prop)
  : Lemma
      (requires source_bf16_region #n bm am start width width model guard /\
        (forall (i:Shared.index n).
          S.index bm i == Some (S.index before i) /\ S.index am i == Some (S.index after i)))
      (ensures dec_segment_frame #BF16.t #n before after start width /\
        (guard ==> dec_bf16_segment_rep #n after start width model)) =
  let frame (i:Shared.index n)
    : Lemma ((i < start \/ start + width <= i) ==>
        S.index after i == S.index before i) = ()
  in C.forall_intro frame;
  let at (j:Shared.index width)
    : Lemma (guard ==> S.index after (start + j) %~ model j) = ()
  in C.forall_intro at

inline_for_extraction noextract
let u_rms_transfer (#n #nw #ny:pos) (x:lseq BF16.t n) (w:lseq BF16.t nw)
  (old after:lseq BF16.t ny) (before_mask after_mask:lseq (option BF16.t) ny)
  (start weight_start output_start:nat)
  (width:pos{start + width <= n /\ weight_start + width <= nw /\ output_start + width <= ny})
  : Lemma
      (requires
        (forall (r gamma:dec_spec_vector width).
          source_bf16_region before_mask after_mask output_start width width
            (dec_spec_rms_norm gamma r) (u_rms_guard x w start weight_start width r gamma)) /\
        (forall (i:Shared.index ny).
          S.index before_mask i == Some (S.index old i) /\
          S.index after_mask i == Some (S.index after i)))
      (ensures u_rms_row_result x w old after start weight_start output_start width) =
  let at (r gamma:dec_spec_vector width)
    : Lemma (dec_segment_frame old after output_start width /\
        (u_rms_guard x w start weight_start width r gamma ==>
          dec_bf16_segment_rep after output_start width (dec_spec_rms_norm gamma r))) =
    source_bf16_region_transfer old after before_mask after_mask output_start width
      (dec_spec_rms_norm gamma r) (u_rms_guard x w start weight_start width r gamma)
  in
  at (fun _ -> 0.0R) (fun _ -> 0.0R);
  C.forall_intro_2 at

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

inline_for_extraction noextract
let u_rms_row_advance (#rows #nw:pos)
  (x:lseq BF16.t (rows * 1024)) (w:lseq BF16.t nw)
  (before after:lseq BF16.t (rows * 1024))
  (offset:nat{offset + 1024 <= nw}) (row:Shared.index rows)
  : Lemma
      (requires u_rms_prefix x w before offset row /\
        u_rms_row_result x w before after (row * 1024) offset (row * 1024) 1024)
      (ensures u_rms_prefix x w after offset (row + 1)) =
  let at (xr:dec_spec_matrix rows 1024) (gamma:dec_spec_vector 1024)
    : Lemma (dec_matrix_rep x xr /\ dec_bf16_segment_rep w offset 1024 gamma ==>
        dec_matrix_prefix after (fun r -> dec_spec_rms_norm gamma (xr r)) (row + 1)) =
    dec_matrix_row_progress before after (fun r -> dec_spec_rms_norm gamma (xr r)) row
      (dec_matrix_rep x xr /\ dec_bf16_segment_rep w offset 1024 gamma)
  in C.forall_intro_2 at

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

inline_for_extraction noextract
let u_rms_input (a:opt_block_addresses) (slot:nat{slot < 3}) : larray BF16.t 8192 =
  if slot = 0 then a.p_embedding else if slot = 1 then a.p_residual else a.p_hidden

inline_for_extraction noextract
let u_rms_output (a:opt_block_addresses) (slot:nat{slot < 3}) : larray BF16.t 8192 =
  if slot = 0 then a.p_normalized0 else if slot = 1 then a.p_postnorm else a.p_normalized1

inline_for_extraction noextract
let u_rms_input_values (v:opt_block_values) (slot:nat{slot < 3}) : lseq BF16.t 8192 =
  if slot = 0 then v.v_embedding else if slot = 1 then v.v_residual else v.v_hidden

inline_for_extraction noextract
let u_rms_output_values (v:opt_block_values) (slot:nat{slot < 3}) : lseq BF16.t 8192 =
  if slot = 0 then v.v_normalized0 else if slot = 1 then v.v_postnorm else v.v_normalized1

inline_for_extraction noextract
let u_rms_updated (v:opt_block_values) (slot:nat{slot < 3}) (ys:lseq BF16.t 8192)
  : opt_block_values =
  if slot = 0 then {v with v_normalized0 = ys}
  else if slot = 1 then {v with v_postnorm = ys}
  else {v with v_normalized1 = ys}

inline_for_extraction noextract
let u_rms_stage_advance (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) (slot:nat{slot < 3}) (ys:lseq BF16.t 8192)
  : Lemma
      (requires u_block_stage weights prompt v (u_rms_step slot) /\
        u_rms_prefix #8 #opt_weight_count (u_rms_input_values v slot) weights ys (u_rms_offset slot) 8)
      (ensures u_block_stage weights prompt (u_rms_updated v slot ys) (u_rms_step slot + 1)) =
  if slot = 0 then () else if slot = 1 then () else ()

inline_for_extraction noextract
let u_rms_frame (a:opt_block_addresses) (v:opt_block_values) (slot:nat{slot < 3}) : slprop =
  a.p_rope |-> v.v_rope ** a.p_qkv |-> v.v_qkv ** a.p_rotated |-> v.v_rotated **
  a.p_attention |-> v.v_attention ** a.p_gateup |-> v.v_gateup ** a.p_activated |-> v.v_activated **
  (if slot = 0 then
    a.p_residual |-> v.v_residual ** a.p_postnorm |-> v.v_postnorm **
    a.p_hidden |-> v.v_hidden ** a.p_normalized1 |-> v.v_normalized1
   else if slot = 1 then
    a.p_embedding |-> v.v_embedding ** a.p_normalized0 |-> v.v_normalized0 **
    a.p_hidden |-> v.v_hidden ** a.p_normalized1 |-> v.v_normalized1
   else
    a.p_embedding |-> v.v_embedding ** a.p_normalized0 |-> v.v_normalized0 **
    a.p_residual |-> v.v_residual ** a.p_postnorm |-> v.v_postnorm)

inline_for_extraction noextract
ghost
fn u_rms_storage_open (a:opt_block_addresses) (#v:erased opt_block_values) (slot:nat{slot < 3})
  requires opt_block_storage a v
  ensures u_rms_input a slot |-> u_rms_input_values v slot **
    u_rms_output a slot |-> u_rms_output_values v slot ** u_rms_frame a v slot

inline_for_extraction noextract
ghost
fn u_rms_storage_close (a:opt_block_addresses) (#v:erased opt_block_values)
  (slot:nat{slot < 3}) (#ys:erased (lseq BF16.t 8192))
  requires u_rms_input a slot |-> u_rms_input_values v slot **
    u_rms_output a slot |-> ys ** u_rms_frame a v slot
  ensures opt_block_storage a (u_rms_updated v slot ys)

inline_for_extraction noextract
fn u_rms_stage_body (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7) (a:opt_block_addresses)
  (slot:nat{slot < 3}) (offset:sz{SZ.v offset == u_rms_offset slot})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased opt_block_values) ()
  preserves gpu
  requires u_stage_resources weights_address prompt_address a weights prompt v (u_rms_step slot)
  ensures u_pending_stage weights_address prompt_address a weights prompt (u_rms_step slot + 1)

inline_for_extraction noextract
let u_rms_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (slot:nat{slot < 3}) (offset:sz{SZ.v offset == u_rms_offset slot})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased opt_block_values)
  : k:KS.kernel_desc
      (u_stage_resources weights_address prompt_address a weights prompt v (u_rms_step slot))
      (u_pending_stage weights_address prompt_address a weights prompt (u_rms_step slot + 1)){
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)} =
  let raw:KS.kernel_desc_1_1
      (u_stage_resources weights_address prompt_address a weights prompt v (u_rms_step slot))
      (u_pending_stage weights_address prompt_address a weights prompt (u_rms_step slot + 1)) = {
    f = u_rms_stage_body weights_address prompt_address a slot offset #weights #prompt #v;
    full_pre_sendable = solve;
    full_post_sendable = solve
  } in opt_scalar_descriptor
    #(u_stage_resources weights_address prompt_address a weights prompt v (u_rms_step slot))
    #(u_pending_stage weights_address prompt_address a weights prompt (u_rms_step slot + 1)) raw

inline_for_extraction noextract
let u_rms_stage
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (a:opt_block_addresses{opt_block_global a})
  (slot:nat{slot < 3}) (offset:sz{SZ.v offset == u_rms_offset slot})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family opt_block_values
      (fun v -> u_stage_resources weights_address prompt_address a weights prompt v (u_rms_step slot))
      (fun _ -> u_pending_stage weights_address prompt_address a weights prompt (u_rms_step slot + 1)) = {
    pending_threads = 1sz;
    pending_shared = opt_scalar_shared_desc;
    pending_code = u_rms_stage_descriptor weights_address prompt_address a slot offset #weights #prompt
  }

inline_for_extraction noextract
ghost
fn u_queue_step (stream:ST.stream_t) (#start:E.epoch_t) (index:nat) (post:slprop)
  requires E.epoch_live stream (E.epoch_next (start + index)) **
    P.pledge0 (E.epoch_done stream (E.epoch_next (start + index))) post
  ensures E.epoch_live stream (start + (index + 1)) **
    P.pledge0 (E.epoch_done stream (start + (index + 1))) post

inline_for_extraction noextract
let u_source_offsets ()
  : Lemma (
    opt_layer0_start + 2048 == 155585536 /\
    opt_layer0_start + 6293504 == 161876992 /\
    opt_layer0_start + 4196352 == 159779840 /\
    opt_layer0_start + 6293760 == 161877248 /\
    opt_layer0_start + 12585216 == 168168704 /\
    opt_layer0_start + dec_weight_elements == 171314432) = ()

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

inline_for_extraction noextract
fn opt_buffer_data_ptr
  (stream:Stream.stream_t) (buffers:list (RT.tensor BF16.t))
  (index:nat{index < List.length buffers})
  preserves KL.cpu ** RuntimeHelpers.all_owners stream buffers
  returns address:PA.array BF16.t
  ensures pure (address == RT.data (List.index buffers index))


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
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn u_read_block_addresses (stream:ST.stream_t)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  preserves cpu ** opt_allocated_owners stream buffers
  returns a:opt_block_addresses
  ensures pure (a == u_allocated_addresses buffers /\ opt_block_global a)
#pop-options

inline_for_extraction noextract
let u_runtime_contents_placeless (#a:Type0) (value:RT.tensor a) : placeless (RT.contents value) = solve

inline_for_extraction noextract
let rec u_all_contents_placeless (values:list (RT.tensor BF16.t))
  : Tot (placeless (RH.all_contents values)) (decreases values) =
  match values with
  | [] -> placeless_emp
  | value::rest -> placeless_star (RT.contents value) (RH.all_contents rest)
      #(u_runtime_contents_placeless value) #(u_all_contents_placeless rest)

inline_for_extraction noextract
let u_remaining_placeless (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  : placeless (u_remaining_storage buffers) =
  placeless_star (RT.contents buffers.token_buffer) (RH.all_contents (u_remaining_tensors buffers))
    #(u_runtime_contents_placeless buffers.token_buffer) #(u_all_contents_placeless (u_remaining_tensors buffers))
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
#pop-options

inline_for_extraction noextract
let u_actual_parameters (weights:lseq BF16.t opt_weight_count) : GTot (Shared.parameters BF16.t) =
  let w = opt_layer0_values weights in {
    input_layernorm = (fun i -> S.index w i);
    post_attention_layernorm = (fun i -> S.index w (1024 + i));
    q_proj = (fun row col -> S.index w (2048 + row * 1024 + col));
    k_proj = (fun row col -> S.index w (2048 + (2048 + row) * 1024 + col));
    v_proj = (fun row col -> S.index w (2048 + (3072 + row) * 1024 + col));
    o_proj = (fun row col -> S.index w (4196352 + row * 2048 + col));
    q_norm = (fun i -> S.index w (6293504 + i));
    k_norm = (fun i -> S.index w (6293632 + i));
    gate_proj = (fun row col -> S.index w (6293760 + row * 1024 + col));
    up_proj = (fun row col -> S.index w (6293760 + (3072 + row) * 1024 + col));
    down_proj = (fun row col -> S.index w (12585216 + row * 3072 + col))
  }

inline_for_extraction noextract
let u_block_numerical (weights:lseq BF16.t opt_weight_count) (v:opt_block_values) : prop =
  (dec_attention_domain v.v_rotated ==>
    Shared.decoder_relation (u_actual_parameters weights) (u_live_matrix #1024 v.v_embedding)
      (u_actual_rotary v.v_rope) (u_native_layer v)) /\
  Shared.final_norm_relation (u_next_gamma_values weights)
    (u_live_matrix #1024 v.v_hidden) (u_live_matrix #1024 v.v_normalized1)

inline_for_extraction noextract
let u_first_block_observation (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) : GTot prop =
  u_block_stage weights prompt v 11 /\
  u_block_numerical weights v /\
  Shared.cache_length (u_first_block_cache v 0) == 7 /\
  (forall (other:Shared.index 28{other > 0}). u_first_block_cache v other == Shared.Empty)

inline_for_extraction noextract
let u_after_first_block (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) : slprop =
  on gpu_loc (
    PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    (exists* (v:opt_block_values).
      opt_block_storage (u_allocated_addresses buffers) v **
      pure (u_first_block_observation weights prompt v))) **
  u_remaining_storage buffers

inline_for_extraction noextract
let u_rope_frame_at (x:lseq BF16.t 32768) (weights:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (ys:lseq BF16.t 32768)
  (row:Shared.index 8) (col:Shared.index 4096{row = 7 \/ col % 512 >= 384})
  : Lemma
      (requires u_rope_prefix x weights rt ys 168)
      (ensures S.index ys (row * 4096 + col) == S.index x (row * 4096 + col)) = ()

inline_for_extraction noextract
let u_rope_value_unchanged (x:lseq BF16.t 32768) (weights:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (ys:lseq BF16.t 32768)
  (row:Shared.index 8) (head:Shared.index 8) (lane:Shared.index 128)
  : Lemma
      (requires u_rope_prefix x weights rt ys 168)
      (ensures S.index ys (row * 4096 + head * 512 + 384 + lane) ==
        S.index x (row * 4096 + head * 512 + 384 + lane)) =
  Kuiper.Math.div_mod_of_mul_add 512 head (384 + lane);
  u_rope_frame_at x weights rt ys row (head * 512 + 384 + lane)

inline_for_extraction noextract
let u_source_score_domain (x:lseq BF16.t 32768)
  : Lemma (source_block_score_domain x <==> dec_attention_domain x) =
  let at (head:Shared.index 8)
    : Lemma (source_score_prefix x head 8 == dec_source_mma_scores x head 8) =
    source_score_fold_related x head 8
  in C.forall_intro at

inline_for_extraction noextract
let u_source_parameters_related (weights:lseq BF16.t opt_weight_count) (w:Shared.parameters real)
  : Lemma
      (requires Shared.parameters_approximates (source_block_parameters weights) w)
      (ensures Shared.parameters_approximates (u_actual_parameters weights) w) = ()

inline_for_extraction noextract
let u_source_rotary_cell (component:Shared.index 2) (row:Shared.index 4096) (col:Shared.index 128)
  : Lemma (emb_rotary_value 64 row component col == source_block_rotary_cell component row col) = ()

inline_for_extraction noextract
let u_public_block_observation (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values)
  : Lemma
      (requires u_first_block_observation weights prompt v)
      (ensures source_first_block_observation weights prompt v) =
  u_source_score_domain v.v_rotated;
  let rotary (component:Shared.index 2) (row:Shared.index 4096) (col:Shared.index 128)
    : Lemma (S.index v.v_rope ((component * 4096 + row) * 128 + col) ==
        source_block_rotary_cell component row col) =
    u_source_rotary_cell component row col
  in C.forall_intro_3 rotary;
  let values (row:Shared.index 8) (head:Shared.index 8) (lane:Shared.index 128)
    : Lemma (S.index v.v_rotated (row * 4096 + head * 512 + 384 + lane) ==
        S.index v.v_qkv (row * 4096 + head * 512 + 384 + lane)) =
    u_rope_value_unchanged v.v_qkv (opt_layer0_values weights) v.v_rope v.v_rotated row head lane
  in C.forall_intro_3 values;
  let padding (col:Shared.index 4096)
    : Lemma (S.index v.v_rotated (28672 + col) == S.index v.v_qkv (28672 + col)) =
    u_rope_frame_at v.v_qkv (opt_layer0_values weights) v.v_rope v.v_rotated 7 col
  in C.forall_intro padding;
  let layer (w:Shared.parameters real) (x:Shared.hidden real) (d:Shared.rotary_tables real)
    : Lemma
        (requires source_block_score_domain v.v_rotated /\
          Shared.parameters_approximates (source_block_parameters weights) w /\
          Shared.matrix_approximates (u_live_matrix #1024 v.v_embedding) x /\
          Shared.rotary_approximates (u_actual_rotary v.v_rope) d)
        (ensures Shared.layer_approximates (u_native_layer v) (Shared.decoder_layer w d x)) =
    u_source_parameters_related weights w
  in C.forall_intro_3 (C.move_requires_3 layer)

inline_for_extraction noextract
ghost
fn u_publish_first_block (weights_address:larray BF16.t opt_weight_count)
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  requires u_after_first_block weights_address prompt_address buffers weights prompt
  ensures source_after_first_block weights_address prompt_address buffers weights prompt

inline_for_extraction noextract
let u_metadata_mask (contents:lseq BF16.t opt_weight_count)
  : GTot (lseq (option BF16.t) opt_weight_count) =
  S.init_ghost opt_weight_count (fun i -> Some (S.index contents i))

inline_for_extraction noextract
ghost
fn u_empty_mask_localize (weights:larray BF16.t opt_weight_count)
  (#contents:erased (lseq BF16.t opt_weight_count)) (ambient:slprop)
  preserves cpu ** ambient ** on gpu_loc (PA.pts_to weights contents)
  ensures PA.pts_to_mask weights #1.0R (u_metadata_mask contents) (fun _ -> False)

inline_for_extraction noextract
fn u_materialize_weight_window
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (lo:HostSize.t) (hi:erased nat{HostSize.v lo <= hi /\ hi <= opt_weight_count})
  (#contents:erased (lseq BF16.t opt_weight_count)) (ambient:slprop)
  preserves cpu ** ambient ** on gpu_loc (PA.pts_to weights contents)
  returns window:PA.array BF16.t
  ensures rewrites_to window (u_weight_window weights (HostSize.v lo) hi)
  ensures pure (PA.length window == hi - HostSize.v lo /\ KA.is_global_array window)

inline_for_extraction noextract
let dec_rms_pointwise_equal (#n:pos) (g0 g1 x0 x1:dec_spec_vector n) (i:dec_spec_index n)
  : Lemma
      (requires forall (j:dec_spec_index n). g0 j == g1 j /\ x0 j == x1 j)
      (ensures dec_spec_rms_norm g0 x0 i == dec_spec_rms_norm g1 x1 i) =
  dec_spec_sum_nonnegative n (fun j -> x0 j *. x0 j);
  dec_spec_sum_nonnegative n (fun j -> x1 j *. x1 j);
  dec_sum_extensional n (fun j -> x0 j *. x0 j) (fun j -> x1 j *. x1 j);
  assert_norm (dec_spec_rms_norm g0 x0 i ==
    g0 i *. (x0 i *. ObRoot.rsqrt
      (dec_spec_sum n (fun j -> x0 j *. x0 j) /. of_int n +. 0.000001R)));
  assert_norm (dec_spec_rms_norm g1 x1 i ==
    g1 i *. (x1 i *. ObRoot.rsqrt
      (dec_spec_sum n (fun j -> x1 j *. x1 j) /. of_int n +. 0.000001R)))

inline_for_extraction noextract
let u_extend (#cols:pos) (x:lseq BF16.t (8 * cols)) (xr:Shared.matrix real 7 cols)
  : GTot (dec_spec_matrix 8 cols) =
  fun row col -> if row < 7 then xr row col else to_real (S.index x (row * cols + col))

inline_for_extraction noextract
let u_extend_related (#cols:pos) (x:lseq BF16.t (8 * cols)) (xr:Shared.matrix real 7 cols)
  : Lemma
      (requires Shared.matrix_approximates (u_live_matrix #cols x) xr)
      (ensures dec_matrix_rep x (u_extend #cols x xr)) =
  let at (row:Shared.index 8) (col:Shared.index cols)
    : Lemma (S.index x (row * cols + col) %~ u_extend #cols x xr row col) =
    if row >= 7 then to_real_ok (S.index x (row * cols + col))
  in C.forall_intro_2 at

inline_for_extraction noextract
let u_rms_math_at (#n:pos) (g x:Shared.vector real n) (i:Shared.index n)
  : Lemma (dec_spec_rms_norm #n g x i == Shared.rms_norm #n g x i) =
  assert (dec_spec_rms_norm #n g x i == Shared.rms_norm #n g x i) by
    (FStar.Tactics.V2.norm [delta_only [
      `%dec_spec_rms_norm; `%Shared.rms_norm; `%dec_spec_sum; `%Shared.sum;
      `%dec_spec_vector; `%dec_spec_index; `%Shared.vector; `%Shared.index]; iota; zeta; unascribe];
     FStar.Tactics.V2.trefl ())
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_rms_apply (#nw:pos) (weights:lseq BF16.t nw) (x ys:lseq BF16.t 8192)
  (offset:nat{offset + 1024 <= nw}) (xr:dec_spec_matrix 8 1024) (gamma:dec_spec_vector 1024)
  : Lemma
      (requires u_rms_prefix #8 #nw x weights ys offset 8 /\
        dec_matrix_rep #8 #1024 x xr /\ dec_bf16_segment_rep weights offset 1024 gamma)
      (ensures dec_matrix_rep #8 #1024 ys (fun row -> dec_spec_rms_norm gamma (xr row))) =
  assert (dec_matrix_prefix #8 #1024 ys (fun row -> dec_spec_rms_norm gamma (xr row)) 8)
#pop-options

inline_for_extraction noextract
let u_rms_live_related (#nw:pos) (weights:lseq BF16.t nw)
  (x ys:lseq BF16.t 8192) (offset:nat{offset + 1024 <= nw})
  (gamma:Shared.vector real 1024) (xr:Shared.hidden real)
  : Lemma
      (requires u_rms_prefix #8 #nw x weights ys offset 8 /\
        dec_bf16_segment_rep weights offset 1024 gamma /\
        Shared.matrix_approximates (u_live_matrix #1024 x) xr)
      (ensures Shared.matrix_approximates (u_live_matrix #1024 ys) (Shared.normalize_hidden gamma xr)) =
  u_extend_related #1024 x xr;
  let extended = u_extend #1024 x xr in
  u_rms_apply weights x ys offset extended gamma;
  let at (row:Shared.index 7) (col:Shared.index 1024)
    : Lemma (S.index ys (row * 1024 + col) %~ Shared.normalize_hidden gamma xr row col) =
    dec_rms_pointwise_equal #1024 gamma gamma (extended row) (xr row) col;
    u_rms_math_at #1024 gamma (xr row) col
  in C.forall_intro_2 at

[@@erasable]
noeq
type dec_spec_parameters = {
  dec_spec_input_layernorm: dec_spec_vector 1024;
  dec_spec_q_proj: dec_spec_matrix 2048 1024;
  dec_spec_k_proj: dec_spec_matrix 1024 1024;
  dec_spec_v_proj: dec_spec_matrix 1024 1024;
  dec_spec_q_norm: dec_spec_vector 128;
  dec_spec_k_norm: dec_spec_vector 128;
  dec_spec_o_proj: dec_spec_matrix 1024 2048;
  dec_spec_post_attention_layernorm: dec_spec_vector 1024;
  dec_spec_gate_proj: dec_spec_matrix 3072 1024;
  dec_spec_up_proj: dec_spec_matrix 3072 1024;
  dec_spec_down_proj: dec_spec_matrix 1024 3072
}

inline_for_extraction noextract
let dec_qkv_weight (w:dec_spec_parameters) (channel:dec_spec_index 4096)
  : GTot (dec_spec_vector 1024) =
  let group = channel / 512 in
  let inner = channel % 512 in
  if inner < 256 then w.dec_spec_q_proj (group * 256 + inner)
  else if inner < 384 then w.dec_spec_k_proj (group * 128 + inner - 256)
  else w.dec_spec_v_proj (group * 128 + inner - 384)

inline_for_extraction noextract
let dec_weights_representation
  (storage:dec_bf16_values dec_weight_elements) (logical:dec_spec_parameters) : prop =
  (forall (channel:dec_spec_index 1024).
    v_approximates (Seq.index storage channel)
      (logical.dec_spec_input_layernorm channel) /\
    v_approximates (Seq.index storage (1024 + channel))
      (logical.dec_spec_post_attention_layernorm channel)) /\
  (forall (lane:dec_spec_index 128).
    v_approximates (Seq.index storage (6293504 + lane))
      (logical.dec_spec_q_norm lane) /\
    v_approximates (Seq.index storage (6293504 + 128 + lane))
      (logical.dec_spec_k_norm lane)) /\
  (forall (row:dec_spec_index 2048) (column:dec_spec_index 1024).
    v_approximates (Seq.index storage (2048 + row * 1024 + column))
      (logical.dec_spec_q_proj row column)) /\
  (forall (row column:dec_spec_index 1024).
    v_approximates (Seq.index storage (2048 + (2048 + row) * 1024 + column))
      (logical.dec_spec_k_proj row column) /\
    v_approximates (Seq.index storage (2048 + (3072 + row) * 1024 + column))
      (logical.dec_spec_v_proj row column)) /\
  (forall (row:dec_spec_index 1024) (column:dec_spec_index 2048).
    v_approximates (Seq.index storage (4196352 + row * 2048 + column))
      (logical.dec_spec_o_proj row column)) /\
  (forall (row:dec_spec_index 3072) (column:dec_spec_index 1024).
    v_approximates (Seq.index storage (6293760 + row * 1024 + column))
      (logical.dec_spec_gate_proj row column) /\
    v_approximates (Seq.index storage (6293760 + (3072 + row) * 1024 + column))
      (logical.dec_spec_up_proj row column)) /\
  (forall (row:dec_spec_index 1024) (column:dec_spec_index 3072).
    v_approximates (Seq.index storage (12585216 + row * 3072 + column))
      (logical.dec_spec_down_proj row column))

inline_for_extraction noextract
let dec_gateup_weight (w:dec_spec_parameters) (channel:dec_spec_index 6144)
  : GTot (dec_spec_vector 1024) =
  let index = channel / 128 * 64 + channel % 64 in
  if channel % 128 < 64 then w.dec_spec_gate_proj index else w.dec_spec_up_proj index

inline_for_extraction noextract
let dec_projection_weights_rep
  (width:nat{dec_dense_width width}) (cols:nat)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (weights:dec_bf16_values dec_weight_elements) (logical:dec_spec_matrix cols width) : prop =
  forall (col:dec_spec_index cols) (k:dec_spec_index width).
    Seq.index weights (offset + dec_projection_row cols layout col * width + k) %~ logical col k

inline_for_extraction noextract
let dec_source_weight_views (weights:dec_bf16_values dec_weight_elements) (w:dec_spec_parameters)
  : Lemma
      (requires dec_weights_representation weights w)
      (ensures
        dec_projection_weights_rep 1024 4096 1 2048 weights (dec_qkv_weight w) /\
        dec_projection_weights_rep 2048 1024 0 4196352 weights w.dec_spec_o_proj /\
        dec_projection_weights_rep 1024 6144 2 6293760 weights (dec_gateup_weight w) /\
        dec_projection_weights_rep 3072 1024 0 12585216 weights w.dec_spec_down_proj) =
  let qkv_at (col:dec_spec_index 4096) (k:dec_spec_index 1024)
    : Lemma (Seq.index weights (2048 + dec_projection_row 4096 1 col * 1024 + k) %~ dec_qkv_weight w col k) =
    if col % 512 < 256 then () else if col % 512 < 384 then () else ()
  in FStar.Classical.forall_intro_2 qkv_at;
  let gate_at (col:dec_spec_index 6144) (k:dec_spec_index 1024)
    : Lemma (Seq.index weights (6293760 + dec_projection_row 6144 2 col * 1024 + k) %~ dec_gateup_weight w col k) =
    if col % 128 < 64 then () else ()
  in FStar.Classical.forall_intro_2 gate_at

inline_for_extraction noextract
let u_real_parameters (w:Shared.parameters real) : dec_spec_parameters = {
  dec_spec_input_layernorm = w.input_layernorm;
  dec_spec_post_attention_layernorm = w.post_attention_layernorm;
  dec_spec_q_proj = w.q_proj;
  dec_spec_k_proj = w.k_proj;
  dec_spec_v_proj = w.v_proj;
  dec_spec_o_proj = w.o_proj;
  dec_spec_q_norm = w.q_norm;
  dec_spec_k_norm = w.k_norm;
  dec_spec_gate_proj = w.gate_proj;
  dec_spec_up_proj = w.up_proj;
  dec_spec_down_proj = w.down_proj
}

inline_for_extraction noextract
let dec_spec_bf16_minimum () : real =
  of_int (-338953138925153547590470800371487866880)

inline_for_extraction noextract
let dec_spec_prefill_mask (query key:dec_spec_index 7) : real =
  if key <= query then 0.0R else dec_spec_bf16_minimum ()

inline_for_extraction noextract
let u_real_rotary (r:Shared.rotary_tables real) : dec_spec_attention_data = {
  dec_spec_position_ids = (fun row -> row);
  dec_spec_cache_positions = (fun row -> row);
  dec_spec_cosine = r.cosine;
  dec_spec_sine = r.sine;
  dec_spec_additive_mask = dec_spec_prefill_mask
}

inline_for_extraction noextract
let u_weights_related (weights:lseq BF16.t opt_weight_count) (w:Shared.parameters real)
  : Lemma
      (requires Shared.parameters_approximates (u_actual_parameters weights) w)
      (ensures dec_weights_representation (opt_layer0_values weights) (u_real_parameters w) /\
        dec_bf16_segment_rep weights 155583488 1024 w.input_layernorm /\
        dec_bf16_segment_rep weights 155584512 1024 w.post_attention_layernorm) = ()

inline_for_extraction noextract
let dec_spec_linear (#src #dst:pos)
  (dec_spec_weight:dec_spec_matrix dst src) (x:dec_spec_vector src) : GTot (dec_spec_vector dst) =
  fun out_channel -> dec_spec_dot x (dec_spec_weight out_channel)

inline_for_extraction noextract
let dec_projection_math (width cols:pos) (weights:dec_spec_matrix cols width)
  (x:dec_spec_matrix 8 width) (residual:dec_spec_matrix 8 cols) (add_residual:bool)
  : GTot (dec_spec_matrix 8 cols) =
  fun row col ->
    if add_residual then dec_spec_linear weights (x row) col +. residual row col
    else dec_spec_linear weights (x row) col

inline_for_extraction noextract
let dec_projection_math_at (width cols:pos) (weights:dec_spec_matrix cols width)
  (x:dec_spec_matrix 8 width) (residual:dec_spec_matrix 8 cols) (add_residual:bool)
  (row:dec_spec_index 8) (col:dec_spec_index cols)
  : Lemma
      (dec_projection_math width cols weights x residual add_residual row col ==
        (if add_residual then dec_spec_dot (x row) (weights col) +. residual row col
         else dec_spec_dot (x row) (weights col))) = ()

inline_for_extraction noextract
let dec_shared_dot_commute (#n:pos) (x y:dec_spec_vector n)
  : Lemma (dec_spec_dot x y == dec_spec_dot y x) =
  dec_sum_extensional n (fun i -> x i *. y i) (fun i -> y i *. x i)

inline_for_extraction noextract
let dec_projection_refines
  (#nr:nat) (width:nat{dec_dense_width width}) (cols:pos)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (add_residual:bool{add_residual ==> 8 * cols <= nr})
  (x:lseq bf16 (8 * width)) (weights:dec_bf16_values dec_weight_elements) (residual:lseq bf16 nr)
  (xr:dec_spec_matrix 8 width) (wr:dec_spec_matrix cols width) (rr:dec_spec_matrix 8 cols)
  : Lemma
      (requires
        dec_matrix_rep x xr /\ dec_projection_weights_rep width cols layout offset weights wr /\
        (add_residual ==> (forall (row:dec_spec_index 8) (col:dec_spec_index cols).
          Seq.index residual (row * cols + col) %~ rr row col)))
      (ensures dec_matrix_rep
        (dec_projection_values width cols layout offset add_residual x weights residual)
        (dec_projection_math width cols wr xr rr add_residual)) =
  let at (row:dec_spec_index 8) (col:dec_spec_index cols)
    : Lemma (Seq.index (dec_projection_values width cols layout offset add_residual x weights residual)
        (row * cols + col) %~
        dec_projection_math width cols wr xr rr add_residual row col) =
    let block = col / 16 * 16 in
    dec_projection_block_contiguous cols layout block (col % 16);
    let start = offset + dec_projection_row cols layout block * width in
    dec_dense_mma_dot width (Seq.slice weights start (start + 16 * width)) x
      (col % 16) row (wr col) (xr row);
    let score = Chest.acc2 (dec_projection_scores width cols layout offset x weights block) (col % 16) row in
    Casts.fcast_approx #F32.t #bf16 score (dec_spec_dot (wr col) (xr row));
    dec_shared_dot_commute (wr col) (xr row);
    assert (Casts.fcast #F32.t #bf16 score %~ dec_spec_dot (xr row) (wr col));
    dec_projection_math_at width cols wr xr rr add_residual row col;
    dec_projection_values_at width cols layout offset add_residual x weights residual row col;
    if add_residual then (
      a_add (Casts.fcast #F32.t #bf16 score) (Seq.index residual (row * cols + col))
        (dec_spec_dot (xr row) (wr col)) (rr row col)
    )
  in
  FStar.Classical.forall_intro_2 at

inline_for_extraction noextract
let u_linear_math_at (#width #cols:pos) (w:Shared.matrix real cols width)
  (x:Shared.matrix real 7 width) (row:Shared.index 7) (col:Shared.index cols)
  : Lemma (Shared.linear_rows #7 #width #cols w x row col == dec_spec_dot #width (x row) (w col)) =
  assert (Shared.linear_rows #7 #width #cols w x row col == dec_spec_dot #width (x row) (w col)) by
    (FStar.Tactics.V2.norm [delta_only [
      `%Shared.linear_rows; `%Shared.linear; `%Shared.dot; `%Shared.sum;
      `%dec_spec_dot; `%dec_spec_sum;
      `%dec_spec_vector; `%dec_spec_index; `%Shared.vector; `%Shared.index]; iota; zeta; unascribe];
     FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let u_residual_extend (#nr:pos) (cols:pos) (r:lseq BF16.t nr)
  (rr:Shared.matrix real 7 cols) : GTot (dec_spec_matrix 8 cols) =
  fun row col -> if row < 7 then rr row col
    else if row * cols + col < nr then to_real (S.index r (row * cols + col)) else 0.0R

inline_for_extraction noextract
let u_residual_extend_related (#nr:pos) (cols:pos) (r:lseq BF16.t nr)
  (rr:Shared.matrix real 7 cols) (add:bool{add ==> 8 * cols <= nr})
  : Lemma
      (requires add ==> (forall (row:Shared.index 7) (col:Shared.index cols).
        S.index r (row * cols + col) %~ rr row col))
      (ensures add ==> (forall (row:Shared.index 8) (col:Shared.index cols).
        S.index r (row * cols + col) %~ u_residual_extend cols r rr row col)) =
  let at (row:Shared.index 8) (col:Shared.index cols)
    : Lemma (add ==> S.index r (row * cols + col) %~ u_residual_extend cols r rr row col) =
    if add && row >= 7 then to_real_ok (S.index r (row * cols + col))
  in C.forall_intro_2 at

inline_for_extraction noextract
let u_linear_live_related (#nr:pos)
  (width:nat{dec_dense_width width}) (cols:pos)
  (layout:nat{dec_projection_layout cols layout})
  (offset:nat{offset + cols * width <= dec_weight_elements})
  (add:bool{add ==> 8 * cols <= nr})
  (x:lseq BF16.t (8 * width)) (weights:lseq BF16.t dec_weight_elements) (residual:lseq BF16.t nr)
  (xr:Shared.matrix real 7 width) (wr:Shared.matrix real cols width) (rr:Shared.matrix real 7 cols)
  : Lemma
      (requires Shared.matrix_approximates (u_live_matrix #width x) xr /\
        dec_projection_weights_rep width cols layout offset weights wr /\
        (add ==> (forall (row:Shared.index 7) (col:Shared.index cols).
          S.index residual (row * cols + col) %~ rr row col)))
      (ensures Shared.matrix_approximates
        (u_live_matrix #cols (dec_projection_values width cols layout offset add x weights residual))
        (fun row col ->
          if add then Shared.linear_rows #7 #width #cols wr xr row col +. rr row col
          else Shared.linear_rows #7 #width #cols wr xr row col)) =
  u_extend_related #width x xr;
  u_residual_extend_related cols residual rr add;
  let ex = u_extend #width x xr in
  let er = u_residual_extend cols residual rr in
  dec_projection_refines width cols layout offset add x weights residual ex wr er;
  let at (row:Shared.index 7) (col:Shared.index cols)
    : Lemma (
      S.index (dec_projection_values width cols layout offset add x weights residual) (row * cols + col) %~
        (if add then Shared.linear_rows #7 #width #cols wr xr row col +. rr row col
         else Shared.linear_rows #7 #width #cols wr xr row col)) =
    dec_projection_math_at width cols wr ex er add row col;
    dec_dot_pointwise_equal #width (ex row) (wr col) (xr row) (wr col);
    u_linear_math_at #width #cols wr xr row col
  in C.forall_intro_2 at

inline_for_extraction noextract
let dec_qkv_query_weight_at (w:dec_spec_parameters) (head:dec_spec_index 16)
  (lane:dec_spec_index 128) (col:dec_spec_index 1024)
  : Lemma
      (dec_qkv_weight w (head / 2 * 512 + head % 2 * 128 + lane) col ==
        w.dec_spec_q_proj (head * 128 + lane) col) = ()

inline_for_extraction noextract
let dec_qkv_kv_weight_at (w:dec_spec_parameters) (head:dec_spec_index 8)
  (lane:dec_spec_index 128) (col:dec_spec_index 1024)
  : Lemma
      (dec_qkv_weight w (head * 512 + 256 + lane) col == w.dec_spec_k_proj (head * 128 + lane) col /\
       dec_qkv_weight w (head * 512 + 384 + lane) col == w.dec_spec_v_proj (head * 128 + lane) col) = ()

inline_for_extraction noextract
let u_query_view (x:lseq BF16.t 32768) : GTot (Shared.heads BF16.t 16) =
  fun row head lane -> S.index x (row * 4096 + (head / 2) * 512 + (head % 2) * 128 + lane)

inline_for_extraction noextract
let u_key_view (x:lseq BF16.t 32768) : GTot (Shared.heads BF16.t 8) =
  fun row head lane -> S.index x (row * 4096 + head * 512 + 256 + lane)

inline_for_extraction noextract
let u_value_view (x:lseq BF16.t 32768) : GTot (Shared.heads BF16.t 8) =
  fun row head lane -> S.index x (row * 4096 + head * 512 + 384 + lane)

inline_for_extraction noextract
let u_linear_channel_equal (#width #left_cols #right_cols:pos)
  (left:Shared.matrix real left_cols width) (right:Shared.matrix real right_cols width)
  (x:Shared.matrix real 7 width) (row:Shared.index 7)
  (lc:Shared.index left_cols) (rc:Shared.index right_cols)
  : Lemma
      (requires forall (k:Shared.index width). left lc k == right rc k)
      (ensures Shared.linear_rows #7 #width #left_cols left x row lc ==
        Shared.linear_rows #7 #width #right_cols right x row rc) =
  u_linear_math_at #width #left_cols left x row lc;
  u_linear_math_at #width #right_cols right x row rc;
  dec_dot_pointwise_equal #width (x row) (left lc) (x row) (right rc)

inline_for_extraction noextract
let u_qkv_related (qkv:lseq BF16.t 32768) (w:Shared.parameters real) (x:Shared.hidden real)
  : Lemma
      (requires Shared.matrix_approximates (u_live_matrix #4096 qkv)
        (Shared.linear_rows #7 #1024 #4096 (dec_qkv_weight (u_real_parameters w)) x))
      (ensures
        Shared.heads_approximates (u_query_view qkv)
          (Shared.split_heads #real #16 (Shared.linear_rows #7 #1024 #2048 w.q_proj x)) /\
        Shared.heads_approximates (u_key_view qkv)
          (Shared.split_heads #real #8 (Shared.linear_rows #7 #1024 #1024 w.k_proj x)) /\
        Shared.heads_approximates (u_value_view qkv)
          (Shared.split_heads #real #8 (Shared.linear_rows #7 #1024 #1024 w.v_proj x))) =
  let query (row:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 128)
    : Lemma (u_query_view qkv row head lane %~
        Shared.split_heads #real #16 (Shared.linear_rows #7 #1024 #2048 w.q_proj x) row head lane) =
    C.forall_intro (dec_qkv_query_weight_at (u_real_parameters w) head lane);
    u_linear_channel_equal #1024 #4096 #2048 (dec_qkv_weight (u_real_parameters w)) w.q_proj x row
      (head / 2 * 512 + head % 2 * 128 + lane) (head * 128 + lane)
  in C.forall_intro_3 query;
  let kv (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
    : Lemma (
        u_key_view qkv row head lane %~
          Shared.split_heads #real #8 (Shared.linear_rows #7 #1024 #1024 w.k_proj x) row head lane /\
        u_value_view qkv row head lane %~
          Shared.split_heads #real #8 (Shared.linear_rows #7 #1024 #1024 w.v_proj x) row head lane) =
    C.forall_intro (dec_qkv_kv_weight_at (u_real_parameters w) head lane);
    u_linear_channel_equal #1024 #4096 #1024 (dec_qkv_weight (u_real_parameters w)) w.k_proj x row
      (head * 512 + 256 + lane) (head * 128 + lane);
    u_linear_channel_equal #1024 #4096 #1024 (dec_qkv_weight (u_real_parameters w)) w.v_proj x row
      (head * 512 + 384 + lane) (head * 128 + lane)
  in C.forall_intro_3 kv

inline_for_extraction noextract
let u_rope_prefix_at (x:lseq BF16.t 32768) (weights:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (ys:lseq BF16.t 32768)
  (row:Shared.index 7) (col:Shared.index 4096{col % 512 < 384})
  : Lemma
      (requires u_rope_prefix x weights rt ys 168)
      (ensures u_rope_cell x weights rt row (col / 512) ((col % 512) / 128) (col % 128)
        (S.index ys (row * 4096 + col))) =
  assert (dec_qk_head_number row col < 168)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_rope_cell_apply (x:lseq BF16.t 32768) (weights:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3) (lane:Shared.index 128)
  (value:BF16.t) (r gamma:dec_spec_vector 128) (cr sr:real)
  : Lemma
      (requires u_rope_cell x weights rt row group head lane value /\
        u_rms_guard x weights (row * 4096 + group * 512 + head * 128)
          (u_rope_weight_start head) 128 r gamma /\
        S.index rt (row * 128 + lane) %~ cr /\ S.index rt (524288 + row * 128 + lane) %~ sr)
      (ensures value %~ (dec_spec_rms_norm gamma r lane *. cr +.
        (if lane < 64 then 0.0R -. dec_spec_rms_norm gamma r (lane + 64)
         else dec_spec_rms_norm gamma r (lane - 64)) *. sr)) =
  reveal_opaque (`%u_rope_cell) (u_rope_cell x weights rt row group head lane value);
  assert (forall (r0 g0:dec_spec_vector 128) (c0 s0:real).
    u_rms_guard x weights (row * 4096 + group * 512 + head * 128)
      (u_rope_weight_start head) 128 r0 g0 /\
    S.index rt (row * 128 + lane) %~ c0 /\ S.index rt (524288 + row * 128 + lane) %~ s0 ==>
      value %~ (dec_spec_rms_norm g0 r0 lane *. c0 +.
        (if lane < 64 then 0.0R -. dec_spec_rms_norm g0 r0 (lane + 64)
         else dec_spec_rms_norm g0 r0 (lane - 64)) *. s0));
  ()
#pop-options

inline_for_extraction noextract
let u_rope_apply_cell (x:lseq BF16.t 32768) (weights:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (ys:lseq BF16.t 32768)
  (row:Shared.index 7) (group:Shared.index 8) (head:Shared.index 3) (lane:Shared.index 128)
  (r gamma:dec_spec_vector 128) (cr sr:real)
  : Lemma
      (requires u_rope_prefix x weights rt ys 168 /\
        u_rms_guard x weights (row * 4096 + group * 512 + head * 128)
          (u_rope_weight_start head) 128 r gamma /\
        S.index rt (row * 128 + lane) %~ cr /\ S.index rt (524288 + row * 128 + lane) %~ sr)
      (ensures S.index ys (row * 4096 + group * 512 + head * 128 + lane) %~
        (dec_spec_rms_norm gamma r lane *. cr +.
          (if lane < 64 then 0.0R -. dec_spec_rms_norm gamma r (lane + 64)
           else dec_spec_rms_norm gamma r (lane - 64)) *. sr)) =
  let value = S.index ys (row * 4096 + group * 512 + head * 128 + lane) in
  Kuiper.Math.div_mod_of_mul_add 128 head lane;
  Kuiper.Math.div_mod_of_mul_add 512 group (head * 128 + lane);
  u_rope_prefix_at x weights rt ys row (group * 512 + head * 128 + lane);
  assert (u_rope_cell x weights rt row group head lane value);
  u_rope_cell_apply x weights rt row group head lane value r gamma cr sr

inline_for_extraction noextract
let u_rope_heads_related (x:lseq BF16.t 32768) (weights:lseq BF16.t dec_weight_elements)
  (rt:lseq BF16.t 1048576) (ys:lseq BF16.t 32768)
  (w:Shared.parameters real) (d:Shared.rotary_tables real)
  (q:Shared.heads real 16) (k v:Shared.heads real 8)
  : Lemma
      (requires u_rope_prefix x weights rt ys 168 /\
        dec_weights_representation weights (u_real_parameters w) /\
        Shared.rotary_approximates (u_actual_rotary rt) d /\
        Shared.heads_approximates (u_query_view x) q /\
        Shared.heads_approximates (u_key_view x) k /\
        Shared.heads_approximates (u_value_view x) v)
      (ensures
        dec_source_qk_representation ys
          (Shared.rotary #16 d (Shared.normalize_heads #16 w.q_norm q))
          (Shared.rotary #8 d (Shared.normalize_heads #8 w.k_norm k)) /\
        dec_source_values_representation ys v)
  =
  let query (row:Shared.index 7) (head:Shared.index 16) (lane:Shared.index 128)
    : Lemma (u_query_view ys row head lane %~
        Shared.rotary #16 d (Shared.normalize_heads #16 w.q_norm q) row head lane) =
    u_rope_apply_cell x weights rt ys row (head / 2) (head % 2) lane (q row head) w.q_norm
      (d.cosine row lane) (d.sine row lane);
    u_rms_math_at #128 w.q_norm (q row head) lane;
    if lane < 64 then u_rms_math_at #128 w.q_norm (q row head) (lane + 64)
    else u_rms_math_at #128 w.q_norm (q row head) (lane - 64)
  in C.forall_intro_3 query;
  let key_value (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
    : Lemma (
        u_key_view ys row head lane %~
          Shared.rotary #8 d (Shared.normalize_heads #8 w.k_norm k) row head lane /\
        u_value_view ys row head lane %~ v row head lane) =
    u_rope_apply_cell x weights rt ys row head 2 lane (k row head) w.k_norm
      (d.cosine row lane) (d.sine row lane);
    u_rms_math_at #128 w.k_norm (k row head) lane;
    if lane < 64 then u_rms_math_at #128 w.k_norm (k row head) (lane + 64)
    else u_rms_math_at #128 w.k_norm (k row head) (lane - 64);
    u_rope_value_unchanged x weights rt ys row head lane
  in C.forall_intro_3 key_value

inline_for_extraction noextract
let opt_repeat_math (x:dec_spec_heads 8)
  : Lemma (dec_spec_repeat_kv x == Shared.repeat_kv #real x) =
  assert (dec_spec_repeat_kv x == Shared.repeat_kv #real x) by
    (FStar.Tactics.V2.norm [delta_only [
      `%dec_spec_repeat_kv; `%Shared.repeat_kv;
      `%dec_spec_vector; `%dec_spec_index; `%Shared.vector; `%Shared.index]; iota; zeta];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let opt_attention_math (data:dec_spec_attention_data) (q k:dec_spec_heads 16)
  : Lemma (dec_spec_attention_probabilities data q k == Shared.attention_probabilities q k) =
  assert (dec_spec_attention_probabilities data q k == Shared.attention_probabilities q k) by
    (FStar.Tactics.V2.norm [delta_only [
      `%dec_spec_attention_probabilities; `%Shared.attention_probabilities;
      `%dec_spec_prefix_softmax; `%Shared.prefix_softmax;
      `%dec_spec_softmax; `%Shared.softmax;
      `%dec_spec_dot; `%Shared.dot; `%dec_spec_sum; `%Shared.sum;
      `%dec_spec_vector; `%dec_spec_index; `%Shared.vector; `%Shared.index]; iota; zeta; unascribe];
     FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let opt_mix_math (p:dec_spec_attention_rows) (v:dec_spec_heads 16)
  : Lemma (dec_spec_attention_values p v == Shared.attention_values p v) =
  assert (dec_spec_attention_values p v == Shared.attention_values p v) by
    (FStar.Tactics.V2.norm [delta_only [
      `%dec_spec_attention_values; `%Shared.attention_values;
      `%dec_spec_sum; `%Shared.sum;
      `%dec_spec_vector; `%dec_spec_index; `%Shared.vector; `%Shared.index]; iota; zeta];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let dec_spec_concatenate_heads (x:dec_spec_heads 16) : GTot (dec_spec_matrix 7 2048) =
  fun token channel -> x token (channel / 128) (channel % 128)

inline_for_extraction noextract
let opt_concat_math (x:dec_spec_heads 16)
  : Lemma (dec_spec_concatenate_heads x == Shared.concatenate_heads #real x) =
  assert (dec_spec_concatenate_heads x == Shared.concatenate_heads #real x) by
    (FStar.Tactics.V2.norm [delta_only [
      `%dec_spec_concatenate_heads; `%Shared.concatenate_heads;
      `%dec_spec_vector; `%dec_spec_index; `%Shared.vector; `%Shared.index]; iota; zeta];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let u_attention_live_math (m:u_attention_meaning) (row:Shared.index 7) (col:Shared.index 2048)
  : Lemma (
    u_attention_model m row col ==
      Shared.concatenate_heads
        (Shared.attention_values
          (Shared.attention_probabilities m.u_query (Shared.repeat_kv m.u_key))
          (Shared.repeat_kv m.u_value)) row col) =
  opt_repeat_math m.u_key;
  opt_repeat_math m.u_value;
  opt_attention_math m.u_data m.u_query (dec_spec_repeat_kv m.u_key);
  let probabilities = dec_spec_attention_probabilities m.u_data m.u_query (dec_spec_repeat_kv m.u_key) in
  opt_mix_math probabilities (dec_spec_repeat_kv m.u_value);
  opt_concat_math (dec_spec_attention_values probabilities (dec_spec_repeat_kv m.u_value));
  reveal_opaque (`%dec_attention_model)
    (dec_attention_model m.u_data m.u_query m.u_key m.u_value)

inline_for_extraction noextract
let u_attention_live_related (rotated:lseq BF16.t 32768) (ys:lseq BF16.t 16384)
  (m:u_attention_meaning)
  : Lemma
      (requires
        (forall (meaning:u_attention_meaning).
          u_attention_guard rotated meaning ==> dec_matrix_rep ys (u_attention_model meaning)) /\
        u_attention_guard rotated m)
      (ensures Shared.matrix_approximates (u_live_matrix #2048 ys)
        (Shared.concatenate_heads
          (Shared.attention_values (Shared.attention_probabilities m.u_query (Shared.repeat_kv m.u_key))
            (Shared.repeat_kv m.u_value)))) =
  assert (dec_matrix_rep #8 #2048 ys (u_attention_model m));
  let at (row:Shared.index 7) (col:Shared.index 2048)
    : Lemma (S.index ys (row * 2048 + col) %~
        Shared.concatenate_heads
          (Shared.attention_values (Shared.attention_probabilities m.u_query (Shared.repeat_kv m.u_key))
            (Shared.repeat_kv m.u_value)) row col) =
    u_attention_live_math m row col
  in C.forall_intro_2 at

inline_for_extraction noextract
let dec_gateup_weight_at (w:dec_spec_parameters) (col:dec_spec_index 3072) (k:dec_spec_index 1024)
  : Lemma
      (dec_gateup_weight w (col / 64 * 128 + col % 64) k == w.dec_spec_gate_proj col k /\
       dec_gateup_weight w (col / 64 * 128 + col % 64 + 64) k == w.dec_spec_up_proj col k) = ()

inline_for_extraction noextract
let u_gateup_projection_at (w:Shared.parameters real) (x:Shared.hidden real)
  (row:Shared.index 7) (col:Shared.index 3072)
  : Lemma (
    Shared.linear_rows #7 #1024 #6144 (dec_gateup_weight (u_real_parameters w)) x row
      (col / 64 * 128 + col % 64) == Shared.linear_rows #7 #1024 #3072 w.gate_proj x row col /\
    Shared.linear_rows #7 #1024 #6144 (dec_gateup_weight (u_real_parameters w)) x row
      (col / 64 * 128 + col % 64 + 64) == Shared.linear_rows #7 #1024 #3072 w.up_proj x row col) =
  C.forall_intro (dec_gateup_weight_at (u_real_parameters w) col);
  u_linear_channel_equal #1024 #6144 #3072 (dec_gateup_weight (u_real_parameters w)) w.gate_proj x row
    (col / 64 * 128 + col % 64) col;
  u_linear_channel_equal #1024 #6144 #3072 (dec_gateup_weight (u_real_parameters w)) w.up_proj x row
    (col / 64 * 128 + col % 64 + 64) col

inline_for_extraction noextract
let u_silu_math (r:real) : Lemma (Shared.silu r == dec_spec_silu r) =
  assert (Shared.silu r == dec_spec_silu r) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.silu; `%dec_spec_silu]];
     FStar.Tactics.V2.trefl ())
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_silu_apply (gateup:lseq BF16.t 49152) (activated:lseq BF16.t 24576)
  (row:Shared.index 7) (col:Shared.index 3072) (gate up:real)
  : Lemma
      (requires u_silu_prefix gateup activated 24576 /\
        S.index gateup (row * 6144 + col / 64 * 128 + col % 64) %~ gate /\
        S.index gateup (row * 6144 + col / 64 * 128 + col % 64 + 64) %~ up)
      (ensures S.index activated (row * 3072 + col) %~ (Shared.silu gate *. up)) =
  u_silu_math gate
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_silu_gate_related (gateup:lseq BF16.t 49152) (activated:lseq BF16.t 24576)
  (w:Shared.parameters real) (x:Shared.hidden real)
  : Lemma
      (requires u_silu_prefix gateup activated 24576 /\
        Shared.matrix_approximates (u_live_matrix #6144 gateup)
          (Shared.linear_rows #7 #1024 #6144 (dec_gateup_weight (u_real_parameters w)) x))
      (ensures Shared.matrix_approximates (u_live_matrix #3072 activated)
        (fun row col ->
          Shared.silu (Shared.linear_rows #7 #1024 #3072 w.gate_proj x row col) *.
            Shared.linear_rows #7 #1024 #3072 w.up_proj x row col)) =
  let gate = Shared.linear_rows #7 #1024 #3072 w.gate_proj x in
  let up = Shared.linear_rows #7 #1024 #3072 w.up_proj x in
  let at (row:Shared.index 7) (col:Shared.index 3072)
    : Lemma (S.index activated (row * 3072 + col) %~
      (Shared.silu (gate row col) *. up row col)) =
    u_gateup_projection_at w x row col;
    u_silu_apply gateup activated row col (gate row col) (up row col)
  in C.forall_intro_2 at
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_mlp_live_related (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) (w:Shared.parameters real) (residual:Shared.hidden real)
  : Lemma
      (requires u_block_stage weights prompt v 11 /\
        Shared.parameters_approximates (u_actual_parameters weights) w /\
        Shared.matrix_approximates (u_live_matrix #1024 v.v_residual) residual)
      (ensures Shared.matrix_approximates (u_live_matrix #1024 v.v_hidden)
        (Shared.residual_add residual
          (Shared.feed_forward w (Shared.normalize_hidden w.post_attention_layernorm residual)))) =
  u_weights_related weights w;
  dec_source_weight_views (opt_layer0_values weights) (u_real_parameters w);
  u_rms_live_related weights v.v_residual v.v_postnorm 155584512 w.post_attention_layernorm residual;
  let normalized = Shared.normalize_hidden w.post_attention_layernorm residual in
  u_linear_live_related 1024 6144 2 6293760 false v.v_postnorm (opt_layer0_values weights) v.v_residual
    normalized (dec_gateup_weight (u_real_parameters w)) (fun _ _ -> 0.0R);
  assert (Shared.matrix_approximates (u_live_matrix #6144 v.v_gateup)
    (Shared.linear_rows #7 #1024 #6144 (dec_gateup_weight (u_real_parameters w)) normalized));
  u_silu_gate_related v.v_gateup v.v_activated w normalized;
  let activated:Shared.matrix real 7 3072 = fun row col ->
    Shared.silu (Shared.linear_rows #7 #1024 #3072 w.gate_proj normalized row col) *.
      Shared.linear_rows #7 #1024 #3072 w.up_proj normalized row col in
  u_linear_live_related 3072 1024 0 12585216 true v.v_activated (opt_layer0_values weights) v.v_residual
    activated w.down_proj residual;
  let down = Shared.linear_rows #7 #3072 #1024 w.down_proj activated in
  assert (Shared.feed_forward w normalized == down) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.feed_forward]];
     FStar.Tactics.V2.smt ());
  assert (Shared.matrix_approximates (u_live_matrix #1024 v.v_hidden) (Shared.residual_add residual down))
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_decoder_live_related (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values) (w:Shared.parameters real) (x:Shared.hidden real) (d:Shared.rotary_tables real)
  : Lemma
      (requires u_block_stage weights prompt v 11 /\ dec_attention_domain v.v_rotated /\
        Shared.parameters_approximates (u_actual_parameters weights) w /\
        Shared.matrix_approximates (u_live_matrix #1024 v.v_embedding) x /\
        Shared.rotary_approximates (u_actual_rotary v.v_rope) d)
      (ensures Shared.layer_approximates (u_native_layer v) (Shared.decoder_layer w d x)) =
  u_weights_related weights w;
  dec_source_weight_views (opt_layer0_values weights) (u_real_parameters w);
  u_rms_live_related weights v.v_embedding v.v_normalized0 155583488 w.input_layernorm x;
  let normalized = Shared.normalize_hidden w.input_layernorm x in
  u_linear_live_related 1024 4096 1 2048 false v.v_normalized0 (opt_layer0_values weights) v.v_embedding
    normalized (dec_qkv_weight (u_real_parameters w)) (fun _ _ -> 0.0R);
  u_qkv_related v.v_qkv w normalized;
  let q0 = Shared.split_heads #real #16 (Shared.linear_rows #7 #1024 #2048 w.q_proj normalized) in
  let k0 = Shared.split_heads #real #8 (Shared.linear_rows #7 #1024 #1024 w.k_proj normalized) in
  let value = Shared.split_heads #real #8 (Shared.linear_rows #7 #1024 #1024 w.v_proj normalized) in
  u_rope_heads_related v.v_qkv (opt_layer0_values weights) v.v_rope v.v_rotated w d q0 k0 value;
  let query = Shared.rotary #16 d (Shared.normalize_heads #16 w.q_norm q0) in
  let key = Shared.rotary #8 d (Shared.normalize_heads #8 w.k_norm k0) in
  let meaning:u_attention_meaning = {
    u_query = query; u_key = key; u_value = value; u_data = u_real_rotary d
  } in
  u_attention_live_related v.v_rotated v.v_attention meaning;
  let attention = Shared.concatenate_heads
    (Shared.attention_values (Shared.attention_probabilities query (Shared.repeat_kv key))
      (Shared.repeat_kv value)) in
  u_linear_live_related 2048 1024 0 4196352 true v.v_attention (opt_layer0_values weights) v.v_embedding
    attention w.o_proj x;
  let projected = Shared.linear_rows #7 #2048 #1024 w.o_proj attention in
  let residual = Shared.residual_add x projected in
  assert (Shared.matrix_approximates (u_live_matrix #1024 v.v_residual) residual);
  u_mlp_live_related weights prompt v w residual;
  let expected:Shared.layer_values real = {
    hidden_after = Shared.residual_add residual
      (Shared.feed_forward w (Shared.normalize_hidden w.post_attention_layernorm residual));
    keys = key;
    values = value
  } in
  assert (Shared.layer_approximates (u_native_layer v) expected);
  assert (Shared.decoder_layer w d x == expected) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.decoder_layer]];
     FStar.Tactics.V2.smt ())
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let u_block_numerical_from_stage (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:opt_block_values)
  : Lemma (requires u_block_stage weights prompt v 11) (ensures u_block_numerical weights v) =
  let layer (w:Shared.parameters real) (x:Shared.hidden real) (d:Shared.rotary_tables real)
    : Lemma
        (requires dec_attention_domain v.v_rotated /\
          Shared.parameters_approximates (u_actual_parameters weights) w /\
          Shared.matrix_approximates (u_live_matrix #1024 v.v_embedding) x /\
          Shared.rotary_approximates (u_actual_rotary v.v_rope) d)
        (ensures Shared.layer_approximates (u_native_layer v) (Shared.decoder_layer w d x)) =
    u_decoder_live_related weights prompt v w x d
  in C.forall_intro_3 (C.move_requires_3 layer);
  let norm (g:Shared.vector real 1024) (x:Shared.hidden real)
    : Lemma
        (requires Shared.vector_approximates (u_next_gamma_values weights) g /\
          Shared.matrix_approximates (u_live_matrix #1024 v.v_hidden) x)
        (ensures Shared.matrix_approximates (u_live_matrix #1024 v.v_normalized1)
          (Shared.normalize_hidden g x)) =
    u_rms_live_related weights v.v_hidden v.v_normalized1 171314432 g x
  in C.forall_intro_2 (C.move_requires_2 norm)
#pop-options

inline_for_extraction noextract
let u_block_cache_frame (before:Shared.cache_state BF16.t) (v:opt_block_values)
  (other:Shared.index 28{other <> 0})
  : Lemma (u_block_cache_after before v other == before other) =
  Shared.cache_frame before 0 (u_native_layer v) other

inline_for_extraction noextract
let u_first_block_cache_binding (v:opt_block_values)
  : Lemma (
      Shared.cache_length (u_first_block_cache v 0) == 7 /\
      (forall (other:Shared.index 28{other > 0}). u_first_block_cache v other == Shared.Empty)) =
  let at (other:Shared.index 28{other > 0})
    : Lemma (u_first_block_cache v other == Shared.Empty) =
    u_block_cache_frame (fun _ -> Shared.Empty) v other
  in C.forall_intro at

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

inline_for_extraction noextract
let source_tile_distribute (tile:larray F32.t 128)
  (#values:erased (lseq F32.t 128)) (#tid:enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** when_ (reveal tid == 0) (tile |-> values))
      (ensures fun _ -> thread_id 32 tid **
        tile |-> Frac (1.0R /. 32.0R) values) =
  Kuiper.Barrier.Warp.warp_barrier_wait ()
    (fun i -> when_ (i == 0) (tile |-> values))
    (fun _ -> tile |-> Frac (1.0R /. 32.0R) values)
    (source_tile_distribute_proof tile #values ()) #32 #tid

inline_for_extraction noextract
type pg_source_descriptor (pre post:slprop) (threads:szp{threads <= max_threads})
  (shared:list SH.shmem_desc) =
  k:KS.kernel_desc pre post{
    SZ.v k.nblk == 1 /\ k.nthr == threads /\ k.shmems_desc == shared /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract threads /\ k.barrier_count bid == 0)}

inline_for_extraction noextract
type pg_remaining_layer = layer:Shared.index 28{layer > 0}

inline_for_extraction noextract
let pg_layer_base (layer:Shared.index 28) : nat =
  155583488 + layer * 15730944

inline_for_extraction noextract
let pg_layer_weights (weights:lseq BF16.t opt_weight_count) (layer:Shared.index 28)
  : GTot (lseq BF16.t dec_weight_elements) =
  S.slice weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)

inline_for_extraction noextract
let pg_size (index:Shared.index 256) : pos =
  if index = 1 then 1048576
  else if index = 254 then 7168
  else if index = 255 then 1228800
  else if index >= 2 then
    let slot = (index - 2) % 9 in
    if slot = 1 || slot = 2 then 32768
    else if slot = 3 then 16384
    else if slot = 6 then 49152
    else if slot = 7 then 24576
    else 8192
  else 8192

inline_for_extraction noextract
let pg_index (layer:Shared.index 28) (slot:Shared.index 11) : Shared.index 256 =
  if slot = 0 then (if layer = 0 then 0 else 9 * layer + 1)
  else if slot = 1 then 1
  else 9 * layer + slot

inline_for_extraction noextract
type pg_values = values:lseq (S.seq BF16.t) 256{
  forall (i:Shared.index 256). S.length (S.index values i) == pg_size i}

inline_for_extraction noextract
let pg_slot (values:pg_values) (index:Shared.index 256) : GTot (lseq BF16.t (pg_size index)) =
  S.index values index

inline_for_extraction noextract
let pg_replace (values:pg_values) (index:Shared.index 256) (value:lseq BF16.t (pg_size index))
  : GTot pg_values = S.upd values index value

inline_for_extraction noextract
let pg_layer_values (values:pg_values) (layer:Shared.index 28) : GTot opt_block_values = {
  v_embedding = pg_slot values (pg_index layer 0);
  v_rope = pg_slot values 1;
  v_normalized0 = pg_slot values (pg_index layer 2);
  v_qkv = pg_slot values (pg_index layer 3);
  v_rotated = pg_slot values (pg_index layer 4);
  v_attention = pg_slot values (pg_index layer 5);
  v_residual = pg_slot values (pg_index layer 6);
  v_postnorm = pg_slot values (pg_index layer 7);
  v_gateup = pg_slot values (pg_index layer 8);
  v_activated = pg_slot values (pg_index layer 9);
  v_hidden = pg_slot values (pg_index layer 10);
  v_normalized1 = pg_slot values (pg_index layer 2)
}

inline_for_extraction noextract
let pg_first_values (v:pg_values) : GTot opt_block_values =
  {pg_layer_values v 0 with v_normalized1 = pg_slot v 11}

inline_for_extraction noextract
let pg_input_gamma (weights:lseq BF16.t opt_weight_count) (layer:Shared.index 28)
  : GTot (Shared.vector BF16.t 1024) =
  fun col -> S.index weights (pg_layer_base layer + col)

inline_for_extraction noextract
let pg_layer_arithmetic (weights:lseq BF16.t opt_weight_count) (layer:Shared.index 28)
  (v:opt_block_values) (step:nat{step <= 8}) : GTot prop =
  Shared.final_norm_relation (pg_input_gamma weights layer)
    (u_live_matrix #1024 v.v_embedding) (u_live_matrix #1024 v.v_normalized0) /\
  (step >= 1 ==> v.v_qkv == dec_projection_values 1024 4096 1 2048 false
    v.v_normalized0 (pg_layer_weights weights layer) v.v_embedding) /\
  (step >= 2 ==> u_rope_prefix v.v_qkv (pg_layer_weights weights layer) v.v_rope v.v_rotated 168) /\
  (step >= 3 ==>
    (forall (m:u_attention_meaning). u_attention_guard v.v_rotated m ==>
      dec_matrix_rep #8 #2048 v.v_attention (u_attention_model m)) /\
    (forall (col:Shared.index 2048). S.index v.v_attention (14336 + col) == (F.zero #BF16.t))) /\
  (step >= 4 ==> v.v_residual == dec_projection_values 2048 1024 0 4196352 true
    v.v_attention (pg_layer_weights weights layer) v.v_embedding) /\
  (step >= 5 ==> u_rms_prefix #8 #opt_weight_count v.v_residual weights v.v_postnorm
    (pg_layer_base layer + 1024) 8) /\
  (step >= 6 ==> v.v_gateup == dec_projection_values 1024 6144 2 6293760 false
    v.v_postnorm (pg_layer_weights weights layer) v.v_residual) /\
  (step >= 7 ==> u_silu_prefix v.v_gateup v.v_activated 24576) /\
  (step >= 8 ==> v.v_hidden == dec_projection_values 3072 1024 0 12585216 true
    v.v_activated (pg_layer_weights weights layer) v.v_residual)

[@@"opaque_to_smt"]
inline_for_extraction noextract
let pg_decoder_stage (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (values:pg_values) (calls:nat{11 <= calls /\ calls <= 253}) : GTot prop =
  source_first_block_observation weights prompt (pg_first_values values) /\
  (forall (layer:Shared.index 28{layer > 0}).
    calls >= 2 + 9 * layer ==>
      pg_layer_arithmetic weights layer (pg_layer_values values layer)
        (if calls >= 10 + 9 * layer then 8 else calls - (2 + 9 * layer)))

inline_for_extraction noextract
let pg_decoder_stage_definition (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (calls:nat{11 <= calls /\ calls <= 253})
  : Lemma (pg_decoder_stage weights prompt v calls <==>
    (source_first_block_observation weights prompt (pg_first_values v) /\
      (forall (layer:Shared.index 28{layer > 0}).
        calls >= 2 + 9 * layer ==>
          pg_layer_arithmetic weights layer (pg_layer_values v layer)
            (if calls >= 10 + 9 * layer then 8 else calls - (2 + 9 * layer))))) =
  reveal_opaque (`%pg_decoder_stage) (pg_decoder_stage weights prompt v calls)

inline_for_extraction noextract
let pg_layer_step_calls (layer:Shared.index 28)
  : Lemma (
    pg_layer_base layer + dec_weight_elements <= opt_weight_count /\
    2 + 9 * layer <= 245 /\ 10 + 9 * layer <= 253 /\
    pg_index layer 10 < 254 /\
    (layer < 27 ==> pg_index (layer + 1) 0 == pg_index layer 10)) = ()

inline_for_extraction noextract
let rec pg_list_index_append (#a:Type0) (left right:list a)
  (index:nat{index < List.length left + List.length right})
  : Lemma (
    List.length (List.append left right) == List.length left + List.length right /\
    List.index (List.append left right) index ==
      (if index < List.length left then List.index left index
       else List.index right (index - List.length left)))
    (decreases left) =
  ListProps.append_length left right;
  match left with
  | [] -> ()
  | _::tail -> if index > 0 then pg_list_index_append tail right (index - 1)
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let rec pg_layer_shape_at (count:pos{count <= 28})
  (index:nat{index < 9 * count /\ index < List.length (opt_layer_allocations count)})
  : Lemma (List.index (opt_layer_allocations count) index ==
      List.index opt_layer_allocation_shapes (index % 9))
    (decreases count) =
  opt_layer_allocation_count (count - 1);
  pg_list_index_append opt_layer_allocation_shapes (opt_layer_allocations (count - 1)) index;
  if index >= 9 then (
    pg_layer_shape_at (count - 1) (index - 9);
    FStar.Math.Lemmas.lemma_mod_plus (index - 9) 1 9;
    assert ((index - 9) % 9 == index % 9))
#pop-options
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_shape_at (index:Shared.index 256)
  : Lemma (List.length opt_bf16_allocation_shapes == 256 /\
    RT.numel (List.index opt_bf16_allocation_shapes index) == pg_size index) =
  opt_allocation_count ();
  opt_layer_allocation_count 28;
  pg_list_index_append [[8;1024];[2;4096;128]]
    (List.append (opt_layer_allocations 28) [[7;1024];[8;153600]]) index;
  if index >= 2 then (
    pg_list_index_append (opt_layer_allocations 28) [[7;1024];[8;153600]] (index - 2);
    if index < 254 then pg_layer_shape_at 28 (index - 2))
#pop-options
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_address (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (index:Shared.index 256)
  : GTot (a:larray BF16.t (pg_size index){KA.is_global_array a}) =
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  pg_shape_at index;
  opt_matching_shape_at opt_bf16_allocation_shapes buffers.bf16_buffers index;
  RT.data (List.index buffers.bf16_buffers index)
#pop-options

inline_for_extraction noextract
let pg_storage (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (v:pg_values) (fraction:perm) : slprop =
  forall+ (index:Shared.index 256).
    PA.pts_to (pg_address buffers index) #fraction (pg_slot v index)

inline_for_extraction noextract
let pg_masked_storage (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (v:pg_values) (fraction:perm) (mask:Shared.index 256 -> prop) : slprop =
  forall+ (index:Shared.index 256{mask index}).
    PA.pts_to (pg_address buffers index) #fraction (pg_slot v index)

inline_for_extraction noextract
ghost
fn pg_take (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (#fraction:perm) (mask:Shared.index 256 -> prop)
  (index:Shared.index 256{mask index})
  requires pg_masked_storage buffers v fraction mask
  ensures PA.pts_to (pg_address buffers index) #fraction (pg_slot v index) **
    pg_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 256) i index))

inline_for_extraction noextract
ghost
fn pg_put (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (#fraction:perm) (mask:Shared.index 256 -> prop)
  (index:Shared.index 256{mask index}) (#value:lseq BF16.t (pg_size index))
  requires PA.pts_to (pg_address buffers index) #fraction value **
    pg_masked_storage buffers v fraction (fun i -> mask i /\ ~(eq2 #(Shared.index 256) i index))
  ensures pg_masked_storage buffers (pg_replace v index value) fraction mask

inline_for_extraction noextract
let pg_replace_same (v:pg_values) (index:Shared.index 256)
  : Lemma (pg_replace v index (pg_slot v index) == v) =
  assert (S.equal (pg_replace v index (pg_slot v index)) v);
  S.lemma_eq_elim (pg_replace v index (pg_slot v index)) v

inline_for_extraction noextract
let pg_target (layer:Shared.index 28) (step:nat{step <= 8 /\ (step = 8 ==> layer < 27)})
  : Shared.index 256 =
  if step = 8 then pg_index (layer + 1) 2 else pg_index layer (step + 3)

inline_for_extraction noextract
let pg_step_fact (weights:lseq BF16.t opt_weight_count) (v:pg_values)
  (layer:Shared.index 28) (step:nat{step <= 8 /\ (step = 8 ==> layer < 27)})
  : GTot prop =
  let b = pg_layer_values v layer in
  if step = 0 then b.v_qkv == dec_projection_values 1024 4096 1 2048 false
    b.v_normalized0 (pg_layer_weights weights layer) b.v_embedding
  else if step = 1 then u_rope_prefix b.v_qkv (pg_layer_weights weights layer) b.v_rope b.v_rotated 168
  else if step = 2 then
    (forall (m:u_attention_meaning). u_attention_guard b.v_rotated m ==>
      dec_matrix_rep #8 #2048 b.v_attention (u_attention_model m)) /\
    (forall (c:Shared.index 2048). S.index b.v_attention (14336 + c) == (F.zero #BF16.t))
  else if step = 3 then b.v_residual == dec_projection_values 2048 1024 0 4196352 true
    b.v_attention (pg_layer_weights weights layer) b.v_embedding
  else if step = 4 then u_rms_prefix #8 #opt_weight_count b.v_residual weights b.v_postnorm
    (pg_layer_base layer + 1024) 8
  else if step = 5 then b.v_gateup == dec_projection_values 1024 6144 2 6293760 false
    b.v_postnorm (pg_layer_weights weights layer) b.v_residual
  else if step = 6 then u_silu_prefix b.v_gateup b.v_activated 24576
  else if step = 7 then b.v_hidden == dec_projection_values 3072 1024 0 12585216 true
    b.v_activated (pg_layer_weights weights layer) b.v_residual
  else u_rms_prefix #8 #opt_weight_count b.v_hidden weights (pg_slot v (pg_target layer step))
    (pg_layer_base (layer + 1)) 8

inline_for_extraction noextract
let pg_prior_layer_frame (v:pg_values) (layer:Shared.index 28)
  (step:nat{step <= 8 /\ (step = 8 ==> layer < 27)})
  (value:lseq BF16.t (pg_size (pg_target layer step)))
  (prior:Shared.index 28{prior < layer})
  : Lemma (
    pg_layer_values (pg_replace v (pg_target layer step) value) prior == pg_layer_values v prior /\
    pg_slot (pg_replace v (pg_target layer step) value) 0 == pg_slot v 0 /\
    pg_slot (pg_replace v (pg_target layer step) value) 1 == pg_slot v 1) = ()

inline_for_extraction noextract
let pg_next_norm_related (weights:lseq BF16.t opt_weight_count)
  (v:pg_values) (layer:Shared.index 28{layer < 27})
  : Lemma
    (requires pg_step_fact weights v layer 8)
    (ensures pg_layer_arithmetic weights (layer + 1) (pg_layer_values v (layer + 1)) 0) =
  let at (gamma:Shared.vector real 1024) (xr:Shared.hidden real)
    : Lemma
      (requires Shared.vector_approximates (pg_input_gamma weights (layer + 1)) gamma /\
        Shared.matrix_approximates (u_live_matrix #1024 (pg_layer_values v (layer + 1)).v_embedding) xr)
      (ensures Shared.matrix_approximates (u_live_matrix #1024 (pg_layer_values v (layer + 1)).v_normalized0)
        (Shared.normalize_hidden gamma xr)) =
    u_rms_live_related weights (pg_layer_values v layer).v_hidden (pg_slot v (pg_target layer 8))
    (pg_layer_base (layer + 1)) gamma xr
  in C.forall_intro_2 (C.move_requires_2 at)

inline_for_extraction noextract
let pg_current_layer_progress (weights:lseq BF16.t opt_weight_count) (v:pg_values)
  (layer:Shared.index 28) (step:nat{step < 8})
  (value:lseq BF16.t (pg_size (pg_target layer step)))
  : Lemma
      (requires pg_layer_arithmetic weights layer (pg_layer_values v layer) step /\
        pg_step_fact weights (pg_replace v (pg_target layer step) value) layer step)
      (ensures pg_layer_arithmetic weights layer
        (pg_layer_values (pg_replace v (pg_target layer step) value) layer) (step + 1)) =
  if step = 0 then () else if step = 1 then () else if step = 2 then ()
  else if step = 3 then () else if step = 4 then () else if step = 5 then ()
  else if step = 6 then () else ()

inline_for_extraction noextract
let pg_stage_reaches_next (layer other:nat) (step:nat{step <= 8})
  : Lemma
      (requires layer < other /\ 2 + 9 * other <= 3 + 9 * layer + step)
      (ensures step == 8 /\ other == layer + 1) = ()

inline_for_extraction noextract
let pg_stage_advance (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) (v:pg_values)
  (layer:Shared.index 28{layer > 0}) (step:nat{step <= 8 /\ (step = 8 ==> layer < 27)})
  (value:lseq BF16.t (pg_size (pg_target layer step)))
  : Lemma
      (requires pg_decoder_stage weights prompt v (2 + 9 * layer + step) /\
        pg_step_fact weights (pg_replace v (pg_target layer step) value) layer step)
      (ensures pg_decoder_stage weights prompt
        (pg_replace v (pg_target layer step) value) (3 + 9 * layer + step)) =
  let after = pg_replace v (pg_target layer step) value in
  pg_decoder_stage_definition weights prompt v (2 + 9 * layer + step);
  pg_decoder_stage_definition weights prompt after (3 + 9 * layer + step);
  assert (pg_first_values after == pg_first_values v);
  let at (other:Shared.index 28{other > 0})
    : Lemma (3 + 9 * layer + step >= 2 + 9 * other ==>
      pg_layer_arithmetic weights other (pg_layer_values after other)
        (if 3 + 9 * layer + step >= 10 + 9 * other then 8 else 3 + 9 * layer + step - (2 + 9 * other))) =
    if other < layer then pg_prior_layer_frame v layer step value other
    else if other = layer then (
      if step < 8 then pg_current_layer_progress weights v layer step value
      else assert (pg_layer_values after layer == pg_layer_values v layer))
    else if 3 + 9 * layer + step >= 2 + 9 * other then (
      pg_stage_reaches_next layer other step;
      pg_next_norm_related weights after layer)
  in C.forall_intro at

inline_for_extraction noextract
let pg_core (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (calls:nat{11 <= calls /\ calls <= 253}) : slprop =
  PA.pts_to weights_address weights ** pg_storage buffers v 1.0R **
  pure (pg_decoder_stage weights prompt v calls)

inline_for_extraction noextract
let pg_pending (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (calls:nat{11 <= calls /\ calls <= 253}) : slprop =
  exists* (v:pg_values). pg_core weights_address buffers weights prompt v calls

inline_for_extraction noextract
instance pg_storage_send (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (v:pg_values) (fraction:perm) : is_send_across gpu_of (pg_storage buffers v fraction) =
  let at (i:Shared.index 256)
    : is_send_across gpu_of (PA.pts_to (pg_address buffers i) #fraction (pg_slot v i)) =
    KA.is_send_pts_to (pg_address buffers i) #fraction (pg_slot v i)
  in is_send_across_forevery (fun (i:Shared.index 256) ->
    PA.pts_to (pg_address buffers i) #fraction (pg_slot v i)) gpu_of #at

inline_for_extraction noextract
instance pg_masked_storage_send (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (v:pg_values) (fraction:perm) (mask:Shared.index 256 -> prop)
  : is_send_across gpu_of (pg_masked_storage buffers v fraction mask) =
  let at (i:Shared.index 256{mask i})
    : is_send_across gpu_of (PA.pts_to (pg_address buffers i) #fraction (pg_slot v i)) =
    KA.is_send_pts_to (pg_address buffers i) #fraction (pg_slot v i)
  in is_send_across_forevery (fun (i:Shared.index 256{mask i}) ->
    PA.pts_to (pg_address buffers i) #fraction (pg_slot v i)) gpu_of #at

inline_for_extraction noextract
instance pg_core_send (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (calls:nat{11 <= calls /\ calls <= 253})
  : is_send_across gpu_of (pg_core weights_address buffers weights prompt v calls) =
  is_send_across_star (PA.pts_to weights_address weights)
    (pg_storage buffers v 1.0R ** pure (pg_decoder_stage weights prompt v calls))
    #(KA.is_send_pts_to weights_address #1.0R weights)
    #(is_send_across_star (pg_storage buffers v 1.0R) (pure (pg_decoder_stage weights prompt v calls))
      #(pg_storage_send buffers v 1.0R)
      #(is_send_across_placeless (pure (pg_decoder_stage weights prompt v calls))
        #(placeless_pure (pg_decoder_stage weights prompt v calls))))

inline_for_extraction noextract
instance pg_pending_send (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (calls:nat{11 <= calls /\ calls <= 253})
  : is_send_across gpu_of (pg_pending weights_address buffers weights prompt calls) =
  let at (v:pg_values) : is_send_across gpu_of (pg_core weights_address buffers weights prompt v calls) =
    pg_core_send weights_address buffers weights prompt v calls
  in is_send_across_exists (fun (v:pg_values) -> pg_core weights_address buffers weights prompt v calls) #at

inline_for_extraction noextract
let pg_mask3 (a b c:Shared.index 256) (i:Shared.index 256) : prop =
  i <> a /\ i <> b /\ i <> c

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

inline_for_extraction noextract
ghost
fn pg_split2 (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (a:Shared.index 256) (b:Shared.index 256{a <> b})
  requires pg_storage buffers v 1.0R
  ensures PA.pts_to (pg_address buffers a) (pg_slot v a) **
    PA.pts_to (pg_address buffers b) (pg_slot v b) **
    pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ i <> b)

inline_for_extraction noextract
ghost
fn pg_join2 (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (a:Shared.index 256) (b:Shared.index 256{a <> b}) (#value:lseq BF16.t (pg_size b))
  requires PA.pts_to (pg_address buffers a) (pg_slot v a) **
    PA.pts_to (pg_address buffers b) value **
    pg_masked_storage buffers v 1.0R (fun i -> i <> a /\ i <> b)
  ensures pg_storage buffers (pg_replace v b value) 1.0R

inline_for_extraction noextract
let rec pg_list_drop_step (#a:Type0) (items:list a) (index:nat{index < List.length items})
  : Lemma ((List.splitAt index items)._2 ==
      List.index items index :: (List.splitAt (index + 1) items)._2)
    (decreases index) =
  if index > 0 then pg_list_drop_step (List.tail items) (index - 1)

inline_for_extraction noextract
let pg_tail (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (start:nat{start <= 256})
  : GTot (items:list (RT.tensor BF16.t){List.length items == 256 - start}) =
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  List.lemma_splitAt_snd_length start buffers.bf16_buffers;
  (List.splitAt start buffers.bf16_buffers)._2

inline_for_extraction noextract
type pg_partial_values (start:nat{start <= 256}) = s:lseq (S.seq BF16.t) 256{
  forall (i:Shared.index 256{start <= i}). S.length (S.index s i) == pg_size i}

inline_for_extraction noextract
let pg_collected (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (start:nat{start <= 256}) (v:pg_partial_values start) : slprop =
  forall+ (i:Shared.index 256{start <= i}). PA.pts_to (pg_address buffers i) (S.index v i)

inline_for_extraction noextract
ghost
fn pg_collect_tail (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (start:nat{start <= 256})
  preserves loc gpu_loc
  requires RH.all_contents (pg_tail buffers start)
  returns observed:erased (pg_partial_values start)
  ensures pg_collected buffers start observed

#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_first_cell (first:opt_block_values) (i:Shared.index 12) : GTot (lseq BF16.t (pg_size i)) =
  if i = 0 then first.v_embedding else if i = 1 then first.v_rope
  else if i = 2 then first.v_normalized0 else if i = 3 then first.v_qkv
  else if i = 4 then first.v_rotated else if i = 5 then first.v_attention
  else if i = 6 then first.v_residual else if i = 7 then first.v_postnorm
  else if i = 8 then first.v_gateup else if i = 9 then first.v_activated
  else if i = 10 then first.v_hidden else first.v_normalized1
#pop-options

inline_for_extraction noextract
let pg_initial_values (first:opt_block_values) (tail:pg_partial_values 12) : GTot pg_values =
  S.init_ghost 256 (fun i -> if i < 12 then pg_first_cell first i else S.index tail i)
#push-options "--fuel 30 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_initial_views (first:opt_block_values) (tail:pg_partial_values 12)
  : Lemma (
    pg_first_values (pg_initial_values first tail) == first /\
    (pg_layer_values (pg_initial_values first tail) 1).v_embedding == first.v_hidden /\
    (pg_layer_values (pg_initial_values first tail) 1).v_normalized0 == first.v_normalized1) = ()
#pop-options

inline_for_extraction noextract
let pg_initial_progress (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (first:opt_block_values) (tail:pg_partial_values 12)
  : Lemma
      (requires source_first_block_observation weights prompt first)
      (ensures pg_decoder_stage weights prompt (pg_initial_values first tail) 11) =
  pg_decoder_stage_definition weights prompt (pg_initial_values first tail) 11;
  pg_initial_views first tail;
  assert (pg_input_gamma weights 1 == u_next_gamma_values weights) by
    (FStar.Tactics.V2.norm [delta_only [`%pg_input_gamma; `%pg_layer_base; `%u_next_gamma_values]; iota; zeta; primops];
     FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
ghost
fn pg_insert_collected (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (index:Shared.index 256)
  requires PA.pts_to (pg_address buffers index) (pg_slot v index) ** pg_collected buffers (index + 1) v
  ensures pg_collected buffers index v
#push-options "--fuel 6 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
ghost
fn pg_join_first_block (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#first:opt_block_values) (#tail:pg_partial_values 12)
  requires opt_block_storage (u_allocated_addresses buffers) first ** pg_collected buffers 12 tail
  ensures pg_storage buffers (pg_initial_values first tail) 1.0R
#pop-options

inline_for_extraction noextract
let pg_after_stage (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (calls:nat{11 <= calls /\ calls <= 253}) : slprop =
  on gpu_loc (pg_pending weights_address buffers weights prompt calls **
    (PA.pts_to prompt_address prompt ** RT.contents buffers.token_buffer))

inline_for_extraction noextract
ghost
fn pg_from_first_block (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) ()
  requires source_after_first_block weights_address prompt_address buffers weights prompt
  ensures pg_after_stage weights_address prompt_address buffers weights prompt 11

inline_for_extraction noextract
let pg_rms_input_index (layer:pg_remaining_layer)
  (step:nat{step = 4 \/ (step = 8 /\ layer < 27)}) : Shared.index 256 =
  pg_index layer (if step = 4 then 6 else 10)

inline_for_extraction noextract
let pg_rms_weight_offset (layer:pg_remaining_layer)
  (step:nat{step = 4 \/ (step = 8 /\ layer < 27)}) : nat =
  if step = 4 then pg_layer_base layer + 1024 else pg_layer_base (layer + 1)

inline_for_extraction noextract
let pg_rms_step_binding (layer:pg_remaining_layer)
  (step:nat{step = 4 \/ (step = 8 /\ layer < 27)})
  : Lemma (
    pg_size (pg_rms_input_index layer step) == 8192 /\
    pg_size (pg_target layer step) == 8192 /\
    pg_rms_input_index layer step <> pg_target layer step /\
    pg_rms_weight_offset layer step + 1024 <= opt_weight_count) = ()

inline_for_extraction noextract
val pg_rms_stage_body
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (layer:pg_remaining_layer) (step:nat{step = 4 \/ (step = 8 /\ layer < 27)})
  (input:larray BF16.t 8192{input == pg_address buffers (pg_rms_input_index layer step)})
  (output:larray BF16.t 8192{output == pg_address buffers (pg_target layer step)})
  (offset:sz{SZ.v offset == pg_rms_weight_offset layer step})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values)
  : unit -> Pulse.Lib.Core.stt unit
      (gpu ** (pg_core weights_address buffers weights prompt v (2 + 9 * layer + step)))
      (fun _ -> gpu ** (pg_pending weights_address buffers weights prompt (3 + 9 * layer + step)))


inline_for_extraction noextract
let pg_rms_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (layer:pg_remaining_layer) (step:nat{step = 4 \/ (step = 8 /\ layer < 27)})
  (input:larray BF16.t 8192{input == pg_address buffers (pg_rms_input_index layer step)})
  (output:larray BF16.t 8192{output == pg_address buffers (pg_target layer step)})
  (offset:sz{SZ.v offset == pg_rms_weight_offset layer step})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased pg_values)
  : pg_source_descriptor
      (pg_core weights_address buffers weights prompt v (2 + 9 * layer + step))
      (pg_pending weights_address buffers weights prompt (3 + 9 * layer + step)) 1sz opt_scalar_shared_desc =
  let raw:KS.kernel_desc_1_1
      (pg_core weights_address buffers weights prompt v (2 + 9 * layer + step))
      (pg_pending weights_address buffers weights prompt (3 + 9 * layer + step)) = {
    f = pg_rms_stage_body weights_address buffers layer step input output offset #weights #prompt #v;
    full_pre_sendable = pg_core_send weights_address buffers weights prompt v (2 + 9 * layer + step);
    full_post_sendable = pg_pending_send weights_address buffers weights prompt (3 + 9 * layer + step)
  } in opt_scalar_descriptor
    #(pg_core weights_address buffers weights prompt v (2 + 9 * layer + step))
    #(pg_pending weights_address buffers weights prompt (3 + 9 * layer + step)) raw

inline_for_extraction noextract
val pg_silu_stage_body
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 49152{input == pg_address buffers (pg_index layer 8)})
  (output:larray BF16.t 24576{output == pg_address buffers (pg_index layer 9)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values)
  : unit -> Pulse.Lib.Core.stt unit
      (gpu ** (pg_core weights_address buffers weights prompt v (8 + 9 * layer)))
      (fun _ -> gpu ** (pg_pending weights_address buffers weights prompt (9 + 9 * layer)))


inline_for_extraction noextract
let pg_silu_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 49152{input == pg_address buffers (pg_index layer 8)})
  (output:larray BF16.t 24576{output == pg_address buffers (pg_index layer 9)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased pg_values)
  : pg_source_descriptor
      (pg_core weights_address buffers weights prompt v (8 + 9 * layer))
      (pg_pending weights_address buffers weights prompt (9 + 9 * layer)) 1sz opt_scalar_shared_desc =
  let raw:KS.kernel_desc_1_1
      (pg_core weights_address buffers weights prompt v (8 + 9 * layer))
      (pg_pending weights_address buffers weights prompt (9 + 9 * layer)) = {
    f = pg_silu_stage_body weights_address buffers layer input output #weights #prompt #v;
    full_pre_sendable = pg_core_send weights_address buffers weights prompt v (8 + 9 * layer);
    full_post_sendable = pg_pending_send weights_address buffers weights prompt (9 + 9 * layer)
  } in opt_scalar_descriptor
    #(pg_core weights_address buffers weights prompt v (8 + 9 * layer))
    #(pg_pending weights_address buffers weights prompt (9 + 9 * layer)) raw

inline_for_extraction noextract
val pg_rope_stage_body
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 3)})
  (output:larray BF16.t 32768{output == pg_address buffers (pg_index layer 4)})
  (table:larray BF16.t 1048576{table == pg_address buffers 1})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values)
  : unit -> Pulse.Lib.Core.stt unit
      (gpu ** (pg_core weights_address buffers weights prompt v (3 + 9 * layer)))
      (fun _ -> gpu ** (pg_pending weights_address buffers weights prompt (4 + 9 * layer)))


inline_for_extraction noextract
let pg_rope_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 3)})
  (output:larray BF16.t 32768{output == pg_address buffers (pg_index layer 4)})
  (table:larray BF16.t 1048576{table == pg_address buffers 1})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (v:erased pg_values)
  : pg_source_descriptor
      (pg_core weights_address buffers weights prompt v (3 + 9 * layer))
      (pg_pending weights_address buffers weights prompt (4 + 9 * layer)) 1sz opt_scalar_shared_desc =
  let raw:KS.kernel_desc_1_1
      (pg_core weights_address buffers weights prompt v (3 + 9 * layer))
      (pg_pending weights_address buffers weights prompt (4 + 9 * layer)) = {
    f = pg_rope_stage_body weights_address buffers layer input output table window #weights #prompt #v;
    full_pre_sendable = pg_core_send weights_address buffers weights prompt v (3 + 9 * layer);
    full_post_sendable = pg_pending_send weights_address buffers weights prompt (4 + 9 * layer)
  } in opt_scalar_descriptor
    #(pg_core weights_address buffers weights prompt v (3 + 9 * layer))
    #(pg_pending weights_address buffers weights prompt (4 + 9 * layer)) raw

inline_for_extraction noextract
let pg_linear_step (slot:Shared.index 4) : nat =
  if slot = 0 then 0 else if slot = 1 then 3 else if slot = 2 then 5 else 7

inline_for_extraction noextract
let pg_linear_input_index (layer:pg_remaining_layer) (slot:Shared.index 4) : Shared.index 256 =
  pg_index layer (if slot = 0 then 2 else if slot = 1 then 5 else if slot = 2 then 7 else 9)

inline_for_extraction noextract
let pg_linear_residual_index (layer:pg_remaining_layer) (slot:Shared.index 4) : Shared.index 256 =
  pg_index layer (if slot < 2 then 0 else 6)

inline_for_extraction noextract
let pg_linear_target_index (layer:pg_remaining_layer) (slot:Shared.index 4) : Shared.index 256 =
  pg_target layer (pg_linear_step slot)

inline_for_extraction noextract
let pg_linear_binding (layer:pg_remaining_layer) (slot:Shared.index 4)
  : Lemma (
    pg_size (pg_linear_input_index layer slot) == 8 * u_linear_width slot /\
    pg_size (pg_linear_residual_index layer slot) == 8192 /\
    pg_size (pg_linear_target_index layer slot) == 8 * u_linear_cols slot /\
    pg_linear_input_index layer slot <> pg_linear_residual_index layer slot /\
    pg_linear_input_index layer slot <> pg_linear_target_index layer slot /\
    pg_linear_residual_index layer slot <> pg_linear_target_index layer slot) =
  if slot = 0 then () else if slot = 1 then () else if slot = 2 then () else ()

inline_for_extraction noextract
let pg_linear_result (weights:lseq BF16.t opt_weight_count) (v:pg_values)
  (layer:pg_remaining_layer) (slot:Shared.index 4)
  : GTot (lseq BF16.t (8 * u_linear_cols slot)) =
  u_linear_shape slot;
  pg_linear_binding layer slot;
  dec_projection_values (u_linear_width slot) (u_linear_cols slot) (u_linear_layout slot)
    (u_linear_offset slot) (u_linear_add slot) (pg_slot v (pg_linear_input_index layer slot))
    (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot))

inline_for_extraction noextract
let pg_linear_progress (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (layer:pg_remaining_layer) (slot:Shared.index 4)
  : Lemma
      (requires pg_decoder_stage weights prompt v (2 + 9 * layer + pg_linear_step slot))
      (ensures pg_decoder_stage weights prompt
        (pg_replace v (pg_linear_target_index layer slot) (pg_linear_result weights v layer slot))
        (3 + 9 * layer + pg_linear_step slot)) =
  pg_linear_binding layer slot;
  assert (pg_step_fact weights
    (pg_replace v (pg_linear_target_index layer slot) (pg_linear_result weights v layer slot))
    layer (pg_linear_step slot));
  pg_stage_advance weights prompt v layer (pg_linear_step slot) (pg_linear_result weights v layer slot)

inline_for_extraction noextract
let pg_linear_frame (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (layer:pg_remaining_layer) (slot:Shared.index 4) : slprop =
  u_weight_window_frame weights_address weights (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements) 1.0R **
  pg_masked_storage buffers v 1.0R
    (pg_mask3 (pg_linear_input_index layer slot) (pg_linear_residual_index layer slot) (pg_linear_target_index layer slot)) **
  pure (pg_decoder_stage weights prompt v (2 + 9 * layer + pg_linear_step slot))

inline_for_extraction noextract
val pg_linear_before
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer) (slot:Shared.index 4)
  (input:larray BF16.t (8 * u_linear_width slot){input == pg_address buffers (pg_linear_input_index layer slot)})
  (residual:larray BF16.t 8192{residual == pg_address buffers (pg_linear_residual_index layer slot)})
  (output:larray BF16.t (8 * u_linear_cols slot){output == pg_address buffers (pg_linear_target_index layer slot)})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (#v:erased pg_values)
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_core weights_address buffers weights prompt v (2 + 9 * layer + pg_linear_step slot))
      (fun _ -> opt_linear_pre input window residual output (pg_slot v (pg_linear_input_index layer slot))
      (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot)) 1.0R 1.0R 1.0R **
    pg_linear_frame weights_address buffers weights prompt v layer slot)


inline_for_extraction noextract
val pg_linear_after
  (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer) (slot:Shared.index 4)
  (input:larray BF16.t (8 * u_linear_width slot){input == pg_address buffers (pg_linear_input_index layer slot)})
  (residual:larray BF16.t 8192{residual == pg_address buffers (pg_linear_residual_index layer slot)})
  (output:larray BF16.t (8 * u_linear_cols slot){output == pg_address buffers (pg_linear_target_index layer slot)})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (#v:erased pg_values)
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (opt_linear_post input window residual output (pg_slot v (pg_linear_input_index layer slot))
      (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot))
      (pg_linear_result weights v layer slot) 1.0R 1.0R 1.0R **
    pg_linear_frame weights_address buffers weights prompt v layer slot)
      (fun _ -> pg_pending weights_address buffers weights prompt (3 + 9 * layer + pg_linear_step slot))


inline_for_extraction noextract
let pg_linear_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer) (slot:Shared.index 4)
  (input:larray BF16.t (8 * u_linear_width slot){input == pg_address buffers (pg_linear_input_index layer slot)})
  (residual:larray BF16.t 8192{residual == pg_address buffers (pg_linear_residual_index layer slot)})
  (output:larray BF16.t (8 * u_linear_cols slot){output == pg_address buffers (pg_linear_target_index layer slot)})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (width:szp{SZ.v width == u_linear_width slot}) (cols:szp{SZ.v cols == u_linear_cols slot})
  (layout:sz{SZ.v layout == u_linear_layout slot}) (offset:sz{SZ.v offset == u_linear_offset slot})
  (add:bool{add == u_linear_add slot})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (v:erased pg_values)
  : pg_source_descriptor
      (pg_core weights_address buffers weights prompt v (2 + 9 * layer + pg_linear_step slot))
      (pg_pending weights_address buffers weights prompt (3 + 9 * layer + pg_linear_step slot)) 32sz opt_linear_shared_desc =
  pg_linear_binding layer slot;
  u_linear_shape slot;
  u_weight_window_shape weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements);
  let code = u_linear_source_descriptor width cols layout offset add input window residual output
    #(hide (pg_slot v (pg_linear_input_index layer slot))) #(hide (pg_layer_weights weights layer))
    #(hide (pg_slot v (pg_linear_residual_index layer slot))) in
  let pre = opt_linear_pre input window residual output (pg_slot v (pg_linear_input_index layer slot))
    (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot)) 1.0R 1.0R 1.0R in
  let post = opt_linear_post input window residual output (pg_slot v (pg_linear_input_index layer slot))
    (pg_layer_weights weights layer) (pg_slot v (pg_linear_residual_index layer slot))
    (pg_linear_result weights v layer slot) 1.0R 1.0R 1.0R in
  let frame = pg_linear_frame weights_address buffers weights prompt v layer slot in
  let framed = opt_descriptor_with_frame #pre #post code frame in
  opt_descriptor_refine #(pre ** frame) #(post ** frame)
    #(pg_core weights_address buffers weights prompt v (2 + 9 * layer + pg_linear_step slot))
    #(pg_pending weights_address buffers weights prompt (3 + 9 * layer + pg_linear_step slot)) framed
    (pg_linear_before weights_address buffers layer slot input residual output window #weights #prompt #v)
    (pg_linear_after weights_address buffers layer slot input residual output window #weights #prompt #v)

inline_for_extraction noextract
let pg_attention_frame (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (layer:pg_remaining_layer) : slprop =
  PA.pts_to weights_address weights **
  pg_masked_storage buffers v 1.0R (fun i -> i <> pg_index layer 4 /\ i <> pg_index layer 5) **
  pure (pg_decoder_stage weights prompt v (4 + 9 * layer))

inline_for_extraction noextract
val pg_attention_before (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 4)})
  (output:larray BF16.t 16384{output == pg_address buffers (pg_index layer 5)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (#v:erased pg_values)
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_core weights_address buffers weights prompt v (4 + 9 * layer))
      (fun _ -> u_attention_pre input output (pg_slot v (pg_index layer 4)) (pg_slot v (pg_index layer 5)) 1.0R **
    pg_attention_frame weights_address buffers weights prompt v layer)


inline_for_extraction noextract
val pg_attention_after (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 4)})
  (output:larray BF16.t 16384{output == pg_address buffers (pg_index layer 5)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (#v:erased pg_values)
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (u_attention_post input output (pg_slot v (pg_index layer 4)) 1.0R **
    pg_attention_frame weights_address buffers weights prompt v layer)
      (fun _ -> pg_pending weights_address buffers weights prompt (5 + 9 * layer))


inline_for_extraction noextract
let pg_attention_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 4)})
  (output:larray BF16.t 16384{output == pg_address buffers (pg_index layer 5)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (v:erased pg_values)
  : pg_source_descriptor
      (pg_core weights_address buffers weights prompt v (4 + 9 * layer))
      (pg_pending weights_address buffers weights prompt (5 + 9 * layer)) 32sz opt_attention_shared_desc =
  let code = u_attention_descriptor input output
    #(hide (pg_slot v (pg_index layer 4))) #(hide (pg_slot v (pg_index layer 5))) #1.0R in
  let pre = u_attention_pre input output (pg_slot v (pg_index layer 4)) (pg_slot v (pg_index layer 5)) 1.0R in
  let post = u_attention_post input output (pg_slot v (pg_index layer 4)) 1.0R in
  let frame = pg_attention_frame weights_address buffers weights prompt v layer in
  let framed = opt_descriptor_with_frame #pre #post code frame in
  opt_descriptor_refine #(pre ** frame) #(post ** frame)
    #(pg_core weights_address buffers weights prompt v (4 + 9 * layer))
    #(pg_pending weights_address buffers weights prompt (5 + 9 * layer)) framed
    (pg_attention_before weights_address buffers layer input output #weights #prompt #v)
    (pg_attention_after weights_address buffers layer input output #weights #prompt #v)

inline_for_extraction noextract
fn pg_data_ptr (stream:ST.stream_t) (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (index:Shared.index 256)
  preserves cpu ** opt_allocated_owners stream buffers
  returns address:larray BF16.t (pg_size index)
  ensures pure (address == pg_address buffers index /\ KA.is_global_array address)

inline_for_extraction noextract
fn pg_pending_weight_window
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (lo:HostSize.t) (hi:erased nat{HostSize.v lo <= hi /\ hi <= opt_weight_count})
  (#contents:erased (lseq BF16.t opt_weight_count))
  preserves cpu ** PA.pts_to_mask weights #1.0R (u_metadata_mask contents) (fun _ -> False)
  returns window:PA.array BF16.t
  ensures rewrites_to window (u_weight_window weights (HostSize.v lo) hi)
  ensures pure (PA.length window == hi - HostSize.v lo /\ KA.is_global_array window)

inline_for_extraction noextract
val pg_expand_pending (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (calls:nat{11 <= calls /\ calls <= 253}) (extra:slprop)
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (on gpu_loc (pg_pending weights_address buffers weights prompt calls ** extra))
      (fun _ -> on gpu_loc (exists* (v:pg_values). pg_core weights_address buffers weights prompt v calls ** extra))


inline_for_extraction noextract
val pg_launch_step (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (calls:nat{11 <= calls /\ calls < 253})
  (after:nat{after == calls + 1})
  (family:opt_pending_family pg_values
    (fun v -> pg_core weights_address buffers weights prompt v calls)
    (fun _ -> pg_pending weights_address buffers weights prompt after))
  (stream:ST.stream_t) (#start:E.epoch_t)
  : Pulse.Lib.Core.stt unit
      (cpu ** ST.stream_live stream ** (E.epoch_live stream (start + calls) **
    P.pledge0 (E.epoch_done stream (start + calls))
      (pg_after_stage weights_address prompt_address buffers weights prompt calls)))
      (fun _ -> cpu ** ST.stream_live stream ** (E.epoch_live stream (start + after) **
    P.pledge0 (E.epoch_done stream (start + after))
      (pg_after_stage weights_address prompt_address buffers weights prompt after)))


inline_for_extraction noextract
let pg_linear_family
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer) (slot:Shared.index 4)
  (input:larray BF16.t (8 * u_linear_width slot){input == pg_address buffers (pg_linear_input_index layer slot)})
  (residual:larray BF16.t 8192{residual == pg_address buffers (pg_linear_residual_index layer slot)})
  (output:larray BF16.t (8 * u_linear_cols slot){output == pg_address buffers (pg_linear_target_index layer slot)})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (width:szp{SZ.v width == u_linear_width slot}) (cols:szp{SZ.v cols == u_linear_cols slot})
  (layout:sz{SZ.v layout == u_linear_layout slot}) (offset:sz{SZ.v offset == u_linear_offset slot})
  (add:bool{add == u_linear_add slot})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family pg_values
      (fun v -> pg_core weights_address buffers weights prompt v (2 + 9 * layer + pg_linear_step slot))
      (fun _ -> pg_pending weights_address buffers weights prompt (3 + 9 * layer + pg_linear_step slot)) = {
    pending_threads = 32sz; pending_shared = opt_linear_shared_desc;
    pending_code = pg_linear_stage_descriptor weights_address buffers layer slot input residual output window
      width cols layout offset add #weights #prompt
  }

inline_for_extraction noextract
let pg_rms_family
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (step:nat{step = 4 \/ (step = 8 /\ layer < 27)})
  (input:larray BF16.t 8192{input == pg_address buffers (pg_rms_input_index layer step)})
  (output:larray BF16.t 8192{output == pg_address buffers (pg_target layer step)})
  (offset:sz{SZ.v offset == pg_rms_weight_offset layer step})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family pg_values
      (fun v -> pg_core weights_address buffers weights prompt v (2 + 9 * layer + step))
      (fun _ -> pg_pending weights_address buffers weights prompt (3 + 9 * layer + step)) = {
    pending_threads = 1sz; pending_shared = opt_scalar_shared_desc;
    pending_code = pg_rms_stage_descriptor weights_address buffers layer step input output offset #weights #prompt
  }

inline_for_extraction noextract
let pg_rope_family
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 3)})
  (output:larray BF16.t 32768{output == pg_address buffers (pg_index layer 4)})
  (table:larray BF16.t 1048576{table == pg_address buffers 1})
  (window:larray BF16.t dec_weight_elements{
    window == u_weight_window weights_address (pg_layer_base layer) (pg_layer_base layer + dec_weight_elements)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family pg_values
      (fun v -> pg_core weights_address buffers weights prompt v (3 + 9 * layer))
      (fun _ -> pg_pending weights_address buffers weights prompt (4 + 9 * layer)) = {
    pending_threads = 1sz; pending_shared = opt_scalar_shared_desc;
    pending_code = pg_rope_stage_descriptor weights_address buffers layer input output table window #weights #prompt
  }

inline_for_extraction noextract
let pg_silu_family
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 49152{input == pg_address buffers (pg_index layer 8)})
  (output:larray BF16.t 24576{output == pg_address buffers (pg_index layer 9)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family pg_values
      (fun v -> pg_core weights_address buffers weights prompt v (8 + 9 * layer))
      (fun _ -> pg_pending weights_address buffers weights prompt (9 + 9 * layer)) = {
    pending_threads = 1sz; pending_shared = opt_scalar_shared_desc;
    pending_code = pg_silu_stage_descriptor weights_address buffers layer input output #weights #prompt
  }

inline_for_extraction noextract
let pg_attention_family
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers}) (layer:pg_remaining_layer)
  (input:larray BF16.t 32768{input == pg_address buffers (pg_index layer 4)})
  (output:larray BF16.t 16384{output == pg_address buffers (pg_index layer 5)})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family pg_values
      (fun v -> pg_core weights_address buffers weights prompt v (4 + 9 * layer))
      (fun _ -> pg_pending weights_address buffers weights prompt (5 + 9 * layer)) = {
    pending_threads = 32sz; pending_shared = opt_attention_shared_desc;
    pending_code = pg_attention_stage_descriptor weights_address buffers layer input output #weights #prompt
  }

inline_for_extraction noextract
let pg_layer_end (layer:pg_remaining_layer) : nat = if layer < 27 then 11 + 9 * layer else 253

inline_for_extraction noextract
val pg_enqueue_layer
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (layer:pg_remaining_layer)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t)
  : Pulse.Lib.Core.stt unit
      (cpu ** ST.stream_live stream ** opt_allocated_owners stream buffers **
    PA.pts_to_mask weights_address #1.0R (u_metadata_mask weights) (fun _ -> False) ** (E.epoch_live stream (start + (2 + 9 * layer)) **
    P.pledge0 (E.epoch_done stream (start + (2 + 9 * layer)))
      (pg_after_stage weights_address prompt_address buffers weights prompt (2 + 9 * layer))))
      (fun _ -> cpu ** ST.stream_live stream ** opt_allocated_owners stream buffers **
    PA.pts_to_mask weights_address #1.0R (u_metadata_mask weights) (fun _ -> False) ** (E.epoch_live stream (start + pg_layer_end layer) **
    P.pledge0 (E.epoch_done stream (start + pg_layer_end layer))
      (pg_after_stage weights_address prompt_address buffers weights prompt (pg_layer_end layer))))


inline_for_extraction noextract
let pg_loop_epoch (layer:nat{1 <= layer /\ layer <= 28}) : nat =
  if layer < 28 then 2 + 9 * layer else 253

inline_for_extraction noextract
let pg_layer_next (layer:pg_remaining_layer)
  : Lemma (pg_layer_end layer == pg_loop_epoch (layer + 1)) = ()

inline_for_extraction noextract
val pg_enqueue_remaining_layers
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t)
  : Pulse.Lib.Core.stt unit
      (cpu ** ST.stream_live stream ** opt_allocated_owners stream buffers **
    PA.pts_to_mask weights_address #1.0R (u_metadata_mask weights) (fun _ -> False) ** (E.epoch_live stream (start + 11) **
    P.pledge0 (E.epoch_done stream (start + 11)) (pg_after_stage weights_address prompt_address buffers weights prompt 11)))
      (fun _ -> cpu ** ST.stream_live stream ** opt_allocated_owners stream buffers **
    PA.pts_to_mask weights_address #1.0R (u_metadata_mask weights) (fun _ -> False) ** (E.epoch_live stream (start + 253) **
    P.pledge0 (E.epoch_done stream (start + 253)) (pg_after_stage weights_address prompt_address buffers weights prompt 253)))


inline_for_extraction noextract
instance pg_projection_width : concrete_sz 1024 = { x = 1024sz; }

inline_for_extraction noextract
instance pg_projection_rows : concrete_sz 7 = { x = 7sz; }

inline_for_extraction noextract
let pg_projection_a (weights:lseq BF16.t opt_weight_count)
  (block:nat{block < 153600 /\ block % 16 == 0}) (step:Shared.index 64)
  : GTot (Chest.chest2 BF16.t 16 16) =
  Chest.mk2 (fun row col ->
    if block + row < 151936 then
      S.index weights ((block + row) * 1024 + step * 16 + col)
    else F.zero)

inline_for_extraction noextract
let pg_projection_b (hidden:lseq BF16.t 7168) (step:Shared.index 64)
  : GTot (Chest.chest2 BF16.t 16 8) =
  Chest.mk2 (fun col row ->
    if row < 7 then S.index hidden (row * 1024 + step * 16 + col) else F.zero)

inline_for_extraction noextract
let pg_projection_live_block (block:nat{block < 151936 /\ block % 16 == 0})
  : Lemma (block + 16 <= 151936) = ()

inline_for_extraction noextract
let pg_projection_padding_a (weights:lseq BF16.t opt_weight_count)
  (block:nat{151936 <= block /\ block < 153600 /\ block % 16 == 0})
  (step:Shared.index 64)
  : Lemma (pg_projection_a weights block step == Chest.const (16 @| 16 @| INil) (F.zero #BF16.t)) =
  Chest.ext (pg_projection_a weights block step) (Chest.const (16 @| 16 @| INil) (F.zero #BF16.t))

inline_for_extraction noextract
let pg_projection_b7_cell (hidden:lseq BF16.t 7168) (step:Shared.index 64)
  (col:Shared.index 16) (row:Shared.index 7)
  : Lemma (Chest.acc2 (ematrix_subtile (Tensor.from_seq (cm 1024 7) hidden) 16 7 step 0) col row ==
      S.index hidden (row * 1024 + step * 16 + col)) =
  Kuiper.EMatrix.Tiling.subtile_acc2 (Tensor.from_seq (cm 1024 7) hidden) 16 7 step 0 col row

inline_for_extraction noextract
let pg_projection_b_matches_registers (hidden:lseq BF16.t 7168) (step:Shared.index 64)
  : Lemma (pg_projection_b hidden step ==
      Registers.pad_b7 (ematrix_subtile (Tensor.from_seq (cm 1024 7) hidden) 16 7 step 0)) =
  let at (col:Shared.index 16) (row:Shared.index 8)
    : Lemma (Chest.acc2 (pg_projection_b hidden step) col row ==
        Chest.acc2 (Registers.pad_b7
          (ematrix_subtile (Tensor.from_seq (cm 1024 7) hidden) 16 7 step 0)) col row) =
    if row < 7 then pg_projection_b7_cell hidden step col row
  in
  C.forall_intro_2 at;
  Chest.ext (pg_projection_b hidden step)
    (Registers.pad_b7 (ematrix_subtile (Tensor.from_seq (cm 1024 7) hidden) 16 7 step 0))

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

inline_for_extraction noextract
let rec pg_projection_acc (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (block:nat{block < 153600 /\ block % 16 == 0}) (steps:nat{steps <= 64})
  : GTot (Chest.chest2 F32.t 16 8) (decreases steps) =
  if steps = 0 then Chest.const (16 @| 8 @| INil) (F.zero #F32.t)
  else MMA.emma (pg_projection_acc weights hidden block (steps - 1))
    (pg_projection_a weights block (steps - 1)) (pg_projection_b hidden (steps - 1))

inline_for_extraction noextract
let pg_projection_tile_values (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (block:nat{block < 153600 /\ block % 16 == 0}) : GTot (lseq F32.t 128) =
  Tensor.to_seq (rm 16 8) (pg_projection_acc weights hidden block 64)

inline_for_extraction noextract
let pg_projection_output (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (old:lseq BF16.t 1228800) : GTot (lseq BF16.t 1228800) =
  S.init_ghost 1228800 (fun i ->
    if i < 1075200 then
      let row = i / 153600 in
      let col = i % 153600 in
      Casts.fcast #F32.t #BF16.t (Chest.acc2 (pg_projection_acc weights hidden (col / 16 * 16) 64) (col % 16) row)
    else S.index old i)

inline_for_extraction noextract
let pg_projection_live_a (weights:lseq BF16.t opt_weight_count)
  (block:nat{block < 151936 /\ block % 16 == 0}) (step:Shared.index 64)
  : Lemma (pg_projection_a weights block step ==
    ematrix_subtile (Tensor.from_seq (rm 16 1024) (S.slice weights (block * 1024) ((block + 16) * 1024)))
      16 16 0 step) =
  pg_projection_live_block block;
  let at (row:Shared.index 16) (col:Shared.index 16)
    : Lemma (Chest.acc2 (pg_projection_a weights block step) row col ==
      Chest.acc2 (ematrix_subtile (Tensor.from_seq (rm 16 1024)
        (S.slice weights (block * 1024) ((block + 16) * 1024))) 16 16 0 step) row col) =
    Kuiper.EMatrix.Tiling.subtile_acc2 (Tensor.from_seq (rm 16 1024)
      (S.slice weights (block * 1024) ((block + 16) * 1024))) 16 16 0 step row col
  in
  C.forall_intro_2 at;
  Chest.ext (pg_projection_a weights block step)
    (ematrix_subtile (Tensor.from_seq (rm 16 1024) (S.slice weights (block * 1024) ((block + 16) * 1024))) 16 16 0 step)

inline_for_extraction noextract
let pg_zero_a_binding (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (block:nat{151936 <= block /\ block < 153600 /\ block % 16 == 0}) (step:Shared.index 64)
  : Lemma (pg_projection_a weights block step == ZeroA.operand_a /\
      pg_projection_b hidden step == Registers.pad_b7 (ZeroA.operand_b7 hidden step)) =
  pg_projection_padding_a weights block step;
  Chest.ext (pg_projection_b hidden step) (Registers.pad_b7 (ZeroA.operand_b7 hidden step))

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

inline_for_extraction noextract
fn pg_projection_tile_mma
  (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168)
  (tile:larray F32.t 128) (block:sz{block < 153600 /\ block % 16 == 0}) (tid:szlt 32)
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq F32.t 128)) (#fw #fh:perm)
  preserves gpu ** thread_id 32 tid ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires tile |-> Frac (1.0R /. 32.0R) old
  ensures tile |-> Frac (1.0R /. 32.0R) (pg_projection_tile_values wv hv block)

inline_for_extraction noextract
let pg_projection_written (ys expected:lseq BF16.t 1228800) (block done:nat) : prop =
  forall (row:Shared.index 7) (col:Shared.index 153600).
    (col < block \/ (block <= col /\ col < block + 16 /\ row * 16 + col - block < done)) ==>
      S.index ys (row * 153600 + col) == S.index expected (row * 153600 + col)

inline_for_extraction noextract
let pg_projection_write (before:lseq BF16.t 1228800) (expected:lseq BF16.t 1228800)
  (block:nat{block < 153600 /\ block % 16 == 0}) (row:Shared.index 7) (col:Shared.index 16)
  : Lemma
      (requires pg_projection_written before expected block (row * 16 + col))
      (ensures pg_projection_written
        (S.upd before (row * 153600 + block + col) (S.index expected (row * 153600 + block + col)))
        expected block (row * 16 + col + 1)) = ()

inline_for_extraction noextract
let pg_projection_outside (old ys:lseq BF16.t 1228800) : prop =
  forall (i:nat{1075200 <= i /\ i < 1228800}). S.index ys i == S.index old i

inline_for_extraction noextract
let pg_projection_owned (output:larray BF16.t 1228800)
  (old expected:lseq BF16.t 1228800) (block:nat) : slprop =
  exists* (ys:lseq BF16.t 1228800). output |-> ys **
    pure (pg_projection_written ys expected block 0 /\ pg_projection_outside old ys)

inline_for_extraction noextract
fn pg_projection_write_tile (output:larray BF16.t 1228800) (tile:larray F32.t 128)
  (block:sz{block < 153600 /\ block % 16 == 0})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  preserves gpu ** tile |-> pg_projection_tile_values weights hidden block
  requires pg_projection_owned output old (pg_projection_output weights hidden old) block
  ensures pg_projection_owned output old (pg_projection_output weights hidden old) (block + 16)

inline_for_extraction noextract
fn pg_projection_owner_write (output:larray BF16.t 1228800) (tile:larray F32.t 128)
  (block:sz{block < 153600 /\ block % 16 == 0}) (tid:szlt 32)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  preserves gpu ** when_ (SZ.v tid == 0) (tile |-> pg_projection_tile_values weights hidden block)
  requires when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) block)
  ensures when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) (block + 16))

inline_for_extraction noextract
ghost
fn pg_projection_begin (output:larray BF16.t 1228800) (tid:szlt 32)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  requires when_ (SZ.v tid == 0) (output |-> old)
  ensures when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) 0)

inline_for_extraction noextract
let pg_projection_finished (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (old ys:lseq BF16.t 1228800)
  : Lemma
      (requires pg_projection_written ys (pg_projection_output weights hidden old) 153600 0 /\
        pg_projection_outside old ys)
      (ensures ys == pg_projection_output weights hidden old) =
  let expected = pg_projection_output weights hidden old in
  let at (i:Shared.index 1228800)
    : Lemma (S.index ys i == S.index expected i) =
    if i < 1075200 then (
      assert (i / 153600 < 7);
      assert (i == (i / 153600) * 153600 + i % 153600))
    else ()
  in
  C.forall_intro at;
  S.lemma_eq_intro ys expected;
  S.lemma_eq_elim ys expected

inline_for_extraction noextract
ghost
fn pg_projection_finish (output:larray BF16.t 1228800) (tid:szlt 32)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#hidden:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  requires when_ (SZ.v tid == 0) (pg_projection_owned output old (pg_projection_output weights hidden old) 153600)
  ensures when_ (SZ.v tid == 0) (output |-> pg_projection_output weights hidden old)

inline_for_extraction noextract
let pg_projection_previous (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (old:lseq F32.t 128) (block:nat{block <= 153600 /\ block % 16 == 0}) : GTot (lseq F32.t 128) =
  if block = 0 then old else pg_projection_tile_values weights hidden (block - 16)

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

inline_for_extraction noextract
let pg_projection_pre (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168)
  (output:larray BF16.t 1228800) (wv:lseq BF16.t opt_weight_count)
  (hv:lseq BF16.t 7168) (old:lseq BF16.t 1228800) : slprop =
  weights |-> wv ** hidden |-> hv ** output |-> old

inline_for_extraction noextract
let pg_projection_post (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168)
  (output:larray BF16.t 1228800) (wv:lseq BF16.t opt_weight_count)
  (hv:lseq BF16.t 7168) (old:lseq BF16.t 1228800) : slprop =
  weights |-> wv ** hidden |-> hv ** output |-> pg_projection_output wv hv old

inline_for_extraction noextract
let pg_projection_kpre (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168)
  (output:larray BF16.t 1228800) (tile:larray F32.t 128) (wv:lseq BF16.t opt_weight_count)
  (hv:lseq BF16.t 7168) (old:lseq BF16.t 1228800) (tid:nat) : slprop =
  (weights |-> Frac (1.0R /. 32.0R) wv ** hidden |-> Frac (1.0R /. 32.0R) hv) **
  (exists* (tv:lseq F32.t 128). tile |-> Frac (1.0R /. 32.0R) tv) **
  when_ (tid == 0) (output |-> old)

inline_for_extraction noextract
let pg_projection_kpost (weights:larray BF16.t opt_weight_count) (hidden:larray BF16.t 7168)
  (output:larray BF16.t 1228800) (tile:larray F32.t 128) (wv:lseq BF16.t opt_weight_count)
  (hv:lseq BF16.t 7168) (old:lseq BF16.t 1228800) (tid:nat) : slprop =
  (weights |-> Frac (1.0R /. 32.0R) wv ** hidden |-> Frac (1.0R /. 32.0R) hv) **
  tile |-> Frac (1.0R /. 32.0R) (pg_projection_tile_values wv hv 153584) **
  when_ (tid == 0) (output |-> pg_projection_output wv hv old)

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

inline_for_extraction noextract
let pg_projection_pre_send
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (hidden:larray BF16.t 7168{KA.is_global_array hidden})
  (output:larray BF16.t 1228800{KA.is_global_array output})
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  (sh:SH.c_shmems opt_linear_shared_desc) (_:squash (SH.c_shmems_inv sh)) (tid:nat)
  : is_send_across block_of (pg_projection_kpre weights hidden output (opt_linear_tile sh) wv hv old tid) =
  opt_linear_tile_block_visible sh;
  let tile = opt_linear_tile sh in
  let send_value (v:lseq F32.t 128) : is_send_across block_of (tile |-> Frac (1.0R /. 32.0R) v) =
    SH.is_send_across_block_array tile #(1.0R /. 32.0R) #v in
  let send_tile = is_send_across_exists
    (fun (v:lseq F32.t 128) -> tile |-> Frac (1.0R /. 32.0R) v) #send_value in
  let send_read:is_send_across block_of
    (weights |-> Frac (1.0R /. 32.0R) wv ** hidden |-> Frac (1.0R /. 32.0R) hv) = solve in
  let send_output:is_send_across block_of (when_ (tid == 0) (output |-> old)) = solve in
  is_send_across_star _ _ #send_read #(is_send_across_star _ _ #send_tile #send_output)

inline_for_extraction noextract
let pg_projection_post_send
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (hidden:larray BF16.t 7168{KA.is_global_array hidden})
  (output:larray BF16.t 1228800{KA.is_global_array output})
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  (sh:SH.c_shmems opt_linear_shared_desc) (_:squash (SH.c_shmems_inv sh)) (tid:nat)
  : is_send_across block_of (pg_projection_kpost weights hidden output (opt_linear_tile sh) wv hv old tid) =
  opt_linear_tile_block_visible sh;
  let tile = opt_linear_tile sh in
  let send_tile = SH.is_send_across_block_array tile #(1.0R /. 32.0R)
    #(hide (pg_projection_tile_values wv hv 153584)) in
  let send_read:is_send_across block_of
    (weights |-> Frac (1.0R /. 32.0R) wv ** hidden |-> Frac (1.0R /. 32.0R) hv) = solve in
  let send_output:is_send_across block_of
    (when_ (tid == 0) (output |-> pg_projection_output wv hv old)) = solve in
  is_send_across_star _ _ #send_read #(is_send_across_star _ _ #send_tile #send_output)

inline_for_extraction noextract
val pg_projection_descriptor
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (hidden:larray BF16.t 7168{KA.is_global_array hidden})
  (output:larray BF16.t 1228800{KA.is_global_array output})
  (#wv:erased (lseq BF16.t opt_weight_count)) (#hv:erased (lseq BF16.t 7168))
  (#old:erased (lseq BF16.t 1228800))
  : k:KS.kernel_desc (pg_projection_pre weights hidden output wv hv old)
      (pg_projection_post weights hidden output wv hv old){
    SZ.v k.nblk == 1 /\ k.nthr == 32sz /\ k.shmems_desc == opt_linear_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 32sz /\ k.barrier_count bid == 0)}

inline_for_extraction noextract
let pg_padded_hidden (hidden:lseq BF16.t 7168) : GTot (lseq BF16.t 8192) =
  S.init_ghost 8192 (fun i -> if i < 7168 then S.index hidden i else F.zero)

inline_for_extraction noextract
let pg_padded_hidden_b (hidden:lseq BF16.t 7168) (step:Shared.index 64)
  : Lemma (pg_projection_b hidden step ==
    ematrix_subtile (Tensor.from_seq (cm 1024 8) (pg_padded_hidden hidden)) 16 8 step 0) =
  let at (row:Shared.index 16) (col:Shared.index 8)
    : Lemma (Chest.acc2 (pg_projection_b hidden step) row col ==
      Chest.acc2 (ematrix_subtile (Tensor.from_seq (cm 1024 8) (pg_padded_hidden hidden)) 16 8 step 0) row col) =
    dec_dense_right_operand_at 1024 (pg_padded_hidden hidden) step row col
  in
  C.forall_intro_2 at;
  Chest.ext (pg_projection_b hidden step)
    (ematrix_subtile (Tensor.from_seq (cm 1024 8) (pg_padded_hidden hidden)) 16 8 step 0)

inline_for_extraction noextract
let rec pg_projection_dense_live (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (block:nat{block < 151936 /\ block % 16 == 0}) (steps:nat{steps <= 64})
  : Lemma (pg_projection_acc weights hidden block steps ==
    dec_dense_mma_fold 1024 (S.slice weights (block * 1024) ((block + 16) * 1024))
      (pg_padded_hidden hidden) steps)
    (decreases steps) =
  pg_projection_live_block block;
  if steps > 0 then (
    pg_projection_dense_live weights hidden block (steps - 1);
    pg_projection_live_a weights block (steps - 1);
    pg_padded_hidden_b hidden (steps - 1);
    dec_dense_mma_step 1024 (S.slice weights (block * 1024) ((block + 16) * 1024))
      (pg_padded_hidden hidden) (steps - 1))

inline_for_extraction noextract
let pg_embedding_weights (weights:lseq BF16.t opt_weight_count)
  : GTot (Shared.matrix BF16.t 151936 1024) =
  fun row col -> S.index weights (row * 1024 + col)

inline_for_extraction noextract
let pg_normalized_view (hidden:lseq BF16.t 7168) : GTot (Shared.hidden BF16.t) =
  fun row col -> S.index hidden (row * 1024 + col)

inline_for_extraction noextract
let pg_projection_output_cell (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (old:lseq BF16.t 1228800) (row:Shared.index 7) (column:Shared.token_id)
  : Lemma (S.index (pg_projection_output weights hidden old) (row * 153600 + column) ==
    Casts.fcast #F32.t #BF16.t
      (Chest.acc2 (pg_projection_acc weights hidden (column / 16 * 16) 64) (column % 16) row)) = ()

inline_for_extraction noextract
let pg_projection_operands_related (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (wr:Shared.matrix real 151936 1024) (xr:Shared.hidden real)
  (row:Shared.index 7) (column:Shared.token_id)
  : Lemma
      (requires Shared.matrix_approximates (pg_embedding_weights weights) wr /\
        Shared.matrix_approximates (pg_normalized_view hidden) xr)
      (ensures
        (forall (i:Shared.index 1024).
          S.index (S.slice weights ((column / 16 * 16) * 1024) ((column / 16 * 16 + 16) * 1024))
            (column % 16 * 1024 + i) %~ wr column i) /\
        (forall (i:Shared.index 1024). S.index (pg_padded_hidden hidden) (row * 1024 + i) %~ xr row i)) =
  pg_projection_live_block (column / 16 * 16);
  let at (i:Shared.index 1024)
    : Lemma (
      S.index (S.slice weights ((column / 16 * 16) * 1024) ((column / 16 * 16 + 16) * 1024))
        (column % 16 * 1024 + i) %~ wr column i /\
      S.index (pg_padded_hidden hidden) (row * 1024 + i) %~ xr row i) =
    FStar.Math.Lemmas.lemma_div_mod column 16
  in C.forall_intro at
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_projection_cell_related (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (old:lseq BF16.t 1228800) (wr:Shared.matrix real 151936 1024) (xr:Shared.hidden real)
  (row:Shared.index 7) (column:Shared.token_id)
  : Lemma
      (requires Shared.matrix_approximates (pg_embedding_weights weights) wr /\
        Shared.matrix_approximates (pg_normalized_view hidden) xr)
      (ensures S.index (pg_projection_output weights hidden old) (row * 153600 + column) %~
        Shared.linear_rows #7 #1024 #151936 wr xr row column) =
  let block = column / 16 * 16 in
  pg_projection_live_block block;
  pg_projection_dense_live weights hidden block 64;
  pg_projection_operands_related weights hidden wr xr row column;
  dec_dense_mma_dot 1024 (S.slice weights (block * 1024) ((block + 16) * 1024))
    (pg_padded_hidden hidden) (column % 16) row (wr column) (xr row);
  let score = Chest.acc2 (pg_projection_acc weights hidden block 64) (column % 16) row in
  dec_shared_dot_commute #1024 (wr column) (xr row);
  Casts.fcast_approx #F32.t #BF16.t score (dec_spec_dot #1024 (xr row) (wr column));
  pg_projection_output_cell weights hidden old row column;
  u_linear_math_at #1024 #151936 wr xr row column
#pop-options

inline_for_extraction noextract
let pg_selector_scores (logits:lseq BF16.t 1228800) (i:Shared.token_id) : F32.t =
  Cast.cast_bf16_to_f32 (S.index logits (921600 + i))
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_projection_related (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (old:lseq BF16.t 1228800)
  : Lemma (Shared.projection_relation (pg_embedding_weights weights) (pg_normalized_view hidden)
    (pg_selector_scores (pg_projection_output weights hidden old))) =
  let at (wr:Shared.matrix real 151936 1024) (xr:Shared.hidden real)
    : Lemma
        (requires Shared.matrix_approximates (pg_embedding_weights weights) wr /\
          Shared.matrix_approximates (pg_normalized_view hidden) xr)
        (ensures Shared.vector_approximates (pg_selector_scores (pg_projection_output weights hidden old))
          (Shared.linear #1024 #151936 wr (xr 6))) =
    let cell (col:Shared.token_id)
      : Lemma (pg_selector_scores (pg_projection_output weights hidden old) col %~
        Shared.linear #1024 #151936 wr (xr 6) col) =
      pg_projection_cell_related weights hidden old wr xr 6 col;
      Casts.fcast_approx #BF16.t #F32.t
        (S.index (pg_projection_output weights hidden old) (921600 + col))
        (Shared.linear #1024 #151936 wr (xr 6) col)
    in C.forall_intro cell
  in C.forall_intro_2 (C.move_requires_2 at)
#pop-options

inline_for_extraction noextract
let pg_hidden_prefix (input:larray BF16.t 8192) : GTot (larray BF16.t 7168) =
  PA.gsub input 0 7168

inline_for_extraction noextract
ghost
fn pg_take_hidden_prefix (input:larray BF16.t 8192)
  (#x:erased (lseq BF16.t 8192)) (#fraction:perm)
  requires input |-> Frac fraction x
  ensures pg_hidden_prefix input |-> Frac fraction (S.slice x 0 7168) **
    PA.pts_to_range input 7168 8192 #fraction (S.slice x 7168 8192)

inline_for_extraction noextract
ghost
fn pg_return_hidden_prefix (input:larray BF16.t 8192)
  (#x:erased (lseq BF16.t 8192)) (#fraction:perm)
  requires pg_hidden_prefix input |-> Frac fraction (S.slice x 0 7168) **
    PA.pts_to_range input 7168 8192 #fraction (S.slice x 7168 8192)
  ensures input |-> Frac fraction x

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

inline_for_extraction noextract
let pg_final_gamma (weights:lseq BF16.t opt_weight_count) : GTot (Shared.vector BF16.t 1024) =
  fun col -> S.index weights (155582464 + col)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_final_rms_apply (weights:lseq BF16.t opt_weight_count)
  (x ys:lseq BF16.t 7168) (xr:dec_spec_matrix 7 1024) (gamma:dec_spec_vector 1024)
  : Lemma
      (requires u_rms_prefix #7 #opt_weight_count x weights ys 155582464 7 /\
        dec_matrix_rep #7 #1024 x xr /\ dec_bf16_segment_rep weights 155582464 1024 gamma)
      (ensures dec_matrix_rep #7 #1024 ys (fun row -> dec_spec_rms_norm gamma (xr row))) =
  assert (dec_matrix_prefix #7 #1024 ys (fun row -> dec_spec_rms_norm gamma (xr row)) 7)
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_final_rms_interpretation (weights:lseq BF16.t opt_weight_count)
  (x:lseq BF16.t 8192) (ys:lseq BF16.t 7168)
  (gamma:Shared.vector real 1024) (xr:Shared.hidden real)
  : Lemma
      (requires u_rms_prefix #7 #opt_weight_count (S.slice x 0 7168) weights ys 155582464 7 /\
        Shared.vector_approximates (pg_final_gamma weights) gamma /\
        Shared.matrix_approximates (u_live_matrix #1024 x) xr)
      (ensures Shared.matrix_approximates (pg_normalized_view ys) (Shared.normalize_hidden gamma xr)) =
  assert (dec_matrix_rep #7 #1024 (S.slice x 0 7168) xr);
  assert (dec_bf16_segment_rep weights 155582464 1024 gamma);
  pg_final_rms_apply weights (S.slice x 0 7168) ys xr gamma;
  let at (row:Shared.index 7) (col:Shared.index 1024)
    : Lemma (pg_normalized_view ys row col %~ Shared.normalize_hidden gamma xr row col) =
    u_rms_math_at #1024 gamma (xr row) col
  in C.forall_intro_2 at
#pop-options

inline_for_extraction noextract
let pg_final_rms_related (weights:lseq BF16.t opt_weight_count)
  (x:lseq BF16.t 8192) (ys:lseq BF16.t 7168)
  : Lemma
      (requires u_rms_prefix #7 #opt_weight_count (S.slice x 0 7168) weights ys 155582464 7)
      (ensures Shared.final_norm_relation (pg_final_gamma weights)
        (u_live_matrix #1024 x) (pg_normalized_view ys)) =
  C.forall_intro_2 (C.move_requires_2 (pg_final_rms_interpretation weights x ys))

inline_for_extraction noextract
let pg_selector_initial : F32.t = F.sub F.zero F.infinity

inline_for_extraction noextract
let pg_selector_irreflexive (x:F32.t) : Lemma (not (F.lt x x)) =
  if F.lt x x then (
    F32.lt_ordered x x; F.eq_spec x x; F.lte_is_lt_or_eq x x; F.negate_lt_is_lte x x)

inline_for_extraction noextract
let pg_selector_order (x y:F32.t)
  : Lemma
      (requires not (F.NaN? (F.kind x)) /\ not (F.NaN? (F.kind y)))
      (ensures (F.lt x y \/ F.eq x y \/ F.lt y x) /\
        (F.lte x y <==> not (F.lt y x))) =
  F.eq_spec x y; F.eq_spec y x;
  F.lte_is_lt_or_eq x y; F.lte_is_lt_or_eq y x;
  F.negate_lt_is_lte x y; F.negate_lt_is_lte y x

inline_for_extraction noextract
let pg_selector_lt_congruent (z x y:F32.t)
  : Lemma
      (requires not (F.NaN? (F.kind z)) /\ not (F.NaN? (F.kind x)) /\
        not (F.NaN? (F.kind y)) /\ F.eq x y)
      (ensures F.lt z x <==> F.lt z y) =
  pg_selector_order x y;
  pg_selector_order z x;
  pg_selector_order z y;
  if F.lt z x && not (F.lt z y) then (
    if F.lt y z then F32.lt_transitive y z x
    else (F.eq_spec z y; F.eq_spec x y; F.eq_spec z x))
  else if F.lt z y && not (F.lt z x) then (
    if F.lt x z then F32.lt_transitive x z y
    else (F.eq_spec z x; F.eq_spec x y; F.eq_spec z y))

inline_for_extraction noextract
let pg_selector_lte_trans (a b c:F32.t)
  : Lemma
      (requires not (F.NaN? (F.kind a)) /\ not (F.NaN? (F.kind b)) /\ not (F.NaN? (F.kind c)) /\
        F.lte a b /\ F.lte b c)
      (ensures F.lte a c) =
  pg_selector_order a b; pg_selector_order b c; pg_selector_order a c;
  if F.lt c a then (
    if F.lt a b then F32.lt_transitive c a b
    else pg_selector_lt_congruent c a b;
    ())

inline_for_extraction noextract
let pg_selector_best (scores:Shared.vector F32.t 151936) (mask:Shared.token_id -> prop)
  (value:F32.t) (index:int) : prop =
  if index = -1 then value == pg_selector_initial /\
    (forall (i:Shared.token_id). mask i ==> not (F.lt pg_selector_initial (scores i)))
  else if 0 <= index && index < 151936 then
    mask index /\ value == scores index /\ F.lt pg_selector_initial value /\
    (forall (i:Shared.token_id). mask i /\ F.lt pg_selector_initial (scores i) ==> F.lte (scores i) value)
  else False

inline_for_extraction noextract
let pg_selector_merge (scores:Shared.vector F32.t 151936)
  (left right:Shared.token_id -> prop) (a b:F32.t) (ai bi:int)
  : Lemma
      (requires pg_selector_best scores left a ai /\ pg_selector_best scores right b bi)
      (ensures pg_selector_best scores (fun i -> left i \/ right i)
        (if F.lt a b then b else a) (if F.lt a b then bi else ai)) =
  if F.lt a b then (
    if bi = -1 then (
      if ai = -1 then pg_selector_irreflexive pg_selector_initial
      else (F32.lt_transitive pg_selector_initial a pg_selector_initial; pg_selector_irreflexive pg_selector_initial))
    else (
      F32.lt_ordered a b;
      let at (i:Shared.token_id)
        : Lemma ((left i \/ right i) /\ F.lt pg_selector_initial (scores i) ==> F.lte (scores i) b) =
        if left i && F.lt pg_selector_initial (scores i) then (
          F32.lt_ordered pg_selector_initial (scores i);
          F.lte_is_lt_or_eq a b;
          pg_selector_lte_trans (scores i) a b)
      in C.forall_intro at))
  else if ai = -1 then ()
  else (
    F32.lt_ordered pg_selector_initial a;
    let at (i:Shared.token_id)
      : Lemma ((left i \/ right i) /\ F.lt pg_selector_initial (scores i) ==> F.lte (scores i) a) =
      if right i && F.lt pg_selector_initial (scores i) then (
        assert (bi <> -1);
        F32.lt_ordered pg_selector_initial b;
        F32.lt_ordered pg_selector_initial (scores i);
        pg_selector_order b a;
        pg_selector_lte_trans (scores i) b a)
    in C.forall_intro at)

inline_for_extraction noextract
let pg_selector_mask_equal (scores:Shared.vector F32.t 151936) (left right:Shared.token_id -> prop)
  (value:F32.t) (index:int)
  : Lemma
      (requires pg_selector_best scores left value index /\ (forall (i:Shared.token_id). left i <==> right i))
      (ensures pg_selector_best scores right value index) = ()

inline_for_extraction noextract
let pg_selector_negative_below (x:F32.t)
  : Lemma (requires F.Finite? (F.kind x)) (ensures F.lt pg_selector_initial x) =
  let nx = F.sub F.zero x in
  F.neg_kind x; F.neg_kind (F.infinity #F32.t); F.neg_kind nx;
  F.infinity_val_spec nx;
  F.lte_is_lt_or_eq nx (F.infinity #F32.t);
  F.eq_spec nx (F.infinity #F32.t);
  if F.is_zero (F.infinity #F32.t) then F.is_zero_spec (F.infinity #F32.t);
  assert (F.lt nx (F.infinity #F32.t));
  F.lt_neg_flip nx (F.infinity #F32.t);
  F.neg_neg x;
  pg_selector_lt_congruent pg_selector_initial (F.sub F.zero nx) x

inline_for_extraction noextract
let pg_selector_result (logits:lseq BF16.t 1228800) (prompt:lseq I64.t 7) (output:lseq I64.t 8) : prop =
  (forall (i:Shared.index 7). S.index output i == S.index prompt i) /\
  (exists (value:F32.t). pg_selector_best (pg_selector_scores logits) (fun _ -> True) value (I64.v (S.index output 7)))

inline_for_extraction noextract
let pg_selector_maximal (logits:lseq BF16.t 1228800) (prompt:lseq I64.t 7) (output:lseq I64.t 8)
  : Lemma
      (requires pg_selector_result logits prompt output /\ Shared.selector_domain (pg_selector_scores logits))
      (ensures 0 <= I64.v (S.index output 7) /\ I64.v (S.index output 7) < 151936 /\
        (forall (i:Shared.token_id). F.lte (pg_selector_scores logits i)
          (pg_selector_scores logits (I64.v (S.index output 7))))) =
  pg_selector_negative_below (pg_selector_scores logits 0);
  let at (i:Shared.token_id)
    : Lemma (F.lte (pg_selector_scores logits i) (pg_selector_scores logits (I64.v (S.index output 7)))) =
    pg_selector_negative_below (pg_selector_scores logits i)
  in C.forall_intro at

inline_for_extraction noextract
let pg_selector_scan (scores:Shared.vector F32.t 151936) (mask:Shared.token_id -> prop)
  (value:F32.t) (index:int) (next:Shared.token_id)
  : Lemma
      (requires pg_selector_best scores mask value index)
      (ensures pg_selector_best scores (fun i -> mask i \/ i == next)
        (if F.lt value (scores next) then scores next else value)
        (if F.lt value (scores next) then next else index)) =
  if F.lt value (scores next) then (
    F32.lt_ordered value (scores next);
    if index <> -1 then F32.lt_transitive pg_selector_initial value (scores next);
    F.eq_spec (scores next) (scores next);
    F.lte_is_lt_or_eq (scores next) (scores next);
    let at (i:Shared.token_id)
      : Lemma ((mask i \/ i == next) /\ F.lt pg_selector_initial (scores i) ==> F.lte (scores i) (scores next)) =
      if mask i && F.lt pg_selector_initial (scores i) then (
        assert (index <> -1);
        F32.lt_ordered pg_selector_initial (scores i);
        F.lte_is_lt_or_eq value (scores next);
        pg_selector_lte_trans (scores i) value (scores next))
    in C.forall_intro at)
  else if index <> -1 && F.lt pg_selector_initial (scores next) then (
    F32.lt_ordered pg_selector_initial value;
    F32.lt_ordered pg_selector_initial (scores next);
    pg_selector_order (scores next) value)

inline_for_extraction noextract
let pg_sel_cell (#n:pos) (values:lseq (option F32.t) n) (indices:lseq (option I64.t) n)
  (i:Shared.index n) (scores:Shared.vector F32.t 151936) (mask:Shared.token_id -> prop)
  (decode:int -> int) (valid:int -> prop) : prop =
  match S.index values i, S.index indices i with
  | Some v, Some j -> valid (I64.v j) /\ pg_selector_best scores mask v (decode (I64.v j))
  | _ -> False

inline_for_extraction noextract
let pg_sel_cover (group:Shared.token_id -> nat) (origin width:nat) (active:pos) (lane:nat)
  (i:Shared.token_id) : prop =
  origin <= group i /\ group i < origin + width /\ (group i - origin) % active == lane

inline_for_extraction noextract
let pg_sel_cell_mask (#n:pos)
  (vs:lseq (option F32.t) n) (is:lseq (option I64.t) n) (at:Shared.index n)
  (scores:Shared.vector F32.t 151936) (left right:Shared.token_id -> prop)
  (decode:int -> int) (valid:int -> prop)
  : Lemma
      (requires pg_sel_cell #n vs is at scores left decode valid /\
        (forall (i:Shared.token_id). left i <==> right i))
      (ensures pg_sel_cell #n vs is at scores right decode valid) =
  match S.index vs at, S.index is at with
  | Some value, Some index -> pg_selector_mask_equal scores left right value (decode (I64.v index))
  | _ -> ()

inline_for_extraction noextract
let pg_sel_cover_leaf (group:Shared.token_id -> nat) (origin:nat) (width:pos) (lane:nat{lane < width})
  : Lemma (forall (i:Shared.token_id).
    pg_sel_cover group origin width width lane i <==> group i == origin + lane) = ()

inline_for_extraction noextract
let pg_sel_reduce_width (width:nat) : prop = width = 4 \/ width = 8 \/ width = 32

inline_for_extraction noextract
let pg_sel_active (width active:nat) : prop =
  pg_sel_reduce_width width /\ dec_reduce_width active /\ active <= width /\ active > 0 /\ width % active == 0

inline_for_extraction noextract
let pg_sel_forest (#n:pos) (values:lseq (option F32.t) n) (indices:lseq (option I64.t) n)
  (base:nat) (width:pos{base + width <= n}) (active:pos)
  (scores:Shared.vector F32.t 151936) (group:Shared.token_id -> nat) (origin:nat)
  (decode:int -> int) (valid:int -> prop) : prop =
  pg_sel_active width active /\
  (forall (lane:nat{lane < active}).
    pg_sel_cell #n values indices (base + lane) scores (pg_sel_cover group origin width active lane) decode valid)

inline_for_extraction noextract
let pg_sel_partial (#n:pos) (values:lseq (option F32.t) n) (indices:lseq (option I64.t) n)
  (base:nat) (width:pos{base + width <= n}) (active:pos) (done:nat)
  (scores:Shared.vector F32.t 151936) (group:Shared.token_id -> nat) (origin:nat)
  (decode:int -> int) (valid:int -> prop) : prop =
  pg_sel_active width active /\ active > 1 /\ done <= active / 2 /\
  (forall (lane:nat{lane < active}).
    pg_sel_cell #n values indices (base + lane) scores
      (pg_sel_cover group origin width (if lane < done then active / 2 else active) lane) decode valid)

inline_for_extraction noextract
let pg_sel_cover_union (group:Shared.token_id -> nat) (origin width:nat) (active:pos{active > 1})
  (lane:nat{lane < active / 2})
  : Lemma
      (requires pg_sel_active width active /\ active > 1)
      (ensures forall (i:Shared.token_id).
        (pg_sel_cover group origin width active lane i \/
         pg_sel_cover group origin width active (lane + active / 2) i) <==>
          pg_sel_cover group origin width (active / 2) lane i) =
  let at (i:Shared.token_id)
    : Lemma (
      (pg_sel_cover group origin width active lane i \/
       pg_sel_cover group origin width active (lane + active / 2) i) <==>
        pg_sel_cover group origin width (active / 2) lane i) =
    if origin <= group i && group i < origin + width then (
      FStar.Math.Lemmas.modulo_modulo_lemma (group i - origin) (active / 2) 2;
      FStar.Math.Lemmas.lemma_div_mod ((group i - origin) % active) (active / 2))
  in C.forall_intro at

inline_for_extraction noextract
let pg_sel_reduce_step (#n:pos)
  (values:lseq (option F32.t) n) (indices:lseq (option I64.t) n)
  (base:nat) (width:pos{base + width <= n}) (active:pos) (lane:nat{lane < active / 2})
  (scores:Shared.vector F32.t 151936) (group:Shared.token_id -> nat) (origin:nat)
  (decode:int -> int) (valid:int -> prop)
  (left right:F32.t) (li ri:I64.t)
  : Lemma
      (requires pg_sel_partial #n values indices base width active lane scores group origin decode valid /\
        S.index values (base + lane) == Some left /\ S.index values (base + lane + active / 2) == Some right /\
        S.index indices (base + lane) == Some li /\ S.index indices (base + lane + active / 2) == Some ri)
      (ensures pg_sel_partial #n
        (if F.lt left right then S.upd values (base + lane) (Some right) else values)
        (if F.lt left right then S.upd indices (base + lane) (Some ri) else indices)
        base width active (lane + 1) scores group origin decode valid) =
  assert (pg_sel_cell #n values indices (base + lane) scores
    (pg_sel_cover group origin width active lane) decode valid);
  assert (lane + active / 2 < active);
  assert (pg_sel_cell #n values indices (base + (lane + active / 2)) scores
    (pg_sel_cover group origin width (if lane + active / 2 < lane then active / 2 else active)
      (lane + active / 2)) decode valid);
  assert (pg_sel_cell #n values indices (base + lane + active / 2) scores
    (pg_sel_cover group origin width active (lane + active / 2)) decode valid);
  pg_sel_cover_union group origin width active lane;
  pg_selector_merge scores (pg_sel_cover group origin width active lane)
    (pg_sel_cover group origin width active (lane + active / 2)) left right (decode (I64.v li)) (decode (I64.v ri));
  pg_selector_mask_equal scores
    (fun i -> pg_sel_cover group origin width active lane i \/ pg_sel_cover group origin width active (lane + active / 2) i)
    (pg_sel_cover group origin width (active / 2) lane)
    (if F.lt left right then right else left)
    (if F.lt left right then decode (I64.v ri) else decode (I64.v li));
  let nv = if F.lt left right then S.upd values (base + lane) (Some right) else values in
  let ni = if F.lt left right then S.upd indices (base + lane) (Some ri) else indices in
  let at (j:nat{j < active})
    : Lemma (pg_sel_cell #n nv ni (base + j) scores
      (pg_sel_cover group origin width (if j < lane + 1 then active / 2 else active) j) decode valid) =
    if j = lane then ()
    else (
      if F.lt left right then (
        S.lemma_index_upd2 values (base + lane) (Some right) (base + j);
        S.lemma_index_upd2 indices (base + lane) (Some ri) (base + j)))
  in C.forall_intro at

inline_for_extraction noextract
let pg_sel_outside (#a:Type0) (#n:pos) (before after:lseq (option a) n) (base width:nat) : prop =
  forall (i:Shared.index n). (i < base \/ base + width <= i) ==> S.index after i == S.index before i

inline_for_extraction noextract
let pg_sel_partial_cell (#n:pos)
  (values:lseq (option F32.t) n) (indices:lseq (option I64.t) n)
  (base:nat) (width:pos{base + width <= n}) (active:pos) (done:nat)
  (scores:Shared.vector F32.t 151936) (group:Shared.token_id -> nat) (origin:nat)
  (decode:int -> int) (valid:int -> prop) (query:nat{query < active})
  : Lemma
      (requires pg_sel_partial #n values indices base width active done scores group origin decode valid)
      (ensures Some? (S.index values (base + query)) /\ Some? (S.index indices (base + query))) =
  assert (pg_sel_cell #n values indices (base + query) scores
    (pg_sel_cover group origin width (if query < done then active / 2 else active) query) decode valid)

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

inline_for_extraction noextract
let pg_local_decode (chunk:Shared.index 96) (index:int) : int =
  if index = -1 then -1 else chunk * 1600 + index

inline_for_extraction noextract
let pg_local_valid (index:int) : prop = index = -1 \/ (0 <= index /\ index < 1600)

inline_for_extraction noextract
let pg_vocabulary_chunk (chunk:nat) (i:Shared.token_id) : prop = i / 1600 == chunk

inline_for_extraction noextract
let pg_thread_mask (chunk:Shared.index 96) (thread:Shared.index 128) (stop:nat)
  (i:Shared.token_id) : prop =
  i / 1600 == chunk /\ (i % 1600) % 128 == thread /\ i % 1600 < stop

inline_for_extraction noextract
let pg_scan_mask_step (chunk:Shared.index 96) (thread:Shared.index 128)
  (at:nat{thread <= at /\ at < 1600 /\ chunk * 1600 + at < 151936 /\ (at - thread) % 128 == 0})
  : Lemma (forall (i:Shared.token_id).
      (pg_thread_mask chunk thread at i \/ i == chunk * 1600 + at) <==>
        pg_thread_mask chunk thread (at + 128) i) =
  let cell (i:Shared.token_id)
    : Lemma ((pg_thread_mask chunk thread at i \/ i == chunk * 1600 + at) <==>
      pg_thread_mask chunk thread (at + 128) i) =
    FStar.Math.Lemmas.lemma_div_mod i 1600;
    FStar.Math.Lemmas.lemma_div_mod (i % 1600) 128;
    FStar.Math.Lemmas.lemma_div_mod (at - thread) 128
  in C.forall_intro cell

inline_for_extraction noextract
let pg_scan_mask_end (chunk:Shared.index 96) (thread:Shared.index 128)
  (stop:nat{thread <= stop /\ (stop - thread) % 128 == 0 /\ (1600 <= stop \/ 151936 <= chunk * 1600 + stop)})
  : Lemma (forall (i:Shared.token_id). pg_thread_mask chunk thread stop i <==>
      (i / 1600 == chunk /\ (i % 1600) % 128 == thread)) = ()

inline_for_extraction noextract
let pg_thread_group (chunk:Shared.index 96) (i:Shared.token_id) : nat =
  if i / 1600 = chunk then (i % 1600) % 128 else 128

inline_for_extraction noextract
let pg_chunk_group (chunk:Shared.index 96) (i:Shared.token_id) : nat =
  if i / 1600 = chunk then ((i % 1600) % 128) / 32 else 4

inline_for_extraction noextract
let pg_thread_leaf (chunk:Shared.index 96) (thread:nat) (i:Shared.token_id) : prop =
  pg_thread_group chunk i == thread

inline_for_extraction noextract
let pg_chunk_leaf (chunk:Shared.index 96) (warp:nat) (i:Shared.token_id) : prop =
  pg_chunk_group chunk i == warp

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

inline_for_extraction noextract
let pg_chunk_scanned (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (chunk:Shared.index 96) (count:nat) : prop =
  forall (thread:nat{thread < count /\ thread < 128}).
    pg_sel_cell #256 vs is thread scores (pg_thread_leaf chunk thread) (pg_local_decode chunk) pg_local_valid

inline_for_extraction noextract
let pg_chunk_warps (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (chunk:Shared.index 96) (done:nat) : prop =
  (forall (warp:nat{warp < done /\ warp < 4}).
    pg_sel_cell #256 vs is warp scores (pg_chunk_leaf chunk warp) (pg_local_decode chunk) pg_local_valid) /\
  (forall (thread:nat{done * 32 <= thread /\ thread < 128}).
    pg_sel_cell #256 vs is thread scores (pg_thread_leaf chunk thread) (pg_local_decode chunk) pg_local_valid)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_chunk_warp_start (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (chunk:Shared.index 96) (warp:Shared.index 4)
  : Lemma
      (requires pg_chunk_warps vs is scores chunk warp)
      (ensures pg_sel_forest #256 vs is (warp * 32) 32 32 scores (pg_thread_group chunk) (warp * 32)
        (pg_local_decode chunk) pg_local_valid) =
  let at (lane:Shared.index 32)
    : Lemma (pg_sel_cell #256 vs is (warp * 32 + lane) scores
      (pg_sel_cover (pg_thread_group chunk) (warp * 32) 32 32 lane)
      (pg_local_decode chunk) pg_local_valid) =
    assert (pg_sel_cell #256 vs is (warp * 32 + lane) scores
      (pg_thread_leaf chunk (warp * 32 + lane)) (pg_local_decode chunk) pg_local_valid);
    pg_sel_cover_leaf (pg_thread_group chunk) (warp * 32) 32 lane;
    pg_sel_cell_mask vs is (warp * 32 + lane) scores
      (pg_thread_leaf chunk (warp * 32 + lane))
      (pg_sel_cover (pg_thread_group chunk) (warp * 32) 32 32 lane) (pg_local_decode chunk) pg_local_valid
  in C.forall_intro at
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_chunk_compress (before_v after_v:lseq (option F32.t) 256)
  (before_i after_i:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (chunk:Shared.index 96) (warp:Shared.index 4)
  (value:F32.t) (index:I64.t)
  : Lemma
      (requires pg_chunk_warps before_v before_i scores chunk warp /\
        pg_sel_outside before_v after_v (warp * 32) 32 /\ pg_sel_outside before_i after_i (warp * 32) 32 /\
        pg_sel_forest #256 after_v after_i (warp * 32) 32 1 scores (pg_thread_group chunk) (warp * 32)
          (pg_local_decode chunk) pg_local_valid /\
        S.index after_v (warp * 32) == Some value /\ S.index after_i (warp * 32) == Some index)
      (ensures pg_chunk_warps (S.upd after_v warp (Some value)) (S.upd after_i warp (Some index))
        scores chunk (warp + 1)) =
  assert (pg_sel_cell #256 after_v after_i (warp * 32) scores
    (pg_sel_cover (pg_thread_group chunk) (warp * 32) 32 1 0) (pg_local_decode chunk) pg_local_valid);
  let mask (i:Shared.token_id)
    : Lemma (pg_sel_cover (pg_thread_group chunk) (warp * 32) 32 1 0 i <==>
      pg_chunk_group chunk i == warp) =
    FStar.Math.Lemmas.lemma_div_mod ((i % 1600) % 128) 32
  in C.forall_intro mask;
  pg_selector_mask_equal scores (pg_sel_cover (pg_thread_group chunk) (warp * 32) 32 1 0)
    (pg_chunk_leaf chunk warp) value (pg_local_decode chunk (I64.v index));
  let complete (j:nat{j < warp + 1 /\ j < 4})
    : Lemma (pg_sel_cell #256 (S.upd after_v warp (Some value)) (S.upd after_i warp (Some index))
      j scores (pg_chunk_leaf chunk j) (pg_local_decode chunk) pg_local_valid) =
    if j = warp then ()
    else (
      assert (j < warp * 32);
      S.lemma_index_upd2 after_v warp (Some value) j;
      S.lemma_index_upd2 after_i warp (Some index) j)
  in C.forall_intro complete;
  let pending (j:nat{(warp + 1) * 32 <= j /\ j < 128})
    : Lemma (pg_sel_cell #256 (S.upd after_v warp (Some value)) (S.upd after_i warp (Some index))
      j scores (pg_thread_leaf chunk j) (pg_local_decode chunk) pg_local_valid) =
    S.lemma_index_upd2 after_v warp (Some value) j;
    S.lemma_index_upd2 after_i warp (Some index) j
  in C.forall_intro pending
#pop-options

inline_for_extraction noextract
let pg_chunk_merge_start (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (chunk:Shared.index 96)
  : Lemma
      (requires pg_chunk_warps vs is scores chunk 4)
      (ensures pg_sel_forest #256 vs is 0 4 4 scores (pg_chunk_group chunk) 0 (pg_local_decode chunk) pg_local_valid) =
  let at (lane:Shared.index 4)
    : Lemma (pg_sel_cell #256 vs is lane scores
      (pg_sel_cover (pg_chunk_group chunk) 0 4 4 lane) (pg_local_decode chunk) pg_local_valid) =
    assert (pg_sel_cell #256 vs is lane scores (pg_chunk_leaf chunk lane) (pg_local_decode chunk) pg_local_valid);
    pg_sel_cover_leaf (pg_chunk_group chunk) 0 4 lane;
    pg_sel_cell_mask vs is lane scores (pg_chunk_leaf chunk lane)
      (pg_sel_cover (pg_chunk_group chunk) 0 4 4 lane) (pg_local_decode chunk) pg_local_valid
  in C.forall_intro at

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

inline_for_extraction noextract
let pg_int_or_positive (a:nat{a < 9223372036854775808}) (b:nat{b < 9223372036854775808})
  : Lemma
      (requires UI.logor #64 a b < 9223372036854775808)
      (ensures SI.logor #64 a b == UI.logor #64 a b) =
  assert (SI.logor #64 a b ==
    (if SI.max_int 64 < UI.logor #64 a b then UI.logor #64 a b - 18446744073709551616
     else UI.logor #64 a b)) by
    (FStar.Tactics.V2.norm [delta_only [
      `%SI.logor; `%SI.from_vec; `%SI.to_vec; `%SI.to_uint; `%UI.logor]; iota; zeta];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let pg_int_and_positive (a:nat{a < 9223372036854775808}) (b:nat{b < 9223372036854775808})
  : Lemma
      (requires UI.logand #64 a b < 9223372036854775808)
      (ensures SI.logand #64 a b == UI.logand #64 a b) =
  assert (SI.logand #64 a b ==
    (if SI.max_int 64 < UI.logand #64 a b then UI.logand #64 a b - 18446744073709551616
     else UI.logand #64 a b)) by
    (FStar.Tactics.V2.norm [delta_only [
      `%SI.logand; `%SI.from_vec; `%SI.to_vec; `%SI.to_uint; `%UI.logand]; iota; zeta];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let pg_int_shift_positive (a:nat{a < 9223372036854775808})
  : Lemma (SI.shift_right #64 a 32 == a / 4294967296) =
  UI.shift_right_value_lemma #64 a 32;
  assert (SI.shift_right #64 a 32 ==
    (if SI.max_int 64 < UI.shift_right #64 a 32 then UI.shift_right #64 a 32 - 18446744073709551616
     else UI.shift_right #64 a 32)) by
    (FStar.Tactics.V2.norm [delta_only [
      `%SI.shift_right; `%SI.from_vec; `%SI.to_vec; `%SI.to_uint; `%UI.shift_right]; iota; zeta];
     FStar.Tactics.V2.smt ())

inline_for_extraction noextract
let pg_selector_pack_bits (chunk:nat{chunk < 96}) (index:nat{index < 1600})
  : Lemma (SI.logor #64 (SI.shift_left #64 chunk 32) index == chunk * 4294967296 + index) =
  SI.shift_left_value_lemma #64 chunk 32;
  assert (SI.shift_left #64 chunk 32 == chunk * 4294967296);
  UI.logor_disjoint #64 (chunk * 4294967296) index 32;
  pg_int_or_positive (chunk * 4294967296) index

inline_for_extraction noextract
let pg_selector_unpack_bits (packed:nat{packed < 96 * 4294967296})
  : Lemma (SI.shift_right #64 packed 32 == packed / 4294967296 /\
    SI.logand #64 packed 4294967295 == packed % 4294967296) =
  UI.logand_mask #64 packed 32;
  pg_int_and_positive packed 4294967295;
  pg_int_shift_positive packed

inline_for_extraction noextract
let pg_packed_valid (value:int) : prop =
  value = -1 \/ (0 <= value /\ value < 96 * 4294967296 /\ value % 4294967296 < 1600)

inline_for_extraction noextract
let pg_packed_decode (value:int) : int =
  if value = -1 then -1 else value / 4294967296 * 1600 + value % 4294967296

inline_for_extraction noextract
let pg_pack_index (chunk:szlt 96) (index:I64.t{0 <= I64.v index /\ I64.v index < 1600})
  : result:I64.t{I64.v result == SZ.v chunk * 4294967296 + I64.v index} =
  pg_selector_pack_bits (SZ.v chunk) (I64.v index);
  let high = I64.shift_left (FStar.Int.Cast.uint32_to_int64 (SZ.sizet_to_u32 chunk)) 32ul in
  I64.logor high index

inline_for_extraction noextract
let pg_decode_index (packed:I64.t{pg_packed_valid (I64.v packed)})
  : result:I64.t{I64.v result == pg_packed_decode (I64.v packed)} =
  if I64.eq packed (-1L) then -1L
  else (
    pg_selector_unpack_bits (I64.v packed);
    let high = I64.shift_right packed 32ul in
    let low = I64.logand packed 4294967295L in
    I64.add (I64.mul high 1600L) low)

inline_for_extraction noextract
let pg_chunks_ready (vs:lseq (option F32.t) 96) (is:lseq (option I64.t) 96)
  (scores:Shared.vector F32.t 151936) (count:nat) : prop =
  forall (chunk:nat{chunk < count /\ chunk < 96}).
    pg_sel_cell #96 vs is chunk scores (pg_vocabulary_chunk chunk)
      (pg_local_decode chunk) pg_local_valid

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

inline_for_extraction noextract
let pg_chunk_pack_related (scores:Shared.vector F32.t 151936)
  (chunk:Shared.index 96) (value:F32.t) (index:I64.t)
  : Lemma
      (requires pg_local_valid (I64.v index) /\
        pg_selector_best scores (pg_vocabulary_chunk chunk) value (pg_local_decode chunk (I64.v index)))
      (ensures
        (F.lt pg_selector_initial value <==> I64.v index <> -1) /\
        pg_packed_valid (if F.lt pg_selector_initial value then chunk * 4294967296 + I64.v index else -1) /\
        pg_selector_best scores (pg_vocabulary_chunk chunk)
          (if F.lt pg_selector_initial value then value else pg_selector_initial)
          (pg_packed_decode (if F.lt pg_selector_initial value then chunk * 4294967296 + I64.v index else -1))) =
  pg_selector_irreflexive pg_selector_initial;
  if I64.v index <> -1 then (
    assert (0 <= I64.v index /\ I64.v index < 1600);
    assert ((chunk * 4294967296 + I64.v index) / 4294967296 == chunk /\
      (chunk * 4294967296 + I64.v index) % 4294967296 == I64.v index))

inline_for_extraction noextract
type pg_local_index = i:I64.t{pg_local_valid (I64.v i)}

inline_for_extraction noextract
val pg_pack_maybe_store
  (values:larray F32.t 256) (indices:larray I64.t 256)
  (chunk_indices:larray I64.t 96) (thread:szlt 96) (take:bool) (candidate:F32.t)
  (#vs:erased (lseq (option F32.t) 256)) (#is:erased (lseq (option I64.t) 256))
  (#ci:erased (lseq (option I64.t) 96)) (#index:erased pg_local_index)
  : Pulse.Lib.Core.stt unit
      (gpu ** PA.pts_to_mask chunk_indices ci (fun _ -> True) ** (PA.pts_to_mask values vs (fun _ -> True) ** PA.pts_to_mask indices is (fun _ -> True) **
    pure (S.index ci thread == Some (Ghost.reveal #pg_local_index index) /\
      (take ==> 0 <= I64.v index /\ I64.v index < 1600))))
      (fun _ -> gpu ** PA.pts_to_mask chunk_indices ci (fun _ -> True) ** (PA.pts_to_mask values (if take then S.upd vs thread (Some candidate) else vs) (fun _ -> True) **
    PA.pts_to_mask indices
      (if take then S.upd is thread (Some (I64.int_to_t (SZ.v thread * 4294967296 + I64.v index)))
       else is) (fun _ -> True)))


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

inline_for_extraction noextract
let pg_final_leaves (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (count:nat) : prop =
  forall (thread:nat{thread < count /\ thread < 256}).
    pg_sel_cell #256 vs is thread scores (pg_vocabulary_chunk thread) pg_packed_decode pg_packed_valid

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

inline_for_extraction noextract
let pg_final_chunk_group (i:Shared.token_id) : nat = i / 1600

inline_for_extraction noextract
let pg_final_group (i:Shared.token_id) : nat = (i / 1600) / 32

inline_for_extraction noextract
let pg_vocabulary_warp (warp:nat) (i:Shared.token_id) : prop = (i / 1600) / 32 == warp

inline_for_extraction noextract
let pg_final_warps (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (done:nat) : prop =
  (forall (warp:nat{warp < done /\ warp < 8}).
    pg_sel_cell #256 vs is warp scores (pg_vocabulary_warp warp) pg_packed_decode pg_packed_valid) /\
  (forall (thread:nat{done * 32 <= thread /\ thread < 256}).
    pg_sel_cell #256 vs is thread scores (pg_vocabulary_chunk thread) pg_packed_decode pg_packed_valid)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_final_warp_start (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (warp:Shared.index 8)
  : Lemma
      (requires pg_final_warps vs is scores warp)
      (ensures pg_sel_forest #256 vs is (warp * 32) 32 32 scores pg_final_chunk_group (warp * 32)
        pg_packed_decode pg_packed_valid) =
  let at (lane:Shared.index 32)
    : Lemma (pg_sel_cell #256 vs is (warp * 32 + lane) scores
      (pg_sel_cover pg_final_chunk_group (warp * 32) 32 32 lane) pg_packed_decode pg_packed_valid) =
    assert (pg_sel_cell #256 vs is (warp * 32 + lane) scores
      (pg_vocabulary_chunk (warp * 32 + lane)) pg_packed_decode pg_packed_valid);
    pg_sel_cover_leaf pg_final_chunk_group (warp * 32) 32 lane;
    pg_sel_cell_mask vs is (warp * 32 + lane) scores
      (pg_vocabulary_chunk (warp * 32 + lane))
      (pg_sel_cover pg_final_chunk_group (warp * 32) 32 32 lane) pg_packed_decode pg_packed_valid
  in C.forall_intro at
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_final_compress (before_v after_v:lseq (option F32.t) 256)
  (before_i after_i:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936) (warp:Shared.index 8) (value:F32.t) (index:I64.t)
  : Lemma
      (requires pg_final_warps before_v before_i scores warp /\
        pg_sel_outside before_v after_v (warp * 32) 32 /\ pg_sel_outside before_i after_i (warp * 32) 32 /\
        pg_sel_forest #256 after_v after_i (warp * 32) 32 1 scores pg_final_chunk_group (warp * 32)
          pg_packed_decode pg_packed_valid /\
        S.index after_v (warp * 32) == Some value /\ S.index after_i (warp * 32) == Some index)
      (ensures pg_final_warps (S.upd after_v warp (Some value)) (S.upd after_i warp (Some index))
        scores (warp + 1)) =
  assert (pg_sel_cell #256 after_v after_i (warp * 32) scores
    (pg_sel_cover pg_final_chunk_group (warp * 32) 32 1 0) pg_packed_decode pg_packed_valid);
  let mask (i:Shared.token_id)
    : Lemma (pg_sel_cover pg_final_chunk_group (warp * 32) 32 1 0 i <==>
      (i / 1600) / 32 == warp) =
    FStar.Math.Lemmas.lemma_div_mod (i / 1600) 32
  in C.forall_intro mask;
  pg_selector_mask_equal scores (pg_sel_cover pg_final_chunk_group (warp * 32) 32 1 0)
    (pg_vocabulary_warp warp) value (pg_packed_decode (I64.v index));
  let complete (j:nat{j < warp + 1 /\ j < 8})
    : Lemma (pg_sel_cell #256 (S.upd after_v warp (Some value)) (S.upd after_i warp (Some index))
      j scores (pg_vocabulary_warp j) pg_packed_decode pg_packed_valid) =
    if j = warp then ()
    else (
      assert (j < warp * 32);
      S.lemma_index_upd2 after_v warp (Some value) j;
      S.lemma_index_upd2 after_i warp (Some index) j)
  in C.forall_intro complete;
  let pending (j:nat{(warp + 1) * 32 <= j /\ j < 256})
    : Lemma (pg_sel_cell #256 (S.upd after_v warp (Some value)) (S.upd after_i warp (Some index))
      j scores (pg_vocabulary_chunk j) pg_packed_decode pg_packed_valid) =
    S.lemma_index_upd2 after_v warp (Some value) j;
    S.lemma_index_upd2 after_i warp (Some index) j
  in C.forall_intro pending
#pop-options

inline_for_extraction noextract
let pg_final_merge_start (vs:lseq (option F32.t) 256) (is:lseq (option I64.t) 256)
  (scores:Shared.vector F32.t 151936)
  : Lemma
      (requires pg_final_warps vs is scores 8)
      (ensures pg_sel_forest #256 vs is 0 8 8 scores pg_final_group 0 pg_packed_decode pg_packed_valid) =
  let at (lane:Shared.index 8)
    : Lemma (pg_sel_cell #256 vs is lane scores
      (pg_sel_cover pg_final_group 0 8 8 lane) pg_packed_decode pg_packed_valid) =
    assert (pg_sel_cell #256 vs is lane scores (pg_vocabulary_warp lane) pg_packed_decode pg_packed_valid);
    pg_sel_cover_leaf pg_final_group 0 8 lane;
    pg_sel_cell_mask vs is lane scores (pg_vocabulary_warp lane)
      (pg_sel_cover pg_final_group 0 8 8 lane) pg_packed_decode pg_packed_valid
  in C.forall_intro at

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

inline_for_extraction noextract
fn pg_selector_body (logits:larray BF16.t 1228800) (prompt:larray I64.t 7) (output:larray I64.t 8)
  (#xs:erased (lseq BF16.t 1228800)) (#ps:erased (lseq I64.t 7))
  (#old:erased (lseq I64.t 8)) (#fx #fp:perm)
  preserves gpu ** logits |-> Frac fx xs ** prompt |-> Frac fp ps
  requires output |-> old
  ensures exists* (ys:lseq I64.t 8). output |-> ys ** pure (pg_selector_result xs ps ys)

inline_for_extraction noextract
let pg_selector_pre (logits:larray BF16.t 1228800) (prompt:larray I64.t 7) (output:larray I64.t 8)
  (xs:lseq BF16.t 1228800) (ps:lseq I64.t 7) (old:lseq I64.t 8) : slprop =
  logits |-> xs ** prompt |-> ps ** output |-> old

inline_for_extraction noextract
let pg_selector_post (logits:larray BF16.t 1228800) (prompt:larray I64.t 7) (output:larray I64.t 8)
  (xs:lseq BF16.t 1228800) (ps:lseq I64.t 7) : slprop =
  logits |-> xs ** prompt |-> ps **
  (exists* (ys:lseq I64.t 8). output |-> ys ** pure (pg_selector_result xs ps ys))

inline_for_extraction noextract
fn pg_selector_call (logits:larray BF16.t 1228800) (prompt:larray I64.t 7) (output:larray I64.t 8)
  (#xs:erased (lseq BF16.t 1228800)) (#ps:erased (lseq I64.t 7)) (#old:erased (lseq I64.t 8)) ()
  norewrite
  preserves gpu
  requires pg_selector_pre logits prompt output xs ps old
  ensures pg_selector_post logits prompt output xs ps

inline_for_extraction noextract
val pg_selector_descriptor
  (logits:larray BF16.t 1228800{KA.is_global_array logits})
  (prompt:larray I64.t 7{KA.is_global_array prompt}) (output:larray I64.t 8{KA.is_global_array output})
  (#xs:erased (lseq BF16.t 1228800)) (#ps:erased (lseq I64.t 7)) (#old:erased (lseq I64.t 8))
  : k:KS.kernel_desc (pg_selector_pre logits prompt output xs ps old) (pg_selector_post logits prompt output xs ps){
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)}

inline_for_extraction noextract
let pg_suffix_arithmetic (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) (step:nat{step <= 3}) : prop =
  pg_decoder_stage weights prompt v 253 /\
  (step >= 1 ==> u_rms_prefix #7 #opt_weight_count (S.slice (pg_slot v 253) 0 7168)
    weights (pg_slot v 254) 155582464 7) /\
  (step >= 2 ==> pg_slot v 255 == pg_projection_output weights (pg_slot v 254) (pg_slot v 255)) /\
  (step >= 3 ==> pg_selector_result (pg_slot v 255) prompt tokens)

inline_for_extraction noextract
let pg_decoder_suffix_frame (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (target:Shared.index 256{254 <= target}) (value:lseq BF16.t (pg_size target))
  : Lemma
      (requires pg_decoder_stage weights prompt v 253)
      (ensures pg_decoder_stage weights prompt (pg_replace v target value) 253) =
  pg_decoder_stage_definition weights prompt v 253;
  pg_decoder_stage_definition weights prompt (pg_replace v target value) 253;
  assert (pg_first_values (pg_replace v target value) == pg_first_values v);
  let at (layer:Shared.index 28{layer > 0})
    : Lemma (pg_layer_arithmetic weights layer (pg_layer_values (pg_replace v target value) layer) 8) =
    assert (pg_layer_values (pg_replace v target value) layer == pg_layer_values v layer)
  in C.forall_intro at

inline_for_extraction noextract
let pg_suffix_rms_advance (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) (ys:lseq BF16.t 7168)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v tokens 0 /\
        u_rms_prefix #7 #opt_weight_count (S.slice (pg_slot v 253) 0 7168) weights ys 155582464 7)
      (ensures pg_suffix_arithmetic weights prompt (pg_replace v 254 ys) tokens 1) =
  pg_decoder_suffix_frame weights prompt v 254 ys

inline_for_extraction noextract
let pg_projection_idempotent (weights:lseq BF16.t opt_weight_count) (hidden:lseq BF16.t 7168)
  (old:lseq BF16.t 1228800)
  : Lemma (pg_projection_output weights hidden (pg_projection_output weights hidden old) ==
    pg_projection_output weights hidden old) =
  let ys = pg_projection_output weights hidden old in
  let at (i:Shared.index 1228800)
    : Lemma (S.index (pg_projection_output weights hidden ys) i == S.index ys i) = ()
  in C.forall_intro at;
  S.lemma_eq_intro (pg_projection_output weights hidden ys) ys;
  S.lemma_eq_elim (pg_projection_output weights hidden ys) ys

inline_for_extraction noextract
let pg_suffix_projection_advance (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v tokens 1)
      (ensures pg_suffix_arithmetic weights prompt
        (pg_replace v 255 (pg_projection_output weights (pg_slot v 254) (pg_slot v 255))) tokens 2) =
  pg_decoder_suffix_frame weights prompt v 255 (pg_projection_output weights (pg_slot v 254) (pg_slot v 255));
  pg_projection_idempotent weights (pg_slot v 254) (pg_slot v 255)

inline_for_extraction noextract
let pg_suffix_selector_advance (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (before after:lseq I64.t 8)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v before 2 /\ pg_selector_result (pg_slot v 255) prompt after)
      (ensures pg_suffix_arithmetic weights prompt v after 3) = ()

inline_for_extraction noextract
let pg_suffix_core (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) (step:nat{step <= 3}) : slprop =
  PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
  pg_storage buffers v 1.0R ** PA.pts_to (RT.data buffers.token_buffer) tokens **
  pure (pg_suffix_arithmetic weights prompt v tokens step)

inline_for_extraction noextract
type pg_suffix_observation = pg_values & lseq I64.t 8

inline_for_extraction noextract
let pg_suffix_tensors (observed:pg_suffix_observation) : GTot pg_values =
  FStar.Pervasives.Native.fst observed

inline_for_extraction noextract
let pg_suffix_tokens (observed:pg_suffix_observation) : GTot (lseq I64.t 8) =
  FStar.Pervasives.Native.snd observed

inline_for_extraction noextract
let pg_suffix_pending (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) (step:nat{step <= 3}) : slprop =
  exists* (observed:pg_suffix_observation).
    pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) step

inline_for_extraction noextract
val pg_adopt_suffix (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_after_stage weights_address prompt_address buffers weights prompt 253)
      (fun _ -> on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 0))


inline_for_extraction noextract
fn pg_materialize_hidden_prefix (input:larray BF16.t 8192)
  (#x:erased (lseq BF16.t 8192))
  preserves gpu ** input |-> x
  returns prefix:PA.array BF16.t
  ensures rewrites_to prefix (pg_hidden_prefix input)
  ensures pure (PA.length prefix == 7168)

inline_for_extraction noextract
let pg_final_rms_pre (weights:larray BF16.t opt_weight_count) (input:larray BF16.t 8192)
  (output:larray BF16.t 7168) (wv:lseq BF16.t opt_weight_count) (x:lseq BF16.t 8192)
  (old:lseq BF16.t 7168) : slprop =
  weights |-> wv ** input |-> x ** output |-> old

inline_for_extraction noextract
let pg_final_rms_post (weights:larray BF16.t opt_weight_count) (input:larray BF16.t 8192)
  (output:larray BF16.t 7168) (wv:lseq BF16.t opt_weight_count) (x:lseq BF16.t 8192) : slprop =
  weights |-> wv ** input |-> x **
  (exists* (ys:lseq BF16.t 7168). output |-> ys **
    pure (u_rms_prefix #7 #opt_weight_count (S.slice x 0 7168) wv ys 155582464 7))

inline_for_extraction noextract
fn pg_final_rms_call (weights:larray BF16.t opt_weight_count) (input:larray BF16.t 8192)
  (output:larray BF16.t 7168)
  (#wv:erased (lseq BF16.t opt_weight_count)) (#x:erased (lseq BF16.t 8192))
  (#old:erased (lseq BF16.t 7168)) ()
  norewrite
  preserves gpu
  requires pg_final_rms_pre weights input output wv x old
  ensures pg_final_rms_post weights input output wv x

inline_for_extraction noextract
val pg_final_rms_descriptor
  (weights:larray BF16.t opt_weight_count{KA.is_global_array weights})
  (input:larray BF16.t 8192{KA.is_global_array input})
  (output:larray BF16.t 7168{KA.is_global_array output})
  (#wv:erased (lseq BF16.t opt_weight_count)) (#x:erased (lseq BF16.t 8192))
  (#old:erased (lseq BF16.t 7168))
  : k:KS.kernel_desc (pg_final_rms_pre weights input output wv x old) (pg_final_rms_post weights input output wv x){
    SZ.v k.nblk == 1 /\ k.nthr == 1sz /\ k.shmems_desc == opt_scalar_shared_desc /\
    (forall (bid:natlt k.nblk) (sh:SH.c_shmems k.shmems_desc).
      k.barrier_contract bid sh == Barrier.empty_contract 1sz /\ k.barrier_count bid == 0)}

inline_for_extraction noextract
let pg_suffix_pair_frame (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) (step:nat{step <= 2})
  (source target:Shared.index 256) : slprop =
  PA.pts_to prompt_address prompt ** PA.pts_to (RT.data buffers.token_buffer) tokens **
  pg_masked_storage buffers v 1.0R (fun i -> i <> source /\ i <> target) **
  pure (pg_suffix_arithmetic weights prompt v tokens step)

inline_for_extraction noextract
val pg_suffix_pair_open (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) (step:nat{step <= 2})
  (source:Shared.index 256) (target:Shared.index 256{source <> target})
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_suffix_core weights_address prompt_address buffers weights prompt v tokens step)
      (fun _ -> PA.pts_to weights_address weights **
    PA.pts_to (pg_address buffers source) (pg_slot v source) **
    PA.pts_to (pg_address buffers target) (pg_slot v target) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens step source target)


inline_for_extraction noextract
val pg_suffix_pair_close (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8)) (step:nat{step < 2})
  (source:Shared.index 256) (target:Shared.index 256{source <> target})
  (#ys:erased (lseq BF16.t (pg_size target)))
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (PA.pts_to weights_address weights **
    PA.pts_to (pg_address buffers source) (pg_slot v source) ** PA.pts_to (pg_address buffers target) ys **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens step source target **
    pure (pg_suffix_arithmetic weights prompt (pg_replace v target ys) tokens (step + 1)))
      (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1))


inline_for_extraction noextract
val pg_final_rms_before (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 8192{input == pg_address buffers 253})
  (output:larray BF16.t 7168{output == pg_address buffers 254})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8))
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 0)
      (fun _ -> pg_final_rms_pre weights_address input output weights (pg_slot v 253) (pg_slot v 254) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 0 253 254)


inline_for_extraction noextract
val pg_final_rms_after (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 8192{input == pg_address buffers 253})
  (output:larray BF16.t 7168{output == pg_address buffers 254})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8))
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_final_rms_post weights_address input output weights (pg_slot v 253) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 0 253 254)
      (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt 1)


inline_for_extraction noextract
let pg_final_rms_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 8192{input == pg_address buffers 253})
  (output:larray BF16.t 7168{output == pg_address buffers 254})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (observed:erased pg_suffix_observation)
  : pg_source_descriptor
      (pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 0)
      (pg_suffix_pending weights_address prompt_address buffers weights prompt 1) 1sz opt_scalar_shared_desc =
  let code = pg_final_rms_descriptor weights_address input output #weights
    #(hide (pg_slot (pg_suffix_tensors observed) 253)) #(hide (pg_slot (pg_suffix_tensors observed) 254)) in
  let pre = pg_final_rms_pre weights_address input output weights
    (pg_slot (pg_suffix_tensors observed) 253) (pg_slot (pg_suffix_tensors observed) 254) in
  let post = pg_final_rms_post weights_address input output weights (pg_slot (pg_suffix_tensors observed) 253) in
  let frame = pg_suffix_pair_frame prompt_address buffers weights prompt
    (pg_suffix_tensors observed) (pg_suffix_tokens observed) 0 253 254 in
  let framed = opt_descriptor_with_frame #pre #post code frame in
  opt_descriptor_refine #(pre ** frame) #(post ** frame)
    #(pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 0)
    #(pg_suffix_pending weights_address prompt_address buffers weights prompt 1) framed
    (pg_final_rms_before weights_address prompt_address buffers input output #weights #prompt
      #(hide (pg_suffix_tensors observed)) #(hide (pg_suffix_tokens observed)))
    (pg_final_rms_after weights_address prompt_address buffers input output #weights #prompt
      #(hide (pg_suffix_tensors observed)) #(hide (pg_suffix_tokens observed)))

inline_for_extraction noextract
val pg_final_projection_before (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 7168{input == pg_address buffers 254})
  (output:larray BF16.t 1228800{output == pg_address buffers 255})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8))
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 1)
      (fun _ -> pg_projection_pre weights_address input output weights (pg_slot v 254) (pg_slot v 255) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 1 254 255)


inline_for_extraction noextract
val pg_final_projection_after (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 7168{input == pg_address buffers 254})
  (output:larray BF16.t 1228800{output == pg_address buffers 255})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8))
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_projection_post weights_address input output weights (pg_slot v 254) (pg_slot v 255) **
    pg_suffix_pair_frame prompt_address buffers weights prompt v tokens 1 254 255)
      (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt 2)


inline_for_extraction noextract
let pg_final_projection_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 7168{input == pg_address buffers 254})
  (output:larray BF16.t 1228800{output == pg_address buffers 255})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (observed:erased pg_suffix_observation)
  : pg_source_descriptor
      (pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 1)
      (pg_suffix_pending weights_address prompt_address buffers weights prompt 2) 32sz opt_linear_shared_desc =
  let code = pg_projection_descriptor weights_address input output #weights
    #(hide (pg_slot (pg_suffix_tensors observed) 254)) #(hide (pg_slot (pg_suffix_tensors observed) 255)) in
  let pre = pg_projection_pre weights_address input output weights
    (pg_slot (pg_suffix_tensors observed) 254) (pg_slot (pg_suffix_tensors observed) 255) in
  let post = pg_projection_post weights_address input output weights
    (pg_slot (pg_suffix_tensors observed) 254) (pg_slot (pg_suffix_tensors observed) 255) in
  let frame = pg_suffix_pair_frame prompt_address buffers weights prompt
    (pg_suffix_tensors observed) (pg_suffix_tokens observed) 1 254 255 in
  let framed = opt_descriptor_with_frame #pre #post code frame in
  opt_descriptor_refine #(pre ** frame) #(post ** frame)
    #(pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 1)
    #(pg_suffix_pending weights_address prompt_address buffers weights prompt 2) framed
    (pg_final_projection_before weights_address prompt_address buffers input output #weights #prompt
      #(hide (pg_suffix_tensors observed)) #(hide (pg_suffix_tokens observed)))
    (pg_final_projection_after weights_address prompt_address buffers input output #weights #prompt
      #(hide (pg_suffix_tensors observed)) #(hide (pg_suffix_tokens observed)))

inline_for_extraction noextract
let pg_selector_stage_frame (weights_address:larray BF16.t opt_weight_count)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) : slprop =
  PA.pts_to weights_address weights ** pg_masked_storage buffers v 1.0R (fun i -> i <> 255) **
  pure (pg_suffix_arithmetic weights prompt v tokens 2)

inline_for_extraction noextract
val pg_final_selector_before (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (logits:larray BF16.t 1228800{logits == pg_address buffers 255})
  (output:larray I64.t 8{output == RT.data buffers.token_buffer})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8))
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 2)
      (fun _ -> pg_selector_pre logits prompt_address output (pg_slot v 255) prompt tokens **
    pg_selector_stage_frame weights_address buffers weights prompt v tokens)


inline_for_extraction noextract
val pg_final_selector_after (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (logits:larray BF16.t 1228800{logits == pg_address buffers 255})
  (output:larray I64.t 8{output == RT.data buffers.token_buffer})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8))
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_selector_post logits prompt_address output (pg_slot v 255) prompt **
    pg_selector_stage_frame weights_address buffers weights prompt v tokens)
      (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt 3)


inline_for_extraction noextract
let pg_final_selector_stage_descriptor
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (logits:larray BF16.t 1228800{logits == pg_address buffers 255})
  (output:larray I64.t 8{output == RT.data buffers.token_buffer})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (observed:erased pg_suffix_observation)
  : pg_source_descriptor
      (pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 2)
      (pg_suffix_pending weights_address prompt_address buffers weights prompt 3) 1sz opt_scalar_shared_desc =
  let code = pg_selector_descriptor logits prompt_address output
    #(hide (pg_slot (pg_suffix_tensors observed) 255))
    #(hide #(lseq I64.t 7) (Ghost.reveal #opt_prompt_values prompt)) #(hide (pg_suffix_tokens observed)) in
  let pre = pg_selector_pre logits prompt_address output (pg_slot (pg_suffix_tensors observed) 255)
    prompt (pg_suffix_tokens observed) in
  let post = pg_selector_post logits prompt_address output (pg_slot (pg_suffix_tensors observed) 255) prompt in
  let frame = pg_selector_stage_frame weights_address buffers weights prompt
    (pg_suffix_tensors observed) (pg_suffix_tokens observed) in
  let framed = opt_descriptor_with_frame #pre #post code frame in
  opt_descriptor_refine #(pre ** frame) #(post ** frame)
    #(pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 2)
    #(pg_suffix_pending weights_address prompt_address buffers weights prompt 3) framed
    (pg_final_selector_before weights_address prompt_address buffers logits output #weights #prompt
      #(hide (pg_suffix_tensors observed)) #(hide (pg_suffix_tokens observed)))
    (pg_final_selector_after weights_address prompt_address buffers logits output #weights #prompt
      #(hide (pg_suffix_tensors observed)) #(hide (pg_suffix_tokens observed)))

inline_for_extraction noextract
val pg_expand_suffix_pending (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (step:nat{step <= 3})
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt step))
      (fun _ -> on gpu_loc (exists* (observed:pg_suffix_observation).
    pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) step))


inline_for_extraction noextract
val pg_launch_suffix_step (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (step:nat{step < 3})
  (family:opt_pending_family pg_suffix_observation
    (fun observed -> pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) step)
    (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1)))
  (stream:ST.stream_t) (#start:E.epoch_t)
  : Pulse.Lib.Core.stt unit
      (cpu ** ST.stream_live stream ** (E.epoch_live stream (start + (253 + step)) **
    P.pledge0 (E.epoch_done stream (start + (253 + step)))
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt step))))
      (fun _ -> cpu ** ST.stream_live stream ** (E.epoch_live stream (start + (253 + (step + 1))) **
    P.pledge0 (E.epoch_done stream (start + (253 + (step + 1))))
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt (step + 1)))))


inline_for_extraction noextract
let pg_final_rms_family
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 8192{input == pg_address buffers 253}) (output:larray BF16.t 7168{output == pg_address buffers 254})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family pg_suffix_observation
      (fun observed -> pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 0)
      (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt 1) = {
    pending_threads = 1sz; pending_shared = opt_scalar_shared_desc;
    pending_code = pg_final_rms_stage_descriptor weights_address prompt_address buffers input output #weights #prompt
  }

inline_for_extraction noextract
let pg_final_projection_family
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (input:larray BF16.t 7168{input == pg_address buffers 254})
  (output:larray BF16.t 1228800{output == pg_address buffers 255})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family pg_suffix_observation
      (fun observed -> pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 1)
      (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt 2) = {
    pending_threads = 32sz; pending_shared = opt_linear_shared_desc;
    pending_code = pg_final_projection_stage_descriptor weights_address prompt_address buffers input output #weights #prompt
  }

inline_for_extraction noextract
let pg_final_selector_family
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (logits:larray BF16.t 1228800{logits == pg_address buffers 255})
  (output:larray I64.t 8{output == RT.data buffers.token_buffer})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : opt_pending_family pg_suffix_observation
      (fun observed -> pg_suffix_core weights_address prompt_address buffers weights prompt (pg_suffix_tensors observed) (pg_suffix_tokens observed) 2)
      (fun _ -> pg_suffix_pending weights_address prompt_address buffers weights prompt 3) = {
    pending_threads = 1sz; pending_shared = opt_scalar_shared_desc;
    pending_code = pg_final_selector_stage_descriptor weights_address prompt_address buffers logits output #weights #prompt
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

inline_for_extraction noextract
noeq type pg_returned_buffers = {
  returned_tokens:RT.tensor I64.t;
  returned_hidden:RT.tensor BF16.t
}

inline_for_extraction noextract
let pg_returned_metadata (output:pg_returned_buffers) : prop =
  RT.metadata output.returned_tokens [1;8] /\ RT.metadata output.returned_hidden [7;1024]

inline_for_extraction noextract
type pg_returned_result = output:pg_returned_buffers{pg_returned_metadata output}

inline_for_extraction noextract
let pg_returned (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  : Tot (output:pg_returned_buffers{pg_returned_metadata output}) =
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  opt_matching_shape_at opt_bf16_allocation_shapes buffers.bf16_buffers 254;
  opt_layer_allocation_count 28;
  pg_list_index_append [[8;1024];[2;4096;128]]
    (List.append (opt_layer_allocations 28) [[7;1024];[8;153600]]) 254;
  pg_list_index_append (opt_layer_allocations 28) [[7;1024];[8;153600]] 252;
  {returned_tokens = buffers.token_buffer; returned_hidden = List.index buffers.bf16_buffers 254}

inline_for_extraction noextract
let pg_return_addresses (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  : Lemma (
    pg_address buffers 254 == RT.data (pg_returned buffers).returned_hidden /\
    RT.data buffers.token_buffer == RT.data (pg_returned buffers).returned_tokens /\
    pg_returned_metadata (pg_returned buffers)) = ()

inline_for_extraction noextract
let rec pg_list_split (#a:Type0) (items:list a) (count:nat{count <= List.length items})
  : Lemma (
    List.length (List.splitAt count items)._1 == count /\
    List.length (List.splitAt count items)._2 == List.length items - count /\
    items == List.append (List.splitAt count items)._1 (List.splitAt count items)._2)
    (decreases count) =
  if count > 0 then pg_list_split (List.tail items) (count - 1)

inline_for_extraction noextract
let rec pg_list_prefix_step (#a:Type0) (items:list a) (count:pos{count <= List.length items})
  : Lemma ((List.splitAt count items)._1 ==
    List.append (List.splitAt (count - 1) items)._1 [List.index items (count - 1)])
    (decreases count) =
  if count > 1 then pg_list_prefix_step (List.tail items) (count - 1)

inline_for_extraction noextract
let pg_retirement (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  : Tot (list (RT.tensor BF16.t)) =
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  List.index buffers.bf16_buffers 255 :: List.rev' (List.splitAt 254 buffers.bf16_buffers)._1

inline_for_extraction noextract
let pg_list_two (#a:Type0) (items:list a{List.length items == 2})
  : Lemma (items == [List.index items 0;List.index items 1]) =
  match items with
  | [_;_] -> ()
  | _ -> ()

inline_for_extraction noextract
let pg_retirement_binding (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  : Lemma (
    List.length buffers.bf16_buffers == 256 /\
    List.length (List.splitAt 254 buffers.bf16_buffers)._1 == 254 /\
    buffers.bf16_buffers == List.append (List.splitAt 254 buffers.bf16_buffers)._1
      [(pg_returned buffers).returned_hidden;List.index buffers.bf16_buffers 255] /\
    List.length (pg_retirement buffers) == 255) =
  opt_allocation_count ();
  opt_matching_count opt_bf16_allocation_shapes buffers.bf16_buffers;
  pg_list_split buffers.bf16_buffers 254;
  let left,right = List.splitAt 254 buffers.bf16_buffers in
  pg_list_index_append left right 254;
  pg_list_index_append left right 255;
  pg_list_two right;
  assert (List.index buffers.bf16_buffers 254 == List.index right 0);
  assert (List.index buffers.bf16_buffers 255 == List.index right 1);
  assert (right == [List.index buffers.bf16_buffers 254;List.index buffers.bf16_buffers 255]);
  ListProps.rev_length left;
  ListProps.rev_rev' left

inline_for_extraction noextract
ghost
fn pg_owners_append (stream:ST.stream_t) (left right:list (RT.tensor BF16.t))
  requires RH.all_owners stream left ** RH.all_owners stream right
  ensures RH.all_owners stream (List.append left right)


inline_for_extraction noextract
ghost
fn pg_owners_split (stream:ST.stream_t) (left right:list (RT.tensor BF16.t))
  requires RH.all_owners stream (List.append left right)
  ensures RH.all_owners stream left ** RH.all_owners stream right


inline_for_extraction noextract
ghost
fn pg_owners_reverse (stream:ST.stream_t) (items:list (RT.tensor BF16.t))
  requires RH.all_owners stream items
  ensures RH.all_owners stream (List.rev' items)


inline_for_extraction noextract
ghost
fn pg_contents_append (left right:list (RT.tensor BF16.t))
  requires RH.all_contents left ** RH.all_contents right
  ensures RH.all_contents (List.append left right)


inline_for_extraction noextract
ghost
fn pg_contents_reverse (items:list (RT.tensor BF16.t))
  requires RH.all_contents items
  ensures RH.all_contents (List.rev' items)


inline_for_extraction noextract
ghost
fn pg_retirement_owners (stream:ST.stream_t) (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  requires opt_allocated_owners stream buffers
  ensures RH.all_owners stream (pg_retirement buffers) **
    RT.owner stream (pg_returned buffers).returned_tokens ** RT.owner stream (pg_returned buffers).returned_hidden

inline_for_extraction noextract
ghost
fn pg_prefix_to_runtime (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#v:pg_values) (count:nat{count <= 254})
  preserves loc gpu_loc
  requires forall+ (i:Shared.index 256{i < count}). PA.pts_to (pg_address buffers i) (pg_slot v i)
  ensures RH.all_contents (List.splitAt count buffers.bf16_buffers)._1


inline_for_extraction noextract
let pg_source_return_frame (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (output:pg_returned_buffers{pg_returned_metadata output}) : slprop =
  on gpu_loc (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    (exists* (v:pg_values) (tokens:lseq I64.t 8).
      PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
      PA.pts_to (RT.data output.returned_tokens) tokens ** pure (pg_suffix_arithmetic weights prompt v tokens 3)))

inline_for_extraction noextract
irreducible
let pg_source_return_placeless (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (output:pg_returned_buffers{pg_returned_metadata output})
  : placeless (pg_source_return_frame weights_address prompt_address weights prompt output) =
  placeless_on gpu_loc
    (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
      (exists* (v:pg_values) (tokens:lseq I64.t 8).
        PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
        PA.pts_to (RT.data output.returned_tokens) tokens ** pure (pg_suffix_arithmetic weights prompt v tokens 3)))

inline_for_extraction noextract
val pg_pack_source_return (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (output:pg_returned_buffers{pg_returned_metadata output})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:pg_values) (#tokens:lseq I64.t 8)
  : Pulse.Lib.Core.stt_ghost unit emp_inames
      (loc gpu_loc ** (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
    PA.pts_to (RT.data output.returned_tokens) tokens ** pure (pg_suffix_arithmetic weights prompt v tokens 3)))
      (fun _ -> loc gpu_loc ** (pg_source_return_frame weights_address prompt_address weights prompt output))


inline_for_extraction noextract
val pg_retirement_snapshot_at (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:erased pg_values) (#tokens:erased (lseq I64.t 8))
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (loc gpu_loc ** (pg_suffix_core weights_address prompt_address buffers weights prompt v tokens 3))
      (fun _ -> loc gpu_loc ** (RH.all_contents (pg_retirement buffers) **
    pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers)))


inline_for_extraction noextract
val pg_retirement_contents_at (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : unit -> Pulse.Lib.Core.stt_ghost unit emp_inames
      (loc gpu_loc ** (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 3)))
      (fun _ -> loc gpu_loc ** (RH.all_contents (pg_retirement buffers) **
    pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers)))


inline_for_extraction noextract
let pg_retirement_contents (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (buffers:opt_allocated_buffers{opt_allocation_metadata buffers})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (_:unit)
  : Pulse.Lib.Core.stt_ghost unit emp_inames
      (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 3))
      (fun _ -> RH.all_contents (pg_retirement buffers) **
        pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers)) =
  let contents_placeless = u_all_contents_placeless (pg_retirement buffers) in
  ghost_impersonate #emp_inames gpu_loc
    (on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 3))
    (RH.all_contents (pg_retirement buffers) **
      pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers))
    #(placeless_on gpu_loc (pg_suffix_pending weights_address prompt_address buffers weights prompt 3))
    #(placeless_star (RH.all_contents (pg_retirement buffers))
      (pg_source_return_frame weights_address prompt_address weights prompt (pg_returned buffers))
      #contents_placeless #(pg_source_return_placeless weights_address prompt_address weights prompt (pg_returned buffers)))
    (pg_retirement_contents_at weights_address prompt_address buffers #weights #prompt)

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

inline_for_extraction noextract
let pg_parameters (weights:lseq BF16.t opt_weight_count) (layer:Shared.index 28)
  : GTot (Shared.parameters BF16.t) =
  let w = pg_layer_weights weights layer in {
    input_layernorm = (fun i -> S.index w i);
    post_attention_layernorm = (fun i -> S.index w (1024 + i));
    q_proj = (fun r c -> S.index w (2048 + r * 1024 + c));
    k_proj = (fun r c -> S.index w (2048 + (2048 + r) * 1024 + c));
    v_proj = (fun r c -> S.index w (2048 + (3072 + r) * 1024 + c));
    o_proj = (fun r c -> S.index w (4196352 + r * 2048 + c));
    q_norm = (fun i -> S.index w (6293504 + i));
    k_norm = (fun i -> S.index w (6293632 + i));
    gate_proj = (fun r c -> S.index w (6293760 + r * 1024 + c));
    up_proj = (fun r c -> S.index w (6293760 + (3072 + r) * 1024 + c));
    down_proj = (fun r c -> S.index w (12585216 + r * 3072 + c))
  }

inline_for_extraction noextract
let pg_weights_related (weights:lseq BF16.t opt_weight_count) (layer:Shared.index 28)
  (wr:Shared.parameters real)
  : Lemma
      (requires Shared.parameters_approximates (pg_parameters weights layer) wr)
      (ensures dec_weights_representation (pg_layer_weights weights layer) (u_real_parameters wr) /\
        Shared.vector_approximates (pg_input_gamma weights layer) wr.input_layernorm /\
        dec_bf16_segment_rep weights (pg_layer_base layer + 1024) 1024 wr.post_attention_layernorm) = ()
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_mlp_related (weights:lseq BF16.t opt_weight_count) (layer:Shared.index 28)
  (v:opt_block_values) (w:Shared.parameters real) (residual:Shared.hidden real)
  : Lemma
      (requires pg_layer_arithmetic weights layer v 8 /\
        Shared.parameters_approximates (pg_parameters weights layer) w /\
        Shared.matrix_approximates (u_live_matrix #1024 v.v_residual) residual)
      (ensures Shared.matrix_approximates (u_live_matrix #1024 v.v_hidden)
        (Shared.residual_add residual
          (Shared.feed_forward w (Shared.normalize_hidden w.post_attention_layernorm residual)))) =
  pg_weights_related weights layer w;
  dec_source_weight_views (pg_layer_weights weights layer) (u_real_parameters w);
  u_rms_live_related weights v.v_residual v.v_postnorm (pg_layer_base layer + 1024) w.post_attention_layernorm residual;
  let normalized = Shared.normalize_hidden w.post_attention_layernorm residual in
  u_linear_live_related 1024 6144 2 6293760 false v.v_postnorm (pg_layer_weights weights layer) v.v_residual
    normalized (dec_gateup_weight (u_real_parameters w)) (fun _ _ -> 0.0R);
  assert (Shared.matrix_approximates (u_live_matrix #6144 v.v_gateup)
    (Shared.linear_rows #7 #1024 #6144 (dec_gateup_weight (u_real_parameters w)) normalized));
  u_silu_gate_related v.v_gateup v.v_activated w normalized;
  let activated:Shared.matrix real 7 3072 = fun row col ->
    Shared.silu (Shared.linear_rows #7 #1024 #3072 w.gate_proj normalized row col) *.
      Shared.linear_rows #7 #1024 #3072 w.up_proj normalized row col in
  u_linear_live_related 3072 1024 0 12585216 true v.v_activated (pg_layer_weights weights layer) v.v_residual
    activated w.down_proj residual;
  let down = Shared.linear_rows #7 #3072 #1024 w.down_proj activated in
  assert (Shared.feed_forward w normalized == down) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.feed_forward]]; FStar.Tactics.V2.smt ());
  assert (Shared.matrix_approximates (u_live_matrix #1024 v.v_hidden) (Shared.residual_add residual down))
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_decoder_related (weights:lseq BF16.t opt_weight_count) (layer:Shared.index 28)
  (v:opt_block_values) (w:Shared.parameters real) (x:Shared.hidden real) (d:Shared.rotary_tables real)
  : Lemma
      (requires pg_layer_arithmetic weights layer v 8 /\ dec_attention_domain v.v_rotated /\
        Shared.parameters_approximates (pg_parameters weights layer) w /\
        Shared.matrix_approximates (u_live_matrix #1024 v.v_embedding) x /\
        Shared.rotary_approximates (u_actual_rotary v.v_rope) d)
      (ensures Shared.layer_approximates (u_native_layer v) (Shared.decoder_layer w d x)) =
  pg_weights_related weights layer w;
  dec_source_weight_views (pg_layer_weights weights layer) (u_real_parameters w);
  let normalized = Shared.normalize_hidden w.input_layernorm x in
  assert (Shared.matrix_approximates (u_live_matrix #1024 v.v_normalized0) normalized);
  u_linear_live_related 1024 4096 1 2048 false v.v_normalized0 (pg_layer_weights weights layer) v.v_embedding
    normalized (dec_qkv_weight (u_real_parameters w)) (fun _ _ -> 0.0R);
  u_qkv_related v.v_qkv w normalized;
  let q0 = Shared.split_heads #real #16 (Shared.linear_rows #7 #1024 #2048 w.q_proj normalized) in
  let k0 = Shared.split_heads #real #8 (Shared.linear_rows #7 #1024 #1024 w.k_proj normalized) in
  let value = Shared.split_heads #real #8 (Shared.linear_rows #7 #1024 #1024 w.v_proj normalized) in
  u_rope_heads_related v.v_qkv (pg_layer_weights weights layer) v.v_rope v.v_rotated w d q0 k0 value;
  let query = Shared.rotary #16 d (Shared.normalize_heads #16 w.q_norm q0) in
  let key = Shared.rotary #8 d (Shared.normalize_heads #8 w.k_norm k0) in
  let meaning:u_attention_meaning = {
    u_query = query; u_key = key; u_value = value; u_data = u_real_rotary d
  } in
  u_attention_live_related v.v_rotated v.v_attention meaning;
  let attention = Shared.concatenate_heads
    (Shared.attention_values (Shared.attention_probabilities query (Shared.repeat_kv key)) (Shared.repeat_kv value)) in
  u_linear_live_related 2048 1024 0 4196352 true v.v_attention (pg_layer_weights weights layer) v.v_embedding
    attention w.o_proj x;
  let projected = Shared.linear_rows #7 #2048 #1024 w.o_proj attention in
  let residual = Shared.residual_add x projected in
  assert (Shared.matrix_approximates (u_live_matrix #1024 v.v_residual) residual);
  pg_mlp_related weights layer v w residual;
  let expected:Shared.layer_values real = {
    hidden_after = Shared.residual_add residual (Shared.feed_forward w (Shared.normalize_hidden w.post_attention_layernorm residual));
    keys = key; values = value
  } in
  assert (Shared.layer_approximates (u_native_layer v) expected);
  assert (Shared.decoder_layer w d x == expected) by
    (FStar.Tactics.V2.norm [delta_only [`%Shared.decoder_layer]]; FStar.Tactics.V2.smt ())
#pop-options
#push-options "--fuel 1 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
let pg_decoder_numerical (weights:lseq BF16.t opt_weight_count) (layer:Shared.index 28) (v:opt_block_values)
  : Lemma
      (requires pg_layer_arithmetic weights layer v 8)
      (ensures dec_attention_domain v.v_rotated ==>
        Shared.decoder_relation (pg_parameters weights layer) (u_live_matrix #1024 v.v_embedding)
          (u_actual_rotary v.v_rope) (u_native_layer v)) =
  let at (w:Shared.parameters real) (x:Shared.hidden real) (d:Shared.rotary_tables real)
    : Lemma
        (requires dec_attention_domain v.v_rotated /\
          Shared.parameters_approximates (pg_parameters weights layer) w /\
          Shared.matrix_approximates (u_live_matrix #1024 v.v_embedding) x /\
          Shared.rotary_approximates (u_actual_rotary v.v_rope) d)
        (ensures Shared.layer_approximates (u_native_layer v) (Shared.decoder_layer w d x)) =
    pg_decoder_related weights layer v w x d
  in C.forall_intro_3 (C.move_requires_3 at)
#pop-options

inline_for_extraction noextract
let pg_layer_output (v:pg_values) (layer:Shared.index 28) : GTot (Shared.layer_values BF16.t) =
  u_native_layer (pg_layer_values v layer)

inline_for_extraction noextract
let pg_cache (v:pg_values) (count:Shared.index 29) : GTot (Shared.cache_state BF16.t) =
  fun layer -> if layer < count then Shared.Prefilled {
    cached_keys = (pg_layer_output v layer).keys;
    cached_values = (pg_layer_output v layer).values
  } else Shared.Empty

inline_for_extraction noextract
let pg_cache_step (v:pg_values) (layer:Shared.index 28)
  : Lemma (Shared.same_cache (pg_cache v (layer + 1))
    (Shared.update_cache (pg_cache v layer) layer (pg_layer_output v layer))) = ()

inline_for_extraction noextract
let pg_attention_domain (v:pg_values) : prop =
  forall (layer:Shared.index 28). source_block_score_domain (pg_layer_values v layer).v_rotated

inline_for_extraction noextract
let pg_parameters_zero_related (weights:lseq BF16.t opt_weight_count) (w:Shared.parameters real)
  : Lemma (
    Shared.parameters_approximates (pg_parameters weights 0) w <==>
      Shared.parameters_approximates (source_block_parameters weights) w) = ()

inline_for_extraction noextract
let pg_layer_input_link (v:pg_values) (layer:Shared.index 28{layer > 0})
  : Lemma ((pg_layer_values v layer).v_embedding == (pg_layer_values v (layer - 1)).v_hidden) = ()

inline_for_extraction noextract
let pg_all_decoder_edges (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) (v:pg_values)
  : Lemma
      (requires pg_decoder_stage weights prompt v 253 /\ pg_attention_domain v)
      (ensures
        Shared.decoder_relation (pg_parameters weights 0) (u_live_matrix #1024 (pg_slot v 0))
          (u_actual_rotary (pg_slot v 1)) (pg_layer_output v 0) /\
        (forall (layer:Shared.index 28{layer > 0}).
          Shared.decoder_relation (pg_parameters weights layer) (pg_layer_output v (layer - 1)).hidden_after
            (u_actual_rotary (pg_slot v 1)) (pg_layer_output v layer))) =
  pg_decoder_stage_definition weights prompt v 253;
  let first (w:Shared.parameters real) (xr:Shared.hidden real) (dr:Shared.rotary_tables real)
    : Lemma
        (requires Shared.parameters_approximates (pg_parameters weights 0) w /\
          Shared.matrix_approximates (u_live_matrix #1024 (pg_slot v 0)) xr /\
          Shared.rotary_approximates (u_actual_rotary (pg_slot v 1)) dr)
        (ensures Shared.layer_approximates (pg_layer_output v 0) (Shared.decoder_layer w dr xr)) =
    pg_parameters_zero_related weights w
  in C.forall_intro_3 (C.move_requires_3 first);
  let at (layer:Shared.index 28{layer > 0})
    : Lemma (
        Shared.decoder_relation (pg_parameters weights layer) (pg_layer_output v (layer - 1)).hidden_after
          (u_actual_rotary (pg_slot v 1)) (pg_layer_output v layer)) =
    u_source_score_domain (pg_layer_values v layer).v_rotated;
    pg_decoder_numerical weights layer (pg_layer_values v layer);
    pg_layer_input_link v layer
  in C.forall_intro at

inline_for_extraction noextract
let pg_same_execution_edges (weights:lseq BF16.t opt_weight_count) (v:pg_values)
  (prompt:opt_prompt_values) (tokens:lseq I64.t 8) : prop =
  (pg_attention_domain v ==>
    Shared.decoder_relation (pg_parameters weights 0) (u_live_matrix #1024 (pg_slot v 0))
      (u_actual_rotary (pg_slot v 1)) (pg_layer_output v 0) /\
    (forall (layer:Shared.index 28{layer > 0}).
      Shared.decoder_relation (pg_parameters weights layer) (pg_layer_output v (layer - 1)).hidden_after
        (u_actual_rotary (pg_slot v 1)) (pg_layer_output v layer))) /\
  Shared.final_norm_relation (pg_final_gamma weights) (pg_layer_output v 27).hidden_after
    (pg_normalized_view (pg_slot v 254)) /\
  Shared.projection_relation (pg_embedding_weights weights) (pg_normalized_view (pg_slot v 254))
    (pg_selector_scores (pg_slot v 255)) /\
  (forall (i:Shared.index 7). S.index tokens i == S.index prompt i) /\
  (Shared.selector_domain (pg_selector_scores (pg_slot v 255)) ==>
    0 <= I64.v (S.index tokens 7) /\ I64.v (S.index tokens 7) < 151936 /\
    (forall (i:Shared.token_id).
      F.lte (pg_selector_scores (pg_slot v 255) i)
        (pg_selector_scores (pg_slot v 255) (I64.v (S.index tokens 7))))) /\
  (forall (layer:Shared.index 28). pg_cache v 0 layer == Shared.Empty) /\
  (forall (layer:Shared.index 28). Shared.same_cache (pg_cache v (layer + 1))
    (Shared.update_cache (pg_cache v layer) layer (pg_layer_output v layer)))

inline_for_extraction noextract
let pg_source_edges_from_provenance (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v tokens 3)
      (ensures pg_same_execution_edges weights v prompt tokens) =
  if t2b (pg_attention_domain v) then pg_all_decoder_edges weights prompt v;
  pg_final_rms_related weights (pg_slot v 253) (pg_slot v 254);
  assert ((pg_layer_output v 27).hidden_after == u_live_matrix #1024 (pg_slot v 253));
  pg_projection_related weights (pg_slot v 254) (pg_slot v 255);
  if t2b (Shared.selector_domain (pg_selector_scores (pg_slot v 255))) then
    pg_selector_maximal (pg_slot v 255) prompt tokens;
  C.forall_intro (pg_cache_step v)

inline_for_extraction noextract
noeq type pg_source_outcome =
  | PGSourceFailed : error:RT.allocation_error -> pg_source_outcome
  | PGSourceReturned : output:pg_returned_result -> pg_source_outcome

inline_for_extraction noextract
val pg_source_program
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t) (ambient:slprop)
  : Pulse.Lib.Core.stt pg_source_outcome
    (cpu ** ST.stream_live stream ** RT.runtime stream ** ambient **
      E.epoch_live stream start ** u_input_storage weights_address prompt_address weights prompt)
    (fun outcome -> cpu ** ST.stream_live stream ** RT.runtime stream ** ambient ** (
    match outcome with
    | PGSourceFailed _ ->
      E.epoch_live stream start **
      P.pledge0 (E.epoch_done stream start) (u_input_storage weights_address prompt_address weights prompt)
    | PGSourceReturned output ->
      E.epoch_live stream (start + 256) ** RT.owner stream output.returned_tokens ** RT.owner stream output.returned_hidden **
      P.pledge0 (E.epoch_done stream (start + 256))
        (pg_source_return_frame weights_address prompt_address weights prompt output)))
#pop-options


#push-options "--fuel 3 --ifuel 1 --z3rlimit 60"
open Kuiper
open FStar.Real
module N = Kuiper.Approximates.Base

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
let source_cell (component:Shared.index 2) (row:Shared.index 4096) (col:Shared.index 128) : BF16.t =
  Casts.fcast #F32.t #BF16.t
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
  Casts.fcast_approx #F32.t #BF16.t value (target *. factor)

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

inline_for_extraction noextract
let mg_rotary_source_cell (component:Shared.index 2) (row:Shared.index 4096) (col:Shared.index 128)
  : Lemma (source_block_rotary_cell component row col == source_cell component row col) =
  assert (source_block_rotary_cell component row col == source_cell component row col) by
    (FStar.Tactics.V2.norm [delta_only [
      `%source_block_rotary_cell; `%source_cell; `%source_angle; `%source_inverse_frequency]; iota; zeta];
     FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let mg_rotary_from_snapshot (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) (v:pg_values)
  : Lemma
      (requires pg_decoder_stage weights prompt v 253)
      (ensures Shared.rotary_initialized (source_rotary_state ()) /\
        (source_rotary_domain () ==>
          Shared.mixed_rotary_generation (source_rotary_state ()) (u_actual_rotary (pg_slot v 1)))) =
  pg_decoder_stage_definition weights prompt v 253;
  let at (row:Shared.index 7) (col:Shared.index 128)
    : Lemma (
      (u_actual_rotary (pg_slot v 1)).cosine row col == source_cell 0 row col /\
      (u_actual_rotary (pg_slot v 1)).sine row col == source_cell 1 row col) =
    mg_rotary_source_cell 0 row col;
    mg_rotary_source_cell 1 row col
  in C.forall_intro_2 at;
  source_rotary_generation (u_actual_rotary (pg_slot v 1))


open Kuiper
open FStar.Real

inline_for_extraction noextract
let mg_prompt_tokens (prompt:opt_prompt_values) : GTot Shared.prompt_tokens =
  S.init_ghost #Shared.token_id 7 (fun i -> I64.v (S.index prompt i))

inline_for_extraction noextract
let mg_model_weights (weights:lseq BF16.t opt_weight_count) : GTot (Shared.model_weights BF16.t) = {
  embedding = pg_embedding_weights weights;
  final_norm = pg_final_gamma weights;
  layers = pg_parameters weights
}

inline_for_extraction noextract
let mg_inputs (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) : GTot Shared.inputs = {
  weight = mg_model_weights weights; prompt = mg_prompt_tokens prompt; decode_steps = 1
}

inline_for_extraction noextract
let mg_tokens_valid (tokens:lseq I64.t 8) : prop =
  forall (i:Shared.index 8). 0 <= I64.v (S.index tokens i) /\ I64.v (S.index tokens i) < 151936

inline_for_extraction noextract
type mg_valid_tokens = tokens:lseq I64.t 8{mg_tokens_valid tokens}

inline_for_extraction noextract
let mg_token_view (tokens:mg_valid_tokens) : GTot Shared.returned_tokens =
  S.init_ghost #Shared.token_id 8 (fun i -> I64.v (S.index tokens i))

inline_for_extraction noextract
let mg_execution (v:pg_values) (selected:Shared.token_id)
  : GTot (Shared.mixed_execution F32.t BF16.t F32.t) = {
  embedded = u_live_matrix #1024 (pg_slot v 0);
  position_ids = (fun i -> i); cache_positions = (fun i -> i);
  checked_mask = Shared.prefill_mask;
  rotary_initial = source_rotary_state (); rotary_final = source_rotary_state ();
  rope = u_actual_rotary (pg_slot v 1);
  layer = pg_layer_output v; cache_at = pg_cache v;
  normalized = pg_normalized_view (pg_slot v 254);
  scores = pg_selector_scores (pg_slot v 255);
  selected = selected
}

inline_for_extraction noextract
let mg_outputs (v:pg_values) (tokens:mg_valid_tokens) : GTot Shared.outputs = {
  tokens = mg_token_view tokens;
  normalized_hidden = pg_normalized_view (pg_slot v 254)
}

inline_for_extraction noextract
let mg_after_state (#frame_t:Type0) (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:mg_valid_tokens) (frame:frame_t)
  : GTot (Shared.mixed_post_state F32.t BF16.t frame_t) = {
  owned_inputs = mg_inputs weights prompt;
  history = mg_token_view tokens;
  cache = pg_cache v 28;
  rotary = source_rotary_state ();
  frame = frame
}

inline_for_extraction noextract
let mg_layer_kv_exact (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (layer:Shared.index 28) (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128)
  : Lemma
      (requires pg_decoder_stage weights prompt v 253)
      (ensures
        (pg_layer_output v layer).keys row head lane ==
          S.index (pg_slot v (pg_index layer 4)) (row * 4096 + head * 512 + 256 + lane) /\
        (pg_layer_output v layer).values row head lane ==
          S.index (pg_slot v (pg_index layer 3)) (row * 4096 + head * 512 + 384 + lane)) =
  pg_decoder_stage_definition weights prompt v 253;
  if layer > 0 then
    u_rope_value_unchanged (pg_layer_values v layer).v_qkv (pg_layer_weights weights layer)
      (pg_layer_values v layer).v_rope (pg_layer_values v layer).v_rotated row head lane

inline_for_extraction noextract
let mg_cache_contents (v:pg_values)
  : Lemma (
    (forall (layer:Shared.index 28). pg_cache v 0 layer == Shared.Empty) /\
    (forall (layer:Shared.index 28).
      Shared.cache_length (pg_cache v 28 layer) == 7 /\
      Shared.same_cache_slot (pg_cache v 28 layer)
        (Shared.Prefilled {cached_keys = (pg_layer_output v layer).keys; cached_values = (pg_layer_output v layer).values}))) = ()

inline_for_extraction noextract
let mg_raw_token_append (prompt:opt_prompt_values) (tokens:lseq I64.t 8)
  : Lemma
      (requires forall (i:Shared.index 7). S.index tokens i == S.index prompt i)
      (ensures tokens == S.append prompt (S.create 1 (S.index tokens 7))) =
  let expected = S.append prompt (S.create 1 (S.index tokens 7)) in
  let at (i:Shared.index 8) : Lemma (S.index tokens i == S.index expected i) =
    if i < 7 then S.lemma_index_app1 prompt (S.create 1 (S.index tokens 7)) i
    else S.lemma_index_app2 prompt (S.create 1 (S.index tokens 7)) i
  in C.forall_intro at;
  S.lemma_eq_intro tokens expected;
  S.lemma_eq_elim tokens expected

inline_for_extraction noextract
let mg_token_range (logits:lseq BF16.t 1228800) (prompt:opt_prompt_values) (tokens:lseq I64.t 8)
  : Lemma
      (requires pg_selector_result logits prompt tokens)
      (ensures
        (I64.v (S.index tokens 7) == -1 \/
          (0 <= I64.v (S.index tokens 7) /\ I64.v (S.index tokens 7) < 151936)) /\
        (Shared.selector_domain (pg_selector_scores logits) ==> mg_tokens_valid tokens)) =
  if t2b (Shared.selector_domain (pg_selector_scores logits)) then (
    pg_selector_maximal logits prompt tokens;
    let at (i:Shared.index 8)
      : Lemma (0 <= I64.v (S.index tokens i) /\ I64.v (S.index tokens i) < 151936) =
      if i < 7 then assert (S.index tokens i == S.index prompt i)
    in C.forall_intro at)

inline_for_extraction noextract
let mg_embedding_exact (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values) (v:pg_values)
  : Lemma
      (requires pg_decoder_stage weights prompt v 253)
      (ensures Shared.same_matrix (u_live_matrix #1024 (pg_slot v 0))
        (Shared.embedding_lookup (mg_model_weights weights) (mg_prompt_tokens prompt))) =
  pg_decoder_stage_definition weights prompt v 253;
  let at (row:Shared.index 7) (col:Shared.index 1024)
    : Lemma (u_live_matrix #1024 (pg_slot v 0) row col ==
      Shared.embedding_lookup (mg_model_weights weights) (mg_prompt_tokens prompt) row col) = ()
  in C.forall_intro_2 at

inline_for_extraction noextract
let mg_raw_structural (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) : prop =
  Shared.same_matrix (u_live_matrix #1024 (pg_slot v 0))
    (Shared.embedding_lookup (mg_model_weights weights) (mg_prompt_tokens prompt)) /\
  tokens == S.append prompt (S.create 1 (S.index tokens 7)) /\
  (I64.v (S.index tokens 7) == -1 \/ (0 <= I64.v (S.index tokens 7) /\ I64.v (S.index tokens 7) < 151936)) /\
  Shared.rotary_initialized (source_rotary_state ()) /\
  (forall (layer:Shared.index 28).
    pg_cache v 0 layer == Shared.Empty /\
    Shared.cache_length (pg_cache v 28 layer) == 7 /\
    Shared.same_cache_slot (pg_cache v 28 layer)
      (Shared.Prefilled {cached_keys = (pg_layer_output v layer).keys; cached_values = (pg_layer_output v layer).values})) /\
  (forall (layer:Shared.index 28) (row:Shared.index 7) (head:Shared.index 8) (lane:Shared.index 128).
    (pg_layer_output v layer).keys row head lane ==
      S.index (pg_slot v (pg_index layer 4)) (row * 4096 + head * 512 + 256 + lane) /\
    (pg_layer_output v layer).values row head lane ==
      S.index (pg_slot v (pg_index layer 3)) (row * 4096 + head * 512 + 384 + lane))

inline_for_extraction noextract
let mg_raw_structural_from_snapshot (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v tokens 3)
      (ensures mg_raw_structural weights prompt v tokens /\
        (Shared.selector_domain (pg_selector_scores (pg_slot v 255)) ==> mg_tokens_valid tokens)) =
  mg_embedding_exact weights prompt v;
  mg_rotary_from_snapshot weights prompt v;
  mg_raw_token_append prompt tokens;
  mg_token_range (pg_slot v 255) prompt tokens;
  mg_cache_contents v;
  C.forall_intro_4 (mg_layer_kv_exact weights prompt v)

inline_for_extraction noextract
let mg_typed_token_append (prompt:opt_prompt_values) (tokens:mg_valid_tokens)
  : Lemma
      (requires forall (i:Shared.index 7). S.index tokens i == S.index prompt i)
      (ensures mg_token_view tokens ==
        Shared.append_token (mg_prompt_tokens prompt) (I64.v (S.index tokens 7))) =
  let actual = mg_token_view tokens in
  let expected = Shared.append_token (mg_prompt_tokens prompt) (I64.v (S.index tokens 7)) in
  let at (i:Shared.index 8) : Lemma (S.index actual i == S.index expected i) =
    if i < 7 then S.lemma_index_app1 (mg_prompt_tokens prompt) (S.create 1 (I64.v (S.index tokens 7))) i
    else S.lemma_index_app2 (mg_prompt_tokens prompt) (S.create 1 (I64.v (S.index tokens 7))) i
  in C.forall_intro at;
  S.lemma_eq_intro actual expected;
  S.lemma_eq_elim actual expected

inline_for_extraction noextract
let mg_structural (#frame_t:Type0) (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:mg_valid_tokens) (frame:frame_t)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v tokens 3)
      (ensures Shared.mixed_structural_postcondition (mg_inputs weights prompt) frame
        (mg_execution v (I64.v (S.index tokens 7))) (mg_outputs v tokens)
        (mg_after_state weights prompt v tokens frame)) =
  mg_embedding_exact weights prompt v;
  mg_rotary_from_snapshot weights prompt v;
  mg_typed_token_append prompt tokens;
  mg_cache_contents v;
  C.forall_intro (pg_cache_step v)

inline_for_extraction noextract
let mg_numerical (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) (selected:Shared.token_id)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v tokens 3 /\
        source_rotary_domain () /\ pg_attention_domain v)
      (ensures Shared.mixed_numerical_execution (mg_inputs weights prompt) (mg_execution v selected)) =
  mg_rotary_from_snapshot weights prompt v;
  pg_source_edges_from_provenance weights prompt v tokens

inline_for_extraction noextract
let mg_typed_result (#frame_t:Type0) (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:mg_valid_tokens) (frame:frame_t) : prop =
  let run = mg_execution v (I64.v (S.index tokens 7)) in
  let output = mg_outputs v tokens in
  let after = mg_after_state weights prompt v tokens frame in
  Shared.mixed_structural_postcondition (mg_inputs weights prompt) frame run output after /\
  (source_rotary_domain () /\ pg_attention_domain v ==>
    Shared.mixed_numerical_execution (mg_inputs weights prompt) run /\
    Shared.mixed_numerical_denotation (mg_inputs weights prompt) run) /\
  (Shared.selector_domain run.scores ==> Shared.maximal_actual_index run.scores run.selected) /\
  (source_rotary_domain () /\ pg_attention_domain v /\ Shared.selector_domain run.scores ==>
    Shared.mixed_postcondition (mg_inputs weights prompt) frame run output after /\
    Shared.matrix_approximates output.normalized_hidden
      (Shared.mathematical_hidden (Shared.real_weights (mg_model_weights weights)) (mg_prompt_tokens prompt)) /\
    Shared.maximal_actual_index run.scores (S.index output.tokens 7))

inline_for_extraction noextract
let mg_typed_result_from_snapshot (#frame_t:Type0) (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:mg_valid_tokens) (frame:frame_t)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v tokens 3)
      (ensures mg_typed_result weights prompt v tokens frame) =
  let run = mg_execution v (I64.v (S.index tokens 7)) in
  let output = mg_outputs v tokens in
  let after = mg_after_state weights prompt v tokens frame in
  mg_structural weights prompt v tokens frame;
  if t2b (source_rotary_domain () /\ pg_attention_domain v) then (
    mg_numerical weights prompt v tokens (I64.v (S.index tokens 7));
    Shared.mixed_numerical_execution_related (mg_inputs weights prompt) run);
  if t2b (Shared.selector_domain run.scores) then
    pg_selector_maximal (pg_slot v 255) prompt tokens;
  if t2b (source_rotary_domain () /\ pg_attention_domain v /\ Shared.selector_domain run.scores) then (
    assert (Shared.mixed_postcondition (mg_inputs weights prompt) frame run output after);
    Shared.mixed_postcondition_meaning (mg_inputs weights prompt) frame run output after)

inline_for_extraction noextract
let mg_program_result (#frame_t:Type0) (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) (frame:frame_t) : prop =
  mg_raw_structural weights prompt v tokens /\
  pg_same_execution_edges weights v prompt tokens /\
  (Shared.selector_domain (pg_selector_scores (pg_slot v 255)) ==> mg_tokens_valid tokens) /\
  (mg_tokens_valid tokens ==> mg_typed_result weights prompt v tokens frame)

inline_for_extraction noextract
let mg_program_result_from_snapshot (#frame_t:Type0) (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (v:pg_values) (tokens:lseq I64.t 8) (frame:frame_t)
  : Lemma
      (requires pg_suffix_arithmetic weights prompt v tokens 3)
      (ensures mg_program_result weights prompt v tokens frame) =
  pg_source_edges_from_provenance weights prompt v tokens;
  mg_raw_structural_from_snapshot weights prompt v tokens;
  if t2b (mg_tokens_valid tokens) then mg_typed_result_from_snapshot weights prompt v tokens frame


open Kuiper
open Pulse.Lib.Send

inline_for_extraction noextract
let program_return_frame (#frame_t:Type0)
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (output:pg_returned_result) (frame:frame_t) : slprop =
  on gpu_loc (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    (exists* (v:pg_values) (tokens:lseq I64.t 8).
      PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
      PA.pts_to (RT.data output.returned_tokens) tokens **
      pure (pg_suffix_arithmetic weights prompt v tokens 3 /\ mg_program_result weights prompt v tokens frame)))

inline_for_extraction noextract
irreducible
let program_return_placeless (#frame_t:Type0)
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (weights:lseq BF16.t opt_weight_count) (prompt:opt_prompt_values)
  (output:pg_returned_result) (frame:frame_t)
  : placeless (program_return_frame weights_address prompt_address weights prompt output frame) =
  placeless_on gpu_loc (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    (exists* (v:pg_values) (tokens:lseq I64.t 8).
      PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
      PA.pts_to (RT.data output.returned_tokens) tokens **
      pure (pg_suffix_arithmetic weights prompt v tokens 3 /\ mg_program_result weights prompt v tokens frame)))

inline_for_extraction noextract
val mg_pack_return (#frame_t:Type0) (#frame:erased frame_t)
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (output:pg_returned_result)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (#v:pg_values) (#tokens:lseq I64.t 8)
  : Pulse.Lib.Core.stt_ghost (unit) emp_inames
      (loc gpu_loc ** (PA.pts_to weights_address weights ** PA.pts_to prompt_address prompt **
    PA.pts_to (RT.data output.returned_hidden) (pg_slot v 254) **
    PA.pts_to (RT.data output.returned_tokens) tokens ** pure (pg_suffix_arithmetic weights prompt v tokens 3)))
      (fun _ -> loc gpu_loc ** (program_return_frame weights_address prompt_address weights prompt output frame))


inline_for_extraction noextract
val mg_strengthen_return_at (#frame_t:Type0) (#frame:erased frame_t)
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (output:pg_returned_result)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  : unit -> Pulse.Lib.Core.stt_ghost (unit) emp_inames
      (loc gpu_loc ** (pg_source_return_frame weights_address prompt_address weights prompt output))
      (fun _ -> loc gpu_loc ** (program_return_frame weights_address prompt_address weights prompt output frame))


inline_for_extraction noextract
let mg_strengthen_return (#frame_t:Type0) (#frame:erased frame_t)
  (weights_address:larray BF16.t opt_weight_count) (prompt_address:larray I64.t 7)
  (output:pg_returned_result)
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values) (_:unit)
  : Pulse.Lib.Core.stt_ghost unit emp_inames
      (pg_source_return_frame weights_address prompt_address weights prompt output)
      (fun _ -> program_return_frame weights_address prompt_address weights prompt output frame) =
  ghost_impersonate #emp_inames gpu_loc
    (pg_source_return_frame weights_address prompt_address weights prompt output)
    (program_return_frame weights_address prompt_address weights prompt output frame)
    #(pg_source_return_placeless weights_address prompt_address weights prompt output)
    #(program_return_placeless weights_address prompt_address weights prompt output frame)
    (mg_strengthen_return_at #frame_t #frame weights_address prompt_address output #weights #prompt)

inline_for_extraction noextract
let program_base_contract_carryover (#frame_t:Type0)
  (input:Shared.inputs) (before:frame_t) (run:Shared.execution F32.t F32.t)
  (output:Shared.outputs) (after:Shared.post_state F32.t frame_t)
  : Lemma (Shared.postcondition input before run output after <==>
    Shared.mixed_postcondition input before (Shared.mixed_legacy_execution run) output (Shared.mixed_legacy_state after)) =
  Shared.mixed_legacy_postcondition_equivalence input before run output after

inline_for_extraction noextract
val program_refines (#frame_t:Type0) (#logical_frame:erased frame_t)
  (weights_address:larray BF16.t opt_weight_count{KA.is_global_array weights_address})
  (prompt_address:larray I64.t 7{KA.is_global_array prompt_address})
  (#weights:erased (lseq BF16.t opt_weight_count)) (#prompt:erased opt_prompt_values)
  (stream:ST.stream_t) (#start:E.epoch_t) (ambient:slprop)
  : Pulse.Lib.Core.stt (pg_source_outcome)
      (cpu ** ST.stream_live stream ** RT.runtime stream ** ambient ** (E.epoch_live stream start ** u_input_storage weights_address prompt_address weights prompt))
      (fun outcome -> cpu ** ST.stream_live stream ** RT.runtime stream ** ambient ** ((
    match outcome with
    | PGSourceFailed _ ->
      E.epoch_live stream start **
      P.pledge0 (E.epoch_done stream start) (u_input_storage weights_address prompt_address weights prompt)
    | PGSourceReturned output ->
      E.epoch_live stream (start + 256) ** RT.owner stream output.returned_tokens ** RT.owner stream output.returned_hidden **
      P.pledge0 (E.epoch_done stream (start + 256))
        (program_return_frame weights_address prompt_address weights prompt output logical_frame))))

#pop-options
