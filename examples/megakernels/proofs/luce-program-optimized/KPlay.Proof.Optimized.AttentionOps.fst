module KPlay.Proof.Optimized.AttentionOps

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Fast = Kuiper.Float32.FastMath
module Neg = Kuiper.Float32.NativeNeg
module Cast = Kuiper.Float.Casts.Base
module Head = KPlay.Proof.Optimized.Head
module N = KPlay.Proof.Optimized.NormMath
module Memory = KPlay.Proof.Optimized.Memory
module Scalar = KPlay.Proof.Optimized.Scalar

inline_for_extraction noextract
let key_index (head:nat{head < 2}) (position:nat{position < 2048})
  (d:nat{d < 256}) : natlt 1048576 =
  head * 524288 + position * 256 + d

let key_index_time (head:nat{head < 2}) (position:nat{position < 2048})
  (d:nat{d < 256})
  : Lemma ((key_index head position d % 524288) / 256 == position) = ()

inline_for_extraction noextract
let cache_frame (position:nat{position < 2048})
  (before after:lseq bf16 1048576) : prop =
  forall (i:nat{i < 1048576}).
    (i % 524288) / 256 <> position ==> S.index after i == S.index before i

noextract
fn rope32 (value partner:f32) (index:sz{index < 64}) (position:sz{position < 2048})
  preserves gpu
  returns result:FP.t
  ensures emp
{
  let j = IC.uint32_to_int64 (SZ.sizet_to_u32 (index %^ 32sz));
  let p = IC.uint32_to_int64 (SZ.sizet_to_u32 position);
  let exponent = F.mul (F.of_int #f32 j) (N.inverse_width 32);
  let product = FP.mul_rn_ftz exponent (F.of_literal #f32 "23.2534961700439453125");
  let inverse = FP.exp2_approx_ftz (Neg.neg product);
  let frequency = FP.mul_rn_ftz (F.of_int #f32 p) inverse;
  let c = Fast.cos frequency;
  let s = Fast.sin frequency;
  if (index <^ 32sz) {
    let cross = FP.mul_rn_ftz partner s;
    FP.fma_rn_ftz value c (Neg.neg cross)
  } else {
    let straight = FP.mul_rn_ftz value c;
    FP.fma_rn_ftz partner s straight
  }
}

noextract
fn head_rstd (input:larray f32 256)
  (#before:erased (lseq f32 256)) (#fraction:perm)
  preserves gpu ** input |-> Frac fraction before
  returns result:f32
  ensures emp
{
  let squared = Head.squared input 256sz;
  let argument = FP.fma_rn_ftz squared (N.inverse_width 256) N.epsilon;
  FP.rsqrt_approx_ftz argument
}

noextract
fn norm_cell (weights:larray bf16 18352640) (projected:larray f32 5120)
  (input_index:sz{input_index < 5120}) (weight_index:sz{weight_index < 18352640})
  (scale:f32)
  (#wv:erased (lseq bf16 18352640)) (#pv:erased (lseq f32 5120)) (#fw #fp:perm)
  preserves gpu ** weights |-> Frac fw wv ** projected |-> Frac fp pv
  returns result:FP.t
  ensures emp
{
  let value = projected.(input_index);
  let scaled = FP.mul_rn_ftz value scale;
  let weight = Cast.cast_bf16_to_f32 (weights.(weight_index));
  let gain = FP.add_rn_ftz (one #f32) weight;
  FP.mul_rn_ftz scaled gain
}

noextract
fn key_cell (weights:larray bf16 18352640) (projected:larray f32 5120)
  (head:sz{head < 2}) (index:sz{index < 256}) (position:sz{position < 2048})
  (scale:f32)
  (#wv:erased (lseq bf16 18352640)) (#pv:erased (lseq f32 5120)) (#fw #fp:perm)
  preserves gpu ** weights |-> Frac fw wv ** projected |-> Frac fp pv
  returns result:FP.t
  ensures emp
{
  let start = 4096sz +^ head *^ 256sz;
  let normed = norm_cell weights projected (start +^ index)
    (SZ.uint32_to_sizet 16255232ul +^ index) scale;
  if (index <^ 64sz) {
    let partner = if (index <^ 32sz) { index +^ 32sz } else { index -^ 32sz };
    let pv = norm_cell weights projected (start +^ partner)
      (SZ.uint32_to_sizet 16255232ul +^ partner) scale;
    rope32 normed pv index position
  } else {
    normed
  }
}

noextract
fn write_keys (weights:larray bf16 18352640) (projected:larray f32 5120)
  (keys values:larray bf16 1048576) (position:sz{position < 2048})
  (#wv:erased (lseq bf16 18352640)) (#pv:erased (lseq f32 5120))
  (#kv #vv:erased (lseq bf16 1048576)) (#fw #fp:perm)
  preserves gpu ** weights |-> Frac fw wv ** projected |-> Frac fp pv
  requires keys |-> kv ** values |-> vv
  ensures exists* (after_keys after_values:lseq bf16 1048576).
    keys |-> after_keys ** values |-> after_values **
    pure (cache_frame (SZ.v position) kv after_keys /\ cache_frame (SZ.v position) vv after_values)
{
  let mut head = 0sz;
  while (!head <^ 2sz)
    invariant exists* (h:sz{h <= 2}) (current_keys current_values:lseq bf16 1048576).
      head |-> h ** keys |-> current_keys ** values |-> current_values **
      pure (cache_frame (SZ.v position) kv current_keys /\ cache_frame (SZ.v position) vv current_values)
    decreases (2 - !head)
  {
    let h = !head;
    let start = 4096sz +^ h *^ 256sz;
    let key_projection = Memory.take projected start 256sz;
    let scale = head_rstd key_projection;
    Memory.give projected (SZ.v start) 256 #pv
      #(S.slice pv (SZ.v start) (SZ.v start + 256)) #fp;
    Memory.overwrite_readonly pv (SZ.v start) 256;
    let mut index = 0sz;
    while (!index <^ 256sz)
      invariant exists* (i:sz{i <= 256}) (current_keys current_values:lseq bf16 1048576).
        index |-> i ** keys |-> current_keys ** values |-> current_values **
        pure (cache_frame (SZ.v position) kv current_keys /\ cache_frame (SZ.v position) vv current_values)
      decreases (256 - !index)
    {
      let i = !index;
      let rotated = key_cell weights projected h i position scale;
      let offset = h *^ (SZ.uint32_to_sizet 524288ul) +^ position *^ 256sz +^ i;
      key_index_time (SZ.v h) (SZ.v position) (SZ.v i);
      keys.(offset) <- Cast.cast_f32_to_bf16 rotated;
      let raw = projected.(4608sz +^ h *^ 256sz +^ i);
      values.(offset) <- Cast.cast_f32_to_bf16 raw;
      index := i +^ 1sz;
    };
    head := h +^ 1sz;
  }
}

inline_for_extraction noextract
let query_frame (head:nat{head < 8}) (before after:lseq f32 5120) : prop =
  forall (i:nat{i < 5120}).
    (i < head * 512 \/ head * 512 + 256 <= i) ==> S.index after i == S.index before i

noextract
fn transform_query (weights:larray bf16 18352640) (projected:larray f32 5120)
  (head:sz{head < 8}) (position:sz{position < 2048})
  (#wv:erased (lseq bf16 18352640)) (#pv:erased (lseq f32 5120)) (#fw:perm)
  preserves gpu ** weights |-> Frac fw wv
  requires projected |-> pv
  ensures exists* (after:lseq f32 5120). projected |-> after **
    pure (query_frame (SZ.v head) pv after)
{
  let start = head *^ 512sz;
  let query = Memory.take projected start 256sz;
  let scale = head_rstd query;
  Memory.give projected (SZ.v start) 256 #pv
    #(S.slice pv (SZ.v start) (SZ.v start + 256)) #1.0R;
  Memory.overwrite_readonly pv (SZ.v start) 256;
  let mut index = 0sz;
  while (!index <^ 32sz)
    invariant exists* (i:sz{i <= 32}) (current:lseq f32 5120).
      index |-> i ** projected |-> current ** pure (query_frame (SZ.v head) pv current)
    decreases (32 - !index)
  {
    let i = !index;
    let lower = norm_cell weights projected (start +^ i)
      (SZ.uint32_to_sizet 16254976ul +^ i) scale;
    let upper = norm_cell weights projected (start +^ i +^ 32sz)
      (SZ.uint32_to_sizet 16255008ul +^ i) scale;
    let next_lower = rope32 lower upper i position;
    projected.(start +^ i) <- next_lower;
    let next_upper = rope32 upper lower (i +^ 32sz) position;
    projected.(start +^ i +^ 32sz) <- next_upper;
    index := i +^ 1sz;
  };
  index := 64sz;
  while (!index <^ 256sz)
    invariant exists* (i:sz{64 <= i /\ i <= 256}) (current:lseq f32 5120).
      index |-> i ** projected |-> current ** pure (query_frame (SZ.v head) pv current)
    decreases (256 - !index)
  {
    let i = !index;
    let normed = norm_cell weights projected (start +^ i)
      (SZ.uint32_to_sizet 16254976ul +^ i) scale;
    projected.(start +^ i) <- normed;
    index := i +^ 1sz;
  }
}

(* reduced.cu:293-300, retaining the eight-term contiguous lane accumulators. *)
noextract
fn score (projected:larray f32 5120) (keys:larray bf16 1048576)
  (head:sz{head < 8}) (time:sz{time < 2048})
  (#pv:erased (lseq f32 5120)) (#kv:erased (lseq bf16 1048576)) (#fp #fk:perm)
  preserves gpu ** projected |-> Frac fp pv ** keys |-> Frac fk kv
  returns result:f32
  ensures emp
{
  let mut lanes:A.array f32 = [| 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i:sz{i <= 32}) (lv:lseq (option f32) 32).
      lane |-> i ** A.pts_to_mask lanes lv Memory.all_cells **
      pure (Memory.ready lv 0 (SZ.v i))
    decreases (32 - !lane)
  {
    let i = !lane;
    let mut element = 0sz;
    let mut accumulator:f32 = zero;
    while (!element <^ 8sz)
      invariant exists* (e:sz{e <= 8}) (v:f32). element |-> e ** accumulator |-> v
      decreases (8 - !element)
    {
      let e = !element;
      let q = projected.(head *^ 512sz +^ i *^ 8sz +^ e);
      let k = Cast.cast_bf16_to_f32
        (keys.((head /^ 4sz) *^ (SZ.uint32_to_sizet 524288ul) +^ time *^ 256sz +^ i *^ 8sz +^ e));
      accumulator := FP.fma_rn_ftz q k (!accumulator);
      element := e +^ 1sz;
    };
    A.mask_write lanes i (!accumulator);
    lane := i +^ 1sz;
  };
  A.from_mask lanes;
  let dot = sum32 lanes;
  let result = FP.mul_rn_ftz dot (F.of_literal #f32 "0.0625");
  A.to_mask lanes;
  result
}
