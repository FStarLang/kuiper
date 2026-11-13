module KPlay.Proof.Optimized.Convolution

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 50"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module C = FStar.Classical
module FP = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module Scalar = KPlay.Proof.Optimized.Scalar
module R = KPlay.Proof.Shared.Recurrent
module Memory = KPlay.Proof.Optimized.Memory
module Masks = KPlay.Proof.Optimized.Dot

inline_for_extraction noextract
let shifted (#a:Type0) (old:lseq a 4) (raw:a) : GTot (lseq a 4) =
  S.init_ghost 4 (fun i -> if i < 3 then S.index old (i + 1) else raw)

inline_for_extraction noextract
let interpretation = lseq real 4 & lseq real 4 & real

inline_for_extraction noextract
let inputs (weights:lseq bf16 4) (old:lseq f32 4) (raw:f32)
  (m:interpretation) : prop =
  weights %~ m._1 /\ old %~ m._2 /\ raw %~ m._3

inline_for_extraction noextract
let term (m:interpretation) (i:nat) : GTot real =
  if i < 4 then S.index (shifted m._2 m._3) i *. S.index m._1 i else 0.0R

let shifted_models (weights:lseq bf16 4) (old:lseq f32 4) (raw:f32)
  : Lemma (forall (m:interpretation). inputs weights old raw m ==>
      shifted old raw %~ shifted m._2 m._3) = ()

let term_step (i:nat{i < 4}) (m:interpretation)
  : Lemma (prefix (i + 1) (term m) ==
      prefix i (term m) +. S.index (shifted m._2 m._3) i *. S.index m._1 i) = ()

(* reduced.cu:196-203: shift, read raw projection, convolve, then SiLU. *)
noextract
fn shift_convolve (kernel:larray bf16 4) (taps:larray f32 4)
  (projected:larray f32 8192) (channel:sz{channel < 6144})
  (#wv:erased (lseq bf16 4)) (#cv:erased (lseq f32 4))
  (#pv:erased (lseq f32 8192)) (#fw #fp:perm)
  preserves gpu ** kernel |-> Frac fw wv ** projected |-> Frac fp pv
  requires taps |-> cv
  returns result:f32
  ensures taps |-> shifted cv (S.index pv (SZ.v channel))
  ensures pure (forall (m:interpretation).
    inputs wv cv (S.index pv (SZ.v channel)) m ==>
      result %~ R.silu (prefix 4 (term m)))
{
  let mut tap = 0sz;
  while (!tap <^ 3sz)
    invariant exists* (t:sz{t <= 3}) (current:lseq f32 4).
      tap |-> t ** taps |-> current **
      pure (forall (j:nat{j < 4}).
        S.index current j ==
          (if j < SZ.v t then S.index cv (j + 1) else S.index cv j))
    decreases (3 - !tap)
  {
    let t = !tap;
    let next = taps.(t +^ 1sz);
    taps.(t) <- next;
    tap := t +^ 1sz;
  };
  let raw = projected.(channel);
  taps.(3sz) <- raw;
  with current. assert taps |-> current;
  assert pure (S.equal current (shifted cv (S.index pv (SZ.v channel))));
  S.lemma_eq_elim current (shifted cv (S.index pv (SZ.v channel)));
  shifted_models wv cv raw;
  let mut index = 0sz;
  let mut accumulator:f32 = zero;
  while (!index <^ 4sz)
    invariant exists* (i:sz{i <= 4}) (value:f32).
      index |-> i ** accumulator |-> value **
      pure (forall (m:interpretation). inputs wv cv raw m ==>
        value %~ prefix (SZ.v i) (term m))
    decreases (4 - !index)
  {
    let i = !index;
    let x = taps.(i);
    let w = Cast.cast_bf16_to_f32 (kernel.(i));
    let previous = !accumulator;
    let next = FP.fma_rn_ftz x w previous;
    lift_fma #interpretation (inputs wv cv raw)
      (fun m -> S.index (shifted m._2 m._3) (SZ.v i))
      (fun m -> S.index m._1 (SZ.v i))
      (fun m -> prefix (SZ.v i) (term m)) x w previous next;
    C.forall_intro (term_step (SZ.v i));
    accumulator := next;
    index := i +^ 1sz;
  };
  Scalar.silu32 (!accumulator)
}

let shared_shift (old:R.convolution_state) (raw:R.vector 6144)
  (channel:R.fin 6144) (tap:R.fin 4)
  : Lemma (
      S.index (shifted (S.init_ghost 4 (old channel)) (raw channel)) tap ==
        R.shift_convolution old raw channel tap) = ()

let shared_convolution (kernel:R.matrix 6144 4)
  (old:R.convolution_state) (raw:R.vector 6144) (channel:R.fin 6144)
  : Lemma (
      R.silu (prefix 4 (term
        (S.init_ghost 4 (kernel channel), S.init_ghost 4 (old channel), raw channel))) ==
      R.convolve kernel (R.shift_convolution old raw) channel) =
  let m:interpretation =
    (S.init_ghost 4 (kernel channel), S.init_ghost 4 (old channel), raw channel) in
  let f = fun (i:nat) ->
    if i < 4 then kernel channel i *. R.shift_convolution old raw channel i else 0.0R in
  let target:Kuiper.Common.between 0 4 -> GTot real =
    fun i -> kernel channel i *. R.shift_convolution old raw channel i in
  C.forall_intro (shared_shift old raw channel);
  prefix_ext 4 (term m) f;
  prefix_kuiper 4 f;
  bounded_sum_ext #4 f target 4;
  FStar.Pervasives.norm_spec [delta_only [`%R.convolve]; zeta; iota]
    (R.convolve kernel (R.shift_convolution old raw) channel)

noextract
fn channel (weights:larray bf16 21555360) (convolution:larray f32 24576)
  (projected:larray f32 8192) (c:sz{c < 6144})
  (#wv:erased (lseq bf16 21555360)) (#cv:erased (lseq f32 24576))
  (#pv:erased (lseq f32 8192)) (#fw #fp:perm)
  preserves gpu ** weights |-> Frac fw wv ** projected |-> Frac fp pv
  requires convolution |-> cv
  returns result:f32
  ensures convolution |->
    Memory.overwrite cv (SZ.v c * 4)
      (shifted (S.slice cv (SZ.v c * 4) (SZ.v c * 4 + 4)) (S.index pv (SZ.v c)))
{
  let offset = c *^ 4sz;
  let weight_offset = SZ.uint32_to_sizet 19433472ul +^ offset;
  let kernel = Memory.take weights weight_offset 4sz;
  let taps = Memory.take convolution offset 4sz;
  let result = shift_convolve kernel taps projected c;
  Memory.give convolution (SZ.v offset) 4 #cv
    #(shifted (S.slice cv (SZ.v offset) (SZ.v offset + 4)) (S.index pv (SZ.v c))) #1.0R;
  Memory.give weights (SZ.v weight_offset) 4 #wv
    #(S.slice wv (SZ.v weight_offset) (SZ.v weight_offset + 4)) #fw;
  Memory.overwrite_readonly wv (SZ.v weight_offset) 4;
  rewrite weights |-> Frac fw
    (Memory.overwrite #bf16 #21555360 #4 wv (SZ.v weight_offset)
      (S.slice wv (SZ.v weight_offset) (SZ.v weight_offset + 4)))
    as weights |-> Frac fw wv;
  result
}

inline_for_extraction noextract
let region_prefix (old:lseq f32 24576) (raw:lseq f32 8192)
  (head:nat{head < 16}) (region:nat{region < 3}) (count:nat{count <= 128})
  : GTot (lseq f32 24576) =
  S.init_ghost 24576 (fun i ->
    let c = i / 4 in
    if region * 2048 + head * 128 <= c &&
       c < region * 2048 + head * 128 + count
    then (if i % 4 < 3 then S.index old (i + 1) else S.index raw c)
    else S.index old i)

let region_zero (old:lseq f32 24576) (raw:lseq f32 8192)
  (head:nat{head < 16}) (region:nat{region < 3})
  : Lemma (region_prefix old raw head region 0 == old) =
  assert (S.equal (region_prefix old raw head region 0) old);
  S.lemma_eq_elim (region_prefix old raw head region 0) old

let region_successor (old:lseq f32 24576) (raw:lseq f32 8192)
  (head:nat{head < 16}) (region:nat{region < 3}) (count:nat{count < 128})
  : Lemma (
      let before = region_prefix old raw head region count in
      let c = region * 2048 + head * 128 + count in
      Memory.overwrite before (c * 4)
        (shifted (S.slice before (c * 4) (c * 4 + 4)) (S.index raw c)) ==
        region_prefix old raw head region (count + 1)) =
  let before = region_prefix old raw head region count in
  let c = region * 2048 + head * 128 + count in
  let after = Memory.overwrite before (c * 4)
    (shifted (S.slice before (c * 4) (c * 4 + 4)) (S.index raw c)) in
  assert (S.equal after (region_prefix old raw head region (count + 1)));
  S.lemma_eq_elim after (region_prefix old raw head region (count + 1))

noextract
fn region (weights:larray bf16 21555360) (convolution:larray f32 24576)
  (projected:larray f32 8192) (output:larray f32 128)
  (head:sz{head < 16}) (which:sz{which < 3})
  (#wv:erased (lseq bf16 21555360)) (#cv:erased (lseq f32 24576))
  (#pv:erased (lseq f32 8192)) (#ov:erased (lseq (option f32) 128)) (#fw #fp:perm)
  preserves gpu ** weights |-> Frac fw wv ** projected |-> Frac fp pv
  requires convolution |-> cv ** A.pts_to_mask output ov Memory.all_cells
  ensures convolution |-> region_prefix cv pv (SZ.v head) (SZ.v which) 128
  ensures exists* (values:lseq f32 128). output |-> values
{
  region_zero cv pv (SZ.v head) (SZ.v which);
  let mut index = 0sz;
  while (!index <^ 128sz)
    invariant exists* (i:sz{i <= 128}) (values:lseq (option f32) 128).
      index |-> i **
      convolution |-> region_prefix cv pv (SZ.v head) (SZ.v which) (SZ.v i) **
      A.pts_to_mask output values Memory.all_cells **
      pure (Memory.ready values 0 (SZ.v i))
    decreases (128 - !index)
  {
    let i = !index;
    let c = which *^ 2048sz +^ head *^ 128sz +^ i;
    let value = channel weights convolution projected c;
    region_successor cv pv (SZ.v head) (SZ.v which) (SZ.v i);
    A.mask_write output i value;
    index := i +^ 1sz;
  };
  A.from_mask output;
}

inline_for_extraction noextract
let heads_prefix (old:lseq f32 24576) (raw:lseq f32 8192) (heads:nat{heads <= 16})
  : GTot (lseq f32 24576) =
  S.init_ghost 24576 (fun i ->
    let c = i / 4 in
    if (c % 2048) / 128 < heads
    then (if i % 4 < 3 then S.index old (i + 1) else S.index raw c)
    else S.index old i)

let heads_zero (old:lseq f32 24576) (raw:lseq f32 8192)
  : Lemma (heads_prefix old raw 0 == old) =
  assert (S.equal (heads_prefix old raw 0) old);
  S.lemma_eq_elim (heads_prefix old raw 0) old

let heads_successor (old:lseq f32 24576) (raw:lseq f32 8192) (head:nat{head < 16})
  : Lemma (
      region_prefix (region_prefix (region_prefix (heads_prefix old raw head)
        raw head 0 128) raw head 1 128) raw head 2 128 ==
      heads_prefix old raw (head + 1)) =
  let after = region_prefix (region_prefix (region_prefix (heads_prefix old raw head)
    raw head 0 128) raw head 1 128) raw head 2 128 in
  assert (S.equal after (heads_prefix old raw (head + 1)));
  S.lemma_eq_elim after (heads_prefix old raw (head + 1))
