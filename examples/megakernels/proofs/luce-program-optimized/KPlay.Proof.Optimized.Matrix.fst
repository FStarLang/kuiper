module KPlay.Proof.Optimized.Matrix

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 50"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module Dot = KPlay.Proof.Optimized.Dot
module Math = KPlay.Proof.Optimized.DotMath
module Scalar = KPlay.Proof.Optimized.Scalar
module R = KPlay.Proof.Shared.Recurrent
module Memory = KPlay.Proof.Optimized.Memory
module C = FStar.Classical

inline_for_extraction noextract
let projected (#nw #n:nat) (rows:nat) (offset:nat{offset + rows * n <= nw})
  (model:Math.interpretation nw n) : GTot (lseq real rows) =
  S.init_ghost rows (fun row -> rsum (S.init_ghost n (Math.term model (offset + row * n))))

let projection_row (#nw #n:nat) (rows:nat) (offset:nat{offset + rows * n <= nw})
  (row:nat{row < rows}) (model:Math.interpretation nw n)
  : Lemma (
      S.index (projected rows offset model) row ==
      rsum (S.init_ghost n (Math.term model (offset + row * n)))) = ()

noextract
fn project_bf16 (#nw:erased pos{nw <= 752393024}) (#capacity:erased pos{capacity <= 8192})
  (weights:larray bf16 nw) (input:larray bf16 1024) (output:larray f32 capacity)
  (rows:sz{0 < SZ.v rows}) (weight_offset:sz{SZ.v weight_offset + SZ.v rows * 1024 <= nw})
  (start:sz{SZ.v start + SZ.v rows <= capacity})
  (#wv:erased (lseq bf16 nw)) (#xv:erased (lseq bf16 1024))
  (#ov:erased (lseq (option f32) capacity)) (#fw #fx:perm)
  preserves gpu ** weights |-> Frac fw wv ** input |-> Frac fx xv
  requires A.pts_to_mask output ov Memory.all_cells
  ensures exists* (after:lseq (option f32) capacity).
    A.pts_to_mask output after Memory.all_cells **
    pure (Memory.ready after (SZ.v start) (SZ.v rows) /\
      (forall (i:nat{i < capacity}). Memory.outside (SZ.v start) (SZ.v rows) i ==>
        S.index after i == S.index ov i) /\
      (forall (model:Math.interpretation nw 1024). Math.bf16_inputs wv xv model ==>
        (forall (row:nat{row < SZ.v rows}).
          Dot.option_models (S.index after (SZ.v start + row))
            (S.index (projected (SZ.v rows) (SZ.v weight_offset) model) row))))
{
  let mut row = 0sz;
  while (!row <^ rows)
    invariant exists* (r:sz{r <= rows}) (values:lseq (option f32) capacity).
      row |-> r ** A.pts_to_mask output values Memory.all_cells **
      pure (Memory.ready values (SZ.v start) (SZ.v r)) **
      pure (forall (i:nat{i < capacity}). Memory.outside (SZ.v start) (SZ.v r) i ==>
        S.index values i == S.index ov i) **
      pure (forall (model:Math.interpretation nw 1024). Math.bf16_inputs wv xv model ==>
        (forall (j:nat{j < SZ.v r}).
          Dot.option_models (S.index values (SZ.v start + j))
            (S.index (projected (SZ.v rows) (SZ.v weight_offset) model) j)))
    decreases (SZ.v rows - SZ.v !row)
  {
    let r = !row;
    let value = Dot.dot_bf16 weights input (weight_offset +^ r *^ 1024sz);
    C.forall_intro (projection_row #nw #1024 (SZ.v rows) (SZ.v weight_offset) (SZ.v r));
    A.mask_write output (start +^ r) value;
    row := r +^ 1sz;
  }
}

inline_for_extraction noextract
let residual_interpretation (nw n:nat) = Math.interpretation nw n & lseq real 1024

inline_for_extraction noextract
let residual_inputs (#nw #n:nat)
  (w:lseq bf16 nw) (x:lseq f32 n) (h:lseq bf16 1024)
  (model:residual_interpretation nw n) : prop =
  Math.float_inputs w x (fst model) /\ h %~ snd model

inline_for_extraction noextract
let residual (#nw #n:nat) (offset:nat{offset + 1024 * n <= nw})
  (model:residual_interpretation nw n) : GTot (lseq real 1024) =
  S.init_ghost 1024 (fun i ->
    S.index (projected 1024 offset (fst model)) i +. S.index (snd model) i)

let residual_dot (#nw #n:nat)
  (w:lseq bf16 nw) (x:lseq f32 n) (h:lseq bf16 1024)
  (offset:nat{offset + 1024 * n <= nw}) (row:nat{row < 1024}) (value:f32)
  : Lemma
      (requires forall (m:Math.interpretation nw n). Math.float_inputs w x m ==>
        value %~ rsum (S.init_ghost n (Math.term m (offset + row * n))))
      (ensures forall (m:residual_interpretation nw n). residual_inputs w x h m ==>
        value %~ S.index (projected 1024 offset (fst m)) row) =
  let related (m:residual_interpretation nw n{residual_inputs w x h m})
    : Lemma (value %~ S.index (projected 1024 offset (fst m)) row) =
    projection_row 1024 offset row (fst m)
  in C.forall_intro related

noextract
fn project_residual (#nw:erased pos{nw <= 752393024})
  (#n:erased pos{n == 2048 \/ n == 3584})
  (weights:larray bf16 nw) (input:larray f32 n) (hidden:larray bf16 1024)
  (count:sz{SZ.v count == n}) (offset:sz{SZ.v offset + 1024 * n <= nw})
  (#wv:erased (lseq bf16 nw)) (#xv:erased (lseq f32 n))
  (#hv:erased (lseq bf16 1024)) (#fw #fx:perm)
  preserves gpu ** weights |-> Frac fw wv ** input |-> Frac fx xv
  requires hidden |-> hv
  ensures exists* (after:lseq bf16 1024).
    hidden |-> after **
    pure (forall (m:residual_interpretation nw n). residual_inputs wv xv hv m ==>
      after %~ residual (SZ.v offset) m)
{
  let mut row = 0sz;
  while (!row <^ 1024sz)
    invariant exists* (r:sz{r <= 1024}) (values:lseq bf16 1024).
      row |-> r ** hidden |-> values **
      pure (forall (i:nat{SZ.v r <= i /\ i < 1024}). S.index values i == S.index hv i) **
      pure (forall (m:residual_interpretation nw n). residual_inputs wv xv hv m ==>
        (forall (i:nat{i < SZ.v r}). S.index values i %~ S.index (residual (SZ.v offset) m) i))
    decreases (1024 - !row)
  {
    let r = !row;
    let projection = Dot.dot_float weights input count (offset +^ r *^ count);
    residual_dot wv xv hv (SZ.v offset) (SZ.v r) projection;
    let previous = Cast.cast_bf16_to_f32 (hidden.(r));
    let next = FP.add_rn_ftz projection previous;
    lift_add #(residual_interpretation nw n) (residual_inputs wv xv hv)
      (fun m -> S.index (projected 1024 (SZ.v offset) (fst m)) (SZ.v r))
      (fun m -> S.index (snd m) (SZ.v r)) projection previous next;
    let stored = Cast.cast_f32_to_bf16 next;
    assert pure (forall (m:residual_interpretation nw n). residual_inputs wv xv hv m ==>
      stored %~ S.index (residual (SZ.v offset) m) (SZ.v r));
    hidden.(r) <- stored;
    row := r +^ 1sz;
  }
}

inline_for_extraction noextract
let mlp_product (#nw:nat)
  (gate_offset up_offset:nat{gate_offset + 3584 * 1024 <= nw /\ up_offset + 3584 * 1024 <= nw})
  (model:Math.interpretation nw 1024) : GTot (lseq real 3584) =
  S.init_ghost 3584 (fun row ->
    R.silu (S.index (projected 3584 gate_offset model) row) *.
    S.index (projected 3584 up_offset model) row)

noextract
fn mlp_rows (#nw:erased pos{nw <= 752393024})
  (weights:larray bf16 nw) (input:larray bf16 1024) (output:larray f32 8192)
  (gate_offset up_offset:sz{
    SZ.v gate_offset + 3584 * 1024 <= nw /\ SZ.v up_offset + 3584 * 1024 <= nw})
  (#wv:erased (lseq bf16 nw)) (#xv:erased (lseq bf16 1024))
  (#ov:erased (lseq (option f32) 8192)) (#fw #fx:perm)
  preserves gpu ** weights |-> Frac fw wv ** input |-> Frac fx xv
  requires A.pts_to_mask output ov Memory.all_cells
  ensures exists* (after:lseq (option f32) 8192).
    A.pts_to_mask output after Memory.all_cells **
    pure (Memory.ready after 0 3584 /\
      (forall (i:nat{3584 <= i /\ i < 8192}). S.index after i == S.index ov i) /\
      (forall (m:Math.interpretation nw 1024). Math.bf16_inputs wv xv m ==>
        (forall (i:nat{i < 3584}). Dot.option_models (S.index after i)
          (S.index (mlp_product (SZ.v gate_offset) (SZ.v up_offset) m) i))))
{
  let mut row = 0sz;
  while (!row <^ 3584sz)
    invariant exists* (r:sz{r <= 3584}) (values:lseq (option f32) 8192).
      row |-> r ** A.pts_to_mask output values Memory.all_cells **
      pure (Memory.ready values 0 (SZ.v r)) **
      pure (forall (i:nat{SZ.v r <= i /\ i < 8192}). S.index values i == S.index ov i) **
      pure (forall (m:Math.interpretation nw 1024). Math.bf16_inputs wv xv m ==>
        (forall (i:nat{i < SZ.v r}). Dot.option_models (S.index values i)
          (S.index (mlp_product (SZ.v gate_offset) (SZ.v up_offset) m) i)))
    decreases (3584 - !row)
  {
    let r = !row;
    let gate = Dot.dot_bf16 weights input (gate_offset +^ r *^ 1024sz);
    let up = Dot.dot_bf16 weights input (up_offset +^ r *^ 1024sz);
    C.forall_intro (projection_row #nw #1024 3584 (SZ.v gate_offset) (SZ.v r));
    C.forall_intro (projection_row #nw #1024 3584 (SZ.v up_offset) (SZ.v r));
    let activated = Scalar.silu32 gate;
    let value = FP.mul_rn_ftz activated up;
    lift_mul #(Math.interpretation nw 1024) (Math.bf16_inputs wv xv)
      (fun m -> R.silu (S.index (projected 3584 (SZ.v gate_offset) m) (SZ.v r)))
      (fun m -> S.index (projected 3584 (SZ.v up_offset) m) (SZ.v r))
      activated up value;
    A.mask_write output r value;
    row := r +^ 1sz;
  }
}
