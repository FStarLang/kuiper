module KPlay.Proof.Optimized.Block

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 50"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module C = FStar.Classical
module Norm = KPlay.Proof.Optimized.Norm
module Matrix = KPlay.Proof.Optimized.Matrix
module Memory = KPlay.Proof.Optimized.Memory
module Dot = KPlay.Proof.Optimized.Dot
module Math = KPlay.Proof.Optimized.DotMath

inline_for_extraction noextract
let interpretation (nw:nat) =
  lseq real nw & lseq real 2048 & lseq real 1024

inline_for_extraction noextract
let inputs (#nw:nat) (w:lseq bf16 nw) (x:lseq f32 2048)
  (h:lseq bf16 1024) (m:interpretation nw) : prop =
  w %~ m._1 /\ x %~ m._2 /\ h %~ m._3

inline_for_extraction noextract
let residual (#nw:nat) (offset:nat{offset + 1024 * 2048 <= nw})
  (m:interpretation nw) : GTot (lseq real 1024) =
  Matrix.residual offset ((m._1, m._2), m._3)

inline_for_extraction noextract
let mlp_input (#nw:nat)
  (out_offset norm_offset:nat{out_offset + 1024 * 2048 <= nw /\ norm_offset + 1024 <= nw})
  (m:interpretation nw) : GTot (lseq real 1024) =
  Norm.normalized (m._1, residual out_offset m) norm_offset

inline_for_extraction noextract
let product (#nw:nat)
  (out_offset norm_offset gate_offset up_offset:nat{
    out_offset + 1024 * 2048 <= nw /\ norm_offset + 1024 <= nw /\
    gate_offset + 3584 * 1024 <= nw /\ up_offset + 3584 * 1024 <= nw})
  (m:interpretation nw) : GTot (lseq real 3584) =
  Matrix.mlp_product gate_offset up_offset
    (m._1, mlp_input out_offset norm_offset m)

inline_for_extraction noextract
let result (#nw:nat)
  (out_offset norm_offset gate_offset up_offset down_offset:nat{
    out_offset + 1024 * 2048 <= nw /\ norm_offset + 1024 <= nw /\
    gate_offset + 3584 * 1024 <= nw /\ up_offset + 3584 * 1024 <= nw /\
    down_offset + 1024 * 3584 <= nw})
  (m:interpretation nw) : GTot (lseq real 1024) =
  Matrix.residual down_offset
    ((m._1, product out_offset norm_offset gate_offset up_offset m),
     residual out_offset m)

let lift_residual (#nw:nat) (offset:nat{offset + 1024 * 2048 <= nw})
  (w:lseq bf16 nw) (x:lseq f32 2048) (h after:lseq bf16 1024)
  : Lemma
      (requires forall (m:Matrix.residual_interpretation nw 2048).
        Matrix.residual_inputs w x h m ==> after %~ Matrix.residual offset m)
      (ensures forall (m:interpretation nw).
        inputs w x h m ==> after %~ residual offset m) =
  let related (m:interpretation nw{inputs w x h m})
    : Lemma (after %~ residual offset m) =
    assert (Matrix.residual_inputs w x h ((m._1, m._2), m._3))
  in C.forall_intro related

let lift_normalization (#nw:nat)
  (out_offset norm_offset:nat{out_offset + 1024 * 2048 <= nw /\ norm_offset + 1024 <= nw})
  (w:lseq bf16 nw) (x:lseq f32 2048) (h mid after:lseq bf16 1024)
  : Lemma
      (requires
        (forall (m:interpretation nw). inputs w x h m ==> mid %~ residual out_offset m) /\
        (forall (m:Norm.interpretation nw).
          Norm.inputs w mid m ==> after %~ Norm.normalized m norm_offset))
      (ensures forall (m:interpretation nw).
        inputs w x h m ==> after %~ mlp_input out_offset norm_offset m) =
  let related (m:interpretation nw{inputs w x h m})
    : Lemma (after %~ mlp_input out_offset norm_offset m) =
    assert (Norm.inputs w mid (m._1, residual out_offset m))
  in C.forall_intro related

let lift_product (#nw:nat)
  (out_offset norm_offset gate_offset up_offset:nat{
    out_offset + 1024 * 2048 <= nw /\ norm_offset + 1024 <= nw /\
    gate_offset + 3584 * 1024 <= nw /\ up_offset + 3584 * 1024 <= nw})
  (w:lseq bf16 nw) (x:lseq f32 2048) (h normalized:lseq bf16 1024)
  (projected:lseq (option f32) 8192)
  (initialized:squash (Memory.ready projected 0 3584))
  : Lemma
      (requires
        (forall (m:interpretation nw). inputs w x h m ==>
          normalized %~ mlp_input out_offset norm_offset m) /\
        (forall (m:Math.interpretation nw 1024). Math.bf16_inputs w normalized m ==>
          (forall (i:nat{i < 3584}). Dot.option_models (S.index projected i)
            (S.index (Matrix.mlp_product gate_offset up_offset m) i))))
      (ensures forall (m:interpretation nw). inputs w x h m ==>
        Memory.values projected 0 3584 initialized %~
          product out_offset norm_offset gate_offset up_offset m) =
  let related (m:interpretation nw{inputs w x h m})
    : Lemma (Memory.values projected 0 3584 initialized %~
        product out_offset norm_offset gate_offset up_offset m) =
    assert (Math.bf16_inputs w normalized (m._1, mlp_input out_offset norm_offset m))
  in C.forall_intro related

let lift_result (#nw:nat)
  (out_offset norm_offset gate_offset up_offset down_offset:nat{
    out_offset + 1024 * 2048 <= nw /\ norm_offset + 1024 <= nw /\
    gate_offset + 3584 * 1024 <= nw /\ up_offset + 3584 * 1024 <= nw /\
    down_offset + 1024 * 3584 <= nw})
  (w:lseq bf16 nw) (x:lseq f32 2048) (h mid after:lseq bf16 1024)
  (activated:lseq f32 3584)
  : Lemma
      (requires
        (forall (m:interpretation nw). inputs w x h m ==>
          mid %~ residual out_offset m /\
          activated %~ product out_offset norm_offset gate_offset up_offset m) /\
        (forall (m:Matrix.residual_interpretation nw 3584).
          Matrix.residual_inputs w activated mid m ==>
            after %~ Matrix.residual down_offset m))
      (ensures forall (m:interpretation nw). inputs w x h m ==>
        after %~ result out_offset norm_offset gate_offset up_offset down_offset m) =
  let related (m:interpretation nw{inputs w x h m})
    : Lemma (after %~ result out_offset norm_offset gate_offset up_offset down_offset m) =
    assert (Matrix.residual_inputs w activated mid
      ((m._1, product out_offset norm_offset gate_offset up_offset m),
       residual out_offset m))
  in C.forall_intro related

(* reduced.cu:328-339, on the same hidden, normalized and projected buffers. *)
noextract
fn finish (#nw:erased pos{nw <= 752393024})
  (weights:larray bf16 nw) (activation:larray f32 2048)
  (hidden normalized:larray bf16 1024) (projected:larray f32 8192)
  (out_offset norm_offset gate_offset up_offset down_offset:sz{
    SZ.v out_offset + 1024 * 2048 <= nw /\ SZ.v norm_offset + 1024 <= nw /\
    SZ.v gate_offset + 3584 * 1024 <= nw /\ SZ.v up_offset + 3584 * 1024 <= nw /\
    SZ.v down_offset + 1024 * 3584 <= nw})
  (#wv:erased (lseq bf16 nw)) (#av:erased (lseq f32 2048))
  (#hv:erased (lseq bf16 1024)) (#nv:erased (lseq (option bf16) 1024))
  (#pv:erased (lseq (option f32) 8192)) (#fw #fa:perm)
  preserves gpu ** weights |-> Frac fw wv ** activation |-> Frac fa av
  requires hidden |-> hv ** A.pts_to_mask normalized nv Memory.all_cells **
    A.pts_to_mask projected pv Memory.all_cells
  ensures exists* (after:lseq bf16 1024) (next_norm:lseq bf16 1024)
                  (next_projected:lseq (option f32) 8192).
    hidden |-> after ** normalized |-> next_norm **
    A.pts_to_mask projected next_projected Memory.all_cells **
    pure (Memory.ready next_projected 0 3584 /\
      (forall (i:nat{3584 <= i /\ i < 8192}). S.index next_projected i == S.index pv i) /\
      (forall (m:interpretation nw). inputs wv av hv m ==>
        after %~ result (SZ.v out_offset) (SZ.v norm_offset)
          (SZ.v gate_offset) (SZ.v up_offset) (SZ.v down_offset) m))
{
  Matrix.project_residual weights activation hidden 2048sz out_offset;
  with mid. assert hidden |-> mid;
  lift_residual (SZ.v out_offset) wv av hv mid;
  Norm.normalize_hidden weights hidden normalized norm_offset;
  with normed. assert normalized |-> normed;
  lift_normalization (SZ.v out_offset) (SZ.v norm_offset) wv av hv mid normed;
  Matrix.mlp_rows weights normalized projected gate_offset up_offset;
  with product_values. assert A.pts_to_mask projected product_values Memory.all_cells;
  let initialized : squash (Memory.ready product_values 0 3584) = ();
  lift_product (SZ.v out_offset) (SZ.v norm_offset)
    (SZ.v gate_offset) (SZ.v up_offset) wv av hv normed product_values initialized;
  let values = Memory.borrow projected 0sz 3584sz #product_values #1.0R initialized;
  Matrix.project_residual weights values hidden 3584sz down_offset;
  with after. assert hidden |-> after;
  lift_result (SZ.v out_offset) (SZ.v norm_offset)
    (SZ.v gate_offset) (SZ.v up_offset) (SZ.v down_offset)
    wv av hv mid after (Memory.values product_values 0 3584 initialized);
  Memory.restore projected 0 3584 #product_values
    #(Memory.values product_values 0 3584 initialized) #1.0R;
  Memory.restore_readonly product_values 0 3584 initialized;
  rewrite A.pts_to_mask projected
    (Memory.replace product_values 0 (Memory.values product_values 0 3584 initialized))
    Memory.all_cells as A.pts_to_mask projected product_values Memory.all_cells;
}
