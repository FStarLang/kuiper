module KPlay.Proof.Optimized.Delta

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module Neg = Kuiper.Float32.NativeNeg
module M = KPlay.Proof.Optimized.DeltaMath
module Strided = KPlay.Proof.Optimized.Strided
module Masks = KPlay.Proof.Optimized.Dot
module C = FStar.Classical

let negative_decay (old key query:lseq f32 128) (value beta decay kq:f32)
  : Lemma (forall (m:M.interpretation).
      M.inputs old key query value beta decay kq m ==>
      Neg.neg decay %~ (0.0R -. m.M.decay)) =
  let related (m:M.interpretation{M.inputs old key query value beta decay kq m})
    : Lemma (Neg.neg decay %~ (0.0R -. m.M.decay)) =
    Neg.neg_approx decay m.M.decay
  in C.forall_intro related

(* The activation store precedes every state-row store, as in reduced.cu. *)
noextract
fn update_row (row key query:larray f32 128) (activation:larray f32 2048)
  (output:sz{output < 2048}) (value beta decay kq:f32)
  (#old #kv #qv:erased (lseq f32 128))
  (#av:erased (lseq (option f32) 2048)) (#fk #fq:perm)
  preserves gpu ** key |-> Frac fk kv ** query |-> Frac fq qv
  requires row |-> old ** A.pts_to_mask activation av Masks.all_cells
  returns result:f32
  ensures exists* (after:lseq f32 128).
    row |-> after **
    A.pts_to_mask activation (S.upd av (SZ.v output) (Some result)) Masks.all_cells **
    pure (forall (m:M.interpretation). M.inputs old kv qv value beta decay kq m ==>
      after %~ M.result m /\ result %~ M.dot (M.result m) m.M.query)
{
  let stk = Strided.strided_dot row key 128sz;
  let sqv = Strided.strided_dot row query 128sz;
  M.dots old kv qv value beta decay kq stk sqv;
  negative_decay old kv qv value beta decay kq;
  let residual = FP.fma_rn_ftz (Neg.neg decay) stk value;
  lift_fma #M.interpretation (M.inputs old kv qv value beta decay kq)
    (fun m -> 0.0R -. m.M.decay) (fun m -> M.dot m.M.old m.M.key)
    (fun m -> m.M.value) (Neg.neg decay) stk value residual;
  let change = FP.mul_rn_ftz residual beta;
  lift_mul #M.interpretation (M.inputs old kv qv value beta decay kq)
    (fun m -> (0.0R -. m.M.decay) *. M.dot m.M.old m.M.key +. m.M.value)
    (fun m -> m.M.beta) residual beta change;
  assert pure (forall (m:M.interpretation).
    M.inputs old kv qv value beta decay kq m ==> change %~ M.change m);
  let correction = FP.mul_rn_ftz change kq;
  lift_mul #M.interpretation (M.inputs old kv qv value beta decay kq)
    M.change (fun m -> M.dot m.M.key m.M.query) change kq correction;
  let readout = FP.fma_rn_ftz decay sqv correction;
  lift_fma #M.interpretation (M.inputs old kv qv value beta decay kq)
    (fun m -> m.M.decay) (fun m -> M.dot m.M.old m.M.query)
    (fun m -> M.change m *. M.dot m.M.key m.M.query) decay sqv correction readout;
  C.forall_intro M.readout;
  A.mask_write activation output readout;
  let mut column = 0sz;
  while (!column <^ 128sz)
    invariant exists* (i:sz{i <= 128}) (values:lseq f32 128).
      column |-> i ** row |-> values **
      pure (forall (j:nat{SZ.v i <= j /\ j < 128}). S.index values j == S.index old j) **
      pure (forall (m:M.interpretation). M.inputs old kv qv value beta decay kq m ==>
        (forall (j:nat{j < SZ.v i}). S.index values j %~ S.index (M.result m) j))
    decreases (128 - !column)
  {
    let i = !column;
    let prior = row.(i);
    let k = key.(i);
    let update = FP.mul_rn_ftz k change;
    lift_mul #M.interpretation (M.inputs old kv qv value beta decay kq)
      (fun m -> S.index m.M.key (SZ.v i)) M.change k change update;
    let next = FP.fma_rn_ftz prior decay update;
    lift_fma #M.interpretation (M.inputs old kv qv value beta decay kq)
      (fun m -> S.index m.M.old (SZ.v i)) (fun m -> m.M.decay)
      (fun m -> S.index m.M.key (SZ.v i) *. M.change m) prior decay update next;
    C.forall_intro (M.result_cell (SZ.v i));
    row.(i) <- next;
    column := i +^ 1sz;
  };
  readout
}
