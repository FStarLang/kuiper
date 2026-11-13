module KPlay.Proof.BaseStorage

#lang-pulse

open Kuiper
module PA = Pulse.Lib.Array
module F = Kuiper.Floating
module S = FStar.Seq
module I32 = FStar.Int32
module SZ = Kuiper.SizeT
module IC = FStar.Int.Cast

inline_for_extraction noextract
let value (x:option f32{Some? x}) : f32 =
  match x with | Some v -> v

inline_for_extraction noextract
fn initialize_slots
  (indices:larray i64 256) (probabilities:larray f32 256)
  (#old_indices:erased (lseq i64 256)) (#old_probabilities:erased (lseq f32 256))
  preserves gpu
  requires PA.pts_to indices old_indices ** PA.pts_to probabilities old_probabilities
  ensures
    PA.pts_to indices (S.create 256 (-1L)) **
    PA.pts_to probabilities (S.create 256 (F.zero #f32))
{
  let mut index = 0l;
  while (I32.lt !index 256l)
    invariant exists* (i:i32{0 <= I32.v i /\ I32.v i <= 256})
      (owners:lseq i64 256) (weights:lseq f32 256).
      index |-> i ** PA.pts_to indices owners ** PA.pts_to probabilities weights **
      pure (forall (j:nat{j < I32.v i}).
        S.index owners j == -1L /\ S.index weights j == F.zero)
    decreases (256 - I32.v !index)
  {
    let i = !index;
    let offset = SZ.uint32_to_sizet (IC.int32_to_uint32 i);
    indices.(offset) <- -1L;
    probabilities.(offset) <- F.zero #f32;
    index := I32.add i 1l;
  };
  with owners weights. assert PA.pts_to indices owners ** PA.pts_to probabilities weights;
  assert pure (S.equal owners (S.create 256 (-1L)));
  assert pure (S.equal weights (S.create 256 (F.zero #f32)));
}
