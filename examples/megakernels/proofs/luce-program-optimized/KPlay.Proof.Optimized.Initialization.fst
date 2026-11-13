module KPlay.Proof.Optimized.Initialization

#lang-pulse

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module Memory = KPlay.Proof.Optimized.Memory
module State = KPlay.Proof.Optimized.State

noextract
fn zero_float (#n:erased pos{n <= 4718592}) (buffer:larray f32 n)
  (count:sz{SZ.v count == n}) (#before:erased (lseq (option f32) n))
  preserves gpu
  requires A.pts_to_mask buffer before Memory.all_cells
  ensures buffer |-> S.create n (zero #f32)
{
  let mut index = 0sz;
  while (!index <^ count)
    invariant exists* (i:sz{i <= count}) (current:lseq (option f32) n).
      index |-> i ** A.pts_to_mask buffer current Memory.all_cells **
      pure (forall (j:nat{j < SZ.v i}). S.index current j == Some (zero #f32))
    decreases (SZ.v count - SZ.v !index)
  {
    let i = !index;
    A.mask_write buffer i (zero #f32);
    index := i +^ 1sz;
  };
  A.from_mask buffer;
  with after. assert buffer |-> after;
  assert pure (S.equal after (S.create n (zero #f32)));
  S.lemma_eq_elim after (S.create n (zero #f32));
}

noextract
fn zero_kv (keys values:larray bf16 6291456)
  (#kv #vv:erased (lseq (option bf16) 6291456))
  preserves gpu
  requires A.pts_to_mask keys kv Memory.all_cells ** A.pts_to_mask values vv Memory.all_cells
  ensures keys |-> S.create 6291456 (zero #bf16) ** values |-> S.create 6291456 (zero #bf16)
{
  let mut index = 0sz;
  while (!index <^ SZ.uint32_to_sizet 6291456ul)
    invariant exists* (i:sz{SZ.v i <= 6291456}) (k v:lseq (option bf16) 6291456).
      index |-> i ** A.pts_to_mask keys k Memory.all_cells ** A.pts_to_mask values v Memory.all_cells **
      pure (forall (j:nat{j < SZ.v i}).
        S.index k j == Some (zero #bf16) /\ S.index v j == Some (zero #bf16))
    decreases (6291456 - !index)
  {
    let i = !index;
    A.mask_write keys i (zero #bf16);
    A.mask_write values i (zero #bf16);
    index := i +^ 1sz;
  };
  A.from_mask keys;
  A.from_mask values;
  with k. assert keys |-> k;
  with v. assert values |-> v;
  assert pure (S.equal k (S.create 6291456 (zero #bf16)));
  assert pure (S.equal v (S.create 6291456 (zero #bf16)));
  S.lemma_eq_elim k (S.create 6291456 (zero #bf16));
  S.lemma_eq_elim v (S.create 6291456 (zero #bf16));
}

noextract
fn source (states:larray f32 4718592) (convolution:larray f32 442368)
  (keys values:larray bf16 6291456)
  (#sv:erased (lseq (option f32) 4718592)) (#cv:erased (lseq (option f32) 442368))
  (#kv #vv:erased (lseq (option bf16) 6291456))
  preserves gpu
  requires A.pts_to_mask states sv Memory.all_cells ** A.pts_to_mask convolution cv Memory.all_cells **
    A.pts_to_mask keys kv Memory.all_cells ** A.pts_to_mask values vv Memory.all_cells
  ensures A.pts_to states State.zeroed.State.states **
    A.pts_to convolution State.zeroed.State.convolution **
    A.pts_to keys State.zeroed.State.keys ** A.pts_to values State.zeroed.State.values
{
  zero_float #4718592 states (SZ.uint32_to_sizet 4718592ul);
  zero_float #442368 convolution (SZ.uint32_to_sizet 442368ul);
  zero_kv keys values;
}
