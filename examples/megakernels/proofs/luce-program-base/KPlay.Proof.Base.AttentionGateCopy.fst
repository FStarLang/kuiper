module KPlay.Proof.Base.AttentionGateCopy

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module P = KPlay.Proof.Shared.Projection
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let writes (before:Transfer.scratch) (count:nat{count <= 2048}) : GTot Transfer.scratch =
  Seq.init_ghost 16000 (fun i ->
    if 8192 <= i && i < 8192 + count then
      let j = i - 8192 in
      Seq.index before (2048 + (j / 256) * 512 + 256 + j % 256)
    else Seq.index before i)

inline_for_extraction noextract
let zero (before:Transfer.scratch)
  : Lemma (writes before 0 == before) =
  assert (Seq.equal (writes before 0) before);
  Seq.lemma_eq_elim (writes before 0) before

inline_for_extraction noextract
let step (before:Transfer.scratch) (head:P.index 8) (i:P.index 256)
  : Lemma (
      Seq.index (writes before (head * 256 + i)) (2048 + head * 512 + 256 + i) ==
        Seq.index before (2048 + head * 512 + 256 + i) /\
      Seq.upd (writes before (head * 256 + i)) (8192 + head * 256 + i)
        (Seq.index before (2048 + head * 512 + 256 + i)) ==
      writes before (head * 256 + i + 1)) =
  assert (Seq.equal
    (Seq.upd (writes before (head * 256 + i)) (8192 + head * 256 + i)
      (Seq.index before (2048 + head * 512 + 256 + i)))
    (writes before (head * 256 + i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (writes before (head * 256 + i)) (8192 + head * 256 + i)
      (Seq.index before (2048 + head * 512 + 256 + i)))
    (writes before (head * 256 + i + 1))

inline_for_extraction noextract
fn apply
  (hidden:larray BF.t 16000) (#before:erased Transfer.scratch)
  preserves gpu
  requires A.pts_to hidden before
  ensures A.pts_to hidden (writes before 2048)
{
  zero before;
  let mut head = 0sz;
  while (!head <^ 8sz)
    invariant exists* (h:sz{SZ.v h <= 8}).
      head |-> h ** A.pts_to hidden (writes before (SZ.v h * 256))
    invariant gpu
    decreases (8 - SZ.v !head)
  {
    let h = !head;
    let mut i = 0sz;
    while (!i <^ 256sz)
      invariant exists* (j:sz{SZ.v j <= 256}).
        i |-> j ** A.pts_to hidden (writes before (SZ.v h * 256 + SZ.v j))
      invariant gpu
      decreases (256 - SZ.v !i)
    {
      let j = !i;
      step before (SZ.v h) (SZ.v j);
      let value = hidden.(SZ.add (SZ.add (SZ.add 2048sz (SZ.mul h 512sz)) 256sz) j);
      hidden.(SZ.add (SZ.add 8192sz (SZ.mul h 256sz)) j) <- value;
      i := SZ.add j 1sz;
    };
    head := SZ.add h 1sz;
  }
}
