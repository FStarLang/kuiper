module KPlay.Proof.Base.HistoryNative

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module BF = Kuiper.BFloat16
module P = KPlay.Proof.Shared.Projection
module Model = KPlay.Proof.Shared.Model
module Transfer = KPlay.Proof.Base.Transfer
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
let retained_row (call:Model.invocation)
  (position:nat{position < Model.calls call /\ call.prompt_length - 1 <= position})
  : P.index call.decode_steps =
  position - call.prompt_length + 1

inline_for_extraction noextract
let hidden_writes (call:Model.invocation)
  (before:lseq (option BF.t) (call.decode_steps * 1024))
  (hidden:Transfer.scratch) (row:P.index call.decode_steps) (count:nat{count <= 1024})
  : GTot (lseq (option BF.t) (call.decode_steps * 1024)) =
  Seq.init_ghost (call.decode_steps * 1024) (fun i ->
    if row * 1024 <= i && i < row * 1024 + count
    then Some (Seq.index hidden (i - row * 1024)) else Seq.index before i)

inline_for_extraction noextract
let hidden_zero (call:Model.invocation)
  (before:lseq (option BF.t) (call.decode_steps * 1024))
  (hidden:Transfer.scratch) (row:P.index call.decode_steps)
  : Lemma (hidden_writes call before hidden row 0 == before) =
  assert (Seq.equal (hidden_writes call before hidden row 0) before);
  Seq.lemma_eq_elim (hidden_writes call before hidden row 0) before

inline_for_extraction noextract
let hidden_step (call:Model.invocation)
  (before:lseq (option BF.t) (call.decode_steps * 1024))
  (hidden:Transfer.scratch) (row:P.index call.decode_steps) (i:P.index 1024)
  : Lemma (
      Seq.upd (hidden_writes call before hidden row i) (row * 1024 + i) (Some (Seq.index hidden i)) ==
      hidden_writes call before hidden row (i + 1)) =
  assert (Seq.equal
    (Seq.upd (hidden_writes call before hidden row i) (row * 1024 + i) (Some (Seq.index hidden i)))
    (hidden_writes call before hidden row (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (hidden_writes call before hidden row i) (row * 1024 + i) (Some (Seq.index hidden i)))
    (hidden_writes call before hidden row (i + 1))

inline_for_extraction noextract
fn capture
  (call:Model.invocation)
  (position:sz{SZ.v position < Model.calls call /\ call.prompt_length - 1 <= SZ.v position})
  (prompt_length:sz{SZ.v prompt_length == call.prompt_length})
  (hidden:larray BF.t 16000) (output:larray BF.t (call.decode_steps * 1024))
  (#vh:erased Transfer.scratch) (#before:erased (lseq (option BF.t) (call.decode_steps * 1024)))
  (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction vh
  requires A.pts_to_mask output before Init.all_cells
  ensures A.pts_to_mask output
    (hidden_writes call before vh (retained_row call (SZ.v position)) 1024) Init.all_cells
{
  let row = SZ.add position 1sz -^ prompt_length;
  let offset = SZ.mul row 1024sz;
  hidden_zero call before vh (SZ.v row);
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (j:sz{SZ.v j <= 1024}).
      i |-> j ** A.pts_to_mask output (hidden_writes call before vh (SZ.v row) (SZ.v j)) Init.all_cells
    invariant gpu ** A.pts_to hidden #fraction vh
    decreases (1024 - SZ.v !i)
  {
    let j = !i;
    hidden_step call before vh (SZ.v row) (SZ.v j);
    let value = hidden.(j);
    A.mask_write output (SZ.add offset j) value;
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
fn emit
  (call:Model.invocation)
  (position:sz{SZ.v position < Model.calls call /\ call.prompt_length - 1 <= SZ.v position})
  (prompt_length:sz{SZ.v prompt_length == call.prompt_length})
  (next:larray I64.t 1) (tokens:larray I64.t call.decode_steps)
  (#selected:erased (lseq I64.t 1)) (#before:erased (lseq (option I64.t) call.decode_steps))
  preserves gpu ** next |-> selected
  requires A.pts_to_mask tokens before Init.all_cells
  ensures A.pts_to_mask tokens
    (Seq.upd before (retained_row call (SZ.v position)) (Some (Seq.index selected 0))) Init.all_cells
{
  let row = SZ.add position 1sz -^ prompt_length;
  let value = next.(0sz);
  A.mask_write tokens row value;
}

inline_for_extraction noextract
let captured_cell (call:Model.invocation)
  (before:lseq (option BF.t) (call.decode_steps * 1024))
  (hidden:Transfer.scratch) (row:P.index call.decode_steps) (i:P.index 1024)
  : Lemma (
      Seq.index (hidden_writes call before hidden row 1024) (row * 1024 + i) ==
      Some (Seq.index hidden i)) = ()

inline_for_extraction noextract
let capture_frame (call:Model.invocation)
  (before:lseq (option BF.t) (call.decode_steps * 1024))
  (hidden:Transfer.scratch) (row:P.index call.decode_steps) (i:P.index (call.decode_steps * 1024))
  : Lemma (
      ~(row * 1024 <= i /\ i < (row + 1) * 1024) ==>
      Seq.index (hidden_writes call before hidden row 1024) i == Seq.index before i) = ()
