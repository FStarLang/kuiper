module KPlay.Proof.Optimized.History

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 40"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module Model = KPlay.Proof.Shared.Model
module Memory = KPlay.Proof.Optimized.Memory

inline_for_extraction noextract
let writes (call:Model.invocation) (before:lseq (option bf16) (call.decode_steps * 1024))
  (hidden:lseq bf16 1024) (row:nat{row < call.decode_steps}) (count:nat{count <= 1024})
  : GTot (lseq (option bf16) (call.decode_steps * 1024)) =
  S.init_ghost (call.decode_steps * 1024) (fun i ->
    if row * 1024 <= i && i < row * 1024 + count
    then Some (S.index hidden (i - row * 1024)) else S.index before i)

let writes_zero (call:Model.invocation) (before:lseq (option bf16) (call.decode_steps * 1024))
  (hidden:lseq bf16 1024) (row:nat{row < call.decode_steps})
  : Lemma (writes call before hidden row 0 == before) =
  assert (S.equal (writes call before hidden row 0) before);
  S.lemma_eq_elim (writes call before hidden row 0) before

let writes_next (call:Model.invocation) (before:lseq (option bf16) (call.decode_steps * 1024))
  (hidden:lseq bf16 1024) (row:nat{row < call.decode_steps}) (i:nat{i < 1024})
  : Lemma (S.upd (writes call before hidden row i) (row * 1024 + i) (Some (S.index hidden i)) ==
      writes call before hidden row (i + 1)) =
  assert (S.equal
    (S.upd (writes call before hidden row i) (row * 1024 + i) (Some (S.index hidden i)))
    (writes call before hidden row (i + 1)));
  S.lemma_eq_elim
    (S.upd (writes call before hidden row i) (row * 1024 + i) (Some (S.index hidden i)))
    (writes call before hidden row (i + 1))

inline_for_extraction noextract
let hidden_after (call:Model.invocation) (position:nat{position < Model.calls call})
  (before:lseq (option bf16) (call.decode_steps * 1024)) (hidden:lseq bf16 1024)
  : GTot (lseq (option bf16) (call.decode_steps * 1024)) =
  if position < call.prompt_length - 1 then before
  else writes call before hidden (position - call.prompt_length + 1) 1024

inline_for_extraction noextract
let tokens_after (call:Model.invocation) (position:nat{position < Model.calls call})
  (before:lseq (option I64.t) call.decode_steps) (selected:I64.t)
  : GTot (lseq (option I64.t) call.decode_steps) =
  if position < call.prompt_length - 1 then before
  else S.upd before (position - call.prompt_length + 1) (Some selected)

noextract
fn capture (call:Model.invocation)
  (position:sz{SZ.v position < Model.calls call}) (length:sz{SZ.v length == call.prompt_length})
  (hidden:larray bf16 1024) (output:larray bf16 (call.decode_steps * 1024))
  (#hv:erased (lseq bf16 1024)) (#before:erased (lseq (option bf16) (call.decode_steps * 1024)))
  (#fh:perm)
  preserves gpu ** hidden |-> Frac fh hv
  requires A.pts_to_mask output before Memory.all_cells
  ensures A.pts_to_mask output (hidden_after call (SZ.v position) before hv) Memory.all_cells
{
  if (length -^ 1sz <=^ position) {
    let row = position +^ 1sz -^ length;
    writes_zero call before hv (SZ.v row);
    let mut index = 0sz;
    while (!index <^ 1024sz)
      invariant exists* (i:sz{i <= 1024}).
        index |-> i ** A.pts_to_mask output (writes call before hv (SZ.v row) (SZ.v i)) Memory.all_cells
      decreases (1024 - !index)
    {
      let i = !index;
      writes_next call before hv (SZ.v row) (SZ.v i);
      let value = hidden.(i);
      A.mask_write output (row *^ 1024sz +^ i) value;
      index := i +^ 1sz;
    };
  } else {
    ();
  }
}

noextract
fn emit (call:Model.invocation)
  (position:sz{SZ.v position < Model.calls call}) (length:sz{SZ.v length == call.prompt_length})
  (selected:I64.t) (output:larray I64.t call.decode_steps)
  (#before:erased (lseq (option I64.t) call.decode_steps))
  preserves gpu
  requires A.pts_to_mask output before Memory.all_cells
  ensures A.pts_to_mask output (tokens_after call (SZ.v position) before selected) Memory.all_cells
{
  if (length -^ 1sz <=^ position) {
    let row = position +^ 1sz -^ length;
    A.mask_write output row selected;
  } else {
    ();
  }
}
