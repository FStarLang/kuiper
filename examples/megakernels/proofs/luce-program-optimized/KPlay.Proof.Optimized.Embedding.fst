module KPlay.Proof.Optimized.Embedding

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module I32 = FStar.Int32
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module Memory = KPlay.Proof.Optimized.Memory
module Layout = KPlay.Proof.Optimized.Layout
module D = KPlay.Proof.Shared.Selection
module E = KPlay.Proof.Shared.Embedding
module Model = KPlay.Proof.Shared.Model
module O = KPlay.Proof.Shared.Observations

inline_for_extraction noextract
let token = t:I32.t{0 <= I32.v t /\ I32.v t < 248320}

inline_for_extraction noextract
let row (weights:Layout.packed) (token:D.token 248320) : GTot (lseq bf16 1024) =
  S.init_ghost 1024 (fun i -> S.index weights (token * 1024 + i))

let row_binding (weights:Layout.packed) (token:D.token 248320)
  : Lemma (
      row weights token == E.embedding (Layout.parameter_values weights).O.embedding token) =
  Layout.embedding_row (fun i -> S.index weights i) token;
  FStar.Pervasives.norm_spec [delta_only [`%Layout.parameter_values]; zeta; iota]
    (Layout.parameter_values weights).O.embedding;
  FStar.Pervasives.norm_spec [delta_only [`%row]; zeta; iota] (row weights token);
  FStar.Pervasives.norm_spec [delta_only [`%E.embedding]; zeta; iota]
    (E.embedding (Layout.parameter_values weights).O.embedding token)

inline_for_extraction noextract
let narrow (value:I64.t{0 <= I64.v value /\ I64.v value < 248320})
  : result:token{I32.v result == I64.v value} =
  IC.int64_to_int32 value

noextract
fn copy (weights:larray bf16 752393024) (hidden:larray bf16 1024) (selected:token)
  (#wv:erased (lseq bf16 752393024)) (#hv:erased (lseq (option bf16) 1024)) (#fw:perm)
  preserves gpu ** weights |-> Frac fw wv
  requires A.pts_to_mask hidden hv Memory.all_cells
  ensures hidden |-> row wv (I32.v selected)
{
  let token = SZ.uint32_to_sizet (IC.int32_to_uint32 selected);
  let offset = token *^ 1024sz;
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (i:sz{i <= 1024}) (current:lseq (option bf16) 1024).
      index |-> i ** A.pts_to_mask hidden current Memory.all_cells **
      pure (Memory.ready current 0 (SZ.v i)) **
      pure (forall (j:nat{j < SZ.v i}).
        S.index current j == Some (S.index wv (I32.v selected * 1024 + j)))
    decreases (1024 - !index)
  {
    let i = !index;
    let value = weights.(offset +^ i);
    A.mask_write hidden i value;
    index := i +^ 1sz;
  };
  A.from_mask hidden;
  with actual. assert hidden |-> actual;
  assert pure (S.equal actual (row wv (I32.v selected)));
  S.lemma_eq_elim actual (row wv (I32.v selected));
}

noextract
fn select_input (call:Model.invocation) (prompt:larray I64.t call.prompt_length)
  (position:sz{SZ.v position < Model.calls call})
  (length:sz{SZ.v length == call.prompt_length}) (previous:token)
  (#pv:erased (lseq I64.t call.prompt_length)) (#fp:perm)
  preserves gpu ** prompt |-> Frac fp pv
  requires pure (forall (i:nat{i < call.prompt_length}).
    0 <= I64.v (S.index pv i) /\ I64.v (S.index pv i) < 248320)
  returns result:token
  ensures pure (I32.v result ==
    (if SZ.v position < call.prompt_length
     then I64.v (S.index pv (SZ.v position)) else I32.v previous))
{
  if (position <^ length) {
    narrow (prompt.(position))
  } else {
    previous
  }
}
