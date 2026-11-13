module KPlay.Proof.Base.SIMT

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module Approx = Kuiper.Approximates.Base
module S = KPlay.Proof.Shared.Attention
module Model = KPlay.Proof.Shared.Model
module Layout = KPlay.Proof.Base.MatvecLayout
module Schedule = KPlay.Proof.Base.Schedule
module Source = KPlay.Proof.Base.SourceProjection
module SM = KPlay.Proof.Base.SIMTMath
module Native = KPlay.Proof.Base.SIMTNative

inline_for_extraction noextract
let frame (#length:pos{length <= 12582912}) (g:SM.geometry length) (before current:lseq BF.t 16000)
  (count:nat{count <= SZ.v g.Layout.rows}) : prop =
  forall (i:nat{i < 16000}).
    ~(SZ.v g.Layout.output_offset <= i /\ i < SZ.v g.Layout.output_offset + count) ==>
    Seq.index current i == Seq.index before i

inline_for_extraction noextract
let produced (#length:pos{length <= 12582912}) (g:SM.geometry length) (current:lseq BF.t 16000)
  (wr:S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns))
  (xr:S.vector (SZ.v g.Layout.columns))
  (count:nat{count <= SZ.v g.Layout.rows}) : prop =
  forall (i:nat{i < count}).
    Approx.v_approximates (Seq.index current (SZ.v g.Layout.output_offset + i))
      (S.linear (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr i)

inline_for_extraction noextract
let input_frame (#length:pos{length <= 12582912}) (g:SM.geometry length) (before current:lseq BF.t 16000)
  (count:nat{count <= SZ.v g.Layout.rows}) (logical:S.vector (SZ.v g.Layout.columns))
  : Lemma
      (requires frame g before current count)
      (ensures SM.input_rep g before logical ==> SM.input_rep g current logical) =
  let at_index (i:S.index (SZ.v g.Layout.columns))
    : Lemma (Seq.index current (SZ.v g.Layout.input_offset + i) ==
        Seq.index before (SZ.v g.Layout.input_offset + i)) = ()
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let stored (#length:pos{length <= 12582912}) (g:SM.geometry length) (before current:lseq BF.t 16000)
  (wr:S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns))
  (xr:S.vector (SZ.v g.Layout.columns)) (scope:prop)
  (count:nat{count < SZ.v g.Layout.rows}) (value:BF.t)
  : Lemma
      (requires frame g before current count /\
        (scope ==> produced g current wr xr count /\
          Approx.v_approximates value (S.linear (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr count)))
      (ensures
        frame g before (Seq.upd current (SZ.v g.Layout.output_offset + count) value) (count + 1) /\
        (scope ==> produced g (Seq.upd current (SZ.v g.Layout.output_offset + count) value) wr xr (count + 1))) =
  ()

inline_for_extraction noextract
fn projection
  (#length:pos{length <= 12582912}) (g:SM.geometry length)
  (cache:larray BF.t length) (hidden:larray BF.t 16000) (partial:larray FP.t 32)
  (groups:sz{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:sz{0 < SZ.v chunk /\ SZ.v chunk <= 64}) (linear:bool)
  (#fraction:perm)
  (#sw:erased (lseq BF.t length)) (#sx:erased (lseq BF.t 16000)) (#sp:erased (lseq FP.t 32))
  (#wr:erased (S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)))
  (#xr:erased (S.vector (SZ.v g.Layout.columns))) (#scope:prop)
  preserves gpu ** cache |-> Frac fraction sw
  requires hidden |-> sx ** partial |-> sp
  requires pure (scope ==> SM.matrix_rep g sw wr /\ SM.input_rep g sx xr)
  ensures exists* (result:lseq BF.t 16000) (temporary:lseq FP.t 32).
    hidden |-> result ** partial |-> temporary **
    pure (frame g sx result (SZ.v g.Layout.rows) /\
      (scope ==> produced g result wr xr (SZ.v g.Layout.rows)))
{
  let mut r = 0sz;
  while (!r <^ g.Layout.rows)
    invariant exists* (row:sz{SZ.v row <= SZ.v g.Layout.rows})
      (current:lseq BF.t 16000) (temporary:lseq FP.t 32).
      r |-> row ** hidden |-> current ** partial |-> temporary **
      pure (frame g sx current (SZ.v row) /\
        (scope ==> produced g current wr xr (SZ.v row)))
    invariant gpu ** cache |-> Frac fraction sw
    decreases (SZ.v g.Layout.rows - SZ.v !r)
  {
    let row = !r;
    with current temporary. assert hidden |-> current ** partial |-> temporary;
    input_frame g sx current (SZ.v row) xr;
    let value = Native.row g cache hidden partial row groups chunk linear
      #fraction #1.0R #sw #current #temporary #wr #xr #scope;
    stored g sx current wr xr scope (SZ.v row) value;
    hidden.(SZ.add g.Layout.output_offset row) <- value;
    r := SZ.add row 1sz;
  }
}

inline_for_extraction noextract
let source_bounds (length:pos{length <= 647}) (score:bool)
  : Lemma (
      (Schedule.simt_groups score length = 16 \/ Schedule.simt_groups score length = 32) /\
      0 < Schedule.simt_chunk score length /\ Schedule.simt_chunk score length <= 64) = ()

inline_for_extraction noextract
fn source_projection
  (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:S.full_attention_layer)
  (phase:Schedule.phase layer{
    Schedule.attention layer phase /\ not (Schedule.mma_path layer phase position)})
  (cache:larray BF.t 12582912) (hidden:larray BF.t 16000) (partial:larray FP.t 32)
  (#fraction:perm)
  (#sw:erased (lseq BF.t 12582912)) (#sx:erased (lseq BF.t 16000)) (#sp:erased (lseq FP.t 32))
  (#wr:erased (S.matrix (Schedule.projection_at call position layer phase).Schedule.rows
    (Schedule.projection_at call position layer phase).Schedule.columns))
  (#xr:erased (S.vector (Schedule.projection_at call position layer phase).Schedule.columns)) (#scope:prop)
  preserves gpu ** cache |-> Frac fraction sw
  requires hidden |-> sx ** partial |-> sp
  requires pure (scope ==>
    SM.matrix_rep (Source.cache_geometry call position layer phase) sw wr /\
    SM.input_rep (Source.cache_geometry call position layer phase) sx xr)
  ensures exists* (result:lseq BF.t 16000) (temporary:lseq FP.t 32).
    hidden |-> result ** partial |-> temporary **
    pure (frame (Source.cache_geometry call position layer phase) sx result
        (Schedule.projection_at call position layer phase).Schedule.rows /\
      (scope ==> produced (Source.cache_geometry call position layer phase) result wr xr
        (Schedule.projection_at call position layer phase).Schedule.rows))
{
  let g = Source.cache_geometry call position layer phase;
  let score = phase % 2 = 1;
  source_bounds (position + 1) score;
  let groups = SZ.uint_to_t (Schedule.simt_groups score (position + 1));
  let chunk = SZ.uint_to_t (Schedule.simt_chunk score (position + 1));
  let linear = Schedule.simt_linear_finish score (position + 1);
  projection g cache hidden partial groups chunk linear #fraction #sw #sx #sp #wr #xr #scope;
}
