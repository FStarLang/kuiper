module KPlay.Proof.Base.SIMTLocal

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module S = KPlay.Proof.Shared.Attention
module Math = KPlay.Proof.Base.AttentionAlgebra
module SM = KPlay.Proof.Base.SIMTMath
module SIMT = KPlay.Proof.Base.SIMT
module Native = KPlay.Proof.Base.SIMTNative
module Layout = KPlay.Proof.Base.MatvecLayout
module Local = KPlay.Proof.Base.LocalArray
module Numeric = KPlay.Proof.Base.LocalReductionMath
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
fn row
  (#length:pos{length <= 12582912}) (g:SM.geometry length)
  (source:larray BF.t length) (hidden:larray BF.t 16000) (partial:larray FP.t 32)
  (which:sz{SZ.v which < SZ.v g.Layout.rows})
  (groups:sz{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:sz{0 < SZ.v chunk /\ SZ.v chunk <= 64}) (linear:bool)
  (#source_fraction #hidden_fraction:perm)
  (#sw:erased (lseq BF.t length)) (#sx:erased (lseq BF.t 16000)) (#sp:erased (Local.snapshot 32))
  (#wr:erased (S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)))
  (#xr:erased (S.vector (SZ.v g.Layout.columns))) (#scope:prop)
  preserves gpu ** source |-> Frac source_fraction sw ** hidden |-> Frac hidden_fraction sx
  requires A.pts_to_mask partial sp Init.all_cells
  requires pure (scope ==> SM.matrix_rep g sw wr /\ SM.input_rep g sx xr)
  returns result:BF.t
  ensures exists* (after:Local.snapshot 32). A.pts_to_mask partial after Init.all_cells **
    pure (scope ==> Kuiper.Approximates.Base.v_approximates result
      (S.linear (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which)))
{
  let mut i = 0sz;
  while (!i <^ groups)
    invariant exists* (j:sz{SZ.v j <= SZ.v groups}) (current:Local.snapshot 32).
      i |-> j ** A.pts_to_mask partial current Init.all_cells **
      pure (Local.ready current (SZ.v j) /\
        (scope ==> Numeric.partial_rep current (SZ.v j)
          (Math.chunk_group_sums (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk)
            (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which)))))
    invariant gpu ** source |-> Frac source_fraction sw ** hidden |-> Frac hidden_fraction sx
    decreases (SZ.v groups - SZ.v !i)
  {
    let j = !i;
    with current. assert A.pts_to_mask partial current Init.all_cells;
    let value = Native.group g source hidden which groups chunk j #source_fraction #hidden_fraction #sw #sx #wr #xr #scope;
    Numeric.stored #32 scope current (SZ.v j)
      (Math.chunk_group_sums (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk)
        (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which))) value;
    A.mask_write partial j value;
    i := SZ.add j 1sz;
  };
  with before. assert A.pts_to_mask partial before Init.all_cells;
  let value = Local.reduce #32 partial groups linear #before ();
  Numeric.reduce_approx #32 scope before (SZ.v groups) ()
    (Math.chunk_group_sums (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk)
      (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which))) linear value;
  Math.chunk_groups_sum (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk)
    (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which));
  SM.conditional_round scope value (S.linear (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which));
  Cast.cast_f32_to_bf16 value
}

inline_for_extraction noextract
fn projection
  (#length:pos{length <= 12582912}) (g:SM.geometry length)
  (cache:larray BF.t length) (hidden:larray BF.t 16000)
  (groups:sz{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:sz{0 < SZ.v chunk /\ SZ.v chunk <= 64}) (linear:bool)
  (#fraction:perm) (#sw:erased (lseq BF.t length)) (#sx:erased (lseq BF.t 16000))
  (#wr:erased (S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)))
  (#xr:erased (S.vector (SZ.v g.Layout.columns))) (#scope:prop)
  preserves gpu ** cache |-> Frac fraction sw
  requires hidden |-> sx
  requires pure (scope ==> SM.matrix_rep g sw wr /\ SM.input_rep g sx xr)
  ensures exists* (result:lseq BF.t 16000). hidden |-> result **
    pure (SIMT.frame g sx result (SZ.v g.Layout.rows) /\ (scope ==> SIMT.produced g result wr xr (SZ.v g.Layout.rows)))
{
  let mut partial : A.array FP.t = [| 32sz |];
  let mut r = 0sz;
  while (!r <^ g.Layout.rows)
    invariant exists* (which:sz{SZ.v which <= SZ.v g.Layout.rows})
      (current:lseq BF.t 16000) (temporary:Local.snapshot 32).
      r |-> which ** hidden |-> current ** A.pts_to_mask partial temporary Init.all_cells **
      pure (SIMT.frame g sx current (SZ.v which) /\ (scope ==> SIMT.produced g current wr xr (SZ.v which)))
    invariant gpu ** cache |-> Frac fraction sw
    decreases (SZ.v g.Layout.rows - SZ.v !r)
  {
    let which = !r;
    with current temporary. assert hidden |-> current ** A.pts_to_mask partial temporary Init.all_cells;
    SIMT.input_frame g sx current (SZ.v which) xr;
    let value = row g cache hidden partial which groups chunk linear #fraction #1.0R #sw #current #temporary #wr #xr #scope;
    SIMT.stored g sx current wr xr scope (SZ.v which) value;
    hidden.(SZ.add g.Layout.output_offset which) <- value;
    r := SZ.add which 1sz;
  }
}
