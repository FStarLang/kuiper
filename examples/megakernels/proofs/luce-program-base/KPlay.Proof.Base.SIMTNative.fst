module KPlay.Proof.Base.SIMTNative

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
module F = Kuiper.Floating.Base
module Fast = Kuiper.Float32.FastMath
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module S = KPlay.Proof.Shared.Attention
module Layout = KPlay.Proof.Base.MatvecLayout
module Math = KPlay.Proof.Base.AttentionAlgebra
module SM = KPlay.Proof.Base.SIMTMath
module Reduction = KPlay.Proof.Base.ReductionNative
module Norm = KPlay.Proof.Base.Norm

inline_for_extraction noextract
fn accumulate_chunk
  (#length:pos{length <= 12582912}) (g:SM.geometry length)
  (source:larray BF.t length) (hidden:larray BF.t 16000)
  (row:sz{SZ.v row < SZ.v g.Layout.rows})
  (groups:sz{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:sz{0 < SZ.v chunk /\ SZ.v chunk <= 64})
  (group:sz{SZ.v group < SZ.v groups})
  (start:sz{SZ.v start < SZ.v g.Layout.columns /\
    SZ.v start % (SZ.v groups * SZ.v chunk) = SZ.v group * SZ.v chunk})
  (acc:Pulse.Lib.Reference.ref FP.t)
  (#source_fraction #hidden_fraction:perm)
  (#sw:erased (lseq BF.t length)) (#sx:erased (lseq BF.t 16000))
  (#wr:erased (S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)))
  (#xr:erased (S.vector (SZ.v g.Layout.columns))) (#scope:prop) (#before:erased FP.t)
  preserves gpu ** source |-> Frac source_fraction sw ** hidden |-> Frac hidden_fraction sx
  requires acc |-> before
  requires pure (scope ==> SM.matrix_rep g sw wr /\ SM.input_rep g sx xr /\
    Approx.v_approximates (reveal #FP.t before)
      (Math.bounded_sum (SZ.v start / (SZ.v groups * SZ.v chunk) * SZ.v chunk)
        (Math.chunk_terms (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v group)
          (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row)))))
  ensures exists* (result:FP.t). acc |-> result **
    pure (scope ==> Approx.v_approximates result
      (Math.bounded_sum ((SZ.v start / (SZ.v groups * SZ.v chunk) + 1) * SZ.v chunk)
        (Math.chunk_terms (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v group)
          (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row)))))
{
  Math.slots_bounds (SZ.v g.Layout.columns) (SZ.v groups * SZ.v chunk);
  SM.work_bound (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v start);
  let finish = SZ.add start chunk;
  let mut i = start;
  while (!i <^ finish && !i <^ g.Layout.columns)
    invariant exists* (j:sz{SZ.v start <= SZ.v j /\ SZ.v j <= SZ.v finish /\
      SZ.v j <= SZ.v g.Layout.columns}) (current:FP.t).
      i |-> j ** acc |-> current **
      pure (scope ==> Approx.v_approximates current
        (Math.bounded_sum (SZ.v start / (SZ.v groups * SZ.v chunk) * SZ.v chunk + SZ.v j - SZ.v start)
          (Math.chunk_terms (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v group)
            (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row)))))
    invariant gpu ** source |-> Frac source_fraction sw ** hidden |-> Frac hidden_fraction sx
    decreases (SZ.v finish - SZ.v !i)
  {
    let j = !i;
    Math.chunk_coordinate (Math.slots (SZ.v g.Layout.columns) (SZ.v groups * SZ.v chunk))
      (SZ.v groups) (SZ.v chunk) (SZ.v group)
      (SZ.v start / (SZ.v groups * SZ.v chunk)) (SZ.v j - SZ.v start);
    SM.address_bounds g (SZ.v row) (SZ.v j);
    let address = SZ.add (SZ.add g.Layout.offset (SZ.mul row g.Layout.row_stride))
      (SZ.mul j g.Layout.col_stride);
    let left = source.(address);
    let right = hidden.(SZ.add g.Layout.input_offset j);
    let old = !acc;
    let next = Fast.fma_rn (Cast.cast_bf16_to_f32 left) (Cast.cast_bf16_to_f32 right) old;
    SM.conditional_fma scope left right old next (SM.matrix_at g wr (SZ.v row) (SZ.v j)) (SM.vector_at g xr (SZ.v j))
      (Math.bounded_sum (SZ.v start / (SZ.v groups * SZ.v chunk) * SZ.v chunk + SZ.v j - SZ.v start)
        (Math.chunk_terms (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v group)
          (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row))));
    SM.conditional_chunk_step scope (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v group)
      (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row))
      (SZ.v start) (SZ.v j) next;
    let following = SZ.add j 1sz;
    assert pure (SZ.v following == SZ.v j + 1);
    acc := next;
    i := following;
  };
  let j = !i;
  assert pure (SZ.v j == SZ.v start + SZ.v chunk \/ SZ.v j == SZ.v g.Layout.columns);
  SM.chunk_finished (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v group)
    (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row))
    (SZ.v start) (SZ.v j);
  ()
}

inline_for_extraction noextract
fn group
  (#length:pos{length <= 12582912}) (g:SM.geometry length)
  (source:larray BF.t length) (hidden:larray BF.t 16000)
  (row:sz{SZ.v row < SZ.v g.Layout.rows})
  (groups:sz{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:sz{0 < SZ.v chunk /\ SZ.v chunk <= 64})
  (which:sz{SZ.v which < SZ.v groups})
  (#source_fraction #hidden_fraction:perm)
  (#sw:erased (lseq BF.t length)) (#sx:erased (lseq BF.t 16000))
  (#wr:erased (S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)))
  (#xr:erased (S.vector (SZ.v g.Layout.columns))) (#scope:prop)
  preserves gpu ** source |-> Frac source_fraction sw ** hidden |-> Frac hidden_fraction sx
  requires pure (scope ==> SM.matrix_rep g sw wr /\ SM.input_rep g sx xr)
  returns result:FP.t
  ensures pure (scope ==> Approx.v_approximates result
    (S.sum (Math.chunk_work (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk))
      (Math.chunk_terms (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v which)
        (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row)))))
{
  Math.slots_bounds (SZ.v g.Layout.columns) (SZ.v groups * SZ.v chunk);
  let span = SZ.mul groups chunk;
  let mut start = SZ.mul which chunk;
  let mut acc:FP.t = F.zero #FP.t;
  while (!start <^ g.Layout.columns)
    invariant exists* (first:sz{SZ.v first < SZ.v g.Layout.columns + SZ.v span /\
      SZ.v first % SZ.v span = SZ.v which * SZ.v chunk /\
      SZ.v first / SZ.v span <= Math.slots (SZ.v g.Layout.columns) (SZ.v span)}) (current:FP.t).
      start |-> first ** acc |-> current **
      pure (scope ==> Approx.v_approximates current
        (Math.bounded_sum (SZ.v first / SZ.v span * SZ.v chunk)
          (Math.chunk_terms (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v which)
            (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row)))))
    invariant gpu ** source |-> Frac source_fraction sw ** hidden |-> Frac hidden_fraction sx
    decreases (SZ.v g.Layout.columns + SZ.v span - SZ.v !start)
  {
    let first = !start;
    accumulate_chunk g source hidden row groups chunk which first acc
      #source_fraction #hidden_fraction #sw #sx #wr #xr #scope;
    let next = SZ.add first span;
    assert pure (SZ.v next / SZ.v span == SZ.v first / SZ.v span + 1);
    start := next;
  };
  let first = !start;
  SM.group_finished (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk) (SZ.v which)
    (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v row))
    (SZ.v first / SZ.v span);
  !acc
}

inline_for_extraction noextract
fn reduce
  (partial:larray FP.t 32) (groups:sz{SZ.v groups = 16 \/ SZ.v groups = 32})
  (linear:bool)
  (#before:erased (lseq FP.t 32)) (#logical:erased (S.vector 256)) (#scope:prop)
  preserves gpu
  requires partial |-> before
  requires pure (scope ==> SM.partial_rep (SZ.v groups) before logical)
  returns result:FP.t
  ensures exists* (after:lseq FP.t 32).
    partial |-> after ** pure (scope ==> Approx.v_approximates result (Math.partial_sum (SZ.v groups) logical))
{
  if linear {
    let result = Reduction.linear partial groups;
    SM.conditional_reduction scope before logical (SZ.v groups) true result;
    result
  } else {
    let result = Reduction.tree partial groups;
    SM.conditional_reduction scope before logical (SZ.v groups) false result;
    result
  }
}

inline_for_extraction noextract
fn row
  (#length:pos{length <= 12582912}) (g:SM.geometry length)
  (source:larray BF.t length) (hidden:larray BF.t 16000) (partial:larray FP.t 32)
  (which:sz{SZ.v which < SZ.v g.Layout.rows})
  (groups:sz{SZ.v groups = 16 \/ SZ.v groups = 32})
  (chunk:sz{0 < SZ.v chunk /\ SZ.v chunk <= 64}) (linear:bool)
  (#source_fraction #hidden_fraction:perm)
  (#sw:erased (lseq BF.t length)) (#sx:erased (lseq BF.t 16000)) (#sp:erased (lseq FP.t 32))
  (#wr:erased (S.matrix (SZ.v g.Layout.rows) (SZ.v g.Layout.columns)))
  (#xr:erased (S.vector (SZ.v g.Layout.columns))) (#scope:prop)
  preserves gpu ** source |-> Frac source_fraction sw ** hidden |-> Frac hidden_fraction sx
  requires partial |-> sp
  requires pure (scope ==> SM.matrix_rep g sw wr /\ SM.input_rep g sx xr)
  returns result:BF.t
  ensures exists* (after:lseq FP.t 32). partial |-> after **
    pure (scope ==> Approx.v_approximates result
      (S.linear (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which)))
{
  let mut i = 0sz;
  while (!i <^ groups)
    invariant exists* (j:sz{SZ.v j <= SZ.v groups}) (current:lseq FP.t 32).
      i |-> j ** partial |-> current **
      pure (scope ==> SM.partial_rep (SZ.v j) current
        (Math.chunk_group_sums (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk)
          (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which))))
    invariant gpu ** source |-> Frac source_fraction sw ** hidden |-> Frac hidden_fraction sx
    decreases (SZ.v groups - SZ.v !i)
  {
    let j = !i;
    let value = group g source hidden which groups chunk j
      #source_fraction #hidden_fraction #sw #sx #wr #xr #scope;
    partial.(j) <- value;
    i := SZ.add j 1sz;
  };
  let result = reduce partial groups linear #_
    #(Math.chunk_group_sums (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk)
      (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which))) #scope;
  Math.chunk_groups_sum (SZ.v g.Layout.columns) (SZ.v groups) (SZ.v chunk)
    (Math.linear_terms (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which));
  SM.conditional_round scope result
    (S.linear (SZ.v g.Layout.rows) (SZ.v g.Layout.columns) wr xr (SZ.v which));
  Cast.cast_f32_to_bf16 result
}
