module Kuiper.Kernel.GEMM.Copy.Vec2

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.EMatrix

open Kuiper.Array2 { array2 }
open Kuiper.Array2.Strided
module M = Kuiper.Array2
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

let in_chunk
  (chunk : pos)
  (rows cols : nat)
  (nthr : nat)
  (tid : natlt nthr)
  (ij : (natlt rows & natlt cols))
  : prop
=
  let flat_idx = ij._1 * cols + ij._2 <: nat in
  let chunk_idx = flat_idx / chunk in
  chunk_idx % nthr == tid

val in_chunk_covers_all
  (chunk : pos)
  (rows cols : nat)
  (nthr : pos)
  (ij : (natlt rows & natlt cols))
  : Lemma (exists tid. in_chunk chunk rows cols nthr tid ij)

val in_chunk_no_overlap
  (chunk : pos)
  (rows cols : nat)
  (nthr : pos)
  (ij : (natlt rows & natlt cols))
  (tid1 tid2 : natlt nthr)
  : Lemma (requires in_chunk chunk rows cols nthr tid1 ij /\
                    in_chunk chunk rows cols nthr tid2 ij)
          (ensures tid1 == tid2)

let own_strided_chunks
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  ([@@@mkey] m : array2 et l)
  (em : ematrix et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
=
  forall+ (ij : (natlt rows & natlt cols){in_chunk (chunk et #_ #hvc) rows cols nthr tid ij}).
    M.pts_to_cell m (ij._1, ij._2) (macc em ij._1 ij._2)

let live_strided_chunks
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  ([@@@mkey] m : array2 et l)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
=
  exists* em.
    own_strided_chunks m em nthr tid

ghost
fn split_array2_into_strided_chunks
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (m : array2 et l)
  (#em : ematrix et rows cols)
  (nthr : pos)
  requires
    m |-> em
  ensures
    pure (SZ.fits (M.layout_size l))
  ensures
    forall+ (tid : natlt nthr).
      own_strided_chunks m em nthr tid

ghost
fn join_array2_from_strided_chunks
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (m : array2 et l)
  (#em : ematrix et rows cols)
  (nthr : pos)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+ (tid : natlt nthr).
      own_strided_chunks m em nthr tid
  ensures
    m |-> em

ghost
fn join_array2_from_strided_chunks_underspec
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (m : array2 et l)
  (nthr : pos)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+ (tid : natlt nthr).
      live_strided_chunks m nthr tid
  ensures
    live m

inline_for_extraction noextract
fn cp_array2_vec
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (rows cols: sz)
  (#lsrc #ldst : M.layout rows cols)
  {| T.ctlayout lsrc, T.ctlayout ldst |}
  {| src_str : strided_row_major lsrc |}
  (src : array2 et lsrc)
  (#f : perm)
  (#esrc : ematrix et rows cols)
  (dst : array2 et ldst)
  (#edst : ematrix et rows cols)
  (nthr : szp)
  (tid : szlt nthr)
  preserves gpu
  preserves
    src |-> Frac f esrc
  requires
    pure (SZ.fits (rows * cols + nthr - 1)) **
    pure (chunk et /?+ cols) **
    pure (chunk et * nthr /?+ (rows * cols)) **
    pure (aligned 16 (M.core src)) **
    pure (rows * cols > 0) **
    pure (aligned_strided_row_major (chunk et) src_str)
  requires
    own_strided_chunks dst edst nthr tid
  ensures
    own_strided_chunks dst esrc nthr tid
