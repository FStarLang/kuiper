module Kuiper.Poly.GEMM.FlipFlopBarrier

(* This module defines a barrier contract used by several GEMMs. *)

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.EMatrix
open Kuiper.Poly.GEMM.Copy.Vec { own_strided_chunks, live_strided_chunks }
open Kuiper.Math { even, odd }
open Kuiper.Matrix.Tiling

module B = Kuiper.Barrier
module SZ = Kuiper.SizeT

let bp_sharing
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (m : gpu_matrix et l)
  (em : ematrix et rows cols)
  (nthr : pos)
  : slprop
  = m |-> Frac (1.0R /. nthr) em

let barrier_p
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.barrier_side nthr =
  fun it tid ->
    (* Barrier contract must be infinite, currently, but we will
       stop after this amount of steps. *)
    if it >= 2 * shared / bk then
      emp
    else if even it then
      (* On even iterations, we give back shared access over the matrix,
         pointing to any value, as we don't care about the content which
         will be overwritten. This is in fact important for the first
         iteration of the loop which starts from uninitialized shared memory. *)
      (exists* em1. bp_sharing m1 em1 nthr) **
      (exists* em2. bp_sharing m2 em2 nthr)
    else
      (* After populating a bit of this matrix, we will give back
         exclusive access to the properly filled strided chunks. *)
      let mrow = bid / (cols/bn) in
      let mcol = bid % (cols/bn) in
      own_strided_chunks m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr tid **
      own_strided_chunks m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr tid

let barrier_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.barrier_side nthr =
  fun it tid ->
    (* Barrier contract must be infinite, currently, but we will
       stop after this amount of steps. *)
    if it >= 2 * shared / bk then
      emp
    else if even it then
      (* We get back exclusive, strided acess to the matrix. Over unspecified
         contents. *)
      live_strided_chunks m1 nthr tid **
      live_strided_chunks m2 nthr tid
    else
      (* We get back shared, read-only access to the matrix. Over the
         *proper* contents. *)
      let mrow = bid / (cols/bn) in
      let mcol = bid % (cols/bn) in
      bp_sharing m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr **
      bp_sharing m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr

let contract 
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.contract nthr =
{
  B.rin  = barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid;
  B.rout = barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid;
}

let barrier_tok
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (* This is defined over the base shared gpu_arrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : slprop
  = B.barrier_tok (contract eA eB l1 l2 sar1 sar2 nthr bid)

(* The proof of correctness. *)
ghost
fn barrier_p_to_q_transform
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (mlayout_size l1)))
  (#_ : squash (SZ.fits (mlayout_size l2)))
  (it : nat)
  requires
    forall+ (tid : natlt nthr).
      barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
  ensures
    forall+ (tid : natlt nthr).
      barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
