module Kuiper.Kernel.RowBroadcast

(* Generic row-broadcast 2D map.
   Computes [B[i, j] := f (B[i, j]) (A[i])] in-place, where [A] is an
   [Array1] of length m and [B] is an [Array2] of shape m × n. *)

#lang-pulse

open Kuiper
module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Shape

let s_row_broadcast
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (#m #n : nat)
  (a : lseq t m) (b : ematrix t m n)
  : ematrix t m n
  = Kuiper.EMatrix.mkM fun i j -> f (macc b i j) (Seq.index a i)

inline_for_extraction noextract
fn row_broadcast
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array2.is_global b))
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sa)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_row_broadcast f sa sb)
