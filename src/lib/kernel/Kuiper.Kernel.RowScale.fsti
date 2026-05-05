module Kuiper.Kernel.RowScale

 (* Computes B[i*N + j] := A[i] * B[i*N + j] in-place, i.e. C = diag(A) @ B. *)

#lang-pulse

open Kuiper
module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2
open Kuiper.Array1
open Kuiper.EMatrix
open Kuiper.Seq.Common
open Kuiper.Tensor
open Kuiper.Seq.Common { (@!) }

(* Spec too strong? *)
let s_row_scale
  (#t:Type0) {| scalar t |}
  (#m #n : nat)
  (a : lseq t m) (b : ematrix t m n)
  : ematrix t m n
  = Kuiper.EMatrix.mkM fun i j -> (a @! i) `mul` macc b i j

type row_scale_ty (t:Type0) {| scalar t |} =
  fn
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
    on gpu_loc (b |-> s_row_scale sa sb)

inline_for_extraction noextract
val row_scale (t:Type0) {| scalar t |} : row_scale_ty t
