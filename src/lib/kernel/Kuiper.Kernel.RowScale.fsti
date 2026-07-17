module Kuiper.Kernel.RowScale

 (* Computes B[i*N + j] := A[i] * B[i*N + j] in-place, i.e. C = diag(A) @ B. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor

(* Spec too strong? *)
let s_row_scale
  (#t:Type0) {| scalar t |}
  (#m #n : nat)
  (a : chest1 t m) (b : chest2 t m n)
  : chest2 t m n
  = mk2 fun i j -> acc1 a i `mul` acc2 b i j

inline_for_extraction noextract
fn row_scale
  (#t:Type0) {| scalar t |}
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 t la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#fA : perm)
  (#sa : chest1 t m)
  (#sb : chest2 t m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sa)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_row_scale sa sb)
