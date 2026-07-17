module Kuiper.Kernel.RowBroadcast

(* Generic row-broadcast 2D map.
   Computes [B[i, j] := f (B[i, j]) (A[i])] in-place, where [A] is an
   [Array1] of length m and [B] is an [Array2] of shape m × n. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor

let s_row_broadcast
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (#m #n : nat)
  (a : chest1 ta m) (b : chest2 tb m n)
  : chest2 tb m n
  = mk2 fun i j -> acc1 a i `f` acc2 b i j

inline_for_extraction noextract
fn row_broadcast
  (#ta #tb : Type0)
  (f : ta -> tb -> tb)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 ta la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 tb lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#fA : perm)
  (#sa : chest1 ta m)
  (#sb : chest2 tb m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sa)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_row_broadcast f sa sb)
