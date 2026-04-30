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

let tid_to_cell (m n : nat) (tid : natlt (m * n)) : (natlt m & natlt n) =
  (tid / n, tid % n)

unfold
let kpre
  (#t : Type0) {| scalar t |}
  (m n : sz)
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  (tid : natlt (m * n))
  : slprop
  = a |-> Frac fA sa **
    Cell b (tid_to_cell m n tid) |-> macc sb (tid / n) (tid % n)

unfold
let kpost
  (#t : Type0) {| scalar t |}
  (m n : sz)
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  (tid : natlt (m * n))
  : slprop
  = a |-> Frac fA sa **
    Cell b (tid_to_cell m n tid) |-> macc (s_row_scale sa sb) (tid / n) (tid % n)

inline_for_extraction noextract
fn kf
  (#t : Type0) {| scalar t |}
  (m n : sz)
  (#la : Array1.layout m) {| ctlayout la |}
  (a : Array1.t t la)
  (#lb : Array2.layout m n) {| ctlayout lb |}
  (b : Array2.t t lb)
  (#fA : perm)
  (#sa : erased (lseq t m))
  (#sb : ematrix t m n)
  (tid : szlt (m * n))
  ()
  requires
    gpu **
    kpre #t m n #la a #lb b #fA #sa #sb tid
  ensures
    gpu **
    kpost #t m n #la a #lb b #fA #sa #sb tid
{
  let row : sz = tid /^ n; assert rewrites_to row (tid /^ n);
  let col : sz = tid %^ n; assert rewrites_to col (tid %^ n);
  let x = Array2.read_cell b (row, col);
  let x = Array2.read_cell b (row, col);
  let v = Array1.read a row;
  Array2.write_cell b (row, col) (v `mul` x);
}

inline_for_extraction noextract
let kdesc
  (#t : Type0) {| scalar t |}
  (m n : sz)
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
  : kernel_desc (requires a |-> Frac fA sa ** b |-> sb)
                (ensures  a |-> Frac fA sa ** b |-> s_row_scale sa sb)
  = {
    nthr = m *^ n;
    f = kf m n a b #fA #sa #sb;
    frame = emp;
    teardown = magic();
    setup    = magic();
    kpre  = kpre #t m n #la a #lb b #fA #sa #sb;
    kpost = kpost #t m n #la a #lb b #fA #sa #sb;
    kpre_sendable = solve;
    kpost_sendable = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn row_scale
  (t:Type0) {| scalar t |}
  (m n : sz)
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
{
  launch_sync (kdesc m n a b);
}
