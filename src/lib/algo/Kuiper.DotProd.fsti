module Kuiper.DotProd

(* Matmul dot product implemented by extracting a row and column
   as Array1's, then computing a dot product between them. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM
open Kuiper.Sum { sum }
open Kuiper.Tensor
open Kuiper.Shape { ( @| ), INil }
open Kuiper.Chest { chest, chest_slice }
open Kuiper.Container

(* A simple dot product spec over sequences.
   For reals, equivalent to Kuiper.Sum.sum (see seq_dotprod_is_sum). *)
let rec seq_dotprod' (#et : Type0) {| scalar et |}
  (a b : lseq et 'n) (k : nat{k <= 'n})
  : GTot et (decreases k)
  = if k = 0 then zero
    else add (seq_dotprod' a b (k-1)) (mul (a `Seq.index` (k-1)) (b `Seq.index` (k-1)))

let seq_dotprod (#et : Type0) {| scalar et |}
  (a b : lseq et 'n)
  = seq_dotprod' a b 'n

let rec chest1_dotprod' (#et : Type0) {| scalar et |}
  (a b : chest1 et 'n) (k : nat{k <= 'n})
  : GTot et (decreases k)
  = if k = 0 then zero
    else add (chest1_dotprod' a b (k-1)) (mul (acc1 a (k-1)) (acc1 b (k-1)))

let chest1_dotprod (#et : Type0) {| scalar et |}
  (a b : chest1 et 'n)
  = chest1_dotprod' a b 'n

val chest1_dotprod_is_sum
  (#n : nat)
  (a b : chest1 real n)
  (k : nat{k <= n})
  : Lemma (ensures
            chest1_dotprod'  a b k
            ==
            sum 0 k (fun (i : natlt n) -> (acc1 a i) *. (acc1  b i)))

val chest1_dotprod_is_matmul_single
  (#et : Type0) {| scalar et |}
  (#m #n #k : nat)
  (eA : chest2 et m k) (eB : chest2 et k n)
  (i : natlt m) (j : natlt n)
  (l : nat{l <= k})
  : Lemma (ensures
            chest1_dotprod' #k (chest_slice 0 i eA) (chest_slice 1 j eB) l
            ==
            MS.__matmul_single eA eB i j l)
          [SMTPat (chest1_dotprod' #k (chest_slice 0 i eA) (chest_slice 1 j eB) l)]

(* Lemma: for reals, seq_dotprod equals Kuiper.Sum.sum *)
val seq_dotprod_is_sum
  (#n : nat)
  (a b : lseq real n)
  (k : nat{k <= n})
  : Lemma (ensures
            seq_dotprod'  a b k
            ==
            sum 0 k (fun (i : natlt n) -> (Seq.index a i) *. (Seq.index b i)))

(* Lemma: seq_dotprod over ematrix_row/ematrix_col equals matmul_single *)
val seq_dotprod_is_matmul_single
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (k : nat{k <= shared})
  : Lemma (ensures
            seq_dotprod' (ematrix_row eA i) (ematrix_col eB j) k
            ==
            MS.__matmul_single eA eB i j k)
          [SMTPat (seq_dotprod' (ematrix_row eA i) (ematrix_col eB j) k)]

(* A generic dot product between two array1 of the same length. *)
inline_for_extraction noextract
fn dotprod
  (#et : Type0) {| scalar et |}
  (#len : sz)
  (#lA #lB : layout1 len)
  {| ctlayout lA, ctlayout lB |}
  (a : array1 et lA)
  (b : array1 et lB)
  (#sA #sB : chest1 et len)
  (#fA #fB : perm)
  preserves
    gpu **
    a |-> Frac fA sA **
    b |-> Frac fB sB
  returns
    res : et
  ensures
    pure (res == chest1_dotprod sA sB)

(* A generic dot product between two array1 of the same length. *)
inline_for_extraction noextract
fn kahan_dotprod
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#len : sz)
  (#lA #lB : layout1 len)
  {| ctlayout lA, ctlayout lB |}
  (a : array1 et lA)
  (b : array1 et lB)
  (#sA #sB : chest1 et len)
  (rA rB : chest1 real len)
  (#fA #fB : perm)
  preserves
    gpu **
    a |-> Frac fA sA **
    b |-> Frac fB sB
  requires
    pure (sA %~ rA /\ sB %~ rB)
  returns
    res : et
  ensures
    pure (res %~ chest1_dotprod rA rB)

(* Specialized to compute a cell of a matmul by extracting the appropriate row
and column as Array1's, then calling dotprod above. *)
inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#m #n #k : sz)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  {| ctlayout lA, ctlayout lB |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (i : szlt m)
  (j : szlt n)
  (#eA #eB : chest2 et _ _)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)

(* As above, but using Kahan summation. *)
inline_for_extraction noextract
fn matmul_kahan_dotprod
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#m #n #k : sz)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  {| ctlayout lA, ctlayout lB |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (i : szlt m)
  (j : szlt n)
  (#eA #eB : chest2 et _ _)
  (rA rB : chest2 real _ _)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (eA %~ rA /\ eB %~ rB)
  returns
    res : et
  ensures
    pure (res %~ MS.matmul_single rA rB i j)
