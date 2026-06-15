module Kuiper.DotProd

(* Matmul dot product implemented by extracting a row and column
   as Array1's, then computing a dot product between them. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM
module Array1 = Kuiper.Array1
open Kuiper.Sum { sum }
open Kuiper.Tensor
open Kuiper.Index { ( @| ), INil }
open Kuiper.Chest { chest, chest_slice }
open Kuiper.Container

(* A simple dot product spec over sequences.
   For reals, equivalent to Kuiper.Sum.sum (see seq_dotprod_is_sum). *)
let rec seq_dotprod' (#et : Type0) {| scalar et |}
  (a b : lseq et 'n) (k : nat{k <= 'n})
  : GTot et (decreases k)
  = if k = 0 then zero
    else add (seq_dotprod' a b (k-1)) (mul (a @! k-1) (b @! k-1))

let seq_dotprod (#et : Type0) {| scalar et |}
  (a b : lseq et 'n)
  = seq_dotprod' a b 'n

(* Lemma: for reals, seq_dotprod equals Kuiper.Sum.sum *)
val seq_dotprod_is_sum
  (#n : nat)
  (a b : lseq real n)
  (k : nat{k <= n})
  : Lemma (ensures
            seq_dotprod'  a b k
            ==
            sum 0 k (fun (i : natlt n) -> (a @! i) *. (b @! i)))

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

(* A generic dot product between two Array1.t of the same length. *)
inline_for_extraction noextract
fn dotprod
  (#et : Type0) {| scalar et |}
  (#len : sz)
  (#lA #lB : Array1.layout len)
  {| ctlayout lA, ctlayout lB |}
  (a : Array1.t et lA)
  (b : Array1.t et lB)
  (#sA #sB : erased (lseq et len))
  (#fA #fB : perm)
  preserves
    gpu **
    a |-> Frac fA sA **
    b |-> Frac fB sB
  returns
    res : et
  ensures
    pure (res == seq_dotprod sA sB)

(* A generic dot product between two Array1.t of the same length. *)
inline_for_extraction noextract
fn kahan_dotprod
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#len : sz)
  (#lA #lB : Array1.layout len)
  {| ctlayout lA, ctlayout lB |}
  (a : Array1.t et lA)
  (b : Array1.t et lB)
  (#sA #sB : erased (lseq et len))
  (rA rB : erased (lseq real len))
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
    pure (res %~ seq_dotprod rA rB)

(* Specialized to compute a cell of a matmul by extracting the appropriate row
and column as Array1's, then calling dotprod above. *)
inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#m #n #k : sz)
  (#lA : Array2.layout m k)
  (#lB : Array2.layout k n)
  {| ctlayout lA, ctlayout lB |}
  (gA : Array2.t et lA)
  (gB : Array2.t et lB)
  (i : szlt m)
  (j : szlt n)
  (#eA #eB : ematrix et _ _)
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
  (#lA : Array2.layout m k)
  (#lB : Array2.layout k n)
  {| ctlayout lA, ctlayout lB |}
  (gA : Array2.t et lA)
  (gB : Array2.t et lB)
  (i : szlt m)
  (j : szlt n)
  (#eA #eB : ematrix et _ _)
  (rA rB : ematrix real _ _)
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


(***** Below, similar but for tensors and chest *)

let rec edotprod' (#et : Type0) {| scalar et |}
  (a b : chest1 et 'n) (k : nat{k <= 'n})
  : GTot et
  = if k = 0 then zero
    else add (edotprod' a b (k-1)) (mul (Chest.acc a (k-1, ())) (Chest.acc b (k-1, ())))

let edotprod (#et : Type0) {| scalar et |}
  (a b : chest1 et 'n)
  : GTot et
  = edotprod' a b 'n

val edotprod_is_matmul_single
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (k : nat{k <= shared})
  : Lemma (ensures
            edotprod' #shared #_ #_ (chest_slice 0 i eA) (chest_slice 1 j eB) k
            ==
            MS.__matmul_single eA eB i j k)
          [SMTPat (edotprod' #shared (chest_slice 0 i eA) (chest_slice 1 j eB) k)]

inline_for_extraction noextract
fn dotprod_t
  (#et : Type0) {| scalar et |}
  (#len : sz)
  (#lA #lB : layout1 len)
  {| ctlayout lA, ctlayout lB |}
  (a : tensor et lA)
  (b : tensor et lB)
  (#sA #sB : erased (chest1 et len))
  (#fA #fB : perm)
  preserves
    gpu **
    a |-> Frac fA sA **
    b |-> Frac fB sB
  returns
    res : et
  ensures
    pure (res == edotprod sA sB)

(* As matmul dotprod but for tensors *)
inline_for_extraction noextract
fn matmul_dotprod_t
  (#et : Type0) {| scalar et |}
  (#m #n #k : sz)
  (#lA : tlayout (m @| k @| INil))
  (#lB : tlayout (k @| n @| INil))
  {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (i : szlt m)
  (j : szlt n)
  (#eA #eB : chest _ et)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)

(* As kahan_dotprod but for tensors. *)
inline_for_extraction noextract
fn kahan_dotprod_t
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#len : sz)
  (#lA #lB : layout1 len)
  {| ctlayout lA, ctlayout lB |}
  (a : tensor et lA)
  (b : tensor et lB)
  (#sA #sB : erased (chest1 et len))
  (rA rB : erased (chest1 real len))
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
    pure (res %~ edotprod rA rB)

(* As matmul_kahan_dotprod but for tensors. *)
inline_for_extraction noextract
fn matmul_kahan_dotprod_t
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#m #n #k : sz)
  (#lA : tlayout (m @| k @| INil))
  (#lB : tlayout (k @| n @| INil))
  {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (i : szlt m)
  (j : szlt n)
  (#eA #eB : chest _ et)
  (rA rB : chest _ real)
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
