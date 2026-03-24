module Kuiper.Example.DotProdSlice

(* Matmul dot product implemented by extracting a row and column
   as farrays, then computing a dot product between them. *)

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
open Kuiper.FArray
open Kuiper.Matrix.Slice
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

(* A simple dot product spec over sequences. *)
let rec seq_dotprod (#et : Type0) {| scalar et |}
  (a b : lseq et 'n) (k : nat{k <= 'n})
  : GTot et (decreases k)
  = if k = 0 then zero
    else add (seq_dotprod a b (k-1)) (mul (Seq.index a (k-1)) (Seq.index b (k-1)))

(* A generic dot product between two farrays of the same length. *)
inline_for_extraction noextract
fn farray_dotprod
  (#et : Type0) {| scalar et |}
  (#len : SZ.t)
  (#lA #lB : flayout len)
  {| cflayout lA, cflayout lB |}
  (a : farray et lA)
  (b : farray et lB)
  (#sA #sB : erased (lseq et len))
  (#fA #fB : perm)
  preserves
    gpu **
    a |-> Frac fA sA **
    b |-> Frac fB sB
  returns res : et
  ensures
    pure (res == seq_dotprod sA sB len)
{
  let mut k : sz = 0sz;
  let mut sum : et = zero;

  while (!k <^ len)
    invariant
      exists* (vk : SZ.t{vk <= len}).
        k |-> vk **
        sum |-> seq_dotprod sA sB vk
  {
    sum := !sum `add` mul (farray_read a !k) (farray_read b !k);
    k := SZ.add !k 1sz;
    ();
  };
  !sum
}

(* Lemma: seq_dotprod over ematrix_row/ematrix_col equals matmul_single *)
#push-options "--fuel 4 --ifuel 2 --z3rlimit 20"
let rec seq_dotprod_is_matmul_single
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (k : nat{k <= shared})
  : Lemma (ensures
      seq_dotprod (ematrix_row eA i) (ematrix_col eB j) k
      ==
      MS.__matmul_single eA eB i j k)
    (decreases k)
  = if k > 0 then begin
      seq_dotprod_is_matmul_single eA eB i j (k-1);
      assert (Seq.index (ematrix_row eA i) (k-1) == macc eA i (k-1));
      assert (Seq.index (ematrix_col eB j) (k-1) == macc eB (k-1) j);
      MS.matmul_single_lemma eA eB i j k
    end
#pop-options

(* Matmul dot product: extract row i and column j, then dot product them. *)
inline_for_extraction noextract
fn matmul_dotprod_via_slice
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  gpu_matrix_extract_row gA (SZ.v i);
  gpu_matrix_extract_col gB (SZ.v j);

  unfold factored _ (M.gpu_matrix_pts_to gA #fA eA);
  unfold factored _ (M.gpu_matrix_pts_to gB #fB eB);

  let res = farray_dotprod (row_farray gA (SZ.v i)) (col_farray gB (SZ.v j));

  ambig_trade_elim ();
  ambig_trade_elim ();

  seq_dotprod_is_matmul_single eA eB (SZ.v i) (SZ.v j) shared;
  res
}

#set-options "--debug SMTFail --split_queries always"

fn matmul_dotprod_via_slice_f32
  (rows shared cols : SZ.t)
  (gA : M.gpu_matrix f32 (row_major rows shared))
  (gB : M.gpu_matrix f32 (row_major shared cols))
  (eA : ematrix f32 rows shared)
  (eB : ematrix f32 shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : f32
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  matmul_dotprod_via_slice gA gB i j;
}
