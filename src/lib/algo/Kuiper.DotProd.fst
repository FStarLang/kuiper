module Kuiper.DotProd

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT

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
          [SMTPat (seq_dotprod (ematrix_row eA i) (ematrix_col eB j) k)]
  = if k > 0 then begin
      seq_dotprod_is_matmul_single eA eB i j (k-1);
      assert (Seq.index (ematrix_row eA i) (k-1) == macc eA i (k-1));
      assert (Seq.index (ematrix_col eB j) (k-1) == macc eB (k-1) j);
      MS.matmul_single_lemma eA eB i j k
    end
#pop-options

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
    pure (res == seq_dotprod sA sB len)
{
  let mut k : szle len = 0sz;
  let mut sum : et = zero;

  while (!k <^ len)
    invariant live k
    invariant sum |-> seq_dotprod sA sB !k
    decreases (len - !k)
  {
    sum := !sum `add` mul (Array1.(a.(!k))) (Array1.(b.(!k)));
    k   := !k +^ 1sz;
  };
  !sum
}

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
{
  Array2.extract_row_ro gA i;
  Array2.extract_col_ro gB j;

  let s = dotprod #_ #_ #_ #_ #_
           #(Kuiper.Tensor.ctlayout_slice _ 0sz i) // should not be needed
           #(Kuiper.Tensor.ctlayout_slice _ 1sz j) // should not be needed
           (Array2.row gA (SZ.v i)) (Array2.col gB (SZ.v j));

  Array2.restore_row gA i;
  Array2.restore_col gB j;

  s;
}
