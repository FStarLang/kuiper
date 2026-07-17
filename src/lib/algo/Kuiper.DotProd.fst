module Kuiper.DotProd

#lang-pulse

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT
open Kuiper.Sum { sum, sum_pop_right }
open Kuiper.Chest { chest_slice }

#push-options "--fuel 4 --ifuel 2 --z3rlimit 20"
let rec chest1_dotprod_is_sum
  (#n : nat)
  (a b : chest1 real n)
  (k : nat{k <= n})
  : Lemma (ensures
            chest1_dotprod' a b k
            ==
            sum 0 k (fun (i : natlt n) -> (acc1 a i) *. (acc1  b i)))
          (decreases k)
  = if k > 0 then begin
      chest1_dotprod_is_sum a b (k-1);
      sum_pop_right 0 k (fun (i : natlt n) -> (acc1 a i) *. (acc1 b i))
    end
#pop-options

let rec chest1_dotprod_is_matmul_single
  (#et : Type0) {| scalar et |}
  (#m #n #k : nat)
  (eA : chest2 et m k) (eB : chest2 et k n)
  (i : natlt m) (j : natlt n)
  (l : nat{l <= k})
  : Lemma (ensures
            chest1_dotprod' #k #_ #_ (chest_slice 0 i eA) (chest_slice 1 j eB) l
            ==
            MS.__matmul_single eA eB i j l)
          (decreases l)
          [SMTPat (chest1_dotprod' #k (chest_slice 0 i eA) (chest_slice 1 j eB) l)]
  = if l > 0 then
      chest1_dotprod_is_matmul_single eA eB i j (l-1)

#push-options "--fuel 4 --ifuel 2 --z3rlimit 20"
let rec seq_dotprod_is_sum
  (#n : nat)
  (a b : lseq real n)
  (k : nat{k <= n})
  : Lemma (ensures
            seq_dotprod' a b k
            ==
            sum 0 k (fun (i : natlt n) -> (Seq.index a i) *. (Seq.index b i)))
          (decreases k)
  = if k > 0 then begin
      seq_dotprod_is_sum a b (k-1);
      sum_pop_right 0 k (fun (i : natlt n) -> (Seq.index a i) *. (Seq.index b i))
    end

let rec seq_dotprod_is_matmul_single
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared) (eB : chest2 et shared cols)
  (i : natlt rows) (j : natlt cols)
  (k : nat{k <= shared})
  : Lemma (ensures
            seq_dotprod' (ematrix_row eA i) (ematrix_col eB j) k
            ==
            MS.__matmul_single eA eB i j k)
          (decreases k)
          [SMTPat (seq_dotprod' (ematrix_row eA i) (ematrix_col eB j) k)]
  = if k > 0 then begin
      seq_dotprod_is_matmul_single eA eB i j (k-1);
      assert (Seq.index (ematrix_row eA i) (k-1) == acc2 eA i (k-1));
      assert (Seq.index (ematrix_col eB j) (k-1) == acc2 eB (k-1) j);
      MS.matmul_single_lemma eA eB i j k
    end
#pop-options

(* A generic dot product between two Array1.t of the same length. *)
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
{
  let mut k : szle len = 0sz;
  let mut sum : et = zero;

  while (!k <^ len)
    invariant live k
    invariant sum |-> chest1_dotprod' sA sB !k
    decreases (len - !k)
  {
    let vk = !k;
    sum := !sum `add` mul (tensor_read a (((vk <: szlt len), ()))) (tensor_read b (((vk <: szlt len), ())));
    k   := !k +^ 1sz;
  };
  !sum
}

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
{
  let res =
    Kuiper.Kahan.kahan_sum #et
      len
      (gpu ** a |-> Frac fA sA ** b |-> Frac fB sB)
      (fun (i : natlt len) -> (rA `acc1` i) *. (rB `acc1` i))
      fn (i : szlt len) {
        a.(i, ()) `mul` b.(i, ());
      };
  chest1_dotprod_is_sum rA rB len;
  res
}

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
{
  tensor_extract_slice_ro gA 0 i;
  tensor_extract_slice_ro gB 1 j;

  let s = dotprod (sliceof gA 0 (SZ.v i)) (sliceof gB 1 (SZ.v j));

  tensor_restore_slice gA _ _;
  tensor_restore_slice gB _ _;

  s;
}

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
{
  tensor_extract_slice_ro gA 0 i;
  tensor_extract_slice_ro gB 1 j;

  let s = kahan_dotprod (sliceof gA 0 (SZ.v i)) (sliceof gB 1 (SZ.v j))
           (chest_slice 0 i rA) (chest_slice 1 j rB);

  tensor_restore_slice gA 0 i;
  tensor_restore_slice gB 1 j;

  s;
}
