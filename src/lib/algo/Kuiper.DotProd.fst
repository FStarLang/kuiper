module Kuiper.DotProd

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Slice
module SZ = Kuiper.SizeT
open Kuiper.Sum { sum, sum_pop_right }
open Kuiper.Chest { chest_slice, chest_map, equal }

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

let chest_map_slice_commute
  (#et1 #et2 : Type0) (#r : nat) (#d : shape r)
  (f : et1 -> et2)
  (i : natlt r) (j : natlt (d @! i))
  (s : chest d et1)
  : Lemma (chest_slice i j (chest_map f s) == chest_map f (chest_slice i j s))
          [SMTPat (chest_slice i j (chest_map f s))]
  = assert (chest_slice i j (chest_map f s) `equal` chest_map f (chest_slice i j s))

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

inline_for_extraction noextract
fn gdotprod
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#len : sz)
  (#lA #lB : layout1 len)
  {| ctlayout lA, ctlayout lB |}
  (a : array1 ta lA)
  (b : array1 tb lB)
  (#sA : chest1 ta len) (#sB : chest1 tb len)
  (#fA #fB : perm)
  preserves
    gpu **
    a |-> Frac fA sA **
    b |-> Frac fB sB
  returns
    res : tacc
  ensures
    pure (res == chest1_dotprod (chest_map mapA sA) (chest_map mapB sB))
{
  let mut k : szle len = 0sz;
  let mut sum : tacc = zero;

  while (!k <^ len)
    invariant live k
    invariant sum |-> chest1_dotprod' (chest_map mapA sA) (chest_map mapB sB) !k
    decreases (len - !k)
  {
    let vk = !k;
    sum := !sum `add` mul (mapA (tensor_read a (((vk <: szlt len), ())))) (mapB (tensor_read b (((vk <: szlt len), ()))));
    k   := !k +^ 1sz;
  };
  !sum
}

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
  let r = gdotprod id id a b;
  assert pure (equal (chest_map id sA) sA);
  assert pure (equal (chest_map id sB) sB);
  r
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
fn gmatmul_dotprod
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #n #k : sz)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  {| ctlayout lA, ctlayout lB |}
  (gA : array2 ta lA)
  (gB : array2 tb lB)
  (i : szlt m)
  (j : szlt n)
  (#eA : chest2 ta _ _) (#eB : chest2 tb _ _)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : tacc
  ensures
    pure (res == MS.matmul_single (chest_map mapA eA) (chest_map mapB eB) i j)
{
  tensor_extract_slice_ro gA 0 i;
  tensor_extract_slice_ro gB 1 j;

  let s = gdotprod mapA mapB (sliceof gA 0 (SZ.v i)) (sliceof gB 1 (SZ.v j));

  tensor_restore_slice gA _ _;
  tensor_restore_slice gB _ _;

  chest_map_slice_commute mapA 0 (SZ.v i) eA;
  chest_map_slice_commute mapB 1 (SZ.v j) eB;

  s;
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
  let r = gmatmul_dotprod id id gA gB i j;
  assert pure (equal (chest_map id eA) eA);
  assert pure (equal (chest_map id eB) eB);
  r
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
