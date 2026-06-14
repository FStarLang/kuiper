module Kuiper.DotProd

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
open Kuiper.Sum { sum, sum_pop_right }
open Kuiper.Chest { chest_slice }

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
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
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
    pure (res == seq_dotprod sA sB)
{
  let mut k : szle len = 0sz;
  let mut sum : et = zero;

  while (!k <^ len)
    invariant live k
    invariant sum |-> seq_dotprod' sA sB !k
    decreases (len - !k)
  {
    sum := !sum `add` mul (Array1.(a.(!k))) (Array1.(b.(!k)));
    k   := !k +^ 1sz;
  };
  !sum
}

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
{
  let res =
    Kuiper.Kahan.kahan_sum #et
      len
      (gpu ** a |-> Frac fA sA ** b |-> Frac fB sB)
      (fun (i : natlt len) -> (rA @! i) *. (rB @! i))
      fn (i : szlt len) {
        open Array1;
        a.(i) `mul` b.(i);
      };
  seq_dotprod_is_sum rA rB len;
  res
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

  let s = dotprod (Array2.row gA (SZ.v i)) (Array2.col gB (SZ.v j));

  Array2.restore_row gA i;
  Array2.restore_col gB j;

  s;
}

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
{
  Array2.extract_row_ro gA i;
  Array2.extract_col_ro gB j;

  let s = kahan_dotprod (Array2.row gA (SZ.v i)) (Array2.col gB (SZ.v j))
           (ematrix_row rA i) (ematrix_col rB j);

  Array2.restore_row gA i;
  Array2.restore_col gB j;

  s;
}

let rec edotprod_is_matmul_single
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (k : nat{k <= shared})
  : Lemma (ensures
            edotprod' #shared #_ #_ (chest_slice 0 i eA) (chest_slice 1 j eB) k
            ==
            MS.__matmul_single eA eB i j k)
          (decreases k)
          [SMTPat (edotprod' #shared (chest_slice 0 i eA) (chest_slice 1 j eB) k)]
  = if k > 0 then begin
      edotprod_is_matmul_single eA eB i j (k-1);
      assert (Chest.acc (chest_slice 0 i eA) ((k-1 <: natlt shared), ()) == macc eA i (k-1));
      assert (Chest.acc (chest_slice 1 j eB) ((k-1 <: natlt shared), ()) == macc eB (k-1) j);
      MS.matmul_single_lemma eA eB i j k
    end


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
{
  let mut k : szle len = 0sz;
  let mut sum : et = zero;

  while (!k <^ len)
    invariant live k
    invariant sum |-> edotprod' sA sB !k
    decreases (len - !k)
  {
    let vk = !k;
    sum := !sum `add` mul (tensor_read a ((vk <: szlt len), ())) (tensor_read b ((vk <: szlt len), ()));
    k   := !k +^ 1sz;
  };
  !sum
}

(* As matmul dotprod but for tensors *)
inline_for_extraction noextract
fn matmul_dotprod_t
  (#et : Type0) {| scalar et |}
  (#m #n #k : sz)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
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
{
  tensor_extract_slice_ro gA 0 i;
  tensor_extract_slice_ro gB 1 j;

  let s = dotprod_t (sliceof gA 0 (SZ.v i)) (sliceof gB 1 (SZ.v j));

  tensor_restore_slice gA 0 i;
  tensor_restore_slice gB 1 j;

  s
}

(* For reals, the chest dot product equals Kuiper.Sum.sum (mirrors
   seq_dotprod_is_sum). Internal helper for the Kahan variant. *)
#push-options "--fuel 4 --ifuel 2 --z3rlimit 20"
let rec edotprod_is_sum
  (#n : nat)
  (a b : chest1 real n)
  (k : nat{k <= n})
  : Lemma (ensures
            edotprod' a b k
            ==
            sum 0 k (fun (i : natlt n) -> (Chest.acc a (i, ())) *. (Chest.acc b (i, ()))))
          (decreases k)
  = if k > 0 then begin
      edotprod_is_sum a b (k-1);
      sum_pop_right 0 k (fun (i : natlt n) -> (Chest.acc a (i, ())) *. (Chest.acc b (i, ())))
    end
#pop-options

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
{
  let res =
    Kuiper.Kahan.kahan_sum #et
      len
      (gpu ** a |-> Frac fA sA ** b |-> Frac fB sB)
      (fun (i : natlt len) -> (Chest.acc rA (i, ())) *. (Chest.acc rB (i, ())))
      fn (i : szlt len) {
        mul (tensor_read a (i, ())) (tensor_read b (i, ()));
      };
  edotprod_is_sum rA rB len;
  res
}

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
{
  tensor_extract_slice_ro gA 0 i;
  tensor_extract_slice_ro gB 1 j;

  let s = kahan_dotprod_t
           (sliceof gA 0 (SZ.v i)) (sliceof gB 1 (SZ.v j))
           (chest_slice 0 i rA) (chest_slice 1 j rB);

  tensor_restore_slice gA 0 i;
  tensor_restore_slice gB 1 j;

  s
}
