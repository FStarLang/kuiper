module Kuiper.Sparse.SPMM.Spec

(* Real-valued reference computations for SPMM. Vector width is a layout
   parameter; these equalities say nothing about computed floating values. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Seq.Common { (@+), seq_replace }
open Kuiper.Sparse.Common
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.FMA
module Math = Kuiper.Sparse.Math

#set-options "--z3rlimit 20"

let tile_vmprod_cell_prop
  (ch : pos)
  (#m1 #n1 : nat { ch /? n1 })
  (acc : erased (lseq real n1))
  (elems : erased (lseq real m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  ch /? n2 })
  (em2 : chest2 real m2 n2)
  (j : nat { ch /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  (k1 : natlt n1)
  (y : lseq real n1)
: prop
=
  let k2 = j + k1 / ch * step * ch + k1 % ch in
  k2 < n2 ==>
  y @! k1 == _sparse_dprod_acc (acc @! k1) elems row_ind (ematrix_col em2 k2) to


let _tile_vmprod_prop
  (ch : pos)
  (#m1 #n1 : nat { ch /? n1 })
  (acc : erased (lseq real n1))
  (elems : erased (lseq real m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  ch /? n2 })
  (em2 : chest2 real m2 n2)
  (j : nat { ch /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  (y : lseq real n1)
: prop
= forall (k1 : natlt n1). tile_vmprod_cell_prop ch acc elems row_ind em2 j step to k1 y


let tile_mm_result
  (ch : pos)
  (#shared #cols : nat {  ch /? cols })
  (#ly : nat { ch /? ly })
  (y0 : erased (lseq real ly))
  (#lA : nat)
  (elems : erased (lseq real lA))
  (row_ind : erased (lseq nat lA) { in_bounds 0 shared row_ind })
  (eB : chest2 real shared cols)
  (j : nat { ch /? j })
  (step : nat)
  (y : lseq real ly)
  : prop
= _tile_vmprod_prop ch y0 elems row_ind eB j step lA y



let tile_mm_lemma0
  (ch : pos)
  (#shared #cols : nat {  ch /? cols })
  (#ly : nat { ch /? ly })
  (y0 : erased (lseq real ly))
  (eB : chest2 real shared cols)
  (j : nat { ch /? j })
  (step : nat)
: Lemma (ensures tile_mm_result ch y0 (Seq.empty <: lseq real 0) Seq.empty eB j step y0)
= ()


let tile_mm_mask_lemma
  (ch : pos)
  (#shared #cols : nat { ch /? cols })
  (#ly : nat { ch /? ly })
  (y0 : lseq real ly)
  (nnz : nat)
  (mask_len : natle nnz)
  (elems : erased (lseq real (nnz - mask_len)))
  (row_ind : erased (lseq nat nnz) { in_bounds 0 shared row_ind })
  (eB : chest2 real shared cols)
  (j : nat { ch /? j })
  (step : nat)
  (y : lseq real ly)
: Lemma
  (requires tile_mm_result ch y0 (Seq.create mask_len zero @+ elems) row_ind eB j step y)
  (ensures  tile_mm_result ch y0 elems (Seq.slice row_ind mask_len nnz) eB j step y)
=
  let mask : lseq real mask_len = Seq.create mask_len zero in
  let elems' : lseq real nnz = mask @+ elems in
  let row_ind' : lseq nat (nnz - mask_len) = Seq.slice row_ind mask_len nnz in
  introduce forall (k1 : natlt ly).
    tile_vmprod_cell_prop ch y0 elems row_ind' eB j step #() (nnz - mask_len) k1 y
  with (
    let k2 = j + k1 / ch * step * ch + k1 % ch in
    if k2 < cols
      then calc(==) {
        y @! k1;
        == {}
        sparse_dprod_acc (y0 @! k1) elems' row_ind (ematrix_col eB k2);
        == {}
        dprod_acc (y0 @! k1) elems' (seq_make_sparse row_ind (ematrix_col eB k2));
        == {
          _dprod_acc_mask_lemma
            (y0 @! k1)
            mask_len
            elems (seq_make_sparse row_ind (ematrix_col eB k2))
            nnz
        }
        dprod_acc
          (y0 @! k1)
          elems
          (Seq.slice (seq_make_sparse row_ind (ematrix_col eB k2)) mask_len nnz);
        == { seq_make_sparse_slice row_ind mask_len nnz (ematrix_col eB k2) }
        dprod_acc
          (y0 @! k1)
          elems
          (seq_make_sparse row_ind' (ematrix_col eB k2));
        == {}
        sparse_dprod_acc (y0 @! k1) elems row_ind' (ematrix_col eB k2);
      }
      else ()
  )



let rec seq_load_vmprod_row
  (ch : pos)
  (#n1 : nat { ch /? n1 })
  (y : lseq real n1)
  (x : real)
  (#n2 : nat { ch /? n2 })
  (row : lseq real n2)
  (j : nat { ch /? j })
  (step : nat)
  (k : natle (n1 / ch))
: GTot (lseq real n1) (decreases k)
=
  if k = 0 then y
    else (
      Math.lineal_divides ch j ch ((k - 1) * step);
      seq_fma'
        ch x row
        (seq_load_vmprod_row ch y x row j step (k - 1))
        ((k - 1) * ch)
        (j + (k - 1) * step * ch)
    )



let rec seq_load_vmprod
  (ch : pos)
  (#m1 #n1 : nat { ch /? n1 })
  (y : lseq real n1)
  (elems : lseq real m1)
  (row_ind : lseq nat m1)
  (#m2 #n2 : pos { ch /? n2 })
  (em : chest2 real m2 n2)
  (j : nat { ch /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
: GTot (lseq real n1) (decreases to)
=
  if to = 0 then y
    else
      seq_load_vmprod_row ch
        (seq_load_vmprod ch y elems row_ind em j step (to - 1))
        (elems @! to - 1)
        (ematrix_row em (row_ind @! to - 1))
        j step (n1 / ch)



(* A thread starting beyond the matrix width has no active vector chunks.
   These lemmas justify skipping its reduction without changing the spec. *)

let rec seq_load_vmprod_row_outside
  (ch : pos)
  (#n1 : nat { ch /? n1 })
  (y : lseq real n1)
  (x : real)
  (#n2 : nat { ch /? n2 })
  (row : lseq real n2)
  (j : nat { ch /? j /\ n2 <= j })
  (step : nat)
  (k : natle (n1 / ch))
  : Lemma (seq_load_vmprod_row ch y x row j step k == y)
  = if k > 0 then seq_load_vmprod_row_outside ch y x row j step (k - 1)


let rec seq_load_vmprod_outside
  (ch : pos)
  (#m1 #n1 : nat { ch /? n1 })
  (y : lseq real n1)
  (elems : lseq real m1)
  (row_ind : lseq nat m1)
  (#m2 #n2 : pos { ch /? n2 })
  (em : chest2 real m2 n2)
  (j : nat { ch /? j /\ n2 <= j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  : Lemma (seq_load_vmprod ch y elems row_ind em j step to == y)
  = if to > 0 then (
      seq_load_vmprod_outside ch y elems row_ind em j step (to - 1);
      seq_load_vmprod_row_outside ch y (elems @! to - 1)
        (ematrix_row em (row_ind @! to - 1)) j step (n1 / ch)
    )




let seq_fma_cell_prop
  (x1 : real)
  (#n : nat)
  (x2 : lseq real n)
  (#sz_y : nat)
  (y0 : lseq real sz_y)
  (k : nat { k + n <= sz_y })
  (to : natle n)
  (y : lseq real sz_y)
  (ix : natlt to)
: prop
= y @! k + ix == add (y0 @! k + ix) (x1 `mul` (x2 @! ix))


let seq_fma_lemma
  (x1 : real)
  (#n : nat)
  (x2 : lseq real n)
  (#sz_y : nat)
  (y : lseq real sz_y)
  (k : nat { k + n <= sz_y })
  (to : natle n)
: Lemma
  (requires true)
  (ensures forall (ix : natlt to).
    seq_fma_cell_prop x1 x2 y k to (seq_fma x1 x2 y k to) ix)
= ()


let seq_fma_cell_prop'
  (cnt : nat)
  (x1 : real)
  (#n : nat { cnt /? n })
  (x2 : lseq real n)
  (#sz_y : nat)
  (y0 : lseq real sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
  (y : lseq real sz_y)
  (ix : natlt cnt)
: prop
=
  k2 < n ==>
    (let _ = __divides_next cnt k2 n in
     y @! k1 + ix == add (y0 @! k1 + ix) (x1 `mul` (x2 @! k2 + ix)))


let seq_fma_lemma0'
  (cnt : nat)
  (x1 : real)
  (#n : nat { cnt /? n })
  (x2 : lseq real n)
  (#sz_y : nat)
  (y0 : lseq real sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
: Lemma
  (requires true)
  (ensures forall (i : natlt sz_y { i < k1 \/ k1 + cnt <= i }).
    seq_fma' cnt x1 x2 y0 k1 k2 @! i == y0 @! i)
= ()


let seq_fma_lemma'
  (cnt : nat)
  (x1 : real)
  (#n : nat { cnt /? n })
  (x2 : lseq real n)
  (#sz_y : nat)
  (y0 : lseq real sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
: Lemma
  (requires true)
  (ensures forall (ix : natlt cnt).
    seq_fma_cell_prop' cnt x1 x2 y0 k1 k2 (seq_fma' cnt x1 x2 y0 k1 k2) ix)
=
  if k2 < n
    then (
      if cnt = 0 then () else Math.divides_leq cnt n k2;
      assert (k2 + cnt <= n);
      let y = seq_fma' cnt x1 x2 y0 k1 k2 in
      assert y == seq_fma x1 #cnt (Seq.slice x2 k2 (k2 + cnt)) y0 k1 cnt;
      introduce forall (ix : natlt cnt).
        seq_fma_cell_prop' cnt x1 x2 y0 k1 k2 y ix
      with (
        assert k1 + ix < sz_y;
        assert y @! k1 + ix == seq_fma x1 #cnt (Seq.slice x2 k2 (k2 + cnt)) y0 k1 cnt @! k1 + ix;
        ()
      )
    )
    else ()


let seq_load_vmprod_row_cell_prop_
  (ch : pos)
  (#n1 : nat { ch /? n1 })
  (y0 : lseq real n1)
  (x : real)
  (#n2 : nat { ch /? n2 })
  (row : lseq real n2)
  (j : nat { ch /? j })
  (step : nat)
  (k : natle (n1 / ch))
  (y : lseq real n1)
  (ik : natlt k)
  (ix : natlt ch)
: prop
=
  Math.lineal_divides ch j ch (ik * step);
  seq_fma_cell_prop'
    ch x row
    y0
    (ik * ch)
    (j + ik * step * ch) y ix


let seq_load_vmprod_row_cell_prop
  (ch : pos)
  (#n1 : nat { ch /? n1 })
  (y0 : lseq real n1)
  (x : real)
  (#n2 : nat { ch /? n2 })
  (row : lseq real n2)
  (j : nat { ch /? j })
  (step : nat)
  (k : natle (n1 / ch))
  (y : lseq real n1)
  (ik : natlt (k * ch))
: prop
=
  j + ik / ch * step * ch + ik % ch < n2 ==>
  y @! ik ==
  add
    (y0 @! ik)
    (x `mul` (row @! j + ik / ch * step * ch + ik % ch))


let seq_load_vmprod_row_cell_prop_equiv
  (ch : pos)
  (#n1 : nat { ch /? n1 })
  (y0 : lseq real n1)
  (x : real)
  (#n2 : nat { ch /? n2 })
  (row : lseq real n2)
  (j : nat { ch /? j })
  (step : nat)
  (y : lseq real n1)
  (k : natle (n1 / ch))
  (i : natlt (k * ch))
: Lemma
  (requires seq_load_vmprod_row_cell_prop_ ch
    y0 x row j step k y (i / ch) (i % ch))
  (ensures  seq_load_vmprod_row_cell_prop ch  y0 x row j step k y i)
= ()


let rec seq_load_vmprod_row_cell_lemma0
  (ch : pos)
  (#n1 : nat { ch /? n1 })
  (y0 : lseq real n1)
  (x : real)
  (#n2 : nat { ch /? n2 })
  (row : lseq real n2)
  (j : nat { ch /? j })
  (step : nat)
  (k : natle (n1 / ch))
  (i : natlt n1 { k * ch <= i })
: Lemma
  (requires true)
  (ensures seq_load_vmprod_row ch y0 x row j step k @! i == y0 @! i)
=
  if k = 0 then ()
  else (
    Math.lineal_divides ch j ch ((k - 1) * step);
    seq_load_vmprod_row_cell_lemma0 ch y0 x row j step (k - 1) i;
    seq_fma_lemma0' ch x row
      (seq_load_vmprod_row ch y0 x row j step (k - 1))
      ((k - 1) * ch) (j + (k - 1) * step * ch);
    ()
  )

#push-options "--z3rlimit 20"

let chunk_prefix_bound (ch : pos) (n : nat) (k : pos)
  : Lemma (requires k <= n / ch)
          (ensures (k - 1) * ch + ch <= n)
  = FStar.Math.Lemmas.lemma_mult_le_left ch k (n / ch)

let last_chunk (k : pos) (i : natlt k)
  : Lemma (requires ~(i < k - 1)) (ensures i == k - 1)
  = ()

let rec seq_load_vmprod_row_cell_lemma_
  (ch : pos)
  (#n1 : nat { ch /? n1 })
  (y0 : lseq real n1)
  (x : real)
  (#n2 : nat { ch /? n2 })
  (row : lseq real n2)
  (j : nat { ch /? j })
  (step : nat)
  (k : natle (n1 / ch))
  (ik : natlt k)
  (ix : natlt ch)
: Lemma
  (requires true)
  (ensures
    seq_load_vmprod_row_cell_prop_ ch
      y0 x row j step k
      (seq_load_vmprod_row ch y0 x row j step k)
      ik ix
  )
  (decreases k)
=
  if k = 0 then ()
  else (
    chunk_prefix_bound ch n1 k;
    Math.lineal_divides ch j ch ((k - 1) * step);
    if ik < k - 1
      then (
        seq_load_vmprod_row_cell_lemma_ ch y0 x row j step (k - 1) ik ix;
        assert ik * ch + ix < (k - 1) * ch;
        seq_fma_lemma0' ch x row
          (seq_load_vmprod_row ch y0 x row j step (k - 1))
          ((k - 1) * ch) (j + (k - 1) * step * ch);
        ()
      )
      else (
        last_chunk k ik;
        seq_fma_lemma' ch x row
          (seq_load_vmprod_row ch y0 x row j step (k - 1))
          ((k - 1) * ch) (j + (k - 1) * step * ch);
        seq_load_vmprod_row_cell_lemma0 ch
          y0 x row j step (k - 1) (ik * ch + ix);
        ()
      )
  )
#pop-options


let tile_vmprod_slice_lemma
  (ch : pos)
  (#m1 #n1 : nat { ch /? n1 })
  (acc : erased (lseq real n1))
  (elems : erased (lseq real m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  ch /? n2 })
  (em2 : chest2 real m2 n2)
  (j : nat { ch /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  (k1 : natlt n1)
  (y : lseq real n1)
: Lemma
  (requires tile_vmprod_cell_prop ch acc elems row_ind em2 j step to k1 y)
  (ensures tile_vmprod_cell_prop ch
    #to
    acc
    (Seq.slice elems 0 to)
    (Seq.slice row_ind 0 to)
    em2 j step to k1 y
  )
=
  let elems' : lseq real to = Seq.slice elems 0 to in
  let row_ind' : lseq nat to = Seq.slice row_ind 0 to in

  let k2 = j + k1 / ch * step * ch + k1 % ch in

  if k2 < n2
    then sparse_dprod_slice_lemma (acc @! k1) elems row_ind (ematrix_col em2 k2) to to
    else ()


let rec seq_load_vmprod_cell_lemma
  (ch : pos)
  (#m1 #n1 : nat { ch /? n1 })
  (y : lseq real n1)
  (elems : lseq real m1)
  (row_ind : lseq nat m1)
  (#m2 #n2 : pos { ch /? n2 })
  (em : chest2 real m2 n2)
  (j : nat { ch /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  (k1 : natlt n1)
: Lemma
  (requires true)
  (ensures
    tile_vmprod_cell_prop ch y elems row_ind em j step to k1
      (seq_load_vmprod ch y elems row_ind em j step to)
  )
=
  if to = 0 then ()
  else (
    seq_load_vmprod_cell_lemma ch y elems row_ind em j step (to - 1) k1;
    seq_load_vmprod_row_cell_lemma_ ch
      (seq_load_vmprod ch y elems row_ind em j step (to - 1))
      (elems @! to - 1)
      (ematrix_row em (row_ind @! to - 1))
      j step (n1 / ch) (k1 / ch) (k1 % ch);
    seq_load_vmprod_row_cell_prop_equiv ch
      (seq_load_vmprod ch y elems row_ind em j step (to - 1))
      (elems @! to - 1)
      (ematrix_row em (row_ind @! to - 1))
      j step
      (seq_load_vmprod_row ch
        (seq_load_vmprod ch y elems row_ind em j step (to - 1))
        (elems @! to - 1)
        (ematrix_row em (row_ind @! to - 1))
        j step (n1 / ch))
      (n1 / ch) k1;
    ()
  )


let seq_load_vmprod_lemma
  (ch : pos)
  (#m1 #n1 : nat { ch /? n1 })
  (y : lseq real n1)
  (elems : lseq real m1)
  (row_ind : lseq nat m1)
  (#m2 #n2 : pos { ch /? n2 })
  (em : chest2 real m2 n2)
  (j : nat { ch /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
: Lemma
  (requires true)
  (ensures
    tile_mm_result ch y
      (Seq.slice elems 0 to <: lseq real to)
      (Seq.slice row_ind 0 to)
      em j step
      (seq_load_vmprod ch y elems row_ind em j step to)
  )
=
  let elems' : lseq real to = Seq.slice elems 0 to in
  let row_ind' : lseq nat to = Seq.slice row_ind 0 to in
  introduce forall (k1 : natlt n1).
    tile_vmprod_cell_prop ch y
      elems'
      row_ind'
      em j step
      #() to k1
      (seq_load_vmprod ch y elems row_ind em j step #() to)
  with (
    seq_load_vmprod_cell_lemma ch
      y elems row_ind em j step #() to k1;
    tile_vmprod_slice_lemma ch
      y elems row_ind em j step to k1
      (seq_load_vmprod ch y elems row_ind em j step #() to)
  )

open FStar.Seq { slice_slice }


let seq_load_vmprod_step_lemma
  (ch : pos)
  (m1 #n1 : nat { ch /? n1 })
  (y0 : lseq real n1)
  (#nnz : erased nat)
  (elems : erased (lseq real nnz))
  (row_ind : erased (lseq nat nnz))
  (to : erased nat { to + m1 <= nnz })
  (cnt : erased nat { cnt <= m1 })
  (#m2 #n2 : pos { ch /? n2 })
  (em : chest2 real m2 n2)
  (j : nat { ch /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (y : lseq real n1)
: Lemma
  (requires
    tile_mm_result ch
      y0
      (Seq.slice elems 0 to <: lseq real to) (Seq.slice row_ind 0 to)
      em
      j step
      y
  )
  (ensures
    tile_mm_result ch
      y0
      (Seq.slice elems 0 (to + cnt) <: lseq real (to + cnt))
      (Seq.slice row_ind 0 (to + cnt))
      em
      j step
      (seq_load_vmprod ch
        y
        (Seq.slice elems to (to + m1) <: lseq real m1)
        (Seq.slice row_ind to (to + m1))
        em j step cnt
      )
  )
=
  let elems2 : lseq real cnt = Seq.slice elems to (to + cnt) in
  let elems2' : lseq real m1 = Seq.slice elems to (to + m1) in
  let elems12 : lseq real (to + cnt) = Seq.slice elems 0 (to + cnt) in

  let row_ind2 : lseq nat cnt = Seq.slice row_ind to (to + cnt) in
  let row_ind2' : lseq nat m1 = Seq.slice row_ind to (to + m1) in
  let row_ind12 : lseq nat (to + cnt) = Seq.slice row_ind 0 (to + cnt) in

  let y' = seq_load_vmprod ch y elems2' row_ind2' em j step cnt in

  seq_load_vmprod_lemma ch y elems2' row_ind2' em j step cnt;

  slice_slice elems   to (to + m1) 0 cnt;
  slice_slice row_ind to (to + m1) 0 cnt;

  assert tile_mm_result ch y elems2 row_ind2 em j step y';

  introduce forall (k1 : natlt n1).
    tile_vmprod_cell_prop ch y0 elems12 row_ind12 em j step #() (to + cnt) k1 y'
  with (
    let k2 = j + k1 / ch * step * ch + k1 % ch in
    if k2 < n2
      then
        sparse_dprod_accum
          (y0 @! k1)
          elems row_ind
          (ematrix_col em k2)
          to (to + cnt)
      else ()
  )
