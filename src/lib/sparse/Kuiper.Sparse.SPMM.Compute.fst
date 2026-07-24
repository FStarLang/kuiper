module Kuiper.Sparse.SPMM.Compute

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Seq.Common { (@+), seq_replace }
open Kuiper.Spec.GEMM
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.Array
open Kuiper.Sparse.Common
open Kuiper.Sparse.SPMM.Defs { chest2_tile_prop }
open Kuiper.Array.Vectorized
open Kuiper.Tensor.Layout.Alg { l2_row_major, c_l2_row_major }
open Kuiper.Tensor
open Kuiper.Seq.Common { op_At_Bang }

// [up] on a 1-D concrete index is the corresponding abstract index.
let up_cidx1_eq (#d0:nat) (i:szlt d0)
  : Lemma (up (cidx1 i) == idx1 (v i))
          [SMTPat (up (cidx1 i))]
  = ()

// [seq_to_chest1] and [chest1_to_seq] are mutually inverse.
let chest1_to_seq_to_chest1 (#et : Type) (#n : nat) (s : lseq et n)
  : Lemma (chest1_to_seq (seq_to_chest1 s) == s)
          [SMTPat (chest1_to_seq (seq_to_chest1 s))]
  = Seq.lemma_eq_elim (chest1_to_seq (seq_to_chest1 s)) s

let seq_to_chest1_to_seq (#et : Type) (#n : nat) (c : chest1 et n)
  : Lemma (seq_to_chest1 (chest1_to_seq c) == c)
          [SMTPat (seq_to_chest1 (chest1_to_seq c))]
  = assert (equal (seq_to_chest1 (chest1_to_seq c)) c)

// The [lseq] view of a chest2 row is exactly [ematrix_row].
let chest2_row_to_seq (#et : Type0) (#rows #cols : nat)
  (em : chest2 et rows cols) (i : natlt rows)
  : Lemma (chest1_to_seq (chest2_row em i) == ematrix_row em i)
          [SMTPat (chest1_to_seq (chest2_row em i))]
  = Seq.lemma_eq_elim (chest1_to_seq (chest2_row em i)) (ematrix_row em i)

// The [lseq] view of a chest2 column is exactly [ematrix_col].
let ematrix_col_is_chest (#et : Type0) (#rows #cols : nat)
  (em : chest2 et rows cols) (j : natlt cols)
  : Lemma (ematrix_col em j == chest1_to_seq (chest2_col em j))
  = Seq.lemma_eq_elim (ematrix_col em j) (chest1_to_seq (chest2_col em j))

noextract
let seq_scalar_prod
  (#et : Type0) {| scalar et |}
  (t : lseq et 'n)
  (k : et)
  (s : lseq et 'n)
: lseq et 'n
= Seq.init 'n fun i -> (t @! i) `add` (k `mul` (s @! i))

inline_for_extraction noextract
fn scalar_prod
  (#et : Type0) {| scalar et |}
  (#n : sz)
  (#ly : layout1 n) {| ctlayout ly |}
  (y : array1 et ly)
  (#vy : chest1 et n)
  (k : et)
  (#lx : layout1 n) {| ctlayout lx |}
  (x : array1 et lx)
  (#fx : perm)
  (#vx : chest1 et n)
  preserves gpu
  preserves x |-> Frac fx vx
  requires  y |-> vy
  ensures   y |-> seq_to_chest1 (seq_scalar_prod (chest1_to_seq vy) k (chest1_to_seq vx))
{
  let mut ix : sz = 0sz;

  while (!ix <^ n)
    invariant exists* vix (vy' : chest1 et n).
      ix |-> vix **
      y  |-> vy' **
      pure (
        vix <= n /\
        (forall (i : natlt n { i < vix }).
          acc1 vy' i == seq_scalar_prod (chest1_to_seq vy) k (chest1_to_seq vx) @! i) /\
        forall (i : natlt n { i >= vix }).
          acc1 vy' i == acc1 vy i
      )
  {
    let ixv = !ix;
    let cur = tensor_read y (cidx1 (ixv <: szlt n));
    let xv = tensor_read x (cidx1 (ixv <: szlt n));
    tensor_write y (cidx1 (ixv <: szlt n)) (cur `add` (k `mul` xv));
    ix := !ix +^ 1sz;
  };

  with vy'. assert y |-> vy';
  assert pure (
    equal vy'
      (seq_to_chest1 (seq_scalar_prod (chest1_to_seq vy) k (chest1_to_seq vx)))
  );
}

let seq_vmprod
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (acc : lseq et cols)
  (v : lseq et rows)
  (m : chest2 et rows cols)
: GTot (lseq et cols)
= Seq.init_ghost cols (fun i -> dprod_acc (acc @! i) v (ematrix_col m i))

inline_for_extraction noextract
fn vmprod
  (#et : Type0) {| scalar et |}
  (#rows #cols : sz)
  (#ly : layout1 cols) {| ctlayout ly |}
  (y : array1 et ly)
  (#vy : chest1 et cols)
  (#lx : layout1 rows) {| ctlayout lx |}
  (x : array1 et lx)
  (#fx : perm)
  (#vx : chest1 et rows)
  (#lm : layout2 rows cols) {| ctlayout lm |}
  (m : array2 et lm)
  (#fm : perm)
  (#vm : chest2 et rows cols)
  norewrite
  preserves gpu
  preserves x |-> Frac fx vx
  preserves m |-> Frac fm vm
  requires  y |-> vy
  ensures   y |-> seq_to_chest1 (seq_vmprod (chest1_to_seq vy) (chest1_to_seq vx) vm)

{
  let mut k : sz = 0sz;
  while (!k <^ rows)
    invariant
      exists* vk (vy' : chest1 et cols).
        k |-> vk **
        y |-> vy' **
        pure (
          vk <= rows /\
          forall (i : natlt cols).
            chest1_to_seq vy' @! i ==
            _dprod_acc (chest1_to_seq vy @! i) (chest1_to_seq vx) (ematrix_col vm i) vk
        )
  {
    let kv = !k;
    let xk = tensor_read x (cidx1 (kv <: szlt rows));
    tensor_extract_row_ro m (v kv);
    scalar_prod
        y xk
        #_ #(Kuiper.Tensor.ctlayout_slice _ 0 (v kv)) // should not be needed
        (tensor_row m (v kv));
    tensor_restore_row m (v kv);

    k := !k +^ 1sz;
  };

  with vy'. assert y |-> vy';
  assert pure (
    equal vy'
      (seq_to_chest1 (seq_vmprod (chest1_to_seq vy) (chest1_to_seq vx) vm))
  );
}

let tile_vmprod_cell_prop
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (acc : erased (lseq et n1))
  (elems : erased (lseq et m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  (k1 : natlt n1)
  (y : lseq et n1)
: prop
=
  let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
  k2 < n2 ==>
  y @! k1 == _sparse_dprod_acc (acc @! k1) elems row_ind (ematrix_col em2 k2) to

let _tile_vmprod_prop
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (acc : erased (lseq et n1))
  (elems : erased (lseq et m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  (y : lseq et n1)
: prop
= forall (k1 : natlt n1). tile_vmprod_cell_prop acc elems row_ind em2 j step to k1 y

let tile_vmprod_prop
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (acc : erased (lseq et n1))
  (elems : erased (lseq et m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (y : lseq et n1)
: prop
= _tile_vmprod_prop acc elems row_ind em2 j step m1 y


let tile_vmprod_lemma
  (#et : Type0) {| scalar et, sized et, hvc : has_vec_cpy et |}
  (#m1 #n1 : sz { chunk et /? n1 })
  (vy : erased (lseq et n1))
  (vy0 : erased (lseq et n1))
  (#nnz : erased nat)
  (elems : erased (lseq et nnz))
  (row_ind : erased (lseq nat nnz))
  (to : erased nat { to + m1 <= nnz })
  (tem : chest2 et m1 n1)
  (#m2 #n2 : nat {  chunk et /? n2 })
  (gem : chest2 et m2 n2)
  (j : sz { chunk et /? j })
  (step : sz)
  (#_ : squash (in_bounds 0 m2 row_ind))
: Lemma
  (requires
    chest2_tile_prop gem (Seq.slice row_ind to (to + m1)) j step tem /\
    tile_vmprod_prop
      vy0
      (Seq.slice elems 0 to <: lseq et to) (Seq.slice row_ind 0 to)
      gem
      j step
      vy
  )
  (ensures
    tile_vmprod_prop #_ #_ #_ #hvc
      vy0
      (Seq.slice elems 0 (to + m1) <: lseq et (to + m1)) (Seq.slice row_ind 0 (to + m1))
      gem
      j step
      (seq_vmprod
        vy
        (Seq.slice elems to (to + m1) <: lseq et m1)
        tem)
  )
=
  let elems1 : lseq et to = Seq.slice elems 0 to in
  let elems2 : lseq et m1 = Seq.slice elems to (to + m1) in
  let elems12 : lseq et (to + m1) = Seq.slice elems 0 (to + m1) in

  let row_ind1 : lseq nat to = Seq.slice row_ind 0 to in
  let row_ind2 : lseq nat m1 = Seq.slice row_ind to (to + m1) in
  let row_ind12 : lseq nat (to + m1) = Seq.slice row_ind 0 (to + m1) in

  let r = seq_vmprod vy elems2 tem in

  introduce forall (k1 : natlt n1).
    tile_vmprod_cell_prop vy0 elems12 row_ind12 gem j step #() (to + m1) k1 r
  with
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
    if k2 < n2
      then (
        // [chest2_tile_prop] relates [chest2_col tem k1] to the sparse column
        // of [gem]; bridge it back to the [lseq] view [ematrix_col tem k1] that
        // [seq_vmprod] uses.
        ematrix_col_is_chest tem k1;
        sparse_dprod_accum
          (vy0 @! k1)
          elems row_ind
          (ematrix_col gem k2)
          to (to + m1)
      )
      else ()

let tile_vmprod_prop_lemma0
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (acc : erased (lseq et n1))
  (elems : erased (lseq et m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
: Lemma
  (requires m1 == 0)
  (ensures tile_vmprod_prop acc elems row_ind em2 j step acc)
= ()

// let tile_result_cell_prop
//   (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
//   (#rows #shared #cols : nat)
//   (em1 : chest2 et rows shared)
//   (i : natlt rows)
//   (em2 : chest2 et shared cols)
//   (j : nat)
//   (step : nat)
//   (#tlen : nat)
//   (tile : lseq et tlen)
//   (k1 : natlt tlen)
// : prop
// =
//   let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
//   k2 < cols ==>
//   tile @! k1 == matmul_single em1 em2 i k2

// let tile_result_prop
//   (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
//   (#rows #shared #cols : nat)
//   (em1 : chest2 et rows shared)
//   (i : natlt rows)
//   (em2 : chest2 et shared cols)
//   (j : nat)
//   (step : nat)
//   (#tlen : nat)
//   (tile : lseq et tlen)
// : prop
// =
//   forall (k1 : natlt tlen).
//     tile_result_cell_prop em1 i em2 j step tile k1

// let tile_result_lemma'
//   (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
//   (#rows #shared #cols : nat)
//   (em1 : chest2 et rows shared)
//   (i : natlt rows)
//   (em2 : chest2 et shared cols)
//   (j : nat)
//   (step : nat)
//   (#tlen : nat)
//   (tile : lseq et tlen)
//   (k1 : natlt tlen)
// : Lemma true
// = admit()


let tile_mask_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#rows #shared #cols : nat { chunk et /? cols })
  (em1 : chest2 et rows shared)
  (i : natlt rows)
  (#nnz : nat)
  (mask_len : natle nnz)
  (elems : erased (lseq et (nnz - mask_len)))
  (row_ind : erased (lseq nat nnz))
  (#_ : squash (in_bounds 0 shared row_ind /\ sorted row_ind))
  (em2 : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (#tlen : nat { chunk et /? tlen })
  (tile0 : lseq et tlen)
  (tile : lseq et tlen)
: Lemma
  (requires
    tile_vmprod_prop
      tile0
      (Seq.create mask_len zero @+ elems) row_ind
      em2 j step tile
  )
  (ensures
    tile_vmprod_prop #_ #_ #_ #solve
      tile0
      elems (Seq.slice row_ind mask_len nnz)
      em2 j step tile
  )
=
  let mask : lseq et mask_len = Seq.create mask_len zero in
  let elems' : lseq et nnz = mask @+ elems in
  let row_ind' : lseq nat (nnz - mask_len) = Seq.slice row_ind mask_len nnz in
  introduce forall (k1 : natlt tlen).
    tile_vmprod_cell_prop tile0 elems row_ind' em2 j step #() (nnz - mask_len) k1 tile 
  with (
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
    if k2 < cols
      then calc(==) {
        tile @! k1;
        == {}
        sparse_dprod_acc (tile0 @! k1) elems' row_ind (ematrix_col em2 k2);
        == {}
        dprod_acc (tile0 @! k1) elems' (seq_make_sparse row_ind (ematrix_col em2 k2));
        == {
          _dprod_acc_mask_lemma
            (tile0 @! k1)
            mask_len
            elems (seq_make_sparse row_ind (ematrix_col em2 k2))
            nnz
        }
        dprod_acc
          (tile0 @! k1)
          elems
          (Seq.slice (seq_make_sparse row_ind (ematrix_col em2 k2)) mask_len nnz);
        == { seq_make_sparse_slice row_ind mask_len nnz (ematrix_col em2 k2) }
        dprod_acc
          (tile0 @! k1)
          elems
          (seq_make_sparse row_ind' (ematrix_col em2 k2));
        == {}
        sparse_dprod_acc (tile0 @! k1) elems row_ind' (ematrix_col em2 k2);
      }
      else ()
  )

let tile_result_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#rows #shared #cols : nat { chunk et /? cols })
  (em1 : chest2 et rows shared)
  (i : natlt rows)
  (#nnz : nat)
  (elems : erased (lseq et nnz))
  (row_ind : erased (lseq nat nnz))
  (#_ : squash (in_bounds 0 shared row_ind /\ sorted row_ind))
  (em2 : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (#tlen : nat { chunk et /? tlen })
  (tile : lseq et tlen)
: Lemma
  (requires
    unsparse _ _ elems row_ind == ematrix_row em1 i /\
    tile_vmprod_prop
      (Seq.create tlen zero)
      elems row_ind
      em2
      j step
      tile
  )
  (ensures tile_result_prop #_ #_ #_ #solve em1 i em2 j step tile)
=
  let tile0 : lseq et tlen = Seq.create tlen zero in
  introduce forall (k1 : natlt tlen).
    tile_result_cell_prop em1 i em2 j step tile k1
  with (
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
    if k2 < cols
      then calc (==) {
        tile @! k1;
        == { sparse_dprod_lemma elems row_ind (ematrix_col em2 k2) }
        dprod (ematrix_row em1 i) (ematrix_col em2 k2);
        == { dprod_is_matmul_single em1 em2 i k2 }
        matmul_single em1 em2 i k2;
      }
      else ()
  )


inline_for_extraction noextract
fn tile_vmprod
  (#et : Type0) {| scalar et, sized et, hvc : has_vec_cpy et |}
  (#m1 #n1 : sz { chunk et /? n1 })
  (#ly : layout1 n1) {| ctlayout ly |}
  (y : array1 et ly)
  (#vy : chest1 et n1)
  (vy0 : erased (lseq et n1))
  (#lx : layout1 m1) {| ctlayout lx |}
  (x : array1 et lx)
  (#fx : perm)
  (#nnz : erased nat)
  (elems : erased (lseq et nnz))
  (row_ind : erased (lseq nat nnz))
  (to : erased nat { to + m1 <= nnz })
  (#ltm : layout2 m1 n1) {| ctlayout ltm |}
  (tm : array2 et ltm)
  (#tem : chest2 et m1 n1)
  (#ftm : perm)
  (#m2 #n2 : nat {  chunk et /? n2 })
  (gem : chest2 et m2 n2)
  (j : sz { chunk et /? j })
  (step : sz)
  (#_ : squash (in_bounds 0 m2 row_ind /\ sorted row_ind))
  norewrite
  preserves gpu
  preserves x  |-> Frac fx (seq_to_chest1 (Seq.slice elems to (to + m1) <: lseq et m1))
  preserves tm |-> Frac ftm tem
  requires  pure (chest2_tile_prop #_ #_ #hvc gem (Seq.slice row_ind to (to + m1)) j step tem)
  requires  y  |-> vy
  requires
    pure (
      tile_vmprod_prop
        vy0
        (Seq.slice elems 0 to <: lseq et to) (Seq.slice row_ind 0 to)
        gem
        j step
        (chest1_to_seq vy)
    )
  ensures exists* (vy' : chest1 et n1).
    y |-> vy' **
    pure (
      tile_vmprod_prop
        vy0
        (Seq.slice elems 0 (to + m1) <: lseq et (to + m1)) (Seq.slice row_ind 0 (to + m1))
        gem
        j step
        (chest1_to_seq vy')
    )
{
  vmprod y x tm;
  tile_vmprod_lemma (chest1_to_seq vy) vy0 elems row_ind to tem gem j step #_;
  ();
}


inline_for_extraction noextract
let fma
  (#et : Type0) {| scalar et |}
  (x1 x2 y : et)
: et = y `add` (x1 `mul` x2)

noextract
let seq_fma
  (#et : Type0) {| scalar et |}
  (x1 : et)
  (#n : nat)
  (x2 : lseq et n)
  (#sz_y : nat)
  (y : lseq et sz_y)
  (k : nat { k + n <= sz_y })
  (to : natle n)
: lseq et sz_y
= seq_replace y k (k + to) (Seq.init to fun i -> fma x1 (x2 @! i) (y @! k + i))

// esto es scalar_prod pero sobre un fragmento del array
// TODO unificar definiciones?
inline_for_extraction noextract
fn fma_arr
  (#et : Type0) {| scalar et |}
  (x1 : et)
  (n : sz)
  (x2 : larray et n)
  (#vx2 : erased (lseq et n))
  (#sz_y : erased nat)
  (#ly : layout1 sz_y) {| ctlayout ly |}
  (y : array1 et ly)
  (#vy : chest1 et sz_y)
  (k : sz { k + n <= sz_y })
  preserves gpu
  preserves x2 |-> vx2
  requires  y |-> vy 
  ensures  y |-> seq_to_chest1 (seq_fma x1 vx2 (chest1_to_seq vy) k n)
{
  let mut ix : sz = 0sz;
  while (!ix <^ n)
    invariant exists* vix (vy' : chest1 et sz_y).
      ix |-> vix **
      y |-> vy' **
      pure (
        vix <= n /\
        (forall (i : natlt sz_y).
          acc1 vy' i ==
            (if k <= i && i < k + vix
             then fma x1 (vx2 @! (i - k)) (acc1 vy i)
             else acc1 vy i))
      )
  {
    let ixv = !ix;
    // y[k + ix] += x1 * x2[ix]
    let x2v = Pulse.Lib.Array.(x2.(ixv));
    let yv = tensor_read y (cidx1 (k +^ ixv <: szlt sz_y));
    tensor_write y (cidx1 (k +^ ixv <: szlt sz_y)) (fma x1 x2v yv);
    ix := !ix +^ 1sz;
  };

  with vy'. assert y |-> vy';
  assert pure (
    equal vy' (seq_to_chest1 (seq_fma x1 vx2 (chest1_to_seq vy) k n))
  );
}

noextract
let seq_fma'
  (#et : Type0) {| scalar et |}
  (cnt : nat)
  (x1 : et)
  (#n : nat { cnt /? n })
  (x2 : lseq et n)
  (#sz_y : nat)
  (y : lseq et sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
: Tot (lseq et sz_y)
=
  if k2 < n
    then seq_fma x1 #cnt (Seq.slice x2 k2 (k2 + cnt)) y k1 cnt
    else y

open Kuiper.Sparse.Load { array_vec_cpy_dh }

inline_for_extraction noextract
fn load_vmprod_chunk
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : sz { chunk et /? n1 })
  (#ly : layout1 n1) {| ctlayout ly |}
  (y : array1 et ly)
  (#vy : chest1 et n1)
  (x : et)
  (#n2 : sz { chunk et /? n2 })
  (#lrow : layout1 n2) {| ctlayout lrow, clrow : cont_layout lrow |}
  (row : array1 et lrow)
  (#frow : perm)
  (#vrow : chest1 et n2)
  (k1 : sz { k1 + chunk et <= n1 })
  (k2 : sz { chunk et /? k2 })
  preserves gpu
  preserves row |-> Frac frow vrow
  requires  pure (aligned 16 (core row))
  requires  pure (aligned_cont_layout (chunk et) clrow)
  requires  y |-> vy
  ensures   y |-> seq_to_chest1 (seq_fma' (chunk et) x (chest1_to_seq vrow) (chest1_to_seq vy) k1 k2)
{
  if (k2 <^ n2)
  {
    let mut lchunk = [| zero #et #_; chunk et |];
    // Freshly allocated scratch buffer is suitably aligned for vectorized copy
    // (cf. the [assume pure (aligned ...)] allocation idiom in Kuiper.Array.Core).
    assume pure (aligned 16 lchunk);
    rewrite (row |-> Frac frow vrow)
         as (row |-> Frac frow (seq_to_chest1 (chest1_to_seq vrow)));
    array_vec_cpy_dh lchunk row k2;
    rewrite (row |-> Frac frow (seq_to_chest1 (chest1_to_seq vrow)))
         as (row |-> Frac frow vrow);
    fma_arr x (chunk et) lchunk y k1;
  }
  else {}
}

noextract
let rec seq_load_vmprod_row
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (y : lseq et n1)
  (x : et)
  (#n2 : nat { chunk et /? n2 })
  (row : lseq et n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (k : natle (n1 / chunk et))
: Tot (lseq et n1)
= 
  let ch : nat = v (chunk et) in 
  if k = 0 then y
    else (
      lineal_divides ch j ch ((k - 1) * step);
      seq_fma'
        ch x row
        (seq_load_vmprod_row y x row j step (k - 1))
        ((k - 1) * ch)
        (j + (k - 1) * step * ch)
    )


inline_for_extraction noextract
fn load_vmprod_row
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : sz { chunk et /? n1 })
  (#ly : layout1 n1) {| ctlayout ly |}
  (y : array1 et ly)
  (#vy : chest1 et n1)
  (x : et)
  (#n2 : sz { chunk et /? n2 })
  (#lrow : layout1 n2) {| ctlayout lrow, clrow : cont_layout lrow |}
  (row : array1 et lrow)
  (#frow : perm)
  (#vrow : chest1 et n2)
  (j : sz { chunk et /? j })
  (step : sz)
  preserves gpu
  preserves row |-> Frac frow vrow
  requires  pure (aligned 16 (core row))
  requires  pure (aligned_cont_layout (chunk et) clrow)
  requires  pure (fits (j + n1 * step))
  requires  y |-> vy
  ensures   y |-> seq_to_chest1 (seq_load_vmprod_row (chest1_to_seq vy) x (chest1_to_seq vrow) j step (n1 / chunk et))
{
  let mut k : sz = 0sz; 

  while (!k <^ n1 /^ chunk et)
    invariant exists* vk (vy' : chest1 et n1).
      k |-> vk **
      y |-> vy' **
      pure (
        vk <= n1 / chunk et /\
        Seq.equal (chest1_to_seq vy') (seq_load_vmprod_row (chest1_to_seq vy) x (chest1_to_seq vrow) j step vk)
      )
  {
    assert pure (fits (j + !k * step * chunk et));
    lemma_divides_product (chunk et) !k;
    lineal_divides (chunk et) j (chunk et) (!k * step);

    load_vmprod_chunk
      y x
      row
      (!k *^ chunk et) (j +^ !k *^ step *^ chunk et);
    k := !k +^ 1sz;
  }
}

noextract
let rec seq_load_vmprod
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (y : lseq et n1)
  (elems : lseq et m1)
  (row_ind : lseq nat m1)
  (#m2 #n2 : pos { chunk et /? n2 })
  (em : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
: GTot (lseq et n1)
=
  if to = 0 then y
    else
      seq_load_vmprod_row
        (seq_load_vmprod y elems row_ind em j step (to - 1))
        (elems @! to - 1)
        (ematrix_row em (row_ind @! to - 1))
        j step (n1 / chunk et)
  

open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }

inline_for_extraction noextract
fn load_vmprod
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : sz { chunk et /? n1 })
  (#ly : layout1 n1) {| ctlayout ly |}
  // en realidad y es un larray... por el momento no podemos unificar
  (y : array1 et ly)
  (#vy : chest1 et n1)
  (#lx  : layout1 m1) {| ctlayout lx |}
  (elems : array1 et lx)
  (row_ind : array1 sz lx)
  (#fx : perm)
  (#velems : chest1 et m1)
  (#vrow_ind : chest1 sz m1)
  (#m2 #n2 : szp { chunk et /? n2 })
  (#lm : layout2 m2 n2) {| ctlayout lm, srm : strided_row_major lm |}
  (m : array2 et lm)
  (#fm : perm)
  (#em : chest2 et m2 n2)
  (j : sz { chunk et /? j })
  (step : sz)
  (#_ : squash (in_bounds 0 m2 (cast_pos (chest1_to_seq vrow_ind))))
  (to : szlt m1)
  preserves gpu
  preserves elems |-> Frac fx velems
  preserves row_ind |-> Frac fx vrow_ind
  preserves m |-> Frac fm em
  requires  pure (aligned 16 (core m) /\ aligned_strided_row_major (chunk et) srm)
  requires  pure (fits (j + n1 * step))
  requires  y |-> vy
  ensures   y |-> seq_to_chest1 (seq_load_vmprod (chest1_to_seq vy) (chest1_to_seq velems) (cast_pos (chest1_to_seq vrow_ind)) em j step to)
{
  let mut k : sz = 0sz;

  while (!k <^ to)
    invariant exists* vk (vy' : chest1 et n1).
      k |-> vk **
      y |-> vy' **
      pure (
        vk <= to /\
        Seq.equal (chest1_to_seq vy') (seq_load_vmprod (chest1_to_seq vy) (chest1_to_seq velems) (cast_pos (chest1_to_seq vrow_ind)) em j step vk)
      )
  {
    let kv = !k;
    let kr = tensor_read row_ind (cidx1 (kv <: szlt m1));
    let kx = tensor_read elems (cidx1 (kv <: szlt m1));

    // [kr] indexes a valid row of [m] by the sparsity bound.
    assert pure (v kr == cast_pos (chest1_to_seq vrow_ind) @! (v kv));

    tensor_extract_row_ro m (v kr);
    row_core_lemma m (v kr);
    aligned_cont_strided_row_major lm (chunk et) kr;

    load_vmprod_row
      y kx
      #_ #_ #(Kuiper.Tensor.ctlayout_slice _ 0 (v kr)) // should not be needed
      (tensor_row m (v kr)) j step;

    tensor_restore_row m (v kr);

    k := !k +^ 1sz;
  }
}

noextract
let seq_fma_cell_prop
  (#et : Type0) {| scalar et |}
  (x1 : et)
  (#n : nat)
  (x2 : lseq et n)
  (#sz_y : nat)
  (y0 : lseq et sz_y)
  (k : nat { k + n <= sz_y })
  (to : natle n)
  (y : lseq et sz_y)
  (ix : natlt to)
: prop
= y @! k + ix == add (y0 @! k + ix) (x1 `mul` (x2 @! ix))

noextract
let seq_fma_lemma
  (#et : Type0) {| scalar et |}
  (x1 : et)
  (#n : nat)
  (x2 : lseq et n)
  (#sz_y : nat)
  (y : lseq et sz_y)
  (k : nat { k + n <= sz_y })
  (to : natle n)
: Lemma
  (requires true)
  (ensures forall (ix : natlt to).
    seq_fma_cell_prop x1 x2 y k to (seq_fma x1 x2 y k to) ix)
= ()

noextract
let seq_fma_cell_prop'
  (#et : Type0) {| scalar et |}
  (cnt : nat)
  (x1 : et)
  (#n : nat { cnt /? n })
  (x2 : lseq et n)
  (#sz_y : nat)
  (y0 : lseq et sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
  (y : lseq et sz_y)
  (ix : natlt cnt)
: prop
=
  k2 < n ==> y @! k1 + ix == add (y0 @! k1 + ix) (x1 `mul` (x2 @! k2 + ix))

noextract
let seq_fma_lemma0'
  (#et : Type0) {| scalar et |}
  (cnt : nat)
  (x1 : et)
  (#n : nat { cnt /? n })
  (x2 : lseq et n)
  (#sz_y : nat)
  (y0 : lseq et sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
: Lemma
  (requires true)
  (ensures forall (i : natlt sz_y { i < k1 \/ k1 + cnt <= i }).
    seq_fma' cnt x1 x2 y0 k1 k2 @! i == y0 @! i) 
= ()

noextract
let seq_fma_lemma'
  (#et : Type0) {| scalar et |}
  (cnt : nat)
  (x1 : et)
  (#n : nat { cnt /? n })
  (x2 : lseq et n)
  (#sz_y : nat)
  (y0 : lseq et sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
: Lemma
  (requires true)
  (ensures forall (ix : natlt cnt).
    seq_fma_cell_prop' cnt x1 x2 y0 k1 k2 (seq_fma' cnt x1 x2 y0 k1 k2) ix)
=
  if k2 < n
    then (
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
    
noextract
let seq_load_vmprod_row_cell_prop_
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (y0 : lseq et n1)
  (x : et)
  (#n2 : nat { chunk et /? n2 })
  (row : lseq et n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (k : natle (n1 / chunk et))
  (y : lseq et n1)
  (ik : natlt k)
  (ix : natlt (chunk et))
: prop
=
  lineal_divides (chunk et) j (chunk et) (ik * step);
  seq_fma_cell_prop'
    (chunk et) x row
    y0
    (ik * chunk et)
    (j + ik * step * chunk et) y ix

noextract
let seq_load_vmprod_row_cell_prop
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (y0 : lseq et n1)
  (x : et)
  (#n2 : nat { chunk et /? n2 })
  (row : lseq et n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (k : natle (n1 / chunk et))
  (y : lseq et n1)
  (ik : natlt (k * chunk et))
: prop
=
  j + ik / chunk et * step * chunk et + ik % chunk et < n2 ==>
  y @! ik ==
  add
    (y0 @! ik)
    (x `mul` (row @! j + ik / chunk et * step * chunk et + ik % chunk et))

noextract
let seq_load_vmprod_row_cell_prop_equiv
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (y0 : lseq et n1)
  (x : et)
  (#n2 : nat { chunk et /? n2 })
  (row : lseq et n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (y : lseq et n1)
  (k : natle (n1 / chunk et))
  (i : natlt (k * chunk et))
: Lemma
  (requires seq_load_vmprod_row_cell_prop_
    y0 x row j step k y (i / chunk et) (i % chunk et))
  (ensures  seq_load_vmprod_row_cell_prop  y0 x row j step k y i)
= ()

noextract
let rec seq_load_vmprod_row_cell_lemma0
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (y0 : lseq et n1)
  (x : et)
  (#n2 : nat { chunk et /? n2 })
  (row : lseq et n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (k : natle (n1 / chunk et))
  (i : natlt n1 { k * chunk et <= i })
: Lemma
  (requires true)
  (ensures seq_load_vmprod_row y0 x row j step k @! i == y0 @! i)
=
  if k = 0 then ()
  else (
    lineal_divides (chunk et) j (chunk et) ((k - 1) * step);
    seq_load_vmprod_row_cell_lemma0 y0 x row j step (k - 1) i;
    seq_fma_lemma0' (chunk et) x row
      (seq_load_vmprod_row y0 x row j step (k - 1))
      ((k - 1) * chunk et) (j + (k - 1) * step * chunk et);
    ()
  )

#push-options "--z3rlimit 10"
noextract
let rec seq_load_vmprod_row_cell_lemma_
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (y0 : lseq et n1)
  (x : et)
  (#n2 : nat { chunk et /? n2 })
  (row : lseq et n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (k : natle (n1 / chunk et))
  (ik : natlt k)
  (ix : natlt (chunk et))
: Lemma
  (requires true)
  (ensures
    seq_load_vmprod_row_cell_prop_
      y0 x row j step k
      (seq_load_vmprod_row y0 x row j step k)
      ik ix
  )
= 
  if k = 0 then ()
  else (
    lineal_divides (chunk et) j (chunk et) ((k - 1) * step);
    if ik < k - 1
      then (
        seq_load_vmprod_row_cell_lemma_ y0 x row j step (k - 1) ik ix;
        assert ik * chunk et + ix < (k - 1) * chunk et;
        seq_fma_lemma0' (chunk et) x row
          (seq_load_vmprod_row y0 x row j step (k - 1))
          ((k - 1) * chunk et) (j + (k - 1) * step * chunk et);
        ()
      )
      else (
        assert ik == k - 1;
        seq_fma_lemma' (chunk et) x row
          (seq_load_vmprod_row y0 x row j step (k - 1))
          ((k - 1) * chunk et) (j + (k - 1) * step * chunk et);
        seq_load_vmprod_row_cell_lemma0
          y0 x row j step (k - 1) (ik * chunk et + ix);
        ()
      )
  )
#pop-options

noextract
let tile_vmprod_slice_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (acc : erased (lseq et n1))
  (elems : erased (lseq et m1))
  (row_ind : erased (lseq nat m1))
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  (k1 : natlt n1)
  (y : lseq et n1)
: Lemma 
  (requires tile_vmprod_cell_prop acc elems row_ind em2 j step to k1 y)
  (ensures tile_vmprod_cell_prop #_ #_ #_ #solve
    #to
    acc
    (Seq.slice elems 0 to)
    (Seq.slice row_ind 0 to)
    em2 j step to k1 y
  )
=
  let elems' : lseq et to = Seq.slice elems 0 to in
  let row_ind' : lseq nat to = Seq.slice row_ind 0 to in

  let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in

  if k2 < n2
    then sparse_dprod_slice_lemma (acc @! k1) elems row_ind (ematrix_col em2 k2) to to
    else ()

noextract
let rec seq_load_vmprod_cell_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (y : lseq et n1)
  (elems : lseq et m1)
  (row_ind : lseq nat m1)
  (#m2 #n2 : pos { chunk et /? n2 })
  (em : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  (k1 : natlt n1)
: Lemma
  (requires true)
  (ensures
    tile_vmprod_cell_prop y elems row_ind em j step to k1
      (seq_load_vmprod y elems row_ind em j step to)
  )
=
  if to = 0 then ()
  else (
    seq_load_vmprod_cell_lemma y elems row_ind em j step (to - 1) k1;
    seq_load_vmprod_row_cell_lemma_
      (seq_load_vmprod y elems row_ind em j step (to - 1))
      (elems @! to - 1)
      (ematrix_row em (row_ind @! to - 1))
      j step (n1 / chunk et) (k1 / chunk et) (k1 % chunk et);
    // seq_load_vmprod_row_cell_prop_equiv
    //   (seq_load_vmprod y elems row_ind em j step (to - 1))
    //   (elems @! to - 1)
    //   (ematrix_row em (row_ind @! to - 1))
    //   j step
    //   (seq_load_vmprod_row
    //     (seq_load_vmprod y elems row_ind em j step (to - 1))
    //     (elems @! to - 1)
    //     (ematrix_row em (row_ind @! to - 1))
    //     j step (n1 / chunk et))
    //   (n1 / chunk et) k1;
    ()
  )

noextract
let seq_load_vmprod_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (y : lseq et n1)
  (elems : lseq et m1)
  (row_ind : lseq nat m1)
  (#m2 #n2 : pos { chunk et /? n2 })
  (em : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
: Lemma
  (requires true)
  (ensures
    tile_vmprod_prop y
      (Seq.slice elems 0 to <: lseq et to)
      (Seq.slice row_ind 0 to)
      em j step
      (seq_load_vmprod y elems row_ind em j step to)
  )
=
  let elems' : lseq et to = Seq.slice elems 0 to in
  let row_ind' : lseq nat to = Seq.slice row_ind 0 to in
  introduce forall (k1 : natlt n1).
    tile_vmprod_cell_prop y
      elems'
      row_ind'
      em j step
      #() to k1
      (seq_load_vmprod y elems row_ind em j step #() to)
  with (
    seq_load_vmprod_cell_lemma
      y elems row_ind em j step #() to k1;
    tile_vmprod_slice_lemma
      y elems row_ind em j step to k1 
      (seq_load_vmprod y elems row_ind em j step #() to)
  )

open FStar.Seq { slice_slice }

let seq_load_vmprod_step_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (m1 #n1 : nat { chunk et /? n1 })
  (y0 : lseq et n1)
  (#nnz : erased nat)
  (elems : erased (lseq et nnz))
  (row_ind : erased (lseq nat nnz))
  (to : erased nat { to + m1 <= nnz })
  (cnt : erased (natle m1))
  (#m2 #n2 : pos { chunk et /? n2 })
  (em : chest2 et m2 n2)
  (j : nat { chunk et /? j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (y : lseq et n1)
: Lemma
  (requires
    tile_vmprod_prop
      y0
      (Seq.slice elems 0 to <: lseq et to) (Seq.slice row_ind 0 to)
      em
      j step
      y
  )
  (ensures
    tile_vmprod_prop #_ #_ #_ #solve
      y0
      (Seq.slice elems 0 (to + cnt) <: lseq et (to + cnt))
      (Seq.slice row_ind 0 (to + cnt))
      em
      j step
      (seq_load_vmprod #_ #_ #_ #solve
        y
        (Seq.slice elems to (to + m1) <: lseq et m1)
        (Seq.slice row_ind to (to + m1))
        em j step cnt
      )
  )
=
  let elems2 : lseq et cnt = Seq.slice elems to (to + cnt) in
  let elems2' : lseq et m1 = Seq.slice elems to (to + m1) in
  let elems12 : lseq et (to + cnt) = Seq.slice elems 0 (to + cnt) in

  let row_ind2 : lseq nat cnt = Seq.slice row_ind to (to + cnt) in
  let row_ind2' : lseq nat m1 = Seq.slice row_ind to (to + m1) in
  let row_ind12 : lseq nat (to + cnt) = Seq.slice row_ind 0 (to + cnt) in

  let y' = seq_load_vmprod y elems2' row_ind2' em j step cnt in

  seq_load_vmprod_lemma y elems2' row_ind2' em j step cnt;
  
  slice_slice elems   to (to + m1) 0 cnt;
  slice_slice row_ind to (to + m1) 0 cnt;

  assert tile_vmprod_prop y elems2 row_ind2 em j step y';

  introduce forall (k1 : natlt n1).
    tile_vmprod_cell_prop y0 elems12 row_ind12 em j step #() (to + cnt) k1 y'
  with (
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
    if k2 < n2
      then
        sparse_dprod_accum
          (y0 @! k1)
          elems row_ind
          (ematrix_col em k2)
          to (to + cnt)
      else ()
  )

inline_for_extraction noextract
fn tile_load_vmprod
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : sz { chunk et /? n1 })
  (#ly : layout1 n1) {| ctlayout ly |}
  // en realidad y es un larray... por el momento no podemos unificar
  (y : array1 et ly)
  (#vy : chest1 et n1)
  (vy0 : erased (lseq et n1))
  (#lx  : layout1 m1) {| ctlayout lx |}
  (elems : array1 et lx)
  (row_ind : array1 sz lx)
  (#fx : perm)
  (#nnz : erased nat)
  (#velems : lseq et nnz)
  (#vrow_ind : lseq sz nnz)
  (#m2 #n2 : szp { chunk et /? n2 })
  (#lm : layout2 m2 n2) {| ctlayout lm, srm : strided_row_major lm |}
  (m : array2 et lm)
  (#fm : perm)
  (#em : chest2 et m2 n2)
  (j : sz { chunk et /? j })
  (step : sz)
  (#_ : squash (in_bounds 0 m2 (cast_pos vrow_ind)))
  (from to : erased nat { from + m1 <= nnz })
  (cant : szlt m1 { v cant == to - from })
  preserves gpu
  preserves elems   |-> Frac fx (seq_to_chest1 (Seq.slice velems from (from + m1) <: lseq et m1))
  preserves row_ind |-> Frac fx (seq_to_chest1 (Seq.slice vrow_ind from (from + m1) <: lseq sz m1))
  preserves m |-> Frac fm em
  requires  pure (aligned 16 (core m) /\ aligned_strided_row_major (chunk et) srm)
  requires  pure (fits (j + n1 * step))
  requires  y |-> vy
  requires
    pure (
      tile_vmprod_prop
        vy0
        (Seq.slice velems 0 from <: lseq et from)
        (Seq.slice (cast_pos vrow_ind) 0 from)
        em
        j step
        (chest1_to_seq vy)
    )
  ensures exists* (vy' : chest1 et n1).
    y |-> vy' **
    pure (
      tile_vmprod_prop
        vy0
        (Seq.slice velems 0 to <: lseq et to)
        (Seq.slice (cast_pos vrow_ind) 0 to <: lseq nat to)
        em j step (chest1_to_seq vy')
    )
{
  load_vmprod y elems row_ind m j step cant;
  seq_load_vmprod_step_lemma 
    m1
    vy0
    velems (cast_pos vrow_ind)
    from (v cant)
    em j step (chest1_to_seq vy);

  assert pure (
    Seq.equal
      (cast_pos #m1 (Seq.slice vrow_ind from (from + m1)))
      (Seq.slice (cast_pos vrow_ind) from (from + m1))
  );
}
