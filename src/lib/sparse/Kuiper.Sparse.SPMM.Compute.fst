module Kuiper.Sparse.SPMM.Compute

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Seq.Common { (@+), seq_replace }
open Kuiper.Spec.GEMM
open Kuiper.Array.Vectorized
open Kuiper.Tensor.Layout.Alg { l2_row_major, c_l2_row_major }
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor
open Kuiper.Seq.Common { op_At_Bang }

open Kuiper.Sparse.Tensor
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.Array
open Kuiper.Sparse.Common
open Kuiper.Sparse.SPMM.Defs { chest2_tile_prop }
open Kuiper.Sparse.FMA

module Math = Kuiper.Sparse.Math

(* Auxiliar lemmas for chests and sequences *)

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
  (y : larray et n)
  (#vy : erased (lseq et n))
  (k : et)
  (#lx : layout1 n) {| ctlayout lx |}
  (x : array1 et lx)
  (#fx : perm)
  (#vx : chest1 et n)
  preserves gpu
  preserves x |-> Frac fx vx
  requires  y |-> vy
  ensures   y |-> seq_scalar_prod vy k (chest1_to_seq vx)
{
  let mut ix : sz = 0sz;

  while (!ix <^ n)
    invariant exists* vix (vy' : lseq et n).
      ix |-> vix **
      y  |-> vy' **
      pure (
        vix <= n /\
        (forall (i : natlt n { i < vix }).
          // acc1 vy' i == seq_scalar_prod (chest1_to_seq vy) k (chest1_to_seq vx) @! i) /\
          vy' @! i == seq_scalar_prod vy k (chest1_to_seq vx) @! i) /\
        forall (i : natlt n { i >= vix }).
          // acc1 vy' i == acc1 vy i
          vy' @! i == vy @! i
      )
      decreases (n - !ix)
  {
    let ixv = !ix;
    let cur =  Pulse.Lib.Array.(y.(ixv));
    let xv = tensor_read x (cidx1 (ixv <: szlt n));
    Pulse.Lib.Array.(y.(ixv) <- (cur `add` (k `mul` xv)));
    ix := !ix +^ 1sz;
  };

  with vy'. assert y |-> vy';
  assert pure (
    Seq.equal vy'
      (seq_scalar_prod vy k (chest1_to_seq vx))
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
  (y : larray et cols)
  (#vy : erased (lseq et cols))
  (x : larray et rows)
  (#fx : perm)
  (#vx : erased (lseq et rows))
  (#lm : layout2 rows cols) {| ctlayout lm |}
  (m : array2 et lm)
  (#fm : perm)
  (#vm : chest2 et rows cols)
  norewrite
  preserves gpu
  preserves x |-> Frac fx vx
  preserves m |-> Frac fm vm
  requires  y |-> vy
  ensures   y |-> seq_vmprod vy vx vm

{
  let mut k : sz = 0sz;
  while (!k <^ rows)
    invariant
      exists* vk (vy' : lseq et cols).
        k |-> vk **
        y |-> vy' **
        pure (
          vk <= rows /\
          forall (i : natlt cols).
            vy' @! i ==
            _dprod_acc (vy @! i) vx (ematrix_col vm i) vk
        )
    decreases (rows - !k)
  {
    open Pulse.Lib.Array;
    let kv = !k;
    let xk = x.(kv);
    tensor_extract_row_ro m (v kv);
    scalar_prod
        y xk
        #_ #(Kuiper.Tensor.Layout.Slice.ctlayout_slice _ 0 (v kv)) // should not be needed
        (tensor_row m (v kv));
    tensor_restore_row m (v kv);

    k := !k +^ 1sz;
  };

  with vy'. assert y |-> vy';
  assert pure (
    Seq.equal vy'
      (seq_vmprod vy vx vm)
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

let tile_mm_result
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#shared #cols : nat {  chunk et /? cols })
  (#ly : nat { chunk et /? ly })
  (y0 : erased (lseq et ly))
  (#lA : nat)
  (elems : erased (lseq et lA))
  (row_ind : erased (lseq nat lA) { in_bounds 0 shared row_ind })
  (eB : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (y : lseq et ly)
  : prop
= _tile_vmprod_prop y0 elems row_ind eB j step lA y

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
    tile_mm_result
      vy0
      (Seq.slice elems 0 to <: lseq et to) (Seq.slice row_ind 0 to)
      gem
      j step
      vy
  )
  (ensures
    tile_mm_result #_ #_ #_ #hvc
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
        ematrix_col_is_chest tem k1;
        sparse_dprod_accum
          (vy0 @! k1)
          elems row_ind
          (ematrix_col gem k2)
          to (to + m1)
      )
      else ()

let tile_mm_lemma0
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#shared #cols : nat {  chunk et /? cols })
  (#ly : nat { chunk et /? ly })
  (y0 : erased (lseq et ly))
  (eB : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
: Lemma (ensures tile_mm_result y0 (Seq.empty <: lseq et 0) Seq.empty eB j step y0)
= ()

let tile_mm_mask_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#shared #cols : nat { chunk et /? cols })
  (#ly : nat { chunk et /? ly })
  (y0 : lseq et ly)
  (nnz : nat)
  (mask_len : natle nnz)
  (elems : erased (lseq et (nnz - mask_len)))
  (row_ind : erased (lseq nat nnz) { in_bounds 0 shared row_ind })
  (eB : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (y : lseq et ly)
: Lemma
  (requires tile_mm_result y0 (Seq.create mask_len zero @+ elems) row_ind eB j step y)
  (ensures  tile_mm_result y0 elems (Seq.slice row_ind mask_len nnz) eB j step y)
=
  let mask : lseq et mask_len = Seq.create mask_len zero in
  let elems' : lseq et nnz = mask @+ elems in
  let row_ind' : lseq nat (nnz - mask_len) = Seq.slice row_ind mask_len nnz in
  introduce forall (k1 : natlt ly).
    tile_vmprod_cell_prop y0 elems row_ind' eB j step #() (nnz - mask_len) k1 y
  with (
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
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

let vmprod_is_tile_
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
  (k : natlt (tlen / chunk et))
  (#_: squash (offset_chunk et j k step < cols))
: Lemma
  (requires
    seq_unsparse _ _ elems row_ind == ematrix_row em1 i /\
    tile_mm_result
      (Seq.create tlen zero)
      elems row_ind
      em2
      j step
      tile
  )
  (ensures is_ematrix_tile_at (matmul em1 em2) i j tile step k)
=
  admit();
  let acc0 : lseq et tlen = Seq.create tlen zero in
  let mm = matmul em1 em2 in
  let s = seq_chunk tile (k * chunk et) in
  let t = ematrix_row_chunk mm i (offset_chunk et j k step) in
  introduce forall (x : natlt (chunk et #_ #solve)). s @! x == t @! x
  with (
    let k1 = k * chunk et + x in
    let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in

    // TODO
    assume k2 == offset_chunk et j k step + x;
    // TODO
    assume k2 < cols;

    calc (==) {
      s @! x;
      == {}
      tile @! k1;
      == { assert tile_vmprod_cell_prop acc0 elems row_ind em2 j step nnz k1 tile }
      sparse_dprod_acc zero elems row_ind (ematrix_col em2 k2);
      == { sparse_dprod_lemma elems row_ind (ematrix_col em2 k2) }
      dprod (ematrix_row em1 i) (ematrix_col em2 k2);
      == { dprod_is_matmul_single em1 em2 i k2 }
      matmul_single em1 em2 i k2;
      == {}
      acc2 mm i k2;
      == { assert k2 == offset_chunk et j k step + x }
      t @! x;
    }
  );
  assert Seq.equal s t

let tile_mm_result_lemma
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#rows #shared #cols : nat { chunk et /? cols })
  (eA : chest2 et rows shared)
  (i : natlt rows)
  (#nnz : nat)
  (elems : erased (lseq et nnz))
  (row_ind : erased (lseq nat nnz))
  (#_ : squash (in_bounds 0 shared row_ind /\ sorted row_ind))
  (eB : chest2 et shared cols)
  (j : nat { chunk et /? j })
  (step : nat)
  (#tlen : nat { chunk et /? tlen })
  (tile : lseq et tlen)
: Lemma
  (requires
    seq_unsparse _ _ elems row_ind == ematrix_row eA i /\
    tile_mm_result
      (Seq.create tlen zero)
      elems row_ind
      eB
      j step
      tile
  )
  (ensures is_ematrix_tile #_ #_ #solve (matmul eA eB) i j tile step)
=
  admit();
  let mm = matmul eA eB in
  introduce forall (k : natlt (tlen / chunk et)).
    offset_chunk et j k step < cols ==>
    is_ematrix_tile_at mm i j tile step k
  with (
    if offset_chunk et j k step < cols
      then vmprod_is_tile_ eA i elems row_ind eB j step tile k #()
      else ()
  )

open Kuiper.Sparse.LoadStore { array_cpy_chunk }

let chunk_end_bound (n k : nat) (ch : pos)
  : Lemma (requires ch /? n /\ ch /? k /\ k < n)
          (ensures k + ch <= n)
= lemma_divides_exact ch k;
  FStar.Math.Lemmas.swap_mul ch (k / ch);
  Math.block_lemma n ch (k / ch)

inline_for_extraction noextract
fn load_vmprod_chunk
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : sz { chunk et /? n1 })
  (y : larray et n1)
  (#vy : erased (lseq et n1))
  (x : et)
  (#n2 : sz { chunk et /? n2 })
  (#lrow : layout1 n2) {| ctlayout lrow, clrow : cont_layout lrow |}
  (row : array1 et lrow)
  (#frow : perm)
  (#vrow : chest1 et n2)
  (k1 : sz { k1 + chunk et <= n1 })
  (k2 : sz { chunk et /? k2 })
  (bounds_checked : bool)
  preserves gpu
  preserves row |-> Frac frow vrow
  requires  pure (aligned 16 (core row))
  requires  pure (aligned_cont_layout (chunk et) clrow)
  requires  pure (bounds_checked ==> k2 < n2)
  requires  y |-> vy
  ensures   y |-> seq_fma' (chunk et) x (chest1_to_seq vrow) vy k1 k2
{
  if (bounds_checked || k2 <^ n2)
  {
    let mut lchunk = [| zero #et #_; chunk et |];
    assume pure (aligned 16 lchunk);
    rewrite (row |-> Frac frow vrow)
         as (row |-> Frac frow (seq_to_chest1 (chest1_to_seq vrow)));
    chunk_end_bound n2 k2 (chunk et);
    array_cpy_chunk lchunk #_ #lrow #_ #clrow row k2;
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
      Math.lineal_divides ch j ch ((k - 1) * step);
      seq_fma'
        ch x row
        (seq_load_vmprod_row y x row j step (k - 1))
        ((k - 1) * ch)
        (j + (k - 1) * step * ch)
    )

(* Keep the divisibility proof in a small pure context.  Reconstructing this
   refinement while elaborating the [SizeT] expression below is needlessly
   expensive once the scalar/sized dictionaries carry canonicality proofs. *)
#push-options "--z3rlimit 20"
let lemma_divides_vmprod_offset
  (et : Type0) {| sized et, has_vec_cpy et |}
  (j k step : sz { fits (j + k * step * chunk et) })
  : Lemma
      (requires chunk et /? j)
      (ensures chunk et /? (j +^ k *^ step *^ chunk et))
  = lemma_divides_product (chunk et) (k * step);
    lemma_divides_sum (chunk et) j (k * step * chunk et)
#pop-options


inline_for_extraction noextract
fn load_vmprod_row
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : sz { chunk et /? n1 })
  (y : larray et n1)
  (#vy : erased (lseq et n1))
  (x : et)
  (#n2 : sz { chunk et /? n2 })
  (#lrow : layout1 n2) {| ctlayout lrow, clrow : cont_layout lrow |}
  (row : array1 et lrow)
  (#frow : perm)
  (#vrow : chest1 et n2)
  (j : sz { chunk et /? j })
  (step : sz)
  (bounds_checked : bool)
  preserves gpu
  preserves row |-> Frac frow vrow
  requires  pure (aligned 16 (core row))
  requires  pure (aligned_cont_layout (chunk et) clrow)
  requires  pure (fits (j + n1 * step))
  requires  pure (bounds_checked ==> n1 == chunk et /\ j < n2)
  requires  y |-> vy
  ensures   y |-> seq_load_vmprod_row vy x (chest1_to_seq vrow) j step (n1 / chunk et)
{
  let mut k : sz = 0sz;

  while (!k <^ n1 /^ chunk et)
    invariant exists* vk (vy' : lseq et n1).
      k |-> vk **
      y |-> vy' **
      pure (
        vk <= n1 / chunk et /\
        Seq.equal vy' (seq_load_vmprod_row vy x (chest1_to_seq vrow) j step vk)
      )
    decreases (n1 /^ chunk et - !k)
  {
    assert pure (fits (j + !k * step * chunk et));
    lemma_divides_vmprod_offset et j !k step;
    assert pure (chunk et /? (j +^ !k *^ step *^ chunk et));
    load_vmprod_chunk
      y x
      row
      (!k *^ chunk et) (j +^ !k *^ step *^ chunk et)
      bounds_checked;
    k := !k +^ 1sz;
  };
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



(* A thread starting beyond the matrix width has no active vector chunks.
   These lemmas justify skipping its reduction without changing the spec. *)
noextract
let rec seq_load_vmprod_row_outside
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (y : lseq et n1)
  (x : et)
  (#n2 : nat { chunk et /? n2 })
  (row : lseq et n2)
  (j : nat { chunk et /? j /\ n2 <= j })
  (step : nat)
  (k : natle (n1 / chunk et))
  : Lemma (seq_load_vmprod_row y x row j step k == y)
  = if k > 0 then seq_load_vmprod_row_outside y x row j step (k - 1)

noextract
let rec seq_load_vmprod_outside
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (y : lseq et n1)
  (elems : lseq et m1)
  (row_ind : lseq nat m1)
  (#m2 #n2 : pos { chunk et /? n2 })
  (em : chest2 et m2 n2)
  (j : nat { chunk et /? j /\ n2 <= j })
  (step : nat)
  (#_ : squash (in_bounds 0 m2 row_ind))
  (to : natle m1)
  : Lemma (seq_load_vmprod y elems row_ind em j step to == y)
  = if to > 0 then (
      seq_load_vmprod_outside y elems row_ind em j step (to - 1);
      seq_load_vmprod_row_outside y (elems @! to - 1)
        (ematrix_row em (row_ind @! to - 1)) j step (n1 / chunk et)
    )


open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }

inline_for_extraction noextract
fn tile_mm_
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#m1 #n1 : sz { chunk et /? n1 })
  (y : larray et n1)
  (#vy : erased (lseq et n1))
  (elems : larray et m1)
  (row_ind : larray sz m1)
  (#fx : perm)
  (#m2 #n2 : szp { chunk et /? n2 })
  (#lm : layout2 m2 n2) {| ctlayout lm, srm : strided_row_major lm |}
  (m : array2 et lm)
  (#fm : perm)
  (#em : chest2 et m2 n2)
  (j : sz { chunk et /? j })
  (step : sz)
  (to : szle m1)
  (#velems : erased (lseq et to))
  (#vrow_ind : erased (lseq sz to))
  (#_ : squash (in_bounds 0 m2 (cast_pos vrow_ind)))
  preserves gpu
  preserves pts_to_slice elems   #fx 0 to velems
  preserves pts_to_slice row_ind #fx 0 to vrow_ind
  preserves m |-> Frac fm em
  requires  pure (aligned 16 (core m) /\ aligned_strided_row_major (chunk et) srm)
  requires  pure (fits (j + n1 * step))
  requires  y |-> vy
  ensures   y |-> seq_load_vmprod vy velems (cast_pos vrow_ind) em j step to
{
  // All vector chunks owned by this thread start at or after j. Threads
  // outside the output can skip the reduction, but must still participate
  // in the caller's shared-memory loads and barriers.
  if (j <^ n2) {
    // With one vector chunk per thread, the outer check proves every load
    // in bounds. The static flag removes the check from the reduction loop.
    let mut k : sz = 0sz;

    while (!k <^ to)
      // invariant exists* vk (vy' : chest1 et n1).
      invariant exists* vk (vy' : lseq et n1).
        k |-> vk **
        y |-> vy' **
        pure (
          vk <= to /\
          Seq.equal vy' (seq_load_vmprod vy velems (cast_pos vrow_ind) em j step vk)
        )
      decreases (to - !k)
    {
      let kv = !k;
      let kr = slice_read row_ind kv;
      let kx = slice_read elems kv;

      // [kr] indexes a valid row of [m] by the sparsity bound.
      assert pure (v kr == cast_pos vrow_ind @! kv);

      tensor_extract_row_ro m (v kr);
      aligned_cont_strided_row_major lm (chunk et) kr;

      load_vmprod_row
        y kx
        #_ #_ #(ctlayout_slice _ 0 (v kr)) // should not be needed
        (tensor_row m (v kr)) j step (FStar.SizeT.eq n1 (chunk et));

      tensor_restore_row m (v kr);

      k := !k +^ 1sz;
    }
  } else {
    seq_load_vmprod_outside vy velems (cast_pos vrow_ind) em j step to;
    assert pure (seq_load_vmprod vy velems (cast_pos vrow_ind) em j step to == vy);
    rewrite (y |-> vy)
         as (y |-> seq_load_vmprod vy velems (cast_pos vrow_ind) em j step to);
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
  Math.lineal_divides (chunk et) j (chunk et) (ik * step);
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
    Math.lineal_divides (chunk et) j (chunk et) ((k - 1) * step);
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
    Math.lineal_divides (chunk et) j (chunk et) ((k - 1) * step);
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
    tile_mm_result y
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
    tile_mm_result
      y0
      (Seq.slice elems 0 to <: lseq et to) (Seq.slice row_ind 0 to)
      em
      j step
      y
  )
  (ensures
    tile_mm_result #_ #_ #_ #solve
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

  assert tile_mm_result y elems2 row_ind2 em j step y';

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
fn tile_mm
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (#shared #cols : szp { chunk et /? cols } )
  (#ly : sz { chunk et /? ly })
  (y : larray et ly)
  (vy0 : erased (lseq et ly))
  (#vy : erased (lseq et ly))
  (#lA : sz) (elems : larray et lA) (row_ind : larray sz lA) (#fA : perm)
  (#nnz : erased nat)
  (velems : erased (lseq et nnz))
  (vrow_ind : erased (lseq sz nnz) { in_bounds 0 shared (cast_pos vrow_ind) })
  (#lB : layout2 shared cols) {| ctlayout lB, srm : strided_row_major lB |}
  (mB : array2 et lB) (#fB : perm) (#eB : chest2 et shared cols)
  (j : sz { chunk et /? j })
  (step : sz)
  (from to : erased nat { to <= nnz })
  (cant : szle lA { v cant == to - from })
  preserves gpu
  preserves pts_to_slice elems   #fA 0 cant (Seq.slice velems from to <: lseq et cant)
  preserves pts_to_slice row_ind #fA 0 cant (Seq.slice vrow_ind from to <: lseq sz cant)
  preserves mB |-> Frac fB eB
  requires  pure (aligned 16 (core mB) /\ aligned_strided_row_major (chunk et) srm)
  requires  pure (fits (j + ly * step))
  requires
    y |-> vy **
    pure (
      tile_mm_result
        vy0
        (Seq.slice velems 0 from <: lseq et from)
        (Seq.slice (cast_pos vrow_ind) 0 from)
        eB
        j step
        vy
    )
  ensures exists* (vy' : lseq et ly).
    y |-> vy' **
    pure (
      tile_mm_result
        vy0
        (Seq.slice velems 0 to <: lseq et to)
        (Seq.slice (cast_pos vrow_ind) 0 to <: lseq nat to)
        eB j step vy'
    )
{
  tile_mm_ y elems row_ind mB j step cant;
  seq_load_vmprod_step_lemma cant vy0 velems (cast_pos vrow_ind) from (v cant) eB j step vy;

  assert pure (
    Seq.equal
      (cast_pos #cant (Seq.slice vrow_ind from to))
      (Seq.slice (cast_pos vrow_ind) from to)
  );
}
