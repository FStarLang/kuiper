module Kuiper.Sparse.SPMM.LoadDense

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Seq.Common { seq_blit, seq_replace }
open Kuiper.Sparse
open Kuiper.Sparse.Load
open Kuiper.Sparse.SPMM.Defs { chest2_tile_prop }
open Kuiper.Array.Vectorized
open Kuiper.Tensor
open Kuiper.Array2.Vectorized


// let rec seq_tile_cpy
//   (#et : Type0) {| sized et, has_vec_cpy et |}
//   (#n1 : nat { chunk et /? n1 })
//   (s1 : lseq et n1)
//   (#n2 : nat { chunk et /? n2 })
//   (s2 : lseq et n2)
//   (j2 : nat { chunk et /? j2 })
//   (step : nat)
//   (to : natle (n1 / chunk et))
// : GTot (lseq et n1) (decreases to)
// =
//   if to = 0 then s1 else (
//     lemma_divides_product (chunk et) (to - 1);
//     lemma_divides_product (chunk et) ((to - 1) * step);
//     lemma_divides_sum (chunk et) j2 ((to - 1) * step * chunk et);
//     (seq_blit'
//       (seq_tile_cpy s1 s2 j2 step (to - 1))
//       ((to - 1) * chunk et)
//       s2
//       (j2 + (to - 1) * step * chunk et)
//       (chunk et))
//   )

let rec chest1_tile_cpy
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (s1 : chest1 et n1)
  (#n2 : nat { chunk et /? n2 })
  (s2 : chest1 et n2)
  (j2 : nat { chunk et /? j2 })
  (step : nat)
  (to : natle (n1 / chunk et))
: GTot (chest1 et n1) (decreases to)
=
  if to = 0 then s1 else (
    lemma_divides_product (chunk et) (to - 1);
    lemma_divides_product (chunk et) ((to - 1) * step);
    lemma_divides_sum (chunk et) j2 ((to - 1) * step * chunk et);
    (chest1_blit'
      (chest1_tile_cpy s1 s2 j2 step (to - 1))
      ((to - 1) * chunk et)
      s2
      (j2 + (to - 1) * step * chunk et)
      (chunk et))
  )

let lemma_fits_tile_offset
  (et:Type0) {| sized et, has_vec_cpy et |}
  (n : nat)
  (j : nat)
  (k : natlt (n / chunk et))
  (step : nat)
: Lemma
  (requires fits (j + n * step))
  (ensures fits (j + k * step * chunk et))
= ()

let lemma_divides_tile_offset
  (et:Type0) {| sized et, has_vec_cpy et |}
  (j k step : sz { fits (j + k * step * chunk et) })
: Lemma
  (requires chunk et /? j)
  (ensures chunk et /? (j +^ k *^ step *^ chunk et))
=
  lemma_divides_product (chunk et) (k * step);
  lemma_divides_sum (chunk et) j (k * step * chunk et)

inline_for_extraction noextract
fn tile_vec_cpy
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#n1 : sz { chunk et /? n1})
  (#l1 : layout1 n1) {| ctlayout l1, cl1 : cont_layout l1 |}
  (x1 : array1 et l1)
  (#s1 : chest1 et n1)
  (#n2 : szp { chunk et /? n2})
  (#l2 : layout1 n2) {| ctlayout l2, cl2 : cont_layout l2 |}
  (x2 : array1 et l2)
  (#f : perm)
  (#s2 : chest1 et n2)
  (j : sz { chunk et /? j })
  (step : szp)
  preserves gpu
  requires  x1 |-> s1
  requires  pure (aligned 16 (core x1))
  requires  pure (aligned_cont_layout (chunk et) cl1)
  preserves x2 |-> Frac f s2
  requires  pure (aligned 16 (core x2))
  requires  pure (fits (j + n1 * step))
  requires  pure (aligned_cont_layout (chunk et) cl2)
  ensures   x1 |-> chest1_tile_cpy s1 s2 j step (n1 / chunk et)
{
  let mut k : sz = 0sz;

  while (!k <^ n1 /^ chunk et)
    invariant exists* (vk : szle (n1 / chunk et)).
      k  |-> (vk <: sz) **
      x1 |-> chest1_tile_cpy s1 s2 j step vk
  {
    lemma_fits_tile_offset et n1 j !k step;
    lemma_divides_tile_offset et j !k step;
    assert pure (fits (j + !k * step * chunk et));
    array_vec_cpy' x1 (!k *^ chunk et) x2 (j +^ !k *^ step *^ chunk et);
    k := !k +^ 1sz;

  }
}


// let ematrix_tile_cpy
//   (#et : Type0) {| sized et, has_vec_cpy et |}
//   (#m1 #n1 : nat { chunk et /? n1 })
//   (em1 : chest2 et m1 n1)
//   (#m2 #n2 : nat {  chunk et /? n2 })
//   (em2 : chest2 et m2 n2)
//   (row_ind : lseq nat m1 { in_bounds 0 m2 row_ind })
//   (j : nat { chunk et /? j })
//   (step : nat) // tratemos de no usar tid
// : GTot (chest2 et m1 n1)
// = ematrix_from_rows fun i ->
//     seq_tile_cpy
//       (ematrix_row em1 i) (ematrix_row em2 (Seq.index row_ind i))
//       j step (n1 / chunk et)


let chest2_tile_cpy
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (em1 : chest2 et m1 n1)
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (row_ind : lseq nat m1 { in_bounds 0 m2 row_ind })
  (j : nat { chunk et /? j })
  (step : nat) // tratemos de no usar tid
: GTot (chest2 et m1 n1)
= chest2_from_rows fun i ->
    chest1_tile_cpy
      (chest2_row em1 i) (chest2_row em2 (Seq.index row_ind i))
      j step (n1 / chunk et)


open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }


inline_for_extraction noextract
fn matrix_tile_vec_cpy
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : szp { chunk et /? n1 })
  (#l1 : layout2 m1 n1) {| ctlayout l1, srm1 : strided_row_major l1 |}
  (a1 : array2 et l1)
  (#em1 : chest2 et m1 n1)
  (#m2 #n2 : szp { chunk et /? n2 })
  (#l2 : layout2 m2 n2) {| ctlayout l2, srm2 : strided_row_major l2 |}
  (a2 : array2 et l2)
  (#f : perm)
  (#em2 : chest2 et m2 n2)
  (j : sz { chunk et /? j })
  (step : szp)
  (row_ind : larray sz m1)
  (#fr : perm)
  (#vrow_ind : lseq sz m1)
  (#_: squash (in_bounds 0 m2 (cast_pos vrow_ind)))
  preserves gpu
  requires  a1 |-> em1
  requires  pure (aligned 16 (core a1) /\ aligned_strided_row_major (chunk et) srm1)
  preserves a2 |-> Frac f em2
  requires  pure (aligned 16 (core a2) /\ aligned_strided_row_major (chunk et) srm2)
  requires  pure (fits (j + n1 * step))
  preserves row_ind |-> Frac fr vrow_ind
  ensures   a1 |-> chest2_tile_cpy em1 em2 (cast_pos vrow_ind) j step
{
  let mut k : sz = 0sz;

  while (!k <^ m1)
    invariant exists* vk em.
      k  |-> vk **
      a1 |-> em **
      pure (
        vk <= m1 /\
        (forall (i : natlt m1 { i < vk }).
          chest2_row em i ==
          chest1_tile_cpy
            (chest2_row em1 i)
            (chest2_row em2 (Seq.index vrow_ind i))
            j step (n1 / chunk et)
        ) /\
        forall (i : natlt m1 { vk <= i }).
          chest2_row em i == chest2_row em1 i
      )
  {
    with em. assert a1 |-> em;

    tensor_extract_row a1 !k;
    let ri = Pulse.Lib.Array.(row_ind.(!k));
    tensor_extract_row_ro a2 ri; 
    
    row_core_lemma a1 !k;
    aligned_cont_strided_row_major l1 (chunk et) !k;
    row_core_lemma a2 ri;
    aligned_cont_strided_row_major l2 (chunk et) ri;

    assert tensor_row a1 (v !k) |-> chest2_row em1 !k;
    tile_vec_cpy
      #_ #_ #_ #n1 #_
      #(ctlayout_slice _ 0 (v !k)) // should not be needed
      (tensor_row a1 (v !k))
      #_ #_ #_
      #(Kuiper.Tensor.ctlayout_slice _ 0 (v ri)) // should not be needed
      (tensor_row a2 (v ri)) j step;
    
    tensor_restore_row a2 ri;

    Pulse.Lib.Forall.elim_forall 
      (chest1_tile_cpy
        (chest2_row em1 !k) (chest2_row em2 ri)
        j step (n1 / chunk et));
    Pulse.Lib.Trade.elim_trade _ _;
    

    chest2_upd_row_lemma em !k 
      (chest1_tile_cpy
        (chest2_row em1 !k) (chest2_row em2 ri)
        j step (n1 / chunk et));
    
    k := !k +^ 1sz;
  };

  with em. assert a1 |-> em;
  assert pure (
    chest2_rows_equal em (chest2_tile_cpy em1 em2 (cast_pos vrow_ind) j step)
  );
}

let lem1
  (d : pos) (a b : nat)
: Lemma
  (requires b * d <= a /\ a < (b + 1) * d)
  (ensures a / d = b)
= ()

let lem2
  (d : pos) (a b : nat) (c : pos)
: Lemma
  (requires a < b * d)
  (ensures a / d * c * d + a % d < b * c * d)
= ()

let chest1_blit_lemma1
  (#a:Type)
  (#n1 : nat)
  (s1 : chest1 a n1) (off1 : natlt n1)
  (#n2 : nat)
  (s2 : chest1 a n2) (off2 : nat)
  (cnt : nat { cnt /? n1 /\ cnt /? off1 /\ cnt /? n2 /\ cnt /? off2 } )
  (k : natlt n1)
: Lemma
  (requires k < off1 \/ k >= off1 + cnt)
  (ensures acc1 (chest1_blit' s1 off1 s2 off2 cnt) k == acc1 s1 k)
=
  lemma_divides_leq cnt n1 off1;
  lemma_divides_leq cnt n2 off2

#push-options "--split_queries always"
let chest1_blit_lemma2
  (#a:Type)
  (#n1 : nat)
  (s1 : chest1 a n1) (off1 : natlt n1)
  (#n2 : nat)
  (s2 : chest1 a n2) (off2 : nat)
  (cnt : nat { cnt /? n1 /\ cnt /? off1 /\ cnt /? n2 /\ cnt /? off2 } )
  (k : natlt n1)
: Lemma
  (requires off1 <= k /\ k < off1 + cnt /\ off2 < n2 /\ off2 + k - off1 < n2)
  (ensures
    acc1 (chest1_blit' s1 off1 s2 off2 cnt) k == acc1 s2 (off2 + k - off1)
  )
=
  lemma_divides_leq cnt n1 off1;
  lemma_divides_leq cnt n2 off2;
  ()

let rec chest1_tile_cpy_col_lemma1
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (s1 : chest1 et n1)
  (#n2 : nat { chunk et /? n2 })
  (s2 : chest1 et n2)
  (j2 : nat { chunk et /? j2 })
  (step : pos)
  (to : natle (n1 / chunk et))
  (k : natlt n1 { k < to * chunk et })
: Lemma
  (requires j2 + k / chunk et * step * chunk et + k % chunk et < n2)
  (ensures
    acc1 (chest1_tile_cpy s1 s2 j2 step to) k ==
    acc1 s2 (j2 + k / chunk et * step * chunk et + k % chunk et)
  )
= 
  lemma_divides_product (chunk et) (to - 1);
  lemma_divides_product (chunk et) ((to - 1) * step);
  lemma_divides_sum (chunk et) j2 ((to - 1) * step * chunk et);
  let k1 = j2 + k / chunk et * step * chunk et in
  let k2 = k % chunk et in
  if k >= (to - 1) * chunk et
    then (
      lem1 (chunk et) k (to - 1);
      assert k1 = j2 + (to - 1) * step * chunk et;
      assert k2 = k - (to - 1) * chunk et;

      lem2 (chunk et) k to step;
      assert k1 + k2 < j2 + to * step * chunk et;

      chest1_blit_lemma2
        (chest1_tile_cpy s1 s2 j2 step (to -1))
        ((to - 1) * chunk et)
        s2
        (j2 + (to - 1) * step * chunk et)
        (chunk et) k
    )
    else (
      lem2 (chunk et) k (to - 1) step;
      assert k1 + k2 < j2 + (to - 1) * step * chunk et;

      chest1_tile_cpy_col_lemma1 s1 s2 j2 step (to - 1) k;
      chest1_blit_lemma1
        (chest1_tile_cpy s1 s2 j2 step (to -1))
        ((to - 1) * chunk et)
        s2
        (j2 + (to - 1) * step * chunk et)
        (chunk et) k
    )
#pop-options

let rec chest1_tile_cpy_col_lemma2
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n1 : nat { chunk et /? n1 })
  (s1 : chest1 et n1)
  (#n2 : nat { chunk et /? n2 })
  (s2 : chest1 et n2)
  (j2 : nat { chunk et /? j2 })
  (step : pos)
  (to : natle (n1 / chunk et))
  (k : natlt n1)
: Lemma
  (requires j2 + k / chunk et * step * chunk et >= n2)
  (ensures acc1 (chest1_tile_cpy s1 s2 j2 step to) k == acc1 s1 k)
= 
  if to = 0 then () else (
    chest1_tile_cpy_col_lemma2 s1 s2 j2 step (to - 1) k;

    lemma_divides_product (chunk et) (to - 1);
    lemma_divides_product (chunk et) ((to - 1) * step);
    lemma_divides_sum (chunk et) j2 ((to - 1) * step * chunk et);
    let k1 = j2 + k / chunk et * step * chunk et in
    let k2 = k % chunk et in

    if (to - 1) * chunk et <= k && k < to * chunk et then (
      lem1 (chunk et) k (to - 1);
      assert k1 = j2 + (to - 1) * step * chunk et;
      assert k2 = k - (to - 1) * chunk et
    )
    else
      chest1_blit_lemma1
        (chest1_tile_cpy s1 s2 j2 step (to - 1))
        ((to - 1) * chunk et)
        s2
        (j2 + (to - 1) * step * chunk et)
        (chunk et) k
  )

// let chest1_make_sparse
//   (#et : Type0)
//   (#nnz #n : nat)
//   (pos : lseq nat nnz{in_bounds 0 n pos})
//   (s : lseq et n)
//   : chest1 et nnz
// =
//   Seq.init nnz (fun i -> s @! (pos @! i))

let chest2_tile_col
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (em1 : chest2 et m1 n1)
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (row_ind : lseq nat m1 { in_bounds 0 m2 row_ind })
  (j : nat { chunk et /? j })
  (step : nat)
  (k1 : natlt n1)
: GTot (chest1 et m1)
=
  let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in
  if k2 < n2 
    // mm defino chest1_make_sparse??
    then seq_to_chest1 (seq_make_sparse row_ind (ematrix_col em2 k2))
    else chest2_col em1 k1


#push-options "--split_queries always"
let chest2_tile_col_lemma_
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (em1 : chest2 et m1 n1)
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (row_ind : lseq nat m1 { in_bounds 0 m2 row_ind })
  (j : nat { chunk et /? j })
  (step : pos)
  (k1 : natlt n1)
: Lemma
  (requires true)
  (ensures
    chest2_col (chest2_tile_cpy em1 em2 row_ind j step) k1 ==
    chest2_tile_col em1 em2 row_ind j step k1
  )
=
  let c1 = ematrix_col (chest2_tile_cpy em1 em2 row_ind j step) k1 in
  let c2 = chest2_tile_col em1 em2 row_ind j step k1 in
  let k2 = j + k1 / chunk et * step * chunk et + k1 % chunk et in

  introduce forall i.
    acc1 (chest2_col (chest2_tile_cpy em1 em2 row_ind j step) k1) i ==
    acc1 (chest2_tile_col em1 em2 row_ind j step k1) i
  with (
    if k2 < n2
      then (
        chest1_tile_cpy_col_lemma1
          (chest2_row em1 i)
          (chest2_row em2 (row_ind `Seq.index` i))
          j
          step
          (n1 / chunk et)
          k1
      )
      else (
        chest1_tile_cpy_col_lemma2
          (chest2_row em1 i)
          (chest2_row em2 (row_ind `Seq.index` i))
          j
          step
          (n1 / chunk et)
          k1
      )
  );
  assert equal
    (chest2_col (chest2_tile_cpy em1 em2 row_ind j step) k1)
    (chest2_tile_col em1 em2 row_ind j step k1)
#pop-options

let chest2_tile_col_lemma
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (em1 : chest2 et m1 n1)
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (row_ind : lseq nat m1 { in_bounds 0 m2 row_ind })
  (j : nat { chunk et /? j })
  (step : pos)
: Lemma
  (requires true)
  (ensures
    chest2_tile_cpy em1 em2 row_ind j step ==
    chest2_from_cols (chest2_tile_col em1 em2 row_ind j step)
  )
=
  introduce forall k.
    chest2_col (chest2_tile_cpy em1 em2 row_ind j step) k ==
    chest2_tile_col em1 em2 row_ind j step k
  with chest2_tile_col_lemma_ em1 em2 row_ind j step k;
  assert chest2_cols_equal 
    (chest2_tile_cpy em1 em2 row_ind j step)
    (chest2_from_cols (chest2_tile_col em1 em2 row_ind j step))

// podriamos probar solo esta spec parcial
// aunque la prueba seria bastante similar
let chest2_tile_lemma
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : nat { chunk et /? n1 })
  (em1 : chest2 et m1 n1)
  (#m2 #n2 : nat {  chunk et /? n2 })
  (em2 : chest2 et m2 n2)
  (row_ind : lseq nat m1 { in_bounds 0 m2 row_ind })
  (j : nat { chunk et /? j })
  (step : pos)
: Lemma
  (requires true)
  (ensures
    chest2_tile_prop
      em2 row_ind j step
      (chest2_from_cols (chest2_tile_col em1 em2 row_ind j step))
  )
= ()


inline_for_extraction noextract
fn load_dense_matrix
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#m1 #n1 : szp { chunk et /? n1 })
  (l1 : layout2 m1 n1) {| ctlayout l1, srm1 : strided_row_major l1 |}
  (a1 : array2 et l1)
  (em1 : chest2 et m1 n1)
  (#m2 #n2 : szp { chunk et /? n2 })
  (l2 : layout2 m2 n2) {| ctlayout l2, srm2 : strided_row_major l2 |}
  (a2 : array2 et l2)
  (#f : perm)
  (em2 : chest2 et m2 n2)
  (j : sz { chunk et /? j })
  (step : szp)
  (row_ind : larray sz m1)
  (#fr : perm)
  (#vrow_ind : lseq sz m1)
  (#_: squash (in_bounds 0 m2 (cast_pos vrow_ind)))
  preserves gpu
  requires  a1 |-> em1
  requires  pure (aligned 16 (core a1) /\ aligned_strided_row_major (chunk et) srm1)
  preserves a2 |-> Frac f em2
  requires  pure (aligned 16 (core a2) /\ aligned_strided_row_major (chunk et) srm2)
  requires  pure (fits (j + n1 * step))
  preserves row_ind |-> Frac fr vrow_ind
  ensures exists* em1'.
    a1 |-> em1' **
    pure (chest2_tile_prop em2 (cast_pos vrow_ind) j step em1')
{

  matrix_tile_vec_cpy a1 a2 j step row_ind;
  chest2_tile_col_lemma em1 em2 (cast_pos vrow_ind) j step;
  chest2_tile_lemma em1 em2 (cast_pos vrow_ind) j step;
}