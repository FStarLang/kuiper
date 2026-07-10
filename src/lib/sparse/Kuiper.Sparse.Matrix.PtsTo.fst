module Kuiper.Sparse.Matrix.PtsTo

#lang-pulse

open Kuiper
open FStar.Tactics.V2 { exact }
open Kuiper.EMatrix
open Kuiper.Array.Vectorized
open Kuiper.Sparse.Common
open Kuiper.Array2 { array2, layout }

module M = Kuiper.Array2
module T = Kuiper.Tensor
open Kuiper.Array2.Strided { strided_row_major }

(* SL props sobre cells *)

let matrix_live_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : M.layout rows cols)
  (gm : array2 et lm)
  (i : natlt rows)
  (j : natlt cols)
: slprop
= exists* v. M.pts_to_cell gm (i, j) v

let matrix_pts_to_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
  (v : lseq et (chunk et))
: slprop
= forall+ (k : natlt (chunk et)). M.pts_to_cell gm (i, j + k) (v @! k)

let matrix_live_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
: slprop
= exists* v. matrix_pts_to_vec gm i j v

let seq_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  (s : lseq et n)
  (k : nat { k + chunk et <= n })
: GTot (lseq et (chunk et))
= Seq.slice s k (k + chunk et)

unfold
let matrix_pts_to_vec_slice
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
  (#n : nat)
  (v : lseq et n)
  (k : nat { k + chunk et <= n })
: slprop
// = forall+ (x : natlt (chunk et)). M.pts_to_cell gm (i, j + x) (v @! k + x)
= matrix_pts_to_vec gm i j (seq_chunk v k)

let matrix_pts_to_cell_in_bounds
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : M.layout rows cols)
  (gm : array2 et lm)
  (i : natlt rows)
  (j : nat)
  (v : et)
: slprop
= when__ (j < cols) (fun _ -> M.pts_to_cell gm (i, j) v)

let matrix_live_cell_in_bounds
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : M.layout rows cols)
  (gm : array2 et lm)
  (i : natlt rows)
  (j : nat)
: slprop
= exists* v. matrix_pts_to_cell_in_bounds gm i j v

let matrix_pts_to_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  // (v : (squash (j < cols)) -> GTot (lseq et (chunk et)))
  (v : lseq et (chunk et))
: slprop
=
  when__ (j < cols) (fun _ -> matrix_pts_to_vec gm i j v)

ghost
fn matrix_pts_to_vec_in_bounds_equiv
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v w : lseq et (chunk et))
  requires matrix_pts_to_vec_in_bounds gm i j v
  requires pure (v == w)
  ensures matrix_pts_to_vec_in_bounds gm i j w
{
  unfold matrix_pts_to_vec_in_bounds gm;
  if (j < cols)
  {
    when__elim_true _ _;
    rewrite each v as w;
    when__intro_true (j < cols) (matrix_pts_to_vec gm i j w);
    fold matrix_pts_to_vec_in_bounds gm i j w;
  }
  else
  {
    when__elim_false _ _;
    when__intro_false (j < cols) (fun _ -> matrix_pts_to_vec gm i j w);
    fold matrix_pts_to_vec_in_bounds gm i j w;
  }

}


let matrix_live_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
: slprop
= exists* v. matrix_pts_to_vec_in_bounds gm i j v

unfold
let matrix_pts_to_vec_slice_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#n : nat)
  (v : lseq et n)
  (k : nat { k + chunk et <= n })
: slprop
= matrix_pts_to_vec_in_bounds gm i j (seq_chunk v k)

(* fold/unfold *)

ghost
fn fold_matrix_pts_to_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
  (#_ : squash (j < cols))
  requires matrix_pts_to_vec gm i j v
  ensures matrix_pts_to_vec_in_bounds gm i j v
{
  when__intro_true (j < cols) (matrix_pts_to_vec gm i j v);
  fold matrix_pts_to_vec_in_bounds gm i j v;
}

ghost
fn fold_matrix_pts_to_vec_slice_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#n : nat)
  (v : lseq et n)
  (k : nat { k + chunk et <= n })
  (#_ : squash (j < cols))
  requires matrix_pts_to_vec_slice gm i j v k
  ensures matrix_pts_to_vec_slice_in_bounds gm i j v k
{
  fold_matrix_pts_to_vec_in_bounds gm i j (seq_chunk v k)
}

ghost
fn fold_matrix_pts_to_vec_not_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
  requires pure (j >= cols)
  ensures matrix_pts_to_vec_in_bounds gm i j v
{
  when__intro_false (j < cols) (fun _ ->
    matrix_pts_to_vec gm i j v
  );
  fold matrix_pts_to_vec_in_bounds gm i j v;
}

ghost
fn fold_matrix_pts_to_vec_slice_not_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#n : nat)
  (v : lseq et n)
  (k : nat { k + chunk et <= n })
  requires pure (j >= cols)
  ensures matrix_pts_to_vec_slice_in_bounds gm i j v k
{
  fold_matrix_pts_to_vec_not_in_bounds gm i j (seq_chunk v k);
}

ghost
fn unfold_matrix_pts_to_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
  (#_ : squash (j < cols))
  requires matrix_pts_to_vec_in_bounds gm i j v
  ensures matrix_pts_to_vec gm i j v 
{
  unfold matrix_pts_to_vec_in_bounds gm i j v;
  when__elim_true _ _;
}

ghost
fn unfold_matrix_live_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#_ : squash (j < cols))
  requires matrix_live_vec_in_bounds gm i j
  ensures matrix_live_vec gm i j
{
  unfold matrix_live_vec_in_bounds gm;
  unfold_matrix_pts_to_vec_in_bounds gm i j _;
  fold matrix_live_vec gm i j;
}

ghost
fn unfold_matrix_pts_to_vec_not_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
  requires pure (j >= cols)
  requires matrix_pts_to_vec_in_bounds gm i j v
{
  unfold matrix_pts_to_vec_in_bounds gm i j v;
  when__elim_false _ _;
}

ghost
fn unfold_matrix_live_vec_not_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  requires pure (j >= cols)
  requires matrix_live_vec_in_bounds gm i j
{
  unfold matrix_live_vec_in_bounds gm; 
  unfold_matrix_pts_to_vec_not_in_bounds gm i j _;
}

(* thread owns in matrix *)

let offset_chunk
  (et : Type0) {| sized et, has_vec_cpy et |}
  (j : nat { chunk et /? j })
  (k : nat)
  (nthr : nat)
: Pure nat (requires true) (ensures divides (chunk et))
=
  lemma_divides_product (chunk et) (k * nthr);
  lemma_divides_sum (chunk et) j (k * nthr * chunk et);
  j + k * nthr * v (chunk et)

unfold
let thread_live_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (row_tile : nat)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
: slprop
= matrix_live_vec_in_bounds gm i (offset_chunk et j k nthr)

let thread_live_tile_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (row_tile : nat)
  (nthr : nat)
: slprop
=
  forall+ (k : natlt (row_tile / chunk et)).
    thread_live_vec gm i j row_tile nthr k

unfold
let thread_pts_to_vec_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
: slprop
=
    matrix_pts_to_vec_slice_in_bounds gm
      i (offset_chunk et j k nthr)
      s (k * chunk et)

let thread_pts_to_tile_vec_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
: slprop
=
  forall+ (k : natlt (row_tile / chunk et)).
    thread_pts_to_vec_underspec gm i j s nthr k
    // matrix_pts_to_cell_vec_in_bounds' gm
    //   i (offset_chunk et j k nthr tid)
    //   s (k * chunk et)

let ematrix_row_chunk_
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
: GTot (lseq et (chunk et))
= Seq.init_ghost (chunk et) (fun k -> macc em i (j + k)) 

let ematrix_row_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  // usamos pos y no nat porque garantiza que chunk et <= cols
  (#rows #cols : pos { chunk et /? cols })
  (em : ematrix et rows cols)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
: GTot (lseq et (chunk et))
= ematrix_row_chunk_ em i j

// TODO mover de acá
unfold
let matrix_pts_to_vec_in_matrix_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  // usamos pos y no nat porque garantiza que chunk et <= cols
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : ematrix et rows cols)
: slprop
=
  when__ (j < cols) (fun _ ->
    matrix_pts_to_vec gm i j (ematrix_row_chunk em i j)
  )

unfold
let thread_pts_to_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : ematrix et rows cols)
  (nthr : nat)
  (k : nat)
: slprop
=
    matrix_pts_to_vec_in_matrix_in_bounds gm
      i (offset_chunk et j k nthr) em



let is_ematrix_tile_at
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : ematrix et rows cols)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile }) 
  (s : lseq et row_tile)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
: Pure prop
  (requires offset_chunk et j k nthr < cols)
  (ensures fun _ -> true)
= 
  seq_chunk s (k * chunk et) ==
  ematrix_row_chunk em i (offset_chunk et j k nthr)

let is_ematrix_tile
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (em : ematrix et rows cols)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile }) 
  (s : lseq et row_tile)
  (nthr : nat)
: prop
=
  forall (k : natlt (row_tile / chunk et)).
    offset_chunk et j k nthr < cols ==>
      is_ematrix_tile_at em i j s nthr k

ghost
fn fold_thread_pts_to_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : ematrix et rows cols)
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
  requires thread_pts_to_vec_underspec gm i j s nthr k
  requires pure (is_ematrix_tile em i j s nthr)
  ensures thread_pts_to_vec gm i j em nthr k
{
  if (offset_chunk et j k nthr < cols)
  {
    assert pure (is_ematrix_tile_at em i j s nthr k);
    matrix_pts_to_vec_in_bounds_equiv gm
    i (offset_chunk et j k nthr)
      (seq_chunk s (k * chunk et))
      (ematrix_row_chunk em i (offset_chunk et j k nthr));
    unfold matrix_pts_to_vec_in_bounds gm;
  }
  else
  {
    unfold matrix_pts_to_vec_in_bounds gm;
    when__elim_false _ _;
    when__intro_false (offset_chunk et j k nthr< cols)
      (fun _ ->
        matrix_pts_to_vec gm
          i (offset_chunk et j k nthr)
          (ematrix_row_chunk em i (offset_chunk et j k nthr))
      );
  }
}

let thread_pts_to_tile_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : ematrix et rows cols)
  (row_tile : nat { chunk et /? row_tile })
  (nthr : nat)
: slprop
=
  forall+ (k : natlt (row_tile / chunk et)).
    thread_pts_to_vec gm i j em nthr k


ghost
fn fold_thread_pts_to_tile_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (em : ematrix et rows cols)
  (nthr : nat)
  requires thread_pts_to_tile_vec_underspec gm i j s nthr
  requires pure (is_ematrix_tile em i j s nthr)
  ensures thread_pts_to_tile_vec gm i j em row_tile nthr
{
  unfold thread_pts_to_tile_vec_underspec gm;
  forevery_map #(natlt (row_tile / chunk et))
    (fun k -> thread_pts_to_vec_underspec gm i j s nthr k)
    (fun k -> thread_pts_to_vec gm i j em nthr k)
    fn k {
      fold_thread_pts_to_vec gm i j em s nthr k
    };
  fold thread_pts_to_tile_vec gm i j em row_tile nthr;
}


unfold
let matrix_pts_to_cell_in_matrix_in_bounds
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat)
  (em : ematrix et rows cols)
: slprop
=
  when__ (j < cols) (fun _ ->
    M.pts_to_cell gm (i, j) (macc em i j)
  )

let gpu_pts_to_tile
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat)
  (em : ematrix et rows cols)
  (row_tile : nat)
: slprop
=
  forall+ (k : natlt row_tile).
    matrix_pts_to_cell_in_matrix_in_bounds gm i (j + k) em

ghost
fn unfold_matrix_pts_to_vec_in_matrix_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  // usamos pos y no nat porque garantiza que chunk et <= cols
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : ematrix et rows cols)
  requires matrix_pts_to_vec_in_matrix_in_bounds gm i j em
  ensures forall+ (x : natlt (chunk et)).
    matrix_pts_to_cell_in_matrix_in_bounds gm i (j + x) em

{
  if (j < cols)
  {
    when__elim_true _ _;
    unfold matrix_pts_to_vec gm;
    forevery_ext #(natlt (chunk et))
      _
      (fun x ->
        matrix_pts_to_cell_in_matrix_in_bounds gm
          i (j + x) em
      );
  }
  else
  {
    when__elim_false _ _;
    forevery_intro_fill #(natlt (chunk et))
      (fun x ->
        matrix_pts_to_cell_in_matrix_in_bounds gm
          i (j + x) em
      )
      fn x {
        when__intro_false (j + x < cols) (fun _ ->
          M.pts_to_cell gm (i, j + x) (macc em i (j + x))
        );
      };
  }
}

let thread_offset
  (et : Type0) {| sized et, has_vec_cpy et |}
  (j : nat)
  (tid : nat)
: Pure nat (requires chunk et /? j) (ensures fun off -> chunk et /? off)
=
  lineal_divides (chunk et) j (chunk et) tid;
  j + tid * v (chunk et)

#push-options "--split_queries always --z3rlimit 10"
ghost
fn thread_pts_to_tile_vec_gather
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
  (em : ematrix et rows cols)
  (nthr : nat)
  (trow_tile : nat { chunk et /? trow_tile })
  requires forall+ (tid : natlt nthr).
    thread_pts_to_tile_vec gm i (thread_offset et j tid) em trow_tile nthr
  ensures gpu_pts_to_tile gm i j em (trow_tile * nthr)
{
  forevery_commute _;
  forevery_unfactor' (trow_tile * nthr / chunk et)
    (trow_tile / chunk et) nthr
    _;
  forevery_map #(natlt (trow_tile * nthr / chunk et))
    (fun x ->
      thread_pts_to_vec gm i (thread_offset et j (x % nthr)) em nthr (x / nthr)
    )
    (fun x ->
      forall+ (y : natlt (chunk et)).
        matrix_pts_to_cell_in_matrix_in_bounds gm i (j + (x * chunk et + y)) em
    )
    fn x {
      let lem : squash (
        offset_chunk et (thread_offset et j (x % nthr)) (x / nthr) nthr ==
        j + x * chunk et
      ) = calc (==) {
        offset_chunk et (thread_offset et j (x % nthr)) (x / nthr) nthr;
        == {}
        j + (x % nthr) * chunk et + x / nthr * nthr * chunk et;
        == {}
        j + (x % nthr + x / nthr * nthr) * chunk et;
        == { FStar.Math.Lib.lemma_div_def x nthr }
        j + x * chunk et;
      };
      rewrite each offset_chunk et (thread_offset et j (x % nthr)) (x / nthr) nthr
      as (j + x * chunk et);
      
      unfold_matrix_pts_to_vec_in_matrix_in_bounds gm
        i (j + x * chunk et) em;

      forevery_ext #(natlt (chunk et))
        _
        (fun y ->
          matrix_pts_to_cell_in_matrix_in_bounds gm
            i (j + (x * chunk et + y)) em
        );
    };

  lemma_divides_product_l (chunk et) trow_tile nthr;
  forevery_unfactor (trow_tile * nthr)
    (trow_tile * nthr / chunk et) (chunk et)
    (fun k ->
      matrix_pts_to_cell_in_matrix_in_bounds gm i (j + k) em
    );
  fold gpu_pts_to_tile gm i j em (trow_tile * nthr);
}
#pop-options

ghost
fn gather_gpu_pts_to_tile_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (em : ematrix et rows cols)
  (row_tile : pos)
  requires forall+ (b : natlt (cols `divup` row_tile)).
    gpu_pts_to_tile gm i (b * row_tile) em row_tile
  ensures forall+ (j : natlt cols).
    M.pts_to_cell gm (i, j) (macc em i j)
{
  forevery_map #(natlt (cols `divup` row_tile))
    (fun b ->
      gpu_pts_to_tile gm i (b * row_tile) em row_tile
    )
    (fun b ->
      forall+ (k : natlt row_tile {b * row_tile + k < cols }).
        M.pts_to_cell gm (i, b * row_tile + k) (macc em i (b * row_tile + k))
    )
    fn b {
      unfold gpu_pts_to_tile gm;
      forevery_refine_pred' #(natlt row_tile) (fun k -> b * row_tile + k < cols)
        _;
    };
  forevery_unfactor_ cols row_tile
    (fun j ->
      M.pts_to_cell gm (i, j) (macc em i j)
    );
}

ghost
fn gather_gpu_pts_to_tile
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (gm : array2 et l)
  (em : ematrix et rows cols)
  (row_tile : pos)
  requires pure (fits (M.layout_size l))
  requires forall+ (r : natlt rows) (b : natlt (cols `divup` row_tile)).
    gpu_pts_to_tile gm r (b * row_tile) em row_tile
  ensures gm |-> em
{
  forevery_map #(natlt rows)
    (fun r ->
      forall+ (b : natlt (cols `divup` row_tile)).
        gpu_pts_to_tile gm r (b * row_tile) em row_tile 
    )
    (fun r ->
      forall+ (j : natlt cols).
        M.pts_to_cell gm (r, j) (macc em r j)
    )
    fn r {
      gather_gpu_pts_to_tile_row gm r em row_tile
    };
  M.iraise gm;
}