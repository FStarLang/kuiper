module Kuiper.Sparse.Matrix.PtsTo

#lang-pulse

open Kuiper
open FStar.Tactics.V2 { exact }
open Kuiper.Sparse.Common
open Kuiper.Tensor
open Kuiper.Array.Vectorized
open Kuiper.EMatrix

open Kuiper.Array2.Strided { strided_row_major }

ghost
fn matrix_pts_to_vec_in_bounds_equiv
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
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

(* fold/unfold *)

ghost
fn fold_matrix_pts_to_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
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
  (#l : layout2 rows cols)
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
  (#l : layout2 rows cols)
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
  (#l : layout2 rows cols)
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
  (#l : layout2 rows cols)
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
  (#l : layout2 rows cols)
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
  (#l : layout2 rows cols)
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
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  requires pure (j >= cols)
  requires matrix_live_vec_in_bounds gm i j
{
  unfold matrix_live_vec_in_bounds gm;
  unfold_matrix_pts_to_vec_not_in_bounds gm i j _;
}

ghost
fn fold_thread_pts_to_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : chest2 et rows cols)
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
  requires thread_pts_to_vec_underspec gm i j s nthr k
  requires pure (is_ematrix_tile em i j s nthr)
  ensures thread_pts_to_vec gm i j em nthr k
{
  admit();
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

ghost
fn unfold_matrix_pts_to_vec_in_matrix_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  // usamos pos y no nat porque garantiza que chunk et <= cols
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : chest2 et rows cols)
  requires matrix_pts_to_vec_in_matrix_in_bounds gm i j em
  ensures forall+ (x : natlt (chunk et)).
    matrix_pts_to_cell_in_matrix_in_bounds gm i (j + x) em
{
  admit();
  if (j < cols)
  {
    when__elim_true _ _;
    unfold matrix_pts_to_vec gm i j _;
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
          Cell gm (idx2 i (j + x <: natlt cols)) |-> (acc2 em i (j + x))
        );
      };
  }
}

#push-options "--z3rlimit 25"
ghost
fn fold_thread_pts_to_tile_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (em : chest2 et rows cols)
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
#pop-options

#push-options "--z3rlimit 25"
ghost
fn thread_pts_to_tile_vec_gather
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
  (em : chest2 et rows cols)
  (nthr : nat)
  (trow_tile : nat { chunk et /? trow_tile })
  requires forall+ (tid : natlt nthr).
    thread_pts_to_tile_vec gm i (thread_offset et j tid) em trow_tile nthr
  ensures pts_to_tile_in_matrix gm i j em (trow_tile * nthr)
{
  forevery_commute _;
  forevery_unfactor' (trow_tile * nthr / chunk et)
    (trow_tile / chunk et) nthr
    (fun k tid -> thread_pts_to_vec gm i (thread_offset et j tid) em nthr k);

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
  fold pts_to_tile_in_matrix gm i j em (trow_tile * nthr);
}
#pop-options

ghost
fn gather_pts_to_tile_in_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (em : chest2 et rows cols)
  (row_tile : pos)
  requires forall+ (b : natlt (cols `divup` row_tile)).
    pts_to_tile_in_matrix gm i (b * row_tile) em row_tile
  ensures forall+ (j : natlt cols).
    Cell gm (idx2 i j) |-> acc2 em i j
    // pts_to_cell gm (i, j) (acc2 em i j)
{
  forevery_map #(natlt (cols `divup` row_tile))
    (fun b ->
      pts_to_tile_in_matrix gm i (b * row_tile) em row_tile
    )
    (fun b ->
      forall+ (k : natlt row_tile {b * row_tile + k < cols }).
        Cell gm (idx2 i (b * row_tile + k <: natlt cols)) |-> (acc2 em i (b * row_tile + k))
    )
    fn b {
      unfold pts_to_tile_in_matrix gm;
      forevery_refine_pred' #(natlt row_tile) (fun k -> b * row_tile + k < cols)
        _;
    };
  forevery_unfactor_ cols row_tile
    (fun j ->
      Cell gm (idx2 i j) |-> acc2 em i j
    );
}

ghost
fn gahter_pts_to_tile_in_matrix
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (em : chest2 et rows cols)
  (row_tile : pos)
  requires array_exists (core gm)
  requires pure (fits (tlayout_ulen l))
  requires forall+ (r : natlt rows) (b : natlt (cols `divup` row_tile)).
    pts_to_tile_in_matrix gm r (b * row_tile) em row_tile
  ensures gm |-> em
{
  admit();
  forevery_map #(natlt rows)
    (fun r ->
      forall+ (b : natlt (cols `divup` row_tile)).
        pts_to_tile_in_matrix gm r (b * row_tile) em row_tile
    )
    (fun r ->
      forall+ (j : natlt cols).
        Cell gm (idx2 r j) |-> acc2 em r j
    )
    fn r {
      gather_pts_to_tile_in_row gm r em row_tile
    };
  tensor_iraise2_with_exists gm;
}
