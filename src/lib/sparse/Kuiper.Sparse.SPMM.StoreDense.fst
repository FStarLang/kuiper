module Kuiper.Sparse.SPMM.StoreDense

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.Load
open Kuiper.Array.Vectorized
open Kuiper.Array2.Vectorized
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
open Kuiper.Seq.Common { seq_blit, seq_replace }
open Kuiper.Array2.Strided { strided_row_major, cell_of_pos, aligned_strided_row_major }
module T = Kuiper.Tensor.Layout
module M = Kuiper.Array2
module A = Kuiper.Array1

// TODO tiene sentido pensar en hacer esta operación sobre filas como Array1?
// No se si cambia mucho porque a cada bloque no le toca una fila entera

let aligned_cell_strided_row_major
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : M.layout rows cols) {| strided : strided_row_major l |}
  (gm : M.array2 et l)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
: Lemma
  (requires
    aligned 16 (M.core gm) /\
    aligned_strided_row_major (chunk et) strided
  )
  (ensures aligned' 16 (M.core gm) (cell_of_pos l i j))
=
  strided.pf i j;
  lineal_divides (chunk et) strided.offset strided.stride i;
  lemma_divides_sum (chunk et) (strided.offset + strided.stride * i) j;
  ()

inline_for_extraction noextract
fn matrix_vec_write_in_bounds
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp { fits (rows * cols) /\ chunk et /? cols })
  (#l : M.layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : M.array2 et l)
  (i : szlt rows)
  (j : sz { chunk et /? j })
  (#f : perm)
  (#n : erased nat)
  (arr : larray et n)
  (#s : erased (lseq et n))
  (k : sz { k + chunk et <= n })
  preserves gpu
  preserves arr |-> Frac f s
  requires  matrix_live_vec_in_bounds gm i j
  requires  pure (aligned 16 (M.core gm))
  requires  pure (aligned_strided_row_major (chunk et) strided)
  ensures   matrix_pts_to_vec_slice_in_bounds gm i j s k
{
  if (j <^ cols) {
    unfold_matrix_live_vec_in_bounds gm i j;

    aligned_cell_strided_row_major gm i j;
    lemma_divides_leq (chunk et) cols j;
    matrix_vec_store gm i j arr k;

    fold_matrix_pts_to_vec_slice_in_bounds gm i j s k;
  }
  else
  {
    unfold_matrix_live_vec_not_in_bounds gm i j;
    fold_matrix_pts_to_vec_slice_not_in_bounds gm i j s k;
  }
}


inline_for_extraction noextract
let offset_chunk_
  (et : Type0) {| sized et, has_vec_cpy et |}
  (j : sz { chunk et /? j })
  (k : sz)
  (nthr : sz)
: Pure sz
  (requires fits (offset_chunk et j k nthr))
  (ensures fun r -> v r == offset_chunk et j k nthr)
=
  j +^ (k *^ nthr) *^ chunk et

inline_for_extraction noextract
fn gpu_matrix_store_tile_vec_underspec
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp { fits (rows * cols) /\ chunk et /? cols })
  (#l : M.layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : M.array2 et l)
  (i : szlt rows)
  (j : sz { chunk et /? j })
  (#n : sz { chunk et /? n })
  (arr : larray et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (nthr : sz)
  preserves gpu
  preserves arr |-> Frac f s
  requires  thread_live_tile_vec gm i j n nthr
  requires  pure (aligned 16 (M.core gm))
  requires  pure (aligned_strided_row_major (chunk et) strided)
  requires  pure (fits (j + n * nthr))
  ensures   thread_pts_to_tile_vec_underspec gm i j s nthr
{
  unfold thread_live_tile_vec gm;

  forevery_rw_size (n / chunk et) (n /^ chunk et);
  
  foreach (n /^ chunk et)
    (fun k -> thread_live_vec gm i j n nthr k)
    (fun k -> thread_pts_to_vec_underspec gm i j s nthr k)
    #(gpu ** arr |-> Frac f s)
    fn k {
      assert pure (offset_chunk et j k nthr <= j + n * nthr);
      FStar.SizeT.fits_lte (offset_chunk et j k nthr) (j + n * nthr);
      rewrite each offset_chunk et j k nthr
      as v (offset_chunk_ et j k nthr);
      matrix_vec_write_in_bounds gm
        i (offset_chunk_ et j k nthr)
        arr (k *^ chunk et);
      rewrite each v (offset_chunk_ et j k nthr)
      as offset_chunk et j k nthr;
    };

  forevery_rw_size (n /^ chunk et) (n / chunk et);
  fold thread_pts_to_tile_vec_underspec gm i j s nthr;
}

inline_for_extraction noextract
fn gpu_matrix_store_tile_vec
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp { fits (rows * cols) /\ chunk et /? cols })
  (#l : M.layout rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : M.array2 et l)
  (i : szlt rows)
  (j : sz { chunk et /? j })
  (em : ematrix et rows cols)
  (n : sz { chunk et /? n })
  (arr : larray et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (nthr : sz)
  preserves gpu
  preserves arr |-> Frac f s
  requires  pure (is_ematrix_tile em i j s nthr)
  requires  thread_live_tile_vec gm i j n nthr
  requires  pure (aligned 16 (M.core gm))
  requires  pure (aligned_strided_row_major (chunk et) strided)
  requires  pure (fits (j + n * nthr))
  ensures   thread_pts_to_tile_vec gm i j em n nthr
{
  gpu_matrix_store_tile_vec_underspec gm i j arr nthr;
  fold_thread_pts_to_tile_vec gm i j s em nthr;
}