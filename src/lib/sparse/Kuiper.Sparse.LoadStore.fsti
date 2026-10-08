module Kuiper.Sparse.LoadStore

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Array.Vectorized
open Kuiper.Array2.Vectorized
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Array2.Strided
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn load_array_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : sz)
  (x : larray et n)
  (#_ : squash (aligned 16 x))
  (#m : sz)
  (y : larray et m)
  (#f : perm)
  (#s : erased (lseq et m))
  (i : szle m)
  (#_ : squash (aligned' 16 y i))
  (nthr : sz)
  (tid : szlt nthr)
  (#_ : squash (nthr * chunk et /? n))
  (#_ : squash (i + n <= m))
  preserves gpu ** y |-> Frac f s
  requires thread_live_chunks x nthr tid
  ensures thread_pts_to_chunks x s i nthr tid

inline_for_extraction noextract
fn load2_array
  (#et1 #et2 : Type0)
  (#dsz : sz)
  (dst1 : larray et1 dsz)
  (dst2 : larray et2 dsz)
  (to : szle dsz)
  (#ssz : sz)
  (src1 : larray et1 ssz)
  (src2 : larray et2 ssz)
  (#f : perm)
  (#s1 : erased (lseq et1 ssz))
  (#s2 : erased (lseq et2 ssz))
  (i : szle ssz)
  (nthr : sz)
  (tid : szlt nthr)
  (#_ : squash (i + to <= ssz))
  preserves gpu ** src1 |-> Frac f s1 ** src2 |-> Frac f s2
  requires thread_slice_live dst1 0 to nthr tid
  requires thread_slice_live dst2 0 to nthr tid
  requires pure (fits (dsz + nthr))
  ensures thread_slice_pts_to dst1 0 to s1 i nthr tid
  ensures thread_slice_pts_to dst2 0 to s2 i nthr tid

inline_for_extraction noextract
fn matrix_store_tile_vec
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : szp { fits (rows * cols) /\ chunk et /? cols })
  (#l : layout2 rows cols) {| ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : sz { chunk et /? j })
  (em : chest2 et rows cols)
  (n : sz { chunk et /? n })
  (arr : larray et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (nthr : sz)
  preserves gpu
  preserves arr |-> Frac f s
  requires  pure (aligned 16 arr)
  requires  pure (is_ematrix_tile em i j s nthr)
  requires  thread_live_tile_vec gm i j n nthr
  requires  pure (aligned 16 (core gm))
  requires  pure (aligned_strided_row_major (chunk et) strided)
  requires  pure (fits (j + n * nthr))
  ensures   thread_pts_to_tile_vec gm i j em n nthr

(* Kuiper.Array.Vectorized.array_vec_cpy specialized to a dst array of chunk length *)
inline_for_extraction noextract
fn array_cpy_chunk
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (dst_arr : larray et (chunk et))
  (#src_sz : szp { chunk et /? src_sz })
  (#src_l : layout1 src_sz) {| T.ctlayout src_l, src_cl : cont_layout src_l |}
  (src_arr : array1 et src_l)
  (src_off : szlt src_sz { chunk et /? src_off })
  (#f : perm)
  (#ss : erased (lseq et src_sz))
  preserves gpu
  requires  live dst_arr
  requires  pure (aligned 16 dst_arr)
  preserves src_arr |-> Frac f (seq_to_chest1 ss)
  requires  pure (aligned 16 (core src_arr))
  requires  pure (aligned_cont_layout (chunk et) src_cl)
  ensures   dst_arr |-> Seq.slice ss src_off (src_off + chunk et)