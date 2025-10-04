module Kuiper.Array.Vectorized

#lang-pulse

open FStar.Seq

open Kuiper
open Kuiper.Seq.Common { seq_blit }

module SZ = FStar.SizeT

inline_for_extraction noextract
unfold
class has_vec_cpy (et : Type) = {
  [@@@FStar.Tactics.Typeclasses.no_method] _chunk : szp;
}

unfold
inline_for_extraction noextract
let chunk (et : Type) {| hvc : has_vec_cpy et |} : szp =
  match hvc with
  | Mkhas_vec_cpy chunk -> chunk

unfold
inline_for_extraction noextract
instance has_vec_cpy_float : has_vec_cpy float = { _chunk = 4sz; }

unfold
inline_for_extraction noextract
instance has_vec_cpy_half  : has_vec_cpy half  = { _chunk = 8sz; }

(* These three operations are essentially the same. We need different
   variants since gpu_array is a different type from array. Sadly the
   slicing is also different. *)

[@@noextract_to "krml"]
atomic
fn gpu_array_vec_cpy_dd
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (#dst_sz : erased nat)
  (dst_arr : gpu_array et dst_sz)
  (dst_off : SZ.t)
  (#dst_slice_i : erased nat)
  (#dst_slice_j : erased nat)
  (#src_sz : erased nat)
  (#src_slice_i : erased nat)
  (#src_slice_j : erased nat)
  (src_arr : gpu_array et src_sz)
  (src_off : SZ.t)
  (#f : perm)
  (#ss : erased (seq et))
  (#ds : erased (seq et))
  (#_ : squash (dst_slice_i <= dst_off /\ dst_off + chunk et <= dst_slice_j))
  (#_ : squash (Seq.length ds == dst_slice_j - dst_slice_i))
  (#_ : squash (src_slice_i <= src_off /\ src_off + chunk et <= src_slice_j))
  (#_ : squash (Seq.length ss == src_slice_j - src_slice_i))
  preserves gpu
  preserves gpu_pts_to_slice src_arr #f src_slice_i src_slice_j ss
  requires  gpu_pts_to_slice dst_arr dst_slice_i dst_slice_j ds
  ensures   gpu_pts_to_slice dst_arr dst_slice_i dst_slice_j (seq_blit ds (dst_off - dst_slice_i) ss (src_off - src_slice_i) (chunk et))

[@@noextract_to "krml"]
atomic
fn gpu_array_vec_cpy_dh
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (dst_arr : array et)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_arr : gpu_array et src_sz)
  (src_off : SZ.t)
  (#src_slice_i : erased nat)
  (#src_slice_j : erased nat)
  (#f : perm)
  (#ss : erased (seq et))
  (#ds : erased (seq et))
  (#_ : squash (dst_off + chunk et <= Seq.length ds))
  (#_ : squash (src_slice_i <= src_off /\ src_off + chunk et <= src_slice_j))
  (#_ : squash (Seq.length ss == src_slice_j - src_slice_i))
  preserves gpu
  preserves gpu_pts_to_slice src_arr #f src_slice_i src_slice_j ss
  requires  dst_arr |-> ds
  ensures   dst_arr |-> (seq_blit ds dst_off ss (src_off - src_slice_i) (chunk et))

[@@noextract_to "krml"]
atomic
fn gpu_array_vec_cpy_hd
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (#dst_sz : erased nat)
  (dst_arr : gpu_array et dst_sz)
  (dst_off : SZ.t)
  (#dst_slice_i : erased nat)
  (#dst_slice_j : erased nat)
  (src_arr : array et)
  (src_off : SZ.t)
  (#f : perm)
  (#ss : erased (seq et))
  (#ds : erased (seq et))
  (#_ : squash (dst_slice_i <= dst_off /\ dst_off + chunk et <= dst_slice_j))
  (#_ : squash (Seq.length ds == dst_slice_j - dst_slice_i))
  (#_ : squash (src_off + chunk et <= Seq.length ss))
  preserves gpu
  preserves src_arr |-> Frac f ss
  requires  gpu_pts_to_slice dst_arr dst_slice_i dst_slice_j ds
  ensures   gpu_pts_to_slice dst_arr dst_slice_i dst_slice_j (seq_blit ds (dst_off - dst_slice_i) ss src_off (chunk et))
