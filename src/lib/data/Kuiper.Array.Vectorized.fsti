module Kuiper.Array.Vectorized

#lang-pulse

open FStar.Seq

open Kuiper
open Kuiper.Seq.Common { seq_blit }
module SZ = Kuiper.SizeT

inline_for_extraction noextract
unfold
class has_vec_cpy (et : Type) {| sized et |} = {
  [@@@FStar.Tactics.Typeclasses.no_method] _chunk : szp;
  [@@@FStar.Tactics.Typeclasses.no_method] _pf : squash (_chunk * size #et == 16);
  (* ^ Vectorized copies are always 16 bytes wide, for us. One can do smaller
     vectorized copies in CUDA. *)
}

unfold
inline_for_extraction noextract
let chunk (et : Type) {| sized et, hvc : has_vec_cpy et |} : szp =
  match hvc with
  | Mkhas_vec_cpy chunk _ -> chunk

inline_for_extraction noextract
instance has_vec_cpy_float : has_vec_cpy float =
  Kuiper.Float32.lem_sizeof ();
  { _chunk = 4sz; _pf = ez; }

inline_for_extraction noextract
instance has_vec_cpy_half  : has_vec_cpy half =
  Kuiper.Float16.lem_sizeof ();
  { _chunk = 8sz; _pf = ez; }

(* The single primitive for vectorized copies. *)

noextract (* prevents krml warning *)
fn array_vec_cpy
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (dst_arr : array et)
  (dst_off : SZ.t)
  (#dst_slice_i : erased nat)
  (#dst_slice_j : erased nat)
  (src_arr : array et)
  (src_off : SZ.t)
  (#src_slice_i : erased nat)
  (#src_slice_j : erased nat)
  (#f : perm)
  (#ss : erased (seq et))
  (#ds : erased (seq et))
  (#_ : squash (dst_slice_i <= dst_off /\ dst_off + chunk et <= dst_slice_j))
  (#_ : squash (Seq.length ds == dst_slice_j - dst_slice_i))
  (#_ : squash (src_slice_i <= src_off /\ src_off + chunk et <= src_slice_j))
  (#_ : squash (Seq.length ss == src_slice_j - src_slice_i))
  preserves gpu
  preserves pts_to_slice src_arr #f src_slice_i src_slice_j ss
  requires  pure (aligned' 16 src_arr src_off)
  requires  pure (aligned' 16 dst_arr dst_off)
  requires  pts_to_slice dst_arr dst_slice_i dst_slice_j ds
  ensures
    exists* s'.
      pts_to_slice dst_arr dst_slice_i dst_slice_j s' **
      pure (s' == seq_blit ds (dst_off - dst_slice_i) ss (src_off - src_slice_i) (chunk et))
