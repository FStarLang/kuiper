module Kuiper.Matrix6
#lang-pulse

open Kuiper
open Kuiper.EMatrix6
open Kuiper.Matrix {
  gpu_matrix_pts_to as gpu_matrix_pts_to2,
  gpu_matrix as gpu_matrix2
}
open Kuiper.EMatrix {
  ematrix as ematrix2
}
open Kuiper.Matrix.Common
// open Kuiper.Matrix4
open Kuiper.Matrix.Reprs.Type

module T  = FStar.Tactics.V2
module SZ = FStar.SizeT

unfold
inline_for_extraction noextract
type mlayout6 (mrows mcols brows bcols trows tcols : erased nat) =
  mlayout (mrows * brows) (mcols * bcols)

inline_for_extraction noextract
class clayout6
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (l : mlayout6 mrows mcols brows bcols trows tcols) =
{
  c_mrows : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal mrows});
  c_mcols : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal mcols});
  c_brows : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal brows});
  c_bcols : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal bcols});
  c_trows : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal trows});
  c_tcols : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal tcols});
  parent : clayout l;
}

// inline_for_extraction noextract
// let clayout6_from_clayout4
//   (#rows #cols : szp)
//   (tile : szp)
//   (ttile : szp)
//   (#l : mlayout4 (rows * tile) (cols * tile) ttile ttile)
//   (c : clayout l)
//   : clayout6 l = {
//     parent = c;
//     c_mrows = rows;
//     c_mcols = cols;
//     c_brows = tile;
//     c_bcols = tile;
//     c_trows = ttile;
//     c_tcols = ttile;
// }

inline_for_extraction noextract
type mrepr6 =
  mrows:nat ->
  mcols:nat ->
  brows:nat ->
  bcols:nat ->
  trows:nat ->
  tcols:nat ->
  mlayout6 mrows mcols brows bcols trows tcols

inline_for_extraction noextract
type crepr6_t (r : mrepr6) =
  mrows:SZ.t ->
  mcols:SZ.t ->
  brows:SZ.t ->
  bcols:SZ.t ->
  trows:SZ.t ->
  tcols:SZ.t ->
  squash (SZ.fits (mrows * brows * mcols * bcols * trows * tcols)) ->
  clayout6 (r mrows mcols brows bcols trows tcols)

inline_for_extraction noextract
class crepr6 (r:mrepr6) = {
  map : crepr6_t r;
}

inline_for_extraction noextract
val gpu_matrix
  (et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (l : mlayout6 mrows mcols brows bcols trows tcols)
  : Type0

inline_for_extraction noextract
val from_array
  (#a : Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (l : mlayout6 mrows mcols brows bcols trows tcols)
  (arr : gpu_array a (mlayout_size l))
  : gpu_matrix a l

inline_for_extraction noextract
val core
  (#et : Type)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (g : gpu_matrix et l)
  : gpu_array et (mlayout_size l)

val lem_core_from_array
  (#et : Type)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (g : gpu_matrix et l)
  : Lemma (ensures from_array l (core g) == g)
          [SMTPat (core g)]

val lem_from_array_core
  (#et : Type)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (p : gpu_array et (mlayout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val gpu_matrix_pts_to
  (#et:Type) (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix6 et mrows mcols brows bcols trows tcols)
  : slprop

(* erased is important for the lens! *)
unfold
instance has_pts_to
  (a:Type)
  (mrows mcols brows bcols trows tcols : erased nat)
  (l : _)
  : has_pts_to (gpu_matrix a l) (ematrix6 a mrows mcols brows bcols trows tcols) = {
  pts_to = gpu_matrix_pts_to;
}
ghost
fn gpu_matrix_pts_to_ref
  (#et:Type)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (g : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix6 et mrows mcols brows bcols trows tcols)
  preserves
    gpu_matrix_pts_to g #f em
  ensures
    pure (SZ.fits (mlayout_size l))

// ghost
// fn gpu_matrix_concr
//   (#et:Type)
//   (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
//   (#l : mlayout6 mrows mcols brows bcols trows tcols)
//   (g : gpu_matrix et l)
//   (#em : ematrix6 et mrows mcols brows bcols trows tcols)
//   requires
//     g |-> em
//   ensures
//     core g |-> to_seq l em

// ghost
// fn gpu_matrix_abs
//   (#et:Type)
//   (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
//   (l : mlayout6 mrows mcols brows bcols trows tcols)
//   (p : gpu_array et (mlayout_size l))
//   (#f : perm)
//   (#em : ematrix6 et mrows mcols brows bcols trows tcols)
//   requires
//     gpu_pts_to_array p #f (to_seq l em)
//   ensures
//     gpu_matrix_pts_to (from_array l p) #f em

// ghost
// fn gpu_matrix_abs'
//   (#et:Type)
//   (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
//   (l : mlayout6 mrows mcols brows bcols trows tcols)
//   (p : gpu_array et (mlayout_size l))
//   (#f : perm)
//   (#s : erased (seq et){Seq.length s == mlayout_size l})
//   requires
//     gpu_pts_to_array p #f s
//   ensures
//     gpu_matrix_pts_to (from_array l p) #f (from_seq l s)

inline_for_extraction noextract
fn gpu_matrix_alloc0
  (#et:Type) {| sized et |}
  (mrows mcols brows bcols trows tcols : szp)
  (l : mlayout6 mrows mcols brows bcols trows tcols)
  preserves
    cpu
  requires
    pure (SZ.fits (mlayout_size l))
  returns
    gm : gpu_matrix et l
  ensures
    exists* em. gm |-> em

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (gm : gpu_matrix et l)
  (#em : _)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#uid: int)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : _)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#uid: int)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : _)
  requires
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
  ensures
    gpu_matrix_pts_to gm #f em

inline_for_extraction noextract
fn gpu_matrix_read
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols) {| clayout6 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt brows)
  (j : szlt bcols)
  (k : szlt trows)
  (l : szlt tcols)
  (#f : perm)
  (#em : ematrix6 et mrows mcols brows bcols trows tcols)
  requires
    gpu **
    gpu_matrix_pts_to gm #f em
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to gm #f em **
    pure (v == macc em bi bj i j k l)

inline_for_extraction noextract
fn gpu_matrix_write
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : erased nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols) {| clayout6 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt brows)
  (j : szlt bcols)
  (k : szlt trows)
  (l : szlt tcols)
  (v : et)
  (#em : _)
  requires
    gpu **
    gpu_matrix_pts_to gm em
  ensures
    gpu **
    gpu_matrix_pts_to gm (mupd em bi bj i j k l v)

(* Ownership over a single cell. *)
val gpu_matrix_pts_to_cell
  (#et:Type)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]bi : natlt mrows)
  ([@@@mkey]bj : natlt mcols)
  ([@@@mkey]i : natlt brows)
  ([@@@mkey]j : natlt bcols)
  ([@@@mkey]k : natlt trows)
  ([@@@mkey]l : natlt tcols)
  (v : et)
  : slprop

inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols) {| clayout6 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i  : szlt brows)
  (j  : szlt bcols)
  (k  : szlt trows)
  (l  : szlt tcols)
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm #f bi bj i j k l v0
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm #f bi bj i j k l v **
    pure (v == v0)

inline_for_extraction noextract
fn gpu_matrix_write_cell
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols) {| clayout6 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i  : szlt brows)
  (j  : szlt bcols)
  (k  : szlt trows)
  (l  : szlt tcols)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm bi bj i j k l v0
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm bi bj i j k l v1

ghost
fn gpu_matrix_explode
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : _)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ br bc r c tr tc.
      gpu_matrix_pts_to_cell gm #f br bc r c tr tc (macc em br bc r c tr tc)

ghost
fn gpu_matrix_implode
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : _)
  requires
    forall+ br bc r c tr tc.
      gpu_matrix_pts_to_cell gm #f br bc r c tr tc (macc em br bc r c tr tc)
  ensures
    gpu_matrix_pts_to gm #f em

val mlayout6_to_mlayout2
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (l : mlayout6 mrows mcols brows bcols trows tcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i : natlt brows)
  (j : natlt bcols)
  : mlayout trows tcols

val clayout6_to_clayout2
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (l : mlayout6 mrows mcols brows bcols trows tcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i : natlt brows)
  (j : natlt bcols)
  {| clayout6 l |}
  : clayout (mlayout6_to_mlayout2 #et l bi bj i j)

val gpu_matrix6_to_gpu_matrix2
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (gm : gpu_matrix et l)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i : natlt brows)
  (j : natlt bcols)
  : gpu_matrix2 et (mlayout6_to_mlayout2 #et l bi bj i j)

val ematrix6_to_ematrix2
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (em : ematrix6 et mrows mcols brows bcols trows tcols)
  (bi : natlt mrows)
  (bj : natlt mcols)
  (i : natlt brows)
  (j : natlt bcols)
  : ematrix2 et trows tcols

ghost
fn gpu_matrix_explode'
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : _)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ br bc r c.
      gpu_matrix_pts_to2 (gpu_matrix6_to_gpu_matrix2 gm br bc r c) #f (ematrix6_to_ematrix2 em br bc r c)

ghost
fn gpu_matrix_implode'
  (#et:Type0)
  (#mrows #mcols #brows #bcols #trows #tcols : nat)
  (#l : mlayout6 mrows mcols brows bcols trows tcols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : _)
  requires
    forall+ br bc r c.
      gpu_matrix_pts_to2 (gpu_matrix6_to_gpu_matrix2 gm br bc r c) #f (ematrix6_to_ematrix2 em br bc r c)
  ensures
    gpu_matrix_pts_to gm #f em

// inline_for_extraction noextract
// fn gpu_matrix_from_array
//   (#et:Type0) {| sized et |}
//   (#mrows #mcols #brows #bcols #trows #tcols : SZ.t)
//   (#l : mlayout6 mrows mcols brows bcols trows tcols)
//   (gm : gpu_matrix et l)
//   (a : vec et)
//   (#s : erased (seq et){Seq.length s == mlayout_size l})
//   (#em : _)
//   preserves
//     (a |-> s) **
//     cpu
//   requires
//     pure (mlayout_size l > 0) **
//     (gm |-> em)
//   ensures
//     pure (SZ.fits (mlayout_size l) /\ Pulse.Lib.Vec.length a == (mlayout_size l)) **
//     (gm |-> from_seq l s)

// inline_for_extraction noextract
// fn gpu_matrix_to_array
//   (#et:Type0) {| sized et |}
//   (#mrows #mcols #brows #bcols #trows #tcols : SZ.t)
//   (#l : mlayout6 mrows mcols brows bcols trows tcols)
//   (a : vec et)
//   (gm : gpu_matrix et l)
//   (#s : erased (seq et){Seq.length s == mlayout_size l})
//   (#em : _)
//   preserves
//     (gm |-> em) **
//     cpu
//   requires
//     pure (mlayout_size l > 0) **
//     (a |-> s)
//   ensures
//     pure (SZ.fits (mlayout_size l) /\ Pulse.Lib.Vec.length a == (mlayout_size l)) **
//     (a |-> to_seq l em)
