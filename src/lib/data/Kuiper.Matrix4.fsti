module Kuiper.Matrix4
#lang-pulse

open Kuiper
open Kuiper.EMatrix4
open Kuiper.Matrix.Common
open Kuiper.Matrix.Reprs.Type
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

unfold
inline_for_extraction noextract
type mlayout4 (mrows mcols brows bcols : erased nat) =
  mlayout (mrows * brows) (mcols * bcols)

inline_for_extraction noextract
class clayout4
  (#mrows #mcols #brows #bcols : erased nat)
  (l : mlayout4 mrows mcols brows bcols) =
{
  c_mrows : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal mrows});
  c_mcols : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal mcols});
  c_brows : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal brows});
  c_bcols : (x:SZ.t{SZ.v x > 0 /\ SZ.v x == reveal bcols});
  parent : clayout l;
}

inline_for_extraction noextract
type mrepr4 =
  mrows:nat ->
  mcols:nat ->
  brows:nat ->
  bcols:nat ->
  mlayout4 mrows mcols brows bcols

inline_for_extraction noextract
type crepr4_t (r : mrepr4) =
  mrows:SZ.t ->
  mcols:SZ.t ->
  brows:SZ.t ->
  bcols:SZ.t ->
  squash (SZ.fits (mrows * brows * mcols * bcols)) ->
  clayout4 (r mrows mcols brows bcols)

inline_for_extraction noextract
class crepr4 (r:mrepr4) = {
  map : crepr4_t r;
}

inline_for_extraction noextract
val gpu_matrix
  (et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (l : mlayout4 mrows mcols brows bcols)
  : Type0

inline_for_extraction noextract
val core
  (#et : Type)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (g : gpu_matrix et l)
  : gpu_array et (mlayout_size l)

val core_match
  (#et : Type)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (g1 g2 : gpu_matrix et l)
  : Lemma (requires core g1 == core g2)
          (ensures g1 == g2)

val gpu_matrix_pts_to
  (#et:Type) (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix4 et mrows mcols brows bcols)
  : slprop

(* erased is important for the lens! *)
unfold
instance has_pts_to
  (a:Type)
  (mrows mcols brows bcols : erased nat)
  (l : _)
  : has_pts_to (gpu_matrix a l) (ematrix4 a mrows mcols brows bcols) = {
  pts_to = gpu_matrix_pts_to;
}

inline_for_extraction noextract
fn gpu_matrix_concr
  (#et:Type)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (g : gpu_matrix et l)
  (#em : ematrix4 et mrows mcols brows bcols)
  requires
    g |-> em
  ensures
    core g |-> to_seq l em

inline_for_extraction noextract
fn gpu_matrix_abs
  (#et:Type)
  (#mrows0 #brows0 #mcols0 #bcols0 : erased nat)
  (#l0 : mlayout4 mrows0 brows0 mcols0 bcols0)
  (g : gpu_matrix et l0)
  (#mrows #mcols #brows #bcols : erased nat)
  (l : mlayout4 mrows mcols brows bcols)
  (#em : ematrix4 et mrows mcols brows bcols)
  requires
    core g |-> to_seq l em
  returns
    g' : gpu_matrix et l
  ensures
    pure (mlayout_size l == mlayout_size l0 /\
          core g == core g') **
    (g' |-> em)

inline_for_extraction noextract
fn gpu_matrix_alloc0
  (#et:Type) {| sized et |}
  (mrows mcols brows bcols : szp)
  (l : mlayout4 mrows mcols brows bcols)
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
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
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
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
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
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
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
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols) {| clayout4 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt brows)
  (j : szlt bcols)
  (#f : perm)
  (#em : ematrix4 et mrows mcols brows bcols)
  requires
    gpu **
    gpu_matrix_pts_to gm #f em
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to gm #f em **
    pure (v == macc em bi bj i j)

inline_for_extraction noextract
fn gpu_matrix_write
  (#et:Type0)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols) {| clayout4 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt brows)
  (j : szlt bcols)
  (v : et)
  (#em : _)
  requires
    gpu **
    gpu_matrix_pts_to gm em
  ensures
    gpu **
    gpu_matrix_pts_to gm (mupd em bi bj i j v)

(* Ownership over a single cell. *)
val gpu_matrix_pts_to_cell
  (#et:Type)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]bi : natlt mrows)
  ([@@@mkey]bj : natlt mcols)
  ([@@@mkey]i : natlt brows)
  ([@@@mkey]j : natlt bcols)
  (v : et)
  : slprop

inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols) {| clayout4 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i  : szlt brows)
  (j  : szlt bcols)
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm #f bi bj i j v0
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm #f bi bj i j v **
    pure (v == v0)

inline_for_extraction noextract
fn gpu_matrix_write_cell
  (#et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols) {| clayout4 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i  : szlt brows)
  (j  : szlt bcols)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm bi bj i j v0
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm bi bj i j v1

ghost
fn gpu_matrix_explode
  (#et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : _)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ br bc r c.
      gpu_matrix_pts_to_cell gm #f br bc r c (macc em br bc r c)

ghost
fn gpu_matrix_implode
  (#et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : _)
  requires
    forall+ br bc r c.
      gpu_matrix_pts_to_cell gm #f br bc r c (macc em br bc r c)
  ensures
    gpu_matrix_pts_to gm #f em

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type0) {| sized et |}
  (#mrows #mcols #brows #bcols : szp)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == mlayout_size l})
  (#em : _)
  preserves
    (a |-> s) **
    cpu
  requires
    (gm |-> em)
  ensures
    pure (SZ.fits (mlayout_size l) /\ Pulse.Lib.Vec.length a == (mlayout_size l)) **
    (gm |-> from_seq l s)

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type0) {| sized et |}
  (#mrows #mcols #brows #bcols : SZ.t)
  (#l : mlayout4 mrows mcols brows bcols)
  (a : vec et)
  (gm : gpu_matrix et l)
  (#s : erased (seq et){Seq.length s == mlayout_size l})
  (#em : _)
  preserves
    (gm |-> em) **
    cpu
  requires
    (a |-> s)
  ensures
    pure (SZ.fits (mlayout_size l) /\ Pulse.Lib.Vec.length a == (mlayout_size l)) **
    (a |-> to_seq l em)
