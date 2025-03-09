module Kuiper.Matrix
#lang-pulse

open Kuiper
open Kuiper.EMatrix
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

(* Also row major layout *)
inline_for_extraction noextract
val gpu_matrix (et:Type0) (rows cols : nat) : Type0

val gpu_matrix_pts_to
  (#et:Type) (#rows #cols : nat)
  ([@@@mkey] gm : gpu_matrix et rows cols)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix et rows cols)
  : slprop

unfold
instance has_pts_to (a:Type) (rows cols : _)
  : has_pts_to (gpu_matrix a rows cols) (ematrix a rows cols) = {
  pts_to = gpu_matrix_pts_to;
}

inline_for_extraction noextract
fn gpu_matrix_alloc
  (#et:Type) {| scalar et |}
  (rows cols : szp)
  preserves
    cpu
  requires
    pure (SZ.fits (rows * cols))
  returns
    gm : gpu_matrix et rows cols
  ensures
    exists* em. gm |-> em

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type) {| scalar et |}
  (#rows #cols : erased nat)
  (gm : gpu_matrix et rows cols)
  (#em : _)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp

(* NOTE: row-major in these specs. *)

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type) {| scalar et |}
  (#rows #cols : szp)
  (a : vec et)
  (gA : gpu_matrix et rows cols)
  (#s : erased (seq et){ len s == rows * cols })
  preserves
    (a |-> s) **
    cpu
  requires
    (gA |-> 'm0) **
    pure (SZ.fits (rows * cols))
  ensures
    gA |-> from_row_major_seq s

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type) {| scalar et |}
  (#rows #cols : szp)
  (a : vec et)
  (gA : gpu_matrix et rows cols)
  (#m : ematrix et rows cols)
  preserves
    (gA |-> m) **
    cpu
  requires
    (a |-> 's0) **
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols)
  ensures
    a |-> to_row_major_seq #_ #rows #cols m

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (gm : gpu_matrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (gm : gpu_matrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
  ensures
    gpu_matrix_pts_to gm #f em

inline_for_extraction noextract
fn gpu_matrix_read
  (#et:Type0)
  (#rows : erased nat)
  (#cols : SZ.t) (* NOTE! This is a concrete argument *)
  (gm : gpu_matrix et rows cols)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm #f em
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to gm #f em **
    pure (v == macc em i j)

inline_for_extraction noextract
fn gpu_matrix_write
  (#et:Type0)
  (#rows : erased nat)
  (#cols : SZ.t) (* NOTE! This is a concrete argument *)
  (gm : gpu_matrix et rows cols)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (v : et)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm em
  ensures
    gpu **
    gpu_matrix_pts_to gm (mupd em i j v)

(* Ownership over a single cell. *)
val gpu_matrix_pts_to_cell
  (#et:Type) (#rows #cols : nat)
  ([@@@mkey] gm : gpu_matrix et rows cols)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : natlt rows)
  ([@@@mkey]j : natlt cols)
  (v : et)
  : slprop

inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#rows : erased nat)
  (#cols : SZ.t) (* NOTE! This is a concrete argument *)
  (gm : gpu_matrix et rows cols)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm #f i j v0
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm #f i j v **
    pure (v == v0)

inline_for_extraction noextract
fn gpu_matrix_write_cell
  (#et:Type0)
  (#rows : erased nat)
  (#cols : SZ.t) (* NOTE! This is a concrete argument *)
  (gm : gpu_matrix et rows cols)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm i j v0
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm i j v1

ghost
fn gpu_matrix_explode
  (#et:Type0)
  (#rows #cols : nat)
  (gm : gpu_matrix et rows cols)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)

ghost
fn gpu_matrix_implode
  (#et:Type0)
  (#rows #cols : nat)
  (gm : gpu_matrix et rows cols)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
  ensures
    gpu_matrix_pts_to gm #f em
