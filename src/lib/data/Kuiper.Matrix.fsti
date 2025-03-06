module Kuiper.Matrix
#lang-pulse

open Kuiper
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

[@@erasable]
noeq
type ematrix (et:Type) (rows cols : nat) =
  | M : s:(seq et){ len s == rows * cols } -> ematrix et rows cols

(* Note: row major (though this is erased, so not that important.
Should we even expose it? *)
let macc (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{ i < rows })
  (j : nat{ j < cols })
  : GTot et
  = m.s @! (i * cols + j)

let mupd (#et:Type) (#rows #cols : nat)
  (m : ematrix et rows cols)
  (i : nat{ i < rows })
  (j : nat{ j < cols })
  (v : et)
  : ematrix et rows cols
  = M <| Seq.upd m.s (i * cols + j) v

(* Also row major layout *)
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
