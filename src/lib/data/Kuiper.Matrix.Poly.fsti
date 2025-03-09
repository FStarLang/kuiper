module Kuiper.Matrix.Poly
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Bijection
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

[@@erasable]
noeq
type mlayout (rows cols : nat) = {
  bij     : (natlt rows & natlt cols) =~ natlt (rows * cols);
}

(* Concrete layout accessors. *)
inline_for_extraction
class clayout (#rows #cols : _) (l : mlayout rows cols) = {
  c_to    : (i:SZ.t{i < rows}) -> (j:SZ.t{j < cols}) -> r:SZ.t{SZ.v r == l.bij.ff (SZ.v i, SZ.v j)};
  c_from1 : (idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == fst (l.bij.gg (SZ.v idx))};
  c_from2 : (idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == snd (l.bij.gg (SZ.v idx))};
}

inline_for_extraction
type mrepr = #rows:nat -> #cols:nat -> mlayout rows cols

inline_for_extraction
class crepr (r:mrepr) = {
  map : (rows:SZ.t -> cols:SZ.t{SZ.fits (rows * cols)} -> clayout (r #rows #cols));
}

inline_for_extraction noextract
instance clayout_from_crepr
  (rows : SZ.t) (cols : SZ.t{SZ.fits (rows * cols)})
  (m : mrepr) (d : crepr m)
  : clayout (m #rows #cols)
  = d.map rows cols

let from_seq (#et:Type) (#rows #cols : _)
  (l : mlayout rows cols)
  (s : lseq et (rows * cols))
  : ematrix et rows cols
  = M <| fun i j -> s @! l.bij.ff (i,j)

let to_seq (#et:Type) (#rows #cols : _)
  (l : mlayout rows cols)
  (m : ematrix et rows cols)
  : GTot (lseq et (rows * cols))
  = Seq.init_ghost (rows * cols) (fun i ->
      let (i,j) = l.bij.gg i in
      m.f i j)

inline_for_extraction noextract
val gpu_matrix (et:Type0) (rows cols : nat) (l : mlayout rows cols) : Type0

inline_for_extraction noextract
val core (#et #rows #cols #l : _) (g : gpu_matrix et rows cols l) : gpu_array et (rows * cols)

val gpu_matrix_pts_to
  (#et:Type) (#rows #cols : erased nat) (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et rows cols l)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix et rows cols)
  : slprop

unfold
instance has_pts_to (a:Type) (rows cols l : _)
  : has_pts_to (gpu_matrix a rows cols l) (ematrix a rows cols) = {
  pts_to = gpu_matrix_pts_to;
}

ghost
fn gpu_matrix_concr
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et rows cols l)
  (#em : ematrix et rows cols)
  requires
    g |-> em
  ensures
    core g |-> to_seq l em

ghost
fn gpu_matrix_abs
  (#et:Type)
  (#rows0 #cols0 : nat) (#l0 : mlayout rows0 cols0)
  (g : gpu_matrix et rows0 cols0 l0)
  (rows cols : nat) (l : mlayout rows cols)
  (#em : ematrix et rows cols)
  requires
    core g |-> to_seq l em
  returns
    g' : gpu_matrix et rows cols l
  ensures
    g' |-> em

inline_for_extraction noextract
fn gpu_matrix_alloc
  (#et:Type) {| sized et |}
  (rows cols : szp)
  (l : mlayout rows cols)
  preserves
    cpu
  requires
    pure (SZ.fits (rows * cols))
  returns
    gm : gpu_matrix et rows cols l
  ensures
    exists* em. gm |-> em

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
  (#em : _)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type) {| sized et |}
  (#rows #cols : szp)
  (#l : mlayout rows cols)
  (a : vec et)
  (gA : gpu_matrix et rows cols l)
  (#s : erased (seq et){ len s == rows * cols })
  preserves
    (a |-> s) **
    cpu
  requires
    (gA |-> 'm0) **
    pure (SZ.fits (rows * cols))
  ensures
    gA |-> from_seq l s

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type) {| sized et |}
  (#rows #cols : szp)
  (#l : mlayout rows cols)
  (a : vec et)
  (gA : gpu_matrix et rows cols l)
  (#m : ematrix et rows cols)
  preserves
    (gA |-> m) **
    cpu
  requires
    (a |-> 's0) **
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols)
  ensures
    a |-> to_seq l m

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
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
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
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
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l |}
  (gm : gpu_matrix et rows cols l)
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
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l |}
  (gm : gpu_matrix et rows cols l)
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
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et rows cols l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : natlt rows)
  ([@@@mkey]j : natlt cols)
  (v : et)
  : slprop

inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l |}
  (gm : gpu_matrix et rows cols l)
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
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l |}
  (gm : gpu_matrix et rows cols l)
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
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
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
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
  ensures
    gpu_matrix_pts_to gm #f em
