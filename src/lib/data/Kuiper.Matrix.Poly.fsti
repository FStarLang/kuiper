module Kuiper.Matrix.Poly
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Bijection
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

(* FIXME: The dimensions here must be erased so that, when we try
   to project, we can use erased values to call the projector.
   Otherwise calling l.c_to will incur a ghost effect! This is definitely
   not nice, and shows up below in the form of some annotations becoming
   needed. Is this something uniformly fixable in F*?
*)
noeq
type layout (rows cols : nat) = {
  bij     : erased ((natlt rows & natlt cols) =~ natlt (rows * cols));
  c_to    : (i:SZ.t{i < rows}) -> (j:SZ.t{j < cols}) -> r:SZ.t{SZ.v r == bij.ff (SZ.v i, SZ.v j)};
  c_from1 : (idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == fst (bij.gg (SZ.v idx))};
  c_from2 : (idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == snd (bij.gg (SZ.v idx))};
}

let from_seq (#et:Type) (#rows #cols : _)
  (l : layout rows cols)
  (s : lseq et (rows * cols))
  : ematrix et rows cols
  = M <| fun i j -> s @! l.bij.ff (i,j)

let to_seq (#et:Type) (#rows #cols : _)
  (l : layout rows cols)
  (m : ematrix et rows cols)
  : GTot (lseq et (rows * cols))
  = Seq.init_ghost (rows * cols) (fun i -> 
      let (i,j) = l.bij.gg i in
      m.f i j)

inline_for_extraction noextract
val gpu_matrix (et:Type0) (rows cols : nat) (l : layout rows cols) : Type0

val gpu_matrix_pts_to
  (#et:Type) (#rows #cols : erased nat) (#l : layout rows cols)
  ([@@@mkey] gm : gpu_matrix et rows cols l)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix et rows cols)
  : slprop

unfold
instance has_pts_to (a:Type) (rows cols l : _)
  : has_pts_to (gpu_matrix a rows cols l) (ematrix a rows cols) = {
  pts_to = gpu_matrix_pts_to;
}

inline_for_extraction noextract
fn gpu_matrix_alloc
  (#et:Type) {| scalar et |}
  (rows cols : szp)
  (l : layout rows cols)
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
  (#et:Type) {| scalar et |}
  (#rows #cols : erased nat)
  (#l : layout rows cols)
  (gm : gpu_matrix et rows cols l)
  (#em : _)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type) {| scalar et |}
  (#rows #cols : szp)
  (#l : layout rows cols)
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
  (#et:Type) {| scalar et |}
  (#rows #cols : szp)
  (#l : layout rows cols)
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
  (#l : layout rows cols)
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
  (#l : layout rows cols)
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
  (#l : layout rows cols)
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
  (#l : layout rows cols)
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
  (#l : layout rows cols)
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
  (#l : layout rows cols)
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
  (#l : layout rows cols)
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
  (#l : layout rows cols)
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
  (#l : layout rows cols)
  (gm : gpu_matrix et rows cols l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
  ensures
    gpu_matrix_pts_to gm #f em
