module Kuiper.Matrix
#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Common
open Kuiper.Matrix.Reprs.Type
module T = FStar.Tactics.V2
module SZ = Kuiper.SizeT

(* Move? *)
inline_for_extraction noextract
let clayout_imap
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (c : clayout l)
  : szlt rows & szlt cols -> szlt l.len
  = fun (i, j) -> c.c_to i j

inline_for_extraction noextract
instance cview_from_clayout
  (et : Type)
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (c : clayout l)
  : IView.ciview (aview_from_mlayout et l).iview =
  // I would like to just say:
  // : View.cview (aview_from_mlayout et l) =
  // But F* complains it's not a class.
{
  clen = c.m_len;

  sch = {
    cit = szlt rows & szlt cols;
    bij = Bijection.bij_prod (Bijection.fin_size_t_bij _) (Bijection.fin_size_t_bij _);
  };

  step = {
    cimap = Kuiper.Injection.mk_cinj (clayout_imap c);
    compat = ez;
  };
}

inline_for_extraction noextract
val gpu_matrix (et:Type0) (#rows #cols : nat) (l : mlayout rows cols) : Type0

inline_for_extraction noextract
val from_array
  (#a : Type0)
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (arr : gpu_array a (mlayout_size l))
  : gpu_matrix a l

inline_for_extraction noextract
val core
  (#et : Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  : gpu_array et (mlayout_size l)

val lem_core_from_array
  (#et : Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  : Lemma (ensures from_array l (core g) == g)
          [SMTPat (core g)]

val lem_from_array_core
  (#et : Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (p : gpu_array et (mlayout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val gpu_matrix_pts_to
  (#et:Type) (#rows #cols : nat) (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix et rows cols)
  : slprop

(* erased is important for the lens! *)
unfold
instance has_pts_to_matrix (a:Type) (rows cols : erased nat) (l : _)
  : has_pts_to (gpu_matrix a l) (ematrix a rows cols) = {
  pts_to = gpu_matrix_pts_to;
}

ghost
fn gpu_matrix_pts_to_ref
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  preserves
    gpu_matrix_pts_to g #f em
  ensures
    pure (SZ.fits (rows * cols))

ghost
fn gpu_matrix_concr
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols { is_full_layout l })
  (g : gpu_matrix et l)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    g |-> Frac f em
  ensures
    core g |-> Frac f (to_seq l em)

ghost
fn gpu_matrix_abs
  (#et:Type)
  (#rows #cols : nat)
  (l : mlayout rows cols { is_full_layout l })
  (p : gpu_array et (mlayout_size l))
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    p |-> Frac f (to_seq l em)
  ensures
    from_array l p |-> Frac f em

ghost
fn gpu_matrix_abs'
  (#et:Type)
  (#rows #cols : nat)
  (l : mlayout rows cols { is_full_layout l })
  (p : gpu_array et (mlayout_size l))
  (#f : perm)
  (#s : lseq et (mlayout_size l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)

(* This version does not require a full_layout. *)
ghost
fn gpu_matrix_iconcr
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    g |-> Frac f em
  ensures
    pure (SZ.fits (mlayout_size l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      gpu_pts_to_cell (core g) #f (cell_of_pos l r c) (macc em r c))

ghost
fn gpu_matrix_iabs
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      gpu_pts_to_cell (core g) #f (cell_of_pos l r c) (macc em r c))
  ensures
    g |-> Frac f em

inline_for_extraction noextract
fn gpu_matrix_alloc0
  (#et:Type) {| sized et |}
  (rows cols : szp)
  (l : mlayout rows cols { is_full_layout l })
  preserves
    cpu
  requires
    pure (SZ.fits (rows * cols))
  returns
    gm : gpu_matrix et l
  ensures
    exists* em. gm |-> em

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols { is_full_layout l })
  (gm : gpu_matrix et l)
  (#em : ematrix et rows cols)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ (_:natlt k). gpu_matrix_pts_to gm #(f /. k) em

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    forall+ (_:natlt k). gpu_matrix_pts_to gm #(f /. k) em
  ensures
    gpu_matrix_pts_to gm #f em

ghost
fn gpu_matrix_pts_to_eq
  (#et : Type u#0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (m : gpu_matrix et l)
  (#f1 f2 : perm)
  (#em1 #em2 : ematrix et rows cols)
  requires
    gpu_matrix_pts_to m #f1 em1 **
    gpu_matrix_pts_to m #f2 em2
  ensures
    gpu_matrix_pts_to m #f1 em2 **
    gpu_matrix_pts_to m #f2 em2

ghost
fn gpu_matrix_gather_n_underspec
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  requires
    forall+ (_:natlt k).
      exists* (em: ematrix et rows cols). gpu_matrix_pts_to gm #(f /. k) em
  ensures
    exists* (em : ematrix et rows cols). gpu_matrix_pts_to gm #f em

ghost
fn gpu_matrix_share_2
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#em : ematrix et rows cols)
  requires
    gm |-> em
  ensures
    (gm |-> Frac 0.5R em) ** (gm |-> Frac 0.5R em)

ghost
fn gpu_matrix_gather_2
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#em : ematrix et rows cols)
  requires
    (gm |-> Frac 0.5R em) ** (gm |-> Frac 0.5R em)
  ensures
    gm |-> em

inline_for_extraction noextract
fn gpu_matrix_read
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt cols)
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
  (gm : gpu_matrix et l)
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
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : natlt rows)
  ([@@@mkey]j : natlt cols)
  (v : et)
  : slprop

val gpu_matrix_pts_to_cell_eq
  (#et:Type) (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (i : natlt rows)
  (j : natlt cols)
  (f : perm)
  (v : et)
  : Lemma (gpu_matrix_pts_to_cell gm #f i j v
           ==
           gpu_pts_to_cell (core gm) #f (cell_of_pos l i j) v)

inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l |}
  (gm : gpu_matrix et l)
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
  (gm : gpu_matrix et l)
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
  (gm : gpu_matrix et l)
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
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
  ensures
    gpu_matrix_pts_to gm #f em

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type0) {| sized et |}
  (#rows #cols : SZ.t)
  (#l : mlayout rows cols { is_full_layout l })
  (gm : gpu_matrix et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    a |-> s **
    cpu
  requires
    gm |-> em
  ensures
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols) **
    (gm |-> from_seq l s)

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type0) {| sized et |}
  (#rows #cols : SZ.t)
  (#l : mlayout rows cols { is_full_layout l })
  (a : vec et)
  (gm : gpu_matrix et l)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    gm |-> em **
    cpu
  requires
    a |-> s
  ensures
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols) **
    (a |-> to_seq l em)
