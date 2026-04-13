module Kuiper.Array2

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Index
open Kuiper.EMatrix
open FStar.Tactics.Typeclasses { no_method }
open Pulse.Lib.Trade
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let desc (rows cols : nat) : idesc 2 =
  rows @| cols @| INil

// Even if this is trivial, it seems to help in some contexts.
let sizeof_desc (rows cols : nat)
  : Lemma (sizeof (desc rows cols) == rows * cols)
          [SMTPat (sizeof (desc rows cols))]
  = ()

let ait (rows cols : nat) = natlt rows & natlt cols

let adapt_idx (#rows #cols : nat) (idx : abs (desc rows cols)) : ait rows cols =
  match idx with
  | (i, (j, ())) -> (i, j)

let adapt_idx_back (#rows #cols : nat) (idx : ait rows cols) : abs (desc rows cols) =
  match idx with
  | (i, j) -> (i, (j, ()))

let raw_cit = sz & sz

let cit_fits (rows cols : nat) (idx : raw_cit) : prop =
  pi_2_0 idx < rows /\ pi_2_1 idx < cols

type layout (rows cols : nat) = tlayout (desc rows cols)

type full_layout (rows cols : nat) = l : layout rows cols { is_full l }

let from_seq (#et:Type) (#m #n : nat)
  (l : full_layout m n )
  (s : lseq et (m * n))
  : ematrix et m n
  = EMatrix.mkM (fun i j -> s `Seq.index` l.imap.f (i, (j, ())))

let to_seq (#et:Type) (#m #n : nat)
  (l : full_layout m n)
  (s : ematrix et m n)
  : GTot (lseq et (m * n))
  = Seq.init_ghost (m * n) (fun i ->
      let x = Kuiper.Injection.inverse_f l.imap i in
      macc s x._1 x._2._1)

// Odd that this seems to help.
let to_seq_helper (#et:Type) (#m #n : nat)
  (l : full_layout m n)
  (s : ematrix et m n)
  (i : natlt (m * n))
  : Lemma (to_seq l s `Seq.index` i == (let x = Kuiper.Injection.inverse_f l.imap i in macc s x._1 x._2._1))
          [SMTPat (to_seq l s `Seq.index` i)]
  = ()

val to_from (#et:Type) (#m #n : nat)
  (l : full_layout m n) (s : lseq et (m * n))
  : Lemma (ensures to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]

let layout_size (#rows #cols : nat) (l : layout rows cols) : GTot nat = l.ulen

inline_for_extraction noextract
val t (et : Type0) (#rows #cols : nat) (l : layout rows cols) : Type0

unfold let array2 = t

val is_global (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#rows #cols : erased nat)
  (l : layout rows cols)
  (a : gpu_array et (layout_size l))
  : t et l

inline_for_extraction noextract
val core
  (#et : Type0) (#rows #cols : erased nat) (#l : layout rows cols)
  (a : t et l)
  : gpu_array et (layout_size l)

val lem_core_from_array
  (#et : Type) (#rows #cols : erased nat)
  (#l : layout rows cols)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type) (#rows #cols : erased nat)
  (l : layout rows cols)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val pts_to
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : ematrix et rows cols)
  : slprop

instance
val is_send_across_global
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l { is_global a })
  (#f : perm) (s : ematrix et rows cols)
  : is_send_across gpu_of (pts_to a #f s)

unfold
instance has_pts_to_inst (et : Type) (rows cols : erased nat) (l : _)
  : has_pts_to (t et l) (ematrix et rows cols)
  = { pts_to }

inline_for_extraction noextract
fn alloc0
  (#et:Type) {| sized et |}
  (rows cols : szp)
  (l : layout rows cols { is_full l })
  preserves
    cpu
  requires
    pure (SZ.fits (rows * cols))
  returns
    p : t et l
  ensures
    exists* em. on gpu_loc (p |-> em)
  ensures
    pure (is_global p)

inline_for_extraction noextract
fn free
  (#et:Type)
  (#rows #cols : erased nat)
  (#l : layout rows cols { is_full l })
  (p : t et l)
  (#em : ematrix et rows cols)
  preserves
    cpu
  requires
    on gpu_loc (p |-> em)
  ensures emp

ghost
fn pts_to_ref
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm) (#s : erased (ematrix et rows cols))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))

ghost
fn lower
  (#et:Type)
  (#rows #cols : nat)
  (#l : layout rows cols { is_full l })
  (g : t et l)
  (#s : ematrix et rows cols)
  (#f : perm)
  requires
    g |-> Frac f s
  ensures
    core g |-> Frac f (to_seq l s)

ghost
fn raise
  (#et:Type)
  (#rows #cols : nat)
  (l : layout rows cols { is_full l })
  (p : gpu_array et (layout_size l))
  (#f : perm)
  (#s : ematrix et rows cols)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s

ghost
fn raise'
  (#et:Type)
  (#rows #cols : nat)
  (l : layout rows cols { is_full l })
  (p : gpu_array et (layout_size l))
  (#f : perm)
  (#s : lseq et (layout_size l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)

ghost
fn share_n
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (k : pos)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn gather_n
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (k : pos)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

ghost
fn pts_to_eq
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f1 f2 : perm)
  (#s1 #s2 : ematrix et rows cols)
  requires
    a |-> Frac f1 s1 **
    a |-> Frac f2 s2
  ensures
    a |-> Frac f1 s2 **
    a |-> Frac f2 s2

ghost
fn gather_n_underspec
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (k : pos)
  (#f : perm)
  requires
    forall+ (_:natlt k).
      exists* (s : ematrix et rows cols). pts_to a #(f /. k) s
  ensures
    exists* (s : ematrix et rows cols). pts_to a #f s

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (#f : perm)
  (#s : erased (ematrix et rows cols))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == macc s (pi_2_0 ij) (pi_2_1 ij))

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (v : et)
  (#s : erased (ematrix et rows cols))
  requires
    a |-> s
  ensures
    a |-> mupd s (pi_2_0 ij) (pi_2_1 ij) v

val pts_to_cell
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ij : ait rows cols)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  : has_pts_to (cell (t et l) (ait rows cols)) et
= {
  pts_to = (fun (Cell ar ij) #f v -> pts_to_cell ar #f ij v);
}

val pts_to_cell_eq
  (#et : Type) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l) (ij : ait rows cols) (f : perm) (v : et)
  : Lemma (Cell a ij |-> Frac f v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f (adapt_idx_back ij)) v)

ghost
fn explode
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires a |-> Frac f s
  ensures
    forall+ (ij : ait rows cols).
      Cell a ij |-> Frac f (macc s (fst ij) (snd ij))

ghost
fn implode
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ij : ait rows cols).
      Cell a ij |-> Frac f (macc s (fst ij) (snd ij))
  ensures
    a |-> Frac f s

ghost
fn ilower
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      pts_to_cell a #f (r, c) (macc s r c))

ghost
fn iraise
  (#et : Type0) (#rows #cols : nat) (#l : layout rows cols)
  (a : t et l)
  (#f : perm)
  (#s : ematrix et rows cols)
  requires
    pure (SZ.fits (layout_size l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      pts_to_cell a #f (r, c) (macc s r c))
  ensures
    a |-> Frac f s

inline_for_extraction noextract
fn read_cell
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (#f : perm)
  (#s : erased et)
  preserves
    // Hideous.
    Cell a ((SZ.v (pi_2_0 ij) <: natlt rows),
            (SZ.v (pi_2_1 ij) <: natlt cols)) |-> Frac f s
  returns
    v : et
  ensures
    pure (v == s)

inline_for_extraction noextract
fn read_cell'
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (i : szlt rows) (j : szlt cols)
  (#f : perm)
  (#s : erased et)
  preserves
    // Hideous.
    Cell a ((i <: natlt rows),
            (j <: natlt cols)) |-> Frac f s
  returns
    v : et
  ensures
    pure (v == s)

inline_for_extraction noextract
fn write_cell
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (v : et)
  (#s : erased et)
  requires
    // Hideous.
    Cell a ((SZ.v (pi_2_0 ij) <: natlt rows),
            (SZ.v (pi_2_1 ij) <: natlt cols)) |-> s
  ensures
    Cell a ((SZ.v (pi_2_0 ij) <: natlt rows),
            (SZ.v (pi_2_1 ij) <: natlt cols)) |-> v

inline_for_extraction noextract
fn write_cell'
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (i : szlt rows) (j : szlt cols)
  (v : et)
  (#s : erased et)
  requires
    // Hideous.
    Cell a ((i <: natlt rows),
            (j <: natlt cols)) |-> s
  ensures
    Cell a ((i <: natlt rows),
            (j <: natlt cols)) |-> v

(* Syntax, in lieu of a typeclass *)
inline_for_extraction noextract
unfold let op_Array_Access
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (#f : perm)
  (#s : erased (ematrix et rows cols))
  = read #et #rows #cols #l a ij #f #s

inline_for_extraction noextract
unfold let op_Array_Assignment
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols) {| ctlayout l |}
  (a : t et l)
  (ij : raw_cit{cit_fits rows cols ij})
  (v : et)
  (#s : erased (ematrix et rows cols))
  = write #et #rows #cols #l a ij v #s

let row_layout
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : erased nat{i < rows})
  : Array1.layout cols
  = Tensor.tlayout_slice l 0 i

inline_for_extraction noextract
val row
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : erased nat{i < rows})
  : Array1.t et (row_layout a i)

ghost
fn extract_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : natlt rows)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    row a i |-> Frac f (ematrix_row s i) **
    (forall* (s' : lseq et cols).
      row a i |-> Frac f s' @==>
      a |-> Frac f (ematrix_upd_row s i s'))

ghost
fn extract_row_ro
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : natlt rows)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    factored
      (row a i |-> Frac f (ematrix_row s i))
      (a |-> Frac f s)

// Useful? This is just trade_elim
ghost
fn restore_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : natlt rows)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    factored
      (row a i |-> Frac f (ematrix_row s i))
      (a |-> Frac f s)
  ensures
    a |-> Frac f s

let col_layout
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : erased nat{i < cols})
  : Array1.layout rows
  = Tensor.tlayout_slice l 1 i

inline_for_extraction noextract
val col
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : erased nat{i < cols})
  : Array1.t et (col_layout a i)

ghost
fn extract_col
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : natlt cols)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    col a i |-> Frac f (ematrix_col s i) **
    (forall* (s' : lseq et rows).
      col a i |-> Frac f s' @==>
      a |-> Frac f (ematrix_upd_col s i s'))

ghost
fn extract_col_ro
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : natlt cols)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    factored
      (col a i |-> Frac f (ematrix_col s i))
      (a |-> Frac f s)

// Useful? This is just trade_elim
ghost
fn restore_col
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout rows cols)
  (a : t et l)
  (i : natlt cols)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    factored
      (col a i |-> Frac f (ematrix_col s i))
      (a |-> Frac f s)
  ensures
    a |-> Frac f s

fn copy_from_vec
  (#et:Type0) {| sized et |}
  (#rows #cols : sz)
  (#l : layout rows cols { is_full l })
  (gm : t et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    cpu ** a |-> s
  requires
    on gpu_loc (gm |-> em)
  ensures
    on gpu_loc (gm |-> from_seq l s)

inline_for_extraction noextract
fn copy_to_vec
  (#et:Type0) {| sized et |}
  (#rows #cols : sz)
  (#l : layout rows cols { is_full l })
  (a : vec et)
  (gm : t et l)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    cpu ** on gpu_loc (gm |-> em)
  requires
    a |-> s
  ensures
    a |-> to_seq l em
