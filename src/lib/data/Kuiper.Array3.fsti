module Kuiper.Array3

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Index
open Kuiper.EMatrix { ematrix }
open Pulse.Lib.Trade
open FStar.Tactics.Typeclasses { no_method }
module B = Kuiper.Array
module Array2 = Kuiper.Array2
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let desc (d0 d1 d2 : nat) : idesc 3 =
  d0 @| d1 @| d2 @| INil

// Even if this is trivial, it seems to help in some contexts.
let sizeof_desc (d0 d1 d2 : nat)
  : Lemma (sizeof (desc d0 d1 d2) == d0 * d1 * d2)
          [SMTPat (sizeof (desc d0 d1 d2))]
  = ()

let ait (d0 d1 d2 : nat) = natlt d0 & natlt d1 & natlt d2

let adapt_idx (#d0 #d1 #d2 : nat) (idx : abs (desc d0 d1 d2)) : ait d0 d1 d2 =
  match idx with
  | (i, (j, (k, ()))) -> (i, j, k)

let adapt_idx_back (#d0 #d1 #d2 : nat) (idx : ait d0 d1 d2) : abs (desc d0 d1 d2) =
  match idx with
  | (i, j, k) -> (i, (j, (k, ())))

let raw_cit = sz & sz & sz

let cit_fits (d0 d1 d2 : nat) (idx : raw_cit) : prop =
  pi_3_0 idx < d0 /\ pi_3_1 idx < d1 /\ pi_3_2 idx < d2

[@@erasable]
type layout (d0 d1 d2 : nat) = tlayout (desc d0 d1 d2)

type full_layout (d0 d1 d2 : nat) = l : layout d0 d1 d2 { is_full l }

let from_seq (#et:Type) (#d0 #d1 #d2 : nat)
  (l : full_layout d0 d1 d2)
  (s : lseq et (d0 * d1 * d2))
  : EMatrix3.t et d0 d1 d2
  = EMatrix3.mkM (fun i j k -> s `Seq.index` l.imap.f (i, (j, (k, ()))))

let to_seq (#et:Type) (#d0 #d1 #d2 : nat)
  (l : full_layout d0 d1 d2)
  (s : EMatrix3.t et d0 d1 d2)
  : GTot (lseq et (d0 * d1 * d2))
  = Seq.init_ghost (d0 * d1 * d2) (fun i ->
      let x = Kuiper.Injection.inverse_f l.imap i in
      EMatrix3.macc s x._1 x._2._1 x._2._2._1)

// Odd that this seems to help.
let to_seq_helper (#et:Type) (#d0 #d1 #d2 : nat)
  (l : full_layout d0 d1 d2)
  (s : EMatrix3.t et d0 d1 d2)
  (i : natlt (d0 * d1 * d2))
  : Lemma (to_seq l s `Seq.index` i == (let x = Kuiper.Injection.inverse_f l.imap i in EMatrix3.macc s x._1 x._2._1 x._2._2._1))
          [SMTPat (to_seq l s `Seq.index` i)]
  = ()

val to_from (#et:Type) (#d0 #d1 #d2 : nat)
  (l : full_layout d0 d1 d2) (s : lseq et (d0 * d1 * d2))
  : Lemma (ensures to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]

let layout_size (#d0 #d1 #d2 : nat) (l : layout d0 d1 d2) : GTot nat = l.ulen

inline_for_extraction noextract
val t (et : Type0) (#d0 #d1 #d2 : nat) (l : layout d0 d1 d2) : Type0

unfold let array3 = t

val is_global (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#d0 #d1 #d2 : erased nat)
  (l : layout d0 d1 d2)
  (a : larray et (layout_size l))
  : t et l

inline_for_extraction noextract
val core
  (#et : Type0) (#d0 #d1 #d2 : erased nat) (#l : layout d0 d1 d2)
  (a : t et l)
  : larray et (layout_size l)

val lem_core_from_array
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (l : layout d0 d1 d2)
  (p : larray et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val pts_to
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : EMatrix3.t et d0 d1 d2)
  : slprop

instance
val is_send_across_global
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l { is_global a })
  (#f : perm) (s : EMatrix3.t et d0 d1 d2)
  : is_send_across gpu_of (pts_to a #f s)

unfold
instance has_pts_to_inst (et : Type) (d0 d1 d2 : erased nat) (l : _)
  : has_pts_to (t et l) (EMatrix3.t et d0 d1 d2)
  = { pts_to }

ghost
fn pts_to_ref
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm) (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))

ghost
fn pts_to_ref_located
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#loc : _)
  (#f : perm) (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (layout_size l))

inline_for_extraction noextract
fn alloc0
  (#et:Type) {| sized et |}
  (d0 d1 d2 : szp)
  (l : layout d0 d1 d2 { is_full l })
  preserves
    cpu
  requires
    pure (SZ.fits (layout_size l))
  returns
    p : t et l
  ensures
    exists* em. on gpu_loc (p |-> em)
  ensures
    pure (is_global p)

inline_for_extraction noextract
fn free
  (#et:Type)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2 { is_full l })
  (p : t et l)
  (#em : EMatrix3.t et d0 d1 d2)
  preserves
    cpu
  requires
    on gpu_loc (p |-> em)
  ensures emp

ghost
fn lower
  (#et:Type)
  (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2 { is_full l })
  (g : t et l)
  (#s : EMatrix3.t et d0 d1 d2)
  (#f : perm)
  requires
    g |-> Frac f s
  ensures
    core g |-> Frac f (to_seq l s)

ghost
fn raise
  (#et:Type)
  (#d0 #d1 #d2 : nat)
  (l : layout d0 d1 d2 { is_full l })
  (p : larray et (layout_size l))
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s

ghost
fn raise'
  (#et:Type)
  (#d0 #d1 #d2 : nat)
  (l : layout d0 d1 d2 { is_full l })
  (p : larray et (layout_size l))
  (#f : perm)
  (#s : lseq et (layout_size l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)

inline_for_extraction noextract
fn copy_from_vec
  (#et:Type0) {| sized et |}
  (#d0 #d1 #d2 : szp)
  (#l : layout d0 d1 d2 { is_full l })
  (gm : t et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == d0 * d1 * d2})
  (#em : EMatrix3.t et d0 d1 d2)
  preserves
    cpu ** a |-> s
  requires
    on gpu_loc (gm |-> em)
  ensures
    on gpu_loc (gm |-> from_seq l s)

inline_for_extraction noextract
fn copy_to_vec
  (#et:Type0) {| sized et |}
  (#d0 #d1 #d2 : szp)
  (#l : layout d0 d1 d2 { is_full l })
  (a : vec et)
  (gm : t et l)
  (#s : erased (seq et){Seq.length s == d0 * d1 * d2})
  (#em : EMatrix3.t et d0 d1 d2)
  preserves
    cpu ** on gpu_loc (gm |-> em)
  requires
    a |-> s
  ensures
    a |-> to_seq l em

ghost
fn share_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn gather_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| ctlayout l |}
  (a : t et l)
  (ijk : raw_cit{cit_fits d0 d1 d2 ijk})
  (#f : perm)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| ctlayout l |}
  (a : t et l)
  (ijk : raw_cit{cit_fits d0 d1 d2 ijk})
  (v : et)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  requires
    a |-> s
  ensures
    a |-> EMatrix3.mupd s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk) v

val pts_to_cell
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ijk : ait d0 d1 d2)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  : has_pts_to (cell (t et l) (ait d0 d1 d2)) et
= {
  pts_to = (fun (Cell ar ijk) #f v -> pts_to_cell ar #f ijk v);
}

val pts_to_cell_eq
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (ijk : ait d0 d1 d2) (f : perm) (v : et)
  : Lemma (Cell a ijk |-> Frac f v
           ==
           B.pts_to_cell (core a) #f (l.imap.f (adapt_idx_back ijk)) v)

ghost
fn explode
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires a |-> Frac f s
  ensures
    forall+ (ijk : ait d0 d1 d2).
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))

ghost
fn implode
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ijk : ait d0 d1 d2).
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))
  ensures
    a |-> Frac f s

(* Syntax, in lieu of a typeclass *)
inline_for_extraction noextract
unfold let op_Array_Access
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| ctlayout l |}
  (a : t et l)
  (ijk : raw_cit{cit_fits d0 d1 d2 ijk})
  (#f : perm)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  = read #et #d0 #d1 #d2 #l a ijk #f #s

inline_for_extraction noextract
unfold let op_Array_Assignment
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| ctlayout l |}
  (a : t et l)
  (ijk : raw_cit{cit_fits d0 d1 d2 ijk})
  (v : et)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  = write #et #d0 #d1 #d2 #l a ijk v #s

(* ---- page: extract a 2-D slice (Array2) from a 3-D tensor (Array3) ---- *)

let page_layout
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  (i : erased nat{i < d0})
  : Array2.layout d1 d2
  = Tensor.tlayout_slice l 0 i

inline_for_extraction noextract
val page
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  (i : erased nat{i < d0})
  : Array2.t et (page_layout a i)

val page_is_global
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (i : erased nat{i < d0})
  : Lemma (ensures Array2.is_global (page a i) <==> is_global a)
          [SMTPat (page a i)]

ghost
fn extract_page
  (#et : Type0)
  (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  (i : natlt d0)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    page a i |-> Frac f (EMatrix3.slice_page s i) **
    (forall* (s' : ematrix et d1 d2).
      page a i |-> Frac f s' @==>
      a |-> Frac f (EMatrix3.upd_page s i s'))

ghost
fn extract_page_ro
  (#et : Type0)
  (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  (i : natlt d0)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    factored
      (page a i |-> Frac f (EMatrix3.slice_page s i))
      (a |-> Frac f s)
