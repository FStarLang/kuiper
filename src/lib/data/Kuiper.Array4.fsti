module Kuiper.Array4

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Index
open Kuiper.EMatrix4
open FStar.Tactics.Typeclasses { no_method }
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let desc (d0 d1 d2 d3 : nat) : idesc 4 =
  d0 @| d1 @| d2 @| d3 @| INil

// Even if this is trivial, it seems to help in some contexts.
let sizeof_desc (d0 d1 d2 d3 : nat) : Lemma (sizeof (desc d0 d1 d2 d3) == d0 * d1 * d2 * d3)
          [SMTPat (sizeof (desc d0 d1 d2 d3))]
  = ()

let ait (d0 d1 d2 d3 : nat) = natlt d0 & natlt d1 & natlt d2 & natlt d3

let adapt_idx (#d0 #d1 #d2 #d3 : nat) (idx : abs (desc d0 d1 d2 d3)) : ait d0 d1 d2 d3 =
  match idx with
  | (i, (j, (k, (l, ())))) -> (i, j, k, l)

let adapt_idx_back (#d0 #d1 #d2 #d3 : nat) (idx : ait d0 d1 d2 d3) : abs (desc d0 d1 d2 d3) =
  match idx with
  | (i, j, k, l) -> (i, (j, (k, (l, ()))))

let raw_cit = sz & sz & sz & sz

let cit_fits (d0 d1 d2 d3 : nat) (idx : raw_cit) : prop =
  pi_4_0 idx < d0 /\ pi_4_1 idx < d1 /\ pi_4_2 idx < d2 /\ pi_4_3 idx < d3

[@@erasable]
type layout (d0 d1 d2 d3 : nat) = tlayout (desc d0 d1 d2 d3)

type full_layout (d0 d1 d2 d3 : nat) = l : layout d0 d1 d2 d3 { is_full l }

let from_seq (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (l : full_layout d0 d1 d2 d3)
  (s : lseq et (d0 * d1 * d2 * d3))
  : EMatrix4.t et d0 d1 d2 d3
  = EMatrix4.mkM (fun i j k m -> s `Seq.index` l.imap.f (i, (j, (k, (m, ())))))

let to_seq (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (l : full_layout d0 d1 d2 d3)
  (s : EMatrix4.t et d0 d1 d2 d3)
  : GTot (lseq et (d0 * d1 * d2 * d3))
  = Seq.init_ghost (d0 * d1 * d2 * d3) (fun i ->
      let x = Kuiper.Injection.inverse_f l.imap i in
      macc s x._1 x._2._1 x._2._2._1 x._2._2._2._1)

// Odd that this seems to help.
let to_seq_helper (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (l : full_layout d0 d1 d2 d3)
  (s : EMatrix4.t et d0 d1 d2 d3)
  (i : natlt (d0 * d1 * d2 * d3))
  : Lemma (to_seq l s `Seq.index` i == (let x = Kuiper.Injection.inverse_f l.imap i in macc s x._1 x._2._1 x._2._2._1 x._2._2._2._1))
          [SMTPat (to_seq l s `Seq.index` i)]
  = ()

val to_from (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (l : full_layout d0 d1 d2 d3) (s : lseq et (d0 * d1 * d2 * d3))
  : Lemma (ensures to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]

let layout_size (#d0 #d1 #d2 #d3 : nat) (l : layout d0 d1 d2 d3) : GTot nat = l.ulen

inline_for_extraction noextract
val t (et : Type0) (#d0 #d1 #d2 #d3 : nat) (l : layout d0 d1 d2 d3) : Type0

unfold let array4 = t

val is_global (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#d0 #d1 #d2 #d3 : erased nat)
  (l : layout d0 d1 d2 d3)
  (a : gpu_array et (layout_size l))
  : t et l

inline_for_extraction noextract
val core
  (#et : Type0) (#d0 #d1 #d2 #d3 : erased nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  : gpu_array et (layout_size l)

val lem_core_from_array
  (#et : Type) (#d0 #d1 #d2 #d3 : erased nat)
  (#l : layout d0 d1 d2 d3)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type) (#d0 #d1 #d2 #d3 : erased nat)
  (l : layout d0 d1 d2 d3)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat)
  (#l : layout d0 d1 d2 d3)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val pts_to
  (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : EMatrix4.t et d0 d1 d2 d3)
  : slprop

instance
val is_send_across_global
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l { is_global a })
  (#f : perm) (s : EMatrix4.t et d0 d1 d2 d3)
  : is_send_across gpu_of (pts_to a #f s)

unfold
instance has_pts_to_inst (et : Type) (d0 d1 d2 d3 : erased nat) (l : _)
  : has_pts_to (t et l) (EMatrix4.t et d0 d1 d2 d3)
  = { pts_to }

ghost
fn pts_to_ref
  (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  (#f : perm) (#s : erased (EMatrix4.t et d0 d1 d2 d3))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))

ghost
fn concr
  (#et:Type)
  (#d0 #d1 #d2 #d3 : nat)
  (#l : layout d0 d1 d2 d3 { is_full l })
  (g : t et l)
  (#s : EMatrix4.t et d0 d1 d2 d3)
  (#f : perm)
  requires
    g |-> Frac f s
  ensures
    core g |-> Frac f (to_seq l s)

ghost
fn abs
  (#et:Type)
  (#d0 #d1 #d2 #d3 : nat)
  (l : layout d0 d1 d2 d3 { is_full l })
  (p : gpu_array et (layout_size l))
  (#f : perm)
  (#s : EMatrix4.t et d0 d1 d2 d3)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s

ghost
fn abs'
  (#et:Type)
  (#d0 #d1 #d2 #d3 : nat)
  (l : layout d0 d1 d2 d3 { is_full l })
  (p : gpu_array et (layout_size l))
  (#f : perm)
  (#s : lseq et (layout_size l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)


ghost
fn share_n
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix4.t et d0 d1 d2 d3)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn gather_n
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix4.t et d0 d1 d2 d3)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#d0 #d1 #d2 #d3 : erased nat)
  (#l : layout d0 d1 d2 d3) {| ctlayout l |}
  (a : t et l)
  (ijkl : raw_cit{cit_fits d0 d1 d2 d3 ijkl})
  (#f : perm)
  (#s : erased (EMatrix4.t et d0 d1 d2 d3))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == EMatrix4.macc s (pi_4_0 ijkl) (pi_4_1 ijkl) (pi_4_2 ijkl) (pi_4_3 ijkl))

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#d0 #d1 #d2 #d3 : erased nat)
  (#l : layout d0 d1 d2 d3) {| ctlayout l |}
  (a : t et l)
  (ijkl : raw_cit{cit_fits d0 d1 d2 d3 ijkl})
  (v : et)
  (#s : erased (EMatrix4.t et d0 d1 d2 d3))
  requires
    a |-> s
  ensures
    a |-> EMatrix4.mupd s (pi_4_0 ijkl) (pi_4_1 ijkl) (pi_4_2 ijkl) (pi_4_3 ijkl) v

val pts_to_cell
  (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ijkl : ait d0 d1 d2 d3)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  : has_pts_to (cell (t et l) (ait d0 d1 d2 d3)) et
= {
  pts_to = (fun (Cell ar ijkl) #f v -> pts_to_cell ar #f ijkl v);
}

val pts_to_cell_eq
  (#et : Type) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l) (ijkl : ait d0 d1 d2 d3) (f : perm) (v : et)
  : Lemma (Cell a ijkl |-> Frac f v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f (adapt_idx_back ijkl)) v)

ghost
fn explode
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix4.t et d0 d1 d2 d3)
  requires a |-> Frac f s
  ensures
    forall+ (ijkl : ait d0 d1 d2 d3).
      Cell a ijkl |-> Frac f (EMatrix4.macc s (pi_4_0 ijkl) (pi_4_1 ijkl) (pi_4_2 ijkl) (pi_4_3 ijkl))

ghost
fn implode
  (#et : Type0) (#d0 #d1 #d2 #d3 : nat) (#l : layout d0 d1 d2 d3)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix4.t et d0 d1 d2 d3)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ijkl : ait d0 d1 d2 d3).
      Cell a ijkl |-> Frac f (EMatrix4.macc s (pi_4_0 ijkl) (pi_4_1 ijkl) (pi_4_2 ijkl) (pi_4_3 ijkl))
  ensures
    a |-> Frac f s

(* Syntax, in lieu of a typeclass *)
inline_for_extraction noextract
unfold let op_Array_Access
  (#et : Type0)
  (#d0 #d1 #d2 #d3 : erased nat)
  (#l : layout d0 d1 d2 d3) {| ctlayout l |}
  (a : t et l)
  (ijkl : raw_cit{cit_fits d0 d1 d2 d3 ijkl})
  (#f : perm)
  (#s : erased (EMatrix4.t et d0 d1 d2 d3))
  = read #et #d0 #d1 #d2 #d3 #l a ijkl #f #s

inline_for_extraction noextract
unfold let op_Array_Assignment
  (#et : Type0)
  (#d0 #d1 #d2 #d3 : erased nat)
  (#l : layout d0 d1 d2 d3) {| ctlayout l |}
  (a : t et l)
  (ijkl : raw_cit{cit_fits d0 d1 d2 d3 ijkl})
  (v : et)
  (#s : erased (EMatrix4.t et d0 d1 d2 d3))
  = write #et #d0 #d1 #d2 #d3 #l a ijkl v #s
