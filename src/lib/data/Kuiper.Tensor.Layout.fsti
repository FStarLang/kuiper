module Kuiper.Tensor.Layout

open Kuiper
open Kuiper.Injection
open Kuiper.Index
open Kuiper.Chest
open FStar.Tactics.Typeclasses { no_method }
module V = Kuiper.View
module SZ = Kuiper.SizeT

[@@erasable]
noeq
type tlayout (#r : erased nat) (d : idesc r) = {
  (* Underlying length of base array (Kuiper.Array) *)
  ulen : nat;
  (* Injection from (abstract) index space into base array. *)
  imap : abs d @~> natlt ulen;
}

let is_full (#r : nat) (#d : idesc r) (l : tlayout d) : prop =
  is_surj l.imap.f

let full_tlayout (#r : nat) (d : idesc r) =
  l : tlayout d { is_full l }

(* The underlying array must be large enough to hold all the elements of the tensor. *)
val full_layout_size_lt (#r : nat) (#d : idesc r) (l : tlayout d)
  : Lemma (ensures  l.ulen >= sizeof d)
          [SMTPat (has_type l (tlayout d))]

(* When the layout is full, the underlying array is exactly the size of the tensor. *)
val full_layout_size (#r : nat) (#d : idesc r) (l : tlayout d)
  : Lemma (requires is_full l)
          (ensures l.ulen == sizeof d)
          [SMTPat (is_full l)]

(* And vice versa: when the underlying array is exactly the size of the tensor, the layout is full. *)
val full_layout_size' (#r : nat) (#d : idesc r) (l : tlayout d)
  : Lemma (requires l.ulen == sizeof d)
          (ensures is_full l)
          [SMTPat (is_full l)]

let tlayout_ulen (#d : idesc 'r) (l : tlayout d) : GTot nat = l.ulen
let tlayout_size (#d : idesc 'r) (l : tlayout d) : GTot nat = sizeof d

val size_le_ulen (#r : nat) (#d : idesc r) (l : tlayout d)
  : Lemma (ensures tlayout_size l <= tlayout_ulen l)
          [SMTPat (tlayout_size l); SMTPat (tlayout_ulen l)]

[@@Tactics.Typeclasses.fundeps [0;1]]
inline_for_extraction
class ctlayout (#r : erased nat) (#d : idesc r) (l : tlayout d) = {
  ulen_fits : squash (SZ.fits l.ulen);

  [@@@no_method]
  all_fit : squash (all_fit d);

  [@@@no_method]
  cimap : i:conc d -> r:SZ.t{SZ.v r == l.imap.f (up i)};
}

val ctlayout_must_fit (#r : nat) (#d : idesc r) (#l : tlayout d)
  (c : ctlayout l)
  : Lemma (ensures SZ.fits (sizeof d))
          [SMTPat (has_type c (ctlayout #r #d l))]

let tensor_aview (et : Type) (#r : nat) (#d : idesc r) (l : tlayout d)
  : V.aview et (chest d et)
  = {
      iview = {
        len = l.ulen;
        ait = abs d;
        step = { imap = l.imap; };
      };
      ctn = solve;
    }

let from_seq (#et:Type) (#r : nat) (#d : idesc r)
  (l : full_tlayout d)
  (s : lseq et (sizeof d))
  : chest d et
  = Kuiper.Chest.mk d (fun i -> s `Seq.index` l.imap.f i)

let to_seq (#et:Type) (#r : nat) (#d : idesc r)
  (l : full_tlayout d)
  (m : chest d et)
  : GTot (lseq et (sizeof d))
  = Seq.init_ghost (sizeof d) (fun i -> m.f (Kuiper.Injection.inverse_f l.imap i))

let from_seq_rel (#et:Type) (#r : nat) (#d : idesc r) (l : full_tlayout d)
  (s : lseq et (sizeof d))
  : Lemma (from_seq l s == V.from_seq (tensor_aview et l) s)
  = assert (Kuiper.Chest.equal (from_seq l s) (V.from_seq (tensor_aview et l) s))

let to_seq_rel (#et:Type) (#r : nat) (#d : idesc r) (l : full_tlayout d)
  (s : chest d et)
  : Lemma (to_seq l s == V.to_seq (tensor_aview et l) s)
  = assert (Seq.equal (to_seq l s) (V.to_seq (tensor_aview et l) s))

let to_from (#et:Type) (#r : nat) (#d : idesc r)
  (l : full_tlayout d) (s : lseq et (sizeof d))
  : Lemma (ensures to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]
  = from_seq_rel l s;
    to_seq_rel l (from_seq l s);
    ()

(* Rank-specific shortcuts *)

let layout1 d = tlayout (d @| INil)
let chest1 et d = chest (d @| INil) et
val size_layout_1 (d : nat)
  : Lemma (sizeof (d @| INil) == d)
          [SMTPat (sizeof (d @| INil))]

let layout2 d1 d2 = tlayout (d1 @| d2 @| INil)
let chest2 et d1 d2 = chest (d1 @| d2 @| INil) et
val size_layout_2 (d1 d2 : nat)
  : Lemma (sizeof (d1 @| d2 @| INil) == d1 * d2)
          [SMTPat (sizeof (d1 @| d2 @| INil))]

let layout3 d1 d2 d3 = tlayout (d1 @| d2 @| d3 @| INil)
let chest3 et d1 d2 d3 = chest (d1 @| d2 @| d3 @| INil) et
val size_layout_3 (d1 d2 d3 : nat)
  : Lemma (sizeof (d1 @| d2 @| d3 @| INil) == d1 * d2 * d3)
          [SMTPat (sizeof (d1 @| d2 @| d3 @| INil))]

let layout4 d1 d2 d3 d4 = tlayout (d1 @| d2 @| d3 @| d4 @| INil)
let chest4 et d1 d2 d3 d4 = chest (d1 @| d2 @| d3 @| d4 @| INil) et
val size_layout_4 (d1 d2 d3 d4 : nat)
  : Lemma (sizeof (d1 @| d2 @| d3 @| d4 @| INil) == d1 * d2 * d3 * d4)
          [SMTPat (sizeof (d1 @| d2 @| d3 @| d4 @| INil))]

(* Matrix representations (families). Ideally this should also be
   polymorphic in tensor dimensionality. *)

let trepr2 = m:nat -> n:nat -> tlayout (m @| n @| INil)

inline_for_extraction noextract
class ctrepr2 (f : trepr2) = {
  inst : m:sz -> n:sz -> #_:squash(SZ.fits (m*n)) ->
         ctlayout (f m n);
}

inline_for_extraction noextract
instance ctrepr2_gives_ctlayout2
  (f : trepr2)
  (m n : sz)
  (#_ : squash (SZ.fits (m*n)))
  {| d : ctrepr2 f |}
  : ctlayout (f m n) = d.inst m n
