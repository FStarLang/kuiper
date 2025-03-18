module Kuiper.View

#lang-pulse

open Kuiper
open Kuiper.Bijection
module F = FStar.FunctionalExtensionality

let to_from (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (s : lseq a len)
  : Lemma (ensures to_seq vw (from_seq vw s) == s)
          [SMTPat (to_seq vw (from_seq vw s))]
  = let _ = vw.igm.bij.gg (F.on_g vw.it <| fun i -> s @! it_to_nat vw i) in
    (* funny, mentioning the term above (= from_seq vw s) makes the proof work. *)
    assert (Seq.equal s (to_seq vw (from_seq vw s)))

let to_seq_upd (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (v : vt)
  (i : vw.it)
  (x : a)
  : Lemma (ensures to_seq vw (vw.igm.upd v i x) == Seq.upd (to_seq vw v) (it_to_nat vw i) x)
          [SMTPat (to_seq vw (vw.igm.upd v i x))]
  = assert (to_seq vw (vw.igm.upd v i x) `Seq.equal` Seq.upd (to_seq vw v) (it_to_nat vw i) x)
