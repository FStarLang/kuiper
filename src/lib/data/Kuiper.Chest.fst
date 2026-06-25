module Kuiper.Chest
#lang-pulse

(* An erased trivial container from any index type to
any element type, implemented via a ghost function. Exposes
a container instance. *)

open Kuiper
open Kuiper.Shape
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

let acc_pat c i = ()

let equal (#r : nat) (#d : shape r) (#et : Type)
  (c1 c2 : chest d et)
  : prop
  = forall i. acc c1 i == acc c2 i

let lemma_equal_intro (#r : nat) (#d : shape r) (#et : Type)
  (c1 c2 : chest d et)
  : Lemma (requires forall (i : abs d). acc c1 i == acc c2 i)
          (ensures equal c1 c2)
          [SMTPat (equal c1 c2)]
  = ()

let ext (#r : nat) (#d : shape r) (#et : Type)
  (c1 c2 : chest d et)
  : Lemma (requires equal c1 c2)
          (ensures c1 == c2)
          [SMTPat (equal c1 c2)]
  = F.extensionality_g _ _ c1.f c2.f

let lemma_to_real_chest_approximates (#et : Type0)
  {| scalar et, real_like et |}
  (#r : nat)
  (#d : shape r)
  (c : chest d et)
  : Lemma (ensures c %~ to_real_chest c)
          [SMTPat (to_real_chest c)]
  = ()

let slice_upd_page_same (#et:Type) (#d0 #d1 #d2 : nat)
  (m : chest3 et d0 d1 d2) (i : natlt d0)
  (p : chest2 et d1 d2)
  : Lemma (ensures slice_page (upd_page m i p) i == p)
          [SMTPat (slice_page (upd_page m i p) i)]
  = assert (equal (slice_page (upd_page m i p) i) p)

let slice_upd_page_other (#et:Type) (#d0 #d1 #d2 : nat)
  (m : chest3 et d0 d1 d2) (i i' : natlt d0)
  (p : chest2 et d1 d2)
  : Lemma (requires i' <> i)
          (ensures slice_page (upd_page m i p) i' == slice_page m i')
          [SMTPat (slice_page (upd_page m i p) i')]
  = assert (equal (slice_page (upd_page m i p) i') (slice_page m i'))
