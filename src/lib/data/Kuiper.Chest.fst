module Kuiper.Chest
#lang-pulse

(* An erased trivial container from any index type to
any element type, implemented via a ghost function. Exposes
a container instance. *)

open Kuiper
open Kuiper.Index
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

let acc_pat c i = ()

let equal (#r : nat) (#d : idesc r) (#et : Type)
  (c1 c2 : chest d et)
  : prop
  = forall i. acc c1 i == acc c2 i

let lemma_equal_intro (#r : nat) (#d : idesc r) (#et : Type)
  (c1 c2 : chest d et)
  : Lemma (requires forall (i : abs d). acc c1 i == acc c2 i)
          (ensures equal c1 c2)
          [SMTPat (equal c1 c2)]
  = ()

let ext (#r : nat) (#d : idesc r) (#et : Type)
  (c1 c2 : chest d et)
  : Lemma (requires equal c1 c2)
          (ensures c1 == c2)
          [SMTPat (equal c1 c2)]
  = F.extensionality_g _ _ c1.f c2.f