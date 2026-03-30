module Kuiper.Chest
#lang-pulse

(* An erased trivial container from any index type to
any element type, implemented via a ghost function. Exposes
a container instance. *)

open Kuiper
open Kuiper.Index
open Kuiper.Container
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

[@@erasable]
noeq
type chest (#r : nat) (d : idesc r) (et : Type) =
  | M : f:(abs d ^->> et) -> chest d et

let mk (#r : nat) (d : idesc r) (#et : Type)
  (f : abs d -> GTot et)
  : chest d et
  = M <| F.on_g _ f

let const (#r : nat) (d : idesc r) (#et : Type)
  (v:et)
  : chest d et
  = mk d fun _ -> v

let acc (#r : nat) (#d : idesc r) (#et : Type)
  (c : chest d et)
  (i : abs d)
  : GTot et
  = c.f i

let upd (#r : nat) (#d : idesc r) (#et : Type)
  (c : chest d et)
  (i : abs d)
  (v : et)
  : chest d et
  = mk d fun i' -> if i' = i then v else c.f i'

val acc_pat (#r : nat) (#d : idesc r) (#et : Type)
  (c : chest d et)
  (i : abs d)
  : Lemma (acc c i == c.f i)
          [SMTPat (c.f i)]

let matrix_comb (#r : nat) (#d : idesc r) (#et : Type)
  (f : binop et)
  (c1 c2 : chest d et)
  : chest d et
  = mk _ fun i -> f (acc c1 i) (acc c2 i)

val equal (#r : nat) (#d : idesc r) (#et : Type)
  (c1 c2 : chest d et)
  : prop

val lemma_equal_intro (#r : nat) (#d : idesc r) (#et : Type)
  (c1 c2 : chest d et)
  : Lemma (requires forall (i : abs d). acc c1 i == acc c2 i)
          (ensures equal c1 c2)
          [SMTPat (equal c1 c2)]

val ext (#r : nat) (#d : idesc r) (#et : Type)
  (c1 c2 : chest d et)
  : Lemma (requires equal c1 c2)
          (ensures c1 == c2)
          [SMTPat (equal c1 c2)]

(* Can this be hidden? *)
instance chest_is_container (#r : nat) (d : idesc r) (et : Type)
  : container (chest d et) (abs d) et =
{
  acc;
  upd;
  l1 = ez;
  l2 = ez;
  ext = (fun c1 c2 _ -> assert (equal c1 c2));
  from_fun = mk d;
  from_fun_ok = ez;
}