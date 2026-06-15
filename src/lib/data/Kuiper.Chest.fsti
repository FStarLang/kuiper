module Kuiper.Chest
#lang-pulse

(* An erased trivial container from any index type to
any element type, implemented via a ghost function. Exposes
a container instance. *)

open Kuiper
open Kuiper.Index
open Kuiper.Container
open Kuiper.Approximates.Base
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

[@@erasable]
noeq
type chest (#r : nat) (d : idesc r) (et : Type) =
  | M : f:(abs d ^->> et) -> chest d et

unfold let t (#r:nat) (d : idesc r) (et : Type) = chest d et

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

let chest_comb (#r : nat) (#d : idesc r) (#et : Type)
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

let chest_approximates #et
  {| scalar et, real_like et |}
  (#r : nat)
  (#d : idesc r)
  (c1 : chest d et)
  (c2 : chest d real)
  : prop
  = forall (i : abs d).
      acc c1 i %~ acc c2 i

instance chest_can_approximate
  (#et : Type0) {| scalar et, real_like et |}
  (#r : nat)
  (#d : idesc r)
  : can_approximate (chest d et) (chest d real) =
{
  approximates = chest_approximates;
}

let to_real_chest (#et : Type0)
  {| scalar et, real_like et |}
  (#r : nat)
  (#d : idesc r)
  (c : chest d et)
  : GTot (chest d real)
  = mk d fun i -> to_real (acc c i)

val lemma_to_real_chest_approximates (#et : Type0)
  {| scalar et, real_like et |}
  (#r : nat)
  (#d : idesc r)
  (c : chest d et)
  : Lemma (ensures c %~ to_real_chest c)
          [SMTPat (to_real_chest c)]

let chest_slice
  (#et : Type0) (#r : nat) (#d : idesc r)
  (i : natlt r) (j : natlt (d @! i))
  (s : chest d et)
  : chest (modulo_i i d) et
  = mk _ (fun (idx : abs (modulo_i i d)) ->
            acc s ((abs_bring_forward_bij i d).gg (j, idx)))

let chest_update_slice
  (#et : Type0) (#r : nat) (#d : idesc r)
  (i : natlt r) (j : natlt (d @! i))
  (s : chest d et) (s' : chest (modulo_i i d) et)
  : chest d et
  = mk _ (fun (idx : abs d) ->
            let (j', k) = (abs_bring_forward_bij i d).ff idx in
            if j' = j then acc s' k else acc s idx)
