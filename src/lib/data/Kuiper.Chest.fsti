module Kuiper.Chest
#lang-pulse

(* An erased trivial container from any index type to
any element type, implemented via a ghost function. Exposes
a container instance. *)

open Kuiper
open Kuiper.Shape
open Kuiper.Container
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality

[@@erasable]
noeq
type chest (#r : nat) (d : shape r) (et : Type) =
  | M : f:(abs d ^->> et) -> chest d et

unfold let t (#r:nat) (d : shape r) (et : Type) = chest d et

let mk (#r : nat) (d : shape r) (#et : Type)
  (f : abs d -> GTot et)
  : chest d et
  = M <| F.on_g _ f

let const (#r : nat) (d : shape r) (#et : Type)
  (v:et)
  : chest d et
  = mk d fun _ -> v

let acc (#r : nat) (#d : shape r) (#et : Type)
  (c : chest d et)
  (i : abs d)
  : GTot et
  = c.f i

let upd (#r : nat) (#d : shape r) (#et : Type)
  (c : chest d et)
  (i : abs d)
  (v : et)
  : chest d et
  = mk d fun i' -> if i' = i then v else c.f i'

val acc_pat (#r : nat) (#d : shape r) (#et : Type)
  (c : chest d et)
  (i : abs d)
  : Lemma (acc c i == c.f i)
          [SMTPat (c.f i)]

let chest_foralli (#r : nat) (#d : shape r) (#et : Type)
  (f : abs d -> et -> prop)
  (c : chest d et)
  : prop
  = forall i. f i (acc c i)

let chest_forall (#r : nat) (#d : shape r) (#et : Type)
  (f : et -> prop)
  (c : chest d et)
  : prop
  = chest_foralli (fun _ -> f) c

let chest_map (#r : nat) (#d : shape r) (#et : Type)
  (f : et -> et)
  (c : chest d et)
  : chest d et
  = mk _ fun i -> f (acc c i)

let chest_comb (#r : nat) (#d : shape r) (#et : Type)
  (f : binop et)
  (c1 c2 : chest d et)
  : chest d et
  = mk _ fun i -> f (acc c1 i) (acc c2 i)

val equal (#r : nat) (#d : shape r) (#et : Type)
  (c1 c2 : chest d et)
  : prop

val lemma_equal_intro (#r : nat) (#d : shape r) (#et : Type)
  (c1 c2 : chest d et)
  : Lemma (requires forall (i : abs d). acc c1 i == acc c2 i)
          (ensures equal c1 c2)
          [SMTPat (equal c1 c2)]

val ext (#r : nat) (#d : shape r) (#et : Type)
  (c1 c2 : chest d et)
  : Lemma (requires equal c1 c2)
          (ensures c1 == c2)
          [SMTPat (equal c1 c2)]

(* Can this be hidden? *)
instance chest_is_container (#r : nat) (d : shape r) (et : Type)
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
  (#d : shape r)
  (c1 : chest d et)
  (c2 : chest d real)
  : prop
  = forall (i : abs d).
      acc c1 i %~ acc c2 i

instance chest_can_approximate
  (#et : Type0) {| scalar et, real_like et |}
  (#r : nat)
  (#d : shape r)
  : can_approximate (chest d et) (chest d real) =
{
  approximates = chest_approximates;
}

let to_real_chest (#et : Type0)
  {| scalar et, real_like et |}
  (#r : nat)
  (#d : shape r)
  (c : chest d et)
  : GTot (chest d real)
  = mk d fun i -> to_real (acc c i)

val lemma_to_real_chest_approximates (#et : Type0)
  {| scalar et, real_like et |}
  (#r : nat)
  (#d : shape r)
  (c : chest d et)
  : Lemma (ensures c %~ to_real_chest c)
          [SMTPat (to_real_chest c)]

let chest_slice
  (#et : Type0) (#r : nat) (#d : shape r)
  (i : natlt r) (j : natlt (d @! i))
  (s : chest d et)
  : chest (modulo_i i d) et
  = mk _ (fun (idx : abs (modulo_i i d)) ->
            acc s ((abs_bring_forward_bij i d).gg (j, idx)))

let chest_update_slice
  (#et : Type0) (#r : nat) (#d : shape r)
  (i : natlt r) (j : natlt (d @! i))
  (s : chest d et) (s' : chest (modulo_i i d) et)
  : chest d et
  = mk _ (fun (idx : abs d) ->
            let (j', k) = (abs_bring_forward_bij i d).ff idx in
            if j' = j then acc s' k else acc s idx)

(* Rank-specific shortcuts *)

let chest1 et d = chest (d @| INil) et
let mk1 (#et:Type) (#d0 : nat)
  (f : natlt d0 -> GTot et)
  : chest1 et d0
  = mk (d0 @| INil) fun (i, ()) -> f i
let upd1 (#et:Type) (#d0 : nat)
  (s : chest1 et d0)
  (i0 : natlt d0)
  (x : et)
  : chest1 et d0
  = upd s (i0, ()) x

let chest2 et d1 d2 = chest (d1 @| d2 @| INil) et
let mk2 (#et:Type) (#d0 #d1 : nat)
  (f : natlt d0 -> natlt d1 -> GTot et)
  : chest2 et d0 d1
  = mk (d0 @| d1 @| INil) fun (i, (j, ())) -> f i j
let upd2 (#et:Type) (#d0 #d1 : nat)
  (s : chest2 et d0 d1)
  (i0 : natlt d0)
  (i1 : natlt d1)
  (x : et)
  : chest2 et d0 d1
  = upd s (i0, (i1, ())) x

let chest3 et d1 d2 d3 = chest (d1 @| d2 @| d3 @| INil) et
let mk3 (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  : chest3 et d0 d1 d2
  = mk (d0 @| d1 @| d2 @| INil) fun (i, (j, (k, ()))) -> f i j k
let upd3 (#et:Type) (#d0 #d1 #d2 : nat)
  (s : chest3 et d0 d1 d2)
  (i0 : natlt d0)
  (i1 : natlt d1)
  (i2 : natlt d2)
  (x : et)
  : chest3 et d0 d1 d2
  = upd s (i0, (i1, (i2, ()))) x

(* Extract / update a single "page" (the 2-D slice at batch index i). *)
let slice_page (#et:Type) (#d0 #d1 #d2 : nat)
  (m : chest3 et d0 d1 d2) (i : natlt d0)
  : chest2 et d1 d2
  = chest_slice 0 i m

let upd_page (#et:Type) (#d0 #d1 #d2 : nat)
  (m : chest3 et d0 d1 d2) (i : natlt d0)
  (p : chest2 et d1 d2)
  : chest3 et d0 d1 d2
  = chest_update_slice 0 i m p

val slice_upd_page_same (#et:Type) (#d0 #d1 #d2 : nat)
  (m : chest3 et d0 d1 d2) (i : natlt d0)
  (p : chest2 et d1 d2)
  : Lemma (ensures slice_page (upd_page m i p) i == p)
          [SMTPat (slice_page (upd_page m i p) i)]

val slice_upd_page_other (#et:Type) (#d0 #d1 #d2 : nat)
  (m : chest3 et d0 d1 d2) (i i' : natlt d0)
  (p : chest2 et d1 d2)
  : Lemma (requires i' <> i)
          (ensures slice_page (upd_page m i p) i' == slice_page m i')
          [SMTPat (slice_page (upd_page m i p) i')]

let chest4 et d1 d2 d3 d4 = chest (d1 @| d2 @| d3 @| d4 @| INil) et
let mk4 (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> GTot et)
  : chest4 et d0 d1 d2 d3
  = mk (d0 @| d1 @| d2 @| d3 @| INil)
      fun (i, (j, (k, (l, ())))) -> f i j k l
let upd4 (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (s : chest4 et d0 d1 d2 d3)
  (i0 : natlt d0)
  (i1 : natlt d1)
  (i2 : natlt d2)
  (i3 : natlt d3)
  (x : et)
  : chest4 et d0 d1 d2 d3
  = upd s (i0, (i1, (i2, (i3, ())))) x
