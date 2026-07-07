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
  (f : abs d -> GTot et)
  (i : abs d)
  : Lemma (acc (mk d f) i == f i)
          [SMTPat (acc (mk d f) i)]

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

let chest_forallb (#r : nat) (#d : shape r) (#et : Type)
  (f : et -> GTot bool)
  (c : chest d et)
  : prop
  = chest_forall (fun x -> f x) c

let chest_map (#r : nat) (#d : shape r) (#et1 #et2 : Type)
  (f : et1 -> et2)
  (c : chest d et1)
  : chest d et2
  = mk _ fun i -> f (acc c i)

(* Could be implemented with map. *)
let chest_refine (#r : nat) (#d : shape r) (#et : Type)
  (p : et -> prop)
  (c : chest d et { forall i. p (acc c i) })
  : chest d (x:et{p x})
  = mk _ #(x:et{p x}) fun i -> acc c i

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
  ext = (fun c1 c2 _ -> assert (equal c1 c2));
  from_fun = mk d;
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
  = chest_map to_real c

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
let acc1 (#et:Type) (#d0 : nat)
  (s : chest1 et d0)
  (i0 : natlt d0)
  : GTot et
  = acc s (i0, ())
let chest1_to_seq
  (#et : Type)
  (#n : nat)
  (c : chest1 et n)
  : GTot (lseq et n)
  = Seq.init_ghost n (fun i -> acc1 c i)
let seq_to_chest1
  (#et : Type)
  (#n : nat)
  (s : lseq et n)
  : GTot (chest1 et n)
  = mk1 (fun i -> Seq.index s i)
let chest1_mapi (#n : nat) (#et1 #et2 : Type)
  (f : natlt n -> et1 -> et2)
  (c : chest1 et1 n)
  : chest1 et2 n
  = mk1 (fun i -> f i (acc1 c i))
let chest1_rsum #n (s : chest1 real n) : real =
  Kuiper.Seq.Common.seq_fold_left (+.) 0.0R (chest1_to_seq s)
  (* Do not use seq? *)
let chest1_sub
  (#et : Type0) (#n : nat)
  (i j : natle n{i <= j})
  (s : chest1 et n)
  : chest1 et (j-i)
  = mk1 (fun k -> acc1 s (i + k))
let chest1_append
  (#et : Type0) (#n #m : nat)
  (s1 : chest1 et n)
  (s2 : chest1 et m)
  : chest1 et (n + m)
  = mk1 #_ #(n+m) (fun i ->
      if i < n then acc1 s1 i
      else acc1 s2 (i - n))

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
let acc2 (#et:Type) (#d0 #d1 : nat)
  (s : chest2 et d0 d1)
  (i0 : natlt d0)
  (i1 : natlt d1)
  : GTot et
  = acc s (i0, (i1, ()))

let chest2_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  : chest1 et cols
  = chest_slice 0 i em

let chest2_col
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  (j : natlt cols)
  : chest1 et rows
  = chest_slice 1 j em

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
let acc3 (#et:Type) (#d0 #d1 #d2 : nat)
  (s : chest3 et d0 d1 d2)
  (i0 : natlt d0)
  (i1 : natlt d1)
  (i2 : natlt d2)
  : GTot et
  = acc s (i0, (i1, (i2, ())))

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
let acc4 (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (s : chest4 et d0 d1 d2 d3)
  (i0 : natlt d0)
  (i1 : natlt d1)
  (i2 : natlt d2)
  (i3 : natlt d3)
  : GTot et
  = acc s (i0, (i1, (i2, (i3, ()))))

(* Extract / update a single "page" (the 2-D slice at batch index i, j). *)
// TODO: this is defined a little differently than the 3D version.. probably should be more consistent?
let slice_page4 (#et:Type) (#d0 #d1 #d2 #d3: nat)
  (m : chest4 et d0 d1 d2 d3) (i : natlt d0) (j : natlt d1)
  : chest2 et d2 d3
  = chest_slice 0 i (chest_slice 1 j m)

let upd_page4 (#et:Type) (#d0 #d1 #d2 #d3: nat)
  (m : chest4 et d0 d1 d2 d3) (i : natlt d0) (j : natlt d1)
  (p : chest2 et d2 d3)
  : chest4 et d0 d1 d2 d3
  = mk4 fun i' j' ->
      if i' = i && j' = j
      then acc2 p
      else acc4 m i' j'
