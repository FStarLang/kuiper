module Kuiper.ForEvery

#lang-pulse
open Pulse
open Kuiper.Common
open Kuiper.Bijection
open Kuiper.Enumerable
open Pulse.Lib.BigStar
open Pulse.Lib.Trade

val forevery
  (a:Type) {| enumerable a |}
  (f : a -> slprop)
  : slprop

unfold
let ( forall+ )
  (#a:Type) {| enumerable a |}
  (f : a -> slprop)
  : slprop = forevery a f

val forevery_ext_lem
  (#a:Type0) {| enumerable a |}
  (f : a -> slprop)
  (g : a -> slprop { forall x. f x == g x })
  : Lemma (ensures (forall+ (x:a). f x) == (forall+ (x:a). g x))

ghost
fn forevery_ext
  (#a:Type0) {| enumerable a |}
  (f : a -> slprop)
  (g : a -> slprop { forall x. f x == g x })
  requires
    forall+ (x:a). f x
  ensures
    forall+ (x:a). g x

ghost
fn forevery_ext_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  (g : a -> b -> slprop)
  requires
    pure (forall x y. f x y == g x y)
  requires
    forall+ (x:a) (y:b). f x y
  ensures
    forall+ (x:a) (y:b). g x y

ghost
fn forevery_flatten
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). f x y
  ensures
    forall+ (xy : a & b). f xy._1 xy._2

ghost
fn forevery_flatten'
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a & b -> slprop)
  requires
    forall+ (x:a) (y:b). f (x, y)
  ensures
    forall+ (xy : a & b). f xy

ghost
fn forevery_unflatten
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  requires
    forall+ (xy : a & b). f xy._1 xy._2
  ensures
    forall+ (x:a) (y:b). f x y

ghost
fn forevery_unflatten'
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a & b -> slprop)
  requires
    forall+ (xy : a & b). f xy
  ensures
    forall+ (x:a) (y:b). f (x, y)

ghost
fn forevery_iso
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (bij : erased (a =~ b))
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (y:b). p (bij.gg y)

ghost
fn forevery_iso_back
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (bij : erased (a =~ b))
  (p : a -> slprop)
  requires
    forall+ (y:b). p (bij.gg y)
  ensures
    forall+ (x:a). p x

(* Normally not needed... *)
ghost
fn forevery_permute
  (#a:Type0) {| enumerable a |}
  (bij : erased (a =~ a))
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (x:a). p (bij.ff x)

ghost
fn forevery_permute_back
  (#a:Type0) {| enumerable a |}
  (bij : erased (a =~ a))
  (p : a -> slprop)
  requires
    forall+ (x:a). p (bij.ff x)
  ensures
    forall+ (x:a). p x

ghost
fn forevery_tostar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    bigstar 0 (cardinal a #_) (fun i -> p (of_nat i))

ghost
fn forevery_fromstar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    bigstar 0 (cardinal a #_) (fun i -> p (of_nat i))
  ensures
    forall+ (x:a). p x

ghost
fn forevery_fromnat
  (n : nat)
  (p : natlt n -> slprop)
  requires
    bigstar 0 n (fun i -> p i)
  ensures
    forall+ (x : natlt n). p x

ghost
fn forevery_tonat
  (n : nat)
  (p : natlt n -> slprop)
  requires
    forall+ (x : natlt n). p x
  ensures
    bigstar 0 n (fun i -> p i)

ghost
fn forevery_emp_intro
  (a : Type0) {| enumerable a |}
  requires
    emp
  ensures
    forall+ (_ : a). emp

ghost
fn forevery_emp_elim
  (a : Type0) {| enumerable a |}
  requires
    forall+ (_ : a). emp
  ensures
    emp

ghost
fn forevery_unit_intro
  (p : slprop)
  requires
    p
  ensures
    forall+ (_:unit). p

ghost
fn forevery_unit_elim
  (p : slprop)
  requires
    forall+ (_:unit). p
  ensures
    p

ghost
fn forevery_singleton_intro
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop { cardinal a #_ == 1 })
  requires
    p (of_nat 0)
  ensures
    forall+ (x:a). p x

ghost
fn forevery_singleton_elim
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop { cardinal a #_ == 1 })
  requires
    forall+ (x:a). p x
  ensures
    p (of_nat 0)

(* SHOULD NOT BE NEEDED!
   1) We should mark the p argument of forevery as extensional,
      and have the checker do the work for us.
   2) Using forall+, everything should be uniformly eta-expanded.
 *)
ghost
fn forevery_eta
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a p
  ensures
    forevery a (fun x -> p x)

ghost
fn forevery_uneta
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a (fun x -> p x)
  ensures
    forevery a p

ghost
fn forevery_rw_type
  (a:Type0) {| d : enumerable a |}
  (b:Type{a == b})
  (f : a -> slprop)
  requires
    forall+ (x:a). f x
  ensures
    forevery b #d (fun (x:b) -> f x)

ghost
fn forevery_rw_size
  (n1 : nat)
  (n2 : nat{n1 == n2})
  (#p : natlt n1 -> slprop)
  requires
    forall+ (i : natlt n1). p i
  ensures
    forall+ (i : natlt n2). p i

ghost
fn forevery_rw_size2
  (n1 : nat)
  (n2 : nat{n1 == n2})
  (n3 : nat)
  (n4 : nat{n3 == n4})
  (#p : natlt n1 -> natlt n3 -> slprop)
  requires
    forall+ (i : natlt n1) (j : natlt n3). p i j
  ensures
    forall+ (i : natlt n2) (j : natlt n4). p i j

ghost
fn forevery_factor
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt n -> slprop)
  requires
    forall+ (i:natlt n). p i
  ensures
    forall+ (i1:natlt d1) (i2:natlt d2). p (i1 * d2 + i2)

ghost
fn forevery_unfactor
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt n -> slprop)
  requires
    forall+ (i1:natlt d1) (i2:natlt d2). p (i1 * d2 + i2)
  ensures
    forall+ (i:natlt n). p i

ghost
fn forevery_unfactor'
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt d1 -> natlt d2 -> slprop)
  requires
    forall+ (i1:natlt d1) (i2:natlt d2). p i1 i2
  ensures
    forall+ (i:natlt n). p (i/d2) (i%d2)

ghost
fn forevery_zip
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  requires
    (forall+ (x:a). p1 x) **
    (forall+ (x:a). p2 x)
  ensures
    forall+ (x:a). p1 x ** p2 x

ghost
fn forevery_unzip
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  requires
    forall+ (x:a). p1 x ** p2 x
  ensures
    (forall+ (x:a). p1 x) **
    (forall+ (x:a). p2 x)

ghost
fn forevery_map
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  (f : (x:a -> stt_ghost unit emp_inames (p1 x) (fun _ -> p2 x)))
  requires
    forall+ (x:a). p1 x
  ensures
    forall+ (x:a). p2 x

ghost
fn forevery_map_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (p1 p2 : a -> b -> slprop)
  (f : (x:a -> y:b -> stt_ghost unit emp_inames (p1 x y) (fun _ -> p2 x y)))
  requires
    forall+ (x:a) (y:b). p1 x y
  ensures
    forall+ (x:a) (y:b). p2 x y

unfold
let pad_f (#n1 : nat) (n2 : nat{n1 <= n2})
  (f : natlt n1 -> slprop)
  : natlt n2 -> slprop =
  fun i ->
    if i < n1 then f i else emp

ghost
fn forevery_pad
  (n1 : nat)
  (n2 : nat{n1 <= n2})
  (p : natlt n1 -> slprop)
  requires
    forall+ (i : natlt n1). p i
  ensures
    forall+ (i : natlt n2). pad_f n2 p i


ghost
fn forevery_unpad
  (n1 : nat)
  (n2 : nat{n1 <= n2})
  (p : natlt n1 -> slprop)
  requires
    forall+ (i : natlt n2). pad_f n2 p i
  ensures
    forall+ (i : natlt n1). p i

ghost
fn forevery_extract
  (#a:Type0) {| enumerable a |}
  (z : a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z ** (p z @==> forall+ (x:a). p x)

ghost
fn forevery_extract_if
  (#a:Type0) {| enumerable a |}
  (z : a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z **
    (forall+ (x:a).
      if Enumerable.to_nat x = Enumerable.to_nat z then emp else p x)

ghost
fn forevery_intro_if
  (#a:Type0) {| enumerable a |}
  (z : a)
  (p : a -> slprop)
  requires
    p z
  ensures
    (forall+ (x:a).
      if Enumerable.to_nat x = Enumerable.to_nat z then p x else emp)

ghost
fn forevery_split_either
  (#a #b : Type0) {| enumerable a, enumerable b |}
  (p : either a b -> slprop)
  requires
    forall+ (x:either a b). p x
  ensures
    (forall+ (x:a). p (Inl x)) **
    (forall+ (x:b). p (Inr x))

ghost
fn forevery_join_either
  (#a #b : Type0) {| enumerable a, enumerable b |}
  (p : either a b -> slprop)
  requires
    (forall+ (x:a). p (Inl x)) **
    (forall+ (x:b). p (Inr x))
  ensures
    forall+ (x:either a b). p x
