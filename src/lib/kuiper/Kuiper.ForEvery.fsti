module Kuiper.ForEvery

#lang-pulse
open Pulse
open Kuiper.Common
open Kuiper.Bijection
open Kuiper.Enumerable
open Pulse.Lib.BigStar
open Pulse.Lib.Trade

let t2b =
  FStar.IndefiniteDescription.strong_excluded_middle

val ( forall+ )
  (#a:Type)
  (f : a -> slprop)
  : slprop

val timeless_forevery #a (p: a -> slprop) :
  Lemma (requires forall x. timeless (p x))
    (ensures timeless (op_forall_Plus p))
    [SMTPat (timeless (op_forall_Plus p))]

unfold
let forevery
  (a:Type)
  (f : a -> slprop)
  : slprop
  = op_forall_Plus #a f

ghost
fn forevery_ext
  (#a:Type0)
  (f : a -> slprop)
  (g : a -> slprop { forall x. f x == g x })
  requires
    forall+ (x:a). f x
  ensures
    forall+ (x:a). g x

ghost
fn forevery_intro_empty (#a:Type0) (p: a -> slprop)
  requires
    pure (forall (x:a). False)
  ensures
    forall+ (x:a). p x

ghost
fn forevery_elim_empty (#a:Type0) (p: a -> slprop)
  requires
    pure (forall (x:a). False)
  requires
    forall+ (x:a). p x

ghost
fn forevery_intro_false (#a:Type0) (p: a -> slprop)
  ensures
    forall+ (x:a {False}). p x

ghost
fn forevery_intro_fill (#a: Type0) (p: a -> slprop)
  (f: (x:a -> stt_ghost unit emp_inames emp (fun _ -> p x)))
  ensures
    forall+ x. p x

ghost
fn forevery_insert
  (#a: Type0)
  (#f: a->prop)
  (p: a -> slprop)
  (y: a)
  requires
    forall+ (x:a {f x}). p x
  requires
    p y
  requires
    pure (~(f y))
  ensures
    forall+ (x:a {f x \/ y == x}). p x

ghost
fn forevery_remove'
  (#a: Type0)
  (f: a->prop)
  (p: a -> slprop)
  (y: a { f y })
  requires
    forall+ (x:a {f x}). p x
  ensures
    forall+ (x:a {f x /\ x =!= y}). p x
  ensures
    p y

ghost
fn forevery_remove
  (#a: Type0)
  (p: a -> slprop)
  (y: a)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (x:a {x =!= y}). p x
  ensures
    p y

ghost
fn forevery_fill
  (#a: Type0)
  (#f: a->prop)
  (p: a -> slprop)
  (pred: a -> prop)
  (g: (x:a{pred x} -> stt_ghost unit emp_inames emp (fun _ -> p x)))
  requires
    forall+ (x:a {f x}). p x
  requires
    pure (forall x. pred x ==> ~(f x))
  ensures
    forall+ (x:a {f x \/ pred x}). p x

ghost
fn forevery_refine_ext'
  (#a: Type0)
  (#f: a->prop)
  (g: a->prop { forall x. f x <==> g x })
  (p: (x:a{f x} -> slprop))
  requires
    forall+ (x:a {f x}). p x
  ensures
    forall+ (w:a {g w}). p w

ghost
fn forevery_refine_ext
  (#a: Type0)
  (#f g: a->prop)
  (p: a -> slprop)
  requires
    forall+ (x:a {f x}). p x
  requires
    pure (forall x. f x <==> g x)
  ensures
    forall+ (w:a {g w}). p w

ghost
fn forevery_unrefine
  (#a: Type0)
  (#f: a->prop)
  (p: a -> slprop)
  requires
    forall+ (x:a {f x}). p x
  requires
    pure (forall x. f x)
  ensures
    forall+ x. p x

ghost
fn forevery_refine_split
  (#a:Type0)
  (p: a -> slprop)
  (f: a -> prop)
  requires
    forall+ x. p x
  ensures
    forall+ (x:a{f x}). p x
  ensures
    forall+ (x:a{~(f x)}). p x

ghost
fn forevery_refine_join
  (#a:Type0)
  (p: a -> slprop)
  (f g: a -> prop)
  requires
    forall+ (x:a{f x}). p x
  requires
    forall+ (x:a{g x}). p x
  requires
    pure (forall x. ~(f x /\ g x))
  ensures
    forall+ (x:a{f x \/ g x}). p x

let unless (p: prop) (q: slprop) : slprop =
  if t2b p then emp else q

let when_ (p: prop) (q: slprop) : slprop =
  if t2b p then q else emp

(* Needed for when the rhs is partially defined *)
let when__ (p: prop) (q: squash p -> slprop) : slprop =
  if t2b p then q () else emp

ghost
fn forevery_unrefine_pred
  (#a:Type0)
  (p: a -> slprop)
  (f: a -> prop)
  requires
    forall+ (x:a { f x }). p x
  ensures
    forall+ (x:a). when_ (f x) (p x)

ghost
fn forevery_unrefine_pred'
  (#a:Type0)
  (f: a -> prop)
  (p: (x:a -> squash (f x) -> slprop))
  requires
    forall+ (x:a { f x }). p x ()
  ensures
    forall+ (x:a). when__ (f x) (p x)

ghost
fn forevery_refine_pred
  (#a:Type0)
  (p: a -> slprop)
  (f: a -> prop)
  requires
    forall+ (x:a). when_ (f x) (p x)
  ensures
    forall+ (x:a { f x }). p x

ghost
fn forevery_ext_2
  (#a:Type0)
  (#b:Type0)
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
  (#a:Type0)
  (#b:Type0)
  (f : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). f x y
  ensures
    forall+ (xy : a & b). f xy._1 xy._2

ghost
fn forevery_flatten_dep
  (#a : Type0)
  (#b : a -> Type0)
  (f : (x:a -> b x -> slprop))
  requires
    forall+ (x:a) (y:b x). f x y
  ensures
    forall+ (xy : (x:a & b x)). f xy._1 xy._2

ghost
fn forevery_flatten'
  (#a:Type0)
  (#b:Type0)
  (f : a & b -> slprop)
  requires
    forall+ (x:a) (y:b). f (x, y)
  ensures
    forall+ (xy : a & b). f xy

ghost
fn forevery_unflatten
  (#a:Type0)
  (#b:Type0)
  (f : a -> b -> slprop)
  requires
    forall+ (xy : a & b). f xy._1 xy._2
  ensures
    forall+ (x:a) (y:b). f x y

ghost
fn forevery_unflatten'
  (#a:Type0)
  (#b:Type0)
  (f : a & b -> slprop)
  requires
    forall+ (xy : a & b). f xy
  ensures
    forall+ (x:a) (y:b). f (x, y)

ghost
fn forevery_unflatten_dep
  (#a : Type0)
  (#b : a -> Type0)
  (f : (x:a -> b x -> slprop))
  requires
    forall+ (xy : (x:a & b x)). f xy._1 xy._2
  ensures
    forall+ (x:a) (y:b x). f x y

ghost
fn forevery_unflatten_dep'
  (#a : Type0)
  (#b : a -> Type0)
  (f : (x:a & b x) -> slprop)
  requires
    forall+ (xy : (x:a & b x)). f xy
  ensures
    forall+ (x:a) (y:b x). f (|x, y|)

ghost
fn forevery_commute
  (#a:Type0)
  (#b:Type0)
  (f : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). f x y
  ensures
    forall+ (y:b) (x:a). f x y

ghost
fn forevery_iso
  (#a:Type0)
  (#b:Type0)
  (bij : a =~ b)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (y:b). p (bij.gg y)

ghost
fn forevery_iso_back
  (#a:Type0)
  (#b:Type0)
  (bij : a =~ b)
  (p : a -> slprop)
  requires
    forall+ (y:b). p (bij.gg y)
  ensures
    forall+ (x:a). p x

(* Normally not needed... *)
ghost
fn forevery_permute
  (#a:Type0)
  (bij : a =~ a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (x:a). p (bij.ff x)

ghost
fn forevery_permute_back
  (#a:Type0)
  (bij : a =~ a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p (bij.ff x)
  ensures
    forall+ (x:a). p x

// FIXME: without this, Pulse will not type-check calls to forevery_natlt_{extend,restrict}
let natlt_coerce #m #n (i: natlt n { i < m }) : natlt m = i

ghost
fn forevery_natlt_extend
  (#n: nat)
  (m: nat { m >= n })
  (p: natlt n -> slprop)
  requires
    forall+ (i: natlt n). p i
  ensures
    forall+ (i: natlt m { i < n }). p (natlt_coerce i)

ghost
fn forevery_natlt_restrict
  (#n: nat)
  (m: nat { m >= n })
  (p: natlt n -> slprop)
  requires
    forall+ (i: natlt m { i < n }). p (natlt_coerce i)
  ensures
    forall+ (i: natlt n). p i

ghost
fn forevery_natlt_pop
  (n: nat { n > 0 })
  (p: natlt n -> slprop)
  requires
    forall+ (i: natlt n). p i
  ensures
    forall+ (i: natlt (n-1)). p (natlt_coerce i)
  ensures
    p (n-1)

ghost
fn forevery_natlt_push
  (n: nat { n > 0 })
  (p: natlt n -> slprop)
  requires
    forall+ (i: natlt (n-1)). p (natlt_coerce i)
  requires
    p (n-1)
  ensures
    forall+ (i: natlt n). p i

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
fn forevery_exists
  (#a: Type0) {| enumerable a |}
  (#b: Type0)
  (p: a -> b -> slprop)
  requires
    forall+ (x:a). exists* (y:b). p x y
  returns
    y:(a->GTot b)
  ensures
    forall+ (x:a). p x (y x)

ghost
fn forevery_emp_intro
  (a : Type0)
  requires
    emp
  ensures
    forall+ (_ : a). emp

ghost
fn forevery_emp_elim
  (a : Type0)
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
fn forevery_bool_intro
  (p: bool -> slprop)
  requires
    p false
  requires
    p true
  ensures
    forall+ (x: bool). p x

ghost
fn forevery_bool_elim
  (p: bool -> slprop)
  requires
    forall+ (x: bool). p x
  ensures
    p false
  ensures
    p true

ghost
fn forevery_singleton_intro'
  (#a:Type0)
  (p : a -> slprop)
  (x: a)
  requires
    pure (forall (y: a). x == y)
  requires
    p x
  ensures
    forall+ (x:a). p x

ghost
fn forevery_singleton_elim'
  (#a:Type0)
  (p : a -> slprop)
  (x: a)
  requires
    pure (forall (y: a). x == y)
  requires
    forall+ (x:a). p x
  ensures
    p x

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
  (#a:Type0)
  (p : a -> slprop)
  requires
    forevery a p
  ensures
    forevery a (fun x -> p x)

ghost
fn forevery_uneta
  (#a:Type0)
  (p : a -> slprop)
  requires
    forevery a (fun x -> p x)
  ensures
    forevery a p

ghost
fn forevery_rw_type
  (a:Type0)
  (b:Type{a == b})
  (f : a -> slprop)
  requires
    forall+ (x:a). f x
  ensures
    forall+ (x:b). f x

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
  (#a:Type0)
  (p1 p2 : a -> slprop)
  requires
    (forall+ (x:a). p1 x) **
    (forall+ (x:a). p2 x)
  ensures
    forall+ (x:a). p1 x ** p2 x

ghost
fn forevery_unzip
  (#a:Type0)
  (p1 p2 : a -> slprop)
  requires
    forall+ (x:a). p1 x ** p2 x
  ensures
    (forall+ (x:a). p1 x) **
    (forall+ (x:a). p2 x)

ghost
fn forevery_zip3
  (#a:Type0)
  (p1 p2 p3 : a -> slprop)
  requires
    forall+ (x:a). p1 x
  requires
    forall+ (x:a). p2 x
  requires
    forall+ (x:a). p3 x
  ensures
    forall+ (x:a). p1 x ** p2 x ** p3 x

ghost
fn forevery_unzip3
  (#a:Type0)
  (p1 p2 p3 : a -> slprop)
  requires
    forall+ (x:a). p1 x ** p2 x ** p3 x
  ensures
    forall+ (x:a). p1 x
  ensures
    forall+ (x:a). p2 x
  ensures
    forall+ (x:a). p3 x

ghost
fn forevery_map
  (#a:Type0)
  (p1 p2 : a -> slprop)
  (f : (x:a -> stt_ghost unit emp_inames (p1 x) (fun _ -> p2 x)))
  requires
    forall+ (x:a). p1 x
  ensures
    forall+ (x:a). p2 x

ghost
fn forevery_map_2
  (#a:Type0)
  (#b:Type0)
  (p1 p2 : a -> b -> slprop)
  (f : (x:a -> y:b -> stt_ghost unit emp_inames (p1 x y) (fun _ -> p2 x y)))
  requires
    forall+ (x:a) (y:b). p1 x y
  ensures
    forall+ (x:a) (y:b). p2 x y

ghost
fn forevery_map'
  (#a:Type0)
  (#b:Type0 { a == b })
  (p1 : a -> slprop)
  (p2 : b -> slprop)
  (f : (x:a -> y:b { x === y } -> stt_ghost unit emp_inames (p1 x) (fun _ -> p2 y)))
  requires
    forall+ (x:a). p1 x
  ensures
    forall+ (x:b). p2 x

ghost
fn forevery_zip_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (p1 p2 : a -> b -> slprop)
  requires
    (forall+ (x:a) (y:b). p1 x y) **
    (forall+ (x:a) (y:b). p2 x y)
  ensures
    forall+ (x:a) (y:b). p1 x y ** p2 x y

ghost
fn forevery_unzip_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (p1 p2 : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). p1 x y ** p2 x y
  ensures
    (forall+ (x:a) (y:b). p1 x y) **
    (forall+ (x:a) (y:b). p2 x y)

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
  (#a:Type0)
  (z : a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z ** (p z @==> forall+ (x:a). p x)

ghost
fn forevery_extract'
  (#a:Type0)
  (z : a)
  (p  : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z **
      (forall* (p' : a -> slprop).
        p' z ** pure (forall (x:a{x =!= z}). p' x == p x)
          @==> (forall+ (x:a). p' x))

ghost
fn forevery_extract_2
  (#a:Type0)
  (#b:Type0)
  (z : a) (w : b)
  (p : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). p x y
  ensures
    p z w ** (p z w @==> forall+ (x:a) (y:b). p x y)

ghost
fn forevery_extract_if
  (#a:Type0)
  (z : a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z **
    (forall+ (x:a).
      if t2b (x == z) then emp else p x)

ghost
fn forevery_unextract_if
  (#a:Type0)
  (z : a)
  (p : a -> slprop)
  requires
    p z **
    (forall+ (x:a).
      if t2b (x == z) then emp else p x)
  ensures
    forall+ (x:a). p x

ghost
fn forevery_extract_if_eqtype
  (#a:eqtype)
  (z : a)
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    p z **
    (forall+ (x:a).
      if x = z then emp else p x)

ghost
fn forevery_unextract_if_eqtype
  (#a:eqtype)
  (z : a)
  (p : a -> slprop)
  requires
    p z **
    (forall+ (x:a).
      if x = z then emp else p x)
  ensures
    forall+ (x:a). p x

ghost
fn forevery_extract_if_2
  (#a:Type0)
  (#b:Type0)
  (z : a) (w : b)
  (p : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). p x y
  ensures
    p z w **
    (forall+ (x:a) (y:b).
      if t2b ((x,y) == (z,w)) then emp else p x y)


ghost
fn forevery_intro_if
  (#a:Type0)
  (z : a)
  (p : a -> slprop)
  requires
    p z
  ensures
    (forall+ (x:a).
      if t2b (x == z) then p x else emp)

ghost
fn forevery_split_either
  (#a #b : Type0)
  (p : either a b -> slprop)
  requires
    forall+ (x:either a b). p x
  ensures
    (forall+ (x:a). p (Inl x)) **
    (forall+ (x:b). p (Inr x))

ghost
fn forevery_join_either
  (#a #b : Type0)
  (p : either a b -> slprop)
  requires
    (forall+ (x:a). p (Inl x)) **
    (forall+ (x:b). p (Inr x))
  ensures
    forall+ (x:either a b). p x

ghost
fn forevery_map_extra
  (#a:Type0) {| enumerable a |}
  (k : slprop)
  (p1 p2 : a -> slprop)
  (f : (x:a -> stt_ghost unit emp_inames (k ** p1 x) (fun _ -> k ** p2 x)))
  preserves k
  requires
    forall+ (x:a). p1 x
  ensures
    forall+ (x:a). p2 x

ghost
fn forevery_flatten4'
  (#a #b #c #d : Type0)
  (f : a & b & c & d -> slprop)
  requires
    forall+ (x:a) (y:b) (z:c) (w:d). f (x, y, z, w)
  ensures
    forall+ (xyzw : a & b & c & d). f xyzw

ghost
fn forevery_unflatten4'
  (#a #b #c #d : Type0)
  (f : a & b & c & d -> slprop)
  requires
    forall+ (xyzw : a & b & c & d). f xyzw
  ensures
    forall+ (x:a) (y:b) (z:c) (w:d). f (x, y, z, w)



ghost
fn forevery_split_or_2
  (#a:Type0)
  (r s : a -> prop)
  (p : a -> slprop)
  requires
    pure (~ (exists (x:a). r x /\ s x))
  requires
    forall+ (x:a { r x \/ s x }). p x
  ensures
    (forall+ (x:a { r x }). p x) **
    (forall+ (x:a { s x }). p x)

ghost
fn forevery_split_or_n
  (#a #b:Type0)
  (r : b -> a -> prop)
  (p : a -> slprop)
  requires
    pure (forall (i1 i2 : b) x.
      r i1 x /\ r i2 x ==> i1 == i2)
  requires
    forall+ (x:a {exists i. r i x}). p x
  ensures
    forall+ (i : b).
      forall+ (x:a { r i x }).
        p x

ghost
fn forevery_join_or_n
  (#a #b:Type0)
  (r : b -> a -> prop)
  (p : a -> slprop)
  requires
    pure (forall (i1 i2 : b) x.
      r i1 x /\ r i2 x ==> i1 == i2)
  requires
    forall+ (i : b).
      forall+ (x:a { r i x }).
        p x
  ensures
    forall+ (x:a {exists i. r i x}). p x
