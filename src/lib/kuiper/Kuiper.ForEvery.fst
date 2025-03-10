module Kuiper.ForEvery

#lang-pulse
open Kuiper.Common
open Kuiper.Bijection
open Kuiper.Enumerable
open Pulse.Lib.BigStar

let forevery a p =
  bigstar 0 (cardinal a) (fun i -> p (of_nat i))

let forevery_ext_lem
  (#a:Type0) {| enumerable a |}
  (f : a -> slprop)
  (g : a -> slprop { forall x. f x == g x })
  : Lemma (ensures (forall+ (x:a). f x) == (forall+ (x:a). g x))
  = ()

ghost
fn forevery_ext
  (#a:Type0) {| enumerable a |}
  (f : a -> slprop)
  (g : a -> slprop { forall x. f x == g x })
  requires
    forall+ (x:a). f x
  ensures
    forall+ (x:a). g x
{
  ();
}

ghost
fn forevery_ext_2
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  (g : a -> b -> slprop { forall x y. f x y == g x y})
  requires
    forall+ (x:a) (y:b). f x y
  ensures
    forall+ (x:a) (y:b). g x y
{
  ();
}

ghost
fn forevery_flatten
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  requires
    forevery a (fun x ->
      forevery b (fun y -> f x y))
  ensures
    forevery (a & b) (fun (x, y) -> f x y)
{
  unfold forevery a (fun x -> forevery b (fun y -> f x y));
  ghost
  fn aux1 (i:natlt (cardinal a))
    requires forevery b (fun y -> f (of_nat i) y)
    ensures  bigstar 0 (cardinal b) (fun j -> f (of_nat i) (of_nat j))
  {
    unfold forevery b (fun y -> f (of_nat i) y);
  };
  bigstar_map #_ #_ #0 #(cardinal a) aux1; // optional :-)
  bigstar_flatten #_ #_ #(cardinal a) #(cardinal b);
  fold forevery (a & b) (fun (x, y) -> f x y);
}

ghost
fn forevery_flatten'
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a & b -> slprop)
  requires
    forall+ (x:a) (y:b). f (x, y)
  ensures
    forall+ (xy : a & b). f xy
{
  forevery_flatten (fun x y -> f (x, y));
}

ghost
fn forevery_unflatten
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a -> b -> slprop)
  requires
    forevery (a & b) (fun (x, y) -> f x y)
  ensures
    forevery a (fun x ->
      forevery b (fun y -> f x y))
{
  unfold forevery (a & b) (fun (x, y) -> f x y);
  assert bigstar 0 (cardinal (a & b)) (fun i -> let x, y = of_nat i in f x y);
  rewrite
    bigstar 0 (cardinal (a & b)) (fun i -> let x, y = of_nat i in f x y)
  as
    bigstar 0 (cardinal a * cardinal b) (fun i -> f (of_nat (i / cardinal b)) (of_nat (i % cardinal b)));
  bigstar_unflatten #0 #0 #(cardinal a) #(cardinal b) #(fun x y -> f (of_nat x) (of_nat y));
  fold forevery a (fun x ->
    forevery b (fun y -> f x y));
}

ghost
fn forevery_unflatten'
  (#a:Type0) {| enumerable a |}
  (#b:Type0) {| enumerable b |}
  (f : a & b -> slprop)
  requires
    forall+ (xy : a & b). f xy
  ensures
    forall+ (x:a) (y:b). f (x, y)
{
  forevery_unflatten (fun x y -> f (x, y));
}

let bij2perm (n:nat) (d : natlt n =~ natlt n)
  : permutation (i:nat {0 <= i /\ i < n}) = {
  f = d.ff;
  g = d.gg;
  proof = ez; (* patterns! *)
}

ghost
fn bigstar_permute'
  (#u1 : int)
  (#n : nat)
  (f : natlt n -> slprop)
  (d : bijection (natlt n) (natlt n))
  requires bigstar #u1 0 n f
  ensures  bigstar #u1 0 n (fun i -> f (d.ff i))
{
  let pp = bij2perm n d;
  bigstar_permute #u1 #0 #n #f pp;
}

ghost
fn bigstar_permute''
  (#u1 : int)
  (#n : nat)
  (f : natlt n -> slprop)
  (d : bijection (natlt n) (natlt n))
  requires bigstar #u1 0 n f
  ensures  bigstar #u1 0 n (fun i -> f (d.gg i))
{
  bigstar_permute' #u1 #n f (bij_sym d);
}

ghost
fn forevery_iso
  (#a:Type0) {| ea : enumerable a |}
  (#b:Type0) {| eb : enumerable b |}
  (bij : (a =~ b))
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    forall+ (y:b). p (bij.gg y)
{
  bijection_implies_equal_cardinal a b bij;
  assert (pure (cardinal a == cardinal b));

  unfold forevery a (fun x -> p x);
  assert bigstar 0 (cardinal a) (fun i -> p (of_nat i));

  let bij_n : (natlt (cardinal a) =~ natlt (cardinal a)) =
    bij_sym ea.bij `bij_comp` bij `bij_comp` eb.bij;

  assert bigstar 0 (cardinal a) (fun i -> p (of_nat #a i));
  bigstar_permute'' (fun i -> p (of_nat i)) bij_n;
  assert bigstar 0 (cardinal a) (fun i -> p (of_nat #a (bij_n.gg i)));
  assert bigstar 0 (cardinal a) (fun i -> p (of_nat #a (to_nat #a (bij.gg (eb.bij.gg i)))));
  assert bigstar 0 (cardinal a) (fun i -> p (bij.gg (eb.bij.gg i)));
  rewrite (* rewrite each cardinal a as cardinal b fails *)
    bigstar 0 (cardinal a) (fun i -> p (bij.gg (eb.bij.gg i)))
  as
    bigstar 0 (cardinal b) (fun i -> p (bij.gg (eb.bij.gg i)));

  assert bigstar 0 (cardinal b) (fun i -> p (bij.gg (eb.bij.gg i)));
  fold forevery b (fun y -> p (bij.gg y));
}

ghost
fn forevery_tostar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forall+ (x:a). p x
  ensures
    bigstar 0 (cardinal a) (fun i -> p (of_nat i))
{
  unfold forevery a (fun x -> p x);
}

ghost
fn forevery_fromstar
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    bigstar 0 (cardinal a) (fun i -> p (of_nat i))
  ensures
    forall+ (x:a). p x
{
  fold forevery a (fun x -> p x);
}

ghost
fn forevery_singleton_intro
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop { cardinal a == 1 })
  requires
    p (of_nat 0)
  ensures
    forall+ (x:a). p x
{
  bigstar_single_intro #0 0 (fun x -> p (of_nat x));
  rewrite
    bigstar 0 1 (fun x -> p (of_nat x))
  as
    bigstar 0 (cardinal a) (fun x -> p (of_nat x));
  fold forevery a (fun x -> p x);
}

ghost
fn forevery_singleton_elim
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop { cardinal a == 1 })
  requires
    forall+ (x:a). p x
  ensures
    p (of_nat 0)
{
  unfold forevery a (fun x -> p x);
  rewrite each cardinal a #_ as (0 + 1);
  bigstar_single_elim #0 #0 #(fun x -> p (of_nat #a x));
}

ghost
fn forevery_unit_intro
  (p : slprop)
  requires
    p
  ensures
    forevery unit (fun _ -> p)
{
  forevery_singleton_intro #unit (fun _ -> p);
}

ghost
fn forevery_unit_elim
  (p : slprop)
  requires
    forevery unit (fun _ -> p)
  ensures
    p
{
  forevery_singleton_elim #unit (fun _ -> p);
}

ghost
fn forevery_eta
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a p
  ensures
    forevery a (fun x -> p x)
{
  unfold forevery a p;
  bigstar_eta ();
  fold forevery a (fun x -> p x);
  ();
}

ghost
fn forevery_uneta
  (#a:Type0) {| enumerable a |}
  (p : a -> slprop)
  requires
    forevery a (fun x -> p x)
  ensures
    forevery a p
{
  unfold forevery a (fun x -> p x);
  bigstar_uneta ();
  fold forevery a p;
  ();
}

ghost
fn forevery_rw_size
  (n1 : nat)
  (n2 : nat{n1 == n2})
  (#p : natlt n1 -> slprop)
  requires
    forall+ (i : natlt n1). p i
  ensures
    forall+ (i : natlt n2). p i
{
  ()
}

ghost
fn forevery_factor
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt n -> slprop)
  requires
    forall+ (i:natlt n). p i
  ensures
    forall+ (i1:natlt d1) (i2:natlt d2). p (i1 * d2 + i2)
{
  open Kuiper.Bijection;
  forevery_rw_size n (d1 * d2);
  forevery_iso (bij_sym <| bij_nat_prod #d1 #d2) _;
  forevery_unflatten #(natlt d1) #_ #(natlt d2) (fun i1 i2 -> p (i1 * d2 + i2));
}

#push-options "--z3rlimit 20"
ghost
fn forevery_unfactor
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt n -> slprop)
  requires
    forall+ (i1:natlt d1) (i2:natlt d2). p (i1 * d2 + i2)
  ensures
    forall+ (i:natlt n). p i
{
  open Kuiper.Bijection;
  forevery_flatten #(natlt d1) #_ #(natlt d2) (fun i1 i2 -> p (i1 * d2 + i2));
  forevery_iso (bij_nat_prod #d1 #d2) _;
  forevery_rw_size (d1 * d2) n;
  ()
}
#pop-options

ghost
fn forevery_unfactor'
  (n : nat)
  (d1 : nat) (d2 : nat { n == d1 * d2 })
  (p : natlt d1 -> natlt d2 -> slprop)
  requires
    forall+ (i1:natlt d1) (i2:natlt d2). p i1 i2
  ensures
    forall+ (i:natlt n). p (i/d2) (i%d2)
{
  forevery_unfactor n d1 d2 (fun i -> p (i/d2) (i%d2));
}

ghost
fn forevery_zip
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  requires
    (forall+ (x:a). p1 x) **
    (forall+ (x:a). p2 x)
  ensures
    forall+ (x:a). p1 x ** p2 x
{
  unfold forevery a (fun x -> p1 x);
  unfold forevery a (fun x -> p2 x);
  bigstar_zip 0 (cardinal a) _ _;
  fold forevery a (fun x -> p1 x ** p2 x);
}

ghost
fn forevery_unzip
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  requires
    forall+ (x:a). p1 x ** p2 x
  ensures
    (forall+ (x:a). p1 x) **
    (forall+ (x:a). p2 x)
{
  unfold forevery a (fun x -> p1 x ** p2 x);
  bigstar_unzip 0 (cardinal a) _ _;
  fold forevery a (fun x -> p1 x);
  fold forevery a (fun x -> p2 x);
}

ghost
fn forevery_map
  (#a:Type0) {| enumerable a |}
  (p1 p2 : a -> slprop)
  (f : (x:a -> stt_ghost unit emp_inames (p1 x) (fun _ -> p2 x)))
  requires
    forall+ (x:a). p1 x
  ensures
    forall+ (x:a). p2 x
{
  unfold forevery a (fun x -> p1 x);
  bigstar_map #_ #_ #0 #(cardinal a) (fun x -> f (of_nat x));
  fold forevery a (fun x -> p2 x);
}

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
{
  forevery_map #a
    (fun x -> forevery b (fun y -> p1 x y))
    (fun x -> forevery b (fun y -> p2 x y))
    (fun x -> forevery_map (fun y -> p1 x y)
                          (fun y -> p2 x y)
                          (f x));
}
