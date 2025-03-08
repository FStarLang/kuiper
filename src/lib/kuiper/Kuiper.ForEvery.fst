module Kuiper.ForEvery

#lang-pulse
open Kuiper.Common
open Kuiper.Bijection
open Kuiper.Enumerable
open Pulse.Lib.BigStar

let forevery a p =
  bigstar 0 (cardinal a) (fun i -> p (of_nat i))

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

let bij2perm (n:nat) (d : natlt n =~ natlt n)
  : Pulse.Lib.BigStar.permutation (i:nat {0 <= i /\ i < n}) = {
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
    forevery a p
  ensures
    forevery b (fun y -> p (bij.gg y))
{
  bijection_implies_equal_cardinal a b bij;
  assert (pure (cardinal a == cardinal b));

  unfold forevery a p;
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
