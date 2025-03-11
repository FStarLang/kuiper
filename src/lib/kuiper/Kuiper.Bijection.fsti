module Kuiper.Bijection

#lang-pulse
open Kuiper.Common
open FStar.Tactics.V2
module SZ = FStar.SizeT
open FStar.SizeT { div as (/^), (%^), (+^), (-^), ( *^ )  }

(* A theory of bijections, used to shift views
over ownership and data layouts. There is some delicate
need to mark some of these 'unfold'. Probably due
to a limitation of the pulse checker. *)

noeq
type bijection (a b : Type) = {
  ff : a -> b;
  gg : b -> a;

  ff_gg : x:_ -> squash (ff (gg x) == x);
  gg_ff : x:_ -> squash (gg (ff x) == x);
}

let ( =~ ) a b = bijection a b

(* Move values across bijections. *)
let ( |~> ) (#a #b : Type) (x : a) (bij : a =~ b) : b = bij.ff x
let ( <~| ) (#a #b : Type) (x : b) (bij : a =~ b) : a = bij.gg x

val galois (#a #b : _) (d : a =~ b) (x:a) (y:b)
  : Lemma (d.ff x == y <==> x == d.gg y)
          [SMTPat (d.ff x); SMTPat (d.gg y)]

#push-options "--warn_error -288"
val galois_forall (#a #b : _) (d : a =~ b)
  : Lemma (forall (x:a) (y:b). d.ff x == y <==> x == d.gg y)
          [SMTPat (has_type d (a =~ b))] // OK? Useful?
#pop-options

let bij_self (a:Type) : (a =~ a) =
{
  ff = id;
  gg = id;
  ff_gg = easy;
  gg_ff = easy;
}

unfold
let bij_sym (#a #b : Type) (d : a =~ b) : (b =~ a) =
{
  ff = d.gg;
  gg = d.ff;
  ff_gg = d.gg_ff;
  gg_ff = d.ff_gg;
}

let o f g =
  fun x -> f (g x)

let bij_comp (#a #b #c : Type) (ab : a =~ b) (bc : b =~ c) : (a =~ c) =
{
  ff = bc.ff `o` ab.ff;
  gg = ab.gg `o` bc.gg;
  ff_gg = (fun x -> ab.ff_gg (bc.gg x); bc.ff_gg x);
  gg_ff = (fun x -> bc.gg_ff (ab.ff x); ab.gg_ff x);
}

let bij_prod (#a #b #c #d : Type) (ab : a =~ b) (cd : c =~ d) : (a & c =~ b & d) =
{
  ff = (fun (x, y) -> (ab.ff x, cd.ff y));
  gg = (fun (x, y) -> (ab.gg x, cd.gg y));
  ff_gg = (fun x ->
    let (x1, x2) = x in
    ab.ff_gg x1; cd.ff_gg x2);
  gg_ff = (fun x ->
    let (x1, x2) = x in
    ab.gg_ff x1; cd.gg_ff x2);
}

let bij_flip (#a #b : Type) : (a & b =~ b & a) =
{
  ff = (fun (x, y) -> (y, x));
  gg = (fun (y, x) -> (x, y));
  ff_gg = ez;
  gg_ff = ez;
}

(* weird typing errors without hoisting. *)
unfold
let prod_ff (n1 n2 : nat) : natlt n1 & natlt n2 -> natlt (n1 * n2) =
  // fun (x, y) -> (x * n2 + y)
  fun xy -> (xy._1 * n2 + xy._2)

unfold
let prod_gg (n1 n2 : nat) : natlt (n1 * n2) -> natlt n1 & natlt n2 =
  fun i -> (i / n2, i % n2)

unfold
let bij_nat_prod (#n1 #n2 : nat) : (natlt n1 & natlt n2 =~ natlt (n1 * n2)) =
{
  ff = prod_ff n1 n2;
  gg = prod_gg n1 n2;
  ff_gg = easy;
  gg_ff = easy;
}

val __bij_cardinal (n1 n2 : nat) (bij : natlt n1 =~ natlt n2)
  : Lemma (n1 == n2)

val bij_cardinal (n1 n2 : nat)
  : Lemma (requires exists (b : natlt n1 =~ natlt n2). True)
          (ensures n1 == n2)


(* FIXME: terrible inference here if we remove the ascription from
the body of ff. It seems to try to try to define a bijection into SZ.t,
regardless of the annotation on the letbinding and the annotation on the
binder for m in gg. *)
let fin_size_t_bij (n:nat{SZ.fits n}) : (natlt n =~ szlt n) =
  {
    gg = (fun (m:szlt n) -> SZ.v m);
    ff = (fun (i:natlt n) -> SZ.uint_to_t i <: szlt n);
    ff_gg = ez;
    gg_ff = ez;
  }

(* weird typing errors without hoisting. *)
unfold
let sz_prod_ff (n1:SZ.t) (n2:SZ.t{SZ.fits (SZ.v n1 * SZ.v n2)})
  : szlt (SZ.v n1) & szlt (SZ.v n2) -> szlt (SZ.v n1 * SZ.v n2)
  = fun xy -> (xy._1 *^ n2 +^ xy._2)

unfold
let sz_prod_gg (n1:SZ.t) (n2:SZ.t{SZ.fits (SZ.v n1 * SZ.v n2)})
  : szlt (SZ.v n1 * SZ.v n2) -> szlt (SZ.v n1) & szlt (SZ.v n2)
  = fun i -> (i /^ n2, i %^ n2)

unfold
let bij_sz_prod (n1:SZ.t) (n2:SZ.t{SZ.fits (SZ.v n1 * SZ.v n2)})
  : (szlt (SZ.v n1) & szlt (SZ.v n2) =~ szlt (SZ.v n1 * SZ.v n2))
  = {
    ff = sz_prod_ff n1 n2;
    gg = sz_prod_gg n1 n2;
    ff_gg = easy;
    gg_ff = easy;
  }
