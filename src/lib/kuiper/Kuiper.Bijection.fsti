module Kuiper.Bijection

#lang-pulse
open Kuiper.Common
open Kuiper.SizeT
open FStar.Ghost { erased }
module SZ = FStar.SizeT
open FStar.SizeT { div as (/^), (%^), (+^), (-^), ( *^ )  }

(* A theory of bijections, used to shift views
over ownership and data layouts. There is some delicate
need to mark some of these 'unfold'. Probably due
to a limitation of the pulse checker. *)

noeq
inline_for_extraction noextract (* IMPORTANT! *)
type bijection (a b : Type) = {
  ff : a -> b;
  gg : b -> a;

  ff_gg : x:_ -> squash (ff (gg x) == x);
  gg_ff : x:_ -> squash (gg (ff x) == x);
}
let ( =~ ) a b = bijection a b

let mk_bijection
  (#a #b : _)
  (ff : a -> b)
  (gg : b -> a)
  (ff_gg : (x:b -> squash (ff (gg x) == x)))
  (gg_ff : (x:a -> squash (gg (ff x) == x)))
  : (a =~ b) =
  Mkbijection ff gg ff_gg gg_ff

let bij_unit_natlt1 : bijection unit (natlt 1) = {
  ff = (fun _ -> 0 <: natlt 1);
  gg = (fun _ -> ());
  ff_gg = ez;
  gg_ff = ez;
}

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

inline_for_extraction noextract
let bij_self (a:Type) : (a =~ a) =
{
  ff = id;
  gg = id;
  ff_gg = ez;
  gg_ff = ez;
}

unfold
inline_for_extraction noextract
let bij_sym (#a #b : Type) (d : a =~ b) : (b =~ a) =
{
  ff = d.gg;
  gg = d.ff;
  ff_gg = d.gg_ff;
  gg_ff = d.ff_gg;
}

inline_for_extraction noextract
let bij_comp (#a #b #c : Type) (ab : a =~ b) (bc : b =~ c) : (a =~ c) =
{
  ff = bc.ff `o` ab.ff;
  gg = ab.gg `o` bc.gg;
  ff_gg = (fun x -> ab.ff_gg (bc.gg x); bc.ff_gg x);
  gg_ff = (fun x -> bc.gg_ff (ab.ff x); ab.gg_ff x);
}

inline_for_extraction noextract
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

inline_for_extraction noextract
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
  ff_gg = ez;
  gg_ff = ez;
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
inline_for_extraction noextract
let fin_size_t_bij (n:nat{SZ.fits n}) : (natlt n =~ szlt n) =
  {
    gg = (fun (m:szlt n) -> SZ.v m);
    ff = (fun (i:natlt n) -> SZ.uint_to_t i <: szlt n);
    ff_gg = ez;
    gg_ff = ez;
  }

(* weird typing errors without hoisting. *)
unfold
inline_for_extraction noextract
let sz_prod_ff (n1:SZ.t) (n2:SZ.t{SZ.fits (SZ.v n1 * SZ.v n2)})
  : szlt (SZ.v n1) & szlt (SZ.v n2) -> szlt (SZ.v n1 * SZ.v n2)
  = fun xy -> (xy._1 *^ n2 +^ xy._2)

unfold
inline_for_extraction noextract
let sz_prod_gg (n1:SZ.t) (n2:SZ.t{SZ.fits (SZ.v n1 * SZ.v n2)})
  : szlt (SZ.v n1 * SZ.v n2) -> szlt (SZ.v n1) & szlt (SZ.v n2)
  = fun i -> (i /^ n2, i %^ n2)

unfold
inline_for_extraction noextract
let bij_sz_prod (n1:SZ.t) (n2:SZ.t{SZ.fits (SZ.v n1 * SZ.v n2)})
  : (szlt (SZ.v n1) & szlt (SZ.v n2) =~ szlt (SZ.v n1 * SZ.v n2))
  = {
    ff = sz_prod_ff n1 n2;
    gg = sz_prod_gg n1 n2;
    ff_gg = ez;
    gg_ff = ez;
  }

inline_for_extraction noextract
let bij_either (#a #b #c #d : Type)
  (ab : a =~ b) (cd : c =~ d) : (either a c =~ either b d) =
{
  ff = (fun x -> match x with
    | Inl x -> Inl (ab.ff x)
    | Inr y -> Inr (cd.ff y));
  gg = (fun x -> match x with
    | Inl x -> Inl (ab.gg x)
    | Inr y -> Inr (cd.gg y));
  ff_gg = ez;
  gg_ff = ez;
}

let bij_nat_sum (n1 n2 : nat)
  : (either (natlt n1) (natlt n2) =~ natlt (n1 + n2)) =
{
  ff = (fun (x : either (natlt n1) (natlt n2)) ->
    (match x with
     | Inl i -> i
     | Inr j -> n1 + j) <: natlt (n1 + n2));
  gg = (fun (i : natlt (n1 + n2)) ->
    if i < n1
    then Inl i
    else Inr (i - n1));
  ff_gg = ez;
  gg_ff = ez;
}

inline_for_extraction noextract
let bij_sz_sum_ff (n1 : sz) (n2 : sz{SZ.fits (SZ.v n1 + SZ.v n2)})
  : either (szlt n1) (szlt n2) -> szlt (n1 + n2)
  = fun x -> (match x with
    | Inl i -> i
    | Inr j -> n1 +^ j)

inline_for_extraction noextract
let bij_sz_sum_gg (n1 : sz) (n2 : sz{SZ.fits (SZ.v n1 + SZ.v n2)})
  : szlt (n1 + n2) -> either (szlt n1) (szlt n2)
  = fun i -> if i `SZ.lt` n1
    then Inl i
    else Inr (i -^ n1)

inline_for_extraction noextract
let bij_sz_sum (n1 : sz) (n2 : sz{SZ.fits (SZ.v n1 + SZ.v n2)})
  : (either (szlt n1) (szlt n2) =~ szlt (n1 + n2)) =
{
  ff = bij_sz_sum_ff n1 n2;
  gg = bij_sz_sum_gg n1 n2;
  ff_gg = ez;
  gg_ff = ez;
}

open Kuiper.Injection

let inj_bij (#a #b : Type) (bij : a =~ b) : (a @~> b) =
  {
    f = bij.ff;
    is_inj = ez;
  }

let inj_bij' (#a #b : Type) (bij : a =~ b) : (b @~> a) =
  {
    f = bij.gg;
    is_inj = ez;
  }

let bij_erase (#a #b : Type) (bij : a =~ b) : (erased a =~ erased b) =
{
  ff = (fun (x : erased a) -> bij.ff x <: erased b);
  gg = (fun (x : erased b) -> bij.gg x <: erased a);
  ff_gg = ez;
  gg_ff = ez;
}

(* These are useful *)
let bij_is_surj (#a #b : Type) (bij : a =~ b) :
  Lemma (Functions.is_surj #a #b bij.ff)
        [SMTPat (Functions.is_surj #a #b bij.ff)]
=
  assert (forall x. bij.ff (bij.gg x) == x);
   ()

let bij_is_surj' (#a #b : Type) (bij : a =~ b) :
  Lemma (Functions.is_surj #b #a bij.gg)
        [SMTPat (Functions.is_surj #b #a bij.gg)]
=
  assert (forall x. bij.gg (bij.ff x) == x);
   ()


(* A type of "natural" bijections that we solve via
typeclass resolution. *)
class natural_bijection (a b : Type) = {
  bij : (a =~ b);
}

instance nb_self (a:Type) : natural_bijection a a = {
  bij = bij_self _;
}

instance nb_nat_sz (n:nat{SZ.fits n}) : natural_bijection (natlt n) (szlt n) = {
  bij = fin_size_t_bij _;
}

let natural (#a #b : Type) {| d : natural_bijection a b |} : bijection a b = d.bij