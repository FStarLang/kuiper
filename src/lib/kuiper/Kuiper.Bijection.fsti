module Kuiper.Bijection

#lang-pulse
open Kuiper.Common
open Kuiper.SizeT
open FStar.Ghost { erased }
module SZ = Kuiper.SizeT
open FStar.SizeT { (/^), (%^), (+^), (-^), ( *^ )  }

(* A theory of bijections, used to shift views
over ownership and data layouts. There is some delicate
need to mark some of these 'unfold'. Probably due
to a limitation of the pulse checker. *)

noeq
[@@erasable]
type bijection (a b : Type) = {
  ff : a -> GTot b;
  gg : b -> GTot a;

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
let ( |~> ) (#a #b : Type) (x : a) (bij : a =~ b) : GTot b = bij.ff x
let ( <~| ) (#a #b : Type) (x : b) (bij : a =~ b) : GTot a = bij.gg x

val bij_inv_fwd (#a #b : _) (d : a =~ b) (x:a)
  : Lemma (x == d.gg (d.ff x))
          [SMTPat (d.ff x)]


val bij_inv_bk (#a #b : _) (d : a =~ b) (y:b)
  : Lemma (y == d.ff (d.gg y))
          [SMTPat (d.gg y)]

let bij_self (a:Type) : (a =~ a) =
{
  ff = id;
  gg = id;
  ff_gg = ez;
  gg_ff = ez;
}

unfold
let bij_sym (#a #b : Type) (d : a =~ b) : (b =~ a) =
{
  ff = d.gg;
  gg = d.ff;
  ff_gg = d.gg_ff;
  gg_ff = d.ff_gg;
}

let bij_comp (#a #b #c : Type) (ab : a =~ b) (bc : b =~ c) : (a =~ c) =
{
  ff = bc.ff `oo` ab.ff;
  gg = ab.gg `oo` bc.gg;
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
inline_for_extraction noextract
let prod_ff (n1 n2 : nat) : natlt n1 & natlt n2 -> natlt (n1 * n2) =
  // fun (x, y) -> (x * n2 + y)
  fun xy -> (xy._1 * n2 + xy._2)

unfold
inline_for_extraction noextract
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
let fin_size_t_bij (n:nat{SZ.fits n}) : (natlt n =~ szlt n) =
  {
    ff = (fun (i : natlt n) -> SZ.uint_to_t i <: szlt n);
    gg = (fun (m : szlt n)  -> SZ.v m <: natlt n);
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

#restart-solver
unfold
let bij_sz_prod (n1:SZ.t) (n2:SZ.t{SZ.fits (SZ.v n1 * SZ.v n2)})
  : (szlt (SZ.v n1) & szlt (SZ.v n2) =~ szlt (SZ.v n1 * SZ.v n2))
  = {
    ff = sz_prod_ff n1 n2;
    gg = sz_prod_gg n1 n2;
    ff_gg = ez;
    gg_ff = ez;
  }

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

let bij_inj (#a #b : Type) (inj : a @~> b)
  : (a =~ image_of inj)
= let ff : a -> GTot (image_of inj) = (fun x -> inj.f x) in
  {
    ff = ff;
    gg = FStar.Functions.inverse_of_bij ff;
    ff_gg = ez;
    gg_ff = ez;
  }

let bij_inj' (#a #b : Type) (inj : a @~> b)
  : Ghost (a =~ b)
          (requires Functions.is_surj inj.f)
          (ensures fun _ -> True)
= {
  ff = inj.f;
  gg = FStar.Functions.inverse_of_bij inj.f;
  ff_gg = ez;
  gg_ff = ez;
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

(* Computationally relevant bijections *)
inline_for_extraction noextract
noeq type cbij (a b: Type) = {
  bij: (a =~ b);
  cff: cff: (a -> b) { forall x. cff x == bij.ff x };
  cgg: cgg: (b -> a) { forall x. cgg x == bij.gg x };
}

inline_for_extraction
let (==~) = cbij

inline_for_extraction noextract
let cbij_self (a:Type) : (a ==~ a) = {
  bij = bij_self _;
  cff = id;
  cgg = id;
}

inline_for_extraction noextract
let cbij_prod (#a #b #c #d : Type) (ab : a ==~ b) (cd : c ==~ d) : (a & c ==~ b & d) =
{
  bij = bij_prod ab.bij cd.bij;
  cff = (fun (x, y) -> (ab.cff x, cd.cff y));
  cgg = (fun (x, y) -> (ab.cgg x, cd.cgg y));
}

inline_for_extraction noextract
let cbij_comp (#a #b #c : Type) (ab : a ==~ b) (bc : b ==~ c) : (a ==~ c) =
{
  bij = bij_comp ab.bij bc.bij;
  cff = (fun x -> bc.cff (ab.cff x));
  cgg = (fun x -> ab.cgg (bc.cgg x));
}

inline_for_extraction noextract
let fin_size_t_cbij (n:nat{SZ.fits n}) : (natlt n ==~ szlt n) =
  {
    bij = fin_size_t_bij n;
    cff = (fun (i : natlt n) -> SZ.uint_to_t i <: szlt n);
    cgg = (fun (m : szlt n)  -> SZ.v m <: natlt n);
  }

(* A type of "natural" bijections that we solve via
typeclass resolution. *)
class natural_bijection (a b : Type) = {
  _bij : (a =~ b);
}

instance nb_self (a:Type) : natural_bijection a a = {
  _bij = bij_self _;
}

instance nb_nat_sz (n:nat{SZ.fits n}) : natural_bijection (natlt n) (szlt n) = {
  _bij = fin_size_t_bij _;
}

instance nb_either (a1 a2 b1 b2 : _) (nb1 : natural_bijection a1 b1)
  (nb2 : natural_bijection a2 b2)
  : natural_bijection (either a1 a2) (either b1 b2) = {
  _bij = bij_either nb1._bij nb2._bij;
}

instance nb_prod (a1 a2 b1 b2 : Type) (nb1 : natural_bijection a1 b1)
  (nb2 : natural_bijection a2 b2)
  : natural_bijection (a1 & a2) (b1 & b2) = {
  _bij = bij_prod nb1._bij nb2._bij;
}

let bij_prod3 (#a1 #a2 #a3 #b1 #b2 #b3 : Type)
  (ab1 : a1 =~ b1) (ab2 : a2 =~ b2) (ab3 : a3 =~ b3)
  : (a1 & a2 & a3 =~ b1 & b2 & b3) =
{
  ff = (fun (x1, x2, x3) -> (ab1.ff x1, ab2.ff x2, ab3.ff x3));
  gg = (fun (y1, y2, y3) -> (ab1.gg y1, ab2.gg y2, ab3.gg y3));
  ff_gg = (fun (y1, y2, y3) -> ab1.ff_gg y1; ab2.ff_gg y2; ab3.ff_gg y3);
  gg_ff = (fun (x1, x2, x3) -> ab1.gg_ff x1; ab2.gg_ff x2; ab3.gg_ff x3);
}

let bij_prod4 (#a1 #a2 #a3 #a4 #b1 #b2 #b3 #b4 : Type)
  (ab1 : a1 =~ b1) (ab2 : a2 =~ b2) (ab3 : a3 =~ b3) (ab4 : a4 =~ b4)
  : (a1 & a2 & a3 & a4 =~ b1 & b2 & b3 & b4) =
{
  ff = (fun (x1, x2, x3, x4) -> (ab1.ff x1, ab2.ff x2, ab3.ff x3, ab4.ff x4));
  gg = (fun (y1, y2, y3, y4) -> (ab1.gg y1, ab2.gg y2, ab3.gg y3, ab4.gg y4));
  ff_gg = (fun (y1, y2, y3, y4) -> ab1.ff_gg y1; ab2.ff_gg y2; ab3.ff_gg y3; ab4.ff_gg y4);
  gg_ff = (fun (x1, x2, x3, x4) -> ab1.gg_ff x1; ab2.gg_ff x2; ab3.gg_ff x3; ab4.gg_ff x4);
}

instance nb_prod3 (a1 a2 a3 b1 b2 b3 : Type)
  (nb1 : natural_bijection a1 b1)
  (nb2 : natural_bijection a2 b2)
  (nb3 : natural_bijection a3 b3)
  : natural_bijection (a1 & a2 & a3) (b1 & b2 & b3) = {
  _bij = bij_prod3 nb1._bij nb2._bij nb3._bij;
}

instance nb_prod4 (a1 a2 a3 a4 b1 b2 b3 b4 : Type)
  (nb1 : natural_bijection a1 b1)
  (nb2 : natural_bijection a2 b2)
  (nb3 : natural_bijection a3 b3)
  (nb4 : natural_bijection a4 b4)
  : natural_bijection (a1 & a2 & a3 & a4) (b1 & b2 & b3 & b4) = {
  _bij = bij_prod4 nb1._bij nb2._bij nb3._bij nb4._bij;
}

let natural (#a #b : Type) {| d : natural_bijection a b |} : bijection a b = d._bij

let bij_push_tuple3 (#a #b #c : _) :
  ((a & (b & c)) =~ (b & (a & c))) =
{
  ff = (fun (x, (y, z)) -> (y, (x, z)));
  gg = (fun (y, (x, z)) -> (x, (y, z)));
  ff_gg = ez;
  gg_ff = ez;
}

inline_for_extraction noextract
let cbij_push_tuple3 (#a #b #c : _) :
  ((a & (b & c)) ==~ (b & (a & c))) =
{
  bij = bij_push_tuple3;
  cff = (fun (x, (y, z)) -> (y, (x, z)));
  cgg = (fun (y, (x, z)) -> (x, (y, z)));
}

let bij_tuple3_nest (#a #b #c : _) :
  (a & b & c =~ (a & (b & c))) =
{
  ff = (fun (x, y, z) -> (x, (y, z)));
  gg = (fun (x, (y, z)) -> (x, y, z));
  ff_gg = ez;
  gg_ff = ez;
}

let bij_tuple4_nest (#a #b #c #d : _) :
  (a & b & c & d =~ (a & (b & c & d))) =
{
  ff = (fun (x, y, z, w) -> (x, (y, z, w)));
  gg = (fun (x, (y, z, w)) -> (x, y, z, w));
  ff_gg = ez;
  gg_ff = ez;
}
