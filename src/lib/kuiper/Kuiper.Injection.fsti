module Kuiper.Injection

#lang-pulse
open Kuiper.Common
open Kuiper.GhostPull
open Kuiper.SizeT

open FStar.Functions
module SZ = Kuiper.SizeT
open FStar.SizeT { (/^), (%^), (+^), (-^), ( *^ )  }

(* A theory of injections. *)

noeq
inline_for_extraction noextract (* IMPORTANT! *)
type injection (a b : Type) = {
  f : a -> b;

  is_inj : x:_ -> y:_ -> squash (f x == f y ==> x == y);
}

// Terrible symbol, but F* is limited in operator support.
inline_for_extraction
let ( @~> ) a b = injection a b

let mk_injection
  (#a #b : _)
  (f : a -> b)
  (is_inj : (x:_ -> y:_ -> squash (f x == f y ==> x == y)))
  : (a @~> b) =
  Mkinjection f is_inj

inline_for_extraction noextract
let inj_id #a : (a @~> a) = {
  f = id;
  is_inj = ez;
}

(* Apply an injection to a value. *)
inline_for_extraction
let ( |~> ) (#a #b : Type) (x : a) (i : a `injection` b) : b = i.f x

val lem_pat (#a #b : _) (d : a @~> b) (x y : a)
  : Lemma (d.f x == d.f y ==> x == y)
          [SMTPat (d.f x); SMTPat (d.f y)]

#push-options "--warn_error -288"
val lem_forall_pat (#a #b : _) (d : a @~> b)
  : Lemma (forall x y. d.f x == d.f y ==> x == y)
          [SMTPat (has_type d (a @~> b))] // OK? Useful?
#pop-options

let image_of (#a #b: Type) (i: a @~> b) : Type = FStar.Functions.image_of i.f

let inverse_f (#a #b : Type) (i : a @~> b) (y : image_of i) : GTot a =
  FStar.IndefiniteDescription.indefinite_description_ghost a
    (fun (x:a) -> i.f x == y)

(* An injection can be inverted, but this requires choice. *)
let inverse (#a #b : Type) (i : a @~> b) : Ghost.erased (image_of i @~> a) = {
  f = ghost_pull (inverse_f i);

  is_inj = ez;
}

let ( <~| ) (#a #b : Type) (y : b) (i : a `injection` b{in_image i.f y}) : GTot a =
  y |~> inverse i

inline_for_extraction noextract
let inj_prod (i1 : 'a @~> 'c) (i2 : 'b @~> 'd) : ('a & 'b @~> 'c & 'd) =
{
  f = (fun (a,b) -> (i1.f a, i2.f b));
  is_inj = ez;
}

inline_for_extraction noextract
let inj_either (i1 : 'a @~> 'c) (i2 : 'b @~> 'd) : (either 'a 'b @~> either 'c 'd) =
{
  f = (function | Inl a -> Inl (i1.f a) | Inr b -> Inr (i2.f b));
  is_inj = ez;
}

inline_for_extraction noextract
let inj_comp (i1 : 'a @~> 'b) (i2 : 'b @~> 'c) : ('a @~> 'c) =
{
  f = i2.f `o` i1.f;
  is_inj = ez;
}

unfold
let inj_nat_sum_f (n1 n2 : nat) : either (natlt n1) (natlt n2) -> natlt (n1 + n2) =
  function
    | Inl i -> i
    | Inr j -> n1 + j

let inj_nat_sum (n1 n2 : nat) : (either (natlt n1) (natlt n2) @~> natlt (n1 + n2)) =
{
  f = inj_nat_sum_f n1 n2;
  is_inj = ez; (* does not work if inj_nat_sum_f is inlined. *)
}

inline_for_extraction noextract
let inj_sz_sum_f
  (n1 n2 : Ghost.erased nat{SZ.fits (n1+n2)})
  (s1 : SZ.t{SZ.v s1 == Ghost.reveal n1})
: either (szlt n1) (szlt n2) -> szlt (n1 + n2)
=
  function
    | Inl i -> i
    | Inr j -> s1 +^ j

// n2 is not really needed
inline_for_extraction noextract
let inj_sz_sum
  (n1 n2 : Ghost.erased nat{SZ.fits (n1+n2)})
  (s1 : SZ.t{SZ.v s1 == Ghost.reveal n1})
: (either (szlt n1) (szlt n2) @~> szlt (n1 + n2)) =
{
  f = inj_sz_sum_f n1 n2 s1;
  is_inj = ez; (* does not work if inj_sz_sum_f is inlined. *)
}

val inj_cardinal (n1 n2 : nat)
  : Lemma (requires exists (b : natlt n1 @~> natlt n2). True)
          (ensures n1 <= n2)
