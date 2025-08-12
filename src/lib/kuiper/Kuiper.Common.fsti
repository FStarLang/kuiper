module Kuiper.Common

open Pulse.Lib.Core
open Kuiper.Divides
module SZ = FStar.SizeT

(* Some base definitions we want everywhere, only over F* and Pulse constructs.
This module should have no Kuiper dependencies. *)

include FStar.Mul
include FStar.Tactics.Typeclasses { solve, solve_debug }

type natlt (b:int) = n:nat{n <  b}
type natle (b:int) = n:nat{n <= b}
type posmultiple (k:int) = n:pos{k /? n}

type binop (t : Type) = t -> t -> t

(* Erased version, with refinement **on the outside** to prevent
against invariance of erased wrt types. *)
type enatlt (b:int) = n:(Ghost.erased nat){n <  b}

(* These two are useful when using size_t as a bound, to avoid
mismatches betweeen SZ.v (a *^ b) and SZ.v a * SZ.v b. *)
unfold
type natlt2
  (b1 : SZ.t)
  (b2 : SZ.t{SZ.fits (SZ.v b1 * SZ.v b2)}) = natlt (SZ.v (b1 `SZ.mul` b2))
unfold
type enatlt2
  (b1 : SZ.t)
  (b2 : SZ.t{SZ.fits (SZ.v b1 * SZ.v b2)}) = enatlt (SZ.v (b1 `SZ.mul` b2))

(* really just ez = easy *)
let ez : #a:Type -> (#[Tactics.V2.easy_fill ()] _ : a) -> a = Tactics.V2.easy

let divmod (j:pos) (i : nat) : (nat & natlt j) =
  (i / j, i % j)

let undivmod (j:pos) (xy : nat & natlt j) : nat =
  j * xy._1 + xy._2

let divmod_inv_1 (j:pos) (i:nat)
  : Lemma (undivmod j (divmod j i) == i)
  = ()

let divmod_inv_2 (j:pos) (xy : nat & natlt j)
  : Lemma (divmod j (undivmod j xy) == xy)
  = ()

(* Function composition. *)
inline_for_extraction noextract
let o (f : 'b -> 'c) (g : 'a -> 'b) : 'a -> 'c =
  fun x -> f (g x)

let oo (f : 'b -> GTot 'c) (g : 'a -> GTot 'b) : 'a -> GTot 'c =
  fun x -> f (g x)
