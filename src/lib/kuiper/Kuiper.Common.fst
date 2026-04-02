module Kuiper.Common

open Pulse.Lib.Core
open Kuiper.Divides
module SZ = Kuiper.SizeT

(* Some base definitions we want everywhere, only over F* and Pulse constructs.
This module should have no Kuiper dependencies. *)

include FStar.Mul
include FStar.Tactics.Typeclasses { solve, solve_debug }

type natlt (b:int) = n:nat{n <  b}
type natle (b:int) = n:nat{n <= b}

let between (lo hi : nat) : Type =
  x:nat{lo <= x /\ x < hi}

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

let max x y = if x > y then x else y
let min x y = if x < y then x else y

let rec sum_n (#n:nat) (f : natlt n -> GTot nat) : GTot nat =
  if n = 0 then 0
  else
    f (n-1) + sum_n #(n-1) f

let rec sum_n_pop (#n:nat) (f : natlt (n+1) -> GTot nat)
  : Lemma (ensures sum_n #(n+1) f == sum_n #n f + f n)
  = if n = 0 then
      ()
    else
      sum_n_pop #(n-1) f

let rec max_n (#n:pos) (f : natlt n -> GTot nat) : GTot nat =
  if n = 1 then f 0
  else
    max (max_n #(n-1) f) (f (n-1))

let rec max_n_lem (#n:pos) (f : natlt n -> GTot nat)
  : Lemma (requires True)
          (ensures  (forall i. f i <= max_n f)
                 /\ (exists i. f i == max_n f))
          [SMTPat (max_n f)]
  = if n = 1 then
      assert (max_n f == f 0)
    else
      max_n_lem #(n-1) f

(* Some projectors *)
let pi_2_0 (x, _) = x
let pi_2_1 (_, x) = x
let pi_3_0 (x, _, _) = x
let pi_3_1 (_, x, _) = x
let pi_3_2 (_, _, x) = x
let pi_4_0 (x, _, _, _) = x
let pi_4_1 (_, x, _, _) = x
let pi_4_2 (_, _, x, _) = x
let pi_4_3 (_, _, _, x) = x
let pi_5_0 (x, _, _, _, _) = x
let pi_5_1 (_, x, _, _, _) = x
let pi_5_2 (_, _, x, _, _) = x
let pi_5_3 (_, _, _, x, _) = x
let pi_5_4 (_, _, _, _, x) = x
let pi_6_0 (x, _, _, _, _, _) = x
let pi_6_1 (_, x, _, _, _, _) = x
let pi_6_2 (_, _, x, _, _, _) = x
let pi_6_3 (_, _, _, x, _, _) = x
let pi_6_4 (_, _, _, _, x, _) = x
let pi_6_5 (_, _, _, _, _, x) = x
