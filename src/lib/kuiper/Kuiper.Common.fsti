module Kuiper.Common

(* Some base definitions we want everywhere, only over F* and Pulse constructs.
This module should have no Kuiper dependencies. *)

include FStar.Mul
include FStar.Tactics.Typeclasses { solve }

type natlt (b:int) = n:nat{n <  b}
type natle (b:int) = n:nat{n <= b}

type szlt (n:nat) = i:FStar.SizeT.t{FStar.SizeT.v i < n}

(* really just ez = easy *)
let ez : #a:Type -> (#[Tactics.V2.easy_fill ()] _ : a) -> a = Tactics.V2.easy

