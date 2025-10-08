module Kuiper.GhostPull

(* An axiom. Used to be in the F* library. *)
val ghost_pull (#a #b:Type) (f: a -> GTot b) : GTot (g:(a -> b) { forall x. f x == g x })
