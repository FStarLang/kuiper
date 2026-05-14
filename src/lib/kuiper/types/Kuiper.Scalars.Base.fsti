module Kuiper.Scalars.Base

open Kuiper.Sized
open FStar.Tactics.Typeclasses { solve, tcinstance }

(* There are no scalar instances for signed ints, we do not have
total unconditional operations on them. *)

inline_for_extraction noextract
class scalar (t : Type) = {
  [@@@tcinstance]
  is_sized : sized t;

  add : t -> t -> t;
  mul : t -> t -> t;

  zero : t;
  one : t;

  lt  : t -> t -> bool;
  lte : t -> t -> bool;
  eq  : t -> t -> bool;

  (* Is this a mathematically valid element? I.e., not a NaN. *)
  valid : t -> bool;

  (* Laws. *)
  lte_is_lt_or_eq :
    (x : t) -> (y : t) ->
      Lemma (requires valid x /\ valid y) (ensures lte x y <==> lt x y \/ eq x y);

  (* x < y <==> not (y <= x) *)
  negate_lt_is_lte :
    (x : t) -> (y : t) ->
      Lemma (requires valid x /\ valid y) (ensures lt x y <==> not (lte y x));
}

(* Derived methods *)

inline_for_extraction noextract
let gt (#t:Type) {| scalar t |} (x : t) (y : t) : bool =
  lt y x

inline_for_extraction noextract
let gte (#t:Type) {| scalar t |} (x : t) (y : t) : bool =
  lte y x

inline_for_extraction noextract
let neq (#t:Type) {| scalar t |} (x : t) (y : t) : bool =
  not (eq x y)

(* This instance is a bit fake. Maybe we should remove it. It's useful
to use MS.matmul on real matrices too. *)
noextract
instance _ : scalar Real.real =
  let open FStar.Real in
  {
  is_sized = { size = 0sz; default = 0.0R };
  add = ( +. );
  mul = ( *. );
  zero = 0.0R;
  one = 1.0R;
  // FIXME: reals cannot be compared in Tot.
  // We're overdue for restructuring the class hierarchy.
  eq  = (fun _ _ -> false);
  lt  = (fun _ _ -> false);
  lte = (fun _ _ -> false);
  valid = (fun _ -> false);

  lte_is_lt_or_eq = (fun _ _ -> ());
  negate_lt_is_lte = (fun _ _ -> ());
}
