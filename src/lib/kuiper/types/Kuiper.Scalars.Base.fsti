module Kuiper.Scalars.Base

open Kuiper.Sized
open FStar.Tactics.Easy
open FStar.Tactics.Typeclasses { solve, tcinstance }

(* There are no scalar instances for signed ints, we do not have
total unconditional operations on them. *)

inline_for_extraction noextract
class scalar (t : Type) = {
  [@@@tcinstance]is_sized : sized t;

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

  (* Equality is sound, at least for valid terms. *)
  #[easy_fill ()]
  eq_spec : (x : t) -> (y : t) ->
    valid x /\ valid y -> (eq x y <==> x == y);

  (* x <= y <==> x < y or x == y *)
  #[easy_fill ()]
  lte_is_lt_or_eq :
    (x : t) -> (y : t) ->
      valid x /\ valid y -> (lte x y <==> lt x y \/ eq x y);

  (* x < y <==> not (y <= x) *)
  #[easy_fill ()]
  negate_lt_is_lte :
    (x : t) -> (y : t) ->
      valid x /\ valid y -> (lt x y <==> not (lte y x));

  (* Addition commutes. Note: this is true even for NaNs. *)
  #[easy_fill ()]
  add_comm : (x : t) -> (y : t) ->
    valid x -> valid y ->
    eq (add x y) (add y x);

  #[easy_fill ()]
  mul_comm : (x : t) -> (y : t) ->
    valid x -> valid y ->
    eq (mul x y) (mul y x);

  #[easy_fill ()]
  add_zero : (x : t) ->
    valid x ->
    eq (add x zero) x;
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
}
