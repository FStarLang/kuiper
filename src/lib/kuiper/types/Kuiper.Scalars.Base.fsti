module Kuiper.Scalars.Base

open Kuiper.Sized
open Kuiper.Canonical
open FStar.Tactics.Typeclasses { tcinstance }

(* There are no scalar instances for signed ints, we do not have
total unconditional operations on them. *)

inline_for_extraction noextract
class scalar (t : Type) = {
  (* Pinning the [sized] superclass to the canonical instance makes a
  [sized t] resolved via [scalar t] and one resolved directly the same
  instance -- see [Kuiper.Canonical]. *)
  [@@@tcinstance]
  is_sized : (x : sized t{canonical x});

  add : t -> t -> t;
  mul : t -> t -> t;

  zero : t;
  one : t;

  lt  : t -> t -> bool;
  lte : t -> t -> bool;
  eq  : t -> t -> bool;
}

(* Derived methods *)

inline_for_extraction noextract
let gt (#t:Type) {| scalar t |} (x y : t) : bool =
  lt y x

inline_for_extraction noextract
let gte (#t:Type) {| scalar t |} (x y : t) : bool =
  lte y x

inline_for_extraction noextract
let neq (#t:Type) {| scalar t |} (x y : t) : bool =
  not (eq x y)

(* This instance is a bit fake. Maybe we should remove it. It's useful
to use MS.matmul on real matrices too. *)
(* Deliberately not an [instance]: it only needs a name for the [canonical]
assumption below and for [is_sized] here. Exposing it to resolution would
add a second candidate for [sized Real.real] alongside the one reached
through [scalar Real.real], which is the very ambiguity [canonical] exists
to remove. *)
inline_for_extraction noextract
let fake_sized_real : sized Real.real =
  { size = 0sz; default = 0.0R }

assume Canonical_fake_sized_real : canonical fake_sized_real

inline_for_extraction noextract
instance _ : scalar Real.real =
  let open FStar.Real in
  {
    is_sized = fake_sized_real;
    add = ( +. );
    mul = ( *. );
    zero = 0.0R;
    one = 1.0R;
    // FIXME: reals cannot be compared in Tot.
    // We're overdue for restructuring the class hierarchy.
    eq  = (fun _ _ -> false);
    lt  = (fun _ _ -> false);
    lte = (fun _ _ -> false);
}
