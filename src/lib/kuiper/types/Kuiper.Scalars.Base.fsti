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

  min_val : t;
  max_val : t;

  #[easy_fill()] min_val_is_valid : squash (valid min_val);
  #[easy_fill()] max_val_is_valid : squash (valid max_val);

  (* Laws. *)

  (* Equality is sound, at least for valid terms. *)
  #[easy_fill ()]
  eq_spec : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq x y <==> x == y)
          [SMTPat (eq x y)];

  (* x <= y <==> x < y or x == y *)
  #[easy_fill ()]
  lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures lte x y <==> lt x y \/ eq x y)
          [SMTPat (lte x y)];

  (* x < y <==> not (y <= x) *)
  #[easy_fill ()]
  negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures lt x y <==> not (lte y x))
          [SMTPat (lt x y)];

  (* Addition commutes. Note: this is true even for NaNs. *)
  #[easy_fill ()]
  add_comm : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq (add x y) (add y x))
          [SMTPat (add x y)];

  #[easy_fill ()]
  mul_comm : (x : t) -> (y : t) ->
    Lemma (requires valid x /\ valid y)
          (ensures eq (mul x y) (mul y x))
          [SMTPat (mul x y)];

  #[easy_fill ()]
  add_zero : (x : t) ->
    Lemma (requires valid x)
          (ensures eq (add x zero) x)
          [SMTPat (add x zero)];

  (* min and max are correct. *)
  #[easy_fill ()]
  min_max_val_spec : (x : t) ->
    Lemma (requires valid x)
          (ensures lte min_val x /\ lte x max_val)
          [SMTPat (lte min_val x)];
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
  admit();
  {
    is_sized = { size = 0sz; default = 0.0R };
  add = ( +. );
  mul = ( *. );
  zero = 0.0R;
  one = 1.0R;
  // bogus from here down
  min_val = 0.0R -. 100.0R;
  max_val = 100.0R;
  // FIXME: reals cannot be compared in Tot.
  // We're overdue for restructuring the class hierarchy.
  eq  = (fun _ _ -> false);
  lt  = (fun _ _ -> false);
  lte = (fun _ _ -> false);
  valid = (fun _ -> false);
}
