module Kuiper.Approximates.U16

open Kuiper
open Kuiper.Scalars
open Kuiper.Approximates.Class
open FStar.Real
open Kuiper.Math.Silly { mod_prod }

(* For the integer types, we can actually define the relation
and prove it. BUT, we must consider overflow! So we make the relation
weaker than you may expect. *)
instance real_like_u16 : real_like u16 = {
  to_real = (fun x -> Real.of_int (UInt16.v x));

  approximates = (fun x r ->
    exists (x' : int).
      r == Real.of_int x' /\ UInt16.v x == x' % 0x10000);

  to_real_ok = (fun x -> ());

  a0 = ();
  a1 = ();

  a_add = (fun x y r s ->
    let open FStar.UInt16 in
    assert (exists x'. r == Real.of_int x' /\ v x == x' % 0x10000);
    assert (exists y'. s == Real.of_int y' /\ v y == y' % 0x10000);
    let x' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun x' -> r == Real.of_int x' /\ v x == x' % 0x10000) in
    let y' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun y' -> s == Real.of_int y' /\ v y == y' % 0x10000) in
    let z' = x' + y' in
    assert (r +. s == Real.of_int z');
    ()
  );

  a_mul = (fun x y r s ->
    let open FStar.UInt16 in
    assert (exists x'. r == Real.of_int x' /\ v x == x' % 0x10000);
    assert (exists y'. s == Real.of_int y' /\ v y == y' % 0x10000);
    let x' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun x' -> r == Real.of_int x' /\ v x == x' % 0x10000) in
    let y' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun y' -> s == Real.of_int y' /\ v y == y' % 0x10000) in
    let z' = x' * y' in
    mod_prod x' y' 0x10000;
    assert (z' % 0x10000 == UInt16.v (x *%^ y));
    assert (r *. s == Real.of_int z');
    ()
  );
}
