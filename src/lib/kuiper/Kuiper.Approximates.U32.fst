module Kuiper.Approximates.U32

open Kuiper.IntAliases
open Kuiper.Scalars
open Kuiper.Approximates.Base
open FStar.Real
open Kuiper.Math.Silly { mod_prod }

(* For the integer types, we can actually define the relation
and prove it. BUT, we must consider overflow! So we make the relation
weaker than you may expect. *)
instance real_like_u32 : real_like u32 = {
  to_real = (fun x -> Real.of_int (UInt32.v x));

  v_approximates = (fun x r ->
    exists (x' : int).
      r == Real.of_int x' /\ UInt32.v x == x' % 0x100000000);

  to_real_ok = (fun x -> ());

  a0 = ();
  a1 = ();

  a_add = (fun x y r s ->
    let open FStar.UInt32 in
    assert (exists x'. r == Real.of_int x' /\ v x == x' % 0x100000000);
    assert (exists y'. s == Real.of_int y' /\ v y == y' % 0x100000000);
    let x' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun x' -> r == Real.of_int x' /\ v x == x' % 0x100000000) in
    let y' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun y' -> s == Real.of_int y' /\ v y == y' % 0x100000000) in
    let z' = x' + y' in
    assert (r +. s == Real.of_int z');
    ()
  );

  a_mul = (fun x y r s ->
    let open FStar.UInt32 in
    assert (exists x'. r == Real.of_int x' /\ v x == x' % 0x100000000);
    assert (exists y'. s == Real.of_int y' /\ v y == y' % 0x100000000);
    let x' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun x' -> r == Real.of_int x' /\ v x == x' % 0x100000000) in
    let y' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun y' -> s == Real.of_int y' /\ v y == y' % 0x100000000) in
    let z' = x' * y' in
    mod_prod x' y' 0x100000000;
    assert (z' % 0x100000000 == UInt32.v (x *%^ y));
    assert (r *. s == Real.of_int z');
    ()
  );
}

instance precise_real_like_u32 : precise_real_like u32 = {
  v_approximates_inj = (fun x y r -> ());
}
