module Kuiper.Approximates

open Kuiper
open FStar.Real
open Kuiper.Scalars
module M = FStar.Math.Lemmas

(* We assume these types approximate reals. We cannot really prove it
without a model of floating point (in both devices). *)
let real_like_f16 : real_like f16 = magic ()
let real_like_f32 : real_like f32 = magic ()
let real_like_f64 : real_like f64 = magic ()

let mod_prod (a b : int) (k : pos) :
  Lemma (ensures (a % k) * (b % k) % k == (a * b) % k)
  = M.lemma_mod_mul_distr_l a b k;
    M.lemma_mod_mul_distr_r (a % k) b k;
    ()

(* For the integer types, we can actually define the relation
and prove it. BUT, we must consider overflow! So we make the relation
weaker than you may expect. *)
instance real_like_u8 : real_like u8 = {
  approximates = (fun x r ->
    exists (x' : int).
      r == Real.of_int x' /\ UInt8.v x == x' % 256);

  a0 = ();
  a1 = ();

  a_add = (fun x y r s ->
    let open FStar.UInt8 in
    assert (exists x'. r == Real.of_int x' /\ v x == x' % 256);
    assert (exists y'. s == Real.of_int y' /\ v y == y' % 256);
    let x' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun x' -> r == Real.of_int x' /\ v x == x' % 256) in
    let y' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun y' -> s == Real.of_int y' /\ v y == y' % 256) in
    let z' = x' + y' in
    assert (r +. s == Real.of_int z');
    ()
  );

  a_mul = (fun x y r s ->
    let open FStar.UInt8 in
    assert (exists x'. r == Real.of_int x' /\ v x == x' % 256);
    assert (exists y'. s == Real.of_int y' /\ v y == y' % 256);
    let x' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun x' -> r == Real.of_int x' /\ v x == x' % 256) in
    let y' = FStar.IndefiniteDescription.indefinite_description_ghost int (fun y' -> s == Real.of_int y' /\ v y == y' % 256) in
    let z' = x' * y' in
    mod_prod x' y' 256;
    assert (z' % 256 == UInt8.v (x *%^ y));
    assert (r *. s == Real.of_int z');
    ()
  );
}

(* The rest are essentially the same *)
let real_like_u16 : real_like u16 = magic()
let real_like_u32 : real_like u32 = magic()
let real_like_u64 : real_like u64 = magic()
