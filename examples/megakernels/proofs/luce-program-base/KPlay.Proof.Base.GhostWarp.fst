module KPlay.Proof.Base.GhostWarp

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.ForEvery
open FStar.Ghost

module GR = Pulse.Lib.GhostReference
module Every = Kuiper.ForEvery
module B = Kuiper.Bijection

ghost
fn two (p:slprop)
  requires p ** p
  ensures forall+ (_:Kuiper.Common.natlt 2). p
{
  Every.forevery_singleton_intro' #(Kuiper.Common.natlt 1) (fun _ -> p) 0;
  Every.forevery_natlt_push 2 (fun _ -> p);
}

ghost
fn double
  (#a:Type0) (reference:GR.ref a) (#value:erased a) (#fraction:perm) (count:pos)
  requires forall+ (_:Kuiper.Common.natlt count). GR.pts_to reference #(fraction /. Real.of_int count) value
  ensures forall+ (_:Kuiper.Common.natlt (2 * count)). GR.pts_to reference #(fraction /. Real.of_int (2 * count)) value
{
  Every.forevery_map #(Kuiper.Common.natlt count)
    (fun _ -> GR.pts_to reference #(fraction /. Real.of_int count) value)
    (fun _ -> Every.forevery (Kuiper.Common.natlt 2)
      (fun _ -> GR.pts_to reference #((fraction /. Real.of_int count) /. 2.0R) value))
    fn _ {
      GR.share reference #value #(fraction /. Real.of_int count);
      two (GR.pts_to reference #((fraction /. Real.of_int count) /. 2.0R) value);
    };
  Every.forevery_flatten #(Kuiper.Common.natlt count) #(Kuiper.Common.natlt 2)
    (fun _ _ -> GR.pts_to reference #((fraction /. Real.of_int count) /. 2.0R) value);
  Every.forevery_iso (B.bij_nat_prod #count #2)
    (fun (_:Kuiper.Common.natlt count & Kuiper.Common.natlt 2) ->
      GR.pts_to reference #((fraction /. Real.of_int count) /. 2.0R) value);
  Every.forevery_rw_type (Kuiper.Common.natlt (count * 2)) (Kuiper.Common.natlt (2 * count))
    (fun _ -> GR.pts_to reference #((fraction /. Real.of_int count) /. 2.0R) value);
  Every.forevery_ext #(Kuiper.Common.natlt (2 * count))
    (fun _ -> GR.pts_to reference #((fraction /. Real.of_int count) /. 2.0R) value)
    (fun _ -> GR.pts_to reference #(fraction /. Real.of_int (2 * count)) value);
}

ghost
fn share32
  (#a:Type0) (reference:GR.ref a) (#value:erased a) (#fraction:perm)
  requires GR.pts_to reference #fraction value
  ensures forall+ (_:Kuiper.Common.natlt 32). GR.pts_to reference #(fraction /. 32.0R) value
{
  Every.forevery_singleton_intro' #(Kuiper.Common.natlt 1) (fun _ -> GR.pts_to reference #fraction value) 0;
  Every.forevery_ext #(Kuiper.Common.natlt 1)
    (fun _ -> GR.pts_to reference #fraction value)
    (fun _ -> GR.pts_to reference #(fraction /. Real.of_int 1) value);
  double reference #value #fraction 1;
  double reference #value #fraction 2;
  double reference #value #fraction 4;
  double reference #value #fraction 8;
  double reference #value #fraction 16;
}

ghost
fn rec gather
  (#a:Type0) (reference:GR.ref a) (#value:erased a) (#fraction:perm) (count:pos)
  requires forall+ (_:Kuiper.Common.natlt count). GR.pts_to reference #(fraction /. Real.of_int count) value
  ensures GR.pts_to reference #fraction value
  decreases count
{
  if (count = 1) {
    Every.forevery_singleton_elim' #(Kuiper.Common.natlt count)
      (fun _ -> GR.pts_to reference #(fraction /. Real.of_int count) value) 0;
  } else {
    Every.forevery_natlt_pop count
      (fun _ -> GR.pts_to reference #(fraction /. Real.of_int count) value);
    let remainder = fraction -. (fraction /. Real.of_int count);
    Every.forevery_ext #(Kuiper.Common.natlt (count - 1))
      (fun _ -> GR.pts_to reference #(fraction /. Real.of_int count) value)
      (fun _ -> GR.pts_to reference #(remainder /. Real.of_int (count - 1)) value);
    gather reference #value #remainder (count - 1);
    GR.gather reference #value #value #(fraction /. Real.of_int count) #remainder;
    rewrite GR.pts_to reference #(fraction /. Real.of_int count +. remainder) value
      as GR.pts_to reference #fraction value;
  }
}
