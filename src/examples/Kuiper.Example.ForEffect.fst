module Kuiper.Example.ForEffect

#lang-pulse

open Kuiper
open Kuiper.For

module U32 = FStar.UInt32

(* A [for_loop'] whose body has an observable effect and captures both [r] and
   [k].  TestFor's loops are all [emp]-to-[emp], so Custard is entitled to
   delete them outright; this one has to survive into the C. *)
fn accum (r : ref U32.t) (k : U32.t)
  requires exists* v. r |-> v
  ensures  exists* v. r |-> v
{
  forevery_emp_intro (between 0sz 10sz);
  for_loop' 0sz 10sz
    (fun x -> emp)
    (fun x -> emp)
    (exists* v. r |-> v)
    fn x { let v = !r; r := U32.add_mod v k };
  forevery_emp_elim (between 0sz 10sz);
  ()
}
