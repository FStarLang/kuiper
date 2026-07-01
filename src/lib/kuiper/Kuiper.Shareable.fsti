module Kuiper.Shareable

#lang-pulse
open Pulse
open Kuiper.ForEvery
open Kuiper.Common
open FStar.Tactics.Typeclasses

class shareable (res : perm -> slprop) = {
  [@@@no_method]
  _share_n :
    ghost fn (n : pos) (#p : perm)
      requires res p
      ensures forall+ (_ : natlt n). res (p /. Real.of_int n);
  [@@@no_method]
  _gather_n :
    ghost fn (n : pos) (#p : perm)
      requires forall+ (_ : natlt n). res (p /. Real.of_int n)
      ensures  res p;
}

ghost
fn share_n
  (res : perm -> slprop)  
  {| d : shareable res |}
  (n : pos) (#p : perm)
  requires res p
  ensures forall+ (_ : natlt n). res (p /. Real.of_int n)
{
  let f = d._share_n;
  f n;
}

ghost
fn gather_n
  (res : perm -> slprop)  
  {| d : shareable res |}
  (n : pos) (#p : perm)
  requires forall+ (_ : natlt n). res (p /. Real.of_int n)
  ensures  res p
{
  let f = d._gather_n;
  f n;
}

instance emp_shareable: shareable (fun _ -> emp) = {
  _share_n = (fun (n: pos) (#p : perm) -> forevery_emp_intro (natlt n));
  _gather_n = (fun (n: pos) (#p : perm) -> forevery_emp_elim (natlt n));
}


ghost fn double_share 
  (res1 : perm -> slprop) {| shareable res1 |}
  (res2 : perm -> slprop) {| shareable res2 |}
  (ffr1 ffr2: perm)
  (n: pos) (#fr : perm)
requires res1 (ffr1 *. fr) ** res2 (ffr2 *. fr)
ensures forall+ (_ : natlt n). res1 (ffr1 *. (fr /. Real.of_int n)) ** res2 (ffr2 *. (fr /. Real.of_int n))
{
  share_n res1 n #(ffr1 *. fr);
  share_n res2 n #(ffr2 *. fr);
  forevery_map 
    (fun (_ : natlt n) -> res1 ((ffr1 *. fr) /. Real.of_int n))
    (fun (_ : natlt n) -> res1 (ffr1 *. (fr /. Real.of_int n)))
    fn i { 
      rewrite res1 ((ffr1 *. fr) /. Real.of_int n) as 
        res1 (ffr1 *. (fr /. Real.of_int n));
     };
  forevery_map 
    (fun (_ : natlt n) -> res2 ((ffr2 *. fr) /. Real.of_int n))
    (fun (_ : natlt n) -> res2 (ffr2 *. (fr /. Real.of_int n)))
    fn i { 
      rewrite res2 ((ffr2 *. fr) /. Real.of_int n) as 
        res2 (ffr2 *. (fr /. Real.of_int n));
     };
  forevery_zip (fun (_ : natlt n) -> res1 (ffr1 *. (fr /. Real.of_int n))) _;
}

ghost fn double_gather
  (res1 : perm -> slprop) {| shareable res1 |}
  (res2 : perm -> slprop) {| shareable res2 |}
  (ffr1 ffr2: perm)
  (n: pos) (#fr : perm)
requires forall+ (_ : natlt n). res1 (ffr1 *. (fr /. Real.of_int n)) ** res2 (ffr2 *. (fr /. Real.of_int n))
ensures res1 (ffr1 *. fr) ** res2 (ffr2 *. fr)
{
  forevery_unzip _ _;
  forevery_map 
    (fun (_ : natlt n) -> res1 (ffr1 *. (fr /. Real.of_int n)))
    (fun (_ : natlt n) -> res1 ((ffr1 *. fr) /. Real.of_int n))
    fn i { 
      rewrite 
        res1 (ffr1 *. (fr /. Real.of_int n)) as 
        res1 ((ffr1 *. fr) /. Real.of_int n);
    };
  forevery_map 
    (fun (_ : natlt n) -> res2 (ffr2 *. (fr /. Real.of_int n)))
    (fun (_ : natlt n) -> res2 ((ffr2 *. fr) /. Real.of_int n))
    fn i { 
      rewrite 
        res2 (ffr2 *. (fr /. Real.of_int n)) as 
        res2 ((ffr2 *. fr) /. Real.of_int n);
    };
  gather_n res1 n;
  gather_n res2 n;
}

instance double_shareable
  (res1 : perm -> slprop) {| shareable res1 |}
  (res2 : perm -> slprop) {| shareable res2 |}
  (ffr1 ffr2: perm):
  (shareable (fun fr -> res1 (ffr1 *. fr) ** res2 (ffr2 *. fr))) = {
  _share_n = double_share res1 res2 ffr1 ffr2;
  _gather_n = double_gather res1 res2 ffr1 ffr2;
}