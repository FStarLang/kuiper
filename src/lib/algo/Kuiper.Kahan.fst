module Kuiper.Kahan

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Sum { sum, sum_pop_right }

let sum_step (len : nat) (vf : natlt len -> GTot real) (k : nat{k < len})
  : Lemma (sum 0 (k+1) vf == sum 0 k vf +. vf k)
  = sum_pop_right 0 (k+1) vf

inline_for_extraction noextract
fn kahan_sum
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (len : sz)
  (frame : slprop)
  (vf : natlt len -> real) (* spec function *)
  (f : fn (i:szlt len)
         preserves frame
         returns   r : et
         ensures   pure (r %~ vf i))
  preserves
    frame
  returns
    res : et
  ensures
    pure (res %~ sum 0 len vf)
{
  let mut k : szle len = 0sz;
  let mut acc : et = zero;
  let mut c : et = zero; // compensation

  while (!k <^ len)
    invariant live k
    invariant live acc ** pure (!acc %~ sum 0 !k vf)
    invariant live c   ** pure (!c %~ 0.0R)
    decreases (len - !k)
  {
    let y = f !k;
    let yc = y `sub` !c;
    let t = !acc `add` yc;
    sum_step len vf !k;
    c := (t `sub` !acc) `sub` yc;
    acc := t;
    k   := !k +^ 1sz;
  };
  !acc
}
