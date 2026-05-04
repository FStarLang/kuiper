module Kuiper.Kahan

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Sum { sum, sum_pop_right, real_add_semigroup }

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
    invariant live c ** pure (!c %~ 0.0R)
    decreases (len - !k)
  {
    let y = f !k;
    assert pure (y %~ vf !k);
    let yc = y `sub` !c;
    sub_approx y !c (vf !k) 0.0R;
    assert pure (yc %~ vf !k);
    let t = !acc `add` yc;
    a_add (!acc) yc (sum 0 !k vf) (vf !k);
    sum_step len vf !k;
    assert pure (t %~ sum 0 (!k + 1) vf);
    sub_approx t !acc (sum 0 (!k + 1) vf) (sum 0 !k vf);
    assert pure (t `sub` !acc %~ vf !k);
    c := (t `sub` !acc) `sub` yc;
    sub_approx (t `sub` !acc) yc (vf !k) (vf !k);
    assert pure (!c %~ 0.0R);
    acc := t;
    k   := !k +^ 1sz;
  };
  !acc
}
