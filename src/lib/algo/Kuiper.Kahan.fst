module Kuiper.Kahan

open Kuiper.Sum { sum }
#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Sum { sum, sum_pop_right }

let sum_step (len : nat) (vf : natlt len -> GTot real) (k : nat{k < len})
  : Lemma (sum 0 (k+1) vf == sum 0 k vf +. vf k)
  = sum_pop_right 0 (k+1) vf

#push-options "--z3rlimit 20 --fuel 1 --ifuel 1"
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
    let old_c = !c;
    let old_acc = !acc;
    let yc = y `sub` old_c;
    sub_approx y old_c (vf !k) 0.0R;
    let t = old_acc `add` yc;
    a_add old_acc yc (sum 0 !k vf) (vf !k -. 0.0R);
    sum_step len vf !k;
    assert pure (t %~ sum 0 (!k + 1) vf);

    // Do not leave the approximation rules for this nested expression to
    // SMT's quantifier matching.  On a busy context that made the final loop
    // VC exhaust several successively larger solver limits.
    let delta = t `sub` old_acc;
    sub_approx t old_acc (sum 0 (!k + 1) vf) (sum 0 !k vf);
    let new_c = delta `sub` yc;
    sub_approx delta yc
      (sum 0 (!k + 1) vf -. sum 0 !k vf)
      (vf !k -. 0.0R);
    assert pure (new_c %~ 0.0R);
    c := new_c;
    acc := t;
    k   := !k +^ 1sz;
  };
  !acc
}
#pop-options
