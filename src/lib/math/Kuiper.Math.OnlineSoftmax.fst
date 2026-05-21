module Kuiper.Math.OnlineSoftmax

#lang-pulse
open Kuiper
open Kuiper.Spec.Softmax
open Kuiper.Seq.Common

let exp_sum (s : seq real{len s > 0}) : real =
  rsum (seq_map rexp s)

let adjust_factor (s : seq real{len s > 0}) (x : real) : real =
  exp_sum s
  /.
  (exp_sum s +. rexp x)

let rec seq_max (s : seq real{len s > 0})
  : Tot real (decreases len s)
  =
  let SCons h t = view_seq s in
  match view_seq t with
  | SNil -> h
  | SCons _ _ -> rmax h (seq_max t)

let seq_max_cons_lem (s : seq real{len s > 0}) (x : real)
  : Lemma (ensures seq_max (s @+ seq![x]) == rmax (seq_max s) x)
          [SMTPat (seq_max (s @+ seq![x]))]
  = admit()

// should not restrict len s > 0 eventually,
// but the initial state makes things weird
noeq
type st (s : erased (seq real){len s > 0}) = {
  m : real; // maximum so far
  d : real; // denominator, i.e. sum of exponentials
  #m_ok : m == seq_max s;
  #d_ok : d >. 0.0R /\ d == exp_sum s /. rexp m;
}

let exp_sum_snoc_lem (s : seq real{len s > 0}) (x : real)
  : Lemma (exp_sum (s @+ seq![x]) == exp_sum s +. rexp x)
  = calc (==) {
    exp_sum (s @+ seq![x]);
    == {}
    rsum (seq_map rexp (s @+ seq![x]));
    == {}
    rsum (seq_map rexp s) +. rsum (seq_map rexp (seq![x]));
    == {}
    exp_sum s +. rexp x;
  }

let r_distr_r (a b c : real) : Lemma (a *. (b +. c) == a *. b +. a *. c) = ()
let r_distr_l (a b c : real{c =!= 0.0R}) : Lemma ((a +. b) /. c == a /. c +. b /. c) = ()

let cancel (a : real) (b : real{b =!= 0.0R}) : Lemma (a *. b /. b == a) = ()
let abcd_adcb (a b c d : real{b =!= 0.0R /\ d =!= 0.0R})
  : Lemma (a /. b *. c /. d == a /. d *. c /. b) = ()

let cancel' (a b c : real{b =!= 0.0R /\ c =!= 0.0R}) : Lemma ((a /. c) /. (b /. c) == a /. b) =
  calc (==) {
    (a /. c) /. (b /. c);
    == {}
    (a *. c) /. (c *. b);
  }

#push-options "--z3refresh --split_queries always --z3rlimit 20" // very bad smt performance below
let softmax_step #s (s0 : st s) (x : real) :
  res:(real & st (s @+ seq![x]))
   { fst res /. (snd res).d == softmax_real (s @+ seq![x]) @! len s }
  =
  let m' = rmax s0.m x in
  let y = rexp (x -. m') in
  exp_sum_snoc_lem s x;
  assert s0.d == exp_sum s /. rexp s0.m;
  let d' = s0.d *. rexp (s0.m -. m') +. y in
  (* Prove d' is correct. *)
  calc (==) {
    exp_sum (s @+ seq![x]) /. rexp m';
    == {}
    (exp_sum s +. rexp x) /. rexp m';
    == { r_distr_l (exp_sum s) (rexp x) (rexp m') }
    exp_sum s /. rexp m' +. rexp x /. rexp m';
    == { cancel (exp_sum s /. rexp m') (rexp s0.m) }
    exp_sum s /. rexp m' *. rexp s0.m /. rexp s0.m +. rexp x /. rexp m';
    == { abcd_adcb (exp_sum s) (rexp s0.m) (rexp s0.m) (rexp m') }
    exp_sum s /. rexp s0.m *. rexp s0.m /. rexp m' +. rexp x /. rexp m';
    == { assert rexp (s0.m -. m') == rexp s0.m /. rexp m' }
    s0.d *. rexp (s0.m -. m') +. rexp x /. rexp m';
    == { () }
    d';
  };
  (* Prove result is correct. *)
  calc (==) {
    y /. d';
    == {}
    rexp (x -. m') /. (exp_sum (s @+ seq![x]) /. rexp m');
    == {}
    (rexp x /. rexp m') /. (exp_sum (s @+ seq![x]) /. rexp m');
    == { cancel' (rexp x) (exp_sum (s @+ seq![x])) (rexp m') }
    rexp x  /. exp_sum (s @+ seq![x]);
    == {}
    softmax_real (s @+ seq![x]) @! len s;
  };
  assert (y /. d' == softmax_real (s @+ seq![x]) @! len s);
  y, { m=m'; d=d' }
#pop-options
