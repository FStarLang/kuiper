module Kuiper.Math.OnlineSoftmax

#lang-pulse
open Kuiper
open Kuiper.Spec.Softmax
open Kuiper.Seq.Common

// NB: this file follows naming conventions of the 
// original online softmax paper, whenever possible: https://arxiv.org/pdf/1805.02867
// s = sequence (generic)
// x = softmax input (sequence)
// xi = some element of x
// y = softmax output (sequence)
// yi = some element of y
// m = running max of x
// d = running denominator sum of x
// fxi = exp(xi-m) (actually from flash attention paper)


let exp_sum (x : seq real{len x > 0}) : r:real{r >. 0.0R} =
  rsum (seq_map rexp x)

let adjust_factor (x : seq real{len x > 0}) (xi : real) : real =
  exp_sum x
  /.
  (exp_sum x +. rexp xi)

let rec seq_max (s : seq real{len s > 0})
  : Tot real (decreases len s)
  =
  let SCons h t = view_seq s in
  match view_seq t with
  | SNil -> h
  | SCons _ _ -> rmax h (seq_max t)

let rec seq_max_cons_lem (s : seq real{len s > 0}) (si : real)
  : Lemma (ensures seq_max (s @+ seq![si]) == rmax si (seq_max s))
          (decreases len s)
          [SMTPat (seq_max (s @+ seq![si]))]
  = let SCons h t = view_seq s in
    assert Seq.equal (s @+ seq![si]) (Seq.cons h (t @+ seq![si]));
    match view_seq t with
    | SNil ->
      lem_rmax_comm si (seq_max s)
    | SCons _ _ ->
      calc (==) {
        seq_max (s @+ seq![si]);
        == {}
        rmax h (seq_max (t @+ seq![si]));
        == { seq_max_cons_lem t si }
        rmax h (rmax si (seq_max t));
        == { lem_rmax_assoc h si (seq_max t)}
        rmax (rmax h si) (seq_max t);
        //== { lem_rmax_comm h si }
        //rmax (rmax si h) (seq_max t);
        //== { lem_rmax_assoc si h (seq_max t) }
        //rmax si (rmax h (seq_max t));
        == { }
        rmax si (seq_max s);
      }

// should not restrict len s > 0 eventually,
// but the initial state makes things weird
noeq
type st (x : erased (seq real){len x > 0}) = {
  m : real; // maximum so far
  d : real; // denominator, i.e. the sum of exponentials, corrected
  #m_ok : m == seq_max x;
  #d_ok : d >. 0.0R /\ d == exp_sum x /. rexp m;
}

let exp_sum_snoc_lem (x : seq real{len x > 0}) (xi : real)
  : Lemma (exp_sum (x @+ seq![xi]) == exp_sum x +. rexp xi)
  = calc (==) {
    exp_sum (x @+ seq![xi]);
    == {}
    rsum (seq_map rexp (x @+ seq![xi]));
    == {}
    rsum (seq_map rexp x) +. rsum (seq_map rexp (seq![xi]));
    == {}
    exp_sum x +. rexp xi;
  }

let r_distr_r (a b c : real) : Lemma (a *. (b +. c) == a *. b +. a *. c) = ()
let r_distr_l (a b c : real{c =!= 0.0R}) : Lemma ((a +. b) /. c == a /. c +. b /. c) = ()

let cancel_md (a : real) (b : real{b =!= 0.0R}) : Lemma (a *. b /. b == a) = ()
let cancel_dm (a : real) (b : real{b =!= 0.0R}) : Lemma (a /. b *. b == a) = ()
let abcd_adcb (a b c d : real{b =!= 0.0R /\ d =!= 0.0R})
  : Lemma (a /. b *. c /. d == a /. d *. c /. b) = ()

let assoc_mul (a b c: real) : Lemma ((a *. b) *. c == a *. (b *. c)) = ()

let cancel_ddd (a b c : real{b =!= 0.0R /\ c =!= 0.0R}) : Lemma ((a /. c) /. (b /. c) == a /. b) =
  calc (==) {
    (a /. c) /. (b /. c);
    == {}
    (a /. c *. c) /. (b /. c *. c);
    == {}
    a /. b;
  }

let assoc_mul_div (a b c : real{c =!= 0.0R}) : Lemma ((a *. b) /. c == a *. (b /. c)) =
  ()

#push-options "--split_queries always --z3rlimit 5 --retry 5" // very bad smt performance below
#restart-solver
let softmax_step #x (xst : st x) (xi : real) :
  res:(real & st (x @+ seq![xi]) & real)
   { (let (fxi,xst',adj) = res in
    fxi /. xst'.d == softmax_real (x @+ seq![xi]) @! len x /\
    adj == rexp (xst.m -. xst'.m)) }
  =
  admit ();
  let m' = rmax xst.m xi in
  seq_max_cons_lem x xi;
  assert m' == seq_max (x @+ seq![xi]);
  let fxi = rexp (xi -. m') in
  let adj = rexp (xst.m -. m') in
  exp_sum_snoc_lem x xi;
  assert xst.d == exp_sum x /. rexp xst.m;
  let d' = xst.d *. adj +. fxi in
  assert (d' >. 0.0R);
  (* Prove d' is correct. *)
  calc (==) {
    exp_sum (x @+ seq![xi]) /. rexp m';
    == {}
    (exp_sum x +. rexp xi) /. rexp m';
    == { r_distr_l (exp_sum x) (rexp xi) (rexp m') }
    exp_sum x /. rexp m' +. rexp xi /. rexp m';
    == { cancel_md (exp_sum x /. rexp m') (rexp xst.m) }
    exp_sum x /. rexp m' *. rexp xst.m /. rexp xst.m +. rexp xi /. rexp m';
    == { abcd_adcb (exp_sum x) (rexp xst.m) (rexp xst.m) (rexp m') }
    exp_sum x /. rexp xst.m *. rexp xst.m /. rexp m' +. rexp xi /. rexp m';
    == { assoc_mul_div (exp_sum x /. rexp xst.m) (rexp xst.m) (rexp m') }
    exp_sum x /. rexp xst.m *. (rexp xst.m /. rexp m') +. rexp xi /. rexp m';
    == { exp_sub xst.m m' }
    exp_sum x /. rexp xst.m *. rexp (xst.m -. m') +. rexp xi /. rexp m';
    == {}
    xst.d *. adj +. rexp xi /. rexp m';
    == { () }
    d';
  };
  (* Prove result is correct. *)
  calc (==) {
    fxi /. d';
    == {}
    rexp (xi -. m') /. (exp_sum (x @+ seq![xi]) /. rexp m');
    == {}
    (rexp xi /. rexp m') /. (exp_sum (x @+ seq![xi]) /. rexp m');
    == { cancel_ddd (rexp xi) (exp_sum (x @+ seq![xi])) (rexp m') }
    rexp xi  /. exp_sum (x @+ seq![xi]);
    == {}
    softmax_real (x @+ seq![xi]) @! len x;
  };
  assert (fxi /. d' == softmax_real (x @+ seq![xi]) @! len x);
  fxi, { m=m'; d=d' }, adj
#pop-options

#push-options "--split_queries always --z3rlimit 10"
let softmax_stepi (#n:pos) (x : lseq real n) (i: pos{i < n}) (xst : st (seq_take i x)) :
  res:(real & st (seq_take (i+1) x) & real)
   { (let (fxi,xst',adj) = res in
    fxi /. xst'.d == softmax_real (seq_take (i+1) x) @! (len (seq_take i x)) /\
    (i == (len (seq_take i x))) /\ // TODO shouldnt be necessary 
    adj == rexp (xst.m -. xst'.m)) } = admit ()

open Kuiper.DotProd

#push-options "--split_queries always --z3rlimit 10"
let lem_online_softmax_adj
  (#n : pos)
  (x : lseq real n)
  (i: pos {i < n}) 
  (xst : st (seq_take i x))
  (k: real -> natlt (Seq.length x) -> real)  (* continuation: what to do with e^xi *)
  (#k_comm_div: (nr: real) -> (dr: real { dr =!= 0.0R } ) -> (j: natlt n) -> (k nr j) /. dr == k (nr /. dr) j) 
  (op: real -> real -> real)
  (#op_distrib_mul: (a: real) -> (b: real) -> (r: real) -> (r *. a) `op` (r *. b) == r *. (a `op` b))
  (init: real)
  (#r : real):
  (* TODO: get rid of the extra constraint here. The lemma still holds when k = 0, but we just can't call softmax_real on an empty sequence
    (seq_take 0 z) because that would entail dividing by 0. *)
  (* maybe it just needs to be a "softmax_real_safe" that returns an empty sequence given an empty sequence as input.
    Because it would also not be possible to have the invariant below hold before the loop (when k = 0) on the client side code. *)
  Lemma 
        (requires r /. xst.d == (seq_fold_left op init (seq_mapi (softmax_real (seq_take i x)) k)))
        (ensures 
          (let (fxi,xst',adj) = softmax_stepi #n x i xst in
          (r *. adj `op` (k fxi i)) /. xst'.d == (seq_fold_left op init (seq_mapi (softmax_real (seq_take (i+1) x)) k)))) =
  
    let (fxi,xst',adj) = softmax_stepi #n x i xst in

    (*calc (==) {
       (k fxi i /. xst'.d);
       == { k_comm_div fxi xst'.d i }
       (k (fxi /. xst'.d) i);
       == {}
       (k (softmax_real (seq_take (i+1) x) @! i) i);
    };*)
    
    assert (rexp (xst.m -. xst'.m) == rexp (xst.m) /. rexp (xst'.m));

    let x_upto_i = seq_take i x in
    let x_upto_i' = seq_take (i+1) x in
    let v1 = (seq_fold_left op init (seq_mapi (softmax_real (x_upto_i)) k)) in
    calc (==) {
      r *. adj /. xst'.d;
      == { assoc_mul_div r adj xst'.d }
      r *. (adj /. xst'.d);
      == {}
      r *. (rexp (xst.m) /. rexp (xst'.m) /. xst'.d);
      == {}
      r *. (rexp (xst.m) /. rexp (xst'.m) /. (exp_sum x_upto_i' /. (rexp xst'.m)));
      == { cancel_ddd (rexp xst.m) (exp_sum x_upto_i') (rexp xst'.m)}
      r *. (rexp (xst.m) /. (exp_sum x_upto_i'));
      == { cancel_dm r xst.d }
      v1 *. xst.d *. (rexp (xst.m) /. (exp_sum x_upto_i'));
      == { assoc_mul v1 xst.d (rexp (xst.m) /. (exp_sum x_upto_i')) }
      v1 *. (xst.d *. (rexp (xst.m) /. (exp_sum x_upto_i')));
      == { }
      v1 *. ((exp_sum x_upto_i /. (rexp xst.m)) *. (rexp (xst.m) /. (exp_sum x_upto_i')));
      == { assoc_mul_div (exp_sum x_upto_i /. (rexp xst.m)) (rexp xst.m) (exp_sum x_upto_i') }
      v1 *. (((exp_sum x_upto_i /. (rexp xst.m)) *. rexp (xst.m)) /. (exp_sum x_upto_i'));
      == { cancel_dm (exp_sum x_upto_i) (rexp xst.m) }
      v1 *. ((exp_sum x_upto_i) /. (exp_sum x_upto_i'));
    };

    // assume (forall (a b: real) (r: real { r =!= 0.0R }). (a /. r) `op` (b /. r) == (a `op` b) /. r); // TODO probably provable from above

    admit ()
(*

let rec aux (#n : pos) (x y : lseq real n)  (#i : pos{i <= n})
    (state : st (seq_take i x))
    (r : real{r /. state.d == seq_dotprod' (softmax_real x) y i})
    : Tot (r:real{r == seq_dotprod (softmax_real x) y})
          (decreases n-i)
  = let _ = if i = n
    then (
      r /. state.d
    ) else (
      let fx, state', _ = softmax_step state (x @! i) in
      let adj = rexp (state.m -. state'.m) in
      let r' = r *. adj   +.   fx *. (y @! i) in
      let z = fx *. (y @! i) in
      let tx = seq_take i x in 
      assume Seq.equal (seq_take (i+1) x) (seq_take i x @+ seq![x @! i]); // seems ok...
      let state' : st (seq_take (i+1) x) = coerce_eq () state' in
      assert (len tx == i);
      assert (fx /. state'.d == (softmax_real (tx @+ seq![x @! i])) @! i);
      assert (fx /. state'.d == (softmax_real (seq_take (i+1) x)) @! i);
      assert ((fx /. state'.d) *. (y @! i) == (softmax_real (seq_take (i+1) x) @! i) *. (y @! i));
      assume ((fx /. state'.d) *. (y @! i) == z /. state'.d); // ???
      
      assert (rexp (state.m -. state'.m) == rexp (state.m) /. rexp (state'.m));
      assume (r == seq_dotprod' (softmax_real x) y i *. state.d ); // ???
      calc (==) {
        (r *. adj /. state'.d);
        == {}
        ((seq_dotprod' (softmax_real x) y i) *. state.d *. adj /. state'.d);
        == {}
        ((seq_dotprod' (softmax_real x) y i) *. (exp_sum (seq_take i x) /. rexp state.m) *. adj /. (exp_sum (seq_take (i+1) x) /. rexp state'.m));
        == {}
        ((seq_dotprod' (softmax_real x) y i) *. (exp_sum (seq_take i x) /. exp_sum (seq_take (i+1) x)) *. (rexp state'.m /. rexp state.m) *. adj);
        == {} 
        ((seq_dotprod' (softmax_real x) y i) *. (exp_sum (seq_take i x) /. exp_sum (seq_take (i+1) x)));      
      };
      admit ();

      calc (==) {
        (r' /. state'.d);
        == {}
        (r *. adj +. fx *. (y @! i)) /. state'.d;
        == {}
        (r *. adj /. state'.d +. fx *. (y @! i) /. state'.d);
        == {}
        (r *. adj /. state'.d +. (softmax_real (seq_take (i+1) x) @! i) *. (y @! i));
        == {}
        ((seq_dotprod' (softmax_real x) y i) *. (exp_sum (seq_take i x) /. exp_sum (seq_take (i+1) x)) 
          +. (softmax_real (seq_take (i+1) x) @! i) *. (y @! i));
        // multiple non-trivial steps: the sum i cancels with seq_dotprod' (softmax_real ...), 
        // then we get the correct sum of i+1, then we add the ith element
        == { admit () } 
        ((seq_dotprod' (softmax_real x) y (i+1)));
      };
      admit ();
      
      assert (r /. state.d == seq_dotprod' (softmax_real x) y i);
      assert (r' /. state'.d == seq_dotprod' (softmax_real x) y (i+1));
      aux #(n) x y #(i+1) state' r'
    ) in 
    magic ()
*)

(*
#set-options "--split_queries always --z3rlimit 15"
let softmax_dotprod (#n : pos) (x y : lseq real n) : r:real{r == seq_dotprod (softmax_real x) y} =
  let rec aux (#i : pos{i <= n})
    (state : st (seq_take i x))
    (r : real{r /. state.d == seq_dotprod' (softmax_real x) y i})
    : Tot (r:real{r == seq_dotprod (softmax_real x) y})
          (decreases n-i)
  =
    let _ = if i = n
    then (
      r /. state.d
    ) else (
      let fx, state', _ = softmax_step state (x @! i) in
      let adj = rexp (state.m -. state'.m) in
      let r' = r *. adj   +.   fx *. (y @! i) in
      let z = fx *. (y @! i) in
      assert (seq_take (i+1) x == seq_take i x @+ seq![x @! i]);
      admit();
      assert (z /. state'.d == (softmax_real (seq_take (i+1) x) @! i) *. (y @! i));
      assume Seq.equal (seq_take (i+1) x) (seq_take i x @+ seq![x @! i]); // seems ok...
      let state' : st (seq_take (i+1) x) = coerce_eq () state' in
      assert (r /. state.d == seq_dotprod' (softmax_real x) y i);
      assert (r' /. state'.d == seq_dotprod' (softmax_real x) y (i+1));
      aux #(i+1) state' r'
    ) in
    magic ()
  in
  admit();
  let r : r:real{r == seq_dotprod (softmax_real x) y} = magic() in
  let s : st (seq_take 1 x) = magic() in
  aux s r *)