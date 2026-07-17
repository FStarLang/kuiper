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
  rsum (seq_map exp x)

let adjust_factor (x : seq real{len x > 0}) (xi : real) : real =
  exp_sum x
  /.
  (exp_sum x +. exp xi)

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
  #d_ok : d >. 0.0R /\ d == exp_sum x /. exp m;
}

let exp_sum_snoc_lem (x : seq real{len x > 0}) (xi : real)
  : Lemma (exp_sum (x @+ seq![xi]) == exp_sum x +. exp xi)
  = calc (==) {
    exp_sum (x @+ seq![xi]);
    == {}
    rsum (seq_map exp (x @+ seq![xi]));
    == {}
    rsum (seq_map exp x) +. rsum (seq_map exp (seq![xi]));
    == {}
    exp_sum x +. exp xi;
  }

let r_distr_r (a b c : real) : Lemma (a *. (b +. c) == a *. b +. a *. c) = ()
let r_distr_l (a b c : real{c =!= 0.0R}) : Lemma ((a +. b) /. c == a /. c +. b /. c) = ()
let l_distr_r (a b c: real): Lemma ((a +. b) *. c == (a *. c) +. (b *. c)) = ()

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

let mul_one_div (a : real) (b : real{b =!= 0.0R}) : Lemma (a *. (1.0R /. b) == a /. b) = ()

#push-options "--split_queries always --z3rlimit 10"
let softmax_step #x (xst : st x) (xi : real) :
  res:(real & st (x @+ seq![xi]) & real)
   { (let (fxi,xst',adj) = res in
    fxi /. xst'.d == softmax_real_seq (x @+ seq![xi]) @! len x /\
    adj == exp (xst.m -. xst'.m)) }
  =
  let m' = rmax xst.m xi in
  seq_max_cons_lem x xi;
  assert m' == seq_max (x @+ seq![xi]);
  let fxi = exp (xi -. m') in
  let adj = exp (xst.m -. m') in
  exp_sum_snoc_lem x xi;
  assert xst.d == exp_sum x /. exp xst.m;
  let d' = xst.d *. adj +. fxi in
  assert (d' >. 0.0R);
  (* Prove d' is correct. *)
  calc (==) {
    exp_sum (x @+ seq![xi]) /. exp m';
    == {}
    (exp_sum x +. exp xi) /. exp m';
    == { r_distr_l (exp_sum x) (exp xi) (exp m') }
    exp_sum x /. exp m' +. exp xi /. exp m';
    == { cancel_md (exp_sum x /. exp m') (exp xst.m) }
    exp_sum x /. exp m' *. exp xst.m /. exp xst.m +. exp xi /. exp m';
    == { abcd_adcb (exp_sum x) (exp xst.m) (exp xst.m) (exp m') }
    exp_sum x /. exp xst.m *. exp xst.m /. exp m' +. exp xi /. exp m';
    == { assoc_mul_div (exp_sum x /. exp xst.m) (exp xst.m) (exp m') }
    exp_sum x /. exp xst.m *. (exp xst.m /. exp m') +. exp xi /. exp m';
    == { exp_sub xst.m m' }
    exp_sum x /. exp xst.m *. exp (xst.m -. m') +. exp xi /. exp m';
    == {}
    xst.d *. adj +. exp xi /. exp m';
    == { () }
    d';
  };
  (* Prove result is correct. *)
  calc (==) {
    fxi /. d';
    == {}
    exp (xi -. m') /. (exp_sum (x @+ seq![xi]) /. exp m');
    == {}
    (exp xi /. exp m') /. (exp_sum (x @+ seq![xi]) /. exp m');
    == { cancel_ddd (exp xi) (exp_sum (x @+ seq![xi])) (exp m') }
    exp xi  /. exp_sum (x @+ seq![xi]);
    == {}
    softmax_real_seq (x @+ seq![xi]) @! len x;
  };
  assert (fxi /. d' == softmax_real_seq (x @+ seq![xi]) @! len x);
  fxi, { m=m'; d=d' }, adj
#pop-options

(* [seq_take (i+1)] extends [seq_take i] by the i-th element. *)
let seq_take_snoc (#a:Type) (s : seq a) (i : nat{i < len s})
  : Lemma (seq_take (i+1) s == seq_take i s @+ seq![s @! i])
  = Seq.lemma_eq_elim (seq_take (i+1) s) (seq_take i s @+ seq![s @! i])

#push-options "--split_queries always --z3rlimit 10"
[@@"opaque_to_smt"] (* keep the body out of downstream (fragile) SMT contexts *)
let softmax_stepi (#n:pos) (x : lseq real n) (i: pos{i < n}) (xst : st (seq_take i x)) :
  res:(real & st (seq_take (i+1) x) & real)
   { (let (fxi,xst',adj) = res in
    fxi /. xst'.d == softmax_real_seq (seq_take (i+1) x) @! (len (seq_take i x)) /\
    (i == (len (seq_take i x))) /\ // TODO shouldnt be necessary
    adj == exp (xst.m -. xst'.m)) }
  = (* [seq_take (i+1) x = seq_take i x @+ [x@!i]], so this is just one
       [softmax_step] over the i-th element of [x]; the result state is
       re-indexed through that sequence equality. *)
    let xi : real = x @! i in
    seq_take_snoc x i;
    let (fxi, xst', adj) = softmax_step #(seq_take i x) xst xi in
    (* Re-index the state record from [seq_take i x @+ [xi]] to the equal
       [seq_take (i+1) x]; its [m_ok]/[d_ok] invariants transport along. *)
    let xstf : st (seq_take (i+1) x) = { m = xst'.m; d = xst'.d } in
    (fxi, xstf, adj)
#pop-options

open Kuiper.DotProd

(* Pulling a right-multiplication out of a left-fold whose binary operator
   distributes on the right: this scales both the elements and the initial
   accumulator. *)
let rec lemma_seq_fold_left_distrib_mul
  (op: real -> real -> real)
  (op_distrib_mul: (a: real) -> (b: real) -> (r: real) ->
    (a `op` b) *. r == (a *. r) `op` (b *. r))
  (acc: real) (r: real) (s: Seq.seq real)
  : Lemma
      (ensures seq_fold_left op (acc *. r) (seq_map (fun (e:real) -> e *. r) s)
               == seq_fold_left op acc s *. r)
      (decreases Seq.length s)
  = let f : real -> real = fun e -> e *. r in
    let s_mapped = seq_map f s in
    match view_seq s with
    | SNil ->
      assert (Seq.equal s Seq.empty);
      assert (Seq.equal s_mapped Seq.empty)
    | SCons hd tl ->
      assert (Seq.equal s_mapped (Seq.cons (hd *. r) (seq_map f tl)));
      calc (==) {
        seq_fold_left op (acc *. r) s_mapped;
        == { }
        seq_fold_left op (op (acc *. r) (hd *. r)) (seq_map f tl);
        == { op_distrib_mul acc hd r }
        seq_fold_left op (op acc hd *. r) (seq_map f tl);
        == { lemma_seq_fold_left_distrib_mul op op_distrib_mul (op acc hd) r tl }
        seq_fold_left op (op acc hd) tl *. r;
        == { }
        seq_fold_left op acc s *. r;
      }

#push-options "--split_queries always --z3rlimit 30"
let lem_online_softmax_adj
  (#n : pos)
  (x : lseq real n)
  (i: pos {i < n})
  (xst : st (seq_take i x))
  (k: real -> natlt (Seq.length x) -> real)  (* continuation: what to do with e^xi *)
  (#k_comm_div: (nr: real) -> (dr: real { dr =!= 0.0R } ) -> (j: natlt n) -> (k nr j) /. dr == k (nr /. dr) j)
  (op: real -> real -> real)
  (#op_distrib_mul: (a: real) -> (b: real) -> (r: real) -> (a `op` b) *. r == (a *. r) `op` (b *. r))
  (#r : real):
  Lemma
        (requires r /. xst.d == (seq_fold_left op 0.0R (seq_mapi (softmax_real_seq (seq_take i x)) k)))
        (ensures
          (let (fxi,xst',adj) = softmax_stepi #n x i xst in
          ((r *. adj) `op` (k fxi i)) /. xst'.d == (seq_fold_left op 0.0R (seq_mapi (softmax_real_seq (seq_take (i+1) x)) k)))) =
    let (fxi,xst',adj) = softmax_stepi #n x i xst in
    let x_upto_i = seq_take i x in
    let x_upto_i' = seq_take (i+1) x in
    let v1 = seq_fold_left op 0.0R (seq_mapi (softmax_real_seq x_upto_i) k) in

    (* Facts from the postcondition of softmax_stepi and the st invariant.
       Each is its own assert so it becomes a separate (small) SMT query. *)
    assert (fxi /. xst'.d == softmax_real_seq x_upto_i' @! i);
    assert (adj == exp (xst.m -. xst'.m));
    assert (exp (xst.m -. xst'.m) == exp (xst.m) /. exp (xst'.m));
    assert (adj == exp (xst.m) /. exp (xst'.m));
    assert (xst.d == exp_sum x_upto_i /. exp xst.m);
    assert (xst'.d == exp_sum x_upto_i' /. exp xst'.m);
    assert (xst'.d =!= 0.0R);

    (* ---- Step A: simplify (k fxi i) /. xst'.d ---- *)
    k_comm_div fxi xst'.d i;
    assert ((k fxi i) /. xst'.d == k (fxi /. xst'.d) i);
    assert ((k fxi i) /. xst'.d == k (softmax_real_seq x_upto_i' @! i) i);

    (* ---- Step B: simplify r *. adj /. xst'.d ----
       Goal: r *. adj /. xst'.d == v1 *. (exp_sum x_upto_i /. exp_sum x_upto_i').
       Pull the division inside, cancel exp xst'.m, then use the precondition
       to replace r with v1 *. xst.d and cancel exp xst.m. *)
    assoc_mul_div r adj xst'.d;
    assert (r *. adj /. xst'.d == r *. (adj /. xst'.d));
    assert (adj /. xst'.d
            == exp (xst.m) /. exp (xst'.m) /. (exp_sum x_upto_i' /. exp xst'.m));
    cancel_ddd (exp xst.m) (exp_sum x_upto_i') (exp xst'.m);
    assert (adj /. xst'.d == exp (xst.m) /. exp_sum x_upto_i');
    assert (r *. adj /. xst'.d == r *. (exp (xst.m) /. exp_sum x_upto_i'));

    (* Use the precondition r /. xst.d == v1, i.e. r == v1 *. xst.d. *)
    cancel_dm r xst.d;
    assert (r == v1 *. xst.d);
    assert (r *. (exp (xst.m) /. exp_sum x_upto_i')
            == (v1 *. xst.d) *. (exp (xst.m) /. exp_sum x_upto_i'));
    assoc_mul v1 xst.d (exp xst.m /. exp_sum x_upto_i');
    assert ((v1 *. xst.d) *. (exp (xst.m) /. exp_sum x_upto_i')
            == v1 *. (xst.d *. (exp xst.m /. exp_sum x_upto_i')));

    assert (xst.d *. (exp xst.m /. exp_sum x_upto_i')
            == (exp_sum x_upto_i /. exp xst.m) *. (exp xst.m /. exp_sum x_upto_i'));
    assoc_mul_div (exp_sum x_upto_i /. exp xst.m) (exp xst.m) (exp_sum x_upto_i');
    assert ((exp_sum x_upto_i /. exp xst.m) *. (exp xst.m /. exp_sum x_upto_i')
            == ((exp_sum x_upto_i /. exp xst.m) *. exp xst.m) /. exp_sum x_upto_i');
    cancel_dm (exp_sum x_upto_i) (exp xst.m);
    assert ((exp_sum x_upto_i /. exp xst.m) *. exp xst.m == exp_sum x_upto_i);
    assert (xst.d *. (exp xst.m /. exp_sum x_upto_i')
            == exp_sum x_upto_i /. exp_sum x_upto_i');

    let r2 : real = exp_sum x_upto_i /. exp_sum x_upto_i' in
    assert (r *. adj /. xst'.d == v1 *. r2);

    (* ---- Step C: rewrite the RHS fold using fold-distributivity ---- *)
    let s1 = seq_mapi (softmax_real_seq x_upto_i) k in
    let s2 = seq_mapi (softmax_real_seq x_upto_i') k in
    let f_r2 : real -> real = fun e -> e *. r2 in
    let s1_scaled = seq_map f_r2 s1 in
    let s2_prefix = seq_take i s2 in

    introduce forall (j: nat {j < i}). s2_prefix @! j == s1_scaled @! j
    with begin
      let xj : real = x @! j in
      assert (x_upto_i @! j == xj);
      assert (x_upto_i' @! j == xj);
      assert (softmax_real_seq x_upto_i @! j == exp xj /. exp_sum x_upto_i);
      assert (softmax_real_seq x_upto_i' @! j == exp xj /. exp_sum x_upto_i');
      assert (s2_prefix @! j == s2 @! j);
      assert (s2 @! j == k (exp xj /. exp_sum x_upto_i') j);
      assert (s1_scaled @! j == k (exp xj /. exp_sum x_upto_i) j *. r2);
      k_comm_div (exp xj) (exp_sum x_upto_i) j;
      k_comm_div (exp xj) (exp_sum x_upto_i') j;
      (* Both sides reduce to (k (exp xj) j) /. exp_sum x_upto_i'. *)
      assert (s2 @! j == k (exp xj) j /. exp_sum x_upto_i');
      assert (s1_scaled @! j == (k (exp xj) j /. exp_sum x_upto_i) *. r2);
      assoc_mul_div (k (exp xj) j /. exp_sum x_upto_i) (exp_sum x_upto_i) (exp_sum x_upto_i');
      cancel_dm (k (exp xj) j) (exp_sum x_upto_i);
      assert ((k (exp xj) j /. exp_sum x_upto_i) *. r2 == k (exp xj) j /. exp_sum x_upto_i')
    end;

    assert (Seq.equal s2_prefix s1_scaled);

    (* Lift the rescaling out of the fold over s1. The init=0 is crucial:
       0.0R *. r2 == 0.0R, so the helper's rescaled initial accumulator
       matches our initial 0.0R. *)
    lemma_seq_fold_left_distrib_mul op op_distrib_mul 0.0R r2 s1;
    assert (0.0R *. r2 == 0.0R);
    assert (seq_fold_left op 0.0R s1_scaled == v1 *. r2);
    assert (seq_fold_left op 0.0R s2_prefix == v1 *. r2);

    (* Split off the last element of s2. *)
    lemma_seq_fold_left_slice 0.0R op s2 0 i;
    assert (Seq.equal (Seq.slice s2 0 (i+1)) s2);
    assert (Seq.equal (Seq.slice s2 0 i) s2_prefix);
    assert (s2 @! i == k (softmax_real_seq x_upto_i' @! i) i);
    assert (seq_fold_left op 0.0R s2
            == (v1 *. r2) `op` (k (softmax_real_seq x_upto_i' @! i) i));

    (* ---- Step D: combine via op_distrib_mul ----
       (a `op` b) /. d  ==  (a /. d) `op` (b /. d)
       Done in atomic steps so the LSP can't get stuck on any single rewrite. *)
    mul_one_div ((r *. adj) `op` (k fxi i)) xst'.d;
    assert (((r *. adj) `op` (k fxi i)) /. xst'.d
            == ((r *. adj) `op` (k fxi i)) *. (1.0R /. xst'.d));
    op_distrib_mul (r *. adj) (k fxi i) (1.0R /. xst'.d);
    assert (((r *. adj) `op` (k fxi i)) *. (1.0R /. xst'.d)
            == ((r *. adj) *. (1.0R /. xst'.d)) `op` ((k fxi i) *. (1.0R /. xst'.d)));
    mul_one_div (r *. adj) xst'.d;
    mul_one_div (k fxi i) xst'.d;
    assert ((r *. adj) *. (1.0R /. xst'.d) == (r *. adj) /. xst'.d);
    assert ((k fxi i) *. (1.0R /. xst'.d) == (k fxi i) /. xst'.d);
    assert (((r *. adj) `op` (k fxi i)) /. xst'.d
            == ((r *. adj) /. xst'.d) `op` ((k fxi i) /. xst'.d));

    (* Plug in the results of Steps A and B. *)
    assert (((r *. adj) /. xst'.d) `op` ((k fxi i) /. xst'.d)
            == (v1 *. r2) `op` (k (softmax_real_seq x_upto_i' @! i) i));
    assert (((r *. adj) `op` (k fxi i)) /. xst'.d
            == (v1 *. r2) `op` (k (softmax_real_seq x_upto_i' @! i) i));

    (* Conclude using Step C. *)
    assert (((r *. adj) `op` (k fxi i)) /. xst'.d == seq_fold_left op 0.0R s2)
#pop-options

(*

previous version which also goes through, but fails in LSP; maybe cleaner style?



#push-options "--split_queries always --z3rlimit 30"
let lem_online_softmax_adj
  (#n : pos)
  (x : lseq real n)
  (i: pos {i < n})
  (xst : st (seq_take i x))
  (k: real -> natlt (Seq.length x) -> real)  (* continuation: what to do with e^xi *)
  (#k_comm_div: (nr: real) -> (dr: real { dr =!= 0.0R } ) -> (j: natlt n) -> (k nr j) /. dr == k (nr /. dr) j)
  (op: real -> real -> real)
  (#op_distrib_mul: (a: real) -> (b: real) -> (r: real) -> (a `op` b) *. r == (a *. r) `op` (b *. r))
  (#r : real):
  Lemma
        (requires r /. xst.d == (seq_fold_left op 0.0R (seq_mapi (softmax_real_seq (seq_take i x)) k)))
        (ensures
          (let (fxi,xst',adj) = softmax_stepi #n x i xst in
          ((r *. adj) `op` (k fxi i)) /. xst'.d == (seq_fold_left op 0.0R (seq_mapi (softmax_real_seq (seq_take (i+1) x)) k)))) =

    let (fxi,xst',adj) = softmax_stepi #n x i xst in

    calc (==) {
       (k fxi i /. xst'.d);
       == { k_comm_div fxi xst'.d i }
       (k (fxi /. xst'.d) i);
       == {}
       (k (softmax_real_seq (seq_take (i+1) x) @! i) i);
    };

    assert (exp (xst.m -. xst'.m) == exp (xst.m) /. exp (xst'.m));

    let x_upto_i = seq_take i x in
    let x_upto_i' = seq_take (i+1) x in
    let v1 = (seq_fold_left op 0.0R (seq_mapi (softmax_real_seq (x_upto_i)) k)) in
    calc (==) {
      r *. adj /. xst'.d;
      == { assoc_mul_div r adj xst'.d }
      r *. (adj /. xst'.d);
      == {}
      r *. (exp (xst.m) /. exp (xst'.m) /. xst'.d);
      == {}
      r *. (exp (xst.m) /. exp (xst'.m) /. (exp_sum x_upto_i' /. (exp xst'.m)));
      == { cancel_ddd (exp xst.m) (exp_sum x_upto_i') (exp xst'.m)}
      r *. (exp (xst.m) /. (exp_sum x_upto_i'));
      == { cancel_dm r xst.d }
      v1 *. xst.d *. (exp (xst.m) /. (exp_sum x_upto_i'));
      == { assoc_mul v1 xst.d (exp (xst.m) /. (exp_sum x_upto_i')) }
      v1 *. (xst.d *. (exp (xst.m) /. (exp_sum x_upto_i')));
      == { }
      v1 *. ((exp_sum x_upto_i /. (exp xst.m)) *. (exp (xst.m) /. (exp_sum x_upto_i')));
      == { assoc_mul_div (exp_sum x_upto_i /. (exp xst.m)) (exp xst.m) (exp_sum x_upto_i') }
      v1 *. (((exp_sum x_upto_i /. (exp xst.m)) *. exp (xst.m)) /. (exp_sum x_upto_i'));
      == { cancel_dm (exp_sum x_upto_i) (exp xst.m) }
      v1 *. ((exp_sum x_upto_i) /. (exp_sum x_upto_i'));
    };

    (* Prove the "external lemma":
         (v1 *. (exp_sum x_upto_i /. exp_sum x_upto_i')) `op` (k (softmax_real_seq x_upto_i' @! i) i)
         == seq_fold_left op 0.0R (seq_mapi (softmax_real_seq x_upto_i') k)

       Strategy: split the RHS using lemma_seq_fold_left_slice into a fold over
       the first i elements (the "prefix") and the last element. Show the prefix
       equals a pointwise rescaling of seq_mapi (softmax_real_seq x_upto_i) k by
       r2 = exp_sum x_upto_i /. exp_sum x_upto_i', and lift the rescaling out of
       the fold via lemma_seq_fold_left_distrib_mul. The init=0 is crucial:
       0.0R *. r2 == 0.0R so the helper's "rescaled initial accumulator" matches
       our initial 0.0R. *)
    let r2 : real = exp_sum x_upto_i /. exp_sum x_upto_i' in
    let s1 = seq_mapi (softmax_real_seq x_upto_i) k in
    let s2 = seq_mapi (softmax_real_seq x_upto_i') k in
    let f_r2 : real -> real = fun e -> e *. r2 in
    let s1_scaled = seq_map f_r2 s1 in
    let s2_prefix = seq_take i s2 in

    introduce forall (j: nat {j < i}). s2_prefix @! j == s1_scaled @! j
    with begin
      let xj : real = x @! j in
      assert (x_upto_i @! j == xj);
      assert (x_upto_i' @! j == xj);
      assert (softmax_real_seq x_upto_i @! j == exp xj /. exp_sum x_upto_i);
      assert (softmax_real_seq x_upto_i' @! j == exp xj /. exp_sum x_upto_i');
      assert (s2_prefix @! j == s2 @! j);
      assert (s2 @! j == k (exp xj /. exp_sum x_upto_i') j);
      assert (s1_scaled @! j == k (exp xj /. exp_sum x_upto_i) j *. r2);
      k_comm_div (exp xj) (exp_sum x_upto_i) j;
      k_comm_div (exp xj) (exp_sum x_upto_i') j;
      (* Both sides reduce to (k (exp xj) j) /. exp_sum x_upto_i' *)
      assert (s2 @! j == k (exp xj) j /. exp_sum x_upto_i');
      assert (s1_scaled @! j == (k (exp xj) j /. exp_sum x_upto_i) *. r2);
      assoc_mul_div (k (exp xj) j /. exp_sum x_upto_i) (exp_sum x_upto_i) (exp_sum x_upto_i');
      cancel_dm (k (exp xj) j) (exp_sum x_upto_i);
      assert ((k (exp xj) j /. exp_sum x_upto_i) *. r2 == k (exp xj) j /. exp_sum x_upto_i')
    end;

    assert (Seq.equal s2_prefix s1_scaled);

    (* Lift the rescaling out of the fold over s1. *)
    lemma_seq_fold_left_distrib_mul op op_distrib_mul 0.0R r2 s1;
    assert (0.0R *. r2 == 0.0R);
    assert (seq_fold_left op 0.0R s1_scaled == v1 *. r2);
    assert (seq_fold_left op 0.0R s2_prefix == v1 *. r2);

    (* Split off the last element of s2. *)
    lemma_seq_fold_left_slice 0.0R op s2 0 i;
    assert (Seq.equal (Seq.slice s2 0 (i+1)) s2);
    assert (Seq.equal (Seq.slice s2 0 i) s2_prefix);
    assert (s2 @! i == k (softmax_real_seq x_upto_i' @! i) i);
    assert (seq_fold_left op 0.0R s2
            == (v1 *. r2) `op` (k (softmax_real_seq x_upto_i' @! i) i));

    calc (==) {
      ((r *. adj) `op` (k fxi i)) /. xst'.d;
      == { }
      ((r *. adj) `op` (k fxi i)) *. (1.0R /. xst'.d);
      == { op_distrib_mul (r *. adj) (k fxi i) (1.0R /. xst'.d) }
      ((r *. adj) *. (1.0R /. xst'.d)) `op` (k fxi i *. (1.0R /. xst'.d));
      == { mul_one_div (r *. adj) xst'.d; mul_one_div (k fxi i) xst'.d }
      (r *. adj /. xst'.d) `op` ((k fxi i) /. xst'.d);
      == { }
      (v1 *. ((exp_sum x_upto_i) /. (exp_sum x_upto_i'))) `op` (k (softmax_real_seq (seq_take (i+1) x) @! i) i);
      == { }
      seq_fold_left op 0.0R (seq_mapi (softmax_real_seq (x_upto_i')) k);
    }
#pop-options

*)

unfold
let dp_k (#n: pos) (y: lseq real n) (fxi: real) (i: natlt n) = fxi *. (y @! i)

let dp_k_comm_div (#n: pos) (y: lseq real n) (nr: real) (dr: real { dr =!= 0.0R } ) (j: natlt n): (((dp_k #n y) nr j) /. dr == (dp_k #n y) (nr /. dr) j) =
  let k = (dp_k #n y) in
  calc (==) {
    (k nr j) /. dr;
    == {}
    nr *. (y @! j) /. dr;
    == {}
    (y @! j) *. nr /. dr;
    == {assoc_mul_div (y @! j) nr dr}
    (y @! j) *. (nr /. dr);
    == {}
    (nr /. dr) *. (y @! j);
    == {}
    (k (nr /. dr) j);
  }

unfold
let smx_dotprod_fold (#n: pos) (x y: lseq real n) (i: pos{i <= n}) =
    seq_fold_left (+.) 0.0R (seq_mapi (softmax_real_seq (seq_take i x)) (dp_k #n y))

// #set-options "--z3version 4.15.5"
#set-options "--z3seed 1"
let rec lem_smx_dotprod_is_dotprod' (#n: pos) (x y: lseq real n) (i: natle n):
  Lemma (ensures seq_fold_left (+.) 0.0R (seq_take i (seq_mapi (softmax_real_seq x) (dp_k #n y))) == seq_dotprod' (softmax_real_seq x) y i)
    (decreases i) =
  if i <= 0 then ()
  else (
    assert (Seq.length (softmax_real_seq x) == n);
    let k = dp_k #n y in
    let l = seq_mapi (softmax_real_seq x) k in
    assert (Seq.length l == n);
    calc (==) {
      seq_fold_left (+.) 0.0R (seq_take i l);
      == {lemma_seq_fold_left_slice 0.0R (+.) l 0 (i-1)}
      seq_fold_left (+.) 0.0R (seq_take (i-1) l) +. (l @! (i-1));
      == {lem_smx_dotprod_is_dotprod' #n x y (i-1)}
      seq_dotprod' (softmax_real_seq x) y (i-1) +. (l @! (i-1));
      == {}
      seq_dotprod' (softmax_real_seq x) y (i-1) +. (k ((softmax_real_seq x) @! (i-1)) (i-1));
      == {}
      (seq_dotprod' (softmax_real_seq x) y (i-1)) +. ((softmax_real_seq x) @! (i-1)) *. (y @! (i-1));
      == {}
      seq_dotprod' (softmax_real_seq x) y i;
    }
  )

let lem_smx_dotprod_fold_is_dotprod_smx (#n: pos) (x y: lseq real n):
  Lemma (smx_dotprod_fold #n x y n == seq_dotprod (softmax_real_seq x) y) =
    assert (smx_dotprod_fold #n x y n == seq_fold_left (+.) 0.0R (seq_mapi (softmax_real_seq x) (dp_k #n y)));
    assert (seq_fold_left (+.) 0.0R (seq_take n (seq_mapi (softmax_real_seq x) (dp_k #n y))) == seq_fold_left (+.) 0.0R (seq_mapi (softmax_real_seq x) (dp_k #n y)));
    lem_smx_dotprod_is_dotprod' #n x y n


#push-options "--split_queries always --z3rlimit 20"
let softmax_dotprod (#n: pos) (x y: lseq real n):
  (r: real {r == seq_dotprod (softmax_real_seq x) y}) =
  let k = dp_k #n y in
  let smx_dotprod_fold = smx_dotprod_fold #n x y in
  let rec aux (#i: pos{i <= n}) (xst : st (seq_take i x))
    (r : real {r /. xst.d == smx_dotprod_fold i})
    : Tot (r' : real {r' == smx_dotprod_fold n})
          (decreases n-i) =
    if i = n then (
      r /. xst.d
    ) else (
      let (fxi,xst',adj) = softmax_stepi #n x i xst in
      let r' = r *. adj +. fxi *. (y @! i) in
      assert (r /. xst.d == (seq_fold_left (+.) 0.0R (seq_mapi (softmax_real_seq (seq_take i x)) k)));
      lem_online_softmax_adj #n x i xst k #(dp_k_comm_div #n y) (+.) #l_distr_r #r;
      aux #(i+1) xst' r'
    ) in
  assert (seq_take 1 x) @! 0 == x @! 0;
  let xst : st (seq_take 1 x) = {
    m = x @! 0;
    d = 1.0R;
    m_ok = (
      assert (seq_max (seq![x @! 0]) == x @! 0);
      ()
    );
    d_ok = (
      assert (exp_sum (seq![x @! 0]) == exp (x @! 0));
      assert (exp (x @! 0) /. exp (x @! 0) == 1.0R);
      ()
    );
  } in
  let r : r:real{(r /. xst.d) == smx_dotprod_fold 1} = (
    assert (exp (x @! 0) /. exp (x @! 0) == 1.0R);
    assert (1.0R *. (y @! 0) == (y @! 0));
    assert ((y @! 0) /. 1.0R == 0.0R +. (y @! 0));
    (y @! 0)
  ) in
  lem_smx_dotprod_fold_is_dotprod_smx #n x y;
  aux #1 xst r
#pop-options
