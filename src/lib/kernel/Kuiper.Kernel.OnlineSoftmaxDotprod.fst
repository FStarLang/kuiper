module Kuiper.Kernel.OnlineSoftmaxDotprod

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
module SZ = FStar.SizeT
open Kuiper.Array1
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

open Kuiper.Spec.Softmax
open Kuiper.DotProd

(* Proofs *)

(* ----------------------------------------------------------------------------
   Real-valued spec.
   We define an "online softmax dot product" by mirroring the loop of the
   GPU kernel below: as we sweep through the input, we maintain the running
   max [m] of [ra] and a running accumulator [d] that, when normalised, will
   equal softmax(ra) . rb.

   The iteration step takes a pair (x, y) = (ra@!i, rb@!i) and updates
   (m, d) as follows:
     m' = max(m, x)
     d' = d * exp(m - m') + exp(x - m') * y
   When started with (ra@!0, rb@!0) and applied over the rest of the pairs,
   the final [d] is exactly softmax(ra) . rb (which is shown later, in
   [real_online_softmax_dotprod_lemma]).
   ------------------------------------------------------------------------ *)

(* Pair up two sequences of reals into a sequence of pairs. *)
let dotprod_pair_seq (#n: nat) (ra rb: lseq real n) : GTot (lseq (real & real) n) =
  Seq.init n (fun i -> (Seq.index ra i, Seq.index rb i))

let dotprod_pair_seq_index (#n: nat) (ra rb: lseq real n) (i: natlt n)
  : Lemma (Seq.index (dotprod_pair_seq ra rb) i == (Seq.index ra i, Seq.index rb i))
          [SMTPat (Seq.index (dotprod_pair_seq ra rb) i)]
  = Seq.init_index n (fun i -> (Seq.index ra i, Seq.index rb i))

(* One iteration of the online-softmax-dotprod. *)
let online_softmax_dotprod_real_iter
  (md: erased (real & real & real)) (xy : real & real) : erased (real & real & real) =
  let (m, dn, dd) = md in
  let (x, y) = xy in
  let m' = rmax m x in
  let dn' = dn *. (rexp (m -. m')) +. rexp (x -. m') *. y in
  let dd' = dd *. (rexp (m -. m')) +. rexp (x -. m') in
  (m', dn', dd')

let rec lem_online_softmax_dotprod_real_iter
  (init : erased (real & real & real))
  (s : seq (real & real))
  : Lemma (requires (reveal init)._3 >. 0.0R)
          (ensures  (reveal (seq_fold_left #(real & real) #(erased (real & real & real)) online_softmax_dotprod_real_iter init s))._3 >. 0.0R)
          (decreases Seq.length s)
          [SMTPat (seq_fold_left #(real & real) #(erased (real & real & real)) online_softmax_dotprod_real_iter init s)]
  = match view_seq s with
    | SNil -> ()
    | SCons hd tl -> lem_online_softmax_dotprod_real_iter (online_softmax_dotprod_real_iter init hd) tl

(* The full real-valued result. *)
let online_softmax_dotprod_real
  (#n: nat) (ra rb: lseq real n { n > 0 }) : GTot real
  = let pairs = dotprod_pair_seq ra rb in
    let init : erased (real & real & real) = hide (ra @! 0, rb @! 0, 1.0R) in
    let (_, dn, dd) = reveal (seq_fold_left online_softmax_dotprod_real_iter init (seq_drop 1 pairs)) in
    dn /. dd

(* ---------- Helpers, analogous to Kuiper.Kernel.OnlineSoftmax. ----------- *)

let rsum_cons (x: real) (t: Seq.seq real)
  : Lemma (rsum (Seq.cons x t) == x +. rsum t)
  = assert (Seq.equal (Seq.cons x t) (Seq.append (Seq.create 1 x) t));
    rsum_append (Seq.create 1 x) t

let seq_map_cons (#a #b: Type) (f: a -> b) (x: a) (t: Seq.seq a)
  : Lemma (seq_map f (Seq.cons x t) == Seq.cons (f x) (seq_map f t))
  = assert (Seq.equal (seq_map f (Seq.cons x t)) (Seq.cons (f x) (seq_map f t)))

let rsum_map_cons (#a: Type) (f: a -> real) (x: a) (t: Seq.seq a)
  : Lemma (rsum (seq_map f (Seq.cons x t)) == f x +. rsum (seq_map f t))
  = seq_map_cons f x t;
    rsum_cons (f x) (seq_map f t)

(* Projections used to express the invariants of [fold_correct_dotprod]
   below: [exp_x] forgets the second component (giving the denominator
   sum) and [exp_x_y] computes the weighted product (giving the
   numerator sum). *)
let exp_x   (xy: real & real) : real = rexp (fst xy)
let exp_x_y (xy: real & real) : real = rexp (fst xy) *. snd xy

(* Theorem 1 from the Online Softmax paper, generalised to dot-product.
   Both numerator and denominator accumulators, multiplied by [exp m],
   recover the "unshifted" running sums. *)
let rec fold_correct_dotprod (m0 dn0 dd0: real) (s: Seq.seq (real & real))
  : Lemma (ensures (
      let r = reveal (seq_fold_left online_softmax_dotprod_real_iter
                        (hide (m0, dn0, dd0)) s) in
      let (m', dn', dd') = r in
      dn' *. rexp m' == dn0 *. rexp m0 +. rsum (seq_map exp_x_y s) /\
      dd' *. rexp m' == dd0 *. rexp m0 +. rsum (seq_map exp_x   s)))
    (decreases Seq.length s)
  = rexp_base ();
    match view_seq s with
    | SNil -> ()
    | SCons xy t ->
        let (x, y) = xy in
        let m1 = rmax m0 x in
        let dn1 = dn0 *. rexp (m0 -. m1) +. rexp (x -. m1) *. y in
        let dd1 = dd0 *. rexp (m0 -. m1) +. rexp (x -. m1) in
        // dn1 * exp(m1) = (dn0 * exp(m0-m1) + exp(x-m1)*y) * exp(m1)
        //               = dn0 * exp(m0) + exp(x) * y
        assert ( rexp (m0 -. m1) *. rexp m1 == rexp m0 );
        assert (dn1 *. rexp m1 == (dn0 *. rexp (m0 -. m1) +. rexp (x -. m1) *. y) *. rexp m1);
        assert (dn1 *. rexp m1 == dn0 *. rexp m0 +. rexp x *. y);
        
        assert (dd1 *. rexp m1 == (dd0 *. rexp (m0 -. m1) +. rexp (x -. m1)) *. rexp m1);
        assert (dd1 *. rexp m1 == dd0 *. rexp m0 +. rexp x);
        fold_correct_dotprod m1 dn1 dd1 t;
        rsum_map_cons exp_x_y xy t;
        rsum_map_cons exp_x   xy t

(* [rsum] of an initialised sequence equals the recursive [seq_dotprod]
   of the corresponding mapped/index pair, expanded one step at a time. *)
let rec rsum_init_dotprod_eq
  (#n: nat) (ra rb: lseq real n) (k: nat{k <= n})
  : Lemma (ensures (
      let sub : lseq real k = Seq.init k (fun (i:nat{i<k}) -> rexp (ra @! i) *. (rb @! i)) in
      let mra : lseq real n = seq_map rexp ra in
      rsum sub == seq_dotprod mra rb k))
    (decreases k)
  = if k = 0 then (
      let sub : lseq real 0 = Seq.init 0 (fun (i:nat{i<0}) -> rexp (ra @! i) *. (rb @! i)) in
      assert (Seq.equal sub Seq.empty)
    ) else (
      let sub  : lseq real k     = Seq.init k     (fun (i:nat{i<k})   -> rexp (ra @! i) *. (rb @! i)) in
      let sub' : lseq real (k-1) = Seq.init (k-1) (fun (i:nat{i<k-1}) -> rexp (ra @! i) *. (rb @! i)) in
      let last = rexp (ra @! (k-1)) *. (rb @! (k-1)) in
      assert (Seq.equal sub (Seq.append sub' (Seq.create 1 last)));
      rsum_append sub' (Seq.create 1 last);
      assert (Seq.equal (Seq.create 1 last) (Seq.cons last Seq.empty));
      rsum_cons last Seq.empty;
      rsum_init_dotprod_eq ra rb (k-1);
      let mra : lseq real n = seq_map rexp ra in
      assert (mra @! (k-1) == rexp (ra @! (k-1)))
    )

(* Distribute [summ] out of [seq_dotprod] when the first argument is a
   softmax with denominator [summ = rsum (seq_map rexp ra)]. *)
let rec seq_dotprod_softmax_factor
  (#n: nat) (ra: lseq real n {n > 0}) (rb: lseq real n) (k: nat{k <= n})
  : Lemma (ensures (
            let mra : lseq real n = seq_map rexp ra in
            let summ : real = rsum mra in
            ~(summ == 0.0R) /\
            seq_dotprod (softmax_real ra) rb k *. summ == seq_dotprod mra rb k))
          (decreases k)
  = let mra : lseq real n = seq_map rexp ra in
    sum_non_zero mra 0.0R;
    let summ : real = rsum mra in
    assert (summ >. 0.0R);
    // [softmax_real ra] unfolds to [seq_map (fun x -> rexp x /. summ) ra].
    assert (Seq.equal (softmax_real ra) (seq_map (fun (x:real) -> rexp x /. summ) ra));
    if k = 0 then ()
    else seq_dotprod_softmax_factor ra rb (k-1)

(* [rsum] of [seq_map exp_x_y pairs] equals the unshifted numerator
   sum, and similarly for [exp_x] and the denominator. *)
let pairs_rsum_eq
  (#n: nat) (ra rb: lseq real n)
  : Lemma (
      let pairs = dotprod_pair_seq ra rb in
      let mra : lseq real n = seq_map rexp ra in
      rsum (seq_map exp_x_y pairs) == seq_dotprod mra rb n
      /\ rsum (seq_map exp_x pairs) == rsum mra)
  = let pairs = dotprod_pair_seq ra rb in
    let mra : lseq real n = seq_map rexp ra in
    // Pointwise equality for the numerator-side:
    let lhs_num : lseq real n =
      Seq.init n (fun (i:nat{i<n}) -> rexp (ra @! i) *. (rb @! i)) in
    assert (Seq.equal (seq_map exp_x_y pairs) lhs_num);
    rsum_init_dotprod_eq ra rb n;
    // Pointwise equality for the denominator-side:
    assert (Seq.equal (seq_map exp_x pairs) mra)

(* Small algebraic helpers on reals used by [real_online_softmax_dotprod_lemma]. *)
let real_mul_cancel (a b c: real)
  : Lemma (requires a *. c == b *. c /\ ~(c == 0.0R))
          (ensures a == b)
  = ()

let real_div_mul (a b: real)
  : Lemma (requires ~(b == 0.0R))
          (ensures (a *. b) /. b == a)
  = ()

let real_mul_assoc (a b c: real)
  : Lemma ((a *. b) *. c == a *. (b *. c))
  = ()

let real_online_softmax_dotprod_lemma
  (#n: nat) (ra rb: lseq real n { n > 0 })
  : Lemma (online_softmax_dotprod_real ra rb == seq_dotprod (softmax_real ra) rb n)
  = rexp_base ();
    let pairs = dotprod_pair_seq ra rb in
    let x0 = ra @! 0 in
    let y0 = rb @! 0 in
    let tl = seq_drop 1 pairs in
    let head_pair : real & real = (x0, y0) in
    let cons_pairs : Seq.seq (real & real) = Seq.cons head_pair tl in
    // pairs == cons (x0, y0) tl
    assert (Seq.length cons_pairs == n);
    assert (forall (i: nat{i < n}). Seq.index cons_pairs i == Seq.index pairs i);
    assert (Seq.equal pairs cons_pairs);
    assert (pairs == cons_pairs);
    rsum_map_cons exp_x_y head_pair tl;
    rsum_map_cons exp_x   head_pair tl;
    // Now apply fold_correct_dotprod to the tail:
    fold_correct_dotprod x0 y0 1.0R tl;
    let init : erased (real & real & real) = hide (x0, y0, 1.0R) in
    let r = reveal (seq_fold_left online_softmax_dotprod_real_iter init tl) in
    let (m', dn', dd') = r in
    // Combine the cons step with the tail fold:
    //   dn' * exp(m') = y0 * exp(x0) + rsum (seq_map exp_x_y tl)
    //                 = rsum (seq_map exp_x_y pairs)
    //   dd' * exp(m') = 1   * exp(x0) + rsum (seq_map exp_x   tl)
    //                 = rsum (seq_map exp_x   pairs)
    assert (exp_x_y head_pair == rexp x0 *. y0);
    assert (exp_x   head_pair == rexp x0);
    assert (dn' *. rexp m' == rsum (seq_map exp_x_y pairs));
    assert (dd' *. rexp m' == rsum (seq_map exp_x   pairs));
    // Connect to seq_dotprod and to summ:
    pairs_rsum_eq ra rb;
    let mra : lseq real n = seq_map rexp ra in
    let summ : real = rsum mra in
    // [summ > 0] since each term [rexp _] is positive and [n > 0].
    sum_non_zero mra 0.0R;
    assert (summ >. 0.0R);
    assert (~(summ == 0.0R));
    // dd' > 0 via lem_online_softmax_dotprod_real_iter (init's third comp is 1.0R > 0):
    lem_online_softmax_dotprod_real_iter init tl;
    assert (dd' >. 0.0R);
    // rexp m' > 0:
    rexp_positive m';
    assert (~(rexp m' == 0.0R));
    assert (~(dd' == 0.0R));
    // Algebraic derivation:
    //   sm * summ        == seq_dotprod mra rb n               (factor lemma)
    //   summ             == dd' *. rexp m'                     (established)
    //   seq_dotprod ...  == dn' *. rexp m'                     (established)
    // hence  sm * (dd' * rexp m')  == dn' * rexp m'
    // i.e.   (sm * dd') * rexp m'  == dn' * rexp m'
    // cancel rexp m' to get  sm * dd' == dn'
    // then divide by dd' to get  sm == dn' / dd'.
    seq_dotprod_softmax_factor ra rb n;
    let sm = seq_dotprod (softmax_real ra) rb n in
    assert (sm *. summ == seq_dotprod mra rb n);
    assert (sm *. (dd' *. rexp m') == dn' *. rexp m');
    // Associativity of [*.] on reals:
    real_mul_assoc sm dd' (rexp m');
    assert ((sm *. dd') *. rexp m' == sm *. (dd' *. rexp m'));
    assert ((sm *. dd') *. rexp m' == dn' *. rexp m');
    real_mul_cancel (sm *. dd') dn' (rexp m');
    assert (sm *. dd' == dn');
    real_div_mul sm dd';
    assert ((sm *. dd') /. dd' == sm);
    assert (online_softmax_dotprod_real ra rb == dn' /. dd');
    assert (online_softmax_dotprod_real ra rb == sm)

(* END Proofs *)

(* ----------------------------------------------------------------------------
   Per-thread pre/post conditions.
   - Each thread has read-only access to a and b (fractional permission 1/n).
   - Thread 0 alone owns the output reference r and writes the result there.
   ------------------------------------------------------------------------ *)

unfold
let kpre
  (#et : Type0) {| floating et, real_like et |}
  (lenab : szp{ lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |}
  (a : array1 et l)
  (b : array1 et l)
  (r : gpu_ref et)
  (#va : erased (lseq et lenab))
  (#vb : erased (lseq et lenab))
  (tid : natlt lenab)
  : slprop
= (a |-> Frac (1 /. lenab) va) **
  (b |-> Frac (1 /. lenab) vb) **
  if_ (op_Equality #nat tid 0) (live r)

unfold
let kpost
  (#et : Type0) {| floating et, real_like et |}
  (lenab : szp{ lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |}
  (a : array1 et l)
  (b : array1 et l)
  (r : gpu_ref et)
  (#va : erased (lseq et lenab))
  (#vb : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (rb : erased (lseq real lenab) { vb %~ rb })
  (tid : natlt lenab)
  : slprop
= (a |-> Frac (1 /. lenab) va) **
  (b |-> Frac (1 /. lenab) vb) **
  if_ (op_Equality #nat tid 0)
      (exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb))


(* ----------------------------------------------------------------------------
   Per-thread kernel function.
   ------------------------------------------------------------------------ *)

inline_for_extraction noextract
fn kfonline_softmax_dotprod
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |}
  (a : array1 et l)
  (b : array1 et l)
  (r : gpu_ref et)
  (#va : erased (lseq et lenab))
  (#vb : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (rb : erased (lseq real lenab) { vb %~ rb })
  (#_: squash ( seq_forallb not_nan va ))
  (tid : szlt lenab)
  ()
  preserves
    gpu
  requires
    kpre #et lenab #l a b r #va #vb tid
  ensures
    kpost #et lenab #l a b r #va #vb ra rb tid
{
  rexp_base ();

  let pairs : erased (lseq (real & real) lenab) = dotprod_pair_seq ra rb;

  let mut i = 0sz;
  let mut sum_n: et = zero;
  let mut sum_d: et = zero;
  let mut max: et = neg infinity;
  let mut gsum_n : erased real = 0.0R;
  let mut gsum_d : erased real = 0.0R;
  let mut gmax : erased real = ra @! 0;
  while (!i <^ lenab)
    invariant live i **
      live max ** live gmax **
      live sum_n ** live sum_d ** live gsum_n ** live gsum_d
    invariant pure (!sum_n %~ !gsum_n)
    invariant pure (!sum_d %~ !gsum_d)
    invariant pure (!i > 0 ==> !max %~ !gmax)
    invariant pure (!i > 0 ==> !i <= Seq.length ra /\
      (reveal !gmax, reveal !gsum_n, reveal !gsum_d) ==
        seq_fold_left online_softmax_dotprod_real_iter
        (hide (ra @! 0, rb @! 0, 1.0R)) (Seq.slice (reveal pairs) 1 (!i)))
    invariant pure (!i == 0sz ==>
      (!sum_n == zero /\ !sum_d == zero /\ !gsum_n == 0.0R /\ !gsum_d == 0.0R /\ !max == neg infinity /\ !gmax == ra @! 0))
    decreases (lenab - !i) {

    let x = read a !i;
    let gx = ra @! !i;
    assert pure (x %~ gx);

    let y = read b !i;
    let gy = rb @! !i;
    assert pure (y %~ gy);

    let old_sum_n = !gsum_n;
    let old_sum_d = !gsum_d;
    let old_max = !gmax;

    let max' = fmax #et !max x;
    let gmax' : erased real = rmax (reveal !gmax) gx;
    assert pure (max' %~ gmax');

    let y1 = exp (!max `sub` max');
    let gy1 = rexp (reveal !gmax -. reveal gmax');
    assert pure (!gsum_n == 0.0R \/ y1 %~ gy1);

    (* At this point, we cannot prove y1 is finite. It may not be,
       in the first iteration, since !max was -infinity
       and !max - max' would underflow and return -INFINITY.
       But, exp(-INFINITY) is define to be zero, so we should be good
       in that case too. TODO: extend the scalar (or floating) class
       with a notion of the infinities that allows to prove this. *)
    assume pure (is_finite y1);
    assert pure ( (!sum_n `mul` y1)  %~  (reveal (!gsum_n) *. gy1) );
    assert pure ( (!sum_d `mul` y1)  %~  (reveal (!gsum_d) *. gy1) );

    let y2_n = exp (x `sub` max') `mul` y;
    let gy2_n = rexp (gx -. reveal gmax') *. gy;
    assert pure (y2_n %~ gy2_n);

    let y2_d = exp (x `sub` max');
    let gy2_d = rexp (gx -. reveal gmax');
    assert pure (y2_d %~ gy2_d);

    let sum_n' = !sum_n `mul` y1 `add` y2_n;
    let gsum_n': erased real = (reveal !gsum_n) *. gy1 +. gy2_n;
    assert pure (sum_n' %~ gsum_n');

    let sum_d' = !sum_d `mul` y1 `add` y2_d;
    let gsum_d': erased real = (reveal !gsum_d) *. gy1 +. gy2_d;
    assert pure (sum_d' %~ gsum_d');

    max := max';
    gmax := gmax';
    assert pure (!max %~ !gmax);

    sum_n := sum_n';
    gsum_n := gsum_n';
    assert pure (!sum_n %~ !gsum_n);

    sum_d := sum_d';
    gsum_d := gsum_d';
    assert pure (!sum_d %~ !gsum_d);

    i := !i `SZ.add` 1sz;

    assert pure (gmax' == rmax (reveal old_max) gx);
    assert pure (reveal gsum_n' == reveal old_sum_n *. 
      (rexp (reveal old_max -. reveal gmax'))  +.  rexp (gx -. reveal gmax') *. gy);
    assert pure (reveal gsum_d' == reveal old_sum_d *. 
      (rexp (reveal old_max -. reveal gmax'))  +.  rexp (gx -. reveal gmax'));
    assert pure ((reveal gmax', reveal gsum_n', reveal gsum_d') == seq_fold_left online_softmax_dotprod_real_iter (hide (ra @! 0, rb @! 0, 1.0R)) (Seq.slice (reveal pairs) 1 (!i)));
  };

  let res = !sum_n `div` !sum_d;
  assert pure (res %~ online_softmax_dotprod_real ra rb);

  (* Thread 0 commits the result to r. *)
  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (live r);
    let s = res;
    gpu_write r s;
    if_intro_true' (op_Equality #nat tid 0)
      (exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb));
  } else {
    if_elim_false' (op_Equality #nat tid 0) (live r);
    if_intro_false' (op_Equality #nat tid 0)
      (exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb));
  }
}


(* ----------------------------------------------------------------------------
   Setup / teardown that split and re-gather the per-thread resources.
   ------------------------------------------------------------------------ *)

ghost
fn setup
  (#et : Type0) {| floating et, real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |}
  (a : array1 et l)
  (b : array1 et l)
  (r : gpu_ref et)
  (#va : erased (lseq et lenab))
  (#vb : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (rb : erased (lseq real lenab) { vb %~ rb })
  ()
  norewrite
  requires
    (a |-> va) ** (b |-> vb) ** live r
  ensures
    (forall+ (tid : natlt lenab).
      kpre #et lenab #l a b r #va #vb tid) ** emp
{
  Kuiper.Array1.share_n a lenab;
  Kuiper.Array1.share_n b lenab;
  forevery_if_intro #(natlt lenab) 0 (fun _ -> live r);
  forevery_ext
    (fun (tid:natlt lenab) -> if_ (op_Equality #(natlt lenab) tid 0) (live r))
    (fun (tid:natlt lenab) -> if_ (op_Equality #nat tid 0) (live r));
  forevery_zip3
    (fun (tid:natlt lenab) -> a |-> Frac (1 /. lenab) va)
    (fun (tid:natlt lenab) -> b |-> Frac (1 /. lenab) vb)
    (fun (tid:natlt lenab) -> if_ (op_Equality #nat tid 0) (live r));
}

ghost
fn teardown
  (#et : Type0) {| floating et, real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |}
  (a : array1 et l)
  (b : array1 et l)
  (r : gpu_ref et)
  (#va : erased (lseq et lenab))
  (#vb : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (rb : erased (lseq real lenab) { vb %~ rb })
  ()
  norewrite
  requires
    (forall+ (tid : natlt lenab).
      kpost #et lenab #l a b r #va #vb ra rb tid) ** emp
  ensures
    (a |-> va) ** (b |-> vb) **
    (exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb))
{
  forevery_unzip3
    (fun (tid:natlt lenab) -> a |-> Frac (1 /. lenab) va)
    (fun (tid:natlt lenab) -> b |-> Frac (1 /. lenab) vb)
    (fun (tid:natlt lenab) -> if_ (op_Equality #nat tid 0)
      (exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb)));
  Kuiper.Array1.gather_n a lenab;
  Kuiper.Array1.gather_n b lenab;
  forevery_ext
    (fun (tid:natlt lenab) -> if_ (op_Equality #nat tid 0)
      (exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb)))
    (fun (tid:natlt lenab) -> if_ (op_Equality #(natlt lenab) tid 0)
      (exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb)));
  forevery_if_elim #(natlt lenab) 0
    (fun _ -> exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb));
}

(* ----------------------------------------------------------------------------
   Kernel descriptor and CPU-side launcher.
   ------------------------------------------------------------------------ *)

inline_for_extraction noextract
let konline_softmax_dotprod
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (b : array1 et l { is_global b })
  (r : gpu_ref et)
  (#va : erased (lseq et lenab))
  (#vb : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (rb : erased (lseq real lenab) { vb %~ rb })
  (#_: squash (seq_forallb not_nan va))
  : kernel_desc
      (requires (a |-> va) ** (b |-> vb) ** live r)
      (ensures  (a |-> va) ** (b |-> vb) **
        (exists* (v: et). r |-> v ** pure (v %~ online_softmax_dotprod_real ra rb)))
= {
    nthr = lenab;
    f = kfonline_softmax_dotprod a b r ra rb;

    frame    = emp;
    teardown = teardown a b r #va #vb ra rb;
    setup    = setup    a b r #va #vb ra rb;
    kpre =  (kpre  #et lenab #l a b r #va #vb);
    kpost = (kpost #et lenab #l a b r #va #vb ra rb);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn online_softmax_dotprod_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (b : array1 et l { is_global b })
  (r : gpu_ref et)
  (#va : erased (lseq et lenab))
  (#vb : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (rb : erased (lseq real lenab) { vb %~ rb })
  (#_: squash (seq_forallb not_nan va))
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> va) **
    on gpu_loc (b |-> vb)
  requires
    exists* (vr : et). on gpu_loc (r |-> vr)
  ensures
    exists* (vr : et). on gpu_loc (r |-> vr) **
      pure (vr %~ seq_dotprod (softmax_real ra) rb lenab)
{
  launch_sync (konline_softmax_dotprod a b r #va #vb ra rb);
  real_online_softmax_dotprod_lemma ra rb;
  ()
}

let _test (len : szp{len <= max_blocks * max_threads}) =
  online_softmax_dotprod_gpu #f32 1024sz #len #(l1_forward len)