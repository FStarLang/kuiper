module Kuiper.Kernel.OnlineSoftmax

#lang-pulse
open Kuiper 
open Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec
module SZ = FStar.SizeT
open Kuiper.Array1
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

module SMX = Kuiper.Kernel.Softmax

let max_real (x: real) (y: real) : real = 
  if x >. y then x else y


// TODO: refactor all these real lemma proofs into a separate file (in src/lib/spec)

private let rsum_cons (x: real) (t: Seq.seq real)
  : Lemma (rsum (Seq.cons x t) == x +. rsum t)
  = assert (Seq.equal (Seq.cons x t) (Seq.append (Seq.create 1 x) t));
    rsum_append (Seq.create 1 x) t

private let seq_map_cons (#a #b: Type) (f: a -> b) (x: a) (t: Seq.seq a)
  : Lemma (seq_map f (Seq.cons x t) == Seq.cons (f x) (seq_map f t))
  = assert (Seq.equal (seq_map f (Seq.cons x t)) (Seq.cons (f x) (seq_map f t)))

private let rsum_map_cons (f: real -> real) (x: real) (t: Seq.seq real)
  : Lemma (rsum (seq_map f (Seq.cons x t)) == f x +. rsum (seq_map f t))
  = seq_map_cons f x t;
    rsum_cons (f x) (seq_map f t)

(* Theorem 1 from the Online Softmax paper (unshifted invariant form):
   The online normalizer computation correctly tracks the running max and,
   when multiplied by exp(max), equals the sum of exponentials. *)
private let rec fold_correct (m0: real) (d0: real) (s: Seq.seq real)
  : Lemma (ensures (
      let r = reveal (seq_fold_left online_softmax_real_iter (hide (m0, d0)) s) in
      fst r == seq_fold_left max_real m0 s /\
      snd r *. rexp (fst r) == d0 *. rexp m0 +. rsum (seq_map rexp s)))
    (decreases Seq.length s)
  = rexp_base ();
    match view_seq s with
    | SNil -> ()
    | SCons x t ->
        let m1 = max_real m0 x in
        let d1 = d0 *. rexp (m0 -. m1) +. rexp (x -. m1) in
        // d1 * exp(m1) = (d0 * exp(m0-m1) + exp(x-m1)) * exp(m1)
        //              = d0 * exp(m0-m1) * exp(m1) + exp(x-m1) * exp(m1)
        //              = d0 * exp(m0) + exp(x)
        assert (d1 *. rexp m1 == d0 *. rexp m0 +. rexp x);
        fold_correct m1 d1 t;
        rsum_map_cons rexp x t;
        // IH gives: snd(fold t) * exp(fst(fold t)) == d1 * exp(m1) + rsum(map rexp t)
        //         = d0 * exp(m0) + exp(x) + rsum(map rexp t)
        //         = d0 * exp(m0) + rsum(map rexp (cons x t))
        ()

private let pointwise_eq (xi m d summ : real)
  : Lemma (requires d *. rexp m == summ /\ d >. 0.0R)
          (ensures rexp (xi -. m) /. d == rexp xi /. summ)
  = assert (rexp (xi -. m) == rexp xi /. rexp m);
    assert (rexp (xi -. m) /. d == (rexp xi /. rexp m) /. d);
    assert ((rexp xi /. rexp m) /. d == rexp xi /. (rexp m *. d));
    assert (rexp m *. d == d *. rexp m);
    ()

let real_online_softmax_lemma (s: Seq.seq real{Seq.length s > 0}) : 
  Lemma (online_softmax_real s == SMX.softmax_real s)
  = rexp_base ();
    let x0 = s @! 0 in
    let tl = seq_drop 1 s in
    fold_correct x0 1.0R tl;
    assert (Seq.equal s (Seq.cons x0 tl));
    rsum_map_cons rexp x0 tl;
    let init : erased (tuple2 real real) = hide (x0, 1.0R) in
    assert (snd (reveal init) == 1.0R);
    let fold_result = seq_fold_left online_softmax_real_iter init tl in
    let (m, (d : real)) = reveal fold_result in
    let summ : real = rsum (seq_map rexp s) in
    assert (d *. rexp m == summ);
    let aux (idx: natlt (Seq.length s)) : Lemma (rexp ((s @! idx) -. m) /. d == rexp (s @! idx) /. summ)
      = pointwise_eq (s @! idx) m d summ
    in
    Classical.forall_intro aux;
    assert (Seq.equal (online_softmax_real s) (SMX.softmax_real s))

unfold
let kpre
  (#et : Type0) {| floating et, real_like et |}
  (lenab : szp{ lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (lseq et lenab))
  (tid : natlt lenab)
  : slprop
= (a |-> Frac (1 /. lenab) va) **
  (exists* (v: et). Cell b tid |-> v)

unfold
let kpost
  (#et : Type0) {| floating et, real_like et |}
  (lenab : szp{ lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (tid : natlt lenab)
  : slprop
= (a |-> Frac (1 /. lenab) va) **
  (exists* (v': et). Cell b tid |-> 
    v' ** pure (v' %~ ((online_softmax_real ra) @! tid)))

let online_softmax_float_iter (#et: Type0) {| floating et |} 
  (md: tuple2 et et) (x:et) : tuple2 et et =
  let (m,d) = md in 
  let m' = if x `gt` m then x else m in
  let d' = d `mul` (exp (m `sub` m')) `add` (exp (x `sub` m')) in
  (m',d')

let tup2_approximates (#a #b:Type) (#ar #br:Type) 
  {| can_approximate a ar, can_approximate b br |}
   (x: tuple2 a b) (y: tuple2 ar br): prop = 
      (fst x) %~ (fst y) /\ (snd x) %~ (snd y)

instance tup2_can_approximate (#a #b:Type) (#ar #br:Type) 
  {| can_approximate a ar, can_approximate b br |}
  : can_approximate (tuple2 a b) (tuple2 ar br) = {
  approximates = tup2_approximates;
}

#set-options "--debug SMTFail --split_queries always"

inline_for_extraction noextract
let max_float (#et : Type0) {| floating et |} 
  (x: et) (y: et) : et = 
  if x `gt` y then x else y
 
let max_float_approximates_max_real (#et: Type0) {| floating et, real_like et |}  
  (x: et) (y: et) (xr: real) (yr: real): 
    Lemma
      (requires x %~ xr /\ y %~ yr) 
      (ensures max_float #et x y %~ max_real xr yr)
      [SMTPat (max_float x y); SMTPat (max_real xr yr);
       SMTPat (x %~ xr); SMTPat (y %~ yr);]
      = admit ()


inline_for_extraction noextract
fn kfonline_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (#_: squash ( forall (i:natlt lenab). false == minus_inf `gt` (va @! i)))
  (tid : szlt lenab)
  ()
  preserves
    gpu
  requires 
    kpre #et lenab #l a b #va tid
  ensures
    kpost #et lenab #l a b #va ra tid 
{
  rexp_base ();
  
  let mut i = 0sz;
  let mut sum: et = zero;
  let mut max: et = minus_inf;
  let mut gsum : erased real = 0.0R;
  let mut gmax : erased real = ra @! 0;
  while (!i <^ lenab)
    invariant live i ** 
      live max ** live gmax **
      live sum ** live gsum
    invariant pure (!sum %~ !gsum)
    invariant pure (!i > 0 ==> !max %~ !gmax)
    invariant pure (!i > 0 ==> !i <= Seq.length ra /\
      (reveal !gmax, reveal !gsum) == 
        seq_fold_left online_softmax_real_iter 
        (hide (ra @! 0, 1.0R)) (Seq.slice ra 1 (!i)))
    invariant pure (!i == 0sz ==> 
      (!sum == zero /\ !gsum == 0.0R /\ !max == minus_inf /\ !gmax == ra @! 0))
    decreases (lenab - !i) {

    let x = read a !i;
    let gx = ra @! !i;
    assert pure (x %~ gx);

    let old_sum = !gsum;
    let old_max = !gmax;
    
    let max' = max_float #et !max x;
    let gmax' : erased real = max_real (reveal !gmax) gx; // if (i == 0) then gx else max_real gx (reveal !gmax); // ?
    assert pure (max' %~ gmax');

    let y1 = exp (!max `sub` max');
    let gy1 = rexp (reveal !gmax -. reveal gmax');
    assert pure (!gsum == 0.0R \/ y1 %~ gy1);

    let y2 = exp (x `sub` max');
    let gy2 = rexp (gx -. reveal gmax');
    assert pure (y2 %~ gy2);

    assume pure (zero `mul` y1 == zero #et); // TODO: add to float properties (take care for nans)
    assert pure ( (!sum `mul` y1)  %~  (reveal (!gsum) *. gy1) );
    

    let sum' = !sum `mul` y1 `add` y2;
    let gsum': erased real = (reveal !gsum) *. gy1 +. gy2;
    assert pure (sum' %~ gsum');

    max := max';
    gmax := gmax';
    assert pure (!max %~ !gmax);

    sum := sum';
    gsum := gsum';
    assert pure (!sum %~ !gsum);

    i := !i `SZ.add` 1sz;

    assert pure (gmax' == max_real (reveal old_max) gx);
    assert pure (reveal gsum' == reveal old_sum *. (rexp (reveal old_max -. reveal gmax'))  +.  rexp (gx -. reveal gmax'));
    assert pure ((reveal gmax', reveal gsum') == seq_fold_left online_softmax_real_iter (hide (Seq.index ra 0, 1.0R)) (Seq.slice ra 1 (!i)));
  };
  let x = read a tid;
  let y = (exp (x `sub` !max) `div` !sum);
  write_cell b tid y;
  ()
}

ghost
fn setup
  (#et : Type0) {| floating et, real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  ()
  norewrite
  requires
    (a |-> va) ** (exists* (vb : lseq et lenab). b |-> vb)
  ensures
    (forall+ (bid : natlt lenab).
      kpre #et lenab #l a b #va bid) **
    pure (SZ.fits (layout_size l))
{
  with vb. assert b |-> vb;
  Kuiper.Array1.pts_to_ref b;
  Kuiper.Array1.explode b;
  forevery_map
    (fun (i:natlt lenab) -> Cell b i |-> (vb @! i))
    (fun (i:natlt lenab) -> (exists* (v: et). Cell b i |-> v))
    fn x { () };
  Kuiper.Array1.share_n a lenab;
  forevery_zip 
    (fun (bid:natlt lenab) -> (a |-> Frac (1 /. lenab) va))
    (fun (bid:natlt lenab) -> (exists* (v: et). Cell b bid |-> v)) 
}

ghost
fn teardown
  (#et : Type0) {| floating et, real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  ()
  norewrite
  requires
    (forall+ (bid : natlt lenab).
      kpost #et lenab #l a b #va ra bid) **
    pure (SZ.fits (layout_size l))
  ensures
    (a |-> va) ** (exists* (vb' : lseq et lenab).
      b |-> vb' **
      pure (vb' %~ online_softmax_real ra))
{
  forevery_unzip _ _; (*
    (fun (bid: natlt lenab) -> (a |-> Frac (1 /. lenab) va))
    (fun (bid: natlt lenab) -> (exists* (v': et). Cell b bid |-> 
    v' ** pure (v' %~ ((online_softmax_real ra) @! bid))));*)

  Kuiper.Array1.gather_n a lenab;
  let y = forevery_exists 
    (fun (bid: natlt lenab) (v': et) -> Cell b bid |-> 
    v' ** pure (v' %~ ((online_softmax_real ra) @! bid)));
  let vb' = Seq.init_ghost lenab (fun (bid: natlt lenab) -> y bid);
  forevery_map
    (fun (i:natlt lenab) -> Cell b i |-> y i ** pure (y i %~ ((online_softmax_real ra) @! i)))
    (fun (i:natlt lenab) -> Cell b i |-> (vb' @! i) ** pure ((vb' @! i) %~ ((online_softmax_real ra) @! i)))
    fn x { };
  forevery_extract_pure 
    (fun (i:natlt lenab) -> Cell b i |-> (vb' @! i) ** pure ((vb' @! i) %~ ((online_softmax_real ra) @! i)))
    (fun (i:natlt lenab) -> (vb' @! i) %~ ((online_softmax_real ra) @! i))
    fn x { };
  forevery_unzip _ _;
  Kuiper.Array1.implode b;
  admit(); // need a "array1_collect_approx" (no _tiled) ?
  show_proof_state;
}

inline_for_extraction noextract
let konline_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l { is_global a })
  (b : array1 et l { is_global b })
  (#va : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (#_: squash ( forall (i:natlt lenab). false == minus_inf `gt` (va @! i)))
  : kernel_desc
      (requires a |-> va ** (exists* (vb : lseq et lenab). b |-> vb))
      (ensures  a |-> va ** (exists* (vb' : lseq et lenab).
        b |-> vb' **
        pure (vb' %~ online_softmax_real ra)))
= {
    nthr = lenab;
    f = kfonline_softmax a b ra;

    frame    = pure (SZ.fits (layout_size l));
    teardown = teardown a b #va ra;
    setup    = setup a b #va ra;
    kpre =  (kpre #et lenab #l a b #va);
    kpost = (kpost #et lenab #l a b #va ra);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn online_softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l { is_global a })
  (b : array1 et l { is_global b })
  (#va : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (#_: squash ( forall (i:natlt lenab). false == minus_inf `gt` (va @! i)))
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    exists* (vb : lseq et lenab). on gpu_loc (b |-> vb)
  ensures
    exists* (vb' : lseq et lenab).
      on gpu_loc (b |-> vb') **
      pure (vb' %~ SMX.softmax_real ra)
{
  launch_sync (konline_softmax a b #va ra);
  real_online_softmax_lemma ra;
  ()
}
