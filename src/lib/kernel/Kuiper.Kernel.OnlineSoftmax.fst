module Kuiper.Kernel.OnlineSoftmax

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
module SZ = FStar.SizeT
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Softmax
open Kuiper.Seq.Common { op_At_Bang }
open Kuiper.Bijection { ( =~ ) }
module CH = Kuiper.Chest

(* Bijection between the abstract 1-D tensor index [(k, ())] and a plain
   [natlt len], used to (un)reindex a forevery over tensor cells. *)
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

let chest1_approx_intro
  (#et : Type0) {| scalar et, real_like et |} (#n : nat)
  (c1 : chest1 et n) (c2 : chest1 real n)
  : Lemma (requires forall (bid:natlt n). acc1 c1 bid %~ acc1 c2 bid)
          (ensures c1 %~ c2)
  = introduce forall (i:abs (n @| INil)). acc c1 i %~ acc c2 i
    with (let (b0, ()) = i in ())

(* Proofs *)

let online_softmax_real_iter (md: erased (tuple2 real real)) (x:real) : erased (tuple2 real real) =
  let (m,d) = md in
  let m' = rmax m x in
  let d' = d *. (exp (m -. m')) +. exp (x -. m') in
  (m',d')

let lem_online_softmax_real_iter'
  (md: erased (tuple2 real real))
  (x : real)
  : Lemma (requires snd md >. 0.0R)
          (ensures snd (online_softmax_real_iter md x) >. 0.0R)
  = ()

let rec lem_online_softmax_real_iter
  (init : erased (tuple2 real real))
  (s : seq real)
  : Lemma (requires snd init >. 0.0R)
          (ensures  snd (seq_fold_left #real #(erased (real & real)) online_softmax_real_iter init s) >. 0.0R)
          (decreases Seq.length s)
          [SMTPat (seq_fold_left #real #(erased (real & real)) online_softmax_real_iter init s)]
  = match view_seq s with
    | SNil -> ()
    | SCons hd tl -> lem_online_softmax_real_iter (online_softmax_real_iter init hd) tl

(* Real-value specification for online softmax, closer to the actual implementation. *)
let online_softmax_real (s:Seq.seq real { Seq.length s > 0 }) : GTot (seq real) =
  let x = Seq.index s 0 in
  let (m, (d : real)) = reveal (seq_fold_left online_softmax_real_iter (hide (x, 1.0R)) (seq_drop 1 s)) in
  seq_map (fun x -> exp (x -. m) /. d) s

let rsum_cons (x: real) (t: Seq.seq real)
  : Lemma (rsum (Seq.cons x t) == x +. rsum t)
  = assert (Seq.equal (Seq.cons x t) (Seq.append (Seq.create 1 x) t));
    rsum_append (Seq.create 1 x) t

let seq_map_cons (#a #b: Type) (f: a -> b) (x: a) (t: Seq.seq a)
  : Lemma (seq_map f (Seq.cons x t) == Seq.cons (f x) (seq_map f t))
  = assert (Seq.equal (seq_map f (Seq.cons x t)) (Seq.cons (f x) (seq_map f t)))

let rsum_map_cons (f: real -> real) (x: real) (t: Seq.seq real)
  : Lemma (rsum (seq_map f (Seq.cons x t)) == f x +. rsum (seq_map f t))
  = seq_map_cons f x t;
    rsum_cons (f x) (seq_map f t)

(* Theorem 1 from the Online Softmax paper (unshifted invariant form):
   The online normalizer computation correctly tracks the running max and,
   when multiplied by exp(max), equals the sum of exponentials. *)
let rec fold_correct (m0: real) (d0: real) (s: Seq.seq real)
  : Lemma (ensures (
      let r = reveal (seq_fold_left online_softmax_real_iter (hide (m0, d0)) s) in
      fst r == seq_fold_left rmax m0 s /\
      snd r *. exp (fst r) == d0 *. exp m0 +. rsum (seq_map exp s)))
    (decreases Seq.length s)
  = exp_base ();
    match view_seq s with
    | SNil -> ()
    | SCons x t ->
        let m1 = rmax m0 x in
        let d1 = d0 *. exp (m0 -. m1) +. exp (x -. m1) in
        // d1 * exp(m1) = (d0 * exp(m0-m1) + exp(x-m1)) * exp(m1)
        //              = d0 * exp(m0-m1) * exp(m1) + exp(x-m1) * exp(m1)
        //              = d0 * exp(m0) + exp(x)
        assert (d1 *. exp m1 == d0 *. exp m0 +. exp x);
        fold_correct m1 d1 t;
        rsum_map_cons exp x t;
        // IH gives: snd(fold t) * exp(fst(fold t)) == d1 * exp(m1) + rsum(map exp t)
        //         = d0 * exp(m0) + exp(x) + rsum(map exp t)
        //         = d0 * exp(m0) + rsum(map exp (cons x t))
        ()

let pointwise_eq (xi m d summ : real)
  : Lemma (requires d *. exp m == summ /\ d >. 0.0R)
          (ensures exp (xi -. m) /. d == exp xi /. summ)
  = assert (exp (xi -. m) == exp xi /. exp m);
    assert (exp (xi -. m) /. d == (exp xi /. exp m) /. d);
    assert ((exp xi /. exp m) /. d == exp xi /. (exp m *. d));
    assert (exp m *. d == d *. exp m);
    ()

let online_softmax_is_softmax (s: Seq.seq real{Seq.length s > 0}) :
  Lemma (online_softmax_real s == softmax_real_seq s)
  = exp_base ();
    let x0 = s @! 0 in
    let tl = seq_drop 1 s in
    fold_correct x0 1.0R tl;
    assert (Seq.equal s (Seq.cons x0 tl));
    rsum_map_cons exp x0 tl;
    let init : erased (tuple2 real real) = hide (x0, 1.0R) in
    assert (snd (reveal init) == 1.0R);
    let fold_result = seq_fold_left online_softmax_real_iter init tl in
    let (m, (d : real)) = reveal fold_result in
    let summ : real = rsum (seq_map exp s) in
    assert (d *. exp m == summ);
    let aux (idx: natlt (Seq.length s)) : Lemma (exp ((s @! idx) -. m) /. d == exp (s @! idx) /. summ)
      = pointwise_eq (s @! idx) m d summ
    in
    Classical.forall_intro aux;
    assert (Seq.equal (online_softmax_real s) (softmax_real_seq s))

(* END Proofs *)

unfold
let kpre
  (#et : Type0) {| floating et, real_like et |}
  (lenab : szp{ lenab <= max_blocks * max_threads})
  (#l : layout1 lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (chest1 et lenab))
  (tid : natlt lenab)
  : slprop
= (a |-> Frac (1 /. lenab) va) **
  (exists* (v: et). Cell b (idx1 tid) |-> v)

unfold
let kpost
  (#et : Type0) {| floating et, real_like et |}
  (lenab : szp{ lenab <= max_blocks * max_threads})
  (#l : layout1 lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (chest1 et lenab))
  (ra : erased (chest1 real lenab) { va %~ ra })
  (tid : natlt lenab)
  : slprop
= (a |-> Frac (1 /. lenab) va) **
  (exists* (v': et). Cell b (idx1 tid) |->
    v' ** pure (v' %~ acc1 (softmax_real ra) tid))

let lemma_seq_fold_left_slice' (#a #b:Type) (e:b) (f: b -> a -> b)
  (s : seq a) (i j : nat)
  : Lemma (ensures i <= j /\ j < len s ==>
                     seq_fold_left f e (Seq.slice s i (j + 1)) == seq_fold_left f e (Seq.slice s i j) `f` (s @! j))
  = ()

inline_for_extraction noextract
fn kfonline_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : layout1 lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (chest1 et lenab))
  (ra : erased (chest1 real lenab) { va %~ ra })
  (#_: squash (chest_forallb not_nan va))
  (tid : szlt lenab)
  ()
  preserves
    gpu
  requires
    kpre #et lenab #l a b #va tid
  ensures
    kpost #et lenab #l a b #va ra tid
{
  exp_base ();

  let ras : erased (lseq real lenab) = hide (CH.chest1_to_seq ra);

  let mut i = 0sz;
  let mut sum: et = zero;
  let mut max: et = neg infinity;
  let mut gsum : erased real = 0.0R;
  let mut gmax : erased real = ras @! 0;
  while (!i <^ lenab)
    invariant live i **
      live max ** live gmax **
      live sum ** live gsum
    invariant pure (!sum %~ !gsum)
    invariant pure (!i > 0 ==> !max %~ !gmax)
    invariant pure (!i > 0 ==> !i <= Seq.length ras /\
      (reveal !gmax, reveal !gsum) ==
        seq_fold_left online_softmax_real_iter
        (hide (ras @! 0, 1.0R)) (Seq.slice (reveal ras) 1 (!i)))
    invariant pure (!i == 0sz ==>
      (!sum == zero /\ !gsum == 0.0R /\ !max == neg infinity /\ !gmax == ras @! 0))
    decreases (lenab - !i) {

    let vk = !i;
    let x = tensor_read a ((vk <: szlt lenab), ());
    let gx = ras @! vk;
    assert pure (x %~ gx);

    let old_sum = !gsum;
    let old_max = !gmax;

    let max' = fmax !max x;
    let gmax' : erased real = rmax (reveal !gmax) gx;
    assert pure (max' %~ gmax');

    let y1 = fexp (!max `sub` max');
    let gy1 = exp (reveal !gmax -. reveal gmax');
    assert pure (!gsum == 0.0R \/ y1 %~ gy1);

    let y2 = fexp (x `sub` max');
    let gy2 = exp (gx -. reveal gmax');
    assert pure (y2 %~ gy2);

    (* At this point, we cannot prove y1 is finite. It may not be,
       in the first iteration, since !max was -infinity
       and !max - max' would underflow and return -INFINITY.
       But, exp(-INFINITY) is define to be zero, so we should be good
       in that case too. TODO: extend the scalar (or floating) class
       with a notion of the infinities that allows to prove this. *)
    assume pure (is_finite y1);
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

    assert pure (reveal gmax' == rmax (reveal old_max) gx);
    assert pure (reveal gsum' == reveal old_sum *. (exp (reveal old_max -. reveal gmax'))  +.  exp (gx -. reveal gmax'));
    if (!i = 1sz) {
      assert pure (gx == (ras @! 0));
      assert pure (!gmax == (ras @! 0));
      assert pure (old_sum == 0.0R);
      assert pure (!gsum == gy2);
      assert pure (gy2 == exp 0.0R);
      assert pure (gy2 == 1.0R);
      assert pure (!gsum == 1.0R);
      assert pure (seq_fold_left online_softmax_real_iter (hide (Seq.index ras 0, 1.0R)) (Seq.slice (reveal ras) 1 1)
                   ==
                   (hide (Seq.index ras 0, 1.0R)));
      ();
    } else {
      assert pure ((reveal old_max, reveal old_sum) == seq_fold_left online_softmax_real_iter (hide (Seq.index ras 0, 1.0R)) (Seq.slice (reveal ras) 1 (!i - 1)));
      assert pure ((reveal gmax', reveal gsum') == online_softmax_real_iter (reveal old_max, reveal old_sum) gx);
      lemma_seq_fold_left_slice' (hide (ras @! 0, 1.0R)) online_softmax_real_iter (reveal ras) 1 (!i);
      assert pure ((reveal gmax', reveal gsum') == seq_fold_left online_softmax_real_iter (hide (Seq.index ras 0, 1.0R)) (Seq.slice (reveal ras) 1 (!i)));
    };
  };
  let x = tensor_read a ((tid <: szlt lenab), ());
  let y = (fexp (x `sub` !max) `div` !sum);
  assert pure (y %~ ((online_softmax_real (reveal ras)) @! tid));
  online_softmax_is_softmax (reveal ras);
  lem_softmax_real_to_seq ra;
  assert pure (y %~ acc1 (softmax_real ra) tid);
  tensor_write_cell b ((tid <: szlt lenab), ()) y;
  ()
}

ghost
fn collect_approx_chest
  (#et : Type0) {| floating et, real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : layout1 lenab) {| ctlayout l |}
  (b : array1 et l)
  (target : chest1 real lenab)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (bid : natlt lenab).
      exists* (v': et). Cell b (idx1 bid) |-> v' ** pure (v' %~ acc1 target bid))
  ensures
    (exists* (vb' : chest1 et lenab).
      b |-> vb' ** pure (vb' %~ target))
{
  let fa = forevery_exists (fun (bid:natlt lenab) (v: et) ->
                              Cell b (idx1 bid) |-> v ** pure (v %~ acc1 target bid));
  let vb' : chest1 et lenab = mk1 (fun (bid:natlt lenab) -> fa bid);
  forevery_extract_pure
    (fun (bid:natlt lenab) -> Cell b (idx1 bid) |-> fa bid ** pure (fa bid %~ acc1 target bid))
    (fun (bid:natlt lenab) -> acc1 vb' bid %~ acc1 target bid)
    fn bid { };
  forevery_map
    (fun (bid:natlt lenab) -> Cell b (idx1 bid) |-> fa bid ** pure (fa bid %~ acc1 target bid))
    (fun (bid:natlt lenab) -> Cell b (idx1 bid) |-> (acc1 vb' bid))
    fn x { };
  forevery_ext
    (fun (bid : natlt lenab) -> Cell b (idx1 bid) |-> (acc1 vb' bid))
    (fun (y : natlt lenab) -> Cell b (abs_bij.gg y) |-> (acc vb' (abs_bij.gg y)));
  forevery_iso_back (abs_bij #lenab)
    (fun (i : abs (lenab @| INil)) -> Cell b i |-> (acc vb' i));
  tensor_implode b;
  chest1_approx_intro vb' target;
  ()
}

ghost
fn setup
  (#et : Type0) {| floating et, real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : layout1 lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (chest1 et lenab))
  (ra : erased (chest1 real lenab) { va %~ ra })
  ()
  norewrite
  requires
    (a |-> va) ** (exists* (vb : chest1 et lenab). b |-> vb)
  ensures
    (forall+ (bid : natlt lenab).
      kpre #et lenab #l a b #va bid) **
    pure (SZ.fits (tlayout_ulen l))
{
  with vb. assert b |-> vb;
  tensor_pts_to_ref b;
  tensor_explode b;
  forevery_iso (abs_bij #lenab)
    (fun (i : abs (lenab @| INil)) -> Cell b i |-> (acc vb i));
  forevery_ext
    (fun (y : natlt lenab) -> Cell b (abs_bij.gg y) |-> (acc vb (abs_bij.gg y)))
    (fun (bid : natlt lenab) -> Cell b (idx1 bid) |-> (acc1 vb bid));
  forevery_map
    (fun (bid:natlt lenab) -> Cell b (idx1 bid) |-> (acc1 vb bid))
    (fun (bid:natlt lenab) -> (exists* (v: et). Cell b (idx1 bid) |-> v))
    fn x { () };
  tensor_share_n a lenab;
  forevery_zip
    (fun (bid:natlt lenab) -> (a |-> Frac (1 /. lenab) va))
    (fun (bid:natlt lenab) -> (exists* (v: et). Cell b (idx1 bid) |-> v))
}

ghost
fn teardown
  (#et : Type0) {| floating et, real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : layout1 lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (chest1 et lenab))
  (ra : erased (chest1 real lenab) { va %~ ra })
  ()
  norewrite
  requires
    (forall+ (bid : natlt lenab).
      kpost #et lenab #l a b #va ra bid) **
    pure (SZ.fits (tlayout_ulen l))
  ensures
    (a |-> va) ** (exists* (vb' : chest1 et lenab).
      b |-> vb' **
      pure (vb' %~ softmax_real ra))
{
  forevery_unzip _ _;
  tensor_gather_n a lenab;
  collect_approx_chest b (softmax_real ra);
}

inline_for_extraction noextract
let konline_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : layout1 lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l { is_global a })
  (b : array1 et l { is_global b })
  (#va : erased (chest1 et lenab))
  (ra : erased (chest1 real lenab) { va %~ ra })
  (#_: squash (chest_forallb not_nan va))
  : kernel_desc
      (requires a |-> va ** (exists* (vb : chest1 et lenab). b |-> vb))
      (ensures  a |-> va ** (exists* (vb' : chest1 et lenab).
        b |-> vb' **
        pure (vb' %~ softmax_real ra)))
= {
    nthr = lenab;
    f = kfonline_softmax a b ra;

    frame    = pure (SZ.fits (tlayout_ulen l));
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
  (#l : layout1 lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l { is_global a })
  (b : array1 et l { is_global b })
  (#va : chest1 et lenab)
  (ra : chest1 real lenab { va %~ ra })
  (#_: squash (chest_forallb not_nan va))
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    exists* (vb : chest1 et lenab). on gpu_loc (b |-> vb)
  ensures
    exists* (vb' : chest1 et lenab).
      on gpu_loc (b |-> vb') **
      pure (vb' %~ softmax_real ra)
{
  launch_sync (konline_softmax a b #va ra);
  ()
}
