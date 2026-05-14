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
fn kfonline_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l)
  (b : array1 et l)
  (#va : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (#_: squash ( forall (i:natlt lenab). (va @! i) `gt` minus_inf))
  (tid : szlt lenab)
  ()
  preserves
    gpu
  requires 
    kpre #et lenab #l a b #va tid
  ensures
    kpost #et lenab #l a b #va ra tid 
{
  let mut i = 0sz;
  let mut sum: et = zero;
  let mut max: et = minus_inf;
  let mut gsum : erased real = 0.0R;
  let mut gmax : erased real = 123.0R;
  while (!i <^ lenab)
    invariant live i ** 
      live max ** live gmax **
      live sum **
      live gsum **
      pure (!sum %~ !gsum) **
      pure (!i > 0 ==> !max %~ !gmax)
    //   pure (!i > 0 ==>
    //          (!max, !sum) %~ seq_fold_left online_softmax_real_iter (hide (Seq.index ra 0, 1.0R)) (seq_drop 1 ra)) **
    //   pure (!i == 0sz ==> (!sum == zero /\ !max == minus_inf))
    decreases (lenab - !i) {
    assert pure (!i < lenab);
    let x = read a !i;
    assert pure (x %~ (ra @! !i));
    let max' = (if x `gt` !max then x else !max);
    let vi = !i;
    let vgmax : erased real = !gmax;
    let gmax' : erased real = hide (if (ra `Seq.index` vi) >. reveal vgmax then ra `Seq.index` vi else reveal #real vgmax);
    assume pure (max' %~ gmax');
    admit();
    sum := !sum `mul` (exp (max' `sub` !max)) `add` (exp (x `sub` max'));
    max := max';
    i := !i `SZ.add` 1sz;
    // admit();
    // assert pure (!max %~ fst (seq_fold_left online_softmax_real_iter (hide (Seq.index ra 0, 1.0R)) (seq_drop 1 ra)));
    // assert pure (!sum %~ snd (seq_fold_left online_softmax_real_iter (hide (Seq.index ra 0, 1.0R)) (seq_drop 1 ra)));
    // assert pure ((!max, !sum) %~ seq_fold_left online_softmax_real_iter (hide (Seq.index ra 0, 1.0R)) (seq_drop 1 ra));
    // admit()
  };
  admit();
  let x = read a tid;
  let y = (exp (x `sub` !max) `div` !sum);
  write_cell b tid y;
  ()
}

ghost
fn setup
  (#et : Type0) {| floating et, real_like et |}
  (nth : szp{nth <= max_threads})
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
  (nth : szp{nth <= max_threads})
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
  (nth : szp{nth <= max_threads})
  (#lenab : szp{lenab <= max_blocks * max_threads})
  (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
  (a : array1 et l { is_global a })
  (b : array1 et l { is_global b })
  (#va : erased (lseq et lenab))
  (ra : erased (lseq real lenab) { va %~ ra })
  (#_: squash ( forall (i:natlt lenab). (va @! i) `gt` minus_inf))
  : kernel_desc
      (requires a |-> va ** (exists* (vb : lseq et lenab). b |-> vb))
      (ensures  a |-> va ** (exists* (vb' : lseq et lenab).
        b |-> vb' **
        pure (vb' %~ online_softmax_real ra)))
= {
    nthr = lenab;
    f = kfonline_softmax lenab a b;

    frame    = pure (SZ.fits (layout_size l));
    teardown = teardown lenab a b;
    setup    = setup lenab #l a b #va;
    kpre =  (kpre_post #et lenab #l a b #va);
    kpost = (kpre_post #et lenab #l a b #va);
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
  launch_sync (konline_softmax nth lenab a b #va ra);
  admit();
  ()
}
