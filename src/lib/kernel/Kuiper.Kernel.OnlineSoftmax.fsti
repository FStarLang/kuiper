module Kuiper.Kernel.OnlineSoftmax

#lang-pulse
open Kuiper 
open Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec
open Kuiper.Array1
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

module SMX = Kuiper.Kernel.Softmax

val max_real: real -> real -> real

let online_softmax_real_iter (md: erased (tuple2 real real)) (x:real) : erased (tuple2 real real) =
  let (m,d) = md in 
  let m' = max_real m x in
  let d' = d *. (rexp (m -. m')) +. rexp (x -. m') in
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

let test (x : real { x >. 0.0R }) =
  assert (x =!= 0.0R)

let online_softmax_real (s:Seq.seq real { Seq.length s > 0 }) : GTot (seq real) =
  (* TODO: rewrite using some of the stuff in Kuiper.Seq.Common.fsti ?
  seq_take, seq_drop *)
  let x = Seq.index s 0 in
  let (m, (d : real)) = reveal (seq_fold_left online_softmax_real_iter (hide (x, 1.0R)) (seq_drop 1 s)) in
  // TODO: why ??
  // assert (d >. 0.0R ==> d =!= 0.0R);
  // let d': (r: real{r =!= 0.R}) = d in
  // assert (d =!= 0.0R);
  seq_map (fun x -> rexp (x -. m) /. d) s

unfold
type softmax_notinplace_gpu_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (nth : szp{nth <= max_threads})
     (#lenab : szp{lenab <= max_blocks * max_threads})
     (#l : Kuiper.Array1.layout lenab) {| ctlayout l |} (* TODO: let them have different layouts *)
     (a : array1 et l { is_global a })
     (b : array1 et l { is_global b })
     (#va : erased (lseq et lenab))
     (ra : erased (lseq real lenab) { va %~ ra })
     (#_: squash ( forall (i:natlt lenab). valid (va @! i) /\ min_val `lte` (va @! i)))
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    exists* (vb : lseq et lenab). on gpu_loc (b |-> vb)
  ensures
    exists* (vb' : lseq et lenab).
      on gpu_loc (b |-> vb') **
      pure (vb' %~ SMX.softmax_real ra)

(*
TODO

unfold
type softmax_notinplace_ty (et : Type0) {| floating et, real_like et, floating_real_like et  |} =
  fn (nth : szp{nth <= max_threads})
     (#lena : szp)
     (a : Vec.lvec et lena)
     (#va : erased (lseq et lena))
     (ra : erased (lseq real lena))
  preserves
    cpu **
    a |-> va 
  requires
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
    // ^ This could be removed
*)

inline_for_extraction noextract
val online_softmax_gpu (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : softmax_notinplace_gpu_ty et

// TODO
// inline_for_extraction noextract
// val online_softmax (#et : Type0) {| floating et, real_like et, floating_real_like et |}
//   : softmax_notinplace_ty et
