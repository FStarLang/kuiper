module Kuiper.Kernel.LogSoftmax

#lang-pulse
open Kuiper
open Kuiper.Real { log }
open Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec
module KS = Kuiper.Spec.Softmax
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }

(* Every entry of [softmax_real] is [exp x] divided by a strictly positive sum,
   hence strictly positive.  F* master emits one query per proof obligation, so
   the E-matching that used to establish this for the [chest_refine] refinement
   below no longer fires on its own: spell the argument out. *)
let softmax_real_pos (#n : nat) (s : chest1 real n)
  : Lemma (forall (i : abs (n @| INil)). acc (KS.softmax_real s) i >. 0.0R)
  = introduce forall (i : abs (n @| INil)). acc (KS.softmax_real s) i >. 0.0R
    with (
      let (i0, ()) = i in
      let exps = chest_map exp s in
      let sq = chest1_to_seq exps in
      assert (Seq.length sq == n /\ n > 0);
      assert (forall (j : natlt n). Seq.index sq j == exp (acc1 s j));
      Kuiper.Real.sum_non_zero sq 0.0R;
      assert (chest1_rsum exps >. 0.0R)
    )

// Log of softmax.
let log_softmax_real #n (s : chest1 real n) =
  softmax_real_pos s;
  chest_map log (chest_refine (fun x -> x >. 0.0R) (KS.softmax_real s))

unfold
type log_softmax_gpu_ty (et : Type0) {| floating et, real_like et |} =
  fn (nth : szp{nth <= max_threads})
     (#lena : szp)
     (a : array1 et (l1_forward lena) { is_global a })
     (#va : chest1 et lena)
     (ra  : chest1 real lena)
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : chest1 et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ log_softmax_real ra)

unfold
type log_softmax_ty (et : Type0) {| floating et, real_like et |} =
  fn (nth : szp{nth <= max_threads})
     (#lena : szp)
     (a : Vec.lvec et lena)
     (#va : erased (lseq et lena))
     (ra : erased (lseq real lena))
  preserves
    cpu
  requires
    a |-> va **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
    // ^ This could be removed
  ensures
    exists* (va' : lseq et lena).
      a |-> va' **
      pure (va' %~ chest1_to_seq (log_softmax_real (seq_to_chest1 ra)))

inline_for_extraction noextract
val log_softmax_gpu (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : log_softmax_gpu_ty et

inline_for_extraction noextract
val log_softmax (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : log_softmax_ty et
