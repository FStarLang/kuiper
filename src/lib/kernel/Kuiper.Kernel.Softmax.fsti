module Kuiper.Kernel.Softmax

#lang-pulse
open Kuiper
open Kuiper.Real { rexp }
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT
module Vec = Pulse.Lib.Vec
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }

let softmax_real (s:Seq.seq real { Seq.length s > 0 }) =
  let exps = seq_map rexp s in
  let summ : real = rsum exps in
  seq_map (fun x -> rexp x /. summ) s

unfold
type softmax_gpu_ty (et : Type0) {| floating et, real_like et |} =
  fn (nth : szp{nth <= max_threads})
     (#lena : szp)
     (a : array1 et (l1_forward lena) { is_global a })
     (#va : erased (lseq et lena))
     (ra : erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ softmax_real ra)

unfold
type softmax_ty (et : Type0) {| floating et, real_like et |} =
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
      pure (va' %~ softmax_real ra)

inline_for_extraction noextract
val softmax_gpu (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : softmax_gpu_ty et

inline_for_extraction noextract
val softmax (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : softmax_ty et
