module Klas.Softmax

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
module KS = Kuiper.Spec.Softmax
module Vec = Pulse.Lib.Vec

inline_for_extraction noextract
type softmax_gpu_flat_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
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
        pure (va' %~ KS.softmax_real ra)

val softmax_gpu_n_f16 : softmax_gpu_flat_ty f16
val softmax_gpu_n_f32 : softmax_gpu_flat_ty f32
val softmax_gpu_n_f64 : softmax_gpu_flat_ty f64

inline_for_extraction noextract
type softmax_flat_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (nth : szp{nth <= max_threads})
    (#lena : szp)
    (a : Vec.lvec et lena)
    (#va : erased (lseq et lena))
    (ra  : erased (lseq real lena))
    preserves
      cpu
    requires
      a |-> va **
      pure (va %~ ra) **
      pure (lena <= max_blocks * max_threads)
    ensures
      exists* (va' : lseq et lena).
        a |-> va' **
        pure (va' %~ chest1_to_seq (KS.softmax_real (seq_to_chest1 ra)))

val softmax_n_f16 : softmax_flat_ty f16
val softmax_n_f32 : softmax_flat_ty f32
val softmax_n_f64 : softmax_flat_ty f64
