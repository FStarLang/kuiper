module Klas.Softmax

#lang-pulse
open Kuiper
module K = Kuiper.Kernel.Softmax
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }

inline_for_extraction noextract
type softmax_gpu_flat_ty (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (nth : szp{nth <= max_threads})
    (#lena : szp)
    (a : array1 et (l1_forward lena) { is_global a })
    (#va: erased (lseq et lena))
    (ra: erased (lseq real lena))
    preserves
      cpu
    requires
      on gpu_loc (a |-> va) **
      pure (va %~ ra) **
      pure (lena <= max_blocks * max_threads)
    ensures
      exists* (va' : lseq et lena).
        on gpu_loc (a |-> va') **
        pure (va' %~ K.softmax_real ra)

val softmax_gpu_n_f16 : softmax_gpu_flat_ty f16
val softmax_gpu_n_f32 : softmax_gpu_flat_ty f32
val softmax_gpu_n_f64 : softmax_gpu_flat_ty f64

val softmax_n_f16 : K.softmax_ty f16
val softmax_n_f32 : K.softmax_ty f32
val softmax_n_f64 : K.softmax_ty f64
