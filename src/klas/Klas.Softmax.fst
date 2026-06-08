module Klas.Softmax

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.Softmax
module KS = Kuiper.Spec.Softmax
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Vec = Pulse.Lib.Vec
module Array1 = Kuiper.Array1

inline_for_extraction noextract
fn inst_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
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
      pure (va' %~ KS.softmax_real ra)
{
  K.softmax_gpu #et nth #lena a ra;
}

(* GPU-pointer variants, dynamic thread number *)
let softmax_gpu_n_f16 = inst_gpu
let softmax_gpu_n_f32 = inst_gpu
let softmax_gpu_n_f64 = inst_gpu

(* GPU-pointer variants, full blocks *)
let softmax_gpu_f16 lena = inst_gpu #f16 1024sz #lena
let softmax_gpu_f32 lena = inst_gpu #f32 1024sz #lena
let softmax_gpu_f64 lena = inst_gpu #f64 1024sz #lena

inline_for_extraction noextract
fn inst_cpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
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
      pure (va' %~ KS.softmax_real ra)
{
  let ga = Array1.alloc0 #et lena (l1_forward lena);
  Array1.memcpy_host_to_device ga a lena;
  inst_gpu nth ga ra;
  Array1.memcpy_device_to_host' a 0sz ga 0sz lena;
  Array1.free ga;
  ()
}

(* CPU-side variants, dynamic thread number *)
let softmax_n_f16 = inst_cpu #f16
let softmax_n_f32 = inst_cpu #f32
let softmax_n_f64 = inst_cpu #f64

(* CPU-side variants, full blocks *)
let softmax_f16 lena = inst_cpu #f16 1024sz #lena
let softmax_f32 lena = inst_cpu #f32 1024sz #lena
let softmax_f64 lena = inst_cpu #f64 1024sz #lena

(* TODO: would it help to use smin lena 1024? *)
