module Klas.Softmax

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.Softmax
module KS = Kuiper.Spec.Softmax
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module Vec = Pulse.Lib.Vec

inline_for_extraction noextract
fn inst_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
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
  norewrite
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
{
  let ga = alloc0 #et lena (l1_forward lena);
  with em. assert on gpu_loc (ga |-> em);
  map_loc gpu_loc #(ga |-> em) #(core ga |-> to_seq (l1_forward lena) em)
    fn _ { tensor_concr ga; };
  gpu_memcpy_host_to_device (core ga) a lena;
  map_loc gpu_loc #(core ga |-> reveal va) #(ga |-> from_seq (l1_forward lena) va)
    fn _ {
      tensor_abs' (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> from_seq (l1_forward lena) va)
           as (ga |-> from_seq (l1_forward lena) va);
    };
  inst_gpu nth ga (seq_to_chest1 ra);
  with res. assert on gpu_loc (ga |-> res);
  map_loc gpu_loc #(ga |-> res) #(core ga |-> to_seq (l1_forward lena) res)
    fn _ { tensor_concr ga; };
  gpu_memcpy_device_to_host a (core ga) lena;
  map_loc gpu_loc #(core ga |-> to_seq (l1_forward lena) res) #(ga |-> res)
    fn _ {
      tensor_abs (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> reveal res)
           as (ga |-> reveal res);
    };
  free ga;
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
