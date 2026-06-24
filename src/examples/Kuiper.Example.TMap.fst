module Kuiper.Example.TMap

(* Example/test for the pointwise tensor map kernel (Kuiper.Kernel.TMap).

   map_gpu applies a function f : et -> et to every cell of a global
   tensor, turning [a |-> s] into [a |-> chest_map f s]. The examples
   below instantiate it at concrete element types, layouts and
   functions. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Kernel.TMap

(* Increment every element of a 1-D u32 tensor on the GPU. *)
fn incr_all_1d
  (a : tensor u32 (l1_forward 1024) { is_global a })
  (#s : chest (1024 @| INil) u32)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map (add one) s)
{
  map_gpu (CCons 1024sz CNil) (add one) 1024sz a;
}

inline_for_extraction noextract
instance _crutch_2d : ctlayout (l2_row_major 1024 1024) = c_l2_row_major 1024 1024sz

fn incr_all_1d2
  (a : tensor u32 (l2_row_major 1024 1024) { is_global a })
  (#s : chest (1024 @| 1024 @| INil) u32)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map (add one) s)
{
  map_gpu (CCons 1024sz (CCons 1024sz CNil)) (add one) (1024sz *^ 1024sz) a;
}
