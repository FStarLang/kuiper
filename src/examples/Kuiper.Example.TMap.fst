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


(*
(* The kernel is polymorphic in both the element type and the mapped
   function. This wrapper fixes the (concrete) 1024-wide 1-D layout and
   leaves everything else open, taking the ctlayout as an instance arg. *)
inline_for_extraction noextract
fn map_1024
  (#et : Type0)
  (f : et -> et)
  {| ctlayout (l1_forward 1024) |}
  (a : tensor et (l1_forward 1024) { is_global a })
  (#s : chest (1024 @| INil) et)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)
{
  map_gpu f 1024sz a;
}

(* A 32x64 row-major tensor of f32 elements (32*64 = 2048 cells). *)
inline_for_extraction noextract
instance _crutch_2d : ctlayout (l2_row_major 32 64) = c_l2_row_major 32 64sz

(* Double every element of a 2-D f32 matrix on the GPU. *)
fn double_all_2d
  (a : tensor f32 (l2_row_major 32 64) { is_global a })
  (#s : chest (32 @| 64 @| INil) f32)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map (fun x -> add x x) s)
{
  map_gpu (fun (x:f32) -> add x x) 2048sz a;
}

(* End-to-end: allocate a 1-D u32 tensor on the GPU, increment every
   element, then free it. *)
fn main ()
  requires cpu
  ensures  cpu
{
  let a = alloc0 #u32 1024sz (l1_forward 1024);
  map_gpu (fun (x:u32) -> add x one) 1024sz a;
  free a;
}
