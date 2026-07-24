module Kuiper.Kernel.Cat

(* torch.cat of two n-dimensional tensors along `dim` (binary form).

   Both inputs agree on every dimension except `dim`; the output has the same
   shape with size `(dA @! dim) + (dB @! dim)` along `dim`. *)

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor

module SZ = Kuiper.SizeT

(* Abstract specification: value at output coordinate `x`. Peel the coordinate
   along `dim`; read from the first input when it is `< na`, otherwise from the
   second (shifting that coordinate down by `na`). *)
let abs_cat (#et: Type0) (#r : nat)
  (dim : natlt r) (dA dB dout : shape r)
  (eA : chest dA et) (eB : chest dB et)
  (na : nat { na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (x : abs dout) : GTot et
  = let (j, m) = (abs_bring_forward_bij dim dout).ff x in
    if j < na then
      acc eA ((abs_bring_forward_bij dim dA).gg (j, m))
    else
      acc eB ((abs_bring_forward_bij dim dB).gg (j - na, m))

let cat_chest (#et: Type0) (#r : nat)
  (dim : natlt r) (dA dB dout : shape r)
  (eA : chest dA et) (eB : chest dB et)
  (na : nat { na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  : chest dout et
  = Kuiper.Chest.mk dout (abs_cat dim dA dB dout eA eB na pf_sz pfA pfB)

inline_for_extraction noextract
fn cat_gpu
  (#et : Type0) (#r : nat)
  (dA dB dout : shape r)
  (cdA : cshape dA) (cdB : cshape dB) (cdout : cshape dout)
  (dim : natlt r) (dimsz : szlt r { SZ.v dimsz == dim })
  (na : sz { SZ.v na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (#lA : tlayout dA) (#lB : tlayout dB) (#lOut : tlayout dout)
  {| ctlayout lA, ctlayout lB, ctlayout lOut |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (gOut : tensor et lOut {is_global gOut})
  (n : sz {SZ.v n == sizeof dout /\ n <= max_blocks * max_threads /\ n > 0})
  (eA : chest dA et)
  (eB : chest dB et)
  (#fA #fB : perm)
  preserves cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)
  requires on gpu_loc (live gOut)
  ensures
    on gpu_loc (gOut |-> cat_chest dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB)
