module Kuiper.Kernel.UGS.Copy
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT

(* Materialize an arbitrary logical matrix view into contiguous row-major
   storage. This is deliberately scalar: it exists to separate physical
   packing from the tensor-core kernel's semantic matrix input. *)
inline_for_extraction noextract
fn copy_to_row_major
  (#m #n : szp)
  (#lSrc : layout2 m n) {| ctlayout lSrc |}
  (#_ : squash (SZ.fits (m * n)))
  (gSrc : array2 half lSrc { is_global gSrc })
  (gDst : array2 half (rm m n) { is_global gDst })
  (#fSrc : perm)
  (#vSrc #vDst0 : chest2 half m n)
  requires
    gpu **
    gSrc |-> Frac fSrc vSrc **
    gDst |-> vDst0
  ensures
    gpu **
    gSrc |-> Frac fSrc vSrc **
    gDst |-> vSrc
