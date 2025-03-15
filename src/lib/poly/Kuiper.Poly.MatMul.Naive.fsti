module Kuiper.Poly.MatMul.Naive

(* This is a very naive matmul, spawning MxN blocks of 1 thread
to do the computation. *)

#lang-pulse

open Kuiper
open Kuiper.Poly.MatMulGPU.Type
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
val kdesc : matmul_kernel_ty
