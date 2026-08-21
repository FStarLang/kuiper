module Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

open Kuiper
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array
open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
#lang-pulse

inline_for_extraction let () = ()

