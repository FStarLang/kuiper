module Kuiper.Kernel.GEMM.TensorCore2D.ProofSupport

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 120"

open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.TensorCore
open Pulse.Lib.Array

inline_for_extraction let () = ()
