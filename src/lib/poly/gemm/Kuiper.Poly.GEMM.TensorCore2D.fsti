module Kuiper.Poly.GEMM.TensorCore2D

#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix

open Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = FStar.SizeT

inline_for_extraction noextract
let warp_sz = 32sz
inline_for_extraction noextract
let warp_size = SZ.v warp_sz
