module Kuiper.MatMul.Tiled.SHMem

#lang-pulse

open Kuiper
module R = Kuiper.Matrix.Reprs
module P = Kuiper.Poly.MatMul.Tiled.SHMem
module M4 = Kuiper.Matrix4
module Tiled = Kuiper.MatMul.Tiled
friend Kuiper.MatMul.Tiled

let inst = 
   P.matmul_gpu 32sz
   #f32 #_ #32sz #32sz #32sz
   (R.row_major 1024 1024)
   (R.row_major 1024 1024)
   (R.row_major 1024 1024)
   #(Tiled.clayout4_from_clayout #32sz #32sz 32sz
     #(R.row_major 1024 1024)
     (R.crepr_row_major.map 1024sz 1024sz))
   #(Tiled.clayout4_from_clayout #32sz #32sz 32sz
     #(R.row_major 1024 1024)
     (R.crepr_row_major.map 1024sz 1024sz))
   #(Tiled.clayout4_from_clayout #32sz #32sz 32sz
     #(R.row_major 1024 1024)
     (R.crepr_row_major.map 1024sz 1024sz))
