module Kuiper.MatMul.Tiled.SHMem

#lang-pulse

open Kuiper
module R = Kuiper.Matrix.Reprs
module P = Kuiper.Poly.MatMul.Tiled.SHMem
module M4 = Kuiper.Matrix4
module Tiled = Kuiper.MatMul.Tiled
friend Kuiper.MatMul.Tiled

let inst_gpu = 
   P.matmul_gpu 32sz
   #u64 #_ #32sz #32sz #32sz
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

fn matmul
  (a : vec u64)
  (b : vec u64)
  (#sa : erased (seq u64){ len sa == 1024 * 1024 })
  (#sb : erased (seq u64){ len sb == 1024 * 1024 })
  preserves
   cpu ** (a |-> sa) ** (b |-> sb)
  requires
   pure (three_fits 1024 1024 1024) **
   pure (1024 * 1024 <= max_blocks)
  returns
    c : vec u64
  ensures
   exists* sc. c |-> sc
{
  let gA = M4.gpu_matrix_alloc0 #u64 32sz 32sz 32sz 32sz (R.row_major 1024 1024);
  let gB = M4.gpu_matrix_alloc0 #u64 32sz 32sz 32sz 32sz (R.row_major 1024 1024);
  let gC = M4.gpu_matrix_alloc0 #u64 32sz 32sz 32sz 32sz (R.row_major 1024 1024);

  M4.gpu_matrix_from_array #u64 #_ #32sz #32sz #32sz #32sz gA a;
  M4.gpu_matrix_from_array #u64 #_ #32sz #32sz #32sz #32sz gB b;

  with vc. assert gC |-> vc;

  inst_gpu gA gB gC;

  let c = Pulse.Lib.Vec.alloc #u64 zero (1024sz *^ 1024sz);
  M4.gpu_matrix_to_array #_ #_ #32sz #32sz #32sz #32sz c gC;

  M4.gpu_matrix_free gA;
  M4.gpu_matrix_free gB;
  M4.gpu_matrix_free gC;

  c
}
