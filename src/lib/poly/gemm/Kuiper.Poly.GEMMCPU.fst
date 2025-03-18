module Kuiper.Poly.GEMMCPU

#lang-pulse

open Kuiper
open Kuiper.Matrix.Common
module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
open Kuiper.EMatrix { ematrix }
open Kuiper.EMatrix4 { ematrix4 }
open Kuiper.Matrix.Reprs.Type
module M  = Kuiper.Matrix
module M4 = Kuiper.Matrix4
open Kuiper.Matrix4 { mlayout4, clayout4 }
module R = Kuiper.Matrix.Reprs
module GT = Kuiper.Ghost.Transpose
module MC = Kuiper.Matrix.Casts

inline_for_extraction noextract
fn matmul_cpu
  (matmul_gpu : matmulcomb_gpu_ty)
  (#et : Type0) {| scalar et |}
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| cA : clayout lA |}
  {| cB : clayout lB |}
  {| cC : clayout lC |}
  (a : vec et)
  (b : vec et)
  (#sa : erased (seq et){ len sa == rows * shared })
  (#sb : erased (seq et){ len sb == shared * cols })
  preserves
    cpu **
    (a |-> sa) **
    (b |-> sb)
  requires
    (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
       is not needed for all kernels. *)
    pure (SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks)
  returns
    c : vec et
  ensures
    (c |-> to_seq lC <|
             MS.matmul (from_seq lA sa)
                       (from_seq lB sb))
{
  let gA = M.gpu_matrix_alloc0 #et _ _ lA;
  let gB = M.gpu_matrix_alloc0 #et _ _ lB;
  let gC = M.gpu_matrix_alloc0 #et _ _ lC;

  M.gpu_matrix_from_array gA a;
  M.gpu_matrix_from_array gB b;

  with vc. assert gC |-> vc;

  matmul_gpu MS.comb2 gA gB gC;

  let c = Pulse.Lib.Vec.alloc #et zero (SZ.mul rows cols);
  M.gpu_matrix_to_array c gC;

  M.gpu_matrix_free gA;
  M.gpu_matrix_free gB;
  M.gpu_matrix_free gC;

  c
}

(* This will dinamically abort if the dimensions (rows/shared/cols) are not
   multiples of tile. *)
inline_for_extraction noextract
fn matmul_gpu_tiled
  (tiled_matmul_gpu : tiled_matmulcomb_gpu_ty)
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| cA : clayout lA |}
  {| cB : clayout lB |}
  {| cC : clayout lC |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  preserves
    cpu **
    (gA |-> eA) **
    (gB |-> eB)
  requires
    pure (rows * cols <= max_blocks) **
    (gC |-> eC)
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  dassert (tile `SZ.gt` 0sz);
  dguard (rows   %^ tile = 0sz);
  dguard (shared %^ tile = 0sz);
  dguard (cols   %^ tile = 0sz);
  let mrows   = rows   /^ tile;
  let mshared = shared /^ tile;
  let mcols   = cols   /^ tile;

  let lA4 : mlayout4 mrows   mshared tile tile = lA;
  let lB4 : mlayout4 mshared mcols  tile tile = lB;
  let lC4 : mlayout4 mrows   mcols  tile tile = lC;
  let gA4 = MC.m2_to_m4 (SZ.v tile) (SZ.v mrows) (SZ.v mshared) #_ #_ #lA4 gA;
  let gB4 = MC.m2_to_m4 (SZ.v tile) (SZ.v mshared) (SZ.v mcols) #_ #_ #lB4 gB;
  let gC4 = MC.m2_to_m4 (SZ.v tile) (SZ.v mrows) (SZ.v mcols) #_ #_ #lC4 gC;
  tiled_matmul_gpu tile
    #et #_
    comb
    #mrows #mshared #mcols
    lA4 lB4 lC4
    #(M4.clayout4_from_clayout tile cA)
    #(M4.clayout4_from_clayout tile cB)
    #(M4.clayout4_from_clayout tile cC)
    gA4 gB4 gC4;

  let gA' = MC.m4_to_m2 (SZ.v tile) (SZ.v mrows) (SZ.v mshared) #_ #_ #lA4 gA4;
  let gB' = MC.m4_to_m2 (SZ.v tile) (SZ.v mshared) (SZ.v mcols) #_ #_ #lB4 gB4;
  let gC' = MC.m4_to_m2 (SZ.v tile) (SZ.v mrows) (SZ.v mcols) #_ #_ #lC4 gC4;

  rewrite each gA' as gA;
  rewrite each gB' as gB;
  rewrite each gC' as gC;
  ()
}

(* An example of computing tr(AB) by just shifting a view.
Basically:
  - Instantiating rA=rB=row_major, rC=col_major
  - Do the product, we get C = AB (in col-major)
  - View-shift C to get tr(AB) in row-major

  TODO: It would be nicer to do this just over matmul,
  but there is no view-like interface for CPU arrays.
*)
inline_for_extraction noextract
fn matmul_transpose_gpu
  (matmul_gpu : matmulcomb_gpu_ty)
  (#et : Type0) {| scalar et |}
  (#rows : szp)
  (#shared : szp)
  (#cols : szp)
  (gA : M.gpu_matrix et (R.row_major rows shared))
  (gB : M.gpu_matrix et (R.row_major shared cols))
  (gC : M.gpu_matrix et (R.row_major cols rows))
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et cols rows)
  preserves
    cpu **
    (gA |-> eA) ** (gB |-> eB)
  requires
    pure (SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks) **
    (gC |-> eC)
  ensures
    gC |-> mtranspose (MS.matmul eA eB)
{
  (* Recall that the lengths fit. We don't get a good error without this,
     but the problem is that we cannot call the crepr instance for gA/gB/gC
     without this fact. *)
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  M.gpu_matrix_pts_to_ref gC;
  GT.ghost_transpose1 gC;
  matmul_gpu MS.comb2 gA gB (GT.row2col gC);
  GT.ghost_transpose1_back gC;
  ()
}

inline_for_extraction noextract
fn specialize_as_gemm_to_type_and_reprs_gpu
  (matmul_gpu : matmulcomb_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
  (alpha beta : et)
  (#rows #shared #cols : szp) (* concrete args *)
  (gA : gpu_matrix et (rA rows shared))
  (gB : gpu_matrix et (rB shared cols))
  (gC : gpu_matrix et (rC rows cols))
  (#ma : ematrix et rows shared)
  (#mb : ematrix et shared cols)
  (#mc0 : ematrix et rows cols)
  preserves
    cpu **
    (gA |-> ma) **
    (gB |-> mb)
  requires
    (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
       is not needed for all kernels. *)
    pure (SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks) **
    (gC |-> mc0)
  ensures
    gC |-> MS.gemm alpha beta mc0 ma mb
{
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  M.gpu_matrix_pts_to_ref gC;

  matmul_gpu #et #_ (MS.lincomb alpha beta) #rows #shared #cols #_ #_ #_ #(cA.map _ _) #(cB.map _ _) #(cC.map _ _) gA gB gC;
}

inline_for_extraction noextract
fn specialize_as_matmul_to_type_and_reprs_gpu
  (matmul_gpu : matmulcomb_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
  (#rows #shared #cols : szp) (* concrete args *)
  (gA : gpu_matrix et (rA rows shared))
  (gB : gpu_matrix et (rB shared cols))
  (gC : gpu_matrix et (rC rows cols))
  (#ma : ematrix et rows shared)
  (#mb : ematrix et shared cols)
  (#mc0 : ematrix et rows cols)
  preserves
    cpu **
    (gA |-> ma) **
    (gB |-> mb)
  requires
    (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
       is not needed for all kernels. *)
    pure (SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks) **
    (gC |-> mc0)
  ensures
    gC |-> MS.matmul ma mb
{
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  M.gpu_matrix_pts_to_ref gC;

  matmul_gpu #et #_ MS.comb2 #rows #shared #cols #_ #_ #_ #(cA.map _ _) #(cB.map _ _) #(cC.map _ _) gA gB gC;
}

inline_for_extraction noextract
fn cpu_wrap_matmul
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
  (matmul_gpu : matmulcomb_gpu_ty)
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp)
  (a : vec et)
  (b : vec et)
  (#sa : erased (seq et){ len sa == rows * shared })
  (#sb : erased (seq et){ len sb == shared * cols })
  preserves
    cpu **
    (a |-> sa) **
    (b |-> sb)
  requires
    (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
       is not needed for all kernels. *)
    pure (SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks)
  returns
    c : vec et
  ensures
    (c |-> to_seq (rC rows cols) <|
             MS.matmul (from_seq (rA rows shared) sa)
                       (from_seq (rB shared cols) sb))
{
  Pulse.Lib.Vec.pts_to_len a;
  Pulse.Lib.Vec.pts_to_len b;

  matmul_cpu matmul_gpu #et #_ #rows #shared #cols #_ #_ #_ #(cA.map _ _) #(cB.map _ _) #(cC.map _ _) a b #sa #sb
}

inline_for_extraction noextract
let specialize_as_matmul_to_type_and_reprs_cpu
  (matmul_gpu : matmulcomb_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
  : fixed_repr_matmul_cpu_ty et rA rB rC #cA #cB #cC
  = cpu_wrap_matmul et rA rB rC matmul_gpu
