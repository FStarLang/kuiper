module Kuiper.Poly.MatMulCPU

#lang-pulse

open Kuiper
open Kuiper.Matrix.Common
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module M  = Kuiper.Matrix
module R = Kuiper.Matrix.Reprs

inline_for_extraction noextract
fn matmul_cpu
  (matmul_gpu : matmul_gpu_ty)
  (#et : Type0) {| scalar et |}
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp{three_fits rows shared cols})
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
    pure (SZ.fits (rows * shared) /\ SZ.fits (shared * cols) /\ SZ.fits (rows * cols)) **
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

  matmul_gpu gA gB gC;

  let c = Pulse.Lib.Vec.alloc #et zero (SZ.mul rows cols);
  M.gpu_matrix_to_array c gC;

  M.gpu_matrix_free gA;
  M.gpu_matrix_free gB;
  M.gpu_matrix_free gC;

  c
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
  (matmul_gpu : matmul_gpu_ty)
  (#et : Type0) {| scalar et |}
  (#rows : szp)
  (#shared : szp)
  (#cols : szp{three_fits rows shared cols})
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
    pure (SZ.fits (rows * shared) /\ SZ.fits (shared * cols) /\ SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks) **
    (gC |-> eC)
  ensures
    gC |-> mtranspose (MS.matmul eA eB)
{
  let gC' = GhostTranspose.ghost_transpose1 gC;
  matmul_gpu gA gB gC';
  let gC'' = GhostTranspose.ghost_transpose2 gC';
  M.core_match gC gC'';
  rewrite each gC'' as gC;
  ()
}

inline_for_extraction noextract
fn specialize_to_type_and_reprs
  (matmul_gpu : matmul_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp{three_fits rows shared cols})
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
    pure (SZ.fits (rows * shared) /\ SZ.fits (shared * cols) /\ SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks)
  returns
    c : vec et
  ensures
    (c |-> to_seq (rC rows cols) <|
             MS.matmul (from_seq (rA rows shared) sa)
                       (from_seq (rB shared cols) sb))
{
  matmul_cpu matmul_gpu #et #_ #rows #shared #cols #_ #_ #_ #(cA.map _ _) #(cB.map _ _) #(cC.map _ _) a b #sa #sb
}
