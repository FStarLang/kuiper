module Kuiper.Kernel.GEMMCPU

#lang-pulse

open Kuiper
open Kuiper.Tensor.Layout
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module M = Kuiper.Array2
open Kuiper.EMatrix { ematrix, to_real_matrix }

#set-options "--z3rlimit 20"

inline_for_extraction noextract
fn matmul_cpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (#et : Type0) {| scalar et |}
  (#m #n #k : szp) (* concrete args *)
  (#lA : M.full_layout m k)
  (#lB : M.full_layout k n)
  (#lC : M.full_layout m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (a b : vec et)
  (#sa : erased (seq et){len sa == m * k})
  (#sb : erased (seq et){len sb == k * n})
  norewrite
  preserves
    cpu ** a |-> sa ** b |-> sb
  requires
    pure (size_req m n k)
  returns
    c : vec et
  ensures
    c |-> (to_seq lC <|
             MS.matmul (from_seq lA sa)
                       (from_seq lB sb))
{
  let gA = M.alloc0 #et _ _ lA;
  let gB = M.alloc0 #et _ _ lB;
  let gC = M.alloc0 #et _ _ lC;

  M.copy_from_vec gA a;
  M.copy_from_vec gB b;

  mmcomb_gpu MS.comb2 gA gB gC;

  let c = Pulse.Lib.Vec.alloc #et zero (SZ.mul m n);
  M.copy_to_vec c gC;

  M.free gA;
  M.free gB;
  M.free gC;

  c
}

(* This will dinamically abort if the dimensions (rows/shared/cols) are not
   multiples of tile. *)
inline_for_extraction noextract
fn mmcomb_gpu_tiled
  (#size_req : tiled_size_req_t)
  (mmcomb_gpu : tiled_matmulcomb_gpu_ty size_req)
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| cA : ctlayout lA, cB : ctlayout lB, cC : ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (#eA #eB #eC : ematrix et _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req (m / tile) (n / tile) (k / tile) tile) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  dassert (tile >^ 0sz);
  dguard (m %^ tile = 0sz);
  dguard (n %^ tile = 0sz);
  dguard (k %^ tile = 0sz);
  let mm = m /^ tile;
  let nn = n /^ tile;
  let kk = k /^ tile;

  // None of these implicits should be needed. (Well, maybe the first
  // three until Kuiper.Concrete works really well.)
  mmcomb_gpu tile comb
    #mm #nn #kk
    #_ #_ #_
    #cA #cB #cC
    gA gB gC;

  ()
}
