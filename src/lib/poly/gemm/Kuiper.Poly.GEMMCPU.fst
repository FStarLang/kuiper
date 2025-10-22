module Kuiper.Poly.GEMMCPU

#lang-pulse

open Kuiper
open Kuiper.Matrix.Common
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.EMatrix { ematrix }
open Kuiper.EMatrix4 { ematrix4 }
open Kuiper.Matrix.Reprs.Type
module M  = Kuiper.Matrix
module M4 = Kuiper.Matrix4
open Kuiper.Matrix4 { mlayout4, clayout4 }
module MC = Kuiper.Matrix.Casts

#set-options "--z3rlimit 20"

inline_for_extraction noextract
fn matmul_cpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (#et : Type0) {| scalar et |}
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp)
  (#lA : full_mlayout rows shared)
  (#lB : full_mlayout shared cols)
  (#lC : full_mlayout rows cols)
  {| cA : clayout lA |}
  {| cB : clayout lB |}
  {| cC : clayout lC |}
  (a : vec et)
  (b : vec et)
  (#sa : erased (seq et){ len sa == rows * shared })
  (#sb : erased (seq et){ len sb == shared * cols })
  norewrite
  preserves
    cpu **
    a |-> sa **
    b |-> sb
  requires
    pure (size_req rows shared cols)
  returns
    c : vec et
  ensures
    c |-> (to_seq lC <|
             MS.matmul (from_seq lA sa)
                       (from_seq lB sb))
{
  let gA = M.gpu_matrix_alloc0 #et _ _ lA;
  let gB = M.gpu_matrix_alloc0 #et _ _ lB;
  let gC = M.gpu_matrix_alloc0 #et _ _ lC;

  M.gpu_matrix_from_array gA a;
  M.gpu_matrix_from_array gB b;

  with vc. assert gC |-> vc;

  mmcomb_gpu MS.comb2 gA gB gC;

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
fn mmcomb_gpu_tiled
  (#size_req : tiled_size_req_t)
  (mmcomb_gpu : tiled_matmulcomb_gpu_ty size_req)
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : full_mlayout rows shared)
  (#lB : full_mlayout shared cols)
  (#lC : full_mlayout rows cols)
  {| cA : clayout lA |}
  {| cB : clayout lB |}
  {| cC : clayout lC |}
  (gA : M.gpu_matrix et lA)
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (size_req (rows / tile) (shared / tile) (cols / tile) tile) **
    gC |-> eC
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  dassert (tile >^ 0sz);
  dguard (rows   %^ tile = 0sz);
  dguard (shared %^ tile = 0sz);
  dguard (cols   %^ tile = 0sz);
  let mrows   = rows   /^ tile;
  let mshared = shared /^ tile;
  let mcols   = cols   /^ tile;

  mmcomb_gpu
    tile
    comb
    #mrows #mshared #mcols // should not be needed
    lA lB lC
    #cA #cB #cC            // should not be needed
    gA gB gC;

  ()
}

inline_for_extraction noextract
fn mmcomb_gpu_block_tiled1d
  (tiled_mmcomb_gpu : block_tiled1d_matmulcomb_gpu_ty)
  (bm bn bk : szp)
  (tm : szp{tm /? bm /\ (bm/tm * bn <= max_threads)})
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : full_mlayout rows shared)
  (#lB : full_mlayout shared cols)
  (#lC : full_mlayout rows cols)
  {| cA : clayout lA |}
  {| cB : clayout lB |}
  {| cC : clayout lC |}
  (gA : M.gpu_matrix et lA)
  (#fA : perm)
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (gC : M.gpu_matrix et lC)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure ((rows / bm) * (cols / bn) <= max_blocks) **
    gC |-> eC
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  dassert (bm >^ 0sz);
  dassert (bn >^ 0sz);
  dassert (bk >^ 0sz);
  dassert (tm >^ 0sz);
  dassert (bm %^ tm = 0sz);
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);
  // a HORRIBLE restriction
  dguard ((bm /^ tm *^ bn) = bm *^ bk);
  dguard ((bm /^ tm *^ bn) = bn *^ bk);
  Math.Lemmas.lemma_cancel_mul bm bn bk; // odd...
  dassert (bm = bn);
  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;

  let lA4 : mlayout4 mrows   mshared bm bk = lA;
  let lB4 : mlayout4 mshared mcols  bk bn = lB;
  let lC4 : mlayout4 mrows   mcols  bm bn = lC;
  let gA4 = MC.m2_to_m4 (SZ.v bm) (SZ.v bk) (SZ.v mrows) (SZ.v mshared) gA;
  let gB4 = MC.m2_to_m4 (SZ.v bk) (SZ.v bn) (SZ.v mshared) (SZ.v mcols) gB;
  let gC4 = MC.m2_to_m4 (SZ.v bm) (SZ.v bn) (SZ.v mrows) (SZ.v mcols) gC;
  tiled_mmcomb_gpu
    comb
    bm bn bk
    tm
    lA4 lB4 lC4
    #(M4.clayout4_from_clayout bm bk cA)
    #(M4.clayout4_from_clayout bk bn cB)
    #(M4.clayout4_from_clayout bm bn cC)
    gA4 gB4 gC4;

  let gA' = MC.m4_to_m2 gA4;
  let gB' = MC.m4_to_m2 gB4;
  let gC' = MC.m4_to_m2 gC4;

  rewrite each gA' as gA;
  rewrite each gB' as gB;
  rewrite each gC' as gC;
  ()
}

inline_for_extraction noextract
fn specialize_as_gemm_to_type_and_reprs_gpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA, cB : crepr rB, cC : crepr rC |}
  (alpha beta : et)
  (#rows #shared #cols : szp) (* concrete args *)
  (gA : gpu_matrix et (rA rows shared))
  (gB : gpu_matrix et (rB shared cols))
  (gC : gpu_matrix et (rC rows cols))
  (#ma : ematrix et rows shared)
  (#mb : ematrix et shared cols)
  (#mc0 : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    gA |-> ma **
    gB |-> mb
  requires
    pure (size_req rows shared cols) **
    gC |-> mc0
  ensures
    gC |-> MS.gemm alpha beta mc0 ma mb
{
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  M.gpu_matrix_pts_to_ref gC;

  mmcomb_gpu #et #_ (MS.lincomb beta alpha) #rows #shared #cols #_ #_ #_ #(cA.map _ _) #(cB.map _ _) #(cC.map _ _) gA gB gC;
}

inline_for_extraction noextract
fn specialize_as_matmul_to_type_and_reprs_gpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA, cB : crepr rB, cC : crepr rC |}
  (#rows #shared #cols : szp) (* concrete args *)
  (gA : gpu_matrix et (rA rows shared))
  (gB : gpu_matrix et (rB shared cols))
  (gC : gpu_matrix et (rC rows cols))
  (#ma : ematrix et rows shared)
  (#mb : ematrix et shared cols)
  (#mc0 : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    gA |-> ma **
    gB |-> mb
  requires
    pure (size_req rows shared cols) **
    gC |-> mc0
  ensures
    gC |-> MS.matmul ma mb
{
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  M.gpu_matrix_pts_to_ref gC;

  mmcomb_gpu MS.comb2
    #rows #shared #cols
    #(rA _ _) #(rB _ _) #(rC _ _)
    gA gB gC;
}

inline_for_extraction noextract
fn cpu_wrap_matmul
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA, cB : crepr rB, cC : crepr rC |}
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (#rows #shared : szp) (* concrete args *)
  (#cols : szp)
  (a : vec et)
  (b : vec et)
  (#sa : erased (seq et){ len sa == rows * shared })
  (#sb : erased (seq et){ len sb == shared * cols })
  norewrite
  preserves
    cpu **
    a |-> sa **
    b |-> sb
  requires
    pure (size_req rows shared cols) **
    pure (SZ.fits (rows * cols))
  returns
    c : vec et
  ensures
    c |-> (to_seq (rC rows cols) <|
             MS.matmul (from_seq (rA rows shared) sa)
                       (from_seq (rB shared cols) sb))
{
  Pulse.Lib.Vec.pts_to_len a;
  Pulse.Lib.Vec.pts_to_len b;

  let c = matmul_cpu mmcomb_gpu
    #_ #_ #rows #shared #cols
    #(rA _ _) #(rB _ _) #(rC _ _)
    a b;

  c;
}

inline_for_extraction noextract
let specialize_as_matmul_to_type_and_reprs_cpu
  (#size_req : _)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
  : fixed_repr_matmul_cpu_ty et size_req rA rB rC
  = cpu_wrap_matmul et rA rB rC mmcomb_gpu
