module Kuiper.Poly.GEMMCPU

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor.Layout
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module M = Kuiper.Array2
open Kuiper.EMatrix { ematrix, to_real_matrix, lemma_to_real_matrix_approximates }

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
    c |-> (M.to_seq lC <|
             MS.matmul (M.from_seq lA sa)
                       (M.from_seq lB sb))
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

// inline_for_extraction noextract
// fn mmcomb_gpu_tiled_approx
//   (#size_req : tiled_size_req_t)
//   (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
//   (tile : valid_tile)
//   (#et : Type0) {| scalar et, real_like et |}
//   (comb : binop et)
//   (comb_r : binop real { comb `approx2` comb_r })
//   (#m #n #k : szp)
//   (#lA : M.layout m k)
//   (#lB : M.layout k n)
//   (#lC : M.layout m n)
//   {| cA : ctlayout lA, cB : ctlayout lB, cC : ctlayout lC |}
//   (gA : M.array2 et lA { M.is_global gA })
//   (gB : M.array2 et lB { M.is_global gB })
//   (gC : M.array2 et lC { M.is_global gC })
//   (#eA : ematrix et m k)
//   (#eB : ematrix et k n)
//   (#eC : ematrix et m n)
//   (rA : ematrix real m k)
//   (rB : ematrix real k n)
//   (rC : ematrix real m n)
//   (#fA #fB : perm)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> Frac fA eA) **
//     on gpu_loc (gB |-> Frac fB eB)
//   requires
//     pure (size_req (m / tile) (n / tile) (k / tile) tile) **
//     pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
//     on gpu_loc (gC |-> eC)
//   ensures
//     exists* (eC' : ematrix et m n).
//       on gpu_loc (gC |-> eC') **
//       pure (eC' %~ MS.mmcomb comb_r rC rA rB)
// {
//   dassert (tile >^ 0sz);
//   dguard (m %^ tile = 0sz);
//   dguard (n %^ tile = 0sz);
//   dguard (k %^ tile = 0sz);
//   let mm = m   /^ tile;
//   let nn = n /^ tile;
//   let kk = k /^ tile;

//   mmcomb_gpu_approx tile comb comb_r
//     #mm #nn #kk
//     #_ #_ #_
//     #cA #cB #cC
//     gA gB gC
//     rA rB rC;

//   ()
// }

// inline_for_extraction noextract
// fn specialize_as_gemm_to_type_and_reprs_gpu
//   (#size_req : size_req_t)
//   (mmcomb_gpu : matmulcomb_gpu_ty size_req)
//   (et : Type0) {| scalar et |}
//   (rA rB rC : mrepr)
//   {| cA : crepr rA, cB : crepr rB, cC : crepr rC |}
//   (alpha beta : et)
//   (#rows #shared #cols : szp) (* concrete args *)
//   (gA : gpu_matrix et (rA rows shared) { M.is_global_matrix gA })
//   (gB : gpu_matrix et (rB shared cols) { M.is_global_matrix gB })
//   (gC : gpu_matrix et (rC rows cols) { M.is_global_matrix gC })
//   (#ma : ematrix et rows shared)
//   (#mb : ematrix et shared cols)
//   (#mc0 : ematrix et rows cols)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> ma) **
//     on gpu_loc (gB |-> mb)
//   requires
//     pure (size_req rows shared cols) **
//     on gpu_loc (gC |-> mc0)
//   ensures
//     on gpu_loc (gC |-> MS.gemm alpha beta mc0 ma mb)
// {
//   M.gpu_matrix_pts_to_ref_located gA;
//   M.gpu_matrix_pts_to_ref_located gB;
//   M.gpu_matrix_pts_to_ref_located gC;

//   mmcomb_gpu #et #_ (MS.lincomb alpha beta) #rows #shared #cols #_ #_ #_ #(cA.map _ _) #(cB.map _ _) #(cC.map _ _) gA gB gC;
// }

// inline_for_extraction noextract
// fn specialize_as_matmul_to_type_and_reprs_gpu
//   (#size_req : size_req_t)
//   (mmcomb_gpu : matmulcomb_gpu_ty size_req)
//   (et : Type0) {| scalar et |}
//   (rA rB rC : mrepr)
//   {| cA : crepr rA, cB : crepr rB, cC : crepr rC |}
//   (#rows #shared #cols : szp) (* concrete args *)
//   (gA : gpu_matrix et (rA rows shared) { M.is_global_matrix gA })
//   (gB : gpu_matrix et (rB shared cols) { M.is_global_matrix gB })
//   (gC : gpu_matrix et (rC rows cols) { M.is_global_matrix gC })
//   (#ma : ematrix et rows shared)
//   (#mb : ematrix et shared cols)
//   (#mc0 : ematrix et rows cols)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> ma) **
//     on gpu_loc (gB |-> mb)
//   requires
//     pure (size_req rows shared cols) **
//     on gpu_loc (gC |-> mc0)
//   ensures
//     on gpu_loc (gC |-> MS.matmul ma mb)
// {
//   M.gpu_matrix_pts_to_ref_located gA;
//   M.gpu_matrix_pts_to_ref_located gB;
//   M.gpu_matrix_pts_to_ref_located gC;

//   mmcomb_gpu MS.comb2
//     #rows #shared #cols
//     #(rA _ _) #(rB _ _) #(rC _ _)
//     gA gB gC;
// }

// inline_for_extraction noextract
// fn cpu_wrap_matmul
//   (et : Type0) {| scalar et |}
//   (rA rB rC : mrepr)
//   {| cA : crepr rA, cB : crepr rB, cC : crepr rC |}
//   (#size_req : size_req_t)
//   (mmcomb_gpu : matmulcomb_gpu_ty size_req)
//   (#rows #shared : szp) (* concrete args *)
//   (#cols : szp)
//   (a : vec et)
//   (b : vec et)
//   (#sa : erased (seq et){ len sa == rows * shared })
//   (#sb : erased (seq et){ len sb == shared * cols })
//   norewrite
//   preserves
//     cpu **
//     a |-> sa **
//     b |-> sb
//   requires
//     pure (size_req rows shared cols) **
//     pure (SZ.fits (rows * cols))
//   returns
//     c : vec et
//   ensures
//     c |-> (to_seq (rC rows cols) <|
//              MS.matmul (from_seq (rA rows shared) sa)
//                        (from_seq (rB shared cols) sb))
// {
//   Pulse.Lib.Vec.pts_to_len a;
//   Pulse.Lib.Vec.pts_to_len b;

//   let c = matmul_cpu mmcomb_gpu
//     #_ #_ #rows #shared #cols
//     #(rA _ _) #(rB _ _) #(rC _ _)
//     a b;

//   c;
// }

// inline_for_extraction noextract
// fn specialize_tiled_approx_gpu
//   (#size_req : tiled_size_req_t)
//   (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
//   (tile : valid_tile)
//   (et : Type0) {| scalar et |} {| real_like et |}
//   (repA repB repC : mrepr)
//   {| cA : crepr repA, cB : crepr repB, cC : crepr repC |}
//   (#rows #shared #cols : szp)
//   (gA : gpu_matrix et (repA rows shared) { M.is_global_matrix gA })
//   (gB : gpu_matrix et (repB shared cols) { M.is_global_matrix gB })
//   (gC : gpu_matrix et (repC rows cols) { M.is_global_matrix gC })
//   (#ma : ematrix et rows shared)
//   (#mb : ematrix et shared cols)
//   (#mc0 : ematrix et rows cols)
//   (rA : ematrix real rows shared)
//   (rB : ematrix real shared cols)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> ma) **
//     on gpu_loc (gB |-> mb)
//   requires
//     pure (size_req (rows / tile) (shared / tile) (cols / tile) tile) **
//     pure (ma %~ rA /\ mb %~ rB) **
//     on gpu_loc (gC |-> mc0)
//   ensures
//     exists* (mc' : ematrix et rows cols).
//       on gpu_loc (gC |-> mc') **
//       pure (mc' %~ MS.matmul rA rB)
// {
//   M.gpu_matrix_pts_to_ref_located gA;
//   M.gpu_matrix_pts_to_ref_located gB;
//   M.gpu_matrix_pts_to_ref_located gC;

//   with eC_val. assert on gpu_loc (gC |-> eC_val);

//   let rC : ematrix real rows cols = to_real_matrix eC_val;
//   lemma_to_real_matrix_approximates eC_val;
//   mmcomb_gpu_tiled_approx
//     mmcomb_gpu_approx
//     tile
//     #et #_ #_
//     MS.comb2 MS.comb2
//     #rows #shared #cols
//     #(repA _ _) #(repB _ _) #(repC _ _)
//     #(cA.map _ _) #(cB.map _ _) #(cC.map _ _)
//     gA gB gC
//     rA rB rC;

//   (* mmcomb comb2 rC rA rB == matmul rA rB by matmul_is_gemm *)
//   ()
// }

// inline_for_extraction noextract
// fn specialize_tiled_approx_gemm_gpu
//   (#size_req : tiled_size_req_t)
//   (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
//   (tile : valid_tile)
//   (et : Type0) {| scalar et |} {| real_like et |}
//   (repA repB repC : mrepr)
//   {| cA : crepr repA, cB : crepr repB, cC : crepr repC |}
//   (alpha beta : et)
//   (alpha_r beta_r : real)
//   (#rows #shared #cols : szp)
//   (gA : gpu_matrix et (repA rows shared) { M.is_global_matrix gA })
//   (gB : gpu_matrix et (repB shared cols) { M.is_global_matrix gB })
//   (gC : gpu_matrix et (repC rows cols) { M.is_global_matrix gC })
//   (#ma : ematrix et rows shared)
//   (#mb : ematrix et shared cols)
//   (#mc0 : ematrix et rows cols)
//   (rA : ematrix real rows shared)
//   (rB : ematrix real shared cols)
//   (rC : ematrix real rows cols)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> ma) **
//     on gpu_loc (gB |-> mb)
//   requires
//     pure (size_req (rows / tile) (shared / tile) (cols / tile) tile) **
//     pure (ma %~ rA /\ mb %~ rB /\ mc0 %~ rC /\
//           alpha %~ alpha_r /\ beta %~ beta_r) **
//     on gpu_loc (gC |-> mc0)
//   ensures
//     exists* (mc' : ematrix et rows cols).
//       on gpu_loc (gC |-> mc') **
//       pure (mc' %~ MS.gemm (alpha_r) (beta_r) rC rA rB)
// {
//   M.gpu_matrix_pts_to_ref_located gA;
//   M.gpu_matrix_pts_to_ref_located gB;
//   M.gpu_matrix_pts_to_ref_located gC;

//   // Defining it like this fails to extract, must be inlined below.
//   // let comb = MS.lincomb alpha beta;
//   let comb_r : (real -> real -> real) =
//     (MS.lincomb (alpha_r) (beta_r));

//   mmcomb_gpu_tiled_approx
//     mmcomb_gpu_approx
//     tile
//     #et #_ #_
//     (MS.lincomb alpha beta) comb_r
//     #rows #shared #cols
//     #(repA _ _) #(repB _ _) #(repC _ _)
//     #(cA.map _ _) #(cB.map _ _) #(cC.map _ _)
//     gA gB gC
//     rA rB rC;

//   ()
// }

// inline_for_extraction noextract
// let specialize_as_matmul_to_type_and_reprs_cpu
//   (#size_req : _)
//   (mmcomb_gpu : matmulcomb_gpu_ty size_req)
//   (et : Type0) {| scalar et |}
//   (rA rB rC : mrepr)
//   {| crepr rA, crepr rB, crepr rC |}
//   : fixed_repr_matmul_cpu_ty et size_req rA rB rC
//   = cpu_wrap_matmul et rA rB rC mmcomb_gpu

// inline_for_extraction noextract
// fn specialize_tiled_approx_cpu
//   (#size_req : tiled_size_req_t)
//   (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
//   (tile : valid_tile)
//   (et : Type0) {| scalar et |} {| real_like et |}
//   (repA repB repC : mrepr)
//   {| cA : crepr repA, cB : crepr repB, cC : crepr repC |}
//   (#rows #shared #cols : szp)
//   (a : vec et)
//   (b : vec et)
//   (#sa : erased (seq et){ len sa == rows * shared })
//   (#sb : erased (seq et){ len sb == shared * cols })
//   (rA : ematrix real rows shared)
//   (rB : ematrix real shared cols)
//   norewrite
//   preserves
//     cpu **
//     a |-> sa **
//     b |-> sb
//   requires
//     pure (size_req (rows / tile) (shared / tile) (cols / tile) tile) **
//     pure (SZ.fits (rows * cols)) **
//     pure (from_seq (repA rows shared) sa %~ rA /\
//           from_seq (repB shared cols) sb %~ rB)
//   returns
//     c : vec et
//   ensures
//     exists* (sc : seq et).
//       c |-> sc **
//       pure (len sc == rows * cols /\
//             from_seq (repC rows cols) sc %~ MS.matmul rA rB)
// {
//   Pulse.Lib.Vec.pts_to_len a;
//   Pulse.Lib.Vec.pts_to_len b;

//   let lA = repA rows shared;
//   let lB = repB shared cols;
//   let lC = repC rows cols;

//   let gA = M.gpu_matrix_alloc0 #et _ _ lA;
//   let gB = M.gpu_matrix_alloc0 #et _ _ lB;
//   let gC = M.gpu_matrix_alloc0 #et _ _ lC;

//   M.gpu_matrix_from_array gA a;
//   M.gpu_matrix_from_array gB b;

//   with vc. assert on gpu_loc (gC |-> vc);

//   lemma_to_real_matrix_approximates vc;
//   let rC = to_real_matrix vc;

//   mmcomb_gpu_tiled_approx
//     mmcomb_gpu_approx
//     tile
//     #et #_ #_
//     MS.comb2 MS.comb2
//     #rows #shared #cols
//     #lA #lB #lC
//     #(cA.map _ _) #(cB.map _ _) #(cC.map _ _)
//     gA gB gC
//     rA rB rC;

//   (* mmcomb comb2 rC rA rB == matmul rA rB by matmul_is_gemm *)

//   with eC'. assert on gpu_loc (gC |-> eC');

//   let c = Pulse.Lib.Vec.alloc #et zero (SZ.mul rows cols);
//   M.gpu_matrix_to_array c gC;

//   M.gpu_matrix_free gA;
//   M.gpu_matrix_free gB;
//   M.gpu_matrix_free gC;

//   c
// }
