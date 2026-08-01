module Kuiper.Kernel.GEMM.TensorCore2D.To.BaseSendable

#lang-pulse

open Kuiper

#set-options "--fuel 1 --ifuel 1 --z3rlimit 15 --split_queries no"

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Array.Vectorized { has_vec_cpy }
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

instance lane_cells_sendable
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l { is_global a })
  (lane : natlt warp_size)
  : is_send_across gpu_of (live_lane_cells a lane)
= solve

let output_lane_live_sendable_to
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : szp)
  (gD : array2 et (rm m n) { is_global gD })
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of (
      output_lane_live gD bm bn tm tn wm wn bid tid)
=
  let wid :
    natlt (bm / (wm * tm) * (bn / (wn * tn))) =
      tid / warp_size in
  let lane : natlt warp_size = tid % warp_size in
  assert_norm (
    reveal (block_tile_idx_rows
      (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) bid)
      == bid / (SZ.v n / SZ.v bn));
  assert_norm (
    reveal (block_tile_idx_cols
      (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) bid)
      == bid % (SZ.v n / SZ.v bn));
  assert_norm (
    reveal (warp_tile_idx_rows
      (SZ.v bm) (SZ.v bn) (SZ.v wm * SZ.v tm) (SZ.v wn * SZ.v tn) wid)
      == wid / (SZ.v bn / (SZ.v wn * SZ.v tn)));
  assert_norm (
    reveal (warp_tile_idx_cols
      (SZ.v bm) (SZ.v bn) (SZ.v wm * SZ.v tm) (SZ.v wn * SZ.v tn) wid)
      == wid % (SZ.v bn / (SZ.v wn * SZ.v tn)));
  assert (forall (mi : natlt wm) (nj : natlt wn).
    is_global (output_fragment gD bm bn tm tn wm wn bid wid mi nj));
  solve

let output_lane_approximates_sendable_to
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (#m #n #k : szp)
  (comb_r : binop real)
  (gD : array2 et (rm m n) { is_global gD })
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of (
      output_lane_approximates gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile
            (MS.mmcomb comb_r rC rA rB)
            bm bn
            (bid / (n / bn))
            (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))))
=
  FStar.Math.Lemmas.cancel_mul_div (SZ.v wm) (SZ.v tm);
  FStar.Math.Lemmas.cancel_mul_div (SZ.v wn) (SZ.v tn);
  let wid :
    natlt (bm / (wm * tm) * (bn / (wn * tn))) =
      tid / warp_size in
  let lane : natlt warp_size = tid % warp_size in
  assert_norm (
    reveal (block_tile_idx_rows
      (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) bid)
      == bid / (SZ.v n / SZ.v bn));
  assert_norm (
    reveal (block_tile_idx_cols
      (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) bid)
      == bid % (SZ.v n / SZ.v bn));
  assert_norm (
    reveal (warp_tile_idx_rows
      (SZ.v bm) (SZ.v bn) (SZ.v wm * SZ.v tm) (SZ.v wn * SZ.v tn) wid)
      == wid / (SZ.v bn / (SZ.v wn * SZ.v tn)));
  assert_norm (
    reveal (warp_tile_idx_cols
      (SZ.v bm) (SZ.v bn) (SZ.v wm * SZ.v tm) (SZ.v wn * SZ.v tn) wid)
      == wid % (SZ.v bn / (SZ.v wn * SZ.v tn)));
  assert (forall (mi : natlt wm) (nj : natlt wn).
    is_global (output_fragment gD bm bn tm tn wm wn bid wid mi nj));
  solve

let kpre1_sendable_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA { is_global gA })
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB { is_global gB })
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n) { is_global gC })
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n) { is_global gD })
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of (
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
=
  let output_send =
    output_lane_live_sendable_to gD
      bm bn bk tm tn tk wm wn #_ nblk nthr bid tid in
  let pA = gA |-> Frac (fA /. (nblk * nthr)) eA in
  let pB = gB |-> Frac (fB /. (nblk * nthr)) eB in
  let pC = gC |-> Frac (fC /. (nblk * nthr)) eC in
  let pOutput = output_lane_live gD bm bn tm tn wm wn bid tid in
  let pAlignedA = pure (aligned 16 (core gA)) in
  let pAlignedB = pure (aligned 16 (core gB)) in
  let pAlignedC = pure (aligned 16 (core gC)) in
  let pAlignedD = pure (aligned 16 (core gD)) in
  let pApproxA = pure (eA %~ rA) in
  let pApproxB = pure (eB %~ rB) in
  let pApproxC = pure (eC %~ rC) in
  let pPure =
    pAlignedA ** pAlignedB ** pAlignedC ** pAlignedD **
    pApproxA ** pApproxB ** pApproxC in
  let sendA : is_send_across block_of pA =
    send_across_if_send_across_gpu pA
      (is_send_across_global_tensor gA #(fA /. (nblk * nthr)) eA) in
  let sendB : is_send_across block_of pB =
    send_across_if_send_across_gpu pB
      (is_send_across_global_tensor gB #(fB /. (nblk * nthr)) eB) in
  let sendC : is_send_across block_of pC =
    send_across_if_send_across_gpu pC
      (is_send_across_global_tensor gC #(fC /. (nblk * nthr)) eC) in
  let sendAlignedA : is_send_across block_of pAlignedA =
    is_send_across_placeless pAlignedA
      #(placeless_pure (aligned 16 (core gA))) in
  let sendAlignedB : is_send_across block_of pAlignedB =
    is_send_across_placeless pAlignedB
      #(placeless_pure (aligned 16 (core gB))) in
  let sendAlignedC : is_send_across block_of pAlignedC =
    is_send_across_placeless pAlignedC
      #(placeless_pure (aligned 16 (core gC))) in
  let sendAlignedD : is_send_across block_of pAlignedD =
    is_send_across_placeless pAlignedD
      #(placeless_pure (aligned 16 (core gD))) in
  let sendApproxA : is_send_across block_of pApproxA =
    is_send_across_placeless pApproxA #(placeless_pure (eA %~ rA)) in
  let sendApproxB : is_send_across block_of pApproxB =
    is_send_across_placeless pApproxB #(placeless_pure (eB %~ rB)) in
  let sendApproxC : is_send_across block_of pApproxC =
    is_send_across_placeless pApproxC #(placeless_pure (eC %~ rC)) in
  let sendApproxBC =
    is_send_across_star pApproxB pApproxC #sendApproxB #sendApproxC in
  let sendApproxABC =
    is_send_across_star pApproxA (pApproxB ** pApproxC)
      #sendApproxA #sendApproxBC in
  let sendAlignedDPure =
    is_send_across_star pAlignedD
      (pApproxA ** pApproxB ** pApproxC)
      #sendAlignedD #sendApproxABC in
  let sendAlignedCPure =
    is_send_across_star pAlignedC
      (pAlignedD ** pApproxA ** pApproxB ** pApproxC)
      #sendAlignedC #sendAlignedDPure in
  let sendAlignedBPure =
    is_send_across_star pAlignedB
      (pAlignedC ** pAlignedD ** pApproxA ** pApproxB ** pApproxC)
      #sendAlignedB #sendAlignedCPure in
  let sendPure =
    is_send_across_star pAlignedA
      (pAlignedB ** pAlignedC ** pAlignedD **
       pApproxA ** pApproxB ** pApproxC)
      #sendAlignedA #sendAlignedBPure in
  let sendOutputPure =
    is_send_across_star pOutput pPure #output_send #sendPure in
  let sendCOutput =
    is_send_across_star pC (pOutput ** pPure) #sendC #sendOutputPure in
  let sendBC =
    is_send_across_star pB (pC ** pOutput ** pPure) #sendB #sendCOutput in
  is_send_across_star pA (pB ** pC ** pOutput ** pPure) #sendA #sendBC

let kpost1_sendable_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA { is_global gA })
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB { is_global gB })
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n) { is_global gC })
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n) { is_global gD })
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of (
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
=
  let output_send =
    output_lane_approximates_sendable_to comb_r gD
      bm bn bk tm tn tk wm wn #_ nblk nthr rA rB rC bid tid in
  FStar.Math.Lemmas.cancel_mul_div (SZ.v wm) (SZ.v tm);
  FStar.Math.Lemmas.cancel_mul_div (SZ.v wn) (SZ.v tn);
  let wid :
    natlt (bm / (wm * tm) * (bn / (wn * tn))) =
      tid / warp_size in
  assert_norm (
    reveal (block_tile_idx_rows
      (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) bid)
      == bid / (SZ.v n / SZ.v bn));
  assert_norm (
    reveal (block_tile_idx_cols
      (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) bid)
      == bid % (SZ.v n / SZ.v bn));
  assert_norm (
    reveal (warp_tile_idx_rows
      (SZ.v bm) (SZ.v bn) (SZ.v wm * SZ.v tm) (SZ.v wn * SZ.v tn) wid)
      == wid / (SZ.v bn / (SZ.v wn * SZ.v tn)));
  assert_norm (
    reveal (warp_tile_idx_cols
      (SZ.v bm) (SZ.v bn) (SZ.v wm * SZ.v tm) (SZ.v wn * SZ.v tn) wid)
      == wid % (SZ.v bn / (SZ.v wn * SZ.v tn)));
  let pA = gA |-> Frac (fA /. (nblk * nthr)) eA in
  let pB = gB |-> Frac (fB /. (nblk * nthr)) eB in
  let pC = gC |-> Frac (fC /. (nblk * nthr)) eC in
  let pOutput =
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (ematrix_subtile
        (ematrix_subtile
          (MS.mmcomb comb_r rC rA rB)
          bm bn
          (bid / (n / bn))
          (bid % (n / bn)))
        (wm * tm) (wn * tn)
        ((tid / warp_size) / (bn / (wn * tn)))
        ((tid / warp_size) % (bn / (wn * tn)))) in
  let sendA : is_send_across block_of pA =
    send_across_if_send_across_gpu pA
      (is_send_across_global_tensor gA #(fA /. (nblk * nthr)) eA) in
  let sendB : is_send_across block_of pB =
    send_across_if_send_across_gpu pB
      (is_send_across_global_tensor gB #(fB /. (nblk * nthr)) eB) in
  let sendC : is_send_across block_of pC =
    send_across_if_send_across_gpu pC
      (is_send_across_global_tensor gC #(fC /. (nblk * nthr)) eC) in
  let sendCOutput =
    is_send_across_star pC pOutput #sendC #output_send in
  let sendBC =
    is_send_across_star pB (pC ** pOutput) #sendB #sendCOutput in
  is_send_across_star pA (pB ** pC ** pOutput) #sendA #sendBC
