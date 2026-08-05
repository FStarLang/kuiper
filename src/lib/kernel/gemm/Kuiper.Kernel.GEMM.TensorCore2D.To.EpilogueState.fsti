module Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorRO { vtlayout_of_tlayout }
module RO = Kuiper.TensorRO
open Kuiper.TensorCore
open Pulse.Lib.Array

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

inline_for_extraction noextract
let sz_succ (x : SZ.t { SZ.fits (x + 1) }) : SZ.t = x +^ 1sz

let fragarrayAcc_approximates
  (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wm wn : nat)
  ([@@@mkey] arr : array (fragment et FragAcc tm tn tk FragLAcc))
  (rm : chest2 real (wm * tm) (wn * tn))
  : slprop
= exists* (em : seq (chest2 et tm tn)).
    arr |-> em **
    pure (
      Pulse.Lib.Array.length arr == wm * wn /\
      Seq.length em == wm * wn /\
      forall (i : natlt wm) (j : natlt wn).
        Seq.index em (i * wn + j) %~
          ematrix_subtile rm tm tn i j)

inline_for_extraction noextract
noeq type epilogue_dims (m n : szp) = {
  bm : szp;
  bn : szp;
  bk : szp;
  tm : szp;
  tn : szp;
  tk : szp;
  wm : szp;
  wn : szp;
  nthr : nthr:szp {
    constraints bm bn bk tm tn tk wm wn /\
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size };
  tiles_div : squash (bm /?+ m /\ bn /?+ n);
  shared_fits : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn));
  output_fits : squash (SZ.fits (m * n));
  tile_count_fits : squash (SZ.fits (wm * wn));
  scratch_fits : squash (SZ.fits ((nthr / warp_size) * tm * tn));
  fragment_fits : squash (SZ.fits (tm * tn + warp_size));
}

let epilogue_frame
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd,
     scalar et_acc, real_like et_acc |}
  (#m #n : szp)
  (#lC : RO.vlayout2 m n)
  (gC : RO.roarray2 et_cd lC)
  (fC : perm)
  (eC : chest2 et_cd m n)
  (rC : chest2 real m n)
  (bm bn bk tm tn tk wm wn nthr : szp{
    constraints bm bn bk tm tn tk wm wn /\
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc) {
    Pulse.Lib.Array.length accFrags == wm * wn})
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (tid : szlt nthr)
  : slprop
= pure (eC %~ rC) **
  gpu **
  thread_id nthr tid **
  gC |-> Frac fC eC **
  fragarrayAcc_approximates wm wn accFrags rAcc **
  scratch_tile_live bm bn bk tm tn nthr sh tid

let epilogue_warp_input
  (#m #n : nat)
  (rC : chest2 real m n)
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (tid : natlt (bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
  : chest2 real (wm * tm) (wn * tn)
= ematrix_subtile
    (ematrix_subtile rC bm bn
      (bid / (n / bn)) (bid % (n / bn)))
    (wm * tm) (wn * tn)
    ((tid / warp_size) / (bn / (wn * tn)))
    ((tid / warp_size) % (bn / (wn * tn)))

let output_fragment_post
  (#et : Type0) {| scalar et, real_like et, has_vec_cpy et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : natlt warp_size)
  (rD : chest2 real (wm * tm) (wn * tn))
  (idx : natlt (wm * wn))
  : slprop
= exists* (eD : chest2 et tm tn).
    own_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (idx / wn) (idx % wn))
      eD lane **
    pure (eD %~ ematrix_subtile rD tm tn (idx / wn) (idx % wn))

let if_else_ (b : bool) (p q : slprop) : slprop =
  if_ b p ** if_ (not b) q

let output_fragment_state_at
  (#et : Type0) {| scalar et, real_like et, has_vec_cpy et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : natlt warp_size)
  (rD : chest2 real (wm * tm) (wn * tn))
  (done : natle (wm * wn))
  (idx : natlt (wm * wn))
  : slprop
= if_else_ (idx < done)
    (output_fragment_post
      gD bm bn tm tn wm wn bid wid lane rD idx)
    (live_lane_cells
      (output_fragment gD bm bn tm tn wm wn
        bid wid (idx / wn) (idx % wn))
      lane)

let output_epilogue_state
  (#et : Type0) {| scalar et, real_like et, has_vec_cpy et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : natlt warp_size)
  (rD : chest2 real (wm * tm) (wn * tn))
  (done : natle (wm * wn))
  : slprop
= forall+ (idx : natlt (wm * wn)).
    output_fragment_state_at
      gD bm bn tm tn wm wn bid wid lane rD done idx
