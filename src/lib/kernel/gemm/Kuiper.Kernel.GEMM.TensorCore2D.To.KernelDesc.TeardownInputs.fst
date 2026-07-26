module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownInputs

#lang-pulse

open Kuiper
#set-options "--split_queries no"
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

ghost
fn gather_kernel_outputs
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    teardown_inputs_pre comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn nblk nthr
      fA fB fC rA rB rC
  ensures
    teardown_inputs_post comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn nblk nthr
      fA fB fC rA rB rC
{
  unfold teardown_inputs_pre comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
  forevery_map_2
    #(natlt nblk)
    #(natlt nthr)
    (fun bid tid ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))))
    fn bid tid {
      unfold kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid;
    };
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC)
    (fun bid tid ->
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));

  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_gather_n gA (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_gather_n gB (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  tensor_gather_n gC (nblk * nthr);
  fold teardown_inputs_post comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
}
