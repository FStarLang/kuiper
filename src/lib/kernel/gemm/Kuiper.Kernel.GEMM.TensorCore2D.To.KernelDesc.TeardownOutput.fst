module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownOutput

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownHelpers
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownWarp
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownBlock

ghost
fn gather_output
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_approximates gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))) **
    pure (SZ.fits ((rm m n).ulen))
  ensures
    exists* (eD : chest2 et_cd m n).
      gD |-> eD ** pure (eD %~ MS.mmcomb comb_r rC rA rB)
{
  forevery_map
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        output_lane_approximates gD bm bn tm tn wm wn bid tid
          (ematrix_subtile
            (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
              bm bn (bid / (n / bn)) (bid % (n / bn)))
            (wm * tm) (wn * tn)
            ((tid / warp_size) / (bn / (wn * tn)))
            ((tid / warp_size) % (bn / (wn * tn)))))
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    fn bid {
      forevery_ext
        (fun (tid : natlt nthr) ->
          output_lane_approximates gD bm bn tm tn wm wn bid tid
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              ((tid / warp_size) / (bn / (wn * tn)))
              ((tid / warp_size) % (bn / (wn * tn)))))
        (fun (tid : natlt nthr) ->
          output_lane_approximates gD bm bn tm tn wm wn bid
            ((tid / warp_size) * warp_size + tid % warp_size)
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              ((tid / warp_size) / (bn / (wn * tn)))
              ((tid / warp_size) % (bn / (wn * tn)))));
      forevery_factor' nthr (nthr / warp_size) warp_size
        (fun wid lane ->
          output_lane_approximates gD bm bn tm tn wm wn bid
            (wid * warp_size + lane)
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))));
      forevery_map
        (fun (wid : natlt (nthr / warp_size)) ->
          forall+ (lane : natlt warp_size).
            output_lane_approximates gD bm bn tm tn wm wn bid
              (wid * warp_size + lane)
              (ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn)))))
        (fun (wid : natlt (nthr / warp_size)) ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn) wid |-> eWarp **
            pure (eWarp %~
              ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn)))))
        fn wid {
          gather_warp gD bm bn bk tm tn tk wm wn
            nblk nthr bid wid
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn))));
        };
      gather_block gD bm bn bk tm tn tk wm wn
        nblk nthr bid
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn (bid / (n / bn)) (bid % (n / bn)));
    };
  forevery_map
    #(natlt nblk)
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          ((bid / (n / bn)) * (n / bn) + bid % (n / bn)) |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    fn bid {
      FStar.Math.Lemmas.euclidean_division_definition bid (n / bn);
      rewrite each bid as
        ((bid / (n / bn)) * (n / bn) + bid % (n / bn));
    };
  forevery_factor' nblk (m / bm) (n / bn)
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc));
  forevery_map_2
    #(natlt (m / bm)) #(natlt (n / bn))
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc))
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        array2_subtile gD (SZ.v bm) (SZ.v bn) br bc |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc))
    fn br bc {
      assert pure ((br * (n / bn) + bc) / (n / bn) == br);
      assert pure ((br * (n / bn) + bc) % (n / bn) == bc);
      rewrite each block_tile gD (SZ.v bm) (SZ.v bn)
        (br * (n / bn) + bc)
      as array2_subtile gD (SZ.v bm) (SZ.v bn) br bc;
    };
  array2_untile_approximates gD (SZ.v bm) (SZ.v bn)
    (MS.mmcomb comb_r rC rA rB);
}
