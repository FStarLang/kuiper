module Kuiper.GEMM.BlockTiling2D.Inst

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs { row_major as rm, col_major as cm }

module M = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module P = Kuiper.Poly.GEMM.BlockTiling2D
module SZ = Kuiper.SizeT

#set-options "--z3rlimit 60"

inline_for_extraction noextract
fn spec
  (bm bn bk : szp)
  // (slA : full_mlayout bm bk)
  // (slB : full_mlayout bk bn)
  // {| clayout slA, clayout slB |}
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn /\ (bm/tm * bn/tn <= max_threads)})
  (#_ : squash (sz_fits (bm*bk + (bm/tm * (bn/tn)))))
  (#_ : squash (sz_fits (bk*bn + (bm/tm * (bn/tn)))))
  (et : Type0) {| scalar et, has_vec_cpy et |}
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (alpha beta : et)
  (#rows #shared #cols : szp)
  (gA : M.gpu_matrix et (rm rows shared) { M.is_global_matrix gA })
  (#fA : perm)
  (gB : M.gpu_matrix et (rm shared cols) { M.is_global_matrix gB })
  (#fB : perm)
  (gC : M.gpu_matrix et (rm rows cols) { M.is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (M.core gA)) **
    pure (aligned 16 (M.core gB)) **
    pure (rows * cols <= max_blocks) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gemm alpha beta eC eA eB)
{
  M.gpu_matrix_pts_to_ref_located gA;
  M.gpu_matrix_pts_to_ref_located gB;
  M.gpu_matrix_pts_to_ref_located gC;

  // TODO: add dynamic assert for this.
    // pure (aligned 16 (M.core gA)) **
    // pure (aligned 16 (M.core gB)) **

  dassert (bm >^ 0sz);
  dassert (bn >^ 0sz);
  dassert (bk >^ 0sz);
  dassert (tm >^ 0sz);
  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);
  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;

  lemma_divides_trans (chunk et) bk shared;
  assert pure (chunk et /?+ shared);
  assert pure (aligned_strided_row_major (chunk et)
                (Kuiper.Matrix.Reprs.strided_row_major_base #(SZ.v rows) #(SZ.v shared)));

  lemma_divides_trans (chunk et) bn cols;
  assert pure (chunk et /?+ cols);
  assert pure (aligned_strided_row_major (chunk et)
                (Kuiper.Matrix.Reprs.strided_row_major_base #(SZ.v shared) #(SZ.v cols)));

  // Prove rows/bm * (cols/bn) <= max_blocks from rows * cols <= max_blocks
  // Since bm >= 1 and bn >= 1: rows/bm <= rows and cols/bn <= cols
  FStar.Math.Lemmas.lemma_div_mod (SZ.v rows) (SZ.v bm);
  FStar.Math.Lemmas.nat_over_pos_is_nat (SZ.v rows) (SZ.v bm);
  FStar.Math.Lemmas.lemma_div_mod (SZ.v cols) (SZ.v bn);
  FStar.Math.Lemmas.nat_over_pos_is_nat (SZ.v cols) (SZ.v bn);
  assert pure (SZ.v rows / SZ.v bm <= SZ.v rows);
  assert pure (SZ.v cols / SZ.v bn <= SZ.v cols);
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v cols / SZ.v bn) (SZ.v rows / SZ.v bm) (SZ.v rows);
  FStar.Math.Lemmas.lemma_mult_le_left (SZ.v rows) (SZ.v cols / SZ.v bn) (SZ.v cols);
  assert pure (SZ.v rows / SZ.v bm * (SZ.v cols / SZ.v bn) <= SZ.v rows * SZ.v cols);

  P.mmcomb_gpu_exact
    (fun o n -> mul beta o `add` mul alpha n)
    #rows #shared #cols
    gA #eA gB #eB gC #eC
    bm bn bk
    tm tn
    (cm _ _) (rm _ _) //slA slB
    #_ #_;

  ()
}
