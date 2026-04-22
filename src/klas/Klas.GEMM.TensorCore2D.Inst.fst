module Klas.GEMM.TensorCore2D.Inst
#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
open Kuiper.TensorCore
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D

#push-options "--split_queries always --z3rlimit 40" // very slow without splitting? flaky nevertheless

inline_for_extraction noextract
fn spec
  // specialize
  (et_ab et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab /?+ bn))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * warp_sz) -1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * warp_sz) -1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * (SZ.v warp_sz)) <= max_threads))

  // do not specialize
  (rows shared cols : szp)
  (gA : gpu_matrix et_ab (row_major rows shared) { is_global_matrix gA })
  (gB : gpu_matrix et_ab (row_major shared cols) { is_global_matrix gB })
  (gC : gpu_matrix et_c (row_major rows cols) { is_global_matrix gC })
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#eA : ematrix et_ab rows shared)
  (#eB : ematrix et_ab shared cols)
  (#eC : ematrix et_c rows cols)
  (#fA #fB : perm)
  // non of these are are checked because the functions is only
  //  partially applied
  preserves
    cpu **
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    on gpu_loc (gC |-> eC)
  ensures
    exists* eC'.
      on gpu_loc (gC |-> eC') ** pure (eC' %~ MS.matmul (to_real_matrix eA) (to_real_matrix eB))
{
  gpu_matrix_pts_to_ref_located gA;
  gpu_matrix_pts_to_ref_located gB;
  gpu_matrix_pts_to_ref_located gC;

  // TODO dassert for alignment of A/B

  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dassert (bk %^ tk = 0sz);

  // preconditions checcked at runtime
  // TODO should be checked at runtime but has ghost effect:
  //  dguard (SZ.lte (rows *^ cols) (SZ.uint_to_t max_blocks));
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);

  // Pretty bad that we have to call this explicitly...
  lemma_divides_chain (wm * tm) bm rows;
  lemma_divides_chain (wn * tn) bn cols;

  let nblk = rows/^bm *^ (cols/^bn);
  let nthr = bm/^(wm*^tm) *^ (bn/^(wn*^tn)) *^ warp_sz;

  assert pure ((rows/bm) * (cols/bn) == nblk);
  assert pure ((rows/bm) * (cols/bn) <= max_blocks);
  dassert (nblk <=^ SZ.uint_to_t 2097152); // Inlining max_blocks.. not great.
  assert pure (nblk <= max_blocks);

  dassert ((bm *^ bk) %^ (chunk et_ab *^ nthr) = 0sz);
  dassert ((bk *^ bn) %^ (chunk et_ab *^ nthr) = 0sz);

  lemma_divides_trans (chunk et_ab) bk shared;
  assert pure (chunk et_ab /?+ shared);
  assert pure (aligned_strided_row_major (chunk et_ab)
                (Kuiper.Matrix.Reprs.strided_row_major_base #(SZ.v rows) #(SZ.v shared)));

  lemma_divides_trans (chunk et_ab) bn cols;
  assert pure (chunk et_ab /?+ cols);
  assert pure (aligned_strided_row_major (chunk et_ab)
                (Kuiper.Matrix.Reprs.strided_row_major_base #(SZ.v shared) #(SZ.v cols)));

  (* Instead of threading through approximations, we here pick
     real matrices that are (trivially) approximated
     by the input ematrices, and call the function. This is mostly
     to show that the approximation precondition is not a serious
     requirement. *)
  let rA = to_real_matrix eA;
  let rB = to_real_matrix eB;
  let rC = to_real_matrix eC;
  #set-options "--fuel 0 --ifuel 0 --z3refresh" {
  launch_sync (
    mk_kernel gA #eA gB #eB gC #_ #eC bm bn bk tm tn tk wm wn #_ #_ #_ #_ #_ #_ #_ #_ #fA #fB nblk nthr rA rB rC ()
  )};

  ()
}
#pop-options
