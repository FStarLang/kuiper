module Kuiper.Kernel.SDPA.Naive

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.Bijection
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Bijection
open Kuiper.Float.Casts

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util

open Kuiper.Spec.Attention
open Kuiper.Kernel.BatchedGEMM
open Kuiper.Kernel.RowSoftmax

let fold_chest4_slice_page
  (#et: Type0)
  (#d0 #d1 #d2 #d3 : nat)
  (m : chest4 et d0 d1 d2 d3)
  (i : natlt d0)
  (j : natlt d1)
  : Lemma (
      slice_page #et #(d0 * d1) #d2 #d3
        (fold_chest #et #4 #(d0 @| d1 @| d2 @| d3 @| INil) m)
        (i * d1 + j) ==
      slice_page4 m i j) =
  let ij : natlt (d0 * d1) = i * d1 + j in
  FStar.Math.Lemmas.lemma_div_plus j i d1;
  FStar.Math.Lemmas.lemma_mod_plus j i d1;
  FStar.Math.Lemmas.small_div j d1;
  FStar.Math.Lemmas.small_mod j d1;
  introduce forall (idx : abs (d2 @| d3 @| INil)).
    acc (slice_page #et #(d0 * d1) #d2 #d3
      (fold_chest #et #4 #(d0 @| d1 @| d2 @| d3 @| INil) m) ij) idx ==
    acc (slice_page4 m i j) idx
  with (
    assert (unfold_index #4 #(d0 @| d1 @| d2 @| d3 @| INil) (ij, idx) ==
      (i, (j, idx)))
  );
  Kuiper.Chest.lemma_equal_intro
    (slice_page #et #(d0 * d1) #d2 #d3
      (fold_chest #et #4 #(d0 @| d1 @| d2 @| d3 @| INil) m) ij)
    (slice_page4 m i j);
  Kuiper.Chest.ext
    (slice_page #et #(d0 * d1) #d2 #d3
      (fold_chest #et #4 #(d0 @| d1 @| d2 @| d3 @| INil) m) ij)
    (slice_page4 m i j)

let fold_chest3_row
  (#et : Type0)
  (#d0 #d1 #d2 : nat)
  (m : chest3 et d0 d1 d2)
  (i : natlt d0)
  (j : natlt d1)
  : Lemma (
      chest2_row #et #(d0 * d1) #d2
        (fold_chest #et #3 #(d0 @| d1 @| d2 @| INil) m)
        (i * d1 + j)
      ==
      chest2_row #et #d1 #d2 (slice_page m i) j) =
  let ij : natlt (d0 * d1) = i * d1 + j in
  FStar.Math.Lemmas.lemma_div_plus j i d1;
  FStar.Math.Lemmas.lemma_mod_plus j i d1;
  FStar.Math.Lemmas.small_div j d1;
  FStar.Math.Lemmas.small_mod j d1;
  introduce forall (idx : abs (d2 @| INil)).
    acc
      (chest2_row #et #(d0 * d1) #d2
        (fold_chest #et #3 #(d0 @| d1 @| d2 @| INil) m) ij)
      idx
    ==
    acc (chest2_row #et #d1 #d2 (slice_page m i) j) idx
  with (
    assert (
      unfold_index #3 #(d0 @| d1 @| d2 @| INil) (ij, idx)
      == (i, (j, idx)))
  );
  Kuiper.Chest.lemma_equal_intro
    (chest2_row #et #(d0 * d1) #d2
      (fold_chest #et #3 #(d0 @| d1 @| d2 @| INil) m) ij)
    (chest2_row #et #d1 #d2 (slice_page m i) j);
  Kuiper.Chest.ext
    (chest2_row #et #(d0 * d1) #d2
      (fold_chest #et #3 #(d0 @| d1 @| d2 @| INil) m) ij)
    (chest2_row #et #d1 #d2 (slice_page m i) j)

let sdpa_scores_spec 
  (#n #h : szp)
  (#l #s : szp)
  (#e: szp)
  (rQ : chest4 real n h l e)
  (rK : chest4 real n h e s)
  (rS : chest4 real n h l s)
  (rscale : real) = 
  mk4 (fun i j -> acc2 (attn_scores
          (slice_page4 (rQ) i j)
          (slice_page4 (rK) i j)
          (slice_page4 (rS) i j)
          rscale))

let sdpa_probs_spec
  (#n #h #l #s : nat)
  (scores : chest4 real n h l s)
  : chest4 real n h l s =
  mk4 (fun i j -> acc2 (row_softmax_real (slice_page4 scores i j)))

let sdpa_scores_spec_slice
  (#n #h : szp)
  (#l #s #e : szp)
  (rQ : chest4 real n h l e)
  (rK : chest4 real n h e s)
  (rS : chest4 real n h l s)
  (rscale : real)
  (i : natlt n)
  (j : natlt h)
  : Lemma (
      slice_page4 (sdpa_scores_spec rQ rK rS rscale) i j
      ==
      attn_scores
        (slice_page4 rQ i j)
        (slice_page4 rK i j)
        (slice_page4 rS i j)
        rscale) =
  let lhs = slice_page4 (sdpa_scores_spec rQ rK rS rscale) i j in
  let rhs = attn_scores
    (slice_page4 rQ i j)
    (slice_page4 rK i j)
    (slice_page4 rS i j)
    rscale in
  introduce forall (idx : abs (l @| s @| INil)).
    acc lhs idx == acc rhs idx
  with ();
  Kuiper.Chest.lemma_equal_intro lhs rhs;
  Kuiper.Chest.ext lhs rhs

let sdpa_probs_spec_slice
  (#n #h #l #s : nat)
  (scores : chest4 real n h l s)
  (i : natlt n)
  (j : natlt h)
  : Lemma (
      slice_page4 (sdpa_probs_spec scores) i j
      ==
      row_softmax_real (slice_page4 scores i j)) =
  let lhs = slice_page4 (sdpa_probs_spec scores) i j in
  let rhs = row_softmax_real (slice_page4 scores i j) in
  introduce forall (idx : abs (l @| s @| INil)).
    acc lhs idx == acc rhs idx
  with ();
  Kuiper.Chest.lemma_equal_intro lhs rhs;
  Kuiper.Chest.ext lhs rhs

let sdpa_softmax_aux
  (#n #h #l #s : pos)
  (scores : chest4 real n h l s)
  : Lemma (
      unfold_chest
        (unfold_chest
          (row_softmax_real (fold_chest (fold_chest scores))))
      == sdpa_probs_spec scores) =
  let lhs =
    unfold_chest
      (unfold_chest
        (row_softmax_real (fold_chest (fold_chest scores)))) in
  let rhs = sdpa_probs_spec scores in
  introduce forall (idx : abs (n @| h @| l @| s @| INil)).
    acc lhs idx == acc rhs idx
  with (
    let (i, (j, (k, (_, ())))) = idx in
    let ij : natlt (n * h) = i * h + j in
    FStar.Math.Lemmas.lemma_div_plus j i h;
    FStar.Math.Lemmas.lemma_mod_plus j i h;
    FStar.Math.Lemmas.small_div j h;
    FStar.Math.Lemmas.small_mod j h;
    FStar.Math.Lemmas.lemma_div_plus k ij l;
    FStar.Math.Lemmas.lemma_mod_plus k ij l;
    FStar.Math.Lemmas.small_div k l;
    FStar.Math.Lemmas.small_mod k l;
    fold_chest4_slice_page scores i j;
    fold_chest3_row #real #(n * h) #l #s (fold_chest scores) ij k
  );
  Kuiper.Chest.lemma_equal_intro lhs rhs;
  Kuiper.Chest.ext lhs rhs

let sdpa_softmax_folded_aux
  (#n #h #l #s : pos)
  (scores : chest4 real n h l s)
  : Lemma (
      unfold_chest
        (row_softmax_real #((n * h) * l) #s
          (fold_chest (fold_chest scores)))
      == fold_chest (sdpa_probs_spec scores)) =
  let probs = sdpa_probs_spec scores in
  let softmax2 =
    row_softmax_real #((n * h) * l) #s
      (fold_chest (fold_chest scores)) in
  let probs3 = unfold_chest softmax2 in
  sdpa_softmax_aux scores;
  assert (unfold_chest probs3 == probs);
  assert (fold_chest (unfold_chest probs3) == fold_chest probs);
  ()

let sdpa_output_aux
  (#n #h #l #s #e #ev : szp)
  (rQ : chest4 real n h l e)
  (rKT : chest4 real n h e s)
  (rV : chest4 real n h s ev)
  (rbias : chest4 real n h l s)
  (scale : real)
  : Lemma (
      unfold_chest
        (MS.batched_matmul #real #_ #(n * h) #l #s #ev
          (fold_chest
            (sdpa_probs_spec
              (sdpa_scores_spec rQ rKT rbias scale)))
          (fold_chest rV))
      ==
      attention_real_batched rQ rKT rV rbias scale) =
  let probs =
    sdpa_probs_spec (sdpa_scores_spec rQ rKT rbias scale) in
  let lhs =
    unfold_chest
      (MS.batched_matmul #real #_ #(n * h) #l #s #ev
        (fold_chest probs) (fold_chest rV)) in
  let rhs = attention_real_batched rQ rKT rV rbias scale in
  introduce forall (idx : abs (n @| h @| l @| ev @| INil)).
    acc lhs idx == acc rhs idx
  with (
    let (i, (j, (k, (t, ())))) = idx in
    let ij : natlt (n * h) = i * h + j in
    FStar.Math.Lemmas.lemma_div_plus j i h;
    FStar.Math.Lemmas.lemma_mod_plus j i h;
    FStar.Math.Lemmas.small_div j h;
    FStar.Math.Lemmas.small_mod j h;
    fold_chest4_slice_page probs i j;
    fold_chest4_slice_page rV i j;
    sdpa_probs_spec_slice (sdpa_scores_spec rQ rKT rbias scale) i j;
    sdpa_scores_spec_slice rQ rKT rbias scale i j;
    assert (
      acc lhs idx ==
      acc2
        (MS.matmul
          (slice_page4 probs i j)
          (slice_page4 rV i j))
        k t);
    assert (
      slice_page4 probs i j ==
      row_softmax_real
        (attn_scores
          (slice_page4 rQ i j)
          (slice_page4 rKT i j)
          (slice_page4 rbias i j)
          scale));
    assert (
      acc rhs idx ==
      acc2
        (MS.matmul
          (row_softmax_real
            (attn_scores
              (slice_page4 rQ i j)
              (slice_page4 rKT i j)
              (slice_page4 rbias i j)
              scale))
          (slice_page4 rV i j))
        k t)
  );
  Kuiper.Chest.lemma_equal_intro lhs rhs;
  Kuiper.Chest.ext lhs rhs

let sdpa_naive_aux
  (#n #h : szp { SZ.fits (n * h) })
  (#l #s : szp)
  (#e: szp)
  (rQ : chest4 real n h l e)
  (rK : chest4 real n h e s)
  (rS : chest4 real n h l s)
  (rscale : real):
  Lemma (ensures 
    (sdpa_scores_spec rQ rK rS rscale)
    == 
    (unfold_chest (MS.bmmcomb (fun bias_qk score -> bias_qk +. (score *. rscale)) 
      #(SZ.v (n *^ h)) #l #e #s (fold_chest rS) (fold_chest rQ) (fold_chest rK)))) =
  let lhs = sdpa_scores_spec rQ rK rS rscale in
  let rhs = unfold_chest (MS.bmmcomb
    (fun bias_qk score -> bias_qk +. (score *. rscale))
    #(SZ.v (n *^ h)) #l #e #s
    (fold_chest rS) (fold_chest rQ) (fold_chest rK)) in
  introduce forall (idx : abs (n @| h @| l @| s @| INil)).
    acc lhs idx == acc rhs idx
  with (
    let (i, (j, (_, (_, ())))) = idx in
    let ij : natlt (n * h) = i * h + j in
    FStar.Math.Lemmas.lemma_div_plus j i h;
    FStar.Math.Lemmas.lemma_mod_plus j i h;
    FStar.Math.Lemmas.small_div j h;
    FStar.Math.Lemmas.small_mod j h;
    fold_chest4_slice_page rS i j;
    fold_chest4_slice_page rQ i j;
    fold_chest4_slice_page rK i j
  );
  Kuiper.Chest.lemma_equal_intro lhs rhs;
  Kuiper.Chest.ext lhs rhs

let scaled_add_approx
  (#et: Type0) {| scalar et, real_like et |}
  (scale : et)
  : Lemma (
      approx2
        (fun x y -> x `add` (y `mul` scale))
        (fun x y -> x +. (y *. to_real scale))) =
  let aux (x y : et) (rx ry : real)
    : Lemma
        (requires x %~ rx /\ y %~ ry)
        (ensures (x `add` (y `mul` scale)) %~ (rx +. (ry *. to_real scale))) =
    to_real_ok scale;
    a_mul y scale ry (to_real scale);
    a_add x (y `mul` scale) rx (ry *. to_real scale)
  in
  Classical.forall_intro_4
    (fun x y rx ry -> Classical.move_requires (aux x y rx) ry)

let comb2_approx
  (#et : Type0) {| scalar et, real_like et |}
  : Lemma (approx2 (MS.comb2 #et) (MS.comb2 #real)) =
  let aux (x y : et) (rx ry : real)
    : Lemma
        (requires x %~ rx /\ y %~ ry)
        (ensures MS.comb2 x y %~ MS.comb2 rx ry) =
    ()
  in
  Classical.forall_intro_4
    (fun x y rx ry -> Classical.move_requires (aux x y rx) ry)

// TODO: this shouldn't require so much boilerplate...

#push-options "--split_queries always"
let transpose4_2 (d0 d1 d2 d3 : nat) : 
  (abs (d0 @| d1 @| d2 @| d3 @| INil) =~ abs (d0 @| d1 @| d3 @| d2 @| INil)) =
{
  ff = (fun (i,(j,(k,(l,())))) -> (i,(j,(l,(k,())))));
  gg = (fun (i,(j,(k,(l,())))) -> (i,(j,(l,(k,())))));
  // weird that ez doesn't take care of it...
  ff_gg = (fun x -> (
    let (i,(j,(k,(l,())))) = x in
    assert ((fun (i,(j,(k,(l,())))) -> (i,(j,(l,(k,()))))) x) == (i,(j,(l,(k,()))))
  ));
  gg_ff = (fun x -> (
    let (i,(j,(k,(l,())))) = x in
    assert ((fun (i,(j,(k,(l,())))) -> (i,(j,(l,(k,()))))) x) == (i,(j,(l,(k,()))))
  ));
}

/// Concrete index mapping for [transpose4_2], mirroring its ghost inverse [gg]
/// (which swaps the last two dimensions). Carries no proof obligations beyond the
/// pointwise swap, so [transpose4_2_conc_correct] is definitional.
inline_for_extraction noextract
let transpose4_2_conc (d0 d1 d2 d3 : nat)
  (x : conc (d0 @| d1 @| d3 @| d2 @| INil))
  : conc (d0 @| d1 @| d2 @| d3 @| INil)
  = let (i,(j,(k,(l,())))) = x in (i,(j,(l,(k,()))))

let transpose4_2_conc_correct (d0 d1 d2 d3 : nat)
  (x : conc (d0 @| d1 @| d3 @| d2 @| INil))
  : (up (transpose4_2_conc d0 d1 d2 d3 x) == (transpose4_2 d0 d1 d2 d3).gg (up x))
  = ()

/// Extractable [ctlayout] for the K-transpose relayout: instantiates [ctlayout_bij]
/// with the concrete swap above.
inline_for_extraction noextract
let ctlayout_bij_transpose
  (#d0 #d1 #d2 #d3 : szp)
  (lin : tlayout (d0 @| d1 @| d2 @| d3 @| INil)) {| c : ctlayout lin |}
  : ctlayout (tlayout_bij (transpose4_2 (SZ.v d0) (SZ.v d1) (SZ.v d2) (SZ.v d3)) lin)
  = ctlayout_bij (transpose4_2 (SZ.v d0) (SZ.v d1) (SZ.v d2) (SZ.v d3))
      (transpose4_2_conc (SZ.v d0) (SZ.v d1) (SZ.v d2) (SZ.v d3))
      (transpose4_2_conc_correct (SZ.v d0) (SZ.v d1) (SZ.v d2) (SZ.v d3))
      lin

#pop-options

inline_for_extraction noextract
fn sdpa_naive
  (#et: Type0) {| floating et, real_like et, floating_real_like et |}
  (n h : szp)
  (l s : szp)
  (e ev : szp)
  (#lQ: tlayout    (n @| h @| l @| e @| INil)  { is_full lQ }) // needed for tlayout_bij for now.
  (#lK: tlayout    (n @| h @| s @| e @| INil)  { is_full lK })
  (#lV: tlayout    (n @| h @| s @| ev @| INil) { is_full lV })
  (#lbias: tlayout (n @| h @| l @| s @| INil)  { is_full lbias })
  {| ctlayout lQ, ctlayout lK, ctlayout lV, ctlayout lbias |}
  (gQ    : tensor et lQ    { is_global gQ    })
  (gK    : tensor et lK    { is_global gK    })
  (gV    : tensor et lV    { is_global gV    })
  (gbias : tensor et lbias { is_global gbias })
  (out   : tensor et (l4_batched_row_major n h l ev) { is_global out })
  (scale : et)
  (#eQ : erased    (chest4 et n h l e))
  (#eK : erased    (chest4 et n h s e))
  (#eV : erased    (chest4 et n h s ev))
  (#ebias : erased (chest4 et n h l s))
  (#rKT : erased   (chest4 real n h e s))
  (#fQ #fK #fV : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gQ    |-> Frac fQ eQ) **
    on gpu_loc (gK    |-> Frac fK eK) **
    on gpu_loc (gV    |-> Frac fV eV)
  requires
    on gpu_loc (gbias |-> ebias) **
    on gpu_loc (live out) **
    pure (
      SZ.fits (l * ev) /\
      SZ.fits (n * h * l * e) /\
      SZ.fits (n * h * s * e)  /\
      SZ.fits (n * h * s * ev)  /\
      SZ.fits (n * h * l * ev)  /\ 
      SZ.fits (n * h * l * s)  /\
      SZ.fits (n * h * l) /\
      SZ.fits (h * l) /\
      (mk4 (fun i j k l -> acc4 eK i j l k)) %~ rKT /\
      l * s <= max_blocks * max_threads /\
      l * ev <= max_blocks * max_threads /\
      n * h * l <= max_blocks /\
      n * h * l * s <= max_blocks * max_threads
    )
  ensures
    // For simplicity, bias is used to hold the scores.
    (exists* (eS' : chest4 et n h l s).
      on gpu_loc (gbias |-> eS')) **
    (exists* (eO : chest4 et n h l ev).
      on gpu_loc (out |-> eO) **
      pure (
        eO %~ attention_real_batched
          (to_real_chest eQ)
          rKT
          (to_real_chest eV)
          (to_real_chest ebias)
          (to_real scale))) 
{
  
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gQ);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gK);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gV);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gbias);

  let rQ    = to_real_chest eQ;
  let rV    = to_real_chest eV;
  let rbias = to_real_chest ebias;
  
  // Transpose K via ghost
  let f_transpose = transpose4_2 n h s e; 
  let gKT = tensor_apply_bij_ro_located f_transpose #lK gK #fK #eK;
  with eKT. assert on gpu_loc (gKT |-> Frac fK eKT);
  assert pure (eKT %~ rKT);

  let gQf = tensor_fold_ro_located gQ #fQ;
  let gKf = tensor_fold_ro_located gKT #fK;
  let gSf = tensor_fold_st_located gbias;

  let eQf = fold_chest eQ; let rQf = fold_chest rQ;
  let eKf = fold_chest eKT; let rKf = fold_chest rKT;
  let eSf = fold_chest ebias; let rSf = fold_chest rbias;

  // Compute Q * K^T + bias, scaled by scale, into bias
  bmmcomb_gpu_exact #et (fun bias_qk score -> bias_qk `add` (score `mul` scale))
    (n*^h) l e s #_ #_ #_
    #(ctlayout_fold_outer lQ)
    #(ctlayout_fold_outer
        (tlayout_bij f_transpose lK)
        #(ctlayout_bij_transpose lK))
    #(ctlayout_fold_outer lbias)
    gQf gKf gSf;

  with eSf'. assert on gpu_loc (gSf |-> eSf');

  Pulse.Lib.Trade.elim_trade (on gpu_loc (gKf |-> Frac fK eKf)) _;
  Pulse.Lib.Trade.elim_trade (on gpu_loc (gQf |-> Frac fQ eQf)) _;

  lemma_to_real_chest_approximates eQ;
  lemma_to_real_chest_approximates ebias;
  assert pure (eQf %~ rQf);
  assert pure (eKf %~ rKf);
  assert pure (eSf %~ rSf);
  scaled_add_approx scale;
  MU.bmmcomb_approx_real
    #et
    (fun bias_qk score -> bias_qk `add` (score `mul` scale))
    (fun bias_qk score -> bias_qk +. (score *. to_real scale))
    #(n * h) #l #e #s
    eSf eQf eKf rSf rQf rKf;
  assert pure (eSf' %~ (MS.bmmcomb
    (fun bias_qk score -> bias_qk +. (score *. to_real scale))
    #(n * h) #l #e #s rSf rQf rKf));
  let rScoresf = MS.bmmcomb
    (fun bias_qk score -> bias_qk +. (score *. to_real scale))
    #(n * h) #l #e #s rSf rQf rKf;
  let eScores = unfold_chest eSf';
  let rScores = unfold_chest rScoresf;
  sdpa_naive_aux rQ rKT rbias (to_real scale);
  assert pure (rScores == sdpa_scores_spec rQ rKT rbias (to_real scale));
  assert pure (unfold_chest eSf' %~
    sdpa_scores_spec rQ rKT rbias (to_real scale));

  let scores = sdpa_scores_spec rQ rKT rbias (to_real scale);
  let scoresf = fold_chest scores;
  assert pure (fold_chest (unfold_chest rScoresf) == scoresf);
  assert pure (rScoresf == scoresf);
  let scoresff = fold_chest scoresf;
  let eSff = fold_chest eSf';
  assert pure (eSff %~ scoresff);

  // Compute softmax of scores into bias
  let gSff = tensor_fold_st_located gSf;
  row_softmax_gpu #et
    (n*^h*^l) s max_threads #_
    #(ctlayout_fold_outer
        (tlayout_fold_outer lbias)
        #(ctlayout_fold_outer lbias))
    gSff #eSff scoresff;
  with eSff'. assert on gpu_loc (gSff |-> eSff');

  let eProbsf = unfold_chest eSff';
  elim_forall (eSff' <:
    chest (fold_outer (fold_outer (n @| h @| l @| s @| INil))) et);
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc (gSff |-> eSff'))
    (on gpu_loc (gSf |-> eProbsf));
  assert on gpu_loc (gSf |-> eProbsf);

  let rProbsf0 =
    unfold_chest (row_softmax_real #((n * h) * l) #s scoresff);
  let probs = sdpa_probs_spec scores;
  let rProbsf = fold_chest probs;
  sdpa_softmax_folded_aux scores;
  assert pure (eProbsf %~ rProbsf);

  let gVf = tensor_fold_ro_located gV #fV;
  let eVf = fold_chest eV;
  let rVf = fold_chest rV;

  with eO0. assert on gpu_loc (out |-> eO0);
  let out_f = tensor_fold_st_located out;
  let eO0f = fold_chest eO0;

  // Final batched matmul to compute output: probs * V
  bmmcomb_gpu_exact #et (MS.comb2 #et)
    (n*^h) l s ev #_ #_ #_
    #(ctlayout_fold_outer lbias)
    #(ctlayout_fold_outer lV)
    #(ctlayout_fold_outer (l4_batched_row_major n h l ev))
    gSf gVf out_f;
  with eOf'. assert on gpu_loc (out_f |-> eOf');

  Pulse.Lib.Trade.elim_trade (on gpu_loc (gVf |-> Frac fV eVf)) _;

  let eO = unfold_chest eOf';
  elim_forall (eOf' <:
    chest (fold_outer (n @| h @| l @| ev @| INil)) et);
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc (out_f |-> eOf'))
    (on gpu_loc (out |-> eO));
  assert on gpu_loc (out |-> eO);

  elim_forall (eProbsf <:
    chest (fold_outer (n @| h @| l @| s @| INil)) et);
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc (gSf |-> eProbsf))
    (on gpu_loc (gbias |-> unfold_chest eProbsf));
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc (gKT |-> Frac fK eKT)) _;

  lemma_to_real_chest_approximates eV;
  lemma_to_real_chest_approximates eO0;
  assert pure (eVf %~ rVf);
  let rO0f = fold_chest (to_real_chest eO0);
  assert pure (eO0f %~ rO0f);
  comb2_approx #et #_ #_;
  MU.bmmcomb_approx_real
    #et
    (MS.comb2 #et)
    (MS.comb2 #real)
    #(n * h) #l #s #ev
    eO0f eProbsf eVf rO0f rProbsf rVf;

  let rOf0 =
    MS.bmmcomb (MS.comb2 #real) #(n * h) #l #s #ev
      rO0f rProbsf rVf;
  let rOf =
    MS.batched_matmul #real #_ #(n * h) #l #s #ev rProbsf rVf;
  MS.bmatmul_is_bgemm #real #_ #(n * h) #l #s #ev
    rO0f rProbsf rVf;
  assert pure (rOf0 == rOf);
  assert pure (unfold_chest eOf' %~ unfold_chest rOf);
  sdpa_output_aux rQ rKT rV rbias (to_real scale);
  assert pure (eO %~ attention_real_batched rQ rKT rV rbias (to_real scale));
}