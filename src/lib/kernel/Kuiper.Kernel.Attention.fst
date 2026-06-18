module Kuiper.Kernel.Attention

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.EMatrix
open Kuiper.Bijection

module EM = Kuiper.EMatrix
module EM4 = Kuiper.EMatrix4
module EM3 = Kuiper.EMatrix3
module A2 = Kuiper.Array2
module A1 = Kuiper.Array1
module CH = Kuiper.Chest
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module T = FStar.Tactics.V2
module SM = Kuiper.Spec.Softmax
module GU = Kuiper.Kernel.GEMM.Util

open Kuiper.Spec.Attention
open Kuiper.Kernel.BatchedGEMM
open Kuiper.Kernel.RowSoftmax
open Kuiper.Kernel.HReduce.Block
open Kuiper.Kernel.Map
open Kuiper.ForEvery

open Kuiper.Kernel.Attention.Helpers

module TL = Kuiper.Tensor.Layout

//#push-options "--print_implicits"
#push-options "--split_queries always --z3rlimit 20"
inline_for_extraction noextract
fn scaled_dot_product_efficient_attention
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (n h : szp)
  (l s : szp)
  (e ev : szp)
  (#lQ: tlayout    (n @| h @| l @| e @| INil) { is_full lQ }) // needed for tlayout_bij for now.
  (#lK: tlayout    (n @| h @| s @| e @| INil) { is_full lK })
  (#lV: tlayout    (n @| h @| s @| ev @| INil) { is_full lV })
  (#lbias: tlayout (n @| h @| l @| s @| INil) { is_full lbias })
  {| ctlayout lQ, ctlayout lK, ctlayout lV, ctlayout lbias |}
  (gQ    : tensor et lQ    { is_global gQ    })
  (gK    : tensor et lK    { is_global gK    })
  (gV    : tensor et lV    { is_global gV    })
  (gbias : tensor et lbias { is_global gbias })
  (scale : et)
  (#eQ : erased    (EM4.t et n h l e))
  (#eK : erased    (EM4.t et n h s e))
  (#eV : erased    (EM4.t et n h s ev))
  (#ebias : erased (EM4.t et n h l s))
  (#rKT : erased   (EM4.t real n h e s))
  (#fQ #fK #fV #fbias : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gQ    |-> Frac fQ eQ) **
    on gpu_loc (gK    |-> Frac fK eK) **
    on gpu_loc (gV    |-> Frac fV eV) **
    on gpu_loc (gbias |-> Frac fbias ebias)
  requires
    pure (
      SZ.fits (l * ev) /\
      SZ.fits (n * h * l * e) /\
      SZ.fits (n * h * s * e)  /\
      SZ.fits (n * h * s * ev)  /\
      SZ.fits (n * h * l * ev)  /\ 
      SZ.fits (n * h * l * s)  /\
      SZ.fits (n * h * l) /\
      SZ.fits (h * l) /\
      (EM4.mkM (fun i j k l -> EM4.macc eK i j l k)) %~ rKT /\
      l * s <= max_blocks * max_threads /\
      l * ev <= max_blocks * max_threads /\
      n * h * l <= max_blocks /\
      n * h * l * s <= max_blocks * max_threads
    )
  returns
    // TODO: polymorphic out & LSE layout
    out : tensor et (l4_batched_row_major n h l ev) & 
          tensor et (l3_batched_row_major n h l)
  ensures
    (exists* (eO : EM4.t et n h l ev) (eLSE : EM3.t et n h l).
      on gpu_loc (fst out |-> eO) **
      on gpu_loc (snd out |-> eLSE) **
      pure (
        let out_spec, lse_spec = attention_real_batched
            (EM4.to_real_matrix eQ)
            rKT
            (EM4.to_real_matrix eV)
            (EM4.to_real_matrix ebias)
            (to_real scale) in
          eO %~ out_spec /\ eLSE %~ lse_spec)) **
    pure (is_global (fst out) /\ is_global (snd out)) {

  map_loc gpu_loc (fun () -> tensor_pts_to_ref gQ);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gK);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gV);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gbias);
  
  let rQ = EM4.to_real_matrix eQ;
  let rV = EM4.to_real_matrix eV;
  let rbias = EM4.to_real_matrix ebias;

  // Transpose K via ghost
  let f_transpose = transpose4_2 #n #h #s #e; 
  let gKT: tensor et (tlayout_bij f_transpose lK) = from_array (tlayout_bij f_transpose lK) (core gK);
  assert rewrites_to gKT (from_array (tlayout_bij f_transpose lK) (core gK));
  map_loc gpu_loc (fun () -> tensor_apply_bij f_transpose gK #fK);
  let eKT = CH.mk (n @| h @| e @| s @| INil) (fun i -> CH.acc eK (i <~| f_transpose));
  assert on gpu_loc (gKT |-> Frac fK eKT);
  assert pure (eKT %~ rKT);

  // Fold 2 batch dimensions of K^T, Q, V, bias into one (N * H)
  let gKTf, eKTf, rKTf = fold4_to_3 gKT #fK #eKT #rKT;
  let gQf, eQf, rQf = fold4_to_3 gQ #fQ #eQ #rQ;
  let gVf, eVf, rVf = fold4_to_3 gV #fV #eV #rV;
  let gbiasf, ebiasf, rbiasf = fold4_to_3 gbias #fbias #ebias #rbias;

  let gS = alloc0 #et (n *^ h *^ l *^ s) (l3_batched_row_major (n*^h) l s);
  with eS. assert on gpu_loc (gS |-> eS);

  // ===== Load bias into gS (layout-aware device copy): gS pre-matmul content := ebiasf =====
  // size facts needed to form [fold_bij_l3 (n*^h) l s].
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v n * SZ.v h) 1 (SZ.v l);
  assert pure (SZ.fits (SZ.v n * SZ.v h));
  assert pure (SZ.v (n *^ h) == SZ.v n * SZ.v h);
  assert pure (SZ.fits (SZ.v (n *^ h) * SZ.v l * SZ.v s));
  FStar.Math.Lemmas.paren_mul_right (SZ.v (n *^ h)) (SZ.v l) (SZ.v s);
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v l * SZ.v s) 1 (SZ.v (n *^ h));
  assert pure (SZ.fits (SZ.v l * SZ.v s));

  let lbias3 : tlayout ((n *^ h) @| l @| s @| INil) = tlayout_fold_outer lbias;
  let flat = fold_bij_l3 (n *^ h) l s;
  let lb : tlayout (((n *^ h) *^ l *^ s) @| INil) = tlayout_bij flat lbias3;

  full_layout_size (l1_forward (n *^ h *^ l *^ s));
  full_layout_size (l3_batched_row_major (n*^h) l s);
  full_layout_size lbias3;
  full_layout_size lb;
  assert pure (is_full lb);
  assert pure (is_full (l3_batched_row_major (n*^h) l s));

  let ct_lbias3 : ctlayout lbias3 = ctlayout_bij fold_bij lbias;
  let ct_lb : ctlayout lb = ctlayout_bij flat lbias3 #ct_lbias3;

  // ---- aBias : A1 view of gbiasf whose logical sequence is the row-major flatten of ebiasf ----
  let aBias : tensor et lb = from_array lb (core gbiasf);
  assert rewrites_to aBias (from_array lb (core gbiasf));
  FStar.Classical.forall_intro (bij_self_imap flat lbias3);
  gpu_relayout_to lb (bij_sym flat) () gbiasf aBias #fbias #ebiasf;
  let cb : CH.t (((n *^ h) *^ l *^ s) @| INil) et =
    CH.mk (((n *^ h) *^ l *^ s) @| INil) (fun idx -> CH.acc ebiasf ((bij_sym flat).ff idx));
  // bridge the (relayouted) tensor [aBias] to a fresh A1 handle [aBias_a1] over the
  // same physical core; its logical lseq is [chest_to_seq1 cb] (the flatten of ebiasf).
  let aBias_a1 : A1.t et lb = A1.from_array lb (core aBias);
  assert rewrites_to aBias_a1 (A1.from_array lb (core aBias));
  tensor1_to_a1 lb aBias aBias_a1 #fbias #cb;
  sb_flatten_char (n *^ h) l s ebiasf;

  // ---- aS : A1 view of gS (pre-copy content irrelevant). Built with [A1.from_array]
  //      so the handle matches the [A1.raise'] resource for the copy. ----
  map_loc gpu_loc (fun () -> tensor_concr gS);
  map_loc gpu_loc (fun () -> A1.raise' (l1_forward (n *^ h *^ l *^ s)) (core gS));
  let aS = A1.from_array (l1_forward (n *^ h *^ l *^ s)) (core gS);
  assert rewrites_to aS (A1.from_array (l1_forward (n *^ h *^ l *^ s)) (core gS));

  // ---- the only new executable kernel call: identity copy aBias_a1 -> aS ----
  // [sb] (the logical row-major flattening of [ebiasf]) is referenced only in
  // ghost/spec positions as [chest_to_seq1 cb]; we never bind it to a runtime let
  // (it is GTot), which keeps Pulse's ghost discipline happy.
  map_gpu_notinplace #et #et (fun (x:et) -> x) (n *^ h *^ l *^ s) #lb #ct_lb aBias_a1 aS;
  lseq_map_id (chest_to_seq1 cb);
  rewrite (on gpu_loc (aS |-> (lseq_map (fun (x:et) -> x) (chest_to_seq1 cb) <: lseq et (n *^ h *^ l *^ s))))
       as (on gpu_loc (aS |-> chest_to_seq1 cb));

  // ---- restore gbiasf from aBias_a1 (preserving its fractional permission fbias) ----
  // [a1_to_tensor1] re-establishes the tensor resource directly on [aBias] (which
  // equals [from_array lb (A1.core aBias_a1)]), over the same physical core.
  a1_to_tensor1 lb aBias_a1 aBias #fbias #(chest_to_seq1 cb);
  let gbiasf2 : tensor et lbias3 = from_array lbias3 (core aBias);
  assert rewrites_to gbiasf2 (from_array lbias3 (core aBias));
  FStar.Classical.forall_intro (untranspose_imap_hyp flat lbias3);
  gpu_relayout_to lbias3 flat () aBias gbiasf2 #fbias #(seq1_to_chest (chest_to_seq1 cb));
  post_relayout_chest_eq (n *^ h) l s flat (chest_to_seq1 cb) ebiasf;
  assert pure (gbiasf2 == gbiasf);
  rewrite (on gpu_loc (gbiasf2 |-> Frac fbias
            (CH.mk ((n *^ h) @| l @| s @| INil) (fun idx -> CH.acc (seq1_to_chest (chest_to_seq1 cb)) (flat.ff idx)))))
       as (on gpu_loc (gbiasf |-> Frac fbias ebiasf));

  // ---- raise aS back to the gS tensor; its content is now ebiasf ----
  // Mirror [array1_to_3d_u]'s internal structure: run the A1->3D bridge inside a
  // [map_loc] block (where the produced resource is a plain [exists*], not under
  // [on]), open the witness, prove it equals [ebiasf], and rewrite onto [gS].
  sb_macc_char (n *^ h) l s ebiasf;
  map_loc gpu_loc
    #(aS |-> Frac 1.0R (chest_to_seq1 cb))
    #(gS |-> ebiasf)
    fn () {
      let g' = array1_to_3d (n *^ h) l s aS #1.0R #(chest_to_seq1 cb);
      with s3. assert (g' |-> Frac 1.0R (s3 <: EM3.t et (n *^ h) l s));
      EM3.lemma_equal_intro s3 ebiasf;
      CH.ext s3 ebiasf;
      rewrite each s3 as ebiasf;
      rewrite each g' as gS;
    };

  let lKT : tlayout (n @| h @| e @| s @| INil) = tlayout_bij f_transpose lK;
  let ctlKT : ctlayout lKT = ctlayout_bij f_transpose lK;
  bmmcomb_gpu_exact #et (fun bias_qk score -> (bias_qk `add` score) `mul` scale) 
    (n*^h) l e s #_ #_ #_ #(ctlayout_bij fold_bij lQ) #(ctlayout_bij fold_bij lKT) #_ gQf gKTf gS;
  with eS'. assert on gpu_loc (gS |-> (eS' <: EM3.t et (n *^ h) l s));

  let rS': EM3.t real (n *^ h) l s = MS.bmmcomb 
    (fun bias_qk score -> (bias_qk +. score) *. (to_real scale))
    rbiasf rQf rKTf;
  approx2_addmul_scale #et scale;
  assert pure (approx2
    (fun (bias_qk:et) (score:et) -> (bias_qk `add` score) `mul` scale)
    (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. (to_real scale)));
  bmmcomb_approx
    (fun bias_qk score -> (bias_qk `add` score) `mul` scale)
    (fun bias_qk score -> (bias_qk +. score) *. (to_real scale))
    ebiasf rbiasf eQf rQf eKTf rKTf;
  assert pure ((eS' <: EM3.t et (n *^ h) l s) %~ rS');

  assert pure (is_full_array (core gS));
  let gSf, eSf, rSf = fold3_to_2 gS #_ #eS' #rS';

  // TODO: could fuse some `f` with the sums in this kernel, for the log step
  let sums = row_softmax_gpu_with_sum (n *^ h *^ l) s 
    #(tlayout_fold_outer (l3_batched_row_major (n*^h) l s))
    #(ctlayout_bij fold_bij (l3_batched_row_major (n*^h) l s))
    gSf rSf;
  with esums. assert on gpu_loc (sums |-> esums);
  assert pure (esums %~ Seq.init_ghost (n *^ h *^ l) (fun i -> rsum (lseq_map exp (ematrix_row rSf i))));
  map_gpu flog (n *^ h *^ l) sums;
  with esums'. assert on gpu_loc (sums |-> esums');
  assert pure (esums' == lseq_map flog esums);
  // The log-sum-exp [%~] fact needed by [lse_approx_all] is derived inside
  // [lse_approx_all_from_sums] (see its call below), from [esums' == lseq_map flog esums],
  // the positivity of the row-sums-of-exp, and [esums %~ Seq.init_ghost ... rsum (...)] (above).
  // It is NOT elaborated here, because the term [log (rsum (lseq_map exp ...))] trips an
  // internal Z3 4.13.3 lar_solver assertion violation in this (heavy) context.
  assert pure (SZ.v s > 0);
  assert pure (forall (i: natlt (n *^ h *^ l)). (Seq.length (ematrix_row rSf i) > 0));
  assert pure (forall (i: natlt (n *^ h *^ l)). (rsum (lseq_map exp (ematrix_row rSf i)) >. 0.0R)); 
  assert pure (SZ.v (h *^ l) > 0);
  assert pure (SZ.v l > 0);
  // sums is LSE now.

  let gS, eS, rS = unfold2_to_3 gSf #_ #_ #(row_softmax_real rSf);
  
  assert pure (SZ.fits (n * h * l));
  assert pure (SZ.v (n *^ h *^ l *^ s) == SZ.v (n *^ h) * SZ.v l * SZ.v s);
  assert pure (SZ.fits (n * h * l * s));
  assert pure (SZ.fits (n * h * l));
  assert pure (SZ.fits (n * h));
  assert pure (SZ.v (n *^ h *^ l *^ s) == sizeof (n @| h @| l @| s @| INil));
  assert pure (SZ.v (n *^ h *^ l *^ s) > 0);
  let gO = alloc0 #et (n *^ h *^ l *^ ev) (l3_batched_row_major (n*^h) l ev);
  with eO. assert on gpu_loc (gO |-> eO);

  bmmcomb_gpu_exact #et MS.comb2 
    (n*^h) l s ev #_ #_ #_ #_ #(ctlayout_bij fold_bij lV) #_ gS gVf gO;

  // is_full_array (core gS) now provided by unfold2_to_3's strengthened ensures
  free gS;

  // ===== restore the preserved 4D inputs (preserves clause) =====
  restore_fold4 gQ gQf #fQ #_ #(reveal eQ);
  restore_fold4 gV gVf #fV #_ #(reveal eV);
  restore_fold4 gbias gbiasf #fbias #_ #(reveal ebias);
  // K: undo the outer fold, then undo the transpose
  restore_fold4 gKT gKTf #fK #_ #eKT;
  FStar.Classical.forall_intro (untranspose_imap_hyp f_transpose lK);
  FStar.Classical.forall_intro (kt_chest_eq n h s e (reveal eK) f_transpose);
  gpu_relayout_to lK f_transpose () gKT gK #fK #eKT;
  CH.lemma_equal_intro (reveal eK)
    (CH.mk (n @| h @| s @| e @| INil) (fun idx -> CH.acc eKT (f_transpose.ff idx)));
  CH.ext (reveal eK)
    (CH.mk (n @| h @| s @| e @| INil) (fun idx -> CH.acc eKT (f_transpose.ff idx)));
  rewrite (on gpu_loc (gK |-> Frac fK (CH.mk (n @| h @| s @| e @| INil)
                                         (fun idx -> CH.acc eKT (f_transpose.ff idx)))))
       as (on gpu_loc (gK |-> Frac fK (reveal eK)));

  // ===== output tensor: relayout gO (3D (n*h) l ev) to 4D (n h l ev) =====
  let eOreal : EM3.t et (n*^h) l ev = MS.bmmcomb MS.comb2 eO eS eVf;
  let gO4 : tensor et (l4_batched_row_major n h l ev) =
    from_array (l4_batched_row_major n h l ev) (core gO);
  assert rewrites_to gO4 (from_array (l4_batched_row_major n h l ev) (core gO));
  imap_hyp_l4_all n h l ev;
  gpu_relayout_to (l4_batched_row_major n h l ev) (fold_bij_l4 n h l ev)
    (ulen_eq_l3l4 n h l ev) gO gO4 #_ #eOreal;
  let eO4 : EM4.t et n h l ev =
    CH.mk (n @| h @| l @| ev @| INil) (fun idx -> CH.acc eOreal ((fold_bij_l4 n h l ev).ff idx));
  // functional correctness of the output tensor
  relayout4_macc_all #et n h l ev eOreal;
  out_approx_all #et n h l s e ev eO4 eS eVf rQ rKT rV rbias (to_real scale);

  // ===== LSE tensor: flatten array1 -> 3D (n h l) =====
  FStar.Math.Lemmas.paren_mul_right (SZ.v n) (SZ.v h) (SZ.v l);
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v h * SZ.v l) 1 (SZ.v n);
  assert pure (SZ.fits (SZ.v h * SZ.v l));
  let gLSE : tensor et (l3_batched_row_major n h l) =
    from_array (l3_batched_row_major n h l) (A1.core sums);
  assert rewrites_to gLSE (from_array (l3_batched_row_major n h l) (A1.core sums));
  array1_to_3d_u n h l sums gLSE #_ #esums';
  with s3. assert on gpu_loc (gLSE |-> (s3 <: EM3.t et n h l));
  lse_approx_all_from_sums #et n h l s e ev s3 esums esums' rQ rKT rV rbias (to_real scale);

  // bridge the per-page specs (from out_approx_all / lse_approx_all) to the
  // batched spec [attention_real_batched] used by the interface.
  attention_real_batched_unfold #(SZ.v n) #(SZ.v h) #(SZ.v l) #(SZ.v s) #(SZ.v e) #(SZ.v ev)
    (EM4.to_real_matrix eQ) rKT (EM4.to_real_matrix eV) (EM4.to_real_matrix ebias) (to_real scale);

  // is_global facts
  assert pure (is_global gO4);
  assert pure (is_global gLSE);

  (gO4, gLSE)
}


#pop-options
