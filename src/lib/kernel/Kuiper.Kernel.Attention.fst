module Kuiper.Kernel.Attention

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Bijection

module EM4 = Kuiper.EMatrix4
module EM3 = Kuiper.EMatrix3
module CH = Kuiper.Chest
module SZ = Kuiper.SizeT

open Kuiper.Spec.Attention
open Kuiper.Kernel.BatchedGEMM


#push-options "--split_queries always"
let transpose4_2 (#d0 #d1 #d2 #d3 : nat) : 
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

//#push-options "--print_implicits"
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
      SZ.fits (n * h * l * e) /\
      SZ.fits (n * h * s * e)  /\
      SZ.fits (n * h * s * ev)  /\
      SZ.fits (n * h * l * s)  /\
      SZ.fits (n * h * l) /\
      (EM4.mkM (fun i j k l -> EM4.macc eK i j l k)) %~ rKT /\
      l * s <= max_blocks * max_threads /\
      n * h * l <= max_blocks
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
  let rbias = EM3.to_real_matrix ebias;

  // Transpose K via ghost
  let f_transpose = transpose4_2 #n #h #s #e; 
  let gKT: tensor et (tlayout_bij f_transpose lK) = from_array (tlayout_bij f_transpose lK) (core gK);
  assert rewrites_to gKT (from_array (tlayout_bij f_transpose lK) (core gK));
  map_loc gpu_loc (fun () -> tensor_apply_bij f_transpose gK #fK);
  let eKT = CH.mk (n @| h @| e @| s @| INil) (fun i -> CH.acc eK (i <~| f_transpose));
  assert on gpu_loc (gKT |-> Frac fK eKT);
  assert pure (eKT %~ rKT);

  // Fold 2 batch dimensions of K^T, Q, V into one (N * H)
  let gKTf = from_array (tlayout_fold_outer (tlayout_bij f_transpose lK)) (core gKT);
  let gQf = from_array (tlayout_fold_outer lQ) (core gQ);
  let gVf = from_array (tlayout_fold_outer lV) (core gV);
  let gbiasf = from_array (tlayout_fold_outer lbias) (core gbias);
  assert rewrites_to gKTf (from_array (tlayout_fold_outer (tlayout_bij f_transpose lK)) (core gKT));
  assert rewrites_to gQf (from_array (tlayout_fold_outer lQ) (core gQ));
  assert rewrites_to gVf (from_array (tlayout_fold_outer lV) (core gV));
  assert rewrites_to gbiasf (from_array (tlayout_fold_outer lbias) (core gbias));
  
  map_loc gpu_loc (fun () -> tensor_fold_outer gQ #fQ);
  map_loc gpu_loc (fun () -> tensor_fold_outer gV #fV);
  map_loc gpu_loc (fun () -> tensor_fold_outer gKT #fK);
  map_loc gpu_loc (fun () -> tensor_fold_outer gbias #fbias);

  let eQf = fold_chest eQ; 
  let eVf = fold_chest eV; 
  let eKTf = fold_chest eKT; 
  let ebiasf = fold_chest ebias;
  assert on gpu_loc (gQf |-> Frac fQ eQf) ** pure (eQf %~ fold_chest rQ);
  assert on gpu_loc (gVf |-> Frac fV eVf) ** pure (eVf %~ fold_chest rV);
  assert on gpu_loc (gKTf |-> Frac fK eKTf) ** pure (eKTf %~ fold_chest rKT);
  assert on gpu_loc (gbiasf |-> Frac fbias ebiasf) ** pure (ebiasf %~ fold_chest rbias);
  
  let gS = alloc0 #et (n *^ h *^ l *^ s) (l3_batched_row_major (n*^h) l s);
  with eS. assert on gpu_loc (gS |-> eS);

  bmmcomb_gpu_exact #et (fun bias_qk score -> (bias_qk +. score) *. scale) 
    (n*^h) l e s gQf gKTf gS;

  admit ();
}