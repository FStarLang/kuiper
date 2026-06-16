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

module EM4 = Kuiper.EMatrix4
module EM3 = Kuiper.EMatrix3
module A2 = Kuiper.Array2
module CH = Kuiper.Chest
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Spec.Attention
open Kuiper.Kernel.BatchedGEMM
open Kuiper.Kernel.RowSoftmax
open Kuiper.Kernel.HReduce.Block
open Kuiper.Kernel.Map

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

inline_for_extraction noextract
fn fold4_to_3 
  (#et : Type0) {| floating et, real_like et |}
  (#d0 #d1 #d2 #d3: szp)
  (#lA : tlayout (d0 @| d1 @| d2 @| d3 @| INil) { is_full lA })
  {| ctlayout lA |}
  (gA : tensor et lA { is_global gA })
  (#fA : perm)
  (#eA : erased (EM4.t et d0 d1 d2 d3))
  (#rA : erased (EM4.t real d0 d1 d2 d3) { eA %~ rA })
preserves
  cpu
requires
  on gpu_loc (gA |-> Frac fA eA)
returns
  out : (
    tensor et (tlayout_fold_outer lA <: tlayout ((d0 *^ d1) @| d2 @| d3 @| INil)) &
    EM3.t et (d0 *^ d1) d2 d3 &
    EM3.t real (d0 *^ d1) d2 d3 
  ) 
ensures 
  on gpu_loc (out._1 |-> Frac fA (out._2)) **
  pure (is_global out._1 /\ (out._2 %~ out._3) /\ out._3 == fold_chest rA)
{
  let gAf = from_array (tlayout_fold_outer lA) (core gA);
  let eAf = fold_chest eA;
  let rAf = fold_chest rA;
  assert rewrites_to gAf (from_array (tlayout_fold_outer lA) (core gA));
  map_loc gpu_loc (fun () -> tensor_fold_outer gA #fA);
  return (gAf, eAf, rAf);
}

inline_for_extraction noextract
fn fold3_to_2 
  (#et : Type0) {| floating et, real_like et |}
  (#d0 #d1 #d2: szp)
  (#lA : tlayout (d0 @| d1 @| d2 @| INil) { is_full lA })
  {| ctlayout lA |}
  (gA : tensor et lA { is_global gA })
  (#fA : perm)
  (#eA : erased (EM3.t et d0 d1 d2))
  (#rA : erased (EM3.t real d0 d1 d2) { eA %~ rA })
preserves
  cpu
requires
  on gpu_loc (gA |-> Frac fA eA)
returns
  out : (
    tensor et (tlayout_fold_outer lA <: tlayout ((d0 *^ d1) @| d2 @| INil)) &
    ematrix et (d0 *^ d1) d2 &
    ematrix real (d0 *^ d1) d2 
  ) 
ensures 
  on gpu_loc (out._1 |-> Frac fA (out._2)) **
  pure (is_global out._1 /\ (out._2 %~ out._3) /\ out._3 == fold_chest rA)
{
  let gAf = from_array (tlayout_fold_outer lA) (core gA);
  let eAf = fold_chest eA;
  let rAf = fold_chest rA;
  assert rewrites_to gAf (from_array (tlayout_fold_outer lA) (core gA));
  map_loc gpu_loc (fun () -> tensor_fold_outer gA #fA);
  return (gAf, eAf, rAf);
}

inline_for_extraction noextract
fn unfold2_to_3 
  (#et : Type0) {| floating et, real_like et |}
  (#d0 #d1 #d2: szp)
  (#lA : tlayout (d0 @| d1 @| d2 @| INil) { is_full lA })
  {| ctlayout lA |}
  (gA : tensor et (tlayout_fold_outer lA <: tlayout ((d0 *^ d1) @| d2 @| INil)) { is_global gA })
  (#fA : perm)
  (#eA : erased (ematrix et (d0 *^ d1) d2))
  (#rA : erased (ematrix real (d0 *^ d1) d2) { eA %~ rA })
preserves
  cpu
requires
  on gpu_loc (gA |-> Frac fA eA)
returns
  out : (
    tensor et lA &
    EM3.t et d0 d1 d2 &
    EM3.t real d0 d1 d2 
  ) 
ensures 
  on gpu_loc (out._1 |-> Frac fA (out._2)) **
  pure (is_global out._1 /\ (out._2 %~ out._3) /\ out._3 == unfold_chest rA)
{
  let gAuf = from_array lA (core gA);
  let eAuf = unfold_chest #et #3 #(d0 @| d1 @| d2 @| INil) eA;
  let rAuf = unfold_chest #real #3 #(d0 @| d1 @| d2 @| INil) rA;
  assert rewrites_to gAuf (from_array lA (core gA));
  map_loc gpu_loc (fun () -> tensor_unfold_outer gA #fA);
  return (gAuf, eAuf, rAuf);
}

inline_for_extraction noextract
fn fold4_to_2 
  (#et : Type0) {| floating et, real_like et |}
  (#d0 #d1 #d2 #d3: szp)
  (#lA : tlayout (d0 @| d1 @| d2 @| d3 @| INil) { is_full lA })
  {| ctlayout lA |}
  (gA : tensor et lA { is_global gA })
  (#fA : perm)
  (#eA : erased (EM4.t et d0 d1 d2 d3))
  (#rA : erased (EM4.t real d0 d1 d2 d3) { eA %~ rA })
preserves
  cpu
requires
  on gpu_loc (gA |-> Frac fA eA)
returns
  out : (
    tensor et ((tlayout_fold_outer (tlayout_fold_outer lA)) <: tlayout ((d0 *^ d1 *^ d2) @| d3 @| INil)) &
    ematrix et (d0 *^ d1 *^ d2) d3 &
    ematrix real (d0 *^ d1 *^ d2) d3 
  ) 
ensures 
  on gpu_loc (out._1 |-> Frac fA (out._2)) **
  pure (is_global out._1 /\ (out._2 %~ out._3) /\ out._3 == fold_chest (fold_chest rA)) 
{
  let gAf,eAf,rAf = fold4_to_3 gA #fA #eA #rA;
  let gAff = from_array (tlayout_fold_outer (tlayout_fold_outer lA)) (core gAf);
  assert rewrites_to gAff (from_array (tlayout_fold_outer (tlayout_fold_outer lA)) (core gAf));
  map_loc gpu_loc (fun () -> tensor_fold_outer gAf #fA);
  let eAff = fold_chest eAf;
  let rAff = fold_chest rAf;
  return (gAff, eAff, rAff);
}

//#push-options "--print_implicits"
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
      SZ.fits (n * h * l * e) /\
      SZ.fits (n * h * s * e)  /\
      SZ.fits (n * h * s * ev)  /\
      SZ.fits (n * h * l * ev)  /\ 
      SZ.fits (n * h * l * s)  /\
      SZ.fits (n * h * l) /\
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

  let lKT : tlayout (n @| h @| e @| s @| INil) = tlayout_bij f_transpose lK;
  let ctlKT : ctlayout lKT = ctlayout_bij f_transpose lK;
  bmmcomb_gpu_exact #et (fun bias_qk score -> (bias_qk `add` score) `mul` scale) 
    (n*^h) l e s #_ #_ #_ #(ctlayout_bij fold_bij lQ) #(ctlayout_bij fold_bij lKT) #_ gQf gKTf gS;
  with eS'. assert on gpu_loc (gS |-> (eS' <: EM3.t et (n *^ h) l s));

  let rS': EM3.t real (n *^ h) l s = MS.bmmcomb 
    (fun bias_qk score -> (bias_qk +. score) *. (to_real scale))
    rbiasf rQf rKTf;
  assume pure ((eS' <: EM3.t et (n *^ h) l s) %~ rS'); // TODO

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
  // causes z3 assertion violation (just typechecking the proposition, not proving it).. TODO
  // assume pure (
  //   ((Seq.init_ghost #real (n *^ h *^ l) (fun i -> log (rsum (lseq_map exp (ematrix_row rSf i))))) <: lseq real (n *^ h *^ l)) 
  //   == 
  //   lseq_map #real #real log (Seq.init_ghost #(r: real {r >. 0.0R}) (n *^ h *^ l) (fun i -> 
  //     rsum (lseq_map exp (ematrix_row rSf i)) <: (r: real {r >. 0.0R}))));
  assume pure (esums' %~ Seq.init_ghost (n *^ h *^ l) (fun i -> log (rsum (lseq_map exp (ematrix_row rSf i))))); // TODO

  with eSf. assert on gpu_loc (gSf |-> eSf);
  let eSf: ematrix et (n *^ h *^ l) s = eSf;
  assert pure (eSf %~ row_softmax_real rSf);
  let gS, eS, rS = unfold2_to_3 gSf #_ #eSf #(row_softmax_real rSf);

  let gO = alloc0 #et (n *^ h *^ l *^ ev) (l3_batched_row_major (n*^h) l ev);
  with eO. assert on gpu_loc (gO |-> eO);

  bmmcomb_gpu_exact #et MS.comb2 
    (n*^h) l s ev #_ #_ #_ #_ #(ctlayout_bij fold_bij lV) #_ gS gVf gO;

(*
  complete the proof/function; rough steps:
  1. add an unfold3_to_4 function that lets us convert gKTf, gQf, gVf, gbiasf back to their original 4D shapes (and prove that the unfolded versions are equivalent to the original tensors)
  2. do the same for gO, and prove the functional specification (attention_real)
      - this is conceptually simple because we just need to show that basically doing these operations on 3d matrices is the same as 
        doing them on 2D matrices and then creating a 4D matrix with chest.mk as in the spec.
  3. turn the flattened LSE array1 into a 3D tensor and show it approximtes the spec. 
      raise/lower in Array1 should be helpful for this. you probably want to make it into a separate lemma (array1_to_3d or something.)
  4. free sums.
*)
  admit ()
}