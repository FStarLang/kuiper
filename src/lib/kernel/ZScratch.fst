module ZScratch
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
module CH = Kuiper.Chest
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module T = FStar.Tactics.V2
open Kuiper.Spec.Attention
open Kuiper.Kernel.RowSoftmax
module SM = Kuiper.Spec.Softmax
module A1 = Kuiper.Array1
module GU = Kuiper.Kernel.GEMM.Util

#push-options "--split_queries always --fuel 4 --ifuel 4 --z3rlimit 60"

let major_on0 (#n:nat) (k:nat) (#d:idesc n) (sub : layout_f_for d) (x : natlt k) (rest : abs d)
  : Lemma (ensures (major_on 0 k sub).f ((x, rest) <: abs (insert_i 0 k d))
                   == x * sizeof d + sub.f rest)
  = assert ((major_on 0 k sub).f ((x,rest) <: abs (insert_i 0 k d))
            == major_on_f 0 k sub ((x,rest) <: abs (insert_i 0 k d)));
    assert ((abs_bring_forward_bij 0 (insert_i 0 k d)).ff ((x,rest) <: abs (insert_i 0 k d))
            == ((x,rest) <: (natlt (insert_i 0 k d @! 0) & abs (modulo_i 0 (insert_i 0 k d)))))
      by (T.norm [delta_only [`%abs_bring_forward_bij; `%bij_self]; iota; zeta; primops]);
    ()

// row-major imap of l1_forward
let imap_l1 (ev:nat) (m : natlt ev)
  : Lemma ((l1_forward ev).imap.f ((m, ()) <: abs (ev @| INil)) == m)
  = major_on0 ev lunit m ();
    assert ((lunit).f () == 0)

let imap_l2 (l ev:nat) (k : natlt l) (m : natlt ev)
  : Lemma ((l2_row_major l ev).imap.f ((k,(m,())) <: abs (l @| ev @| INil)) == k * ev + m)
  = major_on0 l (major_on 0 ev lunit) k (m,());
    imap_l1 ev m

let imap_l3 (r l ev:nat) (a : natlt r) (k : natlt l) (m : natlt ev)
  : Lemma ((l3_batched_row_major r l ev).imap.f ((a,(k,(m,()))) <: abs (r @| l @| ev @| INil))
           == a * (l * ev) + k * ev + m)
  = major_on0 r (major_on 0 l (major_on 0 ev lunit)) a (k,(m,()));
    imap_l2 l ev k m

let imap_l4 (n h l ev:nat) (i:natlt n)(j:natlt h)(k : natlt l) (m : natlt ev)
  : Lemma ((l4_batched_row_major n h l ev).imap.f ((i,(j,(k,(m,())))) <: abs (n @| h @| l @| ev @| INil))
           == i * (h * l * ev) + j * (l * ev) + k * ev + m)
  = major_on0 n (major_on 0 h (major_on 0 l (major_on 0 ev lunit))) i (j,(k,(m,())));
    imap_l3 h l ev j k m

open Kuiper.ForEvery

ghost
fn relayout_via
  (#et : Type0)
  (#r1 #r2 : nat) (#d1 : idesc r1) (#d2 : idesc r2)
  (#l1 : tlayout d1)
  (l2 : tlayout d2)
  (f : abs d2 =~ abs d1)
  (uleq : squash (tlayout_ulen l1 == tlayout_ulen l2))
  (a : tensor et l1)
  (#fp : perm) (#s : CH.t d1 et)
  requires
    (a |-> Frac fp s) **
    pure (forall (idx : abs d2). l1.imap.f (f.ff idx) == l2.imap.f idx)
  ensures
    from_array l2 (core a) |-> Frac fp (CH.mk d2 (fun idx -> CH.acc s (f.ff idx)))
{
  tensor_ilower a;
  forevery_iso (bij_sym f)
    (fun (i:abs d1) -> pts_to_cell (core a) #fp (l1.imap.f i) (CH.acc s i));
  let s2 : CH.t d2 et = CH.mk d2 (fun idx -> CH.acc s (f.ff idx));
  forevery_map
    (fun (idx:abs d2) -> pts_to_cell (core a) #fp (l1.imap.f ((bij_sym f).gg idx)) (CH.acc s ((bij_sym f).gg idx)))
    (fun (idx:abs d2) -> pts_to_cell (core (from_array l2 (core a))) #fp (l2.imap.f idx) (CH.acc s2 idx))
    fn idx {
      rewrite
        pts_to_cell (core a) #fp (l1.imap.f ((bij_sym f).gg idx)) (CH.acc s ((bij_sym f).gg idx))
      as
        pts_to_cell (core (from_array l2 (core a))) #fp (l2.imap.f idx) (CH.acc s2 idx);
    };
  tensor_iraise (from_array l2 (core a));
}

unfold let fits3 (n h l ev : szp) : prop =
  SZ.fits (SZ.v n * SZ.v h) /\ SZ.fits (SZ.v n * SZ.v h * SZ.v l * SZ.v ev)

let fold_bij_l4 (n h l ev : szp { SZ.fits (SZ.v n * SZ.v h) })
  : (abs (n @| h @| l @| ev @| INil) =~ abs ((n *^ h) @| l @| ev @| INil))
  = mk_bijection
      (fun (idx : abs (n @| h @| l @| ev @| INil)) ->
        let (i,(j,(k,(m,())))) = idx in
        ((((i * SZ.v h + j) <: natlt (SZ.v (n *^ h))), (k,(m,()))) <: abs ((n *^ h) @| l @| ev @| INil)))
      (fun (idx : abs ((n *^ h) @| l @| ev @| INil)) ->
        let (a,(k,(m,()))) = idx in
        ((((a / SZ.v h) <: natlt (SZ.v n)), (((a % SZ.v h) <: natlt (SZ.v h)), (k,(m,())))) <: abs (n @| h @| l @| ev @| INil)))
      (fun (idx : abs ((n *^ h) @| l @| ev @| INil)) ->
        let (a,(k,(m,()))) = idx in
        FStar.Math.Lemmas.euclidean_division_definition a (SZ.v h);
        ())
      (fun (idx : abs (n @| h @| l @| ev @| INil)) ->
        let (i,(j,(k,(m,())))) = idx in
        FStar.Math.Lemmas.lemma_div_plus j i (SZ.v h);
        FStar.Math.Lemmas.lemma_mod_plus j i (SZ.v h);
        FStar.Math.Lemmas.small_div j (SZ.v h);
        FStar.Math.Lemmas.small_mod j (SZ.v h);
        ())

let imap_hyp_l4 (n h l ev : szp) (idx : abs (n @| h @| l @| ev @| INil))
  : Lemma (requires fits3 n h l ev)
          (ensures
            (l3_batched_row_major (n*^h) l ev).imap.f ((fold_bij_l4 n h l ev).ff idx)
            == (l4_batched_row_major n h l ev).imap.f idx)
  = let (i,(j,(k,(m,())))) = idx in
    imap_l4 (SZ.v n) (SZ.v h) (SZ.v l) (SZ.v ev) i j k m;
    let a : natlt (SZ.v (n *^ h)) = i * (SZ.v h) + j in
    assert ((fold_bij_l4 n h l ev).ff idx == ((a,(k,(m,()))) <: abs ((n *^ h) @| l @| ev @| INil)));
    imap_l3 (SZ.v (n*^h)) (SZ.v l) (SZ.v ev) a k m;
    FStar.Math.Lemmas.distributivity_add_left (i * SZ.v h) j (SZ.v l * SZ.v ev);
    FStar.Math.Lemmas.paren_mul_right i (SZ.v h) (SZ.v l * SZ.v ev);
    FStar.Math.Lemmas.paren_mul_right (SZ.v h) (SZ.v l) (SZ.v ev)

let fold_bij_l3 (n h l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  : (abs (n @| h @| l @| INil) =~ abs ((n *^ h *^ l) @| INil))
  = let hl = SZ.v h * SZ.v l in
    let nn = SZ.v n * SZ.v h * SZ.v l in
    mk_bijection
      (fun (idx : abs (n @| h @| l @| INil)) ->
        let (i,(j,(k,()))) = idx in
        ((((i * hl + j * SZ.v l + k) <: natlt nn), ()) <: abs ((n *^ h *^ l) @| INil)))
      (fun (idx : abs ((n *^ h *^ l) @| INil)) ->
        let (r,()) = idx in
        ((((r / hl) <: natlt (SZ.v n)),
          ((((r % hl) / SZ.v l) <: natlt (SZ.v h)),
           ((((r % hl) % SZ.v l) <: natlt (SZ.v l)), ()))) <: abs (n @| h @| l @| INil)))
      (fun (idx : abs ((n *^ h *^ l) @| INil)) ->
        let (r,()) = idx in
        FStar.Math.Lemmas.euclidean_division_definition r hl;
        FStar.Math.Lemmas.euclidean_division_definition (r % hl) (SZ.v l);
        ())
      (fun (idx : abs (n @| h @| l @| INil)) ->
        let (i,(j,(k,()))) = idx in
        FStar.Math.Lemmas.lemma_div_plus (j * SZ.v l + k) i hl;
        FStar.Math.Lemmas.lemma_mod_plus (j * SZ.v l + k) i hl;
        FStar.Math.Lemmas.small_div (j * SZ.v l + k) hl;
        FStar.Math.Lemmas.small_mod (j * SZ.v l + k) hl;
        FStar.Math.Lemmas.lemma_div_plus k j (SZ.v l);
        FStar.Math.Lemmas.lemma_mod_plus k j (SZ.v l);
        FStar.Math.Lemmas.small_div k (SZ.v l);
        FStar.Math.Lemmas.small_mod k (SZ.v l);
        ())

let imap_hyp_l3 (n h l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  (idx : abs (n @| h @| l @| INil))
  : Lemma
      ((l1_forward (n *^ h *^ l)).imap.f ((fold_bij_l3 n h l).ff idx)
       == (l3_batched_row_major n h l).imap.f idx)
  = let (i,(j,(k,()))) = idx in
    imap_l3 (SZ.v n) (SZ.v h) (SZ.v l) i j k;
    let r : natlt (SZ.v n * SZ.v h * SZ.v l) = i * (SZ.v h * SZ.v l) + j * SZ.v l + k in
    assert ((fold_bij_l3 n h l).ff idx == ((r,()) <: abs ((n *^ h *^ l) @| INil)));
    imap_l1 (SZ.v (n *^ h *^ l)) r;
    FStar.Math.Lemmas.paren_mul_right i (SZ.v h) (SZ.v l)

/// ───────────────────────── page-correspondence lemmas ─────────────────────────

let em3_slice_macc (#et:Type) (#d0 #d1 #d2:nat)
  (m : EM3.t et d0 d1 d2) (i:natlt d0) (j:natlt d1) (k:natlt d2)
  : Lemma (EM.macc (EM3.slice_page m i) j k == EM3.macc m i j k)
  = let d : idesc 3 = d0 @| d1 @| d2 @| INil in
    assert ((abs_bring_forward_bij 0 d).gg ((i,(j,(k,()))) <: (natlt (d @! 0) & abs (modulo_i 0 d)))
            == ((i,(j,(k,()))) <: abs d))
      by (T.norm [delta_only [`%abs_bring_forward_bij; `%bij_self; `%Mkbijection?.gg];
                  iota; zeta; primops])

let em4_slice_macc (#et:Type) (#d0 #d1 #d2 #d3:nat)
  (m : EM4.t et d0 d1 d2 d3) (i:natlt d0) (j:natlt d1) (k:natlt d2) (mm:natlt d3)
  : Lemma (EM.macc (EM4.slice_page m i j) k mm == EM4.macc m i j k mm)
  = ()

let fold4_page_macc (#et:Type) (#a #b #c #d3:nat)
  (m4 : EM4.t et a b c d3)
  (i:natlt a) (j:natlt b) (k:natlt c) (mm:natlt d3)
  : Lemma (EM3.macc (fold_chest #et #4 #(a @| b @| c @| d3 @| INil) m4) ((i*b+j) <: natlt (a*b)) k mm
           == EM4.macc m4 i j k mm)
  = FStar.Math.Lemmas.lemma_div_plus j i b;
    FStar.Math.Lemmas.lemma_mod_plus j i b;
    FStar.Math.Lemmas.small_div j b;
    FStar.Math.Lemmas.small_mod j b;
    assert (unfold_index ((((i*b+j) <: natlt (a*b)),(k,(mm,()))) <: abs (fold_outer (a @| b @| c @| d3 @| INil)))
            == ((i,(j,(k,(mm,())))) <: abs (a @| b @| c @| d3 @| INil)))

let fold3_page_macc (#et:Type) (#bigA #c #d3:nat)
  (m3 : EM3.t et bigA c d3)
  (p:natlt bigA) (k:natlt c) (mm:natlt d3)
  : Lemma (EM.macc (fold_chest #et #3 #(bigA @| c @| d3 @| INil) m3) ((p*c+k) <: natlt (bigA*c)) mm
           == EM3.macc m3 p k mm)
  = FStar.Math.Lemmas.lemma_div_plus k p c;
    FStar.Math.Lemmas.lemma_mod_plus k p c;
    FStar.Math.Lemmas.small_div k c;
    FStar.Math.Lemmas.small_mod k c;
    assert (unfold_index ((((p*c+k) <: natlt (bigA*c)),(mm,())) <: abs (fold_outer (bigA @| c @| d3 @| INil)))
            == ((p,(k,(mm,()))) <: abs (bigA @| c @| d3 @| INil)))

let unfold3_page_macc (#et:Type) (#bigA #c #d3:nat)
  (m2 : EM.ematrix et (bigA*c) d3)
  (p:natlt bigA) (k:natlt c) (mm:natlt d3)
  : Lemma (EM3.macc (unfold_chest #et #3 #(bigA @| c @| d3 @| INil) m2) p k mm
           == EM.macc m2 ((p*c+k) <: natlt (bigA*c)) mm)
  = assert (fold_index ((p,(k,(mm,()))) <: abs (bigA @| c @| d3 @| INil))
            == ((((p*c+k) <: natlt (bigA*c)),(mm,())) <: abs (fold_outer (bigA @| c @| d3 @| INil))))

/// EM3.slice_page of a folded EM4 at page (i*b+j) equals EM4.slice_page at (i,j).
let fold4_slice_eq (#et:Type) (#a #b #c #d3:nat)
  (m4 : EM4.t et a b c d3) (i:natlt a) (j:natlt b)
  : Lemma (requires i*b+j < a*b)
          (ensures EM3.slice_page (fold_chest #et #4 #(a @| b @| c @| d3 @| INil) m4)
                     ((i*b+j) <: natlt (a*b))
                   == EM4.slice_page m4 i j)
  = let p : natlt (a*b) = i*b+j in
    let lhs = EM3.slice_page (fold_chest #et #4 #(a @| b @| c @| d3 @| INil) m4) p in
    let rhs = EM4.slice_page m4 i j in
    introduce forall (k:natlt c) (mm:natlt d3). EM.macc lhs k mm == EM.macc rhs k mm
    with (
      em3_slice_macc (fold_chest #et #4 #(a @| b @| c @| d3 @| INil) m4) p k mm;
      fold4_page_macc m4 i j k mm;
      em4_slice_macc m4 i j k mm
    );
    EM.lemma_equal_intro lhs rhs

/// Core row correspondence: row (i*h+j)*l+k of the flattened scores rSf equals
/// row k of the per-page attention scores at page (i,j).
let row_corr
  (n h l s e : pos)
  (rQ : EM4.t real n h l e)
  (rKT : EM4.t real n h e s)
  (rbias : EM4.t real n h l s)
  (scale : real)
  (i:natlt n) (j:natlt h) (k:natlt l) (m:natlt s)
  : Lemma (requires i*h+j < n*h /\ (i*h+j)*l+k < (n*h)*l)
          (ensures
            EM.macc (fold_chest #real #3 #((n*h) @| l @| s @| INil)
                       (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                          #(n*h) #l #e #s
                          (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                          (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                          (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)))
                    (((i*h+j)*l+k) <: natlt ((n*h)*l)) m
            == EM.macc (attn_scores #l #s #e (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                          (EM4.slice_page rbias i j) scale) k m)
  = let combf = (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale) in
    let rbiasf = fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias in
    let rQf = fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ in
    let rKTf = fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT in
    let rS' = MS.bmmcomb combf #(n*h) #l #e #s rbiasf rQf rKTf in
    let p : natlt (n*h) = i*h+j in
    fold3_page_macc #real #(n*h) #l #s rS' p k m;
    fold4_slice_eq rbias i j;
    fold4_slice_eq rQ i j;
    fold4_slice_eq rKT i j

/// Approximation eliminates to per-element for EM3.
let approx3_macc (#et:Type) {| scalar et, real_like et |} (#d0 #d1 #d2:nat)
  (e : EM3.t et d0 d1 d2) (r : EM3.t real d0 d1 d2)
  (i:natlt d0) (j:natlt d1) (k:natlt d2)
  : Lemma (requires e %~ r) (ensures EM3.macc e i j k %~ EM3.macc r i j k)
  = assert (Kuiper.Chest.chest_approximates e r)

/// Approximation of a whole EM3 gives approximation of each page (as EM matrix).
let slice3_approx (#et:Type) {| scalar et, real_like et |} (#d0 #d1 #d2:nat)
  (e : EM3.t et d0 d1 d2) (r : EM3.t real d0 d1 d2) (p:natlt d0)
  : Lemma (requires e %~ r)
          (ensures EM3.slice_page e p %~ EM3.slice_page r p)
  = introduce forall (j:natlt d1) (k:natlt d2).
        EM.macc (EM3.slice_page e p) j k %~ EM.macc (EM3.slice_page r p) j k
    with (
      em3_slice_macc e p j k;
      em3_slice_macc r p j k;
      approx3_macc e r p j k
    );
    EM.lemma_approximates_intro (EM3.slice_page e p) (EM3.slice_page r p)

/// Row-level version of row_corr: whole row equality.
let row_corr_row
  (n h l s e : pos)
  (rQ : EM4.t real n h l e)
  (rKT : EM4.t real n h e s)
  (rbias : EM4.t real n h l s)
  (scale : real)
  (i:natlt n) (j:natlt h) (k:natlt l)
  : Lemma (requires i*h+j < n*h /\ (i*h+j)*l+k < (n*h)*l)
          (ensures
            ematrix_row (fold_chest #real #3 #((n*h) @| l @| s @| INil)
                          (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                             #(n*h) #l #e #s
                             (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                             (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                             (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)))
                        (((i*h+j)*l+k) <: natlt ((n*h)*l))
            == ematrix_row (attn_scores #l #s #e (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                              (EM4.slice_page rbias i j) scale) k)
  = let rSf = fold_chest #real #3 #((n*h) @| l @| s @| INil)
                (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                   #(n*h) #l #e #s
                   (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                   (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                   (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)) in
    let scores_ij = attn_scores #l #s #e (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                      (EM4.slice_page rbias i j) scale in
    let r : natlt ((n*h)*l) = (i*h+j)*l+k in
    introduce forall (c:natlt s). EM.macc rSf r c == EM.macc scores_ij k c
    with row_corr n h l s e rQ rKT rbias scale i j k c;
    Seq.lemma_eq_intro (ematrix_row #real #((n*h)*l) #s rSf r)
                       (ematrix_row #real #l #s scores_ij k)

/// The (i*h+j)-th page of the softmaxed flattened scores equals the per-page softmaxed scores.
let probs_slice_eq
  (n h l s e : pos)
  (rQ : EM4.t real n h l e)
  (rKT : EM4.t real n h e s)
  (rbias : EM4.t real n h l s)
  (scale : real)
  (i:natlt n) (j:natlt h)
  : Lemma (requires i*h+j < n*h /\ (forall (kk:natlt l). (i*h+j)*l+kk < (n*h)*l))
          (ensures
            EM3.slice_page
              (unfold_chest #real #3 #((n*h) @| l @| s @| INil)
                (row_softmax_real #((n*h)*l) #s
                  (fold_chest #real #3 #((n*h) @| l @| s @| INil)
                    (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                       #(n*h) #l #e #s
                       (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                       (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                       (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)))))
              ((i*h+j) <: natlt (n*h))
            == row_softmax_real (attn_scores #l #s #e (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                                   (EM4.slice_page rbias i j) scale))
  = let rSf = fold_chest #real #3 #((n*h) @| l @| s @| INil)
                (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                   #(n*h) #l #e #s
                   (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                   (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                   (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)) in
    let rS = unfold_chest #real #3 #((n*h) @| l @| s @| INil) (row_softmax_real #((n*h)*l) #s rSf) in
    let scores_ij = attn_scores #l #s #e (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                      (EM4.slice_page rbias i j) scale in
    let probs_ij = row_softmax_real scores_ij in
    let p : natlt (n*h) = i*h+j in
    introduce forall (k:natlt l) (c:natlt s). EM.macc (EM3.slice_page rS p) k c == EM.macc probs_ij k c
    with (
      em3_slice_macc rS p k c;
      unfold3_page_macc #real #(n*h) #l #s (row_softmax_real #((n*h)*l) #s rSf) p k c;
      row_corr_row n h l s e rQ rKT rbias scale i j k
    );
    EM.lemma_equal_intro (EM3.slice_page rS p) probs_ij

/// Per-element output correctness: page (i*h+j) of the et batched-matmul output
/// approximates the per-page real attention output at page (i,j).
let out_corr_elem
  (#et:Type) {| scalar et |} {| real_like et |}
  (n h l s e ev : pos)
  (eS : EM3.t et (n*h) l s)
  (eVf : EM3.t et (n*h) s ev)
  (rQ : EM4.t real n h l e)
  (rKT : EM4.t real n h e s)
  (rV : EM4.t real n h s ev)
  (rbias : EM4.t real n h l s)
  (scale : real)
  (i:natlt n) (j:natlt h) (k:natlt l) (m:natlt ev)
  : Lemma
    (requires
      i*h+j < n*h /\ (forall (kk:natlt l). (i*h+j)*l+kk < (n*h)*l) /\
      eS %~ unfold_chest #real #3 #((n*h) @| l @| s @| INil)
              (row_softmax_real #((n*h)*l) #s
                (fold_chest #real #3 #((n*h) @| l @| s @| INil)
                  (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                     #(n*h) #l #e #s
                     (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                     (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                     (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)))) /\
      eVf %~ fold_chest #real #4 #(n @| h @| s @| ev @| INil) rV)
    (ensures
      EM3.macc (MS.batched_matmul eS eVf) ((i*h+j) <: natlt (n*h)) k m
      %~ EM.macc (fst (attention_real (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                         (EM4.slice_page rV i j) (EM4.slice_page rbias i j) scale)) k m)
  = let p : natlt (n*h) = i*h+j in
    let rS = unfold_chest #real #3 #((n*h) @| l @| s @| INil)
               (row_softmax_real #((n*h)*l) #s
                 (fold_chest #real #3 #((n*h) @| l @| s @| INil)
                   (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                      #(n*h) #l #e #s
                      (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                      (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                      (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)))) in
    let rVf = fold_chest #real #4 #(n @| h @| s @| ev @| INil) rV in
    let scores_ij = attn_scores #l #s #e (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                      (EM4.slice_page rbias i j) scale in
    let probs_ij = row_softmax_real #l #s scores_ij in
    // slice eS p %~ probs_ij
    slice3_approx eS rS p;
    probs_slice_eq n h l s e rQ rKT rbias scale i j;
    // slice eVf p %~ slice rV i j
    slice3_approx eVf rVf p;
    fold4_slice_eq rV i j;
    // matmul_single approximation
    GU.__matmul_single_approx_real
      (EM3.slice_page eS p) (EM3.slice_page eVf p)
      probs_ij (EM4.slice_page rV i j)
      k m s;
    // bridge macc(matmul ...) == matmul_single ...
    MS.lemma_matmul_index (EM3.slice_page eS p) (EM3.slice_page eVf p) k m;
    MS.lemma_matmul_index probs_ij (EM4.slice_page rV i j) k m

/// EM4.macc of the relayout chest equals EM3.macc of the source at page (i*h+j).
let relayout4_macc (#et:Type) (#nn #hh #ll #evv:nat)
  (eO : EM3.t et (nn*hh) ll evv)
  (f : abs (nn @| hh @| ll @| evv @| INil) =~ abs ((nn*hh) @| ll @| evv @| INil))
  (i:natlt nn) (j:natlt hh) (k:natlt ll) (m:natlt evv)
  : Lemma (requires i*hh+j < nn*hh /\
                    f.ff ((i,(j,(k,(m,())))) <: abs (nn @| hh @| ll @| evv @| INil))
                    == (((i*hh+j) <: natlt (nn*hh)),(k,(m,()))))
          (ensures
            EM4.macc (CH.mk (nn @| hh @| ll @| evv @| INil)
                       (fun idx -> CH.acc eO (f.ff idx))) i j k m
            == EM3.macc eO ((i*hh+j) <: natlt (nn*hh)) k m)
  = ()

/// Index a seq-level approximation.
let seq_approx_index (#a:Type) {| scalar a, real_like a |} (#len:nat)
  (s : lseq a len) (r : lseq real len) (idx:natlt len)
  : Lemma (requires s %~ r) (ensures Seq.index s idx %~ Seq.index r idx)
  = assert (seq_approximates (s <: seq a) (r <: seq real))

/// Per-element LSE correctness.
let lse_corr_elem
  (#et:Type) {| scalar et |} {| real_like et |}
  (n h l s e : pos)
  (esums' : lseq et (n*h*l))
  (rQ : EM4.t real n h l e)
  (rKT : EM4.t real n h e s)
  (rbias : EM4.t real n h l s)
  (scale : real)
  (i:natlt n) (j:natlt h) (k:natlt l)
  : Lemma
    (requires
      i*h+j < n*h /\ (i*h+j)*l+k < (n*h)*l /\
      i*(h*l)+j*l+k == (i*h+j)*l+k /\ i*(h*l)+j*l+k < n*h*l /\
      esums' %~ Seq.init_ghost (n*h*l)
                  (fun r -> log (rsum (lseq_map exp
                    (ematrix_row #real #((n*h)*l) #s
                      (fold_chest #real #3 #((n*h) @| l @| s @| INil)
                        (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                           #(n*h) #l #e #s
                           (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                           (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                           (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT))) r)))))
    (ensures
      Seq.index esums' ((i*(h*l)+j*l+k) <: natlt (n*h*l))
      %~ Seq.index (attn_lse (attn_scores #l #s #e (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                                (EM4.slice_page rbias i j) scale)) k)
  = let rSf = fold_chest #real #3 #((n*h) @| l @| s @| INil)
                (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                   #(n*h) #l #e #s
                   (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                   (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                   (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)) in
    let theinit = Seq.init_ghost (n*h*l)
                  (fun r -> log (rsum (lseq_map exp
                    (ematrix_row #real #((n*h)*l) #s rSf r)))) in
    let flatidx : natlt (n*h*l) = i*(h*l)+j*l+k in
    seq_approx_index esums' theinit flatidx;
    Seq.init_ghost_index_ (n*h*l)
      (fun r -> log (rsum (lseq_map exp (ematrix_row #real #((n*h)*l) #s rSf r)))) flatidx;
    row_corr_row n h l s e rQ rKT rbias scale i j k;
    let scores_ij = attn_scores #l #s #e (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                      (EM4.slice_page rbias i j) scale in
    Seq.init_ghost_index_ l
      (fun kk -> log (rsum (seq_map exp (ematrix_row #real #l #s scores_ij kk)))) k

/// ───────────────────────── LSE array1 → 3D ─────────────────────────

let a1_to_seq_l1_id (#et:Type) (m:nat) (s : lseq et m)
  : Lemma (A1.to_seq (l1_forward m) s == s)
  = let l : A1.full_layout m = l1_forward m in
    introduce forall (i:natlt m). Seq.index (A1.to_seq l s) i == Seq.index s i
    with (
      let y = Kuiper.Injection.inverse_f l.imap i in
      Kuiper.Injection.inverse_lem l.imap i;
      imap_l1 m (y._1)
    );
    Seq.lemma_eq_intro (A1.to_seq l s) s

let lse_macc_lemma
  (#et:Type)
  (n : szp)
  (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  (esums : lseq et (n *^ h *^ l))
  (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) (k:natlt (SZ.v l))
  : Lemma
     (ensures
        EM3.macc (CH.mk (n @| h @| l @| INil)
                   (fun idx -> CH.acc (from_seq (l1_forward (n *^ h *^ l))
                                        (A1.to_seq (l1_forward (n *^ h *^ l)) esums))
                                      ((fold_bij_l3 n h l).ff idx))) i j k
        == Seq.index esums ((i * (SZ.v h * SZ.v l) + j * SZ.v l + k) <: natlt (SZ.v (n *^ h *^ l))))
  = let r : natlt (SZ.v (n *^ h *^ l)) = i * (SZ.v h * SZ.v l) + j * SZ.v l + k in
    imap_l1 (SZ.v (n *^ h *^ l)) r;
    a1_to_seq_l1_id (n *^ h *^ l) esums;
    imap_hyp_l3 n h l ((i,(j,(k,()))) <: abs (n @| h @| l @| INil));
    imap_l3 (SZ.v n) (SZ.v h) (SZ.v l) i j k;
    FStar.Math.Lemmas.paren_mul_right i (SZ.v h) (SZ.v l);
    assert ((fold_bij_l3 n h l).ff ((i,(j,(k,()))) <: abs (n @| h @| l @| INil))
            == ((r,()) <: abs ((n *^ h *^ l) @| INil)))

let lse_macc_all
  (#et:Type)
  (n : szp)
  (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  (esums : lseq et (n *^ h *^ l))
  : Lemma
     (ensures
        (forall (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) (k:natlt (SZ.v l)).
           EM3.macc (CH.mk (n @| h @| l @| INil)
                      (fun idx -> CH.acc (from_seq (l1_forward (n *^ h *^ l))
                                           (A1.to_seq (l1_forward (n *^ h *^ l)) esums))
                                         ((fold_bij_l3 n h l).ff idx))) i j k
           == Seq.index esums ((i * (SZ.v h * SZ.v l) + j * SZ.v l + k) <: natlt (SZ.v (n *^ h *^ l)))))
  = introduce forall (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) (k:natlt (SZ.v l)).
        EM3.macc (CH.mk (n @| h @| l @| INil)
                   (fun idx -> CH.acc (from_seq (l1_forward (n *^ h *^ l))
                                        (A1.to_seq (l1_forward (n *^ h *^ l)) esums))
                                      ((fold_bij_l3 n h l).ff idx))) i j k
        == Seq.index esums ((i * (SZ.v h * SZ.v l) + j * SZ.v l + k) <: natlt (SZ.v (n *^ h *^ l)))
    with lse_macc_lemma n h l esums i j k

ghost
fn array1_to_3d
  (#et:Type)
  (n : szp)
  (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  (sums : A1.t et (l1_forward (n *^ h *^ l)))
  (#f : perm) (#esums : lseq et (n *^ h *^ l))
  requires
    (sums |-> Frac f esums) **
    pure (SZ.fits (SZ.v n * SZ.v h * SZ.v l))
  returns g : tensor et (l3_batched_row_major n h l)
  ensures
    (exists* (s3 : CH.t (n @| h @| l @| INil) et).
       (g |-> Frac f s3) **
       pure (forall (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) (k:natlt (SZ.v l)).
               EM3.macc s3 i j k
               == Seq.index esums ((i * (SZ.v h * SZ.v l) + j * SZ.v l + k) <: natlt (SZ.v (n *^ h *^ l))))) **
    pure (g == from_array (l3_batched_row_major n h l) (A1.core sums) /\
          (A1.is_global sums ==> is_global g))
{
  A1.pts_to_ref sums;
  A1.lower sums;
  tensor_abs' (l1_forward (n *^ h *^ l)) (A1.core sums);
  let g1 : tensor et (l1_forward (n *^ h *^ l)) =
    from_array (l1_forward (n *^ h *^ l)) (A1.core sums);
  assert rewrites_to g1 (from_array (l1_forward (n *^ h *^ l)) (A1.core sums));
  FStar.Classical.forall_intro (imap_hyp_l3 n h l);
  relayout_via (l3_batched_row_major n h l) (fold_bij_l3 n h l) () g1;
  a1_to_seq_l1_id (n *^ h *^ l) esums;
  lse_macc_all n h l esums;
  let g : tensor et (l3_batched_row_major n h l) =
    from_array (l3_batched_row_major n h l) (core g1);
  assert rewrites_to g (from_array (l3_batched_row_major n h l) (core g1));
  g
}

/// ───────────────────────── final combining lemmas ─────────────────────────

let page_row_bound (nh l : nat) (p : natlt nh) (kk : natlt l)
  : Lemma (p * l + kk < nh * l)
  = FStar.Math.Lemmas.lemma_mult_le_right l (p+1) nh

let imap_hyp_l4_all (n h l ev : szp)
  : Lemma (requires fits3 n h l ev)
          (ensures
            (forall (idx : abs (n @| h @| l @| ev @| INil)).
               (l3_batched_row_major (n*^h) l ev).imap.f ((fold_bij_l4 n h l ev).ff idx)
               == (l4_batched_row_major n h l ev).imap.f idx))
  = introduce forall (idx : abs (n @| h @| l @| ev @| INil)).
        (l3_batched_row_major (n*^h) l ev).imap.f ((fold_bij_l4 n h l ev).ff idx)
        == (l4_batched_row_major n h l ev).imap.f idx
    with imap_hyp_l4 n h l ev idx

let relayout4_macc_all (#et:Type) (n : szp) (h : szp { SZ.fits (SZ.v n * SZ.v h) }) (l ev : szp)
  (eB : EM3.t et (n*^h) l ev)
  : Lemma
          (ensures
            (forall (i:natlt (SZ.v n))(j:natlt (SZ.v h))(k:natlt (SZ.v l))(m:natlt (SZ.v ev)).
               i * SZ.v h + j < SZ.v (n*^h) /\
               EM4.macc (CH.mk (n @| h @| l @| ev @| INil)
                          (fun idx -> CH.acc eB ((fold_bij_l4 n h l ev).ff idx))) i j k m
               == EM3.macc eB ((i * SZ.v h + j) <: natlt (SZ.v (n*^h))) k m))
  = introduce forall (i:natlt (SZ.v n))(j:natlt (SZ.v h))(k:natlt (SZ.v l))(m:natlt (SZ.v ev)).
        i * SZ.v h + j < SZ.v (n*^h) /\
        EM4.macc (CH.mk (n @| h @| l @| ev @| INil)
                   (fun idx -> CH.acc eB ((fold_bij_l4 n h l ev).ff idx))) i j k m
        == EM3.macc eB ((i * SZ.v h + j) <: natlt (SZ.v (n*^h))) k m
    with (
      FStar.Math.Lemmas.lemma_mult_le_right (SZ.v h) (i+1) (SZ.v n);
      let pidx : natlt (SZ.v (n*^h)) = i * SZ.v h + j in
      assert ((fold_bij_l4 n h l ev).ff ((i,(j,(k,(m,())))) <: abs (n @| h @| l @| ev @| INil))
              == ((pidx,(k,(m,()))) <: abs ((n*^h) @| l @| ev @| INil)))
    )

let out_approx_all
  (#et:Type) {| scalar et |} {| real_like et |}
  (n : szp) (h : szp { SZ.fits (SZ.v n * SZ.v h) }) (l s e ev : szp)
  (eO4 : EM4.t et n h l ev)
  (eS : EM3.t et (n*^h) l s)
  (eVf : EM3.t et (n*^h) s ev)
  (rQ : EM4.t real n h l e)
  (rKT : EM4.t real n h e s)
  (rV : EM4.t real n h s ev)
  (rbias : EM4.t real n h l s)
  (scale : real)
  : Lemma
    (requires
      fits3 n h l ev /\
      (forall (i:natlt (SZ.v n))(j:natlt (SZ.v h))(k:natlt (SZ.v l))(m:natlt (SZ.v ev)).
         EM4.macc eO4 i j k m
         == EM3.macc (MS.batched_matmul eS eVf) ((i * SZ.v h + j) <: natlt (SZ.v (n*^h))) k m) /\
      eS %~ unfold_chest #real #3 #((n*^h) @| l @| s @| INil)
              (row_softmax_real #(SZ.v (n*^h) * SZ.v l) #(SZ.v s)
                (fold_chest #real #3 #((n*^h) @| l @| s @| INil)
                  (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                     #(SZ.v (n*^h)) #(SZ.v l) #(SZ.v e) #(SZ.v s)
                     (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                     (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                     (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT)))) /\
      eVf %~ (fold_chest #real #4 #(n @| h @| s @| ev @| INil) rV <: EM3.t real (n*^h) s ev))
    (ensures
      eO4 %~ EM4.mkM (fun (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) ->
               EM.macc (fst (attention_real
                 (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                 (EM4.slice_page rV i j) (EM4.slice_page rbias i j) scale))))
  = let out_spec : EM4.t real (SZ.v n) (SZ.v h) (SZ.v l) (SZ.v ev) =
      EM4.mkM (fun (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) ->
               EM.macc (fst (attention_real
                 (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                 (EM4.slice_page rV i j) (EM4.slice_page rbias i j) scale))) in
    introduce forall (i:natlt (SZ.v n))(j:natlt (SZ.v h))(k:natlt (SZ.v l))(m:natlt (SZ.v ev)).
       EM4.macc eO4 i j k m %~ EM4.macc out_spec i j k m
    with (
      FStar.Math.Lemmas.lemma_mult_le_right (SZ.v h) (i+1) (SZ.v n);
      introduce forall (kk:natlt (SZ.v l)). (i * SZ.v h + j) * SZ.v l + kk < SZ.v (n*^h) * SZ.v l
      with page_row_bound (SZ.v (n*^h)) (SZ.v l) ((i * SZ.v h + j) <: natlt (SZ.v (n*^h))) kk;
      out_corr_elem (SZ.v n) (SZ.v h) (SZ.v l) (SZ.v s) (SZ.v e) (SZ.v ev)
        eS eVf rQ rKT rV rbias scale i j k m
    );
    EM4.lemma_approximates_intro eO4 out_spec

let lse_approx_all
  (#et:Type) {| scalar et |} {| real_like et |}
  (n : szp) (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) }) (s e ev : szp)
  (eLSE : EM3.t et n h l)
  (esums' : lseq et (n *^ h *^ l))
  (rQ : EM4.t real n h l e)
  (rKT : EM4.t real n h e s)
  (rV : EM4.t real n h s ev)
  (rbias : EM4.t real n h l s)
  (scale : real)
  : Lemma
    (requires
      SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) /\
      (forall (i:natlt (SZ.v n))(j:natlt (SZ.v h))(k:natlt (SZ.v l)).
         EM3.macc eLSE i j k
         == Seq.index esums' ((i * (SZ.v h * SZ.v l) + j * SZ.v l + k) <: natlt (SZ.v (n *^ h *^ l)))) /\
      esums' %~ Seq.init_ghost (SZ.v (n *^ h *^ l))
                  (fun r -> log (rsum (lseq_map exp
                    (ematrix_row #real #(SZ.v (n*^h) * SZ.v l) #(SZ.v s)
                      (fold_chest #real #3 #((n*^h) @| l @| s @| INil)
                        (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                           #(SZ.v (n*^h)) #(SZ.v l) #(SZ.v e) #(SZ.v s)
                           (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                           (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                           (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT))) r)))))
    (ensures
      eLSE %~ EM3.mkM (fun (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) ->
               Seq.index (snd (attention_real
                 (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                 (EM4.slice_page rV i j) (EM4.slice_page rbias i j) scale))))
  = let lse_spec : EM3.t real (SZ.v n) (SZ.v h) (SZ.v l) =
      EM3.mkM (fun (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) ->
               Seq.index (snd (attention_real
                 (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                 (EM4.slice_page rV i j) (EM4.slice_page rbias i j) scale))) in
    introduce forall (i:natlt (SZ.v n))(j:natlt (SZ.v h))(k:natlt (SZ.v l)).
       EM3.macc eLSE i j k %~ EM3.macc lse_spec i j k
    with (
      FStar.Math.Lemmas.lemma_mult_le_right (SZ.v h) (i+1) (SZ.v n);
      page_row_bound (SZ.v (n*^h)) (SZ.v l) ((i * SZ.v h + j) <: natlt (SZ.v (n*^h))) k;
      FStar.Math.Lemmas.paren_mul_right i (SZ.v h) (SZ.v l);
      lse_corr_elem (SZ.v n) (SZ.v h) (SZ.v l) (SZ.v s) (SZ.v e)
        esums' rQ rKT rbias scale i j k
    );
    EM3.lemma_approximates_intro eLSE lse_spec
