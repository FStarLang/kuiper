module Kuiper.Kernel.Attention.Helpers

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
  pure (is_global out._1 /\ (out._2 %~ out._3) /\ out._3 == fold_chest rA /\
        out._2 == fold_chest eA /\
        out._1 == from_array (tlayout_fold_outer lA) (core gA))
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
  on gpu_loc (gA |-> Frac fA eA) **
  pure (is_full_array (core gA))
returns
  out : (
    tensor et (tlayout_fold_outer lA <: tlayout ((d0 *^ d1) @| d2 @| INil)) &
    ematrix et (d0 *^ d1) d2 &
    ematrix real (d0 *^ d1) d2 
  ) 
ensures 
  on gpu_loc (out._1 |-> Frac fA (out._2)) **
  pure (is_global out._1 /\ (out._2 %~ out._3) /\ out._3 == fold_chest rA /\
        is_full_array (core out._1))
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
  on gpu_loc (gA |-> Frac fA eA) **
  pure (is_full_array (core gA))
returns
  out : (
    tensor et lA &
    EM3.t et d0 d1 d2 &
    EM3.t real d0 d1 d2 
  ) 
ensures 
  on gpu_loc (out._1 |-> Frac fA (out._2)) **
  pure (is_global out._1 /\ (out._2 %~ out._3) /\ out._3 == unfold_chest rA /\
        is_full_array (core out._1))
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


/// ===== transplanted helper lemmas from scratch dev =====
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

/// Deterministic proof of the [approx2] fact for the GEMM score combiner
/// `(bias_qk + score) * scale`. Proving it explicitly (rather than via the flaky
/// [a_add]/[a_mul] SMT patterns) keeps this robust against perturbations of the
/// ambient SMT context.
let approx2_addmul_scale
  (#et:Type) {| scalar et |} {| real_like et |}
  (scale : et)
  : Lemma (approx2 #et #et #et
             (fun (bias_qk:et) (score:et) -> (bias_qk `add` score) `mul` scale)
             (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. (to_real scale)))
  = introduce forall (x:et) (y:et) (r:real) (s:real).
       x %~ r /\ y %~ s ==> ((x `add` y) `mul` scale) %~ ((r +. s) *. (to_real scale))
    with introduce _ ==> _
    with _. (
      a_add x y r s;
      to_real_ok scale;
      a_mul (x `add` y) scale (r +. s) (to_real scale)
    )

/// Flat row-major index into an [n*h*l] sequence, with the in-bounds proof
/// carried in the *return type*. This lets the (SMT-free) Pulse core checker accept
/// `Seq.index esums (flat3 n h l i j k)` at call sites without having to (re)discharge
/// the nonlinear bound — which it cannot do — while [flat3_eq] keeps it interchangeable
/// with the explicit `(i*(h*l)+j*l+k <: natlt ...)` form used by the SMT-checked lemmas.
let flat3
  (n : szp) (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) (k:natlt (SZ.v l))
  : natlt (SZ.v (n *^ h *^ l))
  = FStar.Math.Lemmas.lemma_mult_le_right (SZ.v h) (i+1) (SZ.v n);
    assert (SZ.v (n *^ h) == SZ.v n * SZ.v h);
    assert (SZ.v (n *^ h *^ l) == SZ.v n * SZ.v h * SZ.v l);
    assert (i * SZ.v h + j < SZ.v n * SZ.v h);
    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v l) (i * SZ.v h + j + 1) (SZ.v n * SZ.v h);
    FStar.Math.Lemmas.paren_mul_right i (SZ.v h) (SZ.v l);
    assert (i * (SZ.v h * SZ.v l) + j * SZ.v l + k < SZ.v n * SZ.v h * SZ.v l);
    i * (SZ.v h * SZ.v l) + j * SZ.v l + k

let flat3_eq
  (n : szp) (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) (k:natlt (SZ.v l))
  : Lemma (flat3 n h l i j k == i * (SZ.v h * SZ.v l) + j * SZ.v l + k)
  = ()

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
              == Seq.index esums (flat3 n h l i j k))) **
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
  FStar.Classical.forall_intro_3 (flat3_eq n h l);
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

#push-options "--z3rlimit 200"
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
      // Pin the (nonlinear) index arithmetic explicitly so the precondition
      // check of [lse_corr_elem] is robust (it was flaky interactively).
      FStar.Math.Lemmas.lemma_mult_le_right (SZ.v h) (i+1) (SZ.v n);
      assert (SZ.v (n*^h) == SZ.v n * SZ.v h);
      assert (SZ.v (n*^h*^l) == SZ.v n * SZ.v h * SZ.v l);
      assert (SZ.v (n*^h) * SZ.v l == SZ.v n * SZ.v h * SZ.v l);
      assert (i * SZ.v h + j < SZ.v n * SZ.v h);
      page_row_bound (SZ.v (n*^h)) (SZ.v l) ((i * SZ.v h + j) <: natlt (SZ.v (n*^h))) k;
      FStar.Math.Lemmas.paren_mul_right i (SZ.v h) (SZ.v l);
      assert ((i * SZ.v h + j) * SZ.v l + k < SZ.v (n*^h) * SZ.v l);
      assert (i * (SZ.v h * SZ.v l) + j * SZ.v l + k == (i * SZ.v h + j) * SZ.v l + k);
      assert (i * (SZ.v h * SZ.v l) + j * SZ.v l + k < SZ.v n * SZ.v h * SZ.v l);
      lse_corr_elem (SZ.v n) (SZ.v h) (SZ.v l) (SZ.v s) (SZ.v e)
        esums' rQ rKT rbias scale i j k
    );
    EM3.lemma_approximates_intro eLSE lse_spec
#pop-options

/// Elementwise [flog] step for the LSE: if [a] approximates the (positive)
/// row-sums-of-exp sequence [Seq.init_ghost m g], then [lseq_map flog a]
/// approximates the log-sum-exp sequence [Seq.init_ghost m (fun i -> log (g i))].
///
/// Proven here (with [g] abstract) so the caller never has to *elaborate*
/// `log (rsum (lseq_map exp ...))` in its own (heavy) proof context — doing so
/// trips an internal Z3 4.13.3 linear-arithmetic solver assertion violation
/// (lar_solver.cpp). With [g] abstract the positivity is a plain hypothesis.
let lse_flog_corr
  (#et:Type) {| scalar et |} {| floating et |} {| real_like et |} {| floating_real_like et |}
  (m : nat)
  (a : lseq et m)
  (g : (i:natlt m -> GTot real))
  : Lemma
      (requires
        (forall (i:natlt m). g i >. 0.0R) /\
        a %~ Seq.init_ghost m g)
      (ensures
        lseq_map flog a %~ Seq.init_ghost m (fun (i:natlt m) -> log (g i)))
  = let rhs : lseq real m = Seq.init_ghost m (fun (i:natlt m) -> log (g i)) in
    let lhs : lseq et m = lseq_map flog a in
    assert (Seq.length lhs == m);
    assert (Seq.length rhs == m);
    introduce forall (i:natlt m). Seq.index lhs i %~ Seq.index rhs i
    with (
      Seq.init_ghost_index_ m (fun (j:natlt m) -> log (g j)) i;
      Seq.init_ghost_index_ m g i;
      Seq.init_ghost_index_ #et m (fun (j:natlt m) -> flog (Seq.index a j)) i;
      // Seq.index a i %~ g i  (from [a %~ Seq.init_ghost m g]); g i >. 0.0R (hypothesis);
      // hence flog (Seq.index a i) %~ log (g i) by the [log_approx_pat] SMTPat.
      ()
    );
    assert (seq_approximates (lhs <: seq et) (rhs <: seq real))


/// Combined LSE correspondence usable from a *heavy* caller context: the caller
/// supplies the (positive) row-sums sequence and the relation [esums' == lseq_map flog esums],
/// but NEVER needs to elaborate the `log (rsum (lseq_map exp ...))` term itself (doing so in a
/// heavy context trips the Z3 4.13.3 lar_solver assertion violation). The [flog]/[log] bridge is
/// performed here (in this light context, where that term encodes fine) via [lse_flog_corr], and
/// the result is fed to [lse_approx_all].
let lse_approx_all_from_sums
  (#et:Type) {| scalar et |} {| floating et |} {| real_like et |} {| floating_real_like et |}
  (n : szp) (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) }) (s e ev : szp)
  (eLSE : EM3.t et n h l)
  (esums : lseq et (n *^ h *^ l))
  (esums' : lseq et (n *^ h *^ l))
  (rQ : EM4.t real n h l e)
  (rKT : EM4.t real n h e s)
  (rV : EM4.t real n h s ev)
  (rbias : EM4.t real n h l s)
  (scale : real)
  : Lemma
    (requires
      SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) /\
      esums' == lseq_map flog esums /\
      (forall (i:natlt (SZ.v n))(j:natlt (SZ.v h))(k:natlt (SZ.v l)).
         EM3.macc eLSE i j k
         == Seq.index esums' (flat3 n h l i j k)) /\
      (forall (r:natlt (SZ.v (n *^ h *^ l))).
         rsum (lseq_map exp
           (ematrix_row #real #(SZ.v (n*^h) * SZ.v l) #(SZ.v s)
             (fold_chest #real #3 #((n*^h) @| l @| s @| INil)
               (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                  #(SZ.v (n*^h)) #(SZ.v l) #(SZ.v e) #(SZ.v s)
                  (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                  (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                  (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT))) r)) >. 0.0R) /\
      esums %~ Seq.init_ghost (SZ.v (n *^ h *^ l))
                  (fun r -> rsum (lseq_map exp
                    (ematrix_row #real #(SZ.v (n*^h) * SZ.v l) #(SZ.v s)
                      (fold_chest #real #3 #((n*^h) @| l @| s @| INil)
                        (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
                           #(SZ.v (n*^h)) #(SZ.v l) #(SZ.v e) #(SZ.v s)
                           (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
                           (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
                           (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT))) r))))
    (ensures
      eLSE %~ EM3.mkM (fun (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) ->
               Seq.index (snd (attention_real
                 (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                 (EM4.slice_page rV i j) (EM4.slice_page rbias i j) scale))))
  = FStar.Classical.forall_intro_3 (flat3_eq n h l);
    lse_flog_corr (SZ.v (n *^ h *^ l)) esums
      (fun r -> rsum (lseq_map exp
        (ematrix_row #real #(SZ.v (n*^h) * SZ.v l) #(SZ.v s)
          (fold_chest #real #3 #((n*^h) @| l @| s @| INil)
            (MS.bmmcomb (fun (bias_qk:real) (score:real) -> (bias_qk +. score) *. scale)
               #(SZ.v (n*^h)) #(SZ.v l) #(SZ.v e) #(SZ.v s)
               (fold_chest #real #4 #(n @| h @| l @| s @| INil) rbias)
               (fold_chest #real #4 #(n @| h @| l @| e @| INil) rQ)
               (fold_chest #real #4 #(n @| h @| e @| s @| INil) rKT))) r)));
    lse_approx_all #et n h l s e ev eLSE esums' rQ rKT rV rbias scale


let fold_unfold_chest_id (#et:Type0) (#r:nat{r>1}) (#d:idesc r) (m : CH.t d et)
  : Lemma (unfold_chest #et #r #d (fold_chest #et #r #d m) == m)
  = let ICons h1 (ICons h2 ts) = d in
    introduce forall (i : abs d).
      CH.acc (unfold_chest #et #r #d (fold_chest #et #r #d m)) i == CH.acc m i
    with (
      let (i1,(i2,it)) : (natlt h1 & (natlt h2 & abs ts)) = i in
      FStar.Math.Lemmas.lemma_div_plus i2 i1 h2;
      FStar.Math.Lemmas.lemma_mod_plus i2 i1 h2;
      FStar.Math.Lemmas.small_div i2 h2;
      FStar.Math.Lemmas.small_mod i2 h2;
      assert (unfold_index #r #d (fold_index #r #d i) == i)
    );
    CH.lemma_equal_intro (unfold_chest #et #r #d (fold_chest #et #r #d m)) m;
    CH.ext (unfold_chest #et #r #d (fold_chest #et #r #d m)) m

let untranspose_imap_hyp
  (#r1 #r2 : nat) (#dK : idesc r1) (#dT : idesc r2)
  (g : abs dK =~ abs dT) (lK : tlayout dK) (idx : abs dK)
  : Lemma ((tlayout_bij g lK).imap.f (g.ff idx) == lK.imap.f idx)
  = ()

let kt_chest_eq (#et:Type0) (n h s e : szp)
  (eK : EM4.t et n h s e)
  (f_t : abs (n @| h @| s @| e @| INil) =~ abs (n @| h @| e @| s @| INil))
  (idx : abs (n @| h @| s @| e @| INil))
  : Lemma (CH.acc (CH.mk (n @| h @| e @| s @| INil) (fun i -> CH.acc eK (i <~| f_t))) (f_t.ff idx)
           == CH.acc eK idx)
  = ()

let ulen_eq_l3l4 (n h l ev : szp { SZ.fits (SZ.v n * SZ.v h) })
  : squash (tlayout_ulen (l3_batched_row_major (n*^h) l ev)
            == tlayout_ulen (l4_batched_row_major n h l ev))
  = assert (is_full (l3_batched_row_major (n*^h) l ev));
    assert (is_full (l4_batched_row_major n h l ev));
    full_layout_size (l3_batched_row_major (n*^h) l ev);
    full_layout_size (l4_batched_row_major n h l ev)

inline_for_extraction noextract
ghost
fn restore_fold4
  (#et : Type0) {| floating et, real_like et |}
  (#d0 #d1 #d2 #d3 : szp)
  (#lA : tlayout (d0 @| d1 @| d2 @| d3 @| INil) { is_full lA })
  {| ctlayout lA |}
  (gA : tensor et lA)
  (gAf : tensor et (tlayout_fold_outer lA <: tlayout ((d0 *^ d1) @| d2 @| d3 @| INil)))
  (#fA : perm)
  (#eAf : EM3.t et (d0 *^ d1) d2 d3)
  (#eA : EM4.t et d0 d1 d2 d3)
  requires
    on gpu_loc (gAf |-> Frac fA eAf) **
    pure (gAf == from_array (tlayout_fold_outer lA) (core gA) /\
          eAf == fold_chest eA)
  ensures
    on gpu_loc (gA |-> Frac fA eA)
{
  fold_unfold_chest_id #et #4 #(d0 @| d1 @| d2 @| d3 @| INil) eA;
  map_loc gpu_loc (fun () -> tensor_unfold_outer gAf #fA);
  rewrite (on gpu_loc (from_array lA (core gAf) |-> Frac fA (unfold_chest #et #4 #(d0 @| d1 @| d2 @| d3 @| INil) eAf)))
       as (on gpu_loc (gA |-> Frac fA eA));
}

inline_for_extraction noextract
ghost
fn gpu_relayout_to
  (#et : Type0)
  (#r1 #r2 : nat) (#d1 : idesc r1) (#d2 : idesc r2)
  (#l1 : tlayout d1)
  (l2 : tlayout d2)
  (f : abs d2 =~ abs d1)
  (uleq : squash (tlayout_ulen l1 == tlayout_ulen l2))
  (a : tensor et l1)
  (gtarget : tensor et l2)
  (#fp : perm) (#s : CH.t d1 et)
  requires
    on gpu_loc (a |-> Frac fp s) **
    pure ((forall (idx : abs d2). l1.imap.f (f.ff idx) == l2.imap.f idx) /\
          gtarget == from_array l2 (core a))
  ensures
    on gpu_loc (gtarget |-> Frac fp (CH.mk d2 (fun idx -> CH.acc s (f.ff idx))))
{
  map_loc gpu_loc (fun () -> relayout_via l2 f uleq a);
  rewrite (on gpu_loc (from_array l2 (core a) |-> Frac fp (CH.mk d2 (fun idx -> CH.acc s (f.ff idx)))))
       as (on gpu_loc (gtarget |-> Frac fp (CH.mk d2 (fun idx -> CH.acc s (f.ff idx)))));
}

/// Gpu-located, ghost wrapper around [array1_to_3d]: transforms the (gpu-located)
/// flat LSE array [sums] into the (gpu-located) 3D tensor [gLSE], which is supplied
/// as a refined argument (so the result handle stays a pure, extractable value).
ghost
fn array1_to_3d_u
  (#et:Type)
  (n : szp)
  (h : szp { SZ.fits (SZ.v n * SZ.v h) })
  (l : szp { SZ.fits (SZ.v n * SZ.v h * SZ.v l) /\ SZ.fits (SZ.v h * SZ.v l) })
  (sums : A1.t et (l1_forward (n *^ h *^ l)))
  (gLSE : tensor et (l3_batched_row_major n h l)
            { gLSE == from_array (l3_batched_row_major n h l) (A1.core sums) })
  (#f : perm) (#esums : lseq et (n *^ h *^ l))
  requires
    on gpu_loc (sums |-> Frac f esums)
  ensures
    on gpu_loc
      (exists* (s3 : CH.t (n @| h @| l @| INil) et).
         (gLSE |-> Frac f s3) **
         pure (forall (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) (k:natlt (SZ.v l)).
                 EM3.macc s3 i j k
                 == Seq.index esums (flat3 n h l i j k)))
{
  map_loc gpu_loc
    #(sums |-> Frac f esums)
    #(exists* (s3 : CH.t (n @| h @| l @| INil) et).
        (gLSE |-> Frac f s3) **
        pure (forall (i:natlt (SZ.v n)) (j:natlt (SZ.v h)) (k:natlt (SZ.v l)).
                EM3.macc s3 i j k
                == Seq.index esums (flat3 n h l i j k)))
    fn () {
      let g' = array1_to_3d n h l sums #f #esums;
      rewrite each g' as gLSE;
    };
}

#pop-options

/// Unfold the batched attention spec to its per-page form (definitional).
let attention_real_batched_unfold
  (#n #h #l #s #e #ev : pos)
  (rQ : CH.t (n @| h @| l @| e @| INil) real)
  (rKT : CH.t (n @| h @| e @| s @| INil) real)
  (rV : CH.t (n @| h @| s @| ev @| INil) real)
  (rbias : CH.t (n @| h @| l @| s @| INil) real)
  (scale : real)
  : Lemma
      (attention_real_batched rQ rKT rV rbias scale
       == (EM4.mkM (fun (i:natlt n) (j:natlt h) ->
              EM.macc (fst (attention_real
                (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                (EM4.slice_page rV i j) (EM4.slice_page rbias i j) scale))),
           EM3.mkM (fun (i:natlt n) (j:natlt h) ->
              Seq.index (snd (attention_real
                (EM4.slice_page rQ i j) (EM4.slice_page rKT i j)
                (EM4.slice_page rV i j) (EM4.slice_page rbias i j) scale)))))
  = ()

/// ───────────────────────── bias→scores layout-aware copy helpers ─────────────

module TL = Kuiper.Tensor.Layout

#push-options "--split_queries always --fuel 2 --ifuel 2 --z3rlimit 60"

/// Logical lseq of a 1-D chest (mirror of Array1.backtr_val, defined publicly).
let chest_to_seq1 (#et:Type) (#nn:nat) (c : CH.t (nn @| INil) et) : GTot (lseq et nn)
  = Seq.init_ghost nn (fun (i:natlt nn) -> CH.acc c ((i, ()) <: abs (nn @| INil)))

/// 1-D chest from an lseq (mirror of Array1.tr_val, defined publicly).
let seq1_to_chest (#et:Type) (#nn:nat) (s : lseq et nn) : CH.t (nn @| INil) et
  = CH.mk (nn @| INil) (fun (idx : abs (nn @| INil)) -> Seq.index s idx._1)

let chest_seq1_roundtrip (#et:Type) (#nn:nat) (s : lseq et nn)
  : Lemma (chest_to_seq1 (seq1_to_chest s) == s)
  = introduce forall (i:natlt nn). Seq.index (chest_to_seq1 (seq1_to_chest s)) i == Seq.index s i
    with (
      Seq.init_ghost_index_ nn
        (fun (j:natlt nn) -> CH.acc (seq1_to_chest s) ((j, ()) <: abs (nn @| INil))) i
    );
    Seq.lemma_eq_intro (chest_to_seq1 (seq1_to_chest s)) s

let chest_to_seq1_index (#et:Type) (#nn:nat) (c : CH.t (nn @| INil) et) (i:natlt nn)
  : Lemma (Seq.index (chest_to_seq1 c) i == CH.acc c ((i, ()) <: abs (nn @| INil)))
  = Seq.init_ghost_index_ nn (fun (j:natlt nn) -> CH.acc c ((j, ()) <: abs (nn @| INil))) i

/// Reverse roundtrip: re-chesting the logical sequence of a 1-D chest recovers it.
let seq1_chest_roundtrip (#et:Type) (#nn:nat) (c : CH.t (nn @| INil) et)
  : Lemma (seq1_to_chest (chest_to_seq1 c) == c)
  = introduce forall (idx : abs (nn @| INil)).
        CH.acc (seq1_to_chest (chest_to_seq1 c)) idx == CH.acc c idx
    with (
      chest_to_seq1_index c idx._1;
      assert (idx == ((idx._1, ()) <: abs (nn @| INil)))
    );
    CH.lemma_equal_intro (seq1_to_chest (chest_to_seq1 c)) c;
    CH.ext (seq1_to_chest (chest_to_seq1 c)) c

/// The logical sequence [sb] of [aBias] (the chest produced by relaying [eb] through
/// [bij_sym flat]) is the row-major flattening of [eb].
let sb_flatten_char
  (#et:Type)
  (bnh bl bs : szp { SZ.fits (SZ.v bnh * SZ.v bl * SZ.v bs) /\ SZ.fits (SZ.v bl * SZ.v bs) })
  (eb : EM3.t et bnh bl bs)
  : Lemma
    (ensures
      (forall (r:natlt (SZ.v (bnh *^ bl *^ bs))).
        Seq.index
          (chest_to_seq1 (CH.mk ((bnh *^ bl *^ bs) @| INil)
             (fun idx -> CH.acc eb ((bij_sym (fold_bij_l3 bnh bl bs)).ff idx))))
          r
        == CH.acc eb ((fold_bij_l3 bnh bl bs).gg ((r, ()) <: abs ((bnh *^ bl *^ bs) @| INil)))))
  = let flat = fold_bij_l3 bnh bl bs in
    let cb = CH.mk ((bnh *^ bl *^ bs) @| INil)
               (fun idx -> CH.acc eb ((bij_sym flat).ff idx)) in
    introduce forall (r:natlt (SZ.v (bnh *^ bl *^ bs))).
      Seq.index (chest_to_seq1 cb) r
      == CH.acc eb (flat.gg ((r, ()) <: abs ((bnh *^ bl *^ bs) @| INil)))
    with ( chest_to_seq1_index cb r )

/// Macc-indexed characterization of the row-major flattening: the flattened
/// sequence at row-major position [i*(bl*bs)+j*bs+k] equals [EM3.macc eb i j k].
let sb_macc_char
  (#et:Type)
  (bnh bl bs : szp { SZ.fits (SZ.v bnh * SZ.v bl * SZ.v bs) /\ SZ.fits (SZ.v bl * SZ.v bs) })
  (eb : EM3.t et bnh bl bs)
  : Lemma
    (ensures
      (forall (i:natlt (SZ.v bnh)) (j:natlt (SZ.v bl)) (k:natlt (SZ.v bs)).
        Seq.index
          (chest_to_seq1 (CH.mk ((bnh *^ bl *^ bs) @| INil)
             (fun idx -> CH.acc eb ((bij_sym (fold_bij_l3 bnh bl bs)).ff idx))))
          ((i * (SZ.v bl * SZ.v bs) + j * SZ.v bs + k) <: natlt (SZ.v (bnh *^ bl *^ bs)))
        == EM3.macc eb i j k))
  = let flat = fold_bij_l3 bnh bl bs in
    sb_flatten_char bnh bl bs eb;
    let cb = CH.mk ((bnh *^ bl *^ bs) @| INil)
               (fun idx -> CH.acc eb ((bij_sym flat).ff idx)) in
    introduce forall (i:natlt (SZ.v bnh)) (j:natlt (SZ.v bl)) (k:natlt (SZ.v bs)).
        Seq.index (chest_to_seq1 cb)
          ((i * (SZ.v bl * SZ.v bs) + j * SZ.v bs + k) <: natlt (SZ.v (bnh *^ bl *^ bs)))
        == EM3.macc eb i j k
    with (
      let idx3 : abs (bnh @| bl @| bs @| INil) = (i,(j,(k,()))) in
      let r : natlt (SZ.v (bnh *^ bl *^ bs)) = i * (SZ.v bl * SZ.v bs) + j * SZ.v bs + k in
      assert (flat.ff idx3 == ((r, ()) <: abs ((bnh *^ bl *^ bs) @| INil)));
      // bij_inv_fwd SMTPat gives flat.gg (flat.ff idx3) == idx3, hence flat.gg (r,()) == idx3
      assert (flat.gg ((r, ()) <: abs ((bnh *^ bl *^ bs) @| INil)) == idx3);
      ()
    )

let ulen_eq_l1_l3 (bnh bl bs : szp { SZ.fits (SZ.v bnh * SZ.v bl * SZ.v bs) })
  : squash (tlayout_ulen (l1_forward (bnh *^ bl *^ bs))
            == tlayout_ulen (l3_batched_row_major bnh bl bs))
  = assert (is_full (l1_forward (bnh *^ bl *^ bs)));
    assert (is_full (l3_batched_row_major bnh bl bs));
    full_layout_size (l1_forward (bnh *^ bl *^ bs));
    full_layout_size (l3_batched_row_major bnh bl bs)

/// The tensor-level [to_seq] of a 1-D chest equals the array1-level [to_seq]
/// of its logical sequence: both reduce to [c] at the inverse-image index.
let to_seq1_eq (#et:Type) (#nn:nat) (lA : A1.full_layout nn) (c : CH.t (nn @| INil) et)
  : Lemma (TL.to_seq lA c == A1.to_seq lA (chest_to_seq1 c))
  = let lhs = TL.to_seq lA c in
    let rhs = A1.to_seq lA (chest_to_seq1 c) in
    introduce forall (i:natlt nn). Seq.index lhs i == Seq.index rhs i
    with (
      Seq.init_ghost_index_ nn
        (fun (j:natlt nn) -> (c.f (Kuiper.Injection.inverse_f lA.imap j)))
        i;
      let x = Kuiper.Injection.inverse_f lA.imap i in
      Seq.init_ghost_index_ nn
        (fun (j:natlt nn) ->
           (let y = Kuiper.Injection.inverse_f lA.imap j in
            Seq.index (chest_to_seq1 c) y._1))
        i;
      Seq.init_ghost_index_ nn
        (fun (j:natlt nn) -> CH.acc c ((j, ()) <: abs (nn @| INil)))
        x._1;
      assert (x == ((x._1, ()) <: abs (nn @| INil)))
    );
    Seq.lemma_eq_intro lhs rhs

/// Bridge: tensor [pts_to] (chest) of a 1-D-descriptor tensor to array1 [pts_to]
/// (lseq), establishing the A1 resource on the fresh handle [A1.from_array lA (core a)]
/// over the same physical core. We keep everything on [Tensor.core]/[A1.from_array]
/// (never equating [Tensor.core] with the abstract [A1.core]).
ghost
fn tensor1_to_a1
  (#et:Type) (#nn:nat)
  (lA : A1.full_layout nn)
  (a : tensor et lA)
  (aTarget : A1.t et lA { aTarget == A1.from_array lA (core a) })
  (#f:perm) (#c : CH.t (nn @| INil) et)
  requires
    on gpu_loc (tensor_pts_to a #f c)
  ensures
    on gpu_loc (A1.pts_to aTarget #f (chest_to_seq1 c))
{
  map_loc gpu_loc
    #(tensor_pts_to a #f c)
    #(A1.pts_to aTarget #f (chest_to_seq1 c))
    fn () {
      tensor_concr a #c #f;
      to_seq1_eq lA c;
      rewrite (core a |-> Frac f (TL.to_seq lA c))
           as (core a |-> Frac f (A1.to_seq lA (chest_to_seq1 c)));
      A1.raise lA (core a) #f #(chest_to_seq1 c);
      rewrite (A1.from_array lA (core a) |-> Frac f (chest_to_seq1 c))
           as (A1.pts_to aTarget #f (chest_to_seq1 c));
    };
}

/// Bridge: array1 [pts_to] (lseq) of a 1-D-descriptor array to tensor [pts_to]
/// (chest), establishing the tensor resource on the supplied handle [gTarget]
/// (which must equal [from_array lA (A1.core a)]) over the same physical core. We
/// keep everything on [A1.core]/[Tensor.from_array].
ghost
fn a1_to_tensor1
  (#et:Type) (#nn:nat)
  (lA : A1.full_layout nn)
  (a : A1.t et lA)
  (gTarget : tensor et lA { gTarget == from_array lA (A1.core a) })
  (#f:perm) (#s : lseq et nn)
  requires
    on gpu_loc (A1.pts_to a #f s)
  ensures
    on gpu_loc (tensor_pts_to gTarget #f (seq1_to_chest s))
{
  map_loc gpu_loc
    #(A1.pts_to a #f s)
    #(tensor_pts_to gTarget #f (seq1_to_chest s))
    fn () {
      A1.lower a #f #s;
      to_seq1_eq lA (seq1_to_chest s);
      chest_seq1_roundtrip s;
      rewrite (A1.core a |-> Frac f (A1.to_seq lA s))
           as (A1.core a |-> Frac f (TL.to_seq lA (seq1_to_chest s)));
      tensor_abs lA (A1.core a) #f #(seq1_to_chest s);
      rewrite (from_array lA (A1.core a) |-> Frac f (seq1_to_chest s))
           as (tensor_pts_to gTarget #f (seq1_to_chest s));
    };
}

let lseq_map_id (#et:Type) (#nn:nat) (s : lseq et nn)
  : Lemma (lseq_map (fun (x:et) -> x) s == s)
  = introduce forall (i:natlt nn). Seq.index (lseq_map (fun (x:et) -> x) s) i == Seq.index s i
    with Seq.init_ghost_index_ nn (fun (j:natlt nn) -> (fun (x:et) -> x) (Seq.index s j)) i;
    Seq.lemma_eq_intro (lseq_map (fun (x:et) -> x) s) s

/// [(tlayout_bij g l).imap] applied to an arbitrary index equals [l.imap] at the
/// inverse-mapped index (just unfolds inj_comp; no roundtrip needed).
let bij_self_imap
  (#r1 #r2:nat) (#d1:idesc r1) (#d2:idesc r2)
  (g : abs d1 =~ abs d2) (l : tlayout d1) (idx : abs d2)
  : Lemma ((tlayout_bij g l).imap.f idx == l.imap.f (g.gg idx))
  = ()

/// The chest produced by relaying a row-major-flattened 1-D chest back to the
/// 3-D row-major descriptor recovers the original 3-D chest, provided the 1-D
/// sequence is exactly the row-major flattening of [eb].
let post_relayout_chest_eq
  (#et:Type)
  (bnh bl bs : szp { SZ.fits (SZ.v bnh * SZ.v bl * SZ.v bs) /\ SZ.fits (SZ.v bl * SZ.v bs) })
  (flatb : (abs (bnh @| bl @| bs @| INil) =~ abs ((bnh *^ bl *^ bs) @| INil)))
  (s : lseq et (bnh *^ bl *^ bs))
  (eb : CH.t (bnh @| bl @| bs @| INil) et)
  : Lemma
    (requires
      (forall (i:natlt (SZ.v (bnh *^ bl *^ bs))).
        Seq.index s i == CH.acc eb (flatb.gg ((i, ()) <: abs ((bnh *^ bl *^ bs) @| INil)))))
    (ensures
      CH.mk (bnh @| bl @| bs @| INil)
        (fun idx -> CH.acc (seq1_to_chest s) (flatb.ff idx))
      == eb)
  = introduce forall (idx : abs (bnh @| bl @| bs @| INil)).
        CH.acc (CH.mk (bnh @| bl @| bs @| INil)
                  (fun idx -> CH.acc (seq1_to_chest s) (flatb.ff idx))) idx
        == CH.acc eb idx
    with (
      let q : natlt (SZ.v (bnh *^ bl *^ bs)) = (flatb.ff idx)._1 in
      assert (flatb.ff idx == ((q, ()) <: abs ((bnh *^ bl *^ bs) @| INil)));
      // bij roundtrip: flatb.gg (flatb.ff idx) == idx
      assert (flatb.gg (flatb.ff idx) == idx);
      // requires instantiated at i = q
      assert (Seq.index s q
              == CH.acc eb (flatb.gg ((q, ()) <: abs ((bnh *^ bl *^ bs) @| INil))));
      // acc of seq1_to_chest at (q,()) is s.[q]
      assert (CH.acc (seq1_to_chest s) ((q, ()) <: abs ((bnh *^ bl *^ bs) @| INil))
              == Seq.index s q);
      // chain everything to the goal explicitly
      assert (flatb.gg ((q, ()) <: abs ((bnh *^ bl *^ bs) @| INil)) == idx)
    );
    CH.lemma_equal_intro (CH.mk (bnh @| bl @| bs @| INil)
                  (fun idx -> CH.acc (seq1_to_chest s) (flatb.ff idx))) eb;
    CH.ext (CH.mk (bnh @| bl @| bs @| INil)
                  (fun idx -> CH.acc (seq1_to_chest s) (flatb.ff idx))) eb

/// EM (2-D) approximation eliminates to per-element.
let approx2_macc (#et:Type) {| scalar et, real_like et |} (#d0 #d1:nat)
  (e : EM.ematrix et d0 d1) (r : EM.ematrix real d0 d1)
  (i:natlt d0) (j:natlt d1)
  : Lemma (requires e %~ r) (ensures EM.macc e i j %~ EM.macc r i j)
  = assert (Kuiper.Chest.chest_approximates e r)

/// Approximation congruence for the per-page batched gemm-with-comb spec.
let bmmcomb_approx
  (#et:Type) {| scalar et, real_like et |}
  (comb : binop et) (comb_r : binop real)
  (#batch #rows #shared #cols : nat)
  (ec : EM3.t et batch rows cols) (rc : EM3.t real batch rows cols)
  (ea : EM3.t et batch rows shared) (ra : EM3.t real batch rows shared)
  (eb : EM3.t et batch shared cols) (rb : EM3.t real batch shared cols)
  : Lemma
    (requires approx2 comb comb_r /\ ec %~ rc /\ ea %~ ra /\ eb %~ rb)
    (ensures MS.bmmcomb comb ec ea eb %~ MS.bmmcomb comb_r rc ra rb)
  = let le = MS.bmmcomb comb ec ea eb in
    let lr = MS.bmmcomb comb_r rc ra rb in
    introduce forall (i:natlt batch) (j:natlt rows) (k:natlt cols).
      EM3.macc le i j k %~ EM3.macc lr i j k
    with (
      slice3_approx ec rc i;
      slice3_approx ea ra i;
      slice3_approx eb rb i;
      GU.mmcomb_approx_real comb comb_r
        (EM3.slice_page ec i) (EM3.slice_page ea i) (EM3.slice_page eb i)
        (EM3.slice_page ra i) (EM3.slice_page rb i) (EM3.slice_page rc i);
      approx2_macc
        (MS.mmcomb comb (EM3.slice_page ec i) (EM3.slice_page ea i) (EM3.slice_page eb i))
        (MS.mmcomb comb_r (EM3.slice_page rc i) (EM3.slice_page ra i) (EM3.slice_page rb i))
        j k
    );
    EM3.lemma_approximates_intro le lr

#pop-options

#pop-options
