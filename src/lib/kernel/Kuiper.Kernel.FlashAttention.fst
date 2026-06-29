module Kuiper.Kernel.FlashAttention

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Array
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Tiling
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, c_l2_row_major }

module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade
module Array1 = Kuiper.Array1
module B = Kuiper.Barrier
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Array1
open Kuiper.Shape

open Kuiper.Kernel.FlashAttention.KernelDesc

inline_for_extraction noextract
fn flashattention_tile
  (#et : Type0) {| scalar et, floating et |}
  (bc br d: szp)
  (#lKj #lVj: layout2 bc d)
  (#lSt: layout bc)
  (#lQit #lOit: layout d)
  {| ctlayout lSt, ctlayout lKj, ctlayout lVj, ctlayout lQit, ctlayout lOit |}
  (gKj: array2 et lKj) 
  (gVj: array2 et lVj)
  (gSt: array1 et lSt)
  (gQit: array1 et lQit)
  (gOit: array1 et lOit)
  (glit gmit: ref et)
  (#eKj #eVj: ematrix et bc d)
  (#vQit #vOit: erased (lseq et d))
  (#vlit #vmit: erased et)
  (#fKj #fVj #fQit: perm)
  requires 
    gOit |-> vOit ** glit |-> vlit ** gmit |-> vmit
  preserves 
    (gKj |-> Frac fKj eKj) ** (gVj |-> Frac fVj eVj) ** (gQit |-> Frac fQit vQit) **
    live gSt
  ensures 
    live gOit ** live glit ** live gmit // No functional spec
{
  let row_m_prev = !gmit;
  let row_l_prev = !glit;
  let mut row_m: et = neg infinity;
  let mut y: szle bc = 0sz;
  
  while (!y <^ bc) 
    invariant live y ** live row_m ** live gSt
    decreases (bc - !y)
  {
	  let mut sum: et = zero;
    let mut x: szle d = 0sz;
    while (!x <^ d)
      invariant live x ** live sum
      decreases (d - !x)
    {
      assert pure (!x <^ d);
      let vx = !x; let vy = !y;
      let vq: et = read gQit vx;
      let vk: et = tensor_read gKj ((vy <: szlt bc), ((vx <: szlt d), ()));
      sum := !sum `add` (vq `mul` vk);
      x := !x +^ 1sz;
    };
    // TODO: add softmax scale factor
    // sum := !sum * alpha;

    let vy = !y;
    gSt.(vy) <- !sum;
    row_m := fmax !row_m !sum;

    y := !y +^ 1sz;
  };

  let mut row_l: et = zero;
  y := 0sz;
  while (!y <^ bc)
    invariant live y ** live row_l ** live gSt
    decreases (bc - !y)
  {
    let vy = !y;
    let vs: et = (fexp gSt.(vy)) `sub` !row_m;
    gSt.(vy) <- vs;
    row_l := !row_l `add` vs;

    y := !y +^ 1sz;
  };

  let row_m_new = fmax row_m_prev !row_m;
  let row_l_new = row_l_prev `mul` (fexp (row_m_prev `sub` row_m_new)) `add` (!row_l `mul` (fexp (!row_m `sub` row_m_new)));

  let mut x: sz = 0sz;
  while (!x <^ d) 
    invariant live x ** live gOit
    decreases (d - !x) 
  {
    let mut pv: et = zero;
    y := 0sz;
    while (!y <^ bc)
      invariant live y ** live pv
      decreases (bc - !y)
    {
      let vx = !x; let vy = !y;
      let vs: et = gSt.(vy);
      let vv: et = tensor_read gVj ((vy <: szlt bc), ((vx <: szlt d), ()));
      pv := !pv `add` (vs `mul` vv);

      y := !y +^ 1sz;
    };

    let vx = !x;
    let vo: et = gOit.(vx);
    let vo: et = (vo `mul` row_l_prev `mul` (fexp (row_m_prev `sub` row_m_new))) `div` row_l_new;
    let vo: et = vo `add` ((fexp (!row_m `sub` row_m_new)) `mul` !pv);

    gOit.(vx) <- vo;

    x := !x +^ 1sz;
  };

  glit := row_l_new;
  gmit := row_m_new;

  ()
}

// flash attention kernel executed by each thread (no shared memory caching)
inline_for_extraction noextract 
fn flashattention_kf_no_smem (#et : Type0) {| scalar et, floating et |}
  (n d: szp)
  (bc br: szp { bc /? n /\ br /? n })
  (lSt: layout bc)
  (lK lV lQ: layout2 n d)
  (lOt: layout2 (n /^ br) d)
  (llt lmt: layout (n /^ br))
  {| ctlayout lSt, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lOt, ctlayout llt, ctlayout lmt |}
  (gSt: array1 et lSt)
  (gK: array2 et lK) 
  (gV: array2 et lV)
  (gQ: array2 et lQ)
  (gOt: array2 et lOt)
  (glt: array1 et llt)
  (gmt: array1 et lmt)
  (eK eV eQ: ematrix et n d)
  (#fK #fV #fQ: perm)
  (tid: sz { tid <^ br /\ tid <^ bc }) // TODO: impossible to materialize tid in a kernel unless br = bc
  preserves 
    gpu ** 
    kpre_post_inner_fa n d bc br lSt lK lV lQ lOt llt lmt gSt gK gV gQ gOt glt gmt eK eV eQ #fK #fV #fQ
{
  let tc = n /^ bc;
  let tr = n /^ br;
  let mut j: szle tc = 0sz;

  while (!j <^ tc) 
    invariant live j ** live gSt ** live gOt ** live glt ** live gmt
    decreases (tc - !j)
  {
    let gKj = array2_extract_tile_ro' gK (SZ.v bc) (SZ.v d) (SZ.v !j) 0;
    let gVj = array2_extract_tile_ro' gV (SZ.v bc) (SZ.v d) (SZ.v !j) 0;

    let mut i: szle tr = 0sz;
    while (!i <^ tr)
      invariant live i ** live gSt ** live gOt ** live glt ** live gmt
      decreases (tr - !i)
    {
      with eOt. assert gOt |-> eOt;
      with eSt. assert gSt |-> eSt;
      with vlt. assert glt |-> vlt;
      with vmt. assert gmt |-> vmt;
      let ii = !i;
      let qi = br *^ ii +^ tid;

      mextract_row_ro gQ qi;
      let gQit = mrow gQ (SZ.v qi);
      mextract_row gOt ii #1.0R #eOt; // gO has already been split into per-thread chunks
      let gOit = mrow gOt (SZ.v ii);

      extract_cell glt ii #1.0R #vlt;
      array1_cell_to_ref glt ii;
      let glit = get_ref_of_array_cell glt ii;
      assert rewrites_to glit (ref_of_array_cell glt ii);
      
      extract_cell gmt ii #1.0R #vmt;
      array1_cell_to_ref gmt ii;
      let gmit = get_ref_of_array_cell gmt ii;
      assert rewrites_to gmit (ref_of_array_cell gmt ii);

      flashattention_tile bc br d
        #_ #_ #_ #_ #_
        #_ #_ #_ #(ctlayout_slice _ (SZ.v 0sz) (SZ.v qi)) #(ctlayout_slice _ (SZ.v 0sz) (SZ.v ii))
        gKj gVj gSt (mrow gQ (SZ.v qi)) (mrow gOt (SZ.v ii)) glit gmit;

      array1_cell_from_ref glt ii;
      array1_cell_from_ref gmt ii;
      restore_cell glt ii;
      restore_cell gmt ii;

      mrestore_row gQ (SZ.v qi);
      with (eOit: lseq _ _). assert ((mrow gOt ((SZ.v ii) <: natlt n)) <: (array1 et (mrow_layout gOt (SZ.v ii)))) |-> (Frac 1.0R eOit);
      elim_forall (eOit);
      Trade.elim_trade (((mrow gOt ((SZ.v ii) <: natlt n)) <: (array1 et (mrow_layout gOt (SZ.v ii)))) |-> (Frac 1.0R eOit)) _;

      i := !i +^ 1sz; 
    };
    
    Trade.elim_trade (gKj |-> Frac fK (ematrix_subtile eK (SZ.v bc) (SZ.v d) (SZ.v !j) 0)) _;
    Trade.elim_trade (gVj |-> Frac fV (ematrix_subtile eV (SZ.v bc) (SZ.v d) (SZ.v !j) 0)) _;

    j := !j +^ 1sz;
  }
}
// The strided analogue of [subtile_of_update_tile] (which lives, exported,
// in Kuiper.EMatrix.Tiling): reading back the tile we just wrote yields it.
// The stride version is private to KernelDesc.fst, so we re-prove it here.
#push-options "--split_queries always --z3rlimit 40"
let fa_subtile_of_update_stride_tile
  (#et : _)(#rows #cols : _)
  (em : ematrix et rows cols)
  (srows : pos {srows /? rows})(scols : pos {scols /? cols})
  (tr : natlt srows)(tc : natlt scols)
  (etile : ematrix et (rows/srows) (cols/scols))
  : Lemma (ematrix_stride_subtile (update_stride_tile em srows scols tr tc etile) srows scols tr tc == etile)
          [SMTPat (ematrix_stride_subtile (update_stride_tile em srows scols tr tc etile) srows scols tr tc)]
  = let lhs = ematrix_stride_subtile (update_stride_tile em srows scols tr tc etile) srows scols tr tc in
    introduce forall (i:natlt (rows/srows)) (j:natlt (cols/scols)). macc lhs i j == macc etile i j
    with (
      FStar.Math.Lemmas.lemma_mod_plus tr i srows;
      FStar.Math.Lemmas.lemma_div_plus tr i srows;
      FStar.Math.Lemmas.small_mod tr srows;
      FStar.Math.Lemmas.small_div tr srows;
      FStar.Math.Lemmas.lemma_mod_plus tc j scols;
      FStar.Math.Lemmas.lemma_div_plus tc j scols;
      FStar.Math.Lemmas.small_mod tc scols;
      FStar.Math.Lemmas.small_div tc scols
    );
    assert (equal lhs etile)
#pop-options

// Per-thread "outer" wrapper: instantiates the inner no-shmem kernel with
// bc = br = nthr, threading the strided per-thread sub-views from
// kpre_post_outer_fa into the contiguous per-thread arrays the inner expects.
inline_for_extraction noextract
fn flashattention_kf_outer (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr)})
  (#lS:layout2 nthr nthr)(#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS:array2 et lS)(gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  (tid:szlt nthr)
  ()
  preserves gpu ** kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid)
{
  assert pure (SZ.v (n /^ nthr) == SZ.v n / SZ.v nthr);
  assert pure (SZ.v d / 1 == SZ.v d);

  unfold (kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid));
  with eS eO el em. assert (
    array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0 **
    array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid) <: ematrix et (1 / 1) (SZ.v n / SZ.v nthr)) **
    array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid) <: ematrix et (1 / 1) (SZ.v n / SZ.v nthr)) **
    array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> (ematrix_stride_subtile eO (SZ.v nthr) 1 (SZ.v tid) 0 <: ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1)));

  // Extract the single rows the inner kernel needs as array1's.
  mextract_row (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0;
  mextract_row (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0;
  mextract_row (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0;

  flashattention_kf_no_smem n d nthr nthr
    _ _ _ _ _ _ _
    #(ctlayout_slice (subtile_layout lS 1 (SZ.v nthr) (SZ.v tid) 0) #(c_subtile_layout lS 1 (SZ.v nthr) (SZ.v tid) 0) 0 0)
    #solve #solve #solve
    #(c_stride_subtile_layout lO (SZ.v nthr) 1 (SZ.v tid) 0)
    #(ctlayout_slice (stride_subtile_layout ll 1 (SZ.v nthr) 0 (SZ.v tid)) #(c_stride_subtile_layout ll 1 (SZ.v nthr) 0 (SZ.v tid)) 0 0)
    #(ctlayout_slice (stride_subtile_layout lm 1 (SZ.v nthr) 0 (SZ.v tid)) #(c_stride_subtile_layout lm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 0)
    (mrow (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0) gK gV gQ
    (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0)
    (mrow (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0)
    (mrow (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0)
    eK eV eQ
    tid;

  // Rebuild gO's strided sub-view (written by the inner kernel).
  with vO. assert (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> (vO <: ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1)));
  let vO' : ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1) = vO;
  rewrite (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> (vO' <: ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1)))
       as (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> ematrix_stride_subtile (update_stride_tile eO (SZ.v nthr) 1 (SZ.v tid) 0 vO') (SZ.v nthr) 1 (SZ.v tid) 0);

  // Rebuild gS's row -> contiguous sub-tile.
  with (vS: lseq _ _). assert ((mrow (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0 <: array1 et (mrow_layout (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0)) |-> Frac 1.0R vS);
  elim_forall (vS);
  Trade.elim_trade ((mrow (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0 <: array1 et (mrow_layout (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0)) |-> Frac 1.0R vS) _;
  rewrite (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> Frac 1.0R (ematrix_upd_row (ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0) 0 vS))
       as (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> ematrix_subtile (update_tile eS 1 (SZ.v nthr) (SZ.v tid) 0 (ematrix_upd_row (ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0) 0 vS)) 1 (SZ.v nthr) (SZ.v tid) 0);

  // Rebuild gl's row -> strided sub-view.  Two same-typed trades (gl, gm) are
  // live, so we eliminate via [elim_forall_imp] with explicit predicates.
  with (vl: lseq _ _). assert ((mrow (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (mrow_layout (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R vl);
  Pulse.Lib.Forall.Util.elim_forall_imp
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> (mrow (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (mrow_layout (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R s')
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 s'))
    vl;
  rewrite (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vl))
       as (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> ematrix_stride_subtile (update_stride_tile el 1 (SZ.v nthr) 0 (SZ.v tid) (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vl)) 1 (SZ.v nthr) 0 (SZ.v tid));

  // Rebuild gm's row -> strided sub-view.
  with (vm: lseq _ _). assert ((mrow (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (mrow_layout (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R vm);
  Pulse.Lib.Forall.Util.elim_forall_imp
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> (mrow (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (mrow_layout (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R s')
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 s'))
    vm;
  rewrite (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vm))
       as (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> ematrix_stride_subtile (update_stride_tile em 1 (SZ.v nthr) 0 (SZ.v tid) (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vm)) 1 (SZ.v nthr) 0 (SZ.v tid));

  fold (kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid));
}

// Per-thread kernel adapter: matches the full [kernel_desc] [f] field
// signature (with the trivial empty barrier) and delegates to the existing
// per-thread outer kernel, viewing the block's shared scratch as gS.
inline_for_extraction noextract
fn flashattention_kf
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr)})
  (lS:full_layout2 nthr nthr)(#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  (sh : c_shmems (shmems_desc_fa et nthr))
  (bid : szlt 1sz)
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre_post_outer_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid) **
    thread_id nthr tid **
    block_id 1sz bid **
    B.barrier_tok (B.empty_contract nthr) **
    B.barrier_state 0
  ensures
    gpu **
    kpre_post_outer_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid) **
    thread_id nthr tid **
    block_id 1sz bid **
    B.barrier_tok (B.empty_contract nthr) **
    B.barrier_state 0
{
  flashattention_kf_outer n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ tid ();
}

// gpu-level setup / teardown: a single block (nblk = 1), so [block_pre] /
// [block_post] are just the host I/O, with the size-fact frame.
ghost
fn kflashattention_setup
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr)})
  (lS:full_layout2 nthr nthr)(#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  ()
  norewrite
  requires
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
  ensures
    (forall+ (bid:natlt 1sz).
       full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ) **
    frame_fa n d nthr lS lO ll lm
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ);
}

ghost
fn kflashattention_teardown
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr)})
  (lS:full_layout2 nthr nthr)(#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  ()
  norewrite
  requires
    (forall+ (bid:natlt 1sz).
       full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ) **
    frame_fa n d nthr lS lO ll lm
  ensures
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ);
}

// FlashAttention kernel: 1 block, [nthr] threads, scratch gS in shared memory.
// No barriers (each thread owns a disjoint row of gS).
inline_for_extraction noextract
let kflashattention
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr) /\ nthr <= max_threads})
  (lS:full_layout2 nthr nthr)(#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  : kernel_desc
      (requires full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ)
      (ensures  full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ)
= {
    nblk = 1sz;
    nthr = nthr;
    shmems_desc = shmems_desc_fa et nthr;
    barrier_contract = (fun _ _ -> B.empty_contract nthr);
    barrier_count    = (fun _ -> 0);
    barrier_ok       = (fun _ _ -> B.empty_barrier_transform nthr);
    frame = frame_fa n d nthr lS lO ll lm;
    block_pre  = (fun _ -> full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ);
    block_post = (fun _ -> full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ);
    block_frame = (fun _sh _bid -> frame_fa n d nthr lS lO ll lm);
    setup    = kflashattention_setup n d nthr lS gK gV gQ gO gl gm eK eV eQ;
    teardown = kflashattention_teardown n d nthr lS gK gV gQ gO gl gm eK eV eQ;
    kpre  = (fun sh _bid tid -> kpre_post_outer_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid);
    kpost = (fun sh _bid tid -> kpre_post_outer_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid);
    block_setup    = block_setup_fa n d nthr lS gK gV gQ gO gl gm eK eV eQ;
    block_teardown = block_teardown_fa n d nthr lS gK gV gQ gO gl gm eK eV eQ;
    f = flashattention_kf n d nthr lS gK gV gQ gO gl gm eK eV eQ;
    block_pre_sendable  = (fun _ -> magic());
    block_post_sendable = (fun _ -> magic());
    kpre_sendable  = (fun _ _ _ _ -> magic());
    kpost_sendable = (fun _ _ _ _ -> magic());
  }

inline_for_extraction noextract
fn flashattention_gpu
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr) /\ nthr <= max_threads})
  (lS:full_layout2 nthr nthr)(#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  preserves cpu
  requires on gpu_loc (full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ)
  ensures  on gpu_loc (full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ)
{
  launch_sync (kflashattention n d nthr lS gK gV gQ gO gl gm eK eV eQ);
}

(* ═════════════════════════════════════════════════════════════════════════
   SHARED-MEMORY VARIANT: K, V, Q cached in shared memory, with a real barrier
   around the inner loop.  [sK]/[sV] are genuinely shared (every thread reads
   the whole tile), so they go through the barrier; [sQ] is per-thread scratch
   (each thread only reads its own row), so no barrier is needed for it.
   ───────────────────────────────────────────────────────────────────────── *)

// Inner loop (over the row-tiles [i]) for the shared-memory variant.  The K/V
// tile already lives in shared memory ([sK]/[sV], read at frac 1/nthr); each
// thread loads its own global Q row into its private scratch [sQrow] and runs
// [flashattention_tile].  Clean layouts ([lOt] has [d] columns) so the row
// extract/restore of [gOt] matches without the [d/1] coercion (which is
// absorbed at the call boundary, cf. flashattention_kf_outer -> _kf_no_smem).
#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn flashattention_inner_smem
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n})
  (#lKV:full_layout2 nthr d)(#lQ:layout2 n d)(#lSt:layout nthr)(#lQrow:layout d)
  (#lOt:layout2 (n /^ nthr) d)(#llt #lmt:layout (n /^ nthr))
  {| ctlayout lKV, ctlayout lQ, ctlayout lSt, ctlayout lQrow, ctlayout lOt, ctlayout llt, ctlayout lmt |}
  (sK sV:array2 et lKV)(gSt:array1 et lSt)(sQrow:array1 et lQrow)
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gOt:array2 et lOt)(glt:array1 et llt)(gmt:array1 et lmt)
  (eQ:ematrix et n d)(#fQ:perm)(tid:szlt nthr)
  preserves
    gpu **
    (exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x) **
    (exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y) **
    (gQ |-> Frac fQ eQ) **
    live gSt ** live sQrow ** live gOt ** live glt ** live gmt
{
  assert pure (SZ.v (n /^ nthr) == SZ.v n / SZ.v nthr);
  let tr = n /^ nthr;
  let mut i: szle tr = 0sz;
  while (!i <^ tr)
    invariant exists* (vi:szle tr).
      i |-> vi **
      live gSt ** live gOt ** live glt ** live gmt ** live sQrow **
      (gQ |-> Frac fQ eQ) **
      (exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x) **
      (exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y)
    decreases (SZ.v tr - SZ.v !i)
  {
    with eOt. assert gOt |-> eOt;
    let ii = !i;
    assert pure (SZ.v ii < SZ.v tr);
    let qi = nthr *^ ii +^ tid;
    assert pure (SZ.v nthr * SZ.v ii <= SZ.v n);
    assert pure (SZ.v qi == SZ.v nthr * SZ.v ii + SZ.v tid);
    assert pure (SZ.v qi < SZ.v n);

    // LOAD Q: copy global row [qi] into this thread's private scratch row.
    let mut cq: szle d = 0sz;
    while (!cq <^ d)
      invariant exists* (vcq:szle d).
        cq |-> vcq ** live sQrow ** (gQ |-> Frac fQ eQ)
      decreases (SZ.v d - SZ.v !cq)
    {
      let vcq = !cq;
      let vq = tensor_read gQ ((qi <: szlt n), ((vcq <: szlt d), ()));
      (sQrow.(vcq) <- vq);
      cq := !cq +^ 1sz;
    };

    mextract_row gOt ii #1.0R #eOt;

    with vlt. assert glt |-> vlt;
    extract_cell glt ii #1.0R #vlt;
    array1_cell_to_ref glt ii;
    let glit = get_ref_of_array_cell glt ii;
    assert rewrites_to glit (ref_of_array_cell glt ii);

    with vmt. assert gmt |-> vmt;
    extract_cell gmt ii #1.0R #vmt;
    array1_cell_to_ref gmt ii;
    let gmit = get_ref_of_array_cell gmt ii;
    assert rewrites_to gmit (ref_of_array_cell gmt ii);

    with x. assert (sK |-> Frac (1.0R /. (SZ.v nthr)) (x <: ematrix et (SZ.v nthr) (SZ.v d)));
    with y. assert (sV |-> Frac (1.0R /. (SZ.v nthr)) (y <: ematrix et (SZ.v nthr) (SZ.v d)));

    flashattention_tile nthr nthr d
      #_ #_ #_ #_ #_
      #_ #_ #_ #_ #(ctlayout_slice lOt 0 (SZ.v ii))
      sK sV gSt sQrow (mrow gOt (SZ.v ii)) glit gmit;

    array1_cell_from_ref glt ii;
    array1_cell_from_ref gmt ii;
    restore_cell glt ii;
    restore_cell gmt ii;

    with (eOit: lseq _ _). assert ((mrow gOt ((SZ.v ii) <: natlt (SZ.v n / SZ.v nthr))) <: (array1 et (mrow_layout gOt (SZ.v ii)))) |-> (Frac 1.0R eOit);
    elim_forall (eOit);
    Trade.elim_trade (((mrow gOt ((SZ.v ii) <: natlt (SZ.v n / SZ.v nthr))) <: (array1 et (mrow_layout gOt (SZ.v ii)))) |-> (Frac 1.0R eOit)) _;

    i := !i +^ 1sz;
  }
}
#pop-options

// Per-thread f-adapter, matching the full kernel_desc [f] field with the REAL
// (content-free) barrier contract.  Inlines the per-thread sub-view extraction
// (cf. flashattention_kf_outer) and the outer K/V-tile loop, adding K/V/Q
// shared caching with two barriers around the inner loop.
#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn flashattention_kf_smem
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr) /\ SZ.fits (nthr*d)})
  (lS:full_layout2 nthr nthr)(lKV:full_layout2 nthr d)
  (#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lKV, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  (bid : szlt 1sz)
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre_post_outer_fa_smem n d nthr
      (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh)
      gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid) **
    thread_id nthr tid **
    block_id 1sz bid **
    B.barrier_tok (fa_barrier_contract n d nthr (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh)) **
    B.barrier_state 0
  ensures
    gpu **
    kpre_post_outer_fa_smem n d nthr
      (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh)
      gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid) **
    thread_id nthr tid **
    block_id 1sz bid **
    B.barrier_tok (fa_barrier_contract n d nthr (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh)) **
    B.barrier_state (2 * SZ.v (n /^ nthr))
{
  let gS = gS_of_sh' n d nthr lS sh;
  let sK = sK_of_sh n d nthr lKV sh;
  let sV = sV_of_sh n d nthr lKV sh;
  let sQ = sQ_of_sh n d nthr lKV sh;
  rewrite each (gS_of_sh' n d nthr lS sh) as gS;
  rewrite each (sK_of_sh n d nthr lKV sh) as sK;
  rewrite each (sV_of_sh n d nthr lKV sh) as sV;
  rewrite each (sQ_of_sh n d nthr lKV sh) as sQ;

  unfold (kpre_post_outer_fa_smem n d nthr gS sK sV sQ gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid));

  // Euclidean facts.
  assert pure (SZ.v (n /^ nthr) == SZ.v n / SZ.v nthr);
  assert pure (SZ.v d / 1 == SZ.v d);
  FStar.Math.Lemmas.lemma_div_mod (SZ.v n) (SZ.v nthr);
  assert pure (SZ.v n % SZ.v nthr == 0);
  assert pure (SZ.v nthr * SZ.v (n /^ nthr) == SZ.v n);

  // ── Per-thread sub-view extraction (cf. flashattention_kf_outer). ──────
  unfold (kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid));
  with eS eO el em. assert (
    array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0 **
    array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid) <: ematrix et (1 / 1) (SZ.v n / SZ.v nthr)) **
    array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid) <: ematrix et (1 / 1) (SZ.v n / SZ.v nthr)) **
    array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> (ematrix_stride_subtile eO (SZ.v nthr) 1 (SZ.v tid) 0 <: ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1)));

  mextract_row (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0;
  mextract_row (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0;
  mextract_row (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0;

  let gSt = mrow (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0;
  let gOt = array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0;
  let glt = mrow (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0;
  let gmt = mrow (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0;
  rewrite each (mrow (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0) as gSt;
  rewrite each (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0) as gOt;
  rewrite each (mrow (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0) as glt;
  rewrite each (mrow (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0) as gmt;

  // ── Extract this thread's private Q scratch row (no barrier). ──────────
  with rq0. assert (array2_subtile sQ 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R (rq0 <: ematrix et 1 (SZ.v d)));
  mextract_row (array2_subtile sQ 1 (SZ.v d) (SZ.v tid) 0) 0;
  let sQrow = mrow (array2_subtile sQ 1 (SZ.v d) (SZ.v tid) 0) 0;
  rewrite each (mrow (array2_subtile sQ 1 (SZ.v d) (SZ.v tid) 0) 0) as sQrow;

  let tc = n /^ nthr;
  let mut j: szle tc = 0sz;

  while (!j <^ tc)
    invariant exists* (vj:szle tc).
      j |-> vj **
      live gSt ** live gOt ** live glt ** live gmt ** live sQrow **
      (gK |-> Frac (fK /. (SZ.v nthr)) eK) **
      (gV |-> Frac (fV /. (SZ.v nthr)) eV) **
      (gQ |-> Frac (fQ /. (SZ.v nthr)) eQ) **
      (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R r) **
      (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R r) **
      thread_id nthr tid **
      B.barrier_tok (fa_barrier_contract n d nthr sK sV) **
      B.barrier_state (2 * SZ.v vj)
    decreases (SZ.v tc - SZ.v !j)
  {
    let vj = !j;
    assert pure (SZ.v vj < SZ.v tc);
    let kr = nthr *^ vj +^ tid;
    assert pure (SZ.v nthr * SZ.v vj <= SZ.v n);
    assert pure (SZ.v kr == SZ.v nthr * SZ.v vj + SZ.v tid);
    assert pure (SZ.v kr < SZ.v n);

    // LOAD: thread tid copies global row [kr] of K and V into its shared rows.
    let mut c: szle d = 0sz;
    while (!c <^ d)
      invariant exists* (vc:szle d).
        c |-> vc **
        (gK |-> Frac (fK /. (SZ.v nthr)) eK) **
        (gV |-> Frac (fV /. (SZ.v nthr)) eV) **
        (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R r) **
        (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R r)
      decreases (SZ.v d - SZ.v !c)
    {
      let vc = !c;
      let vk = tensor_read gK ((kr <: szlt n), ((vc <: szlt d), ()));
      tensor_write (array2_subtile sK 1 (SZ.v d) (SZ.v tid) 0) ((0sz <: szlt 1), ((vc <: szlt d), ())) vk;
      let vv = tensor_read gV ((kr <: szlt n), ((vc <: szlt d), ()));
      tensor_write (array2_subtile sV 1 (SZ.v d) (SZ.v tid) 0) ((0sz <: szlt 1), ((vc <: szlt d), ())) vv;
      c := !c +^ 1sz;
    };

    // ── EVEN barrier (it = 2*vj): give our rows, receive a fractional read. ─
    even_2x (SZ.v vj);
    assert pure (2 * SZ.v vj < 2 * SZ.v tc);
    assert pure (even (2 * SZ.v vj));
    rewrite ((exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R r) **
             (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R r))
         as ((fa_barrier_contract n d nthr sK sV).rin (2 * SZ.v vj) (SZ.v tid));
    B.barrier_wait ();
    rewrite ((fa_barrier_contract n d nthr sK sV).rout (2 * SZ.v vj) (SZ.v tid))
         as ((exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x) **
             (exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y));

    // ── Inner loop: read the whole shared K/V tile at frac 1/nthr. ─────────
    flashattention_inner_smem n d nthr
      #_ #_ #_ #_ #_ #_ #_
      #_ #_
      #(ctlayout_slice (subtile_layout lS 1 (SZ.v nthr) (SZ.v tid) 0) #(c_subtile_layout lS 1 (SZ.v nthr) (SZ.v tid) 0) 0 0)
      #(ctlayout_slice (subtile_layout lKV 1 (SZ.v d) (SZ.v tid) 0) #(c_subtile_layout lKV 1 (SZ.v d) (SZ.v tid) 0) 0 0)
      #(c_stride_subtile_layout lO (SZ.v nthr) 1 (SZ.v tid) 0)
      #(ctlayout_slice (stride_subtile_layout ll 1 (SZ.v nthr) 0 (SZ.v tid)) #(c_stride_subtile_layout ll 1 (SZ.v nthr) 0 (SZ.v tid)) 0 0)
      #(ctlayout_slice (stride_subtile_layout lm 1 (SZ.v nthr) 0 (SZ.v tid)) #(c_stride_subtile_layout lm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 0)
      sK sV gSt sQrow gQ gOt glt gmt eQ tid;

    // ── ODD barrier (it = 2*vj+1): give our read back, receive our rows. ───
    odd_2x1 (SZ.v vj);
    assert pure (2 * SZ.v vj + 1 < 2 * SZ.v tc);
    assert pure (odd (2 * SZ.v vj + 1));
    rewrite ((exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x) **
             (exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y))
         as ((fa_barrier_contract n d nthr sK sV).rin (2 * SZ.v vj + 1) (SZ.v tid));
    B.barrier_wait ();
    rewrite ((fa_barrier_contract n d nthr sK sV).rout (2 * SZ.v vj + 1) (SZ.v tid))
         as ((exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R r) **
             (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) (SZ.v tid) 0 |-> Frac 1.0R r));

    assert pure (2 * SZ.v vj + 1 + 1 == 2 * (SZ.v vj + 1));
    j := !j +^ 1sz;
  };

  // ── Restore the private Q row back into its subtile. ───────────────────
  with (vQf: lseq _ _). assert (sQrow |-> Frac 1.0R vQf);
  elim_forall (vQf);
  Trade.elim_trade (sQrow |-> Frac 1.0R vQf) _;

  // Revert the per-thread sub-view locals to their unfolded forms so the
  // epilogue rebuilds (copied from flashattention_kf_outer) match the trades.
  rewrite each gOt as (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0);
  rewrite each gSt as (mrow (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0);
  rewrite each glt as (mrow (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0);
  rewrite each gmt as (mrow (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0);

  // ── Rebuild gO/gS/gl/gm sub-views (cf. flashattention_kf_outer epilogue). ─
  with vO. assert (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> (vO <: ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1)));
  let vO' : ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1) = vO;
  rewrite (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> (vO' <: ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1)))
       as (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> ematrix_stride_subtile (update_stride_tile eO (SZ.v nthr) 1 (SZ.v tid) 0 vO') (SZ.v nthr) 1 (SZ.v tid) 0);

  with (vS: lseq _ _). assert ((mrow (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0 <: array1 et (mrow_layout (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0)) |-> Frac 1.0R vS);
  elim_forall (vS);
  Trade.elim_trade ((mrow (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0 <: array1 et (mrow_layout (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0)) |-> Frac 1.0R vS) _;
  rewrite (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> Frac 1.0R (ematrix_upd_row (ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0) 0 vS))
       as (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> ematrix_subtile (update_tile eS 1 (SZ.v nthr) (SZ.v tid) 0 (ematrix_upd_row (ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0) 0 vS)) 1 (SZ.v nthr) (SZ.v tid) 0);

  with (vl: lseq _ _). assert ((mrow (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (mrow_layout (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R vl);
  Pulse.Lib.Forall.Util.elim_forall_imp
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> (mrow (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (mrow_layout (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R s')
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 s'))
    vl;
  rewrite (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vl))
       as (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> ematrix_stride_subtile (update_stride_tile el 1 (SZ.v nthr) 0 (SZ.v tid) (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vl)) 1 (SZ.v nthr) 0 (SZ.v tid));

  with (vm: lseq _ _). assert ((mrow (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (mrow_layout (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R vm);
  Pulse.Lib.Forall.Util.elim_forall_imp
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> (mrow (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (mrow_layout (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R s')
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 s'))
    vm;
  rewrite (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vm))
       as (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> ematrix_stride_subtile (update_stride_tile em 1 (SZ.v nthr) 0 (SZ.v tid) (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vm)) 1 (SZ.v nthr) 0 (SZ.v tid));

  // Revert the shared-cache locals to their [_of_sh] forms so the folds (which
  // normalise the let-bound locals) match the function's pre/postcondition.
  rewrite each gS as (gS_of_sh' n d nthr lS sh);
  rewrite each sK as (sK_of_sh n d nthr lKV sh);
  rewrite each sV as (sV_of_sh n d nthr lKV sh);
  rewrite each sQ as (sQ_of_sh n d nthr lKV sh);

  fold (kpre_post_outer_fa n d nthr (gS_of_sh' n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid));
  fold (kpre_post_outer_fa_smem n d nthr (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid));
}
#pop-options

// Shared-memory FlashAttention kernel: 1 block, [nthr] threads, with K/V/Q
// cached in shared memory and a real barrier around the inner loop.
inline_for_extraction noextract
let kflashattention_smem
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr) /\ SZ.fits (nthr*d) /\ nthr <= max_threads})
  (lS:full_layout2 nthr nthr)(lKV:full_layout2 nthr d)(#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lKV, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  : kernel_desc
      (requires full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ)
      (ensures  full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ)
= {
    nblk = 1sz;
    nthr = nthr;
    shmems_desc = shmems_desc_fa_smem et n d nthr;
    barrier_contract = (fun _bid sh -> fa_barrier_contract n d nthr (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh));
    barrier_count    = (fun _ -> 2 * SZ.v (n /^ nthr));
    barrier_ok       = (fun _bid sh -> fa_barrier_ok n d nthr (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh));
    frame = frame_fa n d nthr lS lO ll lm;
    block_pre  = (fun _ -> full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ);
    block_post = (fun _ -> full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ);
    block_frame = (fun _sh _bid -> frame_fa_smem n d nthr lS lKV lO ll lm);
    setup    = kflashattention_setup n d nthr lS gK gV gQ gO gl gm eK eV eQ;
    teardown = kflashattention_teardown n d nthr lS gK gV gQ gO gl gm eK eV eQ;
    kpre  = (fun sh _bid tid -> kpre_post_outer_fa_smem n d nthr (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid);
    kpost = (fun sh _bid tid -> kpre_post_outer_fa_smem n d nthr (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid);
    block_setup    = block_setup_fa_smem n d nthr lS lKV gK gV gQ gO gl gm eK eV eQ;
    block_teardown = block_teardown_fa_smem n d nthr lS lKV gK gV gQ gO gl gm eK eV eQ;
    f = flashattention_kf_smem n d nthr lS lKV gK gV gQ gO gl gm eK eV eQ;
    block_pre_sendable  = (fun _ -> magic());
    block_post_sendable = (fun _ -> magic());
    kpre_sendable  = (fun _ _ _ _ -> magic());
    kpost_sendable = (fun _ _ _ _ -> magic());
  }

inline_for_extraction noextract
fn flashattention_smem_gpu
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr) /\ SZ.fits (nthr*d) /\ nthr <= max_threads})
  (lS:full_layout2 nthr nthr)(lKV:full_layout2 nthr d)(#lK #lV #lQ #lO:layout2 n d)(#ll #lm:layout2 1 n)
  {| ctlayout lS, ctlayout lKV, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK:array2 et lK{Kuiper.Tensor.is_global gK})(gV:array2 et lV{Kuiper.Tensor.is_global gV})
  (gQ:array2 et lQ{Kuiper.Tensor.is_global gQ})(gO:array2 et lO{Kuiper.Tensor.is_global gO})(gl:array2 et ll{Kuiper.Tensor.is_global gl})(gm:array2 et lm{Kuiper.Tensor.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  preserves cpu
  requires on gpu_loc (full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ)
  ensures  on gpu_loc (full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ)
{
  launch_sync (kflashattention_smem n d nthr lS lKV gK gV gQ gO gl gm eK eV eQ);
}