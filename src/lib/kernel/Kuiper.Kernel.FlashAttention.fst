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

module M = Kuiper.Array2 
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Index

open Kuiper.Kernel.FlashAttention.KernelDesc

inline_for_extraction noextract
fn flashattention_tile
  (#et : Type0) {| scalar et, floating et |}
  (bc br d: szp)
  (#lKj #lVj: M.layout bc d)
  (#lSt: layout bc)
  (#lQit #lOit: layout d)
  {| ctlayout lSt, ctlayout lKj, ctlayout lVj, ctlayout lQit, ctlayout lOit |}
  (gKj: M.array2 et lKj) 
  (gVj: M.array2 et lVj)
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
      let vk: et = M.read gKj ((vy <: sz), (vx <: sz));
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
      let vv: et = M.read gVj ((vy <: sz), (vx <: sz));
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
  (lK lV lQ: M.layout n d)
  (lOt: M.layout (n /^ br) d)
  (llt lmt: layout (n /^ br))
  {| ctlayout lSt, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lOt, ctlayout llt, ctlayout lmt |}
  (gSt: array1 et lSt)
  (gK: M.array2 et lK) 
  (gV: M.array2 et lV)
  (gQ: M.array2 et lQ)
  (gOt: M.array2 et lOt)
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

      M.extract_row_ro gQ qi;
      let gQit = M.row gQ (SZ.v qi);
      M.extract_row gOt ii #1.0R #eOt; // gO has already been split into per-thread chunks
      let gOit = M.row gOt (SZ.v ii);

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
        gKj gVj gSt (M.row gQ (SZ.v qi)) (M.row gOt (SZ.v ii)) glit gmit;

      array1_cell_from_ref glt ii;
      array1_cell_from_ref gmt ii;
      restore_cell glt ii;
      restore_cell gmt ii;

      M.restore_row gQ (SZ.v qi);
      with (eOit: lseq _ _). assert ((M.row gOt ((SZ.v ii) <: natlt n)) <: (array1 et (M.row_layout gOt (SZ.v ii)))) |-> (Frac 1.0R eOit);
      elim_forall (eOit);
      Trade.elim_trade (((M.row gOt ((SZ.v ii) <: natlt n)) <: (array1 et (M.row_layout gOt (SZ.v ii)))) |-> (Frac 1.0R eOit)) _;

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
  (#lS:M.layout nthr nthr)(#lK #lV #lQ #lO:M.layout n d)(#ll #lm:M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS:M.array2 et lS{M.is_global gS})(gK:M.array2 et lK{M.is_global gK})(gV:M.array2 et lV{M.is_global gV})
  (gQ:M.array2 et lQ{M.is_global gQ})(gO:M.array2 et lO{M.is_global gO})(gl:M.array2 et ll{M.is_global gl})(gm:M.array2 et lm{M.is_global gm})
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
  M.extract_row (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0;
  M.extract_row (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0;
  M.extract_row (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0;

  flashattention_kf_no_smem n d nthr nthr
    _ _ _ _ _ _ _
    #(ctlayout_slice (subtile_layout lS 1 (SZ.v nthr) (SZ.v tid) 0) #(c_subtile_layout lS 1 (SZ.v nthr) (SZ.v tid) 0) 0 0)
    #solve #solve #solve
    #(c_stride_subtile_layout lO (SZ.v nthr) 1 (SZ.v tid) 0)
    #(ctlayout_slice (stride_subtile_layout ll 1 (SZ.v nthr) 0 (SZ.v tid)) #(c_stride_subtile_layout ll 1 (SZ.v nthr) 0 (SZ.v tid)) 0 0)
    #(ctlayout_slice (stride_subtile_layout lm 1 (SZ.v nthr) 0 (SZ.v tid)) #(c_stride_subtile_layout lm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 0)
    (M.row (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0) gK gV gQ
    (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0)
    (M.row (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0)
    (M.row (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0)
    eK eV eQ
    tid;

  // Rebuild gO's strided sub-view (written by the inner kernel).
  with vO. assert (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> (vO <: ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1)));
  let vO' : ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1) = vO;
  rewrite (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> (vO' <: ematrix et (SZ.v n / SZ.v nthr) (SZ.v d / 1)))
       as (array2_stride_subtile gO (SZ.v nthr) 1 (SZ.v tid) 0 |-> ematrix_stride_subtile (update_stride_tile eO (SZ.v nthr) 1 (SZ.v tid) 0 vO') (SZ.v nthr) 1 (SZ.v tid) 0);

  // Rebuild gS's row -> contiguous sub-tile.
  with (vS: lseq _ _). assert ((M.row (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0 <: array1 et (M.row_layout (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0)) |-> Frac 1.0R vS);
  elim_forall (vS);
  Trade.elim_trade ((M.row (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0 <: array1 et (M.row_layout (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0) 0)) |-> Frac 1.0R vS) _;
  rewrite (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> Frac 1.0R (ematrix_upd_row (ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0) 0 vS))
       as (array2_subtile gS 1 (SZ.v nthr) (SZ.v tid) 0 |-> ematrix_subtile (update_tile eS 1 (SZ.v nthr) (SZ.v tid) 0 (ematrix_upd_row (ematrix_subtile eS 1 (SZ.v nthr) (SZ.v tid) 0) 0 vS)) 1 (SZ.v nthr) (SZ.v tid) 0);

  // Rebuild gl's row -> strided sub-view.  Two same-typed trades (gl, gm) are
  // live, so we eliminate via [elim_forall_imp] with explicit predicates.
  with (vl: lseq _ _). assert ((M.row (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (M.row_layout (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R vl);
  Pulse.Lib.Forall.Util.elim_forall_imp
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> (M.row (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (M.row_layout (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R s')
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 s'))
    vl;
  rewrite (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vl))
       as (array2_stride_subtile gl 1 (SZ.v nthr) 0 (SZ.v tid) |-> ematrix_stride_subtile (update_stride_tile el 1 (SZ.v nthr) 0 (SZ.v tid) (ematrix_upd_row (ematrix_stride_subtile el 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vl)) 1 (SZ.v nthr) 0 (SZ.v tid));

  // Rebuild gm's row -> strided sub-view.
  with (vm: lseq _ _). assert ((M.row (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (M.row_layout (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R vm);
  Pulse.Lib.Forall.Util.elim_forall_imp
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> (M.row (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0 <: array1 et (M.row_layout (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid)) 0)) |-> Frac 1.0R s')
    (fun (s':lseq et (SZ.v n / SZ.v nthr)) -> array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 s'))
    vm;
  rewrite (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> Frac 1.0R (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vm))
       as (array2_stride_subtile gm 1 (SZ.v nthr) 0 (SZ.v tid) |-> ematrix_stride_subtile (update_stride_tile em 1 (SZ.v nthr) 0 (SZ.v tid) (ematrix_upd_row (ematrix_stride_subtile em 1 (SZ.v nthr) 0 (SZ.v tid)) 0 vm)) 1 (SZ.v nthr) 0 (SZ.v tid));

  fold (kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ (SZ.v tid));
}

inline_for_extraction noextract
let kflashattention
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr) /\ nthr <= max_blocks * max_threads})
  (#lS:M.layout nthr nthr)(#lK #lV #lQ #lO:M.layout n d)(#ll #lm:M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS:M.array2 et lS{M.is_global gS})(gK:M.array2 et lK{M.is_global gK})(gV:M.array2 et lV{M.is_global gV})
  (gQ:M.array2 et lQ{M.is_global gQ})(gO:M.array2 et lO{M.is_global gO})(gl:M.array2 et ll{M.is_global gl})(gm:M.array2 et lm{M.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  : kernel_desc
      (requires full_io_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ)
      (ensures  full_io_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ)
= {
    nthr = nthr;
    frame = frame_fa n d nthr lS lO ll lm;
    setup = setup_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ;
    teardown = teardown_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ;
    kpre  = (fun (tid:natlt nthr) -> kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid);
    kpost = (fun (tid:natlt nthr) -> kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid);
    f = flashattention_kf_outer n d nthr gS gK gV gQ gO gl gm eK eV eQ;
    kpre_sendable = magic();
    kpost_sendable = magic();
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn flashattention_gpu
  (#et:Type0){| scalar et, floating et |}
  (n d nthr:szp{nthr/?n /\ SZ.fits (nthr*nthr) /\ nthr <= max_blocks * max_threads})
  (#lS:M.layout nthr nthr)(#lK #lV #lQ #lO:M.layout n d)(#ll #lm:M.layout 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS:M.array2 et lS{M.is_global gS})(gK:M.array2 et lK{M.is_global gK})(gV:M.array2 et lV{M.is_global gV})
  (gQ:M.array2 et lQ{M.is_global gQ})(gO:M.array2 et lO{M.is_global gO})(gl:M.array2 et ll{M.is_global gl})(gm:M.array2 et lm{M.is_global gm})
  (eK eV eQ:ematrix et n d)(#fK #fV #fQ:perm)
  preserves cpu
  requires on gpu_loc (full_io_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ)
  ensures  on gpu_loc (full_io_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ)
{
  launch_sync (kflashattention n d nthr gS gK gV gQ gO gl gm eK eV eQ);
}
