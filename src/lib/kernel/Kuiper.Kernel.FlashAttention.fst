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
#push-options "--split_queries always --print_implicits"
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
  (tid: sz { tid <^ br /\ tid <^ bc }) // TODO: impossible to materialize tid in a kernel unless br = bc
  (#fK #fV #fQ: perm)
  preserves 
    gpu ** // preserved so this can sit in kernel_desc.f
    (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
    live gSt ** live gOt ** live glt ** live gmt// No functional spec; note that O, l, m would have preconditions here though. S does not
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
        #_ #_ #_ #(ctlayout_slice _ 0sz qi) #(ctlayout_slice _ 0sz ii)
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
#pop-options

(* ─────────────────────────────────────────────────────────────────────────
   kernel_desc and host launch.

   Configuration: 1 block, [br] threads.  We require [bc == br] so that
   every thread id [tid : szlt br] is also valid as [szlt bc] (the
   [flashattention_kf_no_smem] function refines [tid] by both bounds).

   Per-thread resources (in [kpre tid]):
     - [gK |-> Frac (fK /. br) eK]                         (sharded)
     - [gV |-> Frac (fV /. br) eV]                         (sharded)
     - [gQ |-> Frac (fQ /. br) eQ]                         (sharded)
     - per-thread strided sub-views [gOt, glt, gmt]
       (rows {i*br + tid for i = 0..n/br}).
     - row [tid] of the shmem S matrix, viewed as the [array1] [gSt].

   The strided extraction of (n/br) rows starting at offset tid with
   stride br is non-trivial; the ghost proofs are stubbed with [admit()]
   for now. The host wrapper, [flashattention_launch], wires everything
   into [launch_sync].
   ───────────────────────────────────────────────────────────────────── *)

open Kuiper.SHMem

(* The shmem layout we use for S: a single [SHArray et (bc * br)] viewed
   as a [br * bc] row-major matrix, so that row [tid] is the per-thread
   scratch slice expected by [flashattention_kf_no_smem]. *)

inline_for_extraction noextract
let fa_shmems (et : Type0) {| Kuiper.Sized.sized et |}
  (bc br : szp { SZ.fits (bc * br) })
  : list shmem_desc
  = [SHArray et (bc *^ br)]

(* The Array2 layout we use to view the S shmem array. *)
inline_for_extraction noextract
let fa_lS (bc br : szp) : M.layout br bc = l2_row_major br bc

inline_for_extraction noextract
instance fa_lS_ct (bc br : szp { SZ.fits (bc * br) })
  : ctlayout (fa_lS bc br) = c_l2_row_major _ _

(* Lift the raw shmem array to an Array2 view. *)
inline_for_extraction noextract
let fa_gS
  (#et : Type0) {| Kuiper.Sized.sized et |}
  (bc br : szp { SZ.fits (bc * br) })
  (sh : c_shmems (fa_shmems et bc br))
  : M.array2 et (fa_lS bc br)
  = M.from_array (fa_lS bc br) sh._1

(* Per-thread strided slice. Conceptually thread [tid] owns:
     - the rows {i * br + tid | i = 0 .. n/br - 1} of [gO]
     - the cells {i * br + tid | i = 0 .. n/br - 1} of [gl], [gm]
   These are exposed below via an abstract per-thread "view" type
   [fa_thread_view]. The actual realisation (using Kuiper.Array2.Strided
   subtile layouts) is left as TODO in [setup_fa] / [block_setup_fa]; the
   important thing here is that the slprop is well-formed and gets
   re-bundled symmetrically in [kpost_fa]. *)

inline_for_extraction noextract
unfold
let kpre_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK)
  (gV : M.array2 et lV)
  (gQ : M.array2 et lQ)
  (#lO : M.layout n d)
  (gO : M.array2 et lO)
  (#ll #lm : layout n)
  (gl : array1 et ll)
  (gm : array1 et lm)
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  (sh : c_shmems (fa_shmems et bc br))
  (tid : natlt br)
  : slprop
  =
  (gK |-> Frac (fK /. br) eK) **
  (gV |-> Frac (fV /. br) eV) **
  (gQ |-> Frac (fQ /. br) eQ) **
  (* Per-thread write-side strided sub-tiles. The actual sub-layouts
     (subtile_layout of [lO], [ll], [lm] with stride [br] / offset [tid])
     and their corresponding [array2]/[array1] handles are produced by
     [block_setup_fa]; here they appear under an existential. *)
  (exists* (lOt : M.layout (n /^ br) d)
           (gOt : M.array2 et lOt).
     live gOt) **
  (exists* (llt : layout (n /^ br))
           (glt : array1 et llt).
     live glt) **
  (exists* (lmt : layout (n /^ br))
           (gmt : array1 et lmt).
     live gmt) **
  (* This thread's row of the shmem S matrix. *)
  live (M.row (fa_gS bc br sh) tid)

inline_for_extraction noextract
unfold
let kpost_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK)
  (gV : M.array2 et lV)
  (gQ : M.array2 et lQ)
  (#lO : M.layout n d)
  (gO : M.array2 et lO)
  (#ll #lm : layout n)
  (gl : array1 et ll)
  (gm : array1 et lm)
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  (sh : c_shmems (fa_shmems et bc br))
  (tid : natlt br)
  : slprop
  =
  (* Same shape as kpre; the kernel has no functional spec so [live] suffices *)
  kpre_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh tid

(* Per-thread kf: extracts the row of the shmem S matrix and calls into
   flashattention_kf_no_smem. *)
inline_for_extraction noextract
fn fa_kf
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  {| ctlayout lK, ctlayout lV, ctlayout lQ |}
  (gK : M.array2 et lK)
  (gV : M.array2 et lV)
  (gQ : M.array2 et lQ)
  (#lO : M.layout n d) {| ctlayout lO |}
  (gO : M.array2 et lO)
  (#ll #lm : layout n) {| ctlayout ll, ctlayout lm |}
  (gl : array1 et ll)
  (gm : array1 et lm)
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  (sh : c_shmems (fa_shmems et bc br))
  (bid : szlt 1sz)
  (tid : szlt br)
  ()
  requires
    gpu **
    kpre_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh (SZ.v tid) **
    thread_id br tid **
    block_id 1sz bid **
    Kuiper.Barrier.barrier_tok (Kuiper.Barrier.empty_contract br) **
    Kuiper.Barrier.barrier_state 0
  ensures
    gpu **
    kpost_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh (SZ.v tid) **
    thread_id br tid **
    block_id 1sz bid **
    Kuiper.Barrier.barrier_tok (Kuiper.Barrier.empty_contract br) **
    Kuiper.Barrier.barrier_state 0
{
  (* Pull the per-thread strided sub-tiles out of the existentials in
     [kpre_fa] (it's marked [unfold] so the existentials are exposed).
     Then extract the per-thread row of the shmem [S] matrix and invoke
     [flashattention_kf_no_smem]. [kpost_fa == kpre_fa] so re-bundling
     happens automatically. *)
  admit ()
}

(* Outer setup/teardown for the full kernel_desc. nblk = 1, so the
   [forall+ bid : natlt 1. block_pre bid] is just [block_pre 0]. *)
ghost
fn setup_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK)
  (gV : M.array2 et lV)
  (gQ : M.array2 et lQ)
  (#lO : M.layout n d)
  (gO : M.array2 et lO)
  (#ll #lm : layout n)
  (gl : array1 et ll)
  (gm : array1 et lm)
  (eK eV eQ eO : ematrix et n d)
  (vl vm : erased (lseq et n))
  (fK fV fQ : perm)
  ()
  norewrite
  requires
    (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
    (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm)
  ensures
    (forall+ (_bid : natlt 1).
      (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
      (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm)) **
    emp
{
  forevery_singleton_intro #(natlt 1) (fun _bid ->
    (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
    (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm));
}

ghost
fn teardown_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK)
  (gV : M.array2 et lV)
  (gQ : M.array2 et lQ)
  (#lO : M.layout n d)
  (gO : M.array2 et lO)
  (#ll #lm : layout n)
  (gl : array1 et ll)
  (gm : array1 et lm)
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  ()
  norewrite
  requires
    (forall+ (_bid : natlt 1).
      (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
      (exists* (eO' : ematrix et n d). gO |-> eO') **
      (exists* (vl' : lseq et n). gl |-> vl') **
      (exists* (vm' : lseq et n). gm |-> vm')) **
    emp
  ensures
    (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
    (exists* (eO' : ematrix et n d). gO |-> eO') **
    (exists* (vl' : lseq et n). gl |-> vl') **
    (exists* (vm' : lseq et n). gm |-> vm')
{
  forevery_singleton_elim #(natlt 1) _;
}

(* Block-level setup/teardown: split shmem S matrix into per-thread rows;
   shard read-only perms; explode write-side matrices into per-thread
   strided sub-tiles; bundle into [kpre_fa tid]. *)
ghost
fn block_setup_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK)
  (gV : M.array2 et lV)
  (gQ : M.array2 et lQ)
  (#lO : M.layout n d)
  (gO : M.array2 et lO)
  (#ll #lm : layout n)
  (gl : array1 et ll)
  (gm : array1 et lm)
  (eK eV eQ eO : ematrix et n d)
  (vl vm : erased (lseq et n))
  (fK fV fQ : perm)
  (sh : c_shmems (fa_shmems et bc br))
  (_bid : natlt 1)
  ()
  norewrite
  requires
    live_c_shmems sh **
    ((gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
     (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm))
  ensures
    (forall+ (tid : natlt br).
       kpre_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh tid) **
    emp
{
  admit ()
}

ghost
fn block_teardown_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK)
  (gV : M.array2 et lV)
  (gQ : M.array2 et lQ)
  (#lO : M.layout n d)
  (gO : M.array2 et lO)
  (#ll #lm : layout n)
  (gl : array1 et ll)
  (gm : array1 et lm)
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  (sh : c_shmems (fa_shmems et bc br))
  (_bid : natlt 1)
  ()
  norewrite
  requires
    (forall+ (tid : natlt br).
       kpost_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh tid) **
    emp
  ensures
    live_c_shmems sh **
    ((gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
     (exists* (eO' : ematrix et n d). gO |-> eO') **
     (exists* (vl' : lseq et n). gl |-> vl') **
     (exists* (vm' : lseq et n). gm |-> vm'))
{
  admit ()
}

(* Full kernel descriptor: 1 block × br threads, with a shmem S matrix,
   no barrier. *)
inline_for_extraction noextract
let fa_kdesc
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) /\
                     br <= max_threads })
  (#lK #lV #lQ : M.layout n d)
  {| ctlayout lK, ctlayout lV, ctlayout lQ |}
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (#lO : M.layout n d) {| ctlayout lO |}
  (gO : M.array2 et lO { M.is_global gO })
  (#ll #lm : layout n) {| ctlayout ll, ctlayout lm |}
  (gl : array1 et ll { Array1.is_global gl })
  (gm : array1 et lm { Array1.is_global gm })
  (#eK #eV #eQ : ematrix et n d)
  (#eO : ematrix et n d)
  (#vl #vm : erased (lseq et n))
  (#fK #fV #fQ : perm)
  : kernel_desc
      ((gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
       (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm))
      ((gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
       (exists* (eO' : ematrix et n d). gO |-> eO') **
       (exists* (vl' : lseq et n). gl |-> vl') **
       (exists* (vm' : lseq et n). gm |-> vm'))
  =
  {
    nblk             = 1sz;
    nthr             = br;

    shmems_desc      = fa_shmems et bc br;

    (* No barrier used. *)
    barrier_contract = (fun _bid _sh -> Kuiper.Barrier.empty_contract br);
    barrier_count    = (fun _bid -> 0);
    barrier_ok       = (fun _bid _sh -> Kuiper.Barrier.empty_barrier_transform br);

    kpre             = (fun sh _bid tid ->
                          kpre_fa  n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh tid);
    kpost            = (fun sh _bid tid ->
                          kpost_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh tid);

    f                = (fun sh bid tid ->
                          fa_kf n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh bid tid);

    frame            = emp;

    block_pre        = (fun _bid ->
                          (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
                          (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm));
    block_post       = (fun _bid ->
                          (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
                          (exists* (eO' : ematrix et n d). gO |-> eO') **
                          (exists* (vl' : lseq et n). gl |-> vl') **
                          (exists* (vm' : lseq et n). gm |-> vm'));

    setup            = setup_fa    n d bc br gK gV gQ gO gl gm eK eV eQ eO vl vm fK fV fQ;
    teardown         = teardown_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ;

    block_frame      = (fun _sh _bid -> emp);
    block_setup      = (fun sh bid -> block_setup_fa    n d bc br gK gV gQ gO gl gm eK eV eQ eO vl vm fK fV fQ sh bid);
    block_teardown   = (fun sh bid -> block_teardown_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh bid);

    block_pre_sendable  = magic ();
    block_post_sendable = magic ();
    kpre_sendable       = magic ();
    kpost_sendable      = magic ();
  }

(* Host-side launch. Mirrors flash.cu's [forward] but for one head: the
   caller is responsible for iterating over batch/head. *)
inline_for_extraction noextract
fn flashattention_launch
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) /\
                     br <= max_threads })
  (#lK #lV #lQ : M.layout n d)
  {| ctlayout lK, ctlayout lV, ctlayout lQ |}
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (#lO : M.layout n d) {| ctlayout lO |}
  (gO : M.array2 et lO { M.is_global gO })
  (#ll #lm : layout n) {| ctlayout ll, ctlayout lm |}
  (gl : array1 et ll { Array1.is_global gl })
  (gm : array1 et lm { Array1.is_global gm })
  (#eK #eV #eQ : ematrix et n d)
  (#eO : ematrix et n d)
  (#vl #vm : erased (lseq et n))
  (#fK #fV #fQ : perm)
  preserves
    cpu **
    on gpu_loc ((gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ))
  requires
    on gpu_loc ((gO |-> eO) ** (gl |-> vl) ** (gm |-> vm))
  ensures
    on gpu_loc
      ((exists* (eO' : ematrix et n d). gO |-> eO') **
       (exists* (vl' : lseq et n). gl |-> vl') **
       (exists* (vm' : lseq et n). gm |-> vm'))
{
  launch_sync (fa_kdesc n d bc br gK gV gQ gO gl gm)
}

(*

open Kuiper.Tensor.Layout.Alg

let flashattention_tile_f32 =
  flashattention_tile #f32
  32sz 32sz 128sz
  (l2_row_major _ _)
  (l2_row_major _ _)
  (l2_row_major _ _)
  (l2_row_major _ _)
  (l2_row_major _ _)
  #(c_l2_row_major _ _)
  #(c_l2_row_major _ _)
  #(c_l2_row_major _ _)
  #(c_l2_row_major _ _)
  #(c_l2_row_major _ _)
*)