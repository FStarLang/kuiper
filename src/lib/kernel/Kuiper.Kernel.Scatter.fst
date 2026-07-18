module Kuiper.Kernel.Scatter

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Shareable
open Kuiper.ForEvery
open FStar.Tactics.Typeclasses

module SZ = Kuiper.SizeT

(* ---------------------------------------------------------------------------
   Index-mapping infrastructure.

   Scatter writes each input cell `i : abs di` to the output cell

     sphi i = abs_set_at2 di do dim (acc eIdx i) i  : abs do

   The whole proof rests on this map being injective, which is what lets us
   split the resources of `out` between the threads (one cell per input cell)
   exactly like a layout's `imap.is_inj` splits the resources of an array.
   --------------------------------------------------------------------------- *)

// abs_set_at2 overwrites coordinate `dim` with `idx` and embeds the rest. Hence
// the `dim`-coordinate of the result is recoverable: two results can only be
// equal if their `idx` arguments agree.
let rec abs_set_at2_inj_idx (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (dim : natlt r)
  (idx1 idx2 : natlt (d2 @! dim)) (x1 x2 : abs d1)
  : Lemma (requires abs_set_at2 d1 d2 dim idx1 x1 == abs_set_at2 d1 d2 dim idx2 x2)
          (ensures  idx1 == idx2)
          (decreases dim)
  = let (_, b1) = x1 <: natlt (d1 @! 0) & abs (tail d1) in
    let (_, b2) = x2 <: natlt (d1 @! 0) & abs (tail d1) in
    abs_set_at2_cons d1 d2 dim idx1 x1;
    abs_set_at2_cons d1 d2 dim idx2 x2;
    if dim = 0 then ()
    else begin
      lemma_at_tail d2 (dim - 1);
      abs_set_at2_inj_idx (tail d1) (tail d2) (dim - 1) idx1 idx2 b1 b2
    end

// The scatter destination map.
let sphi (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eIdx : chest di (szlt (do @! dim))) (i : abs di)
  : GTot (abs do)
  = abs_set_at2 di do dim (acc eIdx i) i

// Injectivity of the destination map, from injectivity of the index chest.
let sphi_inj (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eIdx : chest di (szlt (do @! dim)))
  (eIdxInj : (i : abs di -> (j : abs di { acc eIdx i == acc eIdx j }) -> squash (i == j)))
  (i j : abs di)
  : Lemma (requires sphi #et di do dim eIdx i == sphi #et di do dim eIdx j)
          (ensures  i == j)
  = abs_set_at2_inj_idx di do dim (acc eIdx i) (acc eIdx j) i j;
    eIdxInj i j

// An output cell is "covered" if some input cell scatters into it.
let covered (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eIdx : chest di (szlt (do @! dim))) (x : abs do)
  : prop
  = exists (i : abs di). sphi #et di do dim eIdx i == x

// Ghost inverse of sphi on the covered cells (well-defined by injectivity).
let preimg (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eIdx : chest di (szlt (do @! dim)))
  (x : abs do { covered #et di do dim eIdx x })
  : GTot (i : abs di { sphi #et di do dim eIdx i == x })
  = FStar.IndefiniteDescription.indefinite_description_ghost
      (abs di) (fun i -> sphi #et di do dim eIdx i == x)

// The output produced by scatter: covered cells take the scattered input value,
// everything else is left untouched.
let scatter_out (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eInp : chest di et) (eIdx : chest di (szlt (do @! dim))) (eOut : chest do et)
  : chest do et
  = Kuiper.Chest.mk do (fun (x : abs do) ->
      if FStar.IndefiniteDescription.strong_excluded_middle (covered #et di do dim eIdx x)
      then acc eInp (preimg #et di do dim eIdx x)
      else acc eOut x)

let scatter_out_covered (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eInp : chest di et) (eIdx : chest di (szlt (do @! dim))) (eOut : chest do et)
  (eIdxInj : (i : abs di -> (j : abs di { acc eIdx i == acc eIdx j }) -> squash (i == j)))
  (i : abs di)
  : Lemma (acc (scatter_out di do dim eInp eIdx eOut) (sphi #et di do dim eIdx i) == acc eInp i)
  = let x = sphi #et di do dim eIdx i in
    assert (covered #et di do dim eIdx x);
    let j = preimg #et di do dim eIdx x in
    sphi_inj #et di do dim eIdx eIdxInj j i

let scatter_out_uncovered (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eInp : chest di et) (eIdx : chest di (szlt (do @! dim))) (eOut : chest do et)
  (x : abs do { ~(covered #et di do dim eIdx x) })
  : Lemma (acc (scatter_out di do dim eInp eIdx eOut) x == acc eOut x)
  = ()

(*

LATER: it would be great to have this, however I realized it will not work when going to
the concrete layout, because the concrete layout requires reading from the gIdx tensor,
and that is an impure operation which we currently do not support in layouts.
We would need to carry around the permission to read gIdx in the layout, which is quite an
invasive change.

// A tensor layout for mapping indices through a lookup table.
// For now, the lookup table only stores 1 dimension of indices,
// but it could store tuples so that it maps N-D.
// The LUT can be smaller than the mapped tensor. The resulting tensor
// will then have the same shape as the LUT.
let tlayout_lut
  (#r : erased nat) (#d1 #d2: shape r { shape_le d1 d2 })
  (dim: szlt r)
  (eIdx: chest d1 (szlt (d2 @! (SZ.v dim))))
  (eIdxInj: (i: (abs d1) -> (j: abs d1 {acc eIdx i == acc eIdx j}) -> squash (i == j)))
  (l: tlayout d2)
  : tlayout d1 = {
    ulen = sizeof d1;
    imap = {
      f = (fun i ->
        let i' = abs_set_at2 d1 d2 dim (acc eIdx i) i in
        l.imap.f i');
      is_inj = (fun i j ->
        l.imap.is_inj (abs_set_at2 d1 d2 dim (acc eIdx i) i) (abs_set_at2 d1 d2 dim (acc eIdx j) j);
          .... // TODO
        eIdxInj);
    }
  }
*)

(* ---------------------------------------------------------------------------
   Read-only frame holding the input and index tensors, shared between threads
   exactly like Kuiops.Gather.gather_frame.
   --------------------------------------------------------------------------- *)

unfold
let scatter_frame (#et #it : Type0) (#r : erased nat) (di : shape r) (cdi : cshape di)
  (#lInp #lIdx : tlayout di) {| ctlayout lInp, ctlayout lIdx |}
  (gInp : tensor et lInp { is_global gInp })
  (gIdx : tensor it lIdx { is_global gIdx })
  (eInp : chest di et)
  (eIdx : chest di it)
  (fInp fIdx : perm)
  (fr : perm) : slprop =
    (tensor_pts_to gInp #((fInp /. (fInp +. fIdx)) *. fr) eInp) **
    (tensor_pts_to gIdx #((fIdx /. (fInp +. fIdx)) *. fr) eIdx)

instance scatter_frame_shareable
  (#et #it : Type0) (#r : erased nat) (di : shape r) (cdi : cshape di)
  (#lInp #lIdx : tlayout di) {| ctlayout lInp, ctlayout lIdx |}
  (gInp : tensor et lInp { is_global gInp })
  (gIdx : tensor it lIdx { is_global gIdx })
  (eInp : chest di et)
  (eIdx : chest di it)
  (fInp fIdx : perm) :
  shareable (scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx) =
    double_shareable
      (fun fr -> tensor_pts_to gInp #fr eInp)
      (fun fr -> tensor_pts_to gIdx #fr eIdx)
      (fInp /. (fInp +. fIdx)) (fIdx /. (fInp +. fIdx))

// Pure fact: the destination relation is injective in the thread index. This is
// what lets `out`'s cells be split among threads, and rejoined afterwards.
let sphi_rel_inj (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (dim : natlt r)
  (eIdx : chest di (szlt (do @! dim)))
  (eIdxInj : (i : abs di -> (j : abs di { acc eIdx i == acc eIdx j }) -> squash (i == j)))
  : Lemma (forall (i1 i2 : abs di) (x : abs do).
             (sphi #et di do dim eIdx i1 == x /\ sphi #et di do dim eIdx i2 == x) ==> i1 == i2)
  = let aux (i1 i2 : abs di) (x : abs do)
      : Lemma (requires (sphi #et di do dim eIdx i1 == x /\ sphi #et di do dim eIdx i2 == x))
              (ensures  i1 == i2)
      = sphi_inj #et di do dim eIdx eIdxInj i1 i2 in
    Classical.forall_intro_3 (fun i1 i2 x -> Classical.move_requires (aux i1 i2) x)

(* ---------------------------------------------------------------------------
   The scatter kernel, built as a `kernel_desc_n` à la Kuiper.Kernel.TMap.kmap,
   but writing each input cell `i` to the permuted output cell `sphi i`.
   --------------------------------------------------------------------------- *)

ghost
fn ssetup
  (#et : Type0) (#r : nat) (di do : shape r { shape_le di do })
  (dim : natlt r)
  (n : sz { SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0 })
  (#lInp #lIdx : tlayout di) (#lOut : tlayout do) {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp : tensor et lInp { is_global gInp })
  (gIdx : tensor (szlt (do @! dim)) lIdx { is_global gIdx })
  (gOut : tensor et lOut)
  (cdi : cshape di)
  (eInp : chest di et)
  (eIdx : chest di (szlt (do @! dim)))
  (eIdxInj : (i : abs di -> (j : abs di { acc eIdx i == acc eIdx j }) -> squash (i == j)))
  (fInp fIdx : perm)
  (#eOut : chest do et)
  (#fr : perm)
  ()
  norewrite
  requires
    scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx fr ** (gOut |-> eOut)
  ensures
    (forall+ (i : natlt n).
      scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx (fr /. n) **
      (Cell gOut (sphi #et di do dim eIdx (unflatten di i)) |-> acc eOut (sphi #et di do dim eIdx (unflatten di i)))) **
    pure (SZ.fits (tlayout_ulen lOut)) **
    (forall+ (x : abs do { ~(covered #et di do dim eIdx x) }). Cell gOut (x <: abs do) |-> acc eOut x)
{
  tensor_pts_to_ref gOut;
  tensor_explode gOut;
  forevery_refine_split (fun (x : abs do) -> Cell gOut (x <: abs do) |-> acc eOut x) (covered #et di do dim eIdx);
  sphi_rel_inj #et di do dim eIdx eIdxInj;
  forevery_split_or_n
    (fun (i : abs di) (x : abs do) -> sphi #et di do dim eIdx i == x)
    (fun (x : abs do) -> Cell gOut (x <: abs do) |-> acc eOut x);
  forevery_map
    (fun (i : abs di) -> forall+ (x : abs do { sphi #et di do dim eIdx i == x }). Cell gOut (x <: abs do) |-> acc eOut x)
    (fun (i : abs di) -> Cell gOut (sphi #et di do dim eIdx i) |-> acc eOut (sphi #et di do dim eIdx i))
    fn i {
       forevery_singleton_elim'
         (fun (x : abs do { sphi #et di do dim eIdx i == x }) -> Cell gOut (x <: abs do) |-> acc eOut x)
         (sphi #et di do dim eIdx i)
     };
  forevery_iso (flatten_bij di)
    (fun (i : abs di) -> Cell gOut (sphi #et di do dim eIdx i) |-> acc eOut (sphi #et di do dim eIdx i));
  forevery_rw_size (sizeof di) (SZ.v n);
  share_n (scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx)
    #(scatter_frame_shareable di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx) n;
  forevery_zip
    (fun (_ : natlt n) -> scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx (fr /. n)) _;
}

ghost
fn steardown
  (#et : Type0) (#r : nat) (di do : shape r { shape_le di do })
  (dim : natlt r)
  (n : sz { SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0 })
  (#lInp #lIdx : tlayout di) (#lOut : tlayout do) {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp : tensor et lInp { is_global gInp })
  (gIdx : tensor (szlt (do @! dim)) lIdx { is_global gIdx })
  (gOut : tensor et lOut)
  (cdi : cshape di)
  (eInp : chest di et)
  (eIdx : chest di (szlt (do @! dim)))
  (eIdxInj : (i : abs di -> (j : abs di { acc eIdx i == acc eIdx j }) -> squash (i == j)))
  (fInp fIdx : perm)
  (#eOut : chest do et)
  (#fr : perm)
  ()
  norewrite
  requires
    (forall+ (i : natlt n).
      scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx (fr /. n) **
      (exists* (v : et).
        tensor_pts_to_cell gOut (sphi #et di do dim eIdx (unflatten di i)) v **
        pure (v == acc eInp (unflatten di i)))) **
    pure (SZ.fits (tlayout_ulen lOut)) **
    (forall+ (x : abs do { ~(covered #et di do dim eIdx x) }). Cell gOut (x <: abs do) |-> acc eOut x)
  ensures
    scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx fr **
    (exists* (eOut' : chest do et).
      (gOut |-> eOut') **
      pure (vscatter_chest di do dim eInp eIdx eOut'))
{
  let eOut' = scatter_out di do dim eInp eIdx eOut;
  forevery_unzip _ _;
  gather_n (scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx)
    #(scatter_frame_shareable di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx) n;
  let accs = forevery_exists #(natlt n) _;
  forevery_unzip _ _;
  forevery_elim_pure _;
  forevery_rw_size (SZ.v n) (sizeof di);
  forevery_ext #(natlt (sizeof di))
    (fun i -> tensor_pts_to_cell gOut (sphi #et di do dim eIdx (unflatten di i)) (accs i))
    (fun i -> tensor_pts_to_cell gOut (sphi #et di do dim eIdx (unflatten di i)) (accs (flatten di (unflatten di i))));
  forevery_iso_back (flatten_bij di)
    (fun (i : abs di) -> tensor_pts_to_cell gOut (sphi #et di do dim eIdx i) (accs (flatten di i)));
  forevery_map
    (fun (i : abs di) -> tensor_pts_to_cell gOut (sphi #et di do dim eIdx i) (accs (flatten di i)))
    (fun (i : abs di) -> Cell gOut (sphi #et di do dim eIdx i) |-> acc eOut' (sphi #et di do dim eIdx i))
    fn i {
       scatter_out_covered di do dim eInp eIdx eOut eIdxInj i;
       rewrite
         tensor_pts_to_cell gOut (sphi #et di do dim eIdx i) (accs (flatten di i))
       as
         (Cell gOut (sphi #et di do dim eIdx i) |-> acc eOut' (sphi #et di do dim eIdx i));
     };
  forevery_map
    (fun (i : abs di) -> Cell gOut (sphi #et di do dim eIdx i) |-> acc eOut' (sphi #et di do dim eIdx i))
    (fun (i : abs di) -> forall+ (x : abs do { sphi #et di do dim eIdx i == x }). Cell gOut (x <: abs do) |-> acc eOut' x)
    fn i {
       forevery_singleton_intro'
         (fun (x : abs do { sphi #et di do dim eIdx i == x }) -> Cell gOut (x <: abs do) |-> acc eOut' x)
         (sphi #et di do dim eIdx i)
     };
  sphi_rel_inj #et di do dim eIdx eIdxInj;
  forevery_join_or_n
    (fun (i : abs di) (x : abs do) -> sphi #et di do dim eIdx i == x)
    (fun (x : abs do) -> Cell gOut (x <: abs do) |-> acc eOut' x);
  forevery_map
    (fun (x : abs do { ~(covered #et di do dim eIdx x) }) -> Cell gOut (x <: abs do) |-> acc eOut x)
    (fun (x : abs do { ~(covered #et di do dim eIdx x) }) -> Cell gOut (x <: abs do) |-> acc eOut' x)
    fn x {
       scatter_out_uncovered di do dim eInp eIdx eOut x;
       rewrite (Cell gOut (x <: abs do) |-> acc eOut x) as (Cell gOut (x <: abs do) |-> acc eOut' x);
     };
  forevery_refine_join
    (fun (x : abs do) -> Cell gOut (x <: abs do) |-> acc eOut' x)
    (covered #et di do dim eIdx)
    (fun (x : abs do) -> ~(covered #et di do dim eIdx x));
  forevery_unrefine (fun (x : abs do) -> Cell gOut (x <: abs do) |-> acc eOut' x);
  tensor_implode gOut;
  Classical.forall_intro (scatter_out_covered di do dim eInp eIdx eOut eIdxInj);
  assert pure (vscatter_chest di do dim eInp eIdx eOut');
}

inline_for_extraction noextract
fn skf
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi : cshape di) (cdo : cshape do)
  (dim : szlt r)
  (#lInp #lIdx : tlayout di) (#lOut : tlayout do) {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp : tensor et lInp { is_global gInp })
  (gIdx : tensor (szlt (do @! (SZ.v dim))) lIdx { is_global gIdx })
  (gOut : tensor et lOut)
  (n : sz { SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0 })
  (eInp : chest di et)
  (eIdx : chest di (szlt (do @! (SZ.v dim))))
  (fInp fIdx : perm)
  (pfr : perm)
  (#eOut : chest do et)
  (i : szlt (sizeof di))
  ()
  requires
    gpu **
    scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx pfr **
    (Cell gOut (sphi #et di do (SZ.v dim) eIdx (unflatten di (SZ.v i))) |->
       acc eOut (sphi #et di do (SZ.v dim) eIdx (unflatten di (SZ.v i))))
  ensures
    gpu **
    scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx pfr **
    (exists* (v : et).
      tensor_pts_to_cell gOut (sphi #et di do (SZ.v dim) eIdx (unflatten di (SZ.v i))) v **
      pure (v == acc eInp (unflatten di (SZ.v i))))
{
  let ci = cunflatten cdi i;
  let idx = tensor_read gIdx ci;
  let cphi = conc_set_at2 cdi cdo dim idx ci;
  let v = tensor_read gInp ci;
  rewrite
    (Cell gOut (sphi #et di do (SZ.v dim) eIdx (unflatten di (SZ.v i))) |->
       acc eOut (sphi #et di do (SZ.v dim) eIdx (unflatten di (SZ.v i))))
  as
    (Cell gOut (up cphi) |-> acc eOut (sphi #et di do (SZ.v dim) eIdx (unflatten di (SZ.v i))));
  tensor_write_cell gOut cphi v;
  rewrite
    (Cell gOut (up cphi) |-> v)
  as
    (tensor_pts_to_cell gOut (sphi #et di do (SZ.v dim) eIdx (unflatten di (SZ.v i))) v);
}

inline_for_extraction noextract
let scatter_kernel
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi : cshape di) (cdo : cshape do)
  (dim : szlt r)
  (#lInp #lIdx : tlayout di) (#lOut : tlayout do) {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp : tensor et lInp { is_global gInp })
  (gIdx : tensor (szlt (do @! (SZ.v dim))) lIdx { is_global gIdx })
  (gOut : tensor et lOut)
  (n : sz { SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0 })
  (eInp : chest di et)
  (eIdx : chest di (szlt (do @! (SZ.v dim))))
  (eIdxInj : (i : abs di -> (j : abs di { acc eIdx i == acc eIdx j }) -> squash (i == j)))
  (fInp fIdx : perm)
  (#eOut : chest do et)
  (#fr : perm)
  : kernel_desc
      (requires scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx fr ** (gOut |-> eOut))
      (ensures  scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx fr **
        exists* eOut'. (gOut |-> eOut') ** pure (vscatter_chest di do (SZ.v dim) eInp eIdx eOut'))
= {
    nthr = n;
    f = skf di do cdi cdo dim #lInp #lIdx #lOut gInp gIdx gOut n eInp eIdx fInp fIdx (fr /. n) #eOut;

    frame =
      pure (SZ.fits (tlayout_ulen lOut)) **
      (forall+ (x : abs do { ~(covered #et di do (SZ.v dim) eIdx x) }). Cell gOut (x <: abs do) |-> acc eOut x);
    setup    = ssetup di do (SZ.v dim) n #lInp #lIdx #lOut gInp gIdx gOut cdi eInp eIdx eIdxInj fInp fIdx #eOut #fr;
    teardown = steardown di do (SZ.v dim) n #lInp #lIdx #lOut gInp gIdx gOut cdi eInp eIdx eIdxInj fInp fIdx #eOut #fr;
    kpre  = (fun (i : natlt (sizeof di)) ->
               scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx (fr /. n) **
               (Cell gOut (sphi #et di do (SZ.v dim) eIdx (unflatten di i)) |-> acc eOut (sphi #et di do (SZ.v dim) eIdx (unflatten di i))));
    kpost = (fun (i : natlt (sizeof di)) ->
               scatter_frame di cdi #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx (fr /. n) **
               exists* v. tensor_pts_to_cell gOut (sphi #et di do (SZ.v dim) eIdx (unflatten di i)) v **
                          pure (v == acc eInp (unflatten di i)));
    kpost_sendable = magic ();
    kpre_sendable  = magic ();
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn scatter_gpu
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi: cshape di) (cdo: cshape do)
  (dim: szlt r)
  (#lInp #lIdx: tlayout di) (#lOut: tlayout do) {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (n : sz{SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0})
  (eInp: chest di et)
  (eIdx: chest di (szlt (do @! (SZ.v dim))) { chest_inj di do (SZ.v dim) eIdx })
  (#fInp #fIdx: perm)
  preserves cpu ** on gpu_loc (gInp |-> Frac fInp eInp) ** on gpu_loc (gIdx |-> Frac fIdx (eIdx <: chest di (szlt (do @! (SZ.v dim)))))
  requires on gpu_loc (live gOut)
  ensures
    exists* eOut.
      on gpu_loc (gOut |-> eOut) **
      pure (vscatter_chest di do dim eInp eIdx eOut) {
  // Recover the function-form injectivity witness from the `chest_inj`
  // refinement on `eIdx`. This proof is computationally irrelevant and erased,
  // so neither it nor the refinement reach the extracted kernel or its ABI.
  let eIdxInj : (i: abs di -> (j: abs di {acc eIdx i == acc eIdx j}) -> squash (i == j)) =
    (fun i j -> ());
  with eOut. assert on gpu_loc (gOut |-> eOut);
  launch_sync (scatter_kernel di do cdi cdo dim #lInp #lIdx #lOut gInp gIdx gOut n eInp eIdx eIdxInj fInp fIdx #eOut #(fInp +. fIdx));
}