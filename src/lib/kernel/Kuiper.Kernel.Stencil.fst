module Kuiper.Kernel.Stencil

#lang-pulse

open Kuiper
open Kuiper.Array2
open Kuiper.EMatrix
open Kuiper.Tensor { ctlayout }
module Array2 = Kuiper.Array2
module STS = Kuiper.Spec.Stencil
module SZ = Kuiper.SizeT

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : Array2.layout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : Array2.layout rows cols)
  {| ctlayout lIn |}
  {| ctlayout lOut |}
  (gIn : array2 et lIn)
  (fIn : perm)
  (gOut : array2 et lOut)
  (eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  gIn |-> Frac (fIn /. (rows * cols)) eIn **
  (exists* vv.
    Array2.pts_to_cell gOut #1.0R (tid / cols, tid % cols) vv)

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : Array2.layout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : Array2.layout rows cols)
  {| ctlayout lIn |}
  {| ctlayout lOut |}
  (gIn : array2 et lIn)
  (fIn : perm)
  (gOut : array2 et lOut)
  (eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  gIn |-> Frac (fIn /. (rows * cols)) eIn **
  Array2.pts_to_cell gOut (tid / cols, tid % cols)
    (STS.stencil_result_at_idx #_ #_ #rows #cols stencil eIn (tid / cols) (tid % cols))

#push-options "--z3rlimit 15"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : Array2.layout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : Array2.layout rows cols)
  {| ctlayout lIn |}
  {| ctlayout lOut |}
  (gIn : array2 et lIn)
  (#fIn : perm)
  (gOut : array2 et lOut)
  (#eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  (bid : szlt (rows *^ cols))
  ()
  norewrite
  requires
    gpu **
    kpre stencil gIn fIn gOut eIn bid **
    block_id (rows *^ cols) bid
  ensures
    gpu **
    kpost stencil gIn fIn gOut eIn bid **
    block_id (rows *^ cols) bid
{
  let i : sz = bid /^ cols; assert (rewrites_to i (bid /^ cols));
  let j : sz = bid %^ cols; assert (rewrites_to j (bid %^ cols));

  let tl = Array2.read gIn (i,      j); let tm = Array2.read gIn (i,      j+^1sz); let tr = Array2.read gIn (i,      j+^2sz);
  let ml = Array2.read gIn (i+^1sz, j); let mm = Array2.read gIn (i+^1sz, j+^1sz); let mr = Array2.read gIn (i+^1sz, j+^2sz);
  let bl = Array2.read gIn (i+^2sz, j); let bm = Array2.read gIn (i+^2sz, j+^1sz); let br = Array2.read gIn (i+^2sz, j+^2sz);

  let sv =
    (tl `mul` stencil 0 0) `add` (tm `mul` stencil 0 1) `add` (tr `mul` stencil 0 2) `add`
    (ml `mul` stencil 1 0) `add` (mm `mul` stencil 1 1) `add` (mr `mul` stencil 1 2) `add`
    (bl `mul` stencil 2 0) `add` (bm `mul` stencil 2 1) `add` (br `mul` stencil 2 2);

  Array2.write_cell gOut (i, j) sv;
}
#pop-options

#push-options "--z3rlimit 20"
ghost
fn setup
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : Array2.layout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : Array2.layout rows cols)
  {| ctlayout lIn |}
  {| ctlayout lOut |}
  (gIn : array2 et lIn)
  (#fIn : perm)
  (gOut : array2 et lOut)
  (#eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  (#eOut : ematrix et rows cols)
  ()
  norewrite
  requires
    gIn |-> Frac fIn eIn **
    gOut |-> eOut
  ensures
    (forall+ (rc : natlt (rows *^ cols)).
      kpre stencil gIn fIn gOut eIn rc) **
    emp (* frame *)
{
  Array2.share_n gIn (rows *^ cols);

  Array2.ilower gOut;

  forevery_unfactor' (rows *^ cols) rows cols (fun r c ->
    Array2.pts_to_cell gOut (r, c) (macc eOut r c));

  ghost
  fn hide_specific_val_behind_exists (rc: natlt (rows *^ cols))
    requires
        Array2.pts_to_cell gOut ((rc / cols <: natlt rows), (rc % cols <: natlt cols)) (macc #_ #(SZ.v rows) #(SZ.v cols) eOut (rc / cols) (rc % cols))
    ensures
      (exists* vv. Array2.pts_to_cell gOut ((rc / cols <: natlt rows), (rc % cols <: natlt cols)) vv)
  {
    ()
  };

  forevery_map _ _ hide_specific_val_behind_exists;

  forevery_zip #(natlt (rows *^ cols))
    _
    (fun rc ->
      (exists* vv. Array2.pts_to_cell gOut ((rc / cols <: natlt rows), (rc % cols <: natlt cols)) vv));

  ghost
  fn hide_specific_val_behind_exists (rc: natlt (rows * cols))
    requires
      Array2.pts_to #et gIn #(fIn /. of_int (SZ.v (SZ.mul rows cols))) eIn **
        Array2.pts_to_cell gOut ((rc / cols <: natlt rows), (rc % cols <: natlt cols)) (macc eOut (rc / cols) (rc % cols))
    ensures
      Array2.pts_to #et gIn #(fIn /. of_int (SZ.v (SZ.mul rows cols))) eIn **
      (exists* vv. Array2.pts_to_cell gOut ((rc / cols <: natlt rows), (rc % cols <: natlt cols)) vv)
  {
    ()
  };

  (* We're done actually, but the encoding will not match the lambdas. *)
  forevery_ext #(natlt2 rows cols)
    (fun i ->
    (* In the context, fractions are computed by mulitplying SizeT and then converting to nat, *)
      (gIn |-> Frac (fIn /. (rows *^ cols)) eIn) **
      exists* vv. Array2.pts_to_cell gOut ((i/cols <: natlt rows), (i%cols <: natlt cols)) vv)
    (* and the goal expects conversion to nat for each factor. *)
    (fun i ->
      kpre stencil gIn fIn gOut eIn i);

}
#pop-options

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : Array2.layout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : Array2.layout rows cols)
  {| ctlayout lIn |}
  {| ctlayout lOut |}
  (gIn : array2 et lIn)
  (#fIn : perm)
  (gOut : array2 et lOut)
  (#eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  ()
  norewrite
  requires
    (forall+ (tid : natlt (rows *^ cols)).
      kpost stencil gIn fIn gOut eIn tid) **
    emp (* frame *)
  ensures
    gIn |-> Frac fIn eIn **
    (gOut |-> STS.stencil_result stencil eIn)
{
  forevery_unzip #(natlt2 rows cols) _ _;

  forevery_rw_type
    (natlt (v (SizeT.mul rows cols)))
    (natlt (v rows * v cols))
    (fun _ ->
      Array2.pts_to #et gIn #(fIn /. (v rows * v cols)) eIn);
  Array2.gather_n gIn _;

  forevery_factor (rows *^ cols) rows cols _;

  (* Simplify arithmetic expressions. *)
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) / cols == r));
  assert (pure (forall (r c : nat). c < cols ==> (r * cols + c) % cols == c));
  forevery_ext_2
    (fun (r:natlt rows) (c:natlt cols) ->
      Array2.pts_to_cell gOut (((r * cols + c) / cols <: natlt rows), ((r * cols + c) % cols <: natlt cols))
         (STS.stencil_result_at_idx #_ #_ #rows #cols stencil eIn
            ((r * cols + c) / cols) ((r * cols + c) % cols)))
    (fun (r:natlt rows) (c:natlt cols) ->
      Array2.pts_to_cell gOut (r, c) (
        STS.stencil_result_at_idx #_ #_ #rows #cols stencil eIn r c));

  ghost
  fn convert_single_res_to_access_on_entire_res (r:natlt rows) (c:natlt cols)
    requires
      Array2.pts_to_cell gOut (r, c) (STS.stencil_result_at_idx stencil eIn r c)
    ensures
      Array2.pts_to_cell gOut (r, c) (macc (STS.stencil_result stencil eIn) r c)
  {
    ()
  };

  forevery_map_2 #(natlt rows) #(natlt cols)
    (fun r c -> Array2.pts_to_cell gOut (r, c) (STS.stencil_result_at_idx stencil eIn r c))
    _
    convert_single_res_to_access_on_entire_res;

  Array2.iraise gOut;
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : Array2.layout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : Array2.layout rows cols)
  {| ctlayout lIn |}
  {| ctlayout lOut |}
  (gIn : array2 et lIn { Array2.is_global gIn})
  (#fIn : perm)
  (gOut : array2 et lOut { Array2.is_global gOut })
  (#eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  (#eOut : ematrix et rows cols)
  (_ : squash (rows * cols <= max_blocks))
  : kernel_desc_m_1
    (gIn |-> Frac fIn eIn ** gOut |-> eOut)
    (gIn |-> Frac fIn eIn ** gOut |-> STS.stencil_result stencil eIn)
= {
  nblk = rows *^ cols;

  frame = emp;

  setup    = setup    stencil gIn gOut;
  teardown = teardown stencil gIn gOut;

  kpre  = kpre  stencil gIn fIn gOut eIn;
  kpost = kpost stencil gIn fIn gOut eIn;

  f = kf stencil gIn gOut;
  kpost_sendable=solve;
  kpre_sendable=solve;
}

inline_for_extraction noextract
fn host_simple_stencil
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : Array2.layout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : Array2.layout rows cols)
  {| ctlayout lIn |}
  {| ctlayout lOut |}
  (gIn : array2 et lIn { Array2.is_global gIn})
  (#fIn : perm)
  (gOut : array2 et lOut { Array2.is_global gOut })
  (#eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  (#eOut : ematrix et rows cols)
  preserves
    cpu **
    on gpu_loc (gIn |-> Frac fIn eIn)
  requires
    pure (rows * cols <= max_blocks) **
    on gpu_loc (gOut |-> eOut)
  ensures
    on gpu_loc (gOut |-> STS.stencil_result stencil eIn)
{
  launch_sync (kdesc stencil gIn #fIn gOut #eIn #eOut ());
}

inline_for_extraction noextract
fn specialize_host_simple_stencil
  (et: Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (rIn rOut : trepr2)
  {| cIn : ctrepr2 rIn, cOut : ctrepr2 rOut |}
  (rows cols : (x:szp{x >= 3}))
  (gIn : array2 et (rIn rows cols) { Array2.is_global gIn })
  (gOut : array2 et (rOut (rows - 2) (cols - 2)) { Array2.is_global gOut })
  (#fIn : perm)
  (#eIn : ematrix et rows cols)
  (#eOut : ematrix et (rows - 2) (cols - 2))
  preserves
    cpu **
    on gpu_loc (gIn |-> Frac fIn eIn)
  requires
    pure (rows * cols <= max_blocks) **
    on gpu_loc (gOut |-> eOut)
  ensures
    on gpu_loc (gOut |-> STS.stencil_result stencil eIn)
  {
    let rows_sub2 = rows -^ 2sz;
    let cols_sub2 = cols -^ 2sz;
    host_simple_stencil stencil #rows_sub2 #cols_sub2 #() #_ #_ #(cIn.inst _ _) #(cOut.inst _ _) gIn gOut;
  }
