module Kuiper.Poly.Stencil

#lang-pulse

open Kuiper
module M = Kuiper.Matrix
module STS = Kuiper.Spec.Stencil
module SZ = Kuiper.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs { row_major }

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : mlayout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : mlayout rows cols)
  {| clayout lIn |}
  {| clayout lOut |}
  (gIn : M.gpu_matrix et lIn)
  (fIn : perm)
  (gOut : M.gpu_matrix et lOut)
  (eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  (gIn |-> Frac (fIn /. (rows * cols)) eIn) **
  (exists* vv.
    M.gpu_matrix_pts_to_cell gOut #1.0R (tid / cols) (tid % cols) vv)

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : mlayout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : mlayout rows cols)
  {| clayout lIn |}
  {| clayout lOut |}
  (gIn : M.gpu_matrix et lIn)
  (fIn : perm)
  (gOut : M.gpu_matrix et lOut)
  (eIn : ematrix et (rows +^ 2sz) (cols +^ 2sz))
  (tid : nat{ tid < rows * cols })
  : slprop
  =
  (gIn |-> Frac (fIn /. (rows * cols)) eIn) **
  M.gpu_matrix_pts_to_cell gOut (tid / cols) (tid % cols)
    (STS.stencil_result_at_idx #_ #_ #rows #cols stencil eIn (tid / cols) (tid % cols))

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : mlayout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : mlayout rows cols)
  {| clayout lIn |}
  {| clayout lOut |}
  (gIn : M.gpu_matrix et lIn)
  (#fIn : perm)
  (gOut : M.gpu_matrix et lOut)
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
  let i = bid /^ cols; assert (rewrites_to i (bid /^ cols));
  let j = bid %^ cols; assert (rewrites_to j (bid %^ cols));

  let tl = M.gpu_matrix_read gIn i j; let tm = M.gpu_matrix_read gIn i (j+^1sz); let tr = M.gpu_matrix_read gIn i (j+^2sz);
  let ml = M.gpu_matrix_read gIn (i+^1sz) j; let mm = M.gpu_matrix_read gIn (i+^1sz) (j+^1sz); let mr = M.gpu_matrix_read gIn (i+^1sz) (j+^2sz);
  let bl = M.gpu_matrix_read gIn (i+^2sz) j; let bm = M.gpu_matrix_read gIn (i+^2sz) (j+^1sz); let br = M.gpu_matrix_read gIn (i+^2sz) (j+^2sz);

  let sv =
    (tl `mul` stencil 0 0) `add` (tm `mul` stencil 0 1) `add` (tr `mul` stencil 0 2) `add`
    (ml `mul` stencil 1 0) `add` (mm `mul` stencil 1 1) `add` (mr `mul` stencil 1 2) `add`
    (bl `mul` stencil 2 0) `add` (bm `mul` stencil 2 1) `add` (br `mul` stencil 2 2);

  M.gpu_matrix_write_cell gOut i j sv;
}
#pop-options

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : mlayout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : mlayout rows cols)
  {| clayout lIn |}
  {| clayout lOut |}
  (gIn : M.gpu_matrix et lIn)
  (#fIn : perm)
  (gOut : M.gpu_matrix et lOut)
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
  M.gpu_matrix_share_n gIn (rows *^ cols);

  M.gpu_matrix_explode gOut;

  forevery_unfactor' (rows *^ cols) rows cols (fun r c ->
    M.gpu_matrix_pts_to_cell gOut r c (macc eOut r c));

  ghost
  fn hide_specific_val_behind_exists (rc: natlt (rows *^ cols))
    requires
        M.gpu_matrix_pts_to_cell gOut (rc / cols) (rc % cols) (macc #_ #(SZ.v rows) #(SZ.v cols) eOut (rc / cols) (rc % cols))
    ensures
      (exists* vv. M.gpu_matrix_pts_to_cell gOut (rc / cols) (rc % cols) vv)
  {
    ()
  };

  forevery_map _ _ hide_specific_val_behind_exists;

  forevery_zip #(natlt (rows *^ cols))
    _
    (fun rc ->
      (exists* vv. M.gpu_matrix_pts_to_cell gOut (rc / cols) (rc % cols) vv));
    // M.gpu_matrix_pts_to_cell gOut (i/cols) (i%cols) (macc eOut (i/cols) (i%cols)));

  ghost
  fn hide_specific_val_behind_exists (rc: natlt (rows * cols))
    requires
      M.gpu_matrix_pts_to #et gIn #(fIn /. of_int (SZ.v (SZ.mul rows cols))) eIn **
        M.gpu_matrix_pts_to_cell gOut (rc / cols) (rc % cols) (macc eOut (rc / cols) (rc % cols))
    ensures
      M.gpu_matrix_pts_to #et gIn #(fIn /. of_int (SZ.v (SZ.mul rows cols))) eIn **
      (exists* vv. M.gpu_matrix_pts_to_cell gOut (rc / cols) (rc % cols) vv)
  {
    ()
  };

  (* We're done actually, but the encoding will not match the lambdas. *)
  forevery_ext #(natlt2 rows cols)
    (fun i ->
    (* In the context, fractions are computed by mulitplying SizeT and then converting to nat, *)
      (gIn |-> Frac (fIn /. (rows *^ cols)) eIn) **
      exists* vv. M.gpu_matrix_pts_to_cell gOut (i/cols) (i%cols) vv)
    (* and the goal expects conversion to nat for each factor. *)
    (fun i ->
      kpre stencil gIn fIn gOut eIn i);

}

#push-options "--z3rlimit 60"
ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : mlayout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : mlayout rows cols)
  {| clayout lIn |}
  {| clayout lOut |}
  (gIn : M.gpu_matrix et lIn)
  (#fIn : perm)
  (gOut : M.gpu_matrix et lOut)
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
      M.gpu_matrix_pts_to #et gIn #(fIn /. (v rows * v cols)) eIn);
  M.gpu_matrix_gather_n gIn _;

  forevery_factor (rows *^ cols) rows cols _;

  (* Simplify arithmetic expressions. *)
  forevery_ext_2
    (fun (r:natlt rows) (c:natlt cols) ->
      M.gpu_matrix_pts_to_cell gOut ((r * cols + c) / cols) ((r * cols + c) % cols)
         (STS.stencil_result_at_idx #_ #_ #rows #cols stencil eIn
            ((r * cols + c) / cols) ((r * cols + c) % cols)))
    (fun (r:natlt rows) (c:natlt cols) ->
      M.gpu_matrix_pts_to_cell gOut r c (
        STS.stencil_result_at_idx #_ #_ #rows #cols stencil eIn r c));

  ghost
  fn convert_single_res_to_access_on_entire_res (r:natlt rows) (c:natlt cols)
    requires
      M.gpu_matrix_pts_to_cell gOut r c (STS.stencil_result_at_idx stencil eIn r c)
    ensures
      M.gpu_matrix_pts_to_cell gOut r c (macc (STS.stencil_result stencil eIn) r c)
  {
    ()
  };

  forevery_map_2 #(natlt rows) #(natlt cols)
    (fun r c -> M.gpu_matrix_pts_to_cell gOut r c (STS.stencil_result_at_idx stencil eIn r c))
    _
    convert_single_res_to_access_on_entire_res;

  M.gpu_matrix_implode gOut;
}
#pop-options

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : mlayout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : mlayout rows cols)
  {| clayout lIn |}
  {| clayout lOut |}
  (gIn : M.gpu_matrix et lIn)
  (#fIn : perm)
  (gOut : M.gpu_matrix et lOut)
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
}

inline_for_extraction noextract
fn host_simple_stencil
  (#et : Type0) {| scalar et |}
  (stencil: (i: natlt 3) -> (j: natlt 3) -> et)
  (#rows #cols : szp)
  (#_ : squash (SZ.fits (rows + 2) /\ SZ.fits (cols + 2)))
  (#lIn : mlayout (rows +^ 2sz) (cols +^ 2sz))
  (#lOut : mlayout rows cols)
  {| clayout lIn |}
  {| clayout lOut |}
  (gIn : M.gpu_matrix et lIn { M.is_global_matrix gIn })
  (#fIn : perm)
  (gOut : M.gpu_matrix et lOut { M.is_global_matrix gOut })
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
  (rIn rOut : mrepr)
  {| cIn : crepr rIn |}
  {| cOut : crepr rOut |}
  (#rows #cols : (x:szp{x >= 3}))
  (gIn : M.gpu_matrix et (rIn rows cols) { M.is_global_matrix gIn })
  (gOut : M.gpu_matrix et (rOut (rows - 2) (cols - 2)) { M.is_global_matrix gOut })
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
    let cols_sub2 = cols -^ 2sz;
    host_simple_stencil 
      stencil #(rows -^ 2sz) #cols_sub2 #() #_ #_ #(cIn.map _ _) #(cOut.map _ _) gIn #fIn gOut #eIn #eOut;
  }
