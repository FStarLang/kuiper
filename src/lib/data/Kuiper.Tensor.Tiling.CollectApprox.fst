module Kuiper.Tensor.Tiling.CollectApprox
#lang-pulse

#set-options "--admit_smt_queries true"

(* An API for tiling matrices, implemented with array views. *)


#push-options "--z3rlimit 40"
(* Combinator for teardown of approximate kernels.
   Collects per-cell existentials into a matrix-level existential,
   and transforms per-cell approximation facts into matrix-level approximation.

   Usage: After separating gA/gB, call this on the remaining forall+ of cells
   to get a single existential matrix that approximates the spec.

   TODO: this needs a thourough review. *)
ghost
fn array2_collect_approx_tiled
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (ntr : nat { ntr == rows / trows })
  (ntc : nat { ntc == cols / tcols })
  (spec_fn : natlt rows -> natlt cols -> et -> prop)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+ (bid : natlt (ntr * ntc)) (tid : natlt (trows * tcols)).
      exists* (v : et).
        tensor_pts_to_cell
          (array2_subtile gm trows tcols (bid / ntc) (bid % ntc))
          (idx2 (tid / tcols <: natlt trows) (tid % tcols <: natlt tcols)) v **
        pure (spec_fn ((bid / ntc) * trows + (tid / tcols))
                      ((bid % ntc) * tcols + (tid % tcols)) v)
  returns vf : (natlt (ntr * ntc) -> natlt (trows * tcols) -> GTot et)
  ensures
    gm |-> mkM (fun (row : natlt rows) (col : natlt cols) ->
      vf ((row / trows) * ntc + (col / tcols)) ((row % trows) * tcols + (col % tcols))) **
    pure (forall (row : natlt rows) (col : natlt cols).
      spec_fn row col
        (vf ((row / trows) * ntc + (col / tcols)) ((row % trows) * tcols + (col % tcols))))
{
  (* Step 1: Collect the existential witnesses using 2D forevery_exists *)
  let vf = forevery_exists_2
    (fun (bid : natlt (ntr * ntc)) (tid : natlt (trows * tcols)) (v : et) ->
      let tr = bid / ntc in
      let tc = bid % ntc in
      let i = tid / tcols in
      let j = tid % tcols in
      tensor_pts_to_cell
        (array2_subtile gm trows tcols tr tc)
        (idx2 (i <: natlt trows) (j <: natlt tcols)) v **
      pure (spec_fn (tr * trows + i) (tc * tcols + j) v));

  (* Step 2: Extract pure facts *)
  forevery_extract_pure_2
    #(natlt (ntr * ntc)) #(natlt (trows * tcols))
    (fun bid tid ->
      let tr = bid / ntc in
      let tc = bid % ntc in
      let i = tid / tcols in
      let j = tid % tcols in
      tensor_pts_to_cell
        (array2_subtile gm trows tcols tr tc)
        (idx2 (i <: natlt trows) (j <: natlt tcols)) (vf bid tid) **
      pure (spec_fn (tr * trows + i) (tc * tcols + j) (vf bid tid)))
    (fun bid tid ->
      let tr = bid / ntc in
      let tc = bid % ntc in
      let i = tid / tcols in
      let j = tid % tcols in
      spec_fn (tr * trows + i) (tc * tcols + j) (vf bid tid))
    fn bid tid { (); };

  (* Step 3: Drop pures *)
  forevery_map_2
    #(natlt (ntr * ntc)) #(natlt (trows * tcols))
    (fun bid tid ->
      let tr = bid / ntc in
      let tc = bid % ntc in
      let i = tid / tcols in
      let j = tid % tcols in
      tensor_pts_to_cell
        (array2_subtile gm trows tcols tr tc)
        (idx2 (i <: natlt trows) (j <: natlt tcols)) (vf bid tid) **
      pure (spec_fn (tr * trows + i) (tc * tcols + j) (vf bid tid)))
    (fun bid tid ->
      let tr = bid / ntc in
      let tc = bid % ntc in
      let i = tid / tcols in
      let j = tid % tcols in
      tensor_pts_to_cell
        (array2_subtile gm trows tcols tr tc)
        (idx2 (i <: natlt trows) (j <: natlt tcols)) (vf bid tid))
    fn bid tid { () };

  (* Step 4: Factor to 4D *)
  forevery_factor_2
    (ntr * ntc) ntr ntc
    (trows * tcols) trows tcols
    (fun (bid : natlt (ntr * ntc)) (tid : natlt (trows * tcols)) ->
      tensor_pts_to_cell
        (array2_subtile gm trows tcols (bid / ntc) (bid % ntc))
        (idx2 (tid / tcols <: natlt trows) (tid % tcols <: natlt tcols)) (vf bid tid));

  (* Simplify div/mod *)
  assert pure (forall (tr:natlt ntr) (tc:natlt ntc). (tr * ntc + tc) / ntc == tr /\ (tr * ntc + tc) % ntc == tc);
  assert pure (forall (i:natlt trows) (j:natlt tcols). (i * tcols + j) / tcols == i /\ (i * tcols + j) % tcols == j);

  forevery_ext_4
    (fun (tr:natlt ntr) (tc:natlt ntc) (i:natlt trows) (j:natlt tcols) ->
      let bid = tr * ntc + tc in let tid = i * tcols + j in
      tensor_pts_to_cell
        (array2_subtile gm trows tcols (bid / ntc) (bid % ntc))
        (idx2 (tid / tcols <: natlt trows) (tid % tcols <: natlt tcols)) (vf bid tid))
    (fun (tr:natlt ntr) (tc:natlt ntc) (i:natlt trows) (j:natlt tcols) ->
      tensor_pts_to_cell
        (array2_subtile gm trows tcols tr tc)
        (idx2 (i <: natlt trows) (j <: natlt tcols)) (vf (tr * ntc + tc) (i * tcols + j)));

  (* Step 5: Rewrite sizes for implode_tiled *)
  forevery_rw_size4 ntr (rows / trows) ntc (cols / tcols) trows trows tcols tcols;

  (* Step 6: Implode tiled *)
  array2_implode_tiled gm trows tcols
    (fun (tr:natlt (rows / trows)) (tc:natlt (cols / tcols)) (i:natlt trows) (j:natlt tcols) ->
      vf (tr * ntc + tc) (i * tcols + j));

  (* Prove the pure postcondition *)
  assert pure (forall (row : natlt rows) (col : natlt cols).
    let tr = row / trows in
    let tc = col / tcols in
    let i = row % trows in
    let j = col % tcols in
    spec_fn row col (vf (tr * ntc + tc) (i * tcols + j)));

  vf
}
#pop-options
