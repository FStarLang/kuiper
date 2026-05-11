module Kuiper.Kernel.GEMM.TiledComb

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Tiling
open Kuiper.Kernel.GEMMGPU.Type
module Tiling = Kuiper.Matrix.Tiling

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  gA |-> Frac fA eA **
  gB |-> Frac fB eB **
  (exists* v.
    gpu_matrix_pts_to_cell
      (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile)
      (tid % tile)
      v)
  **
  emp

(* NO FUNCTIONAL SPEC RIGHT NOW *)
unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA : ematrix et (mrows   * tile) (mshared * tile))
  (eB : ematrix et (mshared * tile) (mcols   * tile))
  (eC : ematrix et (mrows   * tile) (mcols   * tile))
  (fA fB : perm)
  (bid : natlt (mrows * mcols))
  (tid : natlt (tile * tile))
  : slprop
  =
  kpre comb tile gA gB gC eA eB eC fA fB bid tid

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  norewrite
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  let mut k : sz = 0sz;
  let mut sum : et = zero;

  while ((!k <^ shared))
    invariant
      exists* (vk : SZ.t{vk <= shared}).
        k |-> vk **
        sum |-> MS.__matmul_single eA eB i j vk
  {
    let v1 = gpu_matrix_read gA i !k;
    let v2 = gpu_matrix_read gB !k j;

    sum := !sum `add` mul v1 v2;
    k   := !k +^ 1sz;

    (**)MS.matmul_single_lemma eA eB i j !k;
    ();
  };
  !sum
}

inline_for_extraction noextract
type dp_t
  (et : Type0) {| scalar et |}
  (rows shared cols : SZ.t)
=
  fn (#lA : mlayout rows shared)
     (#lB : mlayout shared cols)
     {| clayout lA, clayout lB |}
     (gA : gpu_matrix et lA)
     (gB : gpu_matrix et lB)
     (#eA : ematrix et rows shared)
     (#eB : ematrix et shared cols)
     (i : szlt rows)
     (j : szlt cols) 
     (#fA #fB : perm)
  preserves
    gpu ** gA |-> Frac fA eA ** gB |-> Frac fB eB
  returns res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)

inline_for_extraction noextract
fn with_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : nat { trows > 0 /\ trows /? rows })
  (tcols : nat { tcols > 0 /\ tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  (#pre #post : slprop)
  (k : unit ->
    stt unit
      (requires pre **
         gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (ensures fun _ -> post **
         gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)))
  requires
    gm |-> Frac f em **
    pre
  ensures
    gm |-> Frac f em **
    post
{
    Tiling.gpu_matrix_extract_tile_ro gm trows tcols tr tc;

    k ();

    ambig_trade_elim ();
}

inline_for_extraction noextract
fn tile_and_recurse
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  {| clayout lA, clayout lB |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (#eA #eB : ematrix et _ _)
  (i : szlt (mrows * tile))
  (j : szlt (mcols * tile))
  (#fA #fB : perm)
  (dp_f : dp_t et tile tile tile)
  norewrite
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  let mut s : et = zero;
  let mut bk : sz = 0sz;

  // tile i (which tile), sub i (where inside tile)
  let ti, si = s_divmod tile i;
  let tj, sj = s_divmod tile j;

  while ((!bk <^ mshared))
    invariant
      live bk ** live s **
      pure (!bk <= mshared)
  {
    let tA = Tiling.gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v ti) (SZ.v !bk);
    let tB = Tiling.gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v !bk) (SZ.v tj);
    assert (rewrites_to tA (Tiling.gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v ti) (SZ.v !bk)));
    assert (rewrites_to tB (Tiling.gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v !bk) (SZ.v tj)));

    // with_tile gA tile tile ti !bk fn () {
    //   with_tile gB tile tile !bk tj fn () {
    //     let s' = dp_f tA tB si sj;
    //     s := !s `add` s';
    //   }
    // };

    Tiling.gpu_matrix_extract_tile_ro gA tile tile ti !bk;
    Tiling.gpu_matrix_extract_tile_ro gB tile tile !bk tj;

    let s' = dp_f tA tB si sj;
    s := !s `add` s';

    ambig_trade_elim ();
    ambig_trade_elim ();

    bk := !bk +^ 1sz;
  };

  assume pure (!s == MS.matmul_single eA eB i j);

  !s
}

#set-options "--print_implicits"

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (tile : valid_tile)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, cC : clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (eA eB eC : ematrix et _ _)
  (fA fB : perm)
  (dp_f : dp_t et tile tile tile)
  (bid : szlt (mrows * mcols))
  (tid : szlt (tile * tile))
  ()
  norewrite
  requires
    gpu **
    kpre comb tile gA gB gC eA eB eC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
  ensures
    gpu **
    kpost comb tile gA gB gC eA eB eC fA fB bid tid **
    thread_id (tile * tile) tid **
    block_id (mrows * mcols) bid
{
  let mrow, mcol = s_divmod mcols bid;
  let brow, bcol = s_divmod tile  tid;

  with i0 j0 v0.
    rewrite
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) #1.0R i0 j0 v0
    as
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)) #1.0R brow bcol v0
    ;

  assert (pure (mrow < mrows));
  assert (pure (mcol < mcols));
  assert (pure (brow < tile));
  assert (pure (bcol < tile));

  let tC = gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols);
  assert (rewrites_to tC (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols)));

  // let s = tile_and_recurse tile gA gB (mrow *^ tile +^ brow) (mcol *^ tile +^ bcol) dp_f;
  // This should be inlined.

  let s = tile_and_recurse tile gA gB (mrow *^ tile +^ brow) (mcol *^ tile +^ bcol) fn
    (#lA : mlayout tile tile)
    (#lB : mlayout tile tile)
    {| cA : clayout lA, cB : clayout lB |}
    (gA : gpu_matrix et lA)
    (gB : gpu_matrix et lB)
    (#eA #eB : ematrix et _ _)
    (i j : szlt tile)
    (#fA #fB : perm)
    {
      assume pure False; // FIXME: somehow the functional proof fails
      // tile_and_recurse #_ #_ #1sz #1sz #1sz tile #lA #lB #cA #cB gA gB 0sz 0sz fn
      //   (#lA : mlayout tile tile)
      //   (#lB : mlayout tile tile)
      //   {| clayout lA, clayout lB |}
      //   (gA : gpu_matrix et lA)
      //   (gB : gpu_matrix et lB)
      //   (#eA #eB : ematrix et _ _)
      //   (i j : szlt tile)
      //   (#fA #fB : perm)
      //   {
      //     dp_f gA gB i j;
      //   }
      tile_and_recurse #et #_ #tile #tile #tile 1sz #lA #lB #cA #cB gA gB i j fn
        (#lA : mlayout 1 1)
        (#lB : mlayout 1 1)
        {| clayout lA, clayout lB |}
        (gA : gpu_matrix et lA)
        (gB : gpu_matrix et lB)
        (#eA #eB : ematrix et 1 1)
        (i j : szlt 1)
        (#fA #fB : perm)
        {
          let r = gpu_matrix_read gA 0sz 0sz *^ gpu_matrix_read gB 0sz 0sz;
          r
        }
    }
  ;

  let v0 = gpu_matrix_read_cell tC brow bcol;
  let v1 = comb v0 s;
  gpu_matrix_write_cell tC brow bcol v1;

  rewrite
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      brow bcol v1
  as
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gC (SZ.v tile) (SZ.v tile) (bid / mcols) (bid % mcols))
      (tid / tile) (tid % tile) v1;

  ()
}

ghost
fn setup
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpre comb tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn teardown
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  ()
  norewrite
  requires
    (forall+ (bid : natlt2 mrows mcols)
            (tid : natlt2 tile  tile).
      kpost comb tile gA gB gC eA eB eC fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  admit();
}

inline_for_extraction noextract
fn my_matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  norewrite // So the type matches
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  Kuiper.Kernel.GEMM.Util.matmul_dotprod gA gB i j;
}

inline_for_extraction noextract
let mk_kernel
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mshared #mcols : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#fA : perm)
  (gB : gpu_matrix et lB)
  (#fB : perm)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  (_ : squash (mrows * mcols <= max_blocks
               /\ tile * tile <= max_threads))
  : kernel_desc_m_n
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
= {
  nblk = mrows *^ mcols;
  nthr = tile *^ tile;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt2 tile tile). kpre  comb tile gA gB gC eA eB eC fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt2 tile tile). kpost comb tile gA gB gC eA eB eC fA fB bid tid);
  setup     = setup    tile comb gA gB gC #eA #eB #eC;
  teardown  = teardown tile comb gA gB gC #eA #eB #eC;

  block_frame    = (fun _bid -> emp);
  block_setup    = magic();
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  kpre      = kpre  comb tile gA gB gC eA eB eC fA fB;
  kpost     = kpost comb tile gA gB gC eA eB eC fA fB;

  f = kf #et #_ comb #mrows #mshared #mcols tile gA gB gC eA eB eC fA fB my_matmul_dotprod;

  kpre_sendable = magic();
  kpost_sendable = magic();
  block_pre_sendable = magic();
  block_post_sendable = magic();
}

inline_for_extraction noextract
fn mmcomb_gpu
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#mrows #mcols #mshared : szp)
  (#lA : mlayout (mrows   * tile) (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  (#lC : mlayout (mrows   * tile) (mcols   * tile))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (gB : gpu_matrix et lB)
  (gC : gpu_matrix et lC)
  (#eA #eB #eC : ematrix _ _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks /\
          tile * tile <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  dassert (tile `SZ.gt` 0sz);
  launch_sync (mk_kernel tile comb gA gB gC ());
}
