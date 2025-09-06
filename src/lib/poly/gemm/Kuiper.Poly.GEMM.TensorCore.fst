module Kuiper.Poly.GEMM.TensorCore

#lang-pulse

open Kuiper

#set-options "--z3rlimit 20"


open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }

open Kuiper.Matrix

module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
module B = Kuiper.Barrier

module R = Kuiper.Matrix.Reprs

open Kuiper.EMatrix { ematrix }
open Kuiper.VArray {
  varray,
  varray_pts_to,
  varray_pts_to_cell
}
open Kuiper.TensorCore
open Kuiper.Float16
open Kuiper.Matrix.Tiling

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc
  (et:Type0) {| sized et |}
  (bm bn bk: szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  : list shmem_desc = [
  SHArray et (bm *^ bk);
  SHArray et (bk *^ bn);
]

unfold
let bid_y
  (rows cols : erased nat)
  (bm : erased nat {bm > 0 /\ bm /? rows})
  (bn : erased nat {bn > 0 /\ bn /? cols})
  (bid : enatlt (rows/bm * (cols/bn)))
  : enatlt (rows/bm)
  =
    bid / (cols/bn)

unfold
let bid_x
  (rows cols : erased nat)
  (bm : erased nat {bm > 0 /\ bm /? rows})
  (bn : erased nat {bn > 0 /\ bn /? cols})
  (bid : enatlt (rows/bm * (cols/bn)))
  : enatlt (cols/bn)
  =
  bid % (cols/bn)

unfold
let tid_y
  (bm bn : erased nat)
  (tm : erased nat {tm > 0 /\ tm /? bm})
  (tn : erased nat {tn > 0 /\ tn /? bn})
  (tid : enatlt (bm/tm * (bn/tn)))
  : enatlt (bm/tm)
  =
    tid / (bn/tn)

unfold
let tid_x
  (bm bn : erased nat)
  (tm : erased nat {tm > 0 /\ tm /? bm})
  (tn : erased nat {tn > 0 /\ tn /? bn})
  (tid : enatlt (bm/tm * (bn/tn)))
  : enatlt (bn/tn)
  =
    tid % (bn/tn)

inline_for_extraction noextract
instance concrete_sz_1 : concrete_sz 1 = { x = 1sz }

inline_for_extraction noextract
let block_tile
  (#et : Type0) {| scalar et |}
  (#rows #cols : erased nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : erased nat{bm > 0 /\ bm /? rows})
  (bn : erased nat{bn > 0 /\ bn /? cols})
  (bid : enatlt (rows/bm * (cols/bn)))
  : Tot (gpu_matrix et
          (subtile_layout lC bm bn
            (bid_y rows cols bm bn bid)
            (bid_x rows cols bm bn bid)))
  =
    gpu_matrix_subtile gC bm bn
      (bid_y rows cols bm bn bid) (bid_x rows cols bm bn bid)

inline_for_extraction noextract
let thread_tile
  (#et : Type0) {| scalar et |}
  (#bm #bn : erased nat)
  (#lC_bt : mlayout bm bn)
  (gC_bt : gpu_matrix et lC_bt)
  (tm : erased nat{tm > 0 /\ tm /? bm})
  (tn : erased nat{tn > 0 /\ tn /? bn})
  (tid : enatlt (bm/tm * (bn/tn)))
  : Tot (gpu_matrix et
          (subtile_layout lC_bt tm tn
            (tid_y bm bn tm tn tid) (tid_x bm bn tm tn tid)))
  =
   gpu_matrix_subtile gC_bt tm tn
    (tid_y bm bn tm tn tid) (tid_x bm bn tm tn tid)

let own_thread_tile
  (#et : Type0) {| scalar et |}
  // Since this is an slprop, I would like to not erase the nat.
  // Unfortunately, when unfolding own_thread_tile, after passing
  // a (reveal x) as argument, this leads to (reveal (hide (reveal x)))
  // which creates problems with type equalities.
  (#rows : erased nat)
  (#cols : nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : nat{bm > 0 /\ bm /? rows})
  (bn : nat{bn > 0 /\ bn /? cols})
  (tm : nat{tm > 0 /\ tm /? bm})
  (tn : nat{tn > 0 /\ tn /? bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
    exists* em.
      gpu_matrix_pts_to
        (thread_tile (block_tile gC bm bn bid) tm tn tid) em

// TODO look again at why we cannot use nats here instead of sizet
unfold
let kpre1
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (fA fB : perm)
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  =
  pure (SZ.fits (rows * shared)) **
  pure (SZ.fits (shared * cols)) **
  pure (valid_frag_et_dims et_ab FragA tm tn tk) **
  pure (valid_frag_et_dims et_ab FragB tm tn tk) **
  pure (valid_frag_et_dims et_c FragAcc tm tn tk) **
  pure (valid_frag_et_comb et_ab et_c) **
  // could be added if it wasn't trivially true
  // pure (valid_frag_et_comb et et) **
  gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA **
  gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB **
  // see comment at own_thread_tile for why we explicitly
  // have to convert (so that the reaveal coercion can kick in)
  own_thread_tile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) bid tid

unfold
let kpost1
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  =
  gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA **
  gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB **
  own_thread_tile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) bid tid

let own_shmem_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (sm : gpu_matrix et lm)
  (i : natlt rows)
  (j : natlt cols)
  : slprop
  = exists* v. gpu_matrix_pts_to_cell sm i j v

let div_ceil (a : nat) (b : pos) = (a + (b-1))/b
let own_tile_stride_cells
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  ([@@@mkey] m : gpu_matrix et lm)
  // (em : ematrix et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
  =
  (* using enatlt here would require enumerable enatlt which is not entirely straightforward *)
  forall+ (it : natlt (div_ceil (rows*cols) nthr)).
    let flat_idx = tid + (it * nthr) <: erased nat in
    let i = flat_idx/cols <: erased nat in
    let j = flat_idx%cols <: erased nat in
      if (i < rows && j < cols)
      then own_shmem_cell m i j
      else emp

inline_for_extraction noextract
fn cp_matrix
  (#et : Type0) {| scalar et |}
  (rows cols: sz)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  (src : gpu_matrix et lsrc)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (#fM : perm)
  (nthr : sz)
  (tid : szlt nthr)
  preserves
    gpu **
    pure (SZ.fits (rows * cols + nthr)) **
    src |-> Frac fM esrc **
    own_tile_stride_cells dst nthr tid
{
  let mlen = rows *^ cols;

  let mut i : sz = tid;
  while (SZ.(!i <^ mlen))
    invariant
      exists* (vi : sz).
        pure (vi >= tid) **
        pure (vi % nthr == tid) **
        i |-> vi **
        own_tile_stride_cells dst nthr tid **
        pure (vi < mlen + nthr)
  {
    let v = gpu_matrix_read src (!i /^ cols) (!i %^ cols);
    unfold own_tile_stride_cells dst nthr tid;

    let ite : erased (natlt (div_ceil (rows*cols) nthr)) = (!i - tid) / nthr;
    forevery_extract (reveal ite) _;

    rewrite each
      (((tid + ite * nthr) / cols < rows) &&
       ((tid + ite * nthr) % cols < cols))
    as true;
    let vi = !i;
    assert pure (SZ.v vi == (SZ.v tid + ite * SZ.v nthr));
    rewrite each (tid + ite * nthr) as vi;

    unfold own_shmem_cell dst (vi / cols) (vi % cols);
    gpu_matrix_write_cell dst (!i /^ cols) (!i %^ cols) v;
    fold own_shmem_cell dst (vi / cols) (vi % cols);

    rewrite each SZ.v vi as (tid + ite * nthr);

    Pulse.Lib.Trade.elim_trade _ _;

    fold own_tile_stride_cells dst nthr tid;

    Math.Lemmas.modulo_addition_lemma vi nthr 1;
    i := !i +^ nthr;
  };

  ()
}

let barrier_p
  (#et : Type0)
  (#bm #bn #bk : szp)
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  : B.barrier_side nthr =
  fun it tid ->
    if even it then
      (exists* (x : ematrix _ _ _). m1 |-> Frac (1.0R /. nthr) x) **
      (exists* (x : ematrix _ _ _). m2 |-> Frac (1.0R /. nthr) x)
    else
      own_tile_stride_cells m1 nthr tid **
      own_tile_stride_cells m2 nthr tid

let barrier_q
  (#et : Type0)
  (#bm #bn #bk : szp)
  (#l1 : mlayout bm bk)
  (#l2 : mlayout bk bn)
  (m1 : gpu_matrix et l1)
  (m2 : gpu_matrix et l2)
  (nthr : pos)
  : B.barrier_side nthr =
  fun it tid -> barrier_p m1 m2 nthr (it+1) tid (* flip flop *)

let barrier_tok
  (#et : Type0)
  (#bm #bn #bk : szp)
  (* This is defined over the base shared gpu_arrays, as
  this spec must make sense before the arrays are viewed as
  a matrix. *)
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (it : nat)
  (nthr : pos)
  (tid : natlt nthr)
  : slprop
  =
  B.barrier_tok (barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr)
                (barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr)
                it tid

unfold
let kpre
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpre1 gA eA gB eB gC bm bn bk tm tn tk fA fB bid tid **
  exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * (bn/tn))) x **
  exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * (bn/tn))) x **
  barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) 0 (bm/tm * (bn/tn)) tid

unfold
let kpost
  (#et_ab #et_c : Type0) {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpost1 (* comb *) gA eA gB eB gC bm bn bk tm tn fA fB bid tid **
  exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * (bn/tn))) x **
  exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * (bn/tn))) x **
  barrier_tok (R.row_major bm bk) (R.row_major bk bn) (fst sh) (fst (snd sh)) (2 * (shared/bk)) (bm/tm * (bn/tn)) tid

inline_for_extraction noextract
fn populate_shmem
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : erased nat)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#slA : mlayout bm bk) {| clayout slA |}
  (#slB : mlayout bk bn) {| clayout slB |}
  (sA : gpu_matrix et slA)
  (sB : gpu_matrix et slB)
  (#lA : mlayout rows shared) {| clayout lA |}
  (#lB : mlayout shared cols) {| clayout lB |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#fA #fB : perm)
  (#eB : ematrix et shared cols)
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (tile_row : szlt (rows/bm))
  (tile_shared : szlt (shared/bk))
  (tile_col : szlt (cols/bn))
  (tid : szlt (bm/^tm *^ (bn/^tn)))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id (bm/^tm *^ (bn/^tn)) tid **
    own_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
    own_tile_stride_cells sB (bm/tm * (bn/tn)) tid
{
  let tileA = gpu_matrix_extract_tile_ro' gA
    (hide (SZ.v bm)) (hide (SZ.v bk)) (hide (SZ.v tile_row)) (hide (SZ.v tile_shared));
  cp_matrix bm bk #_ #_ tileA sA (get_bdim()) tid;

  let tileB = gpu_matrix_extract_tile_ro' gB
    (hide (SZ.v bk)) (hide (SZ.v bn)) (hide (SZ.v tile_shared)) (hide (SZ.v tile_col));
  cp_matrix bk bn tileB sB (get_bdim()) tid;

  ambig_trade_elim ();
  ambig_trade_elim ();
  ();
}

inline_for_extraction noextract
fn subproducts_tc
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk: szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (aFrag : fragment et_ab FragA tm tn tk FragLRM)
  (#vaFrag : ematrix et_ab tm tk)
  (bFrag : fragment et_ab FragB tm tn tk FragLRM)
  (#vbFrag : ematrix et_ab tk tn)
  (accumFrag : fragment et_acc FragAcc tm tn tk FragLAcc)
  (#vaccumFrag : ematrix et_acc tm tn)
  (gA : gpu_matrix et_ab (R.row_major bm bk))
  (gB : gpu_matrix et_ab (R.row_major bk bn))
  (#eA : ematrix et_ab bm bk)
  (#eB : ematrix et_ab bk bn)
  (#fA #fB : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt (bn/tn))
  preserves
    gpu **
    pure (valid_frag_et_comb et_ab et_acc) **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    aFrag |-> vaFrag **
    bFrag |-> vbFrag **
    accumFrag |-> vaccumFrag
  ensures
    exists* vaFrag' vbFrag' vaccumFrag'.
      aFrag |-> vaFrag' **
      bFrag |-> vbFrag' **
      accumFrag |-> vaccumFrag'
{
  let mut dotIdx : sz = 0sz;
  while (SZ.(!dotIdx <^ (bk/^tk)))
    invariant
      exists* (vdotIdx : sz{vdotIdx <= (bk/^tk)}) (vaFrag : ematrix et_ab tm tk)
        (vbFrag : ematrix et_ab tk tn) (vaccumFrag : ematrix et_acc tm tn).
        dotIdx |-> vdotIdx **
        aFrag |-> vaFrag **
        bFrag |-> vbFrag **
        accumFrag |-> vaccumFrag
  {
    (* only required because of rewrites_to *)
    let didx = !dotIdx;

    gpu_matrix_extract_tile_ro gA tm tk arow !dotIdx;
    let a_tile = gpu_matrix_subtile gA (hide (SZ.v tm)) (hide (SZ.v tk)) (hide (SZ.v arow)) (hide (SZ.v didx));
    assert (rewrites_to a_tile (gpu_matrix_subtile gA (hide (SZ.v tm)) (hide (SZ.v tk)) (hide (SZ.v arow)) (hide (SZ.v didx))));

    gpu_matrix_extract_tile_ro gB tk tn !dotIdx bcol;
    let b_tile = gpu_matrix_subtile gB (hide (SZ.v tk)) (hide (SZ.v tn)) (hide (SZ.v !dotIdx)) (hide (SZ.v bcol));
    assert (rewrites_to b_tile (gpu_matrix_subtile gB (hide (SZ.v tk)) (hide (SZ.v tn)) (hide (SZ.v didx)) (hide (SZ.v bcol))));

    mma_loadA aFrag a_tile;
    mma_loadB bFrag b_tile;
    mma_sync' aFrag bFrag accumFrag;

    with etA.
      assert (gpu_matrix_pts_to a_tile #fA etA);
      Pulse.Lib.Trade.elim_trade (a_tile |-> Frac fA etA) (gA |-> Frac fA eA);
    with etB.
      assert (gpu_matrix_pts_to b_tile #fB etB);
      Pulse.Lib.Trade.elim_trade (b_tile |-> Frac fB etB) (gB |-> Frac fB eB);

    dotIdx := !dotIdx +^ 1sz;
  };

  ()
}

inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et |}
  // (comb : binop et)

  // rows should be an erased nat because not concerete value is required, but
  // using erased nats here leads to very confusing reveals when calling mma_store
  //  (maybe due to inferred type class instances?)
  // making it a size because otherwise a nat would be extracted
  (#rows : erased nat)
  // cols is concretized so using size is fine I think
  (#cols : sz)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#tk : erased nat)
  (accumFrag : fragment et FragAcc tm tn tk FragLAcc)
  (gC : gpu_matrix et (R.row_major rows cols))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  requires
    gpu **
    // see comment in own_thread_tile for why SZ.v
    own_thread_tile gC bm bn tm tn bid tid **
    (exists* vaccumFrag.
      accumFrag |-> vaccumFrag)
  ensures
    gpu **
    own_thread_tile gC bm bn tm tn bid tid **
    (exists* vaccumFrag.
      accumFrag |-> vaccumFrag)
{
  unfold own_thread_tile gC bm bn tm tn bid tid;

  (* Only create a tile in gC and write the accumulator values. In this version the input from gC
     was added by loading the tile into the accumulator before any other computations *)
  let t_tile = thread_tile (block_tile gC (SZ.v bm) (SZ.v bn) (hide (SZ.v bid)))
    (SZ.v tm) (SZ.v tn) (hide (SZ.v tid));
  assert (rewrites_to t_tile (thread_tile (block_tile gC (SZ.v bm) (SZ.v bn) (hide (SZ.v bid)))
    (SZ.v tm) (SZ.v tn) (hide (SZ.v tid))));

  // from looking at the type of mma_store, it is not clear that cols mut be concretizable
  // 1. know that strided_row_major needs concrete sizes
  // 2. search the code base for the appropriate instance and see which of the arguments
  //   must be concretizable
  // 3. figure out which expression is which argument and make concretizable accordingly
  mma_store accumFrag t_tile;

  // rewrite each t_tile as thread_tile (block_tile gC bm bn bid) tm tn tid;
  fold own_thread_tile gC bm bn tm tn bid tid;
  ()
}

// #push-options "--split_queries always --debug SMTFail"
// #push-options "--z3rlimit 40 --retry 5"
// #push-options "--print_implicits"
inline_for_extraction noextract
fn kf
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB |}
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  ()
  requires
    gpu **
    kpre gA eA gB eB gC bm bn bk tm tn tk fA fB sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (rows/bm * (cols/bn)) bid
  ensures
    gpu **
    kpost (* comb *) gA eA gB eB gC bm bn bk tm tn fA fB sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (rows/bm * (cols/bn)) bid
{
  let sarA : gpu_array et_ab (bm * bk) = fst sh;
  let sarB : gpu_array et_ab (bk * bn) = fst (snd sh);
  rewrite each fst sh as sarA;
  rewrite each fst (snd sh) as sarB;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;
  // This leads to a faillure to resolve the clayout when calling populate_shmem
  // let slA = R.row_major bm bk;
  // assert (rewrites_to slA (R.row_major bm bk));
  // let slB = R.row_major bk bn;
  // assert (rewrites_to slB (R.row_major bk bn));

  unfold barrier_tok (R.row_major bm bk) (R.row_major bk bn) sarA sarB 0 (bm/tm * (bn/tn)) tid;

  gpu_matrix_abs' (R.row_major bm bk) sarA;
  let sA = from_array (R.row_major bm bk) sarA;
  rewrite each from_array (R.row_major bm bk) sarA as sA;

  gpu_matrix_abs' (R.row_major bk bn) sarB;
  let sB = from_array (R.row_major bk bn) sarB;
  rewrite each from_array (R.row_major bk bn) sarB as sB;

  let num_k_tiles = shared /^ bk;
  let num_n_tiles = cols /^ bn;
  let mrow = bid /^ num_n_tiles;
  let mcol = bid %^ num_n_tiles;

  let threadRow = tid /^ (bn/^tn);
  let threadCol = tid %^ (bn/^tn);

  (* tensor core fragments *)
  let aFrag = __alloc_fragment et_ab FragA tm tn tk FragLRM;
  let bFrag = __alloc_fragment et_ab FragB tm tn tk FragLRM;
  let accumFrag = __alloc_fragment et_c FragAcc tm tn tk FragLAcc;

  (* get ownership over the thread's gC tile and load it into the accumulator *)
  unfold own_thread_tile gC bm bn tm tn bid tid;
  let t_tile = thread_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (SZ.v tm) (SZ.v tn) (SZ.v tid);
  assert (rewrites_to t_tile (thread_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (SZ.v tm) (SZ.v tn) (SZ.v tid)));
  mma_loadAccum accumFrag t_tile;
  fold own_thread_tile gC bm bn tm tn bid tid;

  let mut bkIdx  : sz = 0sz;
  while (SZ.(!bkIdx <^ num_k_tiles))
    invariant
      exists* (vbkIdx : SZ.t{vbkIdx <= num_k_tiles})
        (vaFrag : ematrix et_ab tm tk) (vbFrag : ematrix et_ab tk tn) (vaccumFrag : ematrix et_c tm tn).
        bkIdx |-> vbkIdx **
        aFrag |-> vaFrag **
        bFrag |-> vbFrag **
        accumFrag |-> vaccumFrag **
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. ((bm/tm*(bn/tn)))) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. ((bm/tm*(bn/tn)))) x) **
        B.barrier_tok (barrier_p sA sB ((bm/tm*(bn/tn)))) (barrier_q sA sB ((bm/tm*(bn/tn)))) (2 * vbkIdx) tid
  {
    (* This assert should not be needed. I don't know what effect it even has. *)
    let vbkIdx = !bkIdx;
    assert B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn))) (barrier_q sA sB (bm/tm * (bn/tn))) (2 * vbkIdx) tid;
    even_2x vbkIdx;
    assert pure((2 * vbkIdx % 2 = 0) == true);
    rewrite
        // WARNING after reintroducing the paranthesis in barrier_p, this checks
        //  It seems that either there are paranthesis in both or in neither, otherwise the assertion fails
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * (bn/tn))) x)
      as barrier_p sA sB (bm/tm * (bn/tn)) (2 * vbkIdx) tid;

    B.barrier_wait ();
    rewrite (barrier_q sA sB (bm/tm * (bn/tn)) (2 * vbkIdx) tid)
        as own_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
           own_tile_stride_cells sB (bm/tm * (bn/tn)) tid;

    populate_shmem bm bn bk tm tn sA sB gA gB mrow !bkIdx mcol tid;

    assert (B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn))) (barrier_q sA sB (bm/tm * (bn/tn))) (2 * vbkIdx + 1) tid);
    odd_2x1 vbkIdx;
    assert (pure (odd (2 * vbkIdx + 1)));
    rewrite own_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
            own_tile_stride_cells sB (bm/tm * (bn/tn)) tid
         as (barrier_p sA sB (bm/tm * (bn/tn)) (2 * vbkIdx + 1) tid);

    B.barrier_wait ();

    even_2x (vbkIdx + 1);
    assert (pure (2 * (vbkIdx + 1) == 2 * vbkIdx + 2));
    assert (pure (even (2 * vbkIdx + 2)));
    rewrite (barrier_q sA sB (bm/tm * (bn/tn)) (2 * vbkIdx + 1) tid)
    as
      (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
      (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * (bn/tn))) x);

    subproducts_tc bm bn bk tm tn tk aFrag bFrag accumFrag sA sB threadRow threadCol;

    bkIdx := !bkIdx +^ 1sz;
  };

  epilogue bm bn tm tn accumFrag gC bid tid;

  with vaFrag. assert aFrag |-> vaFrag; drop_ (aFrag |-> vaFrag);
  with vbFrag. assert bFrag |-> vbFrag; drop_ (bFrag |-> vbFrag);
  with vaccumFrag. assert accumFrag |-> vaccumFrag; drop_ (accumFrag |-> vaccumFrag);

  gpu_matrix_concr sA; rewrite each core sA as sarA;
  gpu_matrix_concr sB; rewrite each core sB as sarB;

  fold barrier_tok (R.row_major bm bk) (R.row_major bk bn) sarA sarB (2 * num_k_tiles) (bm/tm * (bn/tn)) tid;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);
  ()
}

ghost
fn setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (bm * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpre1 (*comb*) gA eA gB eB gC bm bn bk tm tn tk fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (bm * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    block_setup_tok nthr **
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 (* comb *) gA eA gB eB gC bm bn bk tm tn tk fA fB bid tid)
  ensures
    block_setup_tok nthr **
    (forall+ (tid : natlt nthr).
      kpre (* comb *) gA eA gB eB gC bm bn bk tm tn tk fA fB sh bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (#_: squash (SZ.fits (bm * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost (* comb *) gA eA gB eB gC bm bn bk tm tn fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC bm bn bk tm tn fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et_ab lA)
  (eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c lC)
  (eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (rows * cols)))
  (#_: squash (SZ.fits (bm * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 (* comb *) gA eA gB eB gC bm bn bk tm tn fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    // underspec not implemented anyway
    (exists* eC'. gC |-> eC')
    // (gC |-> MS.mmcomb comb eC eA eB)
{
  // forevery_flatten #(natlt2 mrows mcols) #_ #(natlt tile)
  //   (fun bid tid -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_tostar #(natlt2 mrows mcols & natlt tile) (fun _tid -> m4_pts_to gA #(1.0R /. mlayout_size lC) eA);

    // (fun (bid, tid) -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  admit();
}

inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB |}
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (#_ : squash (SZ.fits (rows * cols)))
  (#eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (rows/bm * (cols/bn) <= max_blocks
               /\ (bm/tm * (bn/tn)) <= max_threads))
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** (exists* eC'. gC |-> eC'))
= {
  nblk;// = rows/^bm *^ (cols/^bn);
  nthr;// = (bm /^ tm *^ (bn /^ tn));

  shmems_desc = shmems_desc et_ab bm bn bk;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  (* comb *) gA eA gB eB gC bm bn bk tm tn tk fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 (* comb *) gA eA gB eB gC bm bn bk tm tn fA fB bid tid);

  setup      = setup    (* comb *) gA eA gB eB gC eC bm bn bk tm tn tk nblk nthr fA fB;
  teardown   = teardown (* comb *) gA eA gB eB gC eC bm bn bk tm tn nblk nthr fA fB;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup (* comb *) gA eA gB eB gC eC bm bn bk tm tn tk nblk nthr fA fB;
  block_teardown = block_teardown (* comb *) gA eA gB eB gC eC bm bn bk tm tn nblk nthr fA fB ;

  kpre      = kpre  (* comb *) gA eA gB eB gC bm bn bk tm tn tk fA fB;
  kpost     = kpost (* comb *) gA eA gB eB gC bm bn bk tm tn fA fB;

  f = kf (* comb *) gA #eA gB #eB gC bm bn bk tm tn tk #() #() #fA #fB;
}

inline_for_extraction noextract
fn mmcomb_gpu
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (#_ : squash (SZ.fits (rows * cols)))
  (#eC : ematrix et_c rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (rows/bm * (cols/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    gC |-> eC
  ensures
    (exists* eC'. gC |-> eC')
{
  launch_sync (mk_kernel gA gB gC bm bn bk tm tn tk (rows/^bm *^ (cols/^bn)) (bm/^tm *^ (bn/^tn)) ());
}
