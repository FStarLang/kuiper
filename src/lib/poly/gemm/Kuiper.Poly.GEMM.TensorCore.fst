module Kuiper.Poly.GEMM.TensorCore

#lang-pulse

open Kuiper

#set-options "--z3rlimit 20"

module EM = Kuiper.EMatrix

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

let block_tile
  (#et : Type0) {| scalar et |}
  (#rows #cols : erased nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bid : enatlt (rows/bm * (cols/bn)))
  : gpu_matrix et
      (subtile_layout lC bm bn
        (bid_y rows cols bm bn bid) (bid_x rows cols bm bn bid))
  =
    gpu_matrix_subtile gC bm bn (bid_y rows cols bm bn bid) (bid_x rows cols bm bn bid)

let thread_tile
  (#et : Type0) {| scalar et |}
  (#bm #bn : sz)
  (#lC_bt : mlayout bm bn)
  (gC_bt : gpu_matrix et lC_bt)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tid : enatlt (bm/tm * (bn/tn)))
  : gpu_matrix et
      (subtile_layout lC_bt tm tn
        (tid_y bm bn tm tn tid) (tid_x bm bn tm tn tid))
  =
   gpu_matrix_subtile gC_bt tm tn (tid_y bm bn tm tn tid) (tid_x bm bn tm tn tid)

let own_thread_tile
  (#et : Type0) {| scalar et |}
  (#rows #cols : erased nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (bid : enatlt ((rows/bm) * (cols/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  = 
    (exists* em.
      gpu_matrix_pts_to
        (thread_tile (block_tile gC bm bn bid) tm tn tid) em)

unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
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
  (gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA) **
  (gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB) **
  own_thread_tile gC bm bn tm tn bid tid

unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
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
  (gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA) **
  (gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB) **
  own_thread_tile gC bm bn tm tn bid tid
  
let own_shmem_cell
  (#et : Type0)
  (#rows #cols : erased nat)
  (#lm : mlayout rows cols)
  (sm : gpu_matrix et lm)
  (i : enatlt rows)
  (j : enatlt cols)
  : slprop
  = exists* v. gpu_matrix_pts_to_cell sm i j v

let div_ceil (a : erased nat) (b : pos) = (a + (b-1))/b
let own_tile_stride_cells
  (#et : Type0)
  (#rows #cols : erased nat)
  (#lm : mlayout rows cols)
  ([@@@mkey] m : gpu_matrix et lm)
  // (em : ematrix et rows cols)
  (nthr : erased nat)
  (tid : enatlt nthr)
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
    (src |-> Frac fM esrc) **
    own_tile_stride_cells dst nthr tid
{
  let mlen = rows *^ cols;

  let mut i : sz = tid;
  while (SZ.(!i <^ mlen))
    invariant
      exists* (vi : sz).
        pure (vi >= tid) **
        pure (vi % nthr == tid) **
        (i |-> vi) **
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
  (it : erased nat)
  (nthr : pos)
  (tid : natlt nthr)
  : slprop
  =
  B.barrier_tok (barrier_p (from_array l1 sar1) (from_array l2 sar2) nthr)
                (barrier_q (from_array l1 sar1) (from_array l2 sar2) nthr)
                it tid

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpre1 comb gA eA gB eB gC bm bn bk tm tn fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
  barrier_tok slA slB (fst sh) (fst (snd sh)) 0 (bm/tm * (bn/tn)) tid

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpost1 comb gA eA gB eB gC bm bn bk tm tn fA fB bid tid **
  (exists* (x : seq _). fst sh |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
  (exists* (x : seq _). fst (snd sh) |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
  barrier_tok slA slB (fst sh) (fst (snd sh)) (2 * (shared/bk)) (bm/tm * (bn/tn)) tid

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
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    thread_id (bm/^tm *^ (bn/^tn)) tid **
    own_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
    own_tile_stride_cells sB (bm/tm * (bn/tn)) tid
{
  let tileA = gpu_matrix_extract_tile' gA bm bk tile_row tile_shared;
  cp_matrix bm bk tileA sA (get_bdim()) tid;

  let tileB = gpu_matrix_extract_tile' gB bk bn tile_shared tile_col;
  cp_matrix bk bn tileB sB (get_bdim()) tid;

  ambig_trade_elim ();
  ambig_trade_elim ();
  ();
}

// #push-options "--debug SMTFail --split_queries always --print_implicits"
inline_for_extraction noextract
fn subproducts2d
  (#et : Type0) {| scalar et |}
  (bm bn bk: szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (rAcol rBrow rchProd: array et)
  (#vrAcol #vrBrow #vrchProd : erased (seq et))
  (#l1 : mlayout bm bk) {| clayout l1 |}
  (#l2 : mlayout bk bn) {| clayout l2 |}
  (gA : gpu_matrix et l1)
  (gB : gpu_matrix et l2)
  (#eA : ematrix et bm bk)
  (#eB : ematrix et bk bn)
  (#f : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt (bn/tn))
  preserves
    gpu **
    (gA |-> Frac f eA) **
    (gB |-> Frac f eB)
  requires
    pure (Seq.length vrAcol == tm /\
          Seq.length vrBrow == tn /\
          Seq.length vrchProd == tm * tn /\
          SZ.fits (tm * tn)) **
    (rAcol |-> vrAcol) **
    (rBrow |-> vrBrow) **
    (rchProd |-> vrchProd)
  ensures
    exists* vrAcol' vrBrow' vrchProd'.
      pure (Seq.length vrAcol' == tm /\
            Seq.length vrBrow' == tn /\
            Seq.length vrchProd' == tm * tn) **
      (rAcol |-> vrAcol') **
      (rBrow |-> vrBrow') **
      (rchProd |-> vrchProd')
{
  let mut dotIdx : sz = 0sz;
  while (SZ.(!dotIdx <^ bk))
    invariant
      exists* (vdotIdx : sz{vdotIdx <= bk}) (vrAcol : erased (lseq et tm))
        (vrBrow : erased (lseq et tn)) (vrchProd : erased (lseq et (tm*tn))).
        (dotIdx |-> vdotIdx) **
        (rAcol |-> vrAcol) **
        (rBrow |-> vrBrow) **
        (rchProd |-> vrchProd)
  {
    open Pulse.Lib.Array;

    let mut i0 = 0sz;
    while (SZ.(!i0 <^ tm))
      invariant
        exists* (vi : sz{vi <= tm}) (vrAcol : erased (lseq et tm)).
          (i0 |-> vi) **
          (rAcol |-> vrAcol)
    {
      (* get rid of a few non-linear arithmetic expressions *)
      let a_tile = gpu_matrix_extract_tile' gA tm 1 arow !dotIdx;
      let va = gpu_matrix_read a_tile !i0 0sz;
      ambig_trade_elim ();
      rAcol.(!i0) <- va;

      i0 := !i0 +^ 1sz;
    };

    let mut i1 = 0sz;
    while (SZ.(!i1 <^ tn))
      invariant
        exists* (vi : sz{vi <= tn}) (vrBrow : erased (lseq et tn)).
          (i1 |-> vi) **
          (rBrow |-> vrBrow)
    {
      let b_tile = gpu_matrix_extract_tile' gB 1 tn !dotIdx bcol;
      let vb = gpu_matrix_read b_tile 0sz !i1;
      ambig_trade_elim ();
      rBrow.(!i1) <- vb;

      i1 := !i1 +^ 1sz;
    };

    let mut resIdxM = 0sz;
    while (SZ.(!resIdxM <^ tm))
      invariant
        exists* (vresIdxM : sz{vresIdxM <= tm}) (vrchProd : erased (lseq et (tm*tn))).
          (resIdxM |-> vresIdxM) **
          (rchProd |-> vrchProd)
    {
      let mut resIdxN = 0sz;
      while (SZ.(!resIdxN <^ tn))
        invariant
          exists* (vresIdxN : sz{vresIdxN <= tn}) (vrchProd : erased (lseq et (tm*tn))).
            (resIdxN |-> vresIdxN) **
            (rchProd |-> vrchProd)
      {
        (* works on arrays and therefore does not have the nice matrix abstraction *)
        let ra = rAcol.(!resIdxM);
        let rb = rBrow.(!resIdxN);
        let iM = !resIdxM;
        let iN = !resIdxN;
        assert(pure(SZ.fits(iM *^ tn +^ iN)));
        let old = rchProd.(!resIdxM *^ tn +^ !resIdxN);
        let mad = old `add` (ra `mul` rb);
        rchProd.(!resIdxM *^ tn +^ !resIdxN) <- mad;

        resIdxN := !resIdxN +^ 1sz;
      };

      resIdxM := !resIdxM +^ 1sz;
    };

    dotIdx := !dotIdx +^ 1sz;
  }
}

#push-options "--debug SMTFail --split_queries always"
inline_for_extraction noextract
fn subproducts_tc
  (* TODO restrict et to valid types only *)
  (#et : Type0) {| scalar et |}
  (bm bn bk: szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (* is FragLCM layout for aFrag relevant here? *)
  (aFrag : fragment et FragA tm tn tk FragLRM)
  (#vaFrag : ematrix et tm tk)
  (bFrag : fragment et FragB tm tn tk FragLRM)
  (#vbFrag : ematrix et tk tn)
  (accumFrag : fragment et FragAccum tm tn tk FragLAccum)
  (#vaccumFrag : ematrix et tm tn)
  (gA : gpu_matrix et (R.row_major bm bk))
  (gB : gpu_matrix et (R.row_major bk bn))
  (#eA : ematrix et bm bk)
  (#eB : ematrix et bk bn)
  (#f : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt (bn/tn))
  preserves
    gpu **
    (gpu_matrix_pts_to gA #f eA) **
    (gpu_matrix_pts_to gB #f eB)
  requires
    (aFrag |-> vaFrag) **
    (bFrag |-> vbFrag) **
    (accumFrag |-> vaccumFrag)
  ensures
    exists* vaFrag' vbFrag' vaccumFrag'.
      (aFrag |-> vaFrag') **
      (bFrag |-> vbFrag') **
      (accumFrag |-> vaccumFrag')
{
  let mut dotIdx : sz = 0sz;
  while (SZ.(!dotIdx <^ (bk/^tk)))
    invariant
      exists* (vdotIdx : sz{vdotIdx <= (bk/^tk)}) (vaFrag : ematrix et tm tk)
        (vbFrag : ematrix et tk tn) (vaccumFrag : ematrix et tm tn).
        (dotIdx |-> vdotIdx) **
        (aFrag |-> vaFrag) **
        (bFrag |-> vbFrag) **
        (accumFrag |-> vaccumFrag)
  {
    let a_tile = gpu_matrix_extract_tile' gA tm tk arow !dotIdx;
    mma_loadA aFrag a_tile;
    let b_tile = gpu_matrix_extract_tile' gB tk tn !dotIdx bcol;
    mma_loadB bFrag b_tile;
    mma_sync' aFrag bFrag accumFrag;

    dotIdx := !dotIdx +^ 1sz;
  }
}

inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #cols : sz)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (rchProd: array et)
  (#lC : mlayout rows cols)
  {| clayout lC |}
  (gC : gpu_matrix et lC)
  // (#_ : squash (SZ.fits (bm/tm * (bn/tn))))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  requires
    gpu **
    own_thread_tile gC bm bn tm tn bid tid **
    (exists* vrchProd.
      pure (Seq.length vrchProd == tm * tn) **
      (rchProd |-> vrchProd))
  ensures
    gpu **
    own_thread_tile gC bm bn tm tn bid tid **
    (exists* vrchProd'.
      pure (Seq.length vrchProd' == tm * tn) **
      (rchProd |-> vrchProd'))
{
  unfold own_thread_tile gC bm bn tm tn bid tid;
  let t_tile = thread_tile (block_tile gC bm bn bid) tm tn tid;
  rewrite each thread_tile (block_tile gC bm bn bid) tm tn tid as t_tile;
  // assert (rewrites_to t_tile (thread_tile (block_tile gC bm bn bid) tm tn tid));

  let mut resIdxM = 0sz;
  while (SZ.(!resIdxM <^ tm))
    invariant
      exists* (vresIdxM : sz{vresIdxM <= tm}) (vrchProd : lseq et (tm*tn)) (v : ematrix et tm tn).
        (resIdxM |-> vresIdxM) **
        (rchProd |-> vrchProd) **
        gpu_matrix_pts_to t_tile v
  {
    let mut resIdxN = 0sz;
    while (SZ.(!resIdxN <^ tn))
      invariant
        exists* (vresIdxN : sz{vresIdxN <= tn}) (vrchProd : lseq et (tm*tn)) (v : ematrix et tm tn).
          (resIdxN |-> vresIdxN) **
          (rchProd |-> vrchProd) **
          gpu_matrix_pts_to t_tile v

    {
      let v0 = gpu_matrix_read t_tile !resIdxM !resIdxN;

      (* add the new result in the register cache to the value from gC and overwrite the the cell in gC *)
      open Pulse.Lib.Array;
      // all obvious but without the asserts the next line fails
      assert pure (SZ.fits (tm * tn));
      assert pure (SZ.fits ((tm-1) * tn + tn));
      with vrchProd. assert Pulse.Lib.Array.Core.pts_to rchProd vrchProd;
      assert pure (Seq.length vrchProd == tm * tn);
      let v1 = rchProd.(!resIdxM *^ tn +^ !resIdxN);
      let v' = comb v0 v1;

      gpu_matrix_write t_tile !resIdxM !resIdxN v';

      resIdxN := !resIdxN +^ 1sz;
    };

    resIdxM := !resIdxM +^ 1sz;
  };

  rewrite each t_tile as thread_tile (block_tile gC bm bn bid) tm tn tid;
  fold own_thread_tile gC bm bn tm tn bid tid;
  ()
}

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#fA #fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  ()
  requires
    gpu **
    kpre comb gA eA gB eB gC bm bn bk slA slB tm tn fA fB sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (rows/bm * (cols/bn)) bid
  ensures
    gpu **
    kpost comb gA eA gB eB gC bm bn bk slA slB tm tn fA fB sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (rows/bm * (cols/bn)) bid
{
  let sarA : gpu_array et (bm * bk) = fst sh;
  let sarB : gpu_array et (bk * bn) = fst (snd sh);
  rewrite each fst sh as sarA;
  rewrite each fst (snd sh) as sarB;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  unfold barrier_tok slA slB sarA sarB 0 (bm/tm * (bn/tn)) tid;

  gpu_matrix_abs' slA sarA;
  let sA = from_array slA sarA;
  rewrite each from_array slA sarA as sA;

  gpu_matrix_abs' slB sarB;
  let sB = from_array slB sarB;
  rewrite each from_array slB sarB as sB;

  let num_k_tiles = shared /^ bk;
  let num_n_tiles = cols /^ bn;
  let mrow = bid /^ num_n_tiles;
  let mcol = bid %^ num_n_tiles;

  let threadRow = tid /^ (bn/^tn);
  let threadCol = tid %^ (bn/^tn);

  // (* tensor core fragments *)
  // let aFrag = __alloc_fragment et FragA tm tn tk FragLCM;
  // let bFrag = __alloc_fragment et FragB tm tn tk FragLRM;
  // let accumFrag = __alloc_fragment et FragAccum tm tn tk FragAccum;
  // fill_fragment accumFrag fill_value_zero;
  (* register caches *)
  let mut rAcol : Pulse.Lib.Array.array et = [| zero #et #_ ; tm |];
  let mut rBrow : Pulse.Lib.Array.array et = [| zero #et #_ ; tn |];
  let mut rchProd : Pulse.Lib.Array.array et = [| zero #et #_ ; tm*^tn |];

  let mut bkIdx  : sz = 0sz;
  while (SZ.(!bkIdx <^ num_k_tiles))
    invariant
      exists* (vbkIdx : SZ.t{vbkIdx <= num_k_tiles}) (vrAcol : lseq et tm) (vrBrow : lseq et tn) (vrchProd : lseq et (tm*tn)).
        (bkIdx |-> vbkIdx) **
        (rAcol |-> vrAcol) **
        (rBrow |-> vrBrow) **
        (rchProd |-> vrchProd) **
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. ((bm/tm*(bn/tn)))) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. ((bm/tm*(bn/tn)))) x) **
        B.barrier_tok (barrier_p sA sB ((bm/tm*(bn/tn)))) (barrier_q sA sB ((bm/tm*(bn/tn)))) (2 * vbkIdx) tid **
        gpu
  {
    (* This assert should not be needed. I don't know what effect it even has. *)
    let vbkIdx = !bkIdx;
    assert B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn))) (barrier_q sA sB (bm/tm * (bn/tn))) (2 * vbkIdx) tid;
    even_2x vbkIdx;
    rewrite
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

    subproducts2d bm bn bk tm tn rAcol rBrow rchProd sA sB threadRow threadCol;

    bkIdx := !bkIdx +^ 1sz;
  };

  epilogue comb bm bn tm tn rchProd gC bid tid;

  gpu_matrix_concr sA; rewrite each core sA as sarA;
  gpu_matrix_concr sB; rewrite each core sB as sarB;

  fold barrier_tok slA slB sarA sarB (2 * num_k_tiles) (bm/tm * (bn/tn)) tid;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);
  ()
}

#push-options "--z3rlimit 20"
ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (bm * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  ()
  requires
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> eC)
  ensures
    (forall+ (bid : natlt (rows/^bm *^ (cols/^bn)))
             (tid : natlt (bm/^tm *^ (bn/^tn))).
      kpre1 comb gA eA gB eB gC bm bn bk tm tn fA fB bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (bm * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (rows/^bm *^ (cols/^bn)))
  ()
  requires
    block_setup_tok (bm/^tm *^ (bn/^tn)) **
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ (bn/^tn))).
      kpre1 comb gA eA gB eB gC bm bn bk tm tn fA fB bid tid)
  ensures
    block_setup_tok (bm/^tm *^ (bn/^tn)) **
    (forall+ (tid : natlt (bm/^tm *^ (bn/^tn))).
      kpre comb gA eA gB eB gC bm bn bk slA slB tm tn fA fB sh bid tid) **
    emp (* frame *)
{
  admit();
}

ghost
fn block_teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (#_: squash (SZ.fits (bm * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (rows/^bm *^ (cols/^bn)))
  ()
  requires
    (forall+ (tid : natlt (bm/^tm *^ (bn/^tn))).
      kpost comb gA eA gB eB gC bm bn bk slA slB tm tn fA fB sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt (bm/^tm *^ (bn/^tn))).
      kpost1 comb gA eA gB eB gC bm bn bk tm tn fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (rows * cols)))
  (#_: squash (SZ.fits (bm * bn)))
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (fA fB : perm)
  ()
  requires
    (forall+ (bid : natlt (rows/^bm *^ (cols/^bn)))
             (tid : natlt (bm/^tm *^ (bn/^tn))).
      kpre1 comb gA eA gB eB gC bm bn bk tm tn fA fB bid tid) **
    emp (* frame *)
  ensures
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> MS.mmcomb comb eC eA eB)
{
  // forevery_flatten #(natlt2 mrows mcols) #_ #(natlt tile)
  //   (fun bid tid -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_unzip #(natlt2 mrows mcols & natlt tile) _ _;
  // forevery_tostar #(natlt2 mrows mcols & natlt tile) (fun _tid -> m4_pts_to gA #(1.0R /. mlayout_size lC) eA);

    // (fun (bid, tid) -> kpost1 comb tile gA gB gC eA eB 1.0R bid tid);
  admit();
}
#pop-options

inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (#eC : ematrix et rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#fA #fB : perm)
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (rows/bm * (cols/bn) <= max_blocks
               /\ (bm/tm * (bn/tn)) <= max_threads))
  ()
  : kernel_desc
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> eC))
      ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> MS.mmcomb comb eC eA eB))
= {
  nblk = rows/^bm *^ (cols/^bn);
  nthr = (bm /^ tm *^ (bn /^ tn));

  shmems_desc = shmems_desc et bm bn bk;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt (bm/^tm *^ (bn/^tn))). kpre1  comb gA eA gB eB gC bm bn bk tm tn fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt (bm/^tm *^ (bn/^tn))). kpost1 comb gA eA gB eB gC bm bn bk tm tn fA fB bid tid);
  setup      = setup    comb gA eA gB eB gC eC bm bn bk tm tn fA fB;
  teardown   = teardown comb gA eA gB eB gC eC bm bn bk tm tn fA fB;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB;
  block_teardown = block_teardown comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB ;

  kpre      = kpre  comb gA eA gB eB gC bm bn bk slA slB tm tn fA fB;
  kpost     = kpost comb gA eA gB eB gC bm bn bk slA slB tm tn fA fB;

  f = kf comb gA #eA gB #eB gC bm bn bk slA slB tm tn #() #() #fA #fB;
}

#push-options "--debug SMTFail --split_queries always"
inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA)
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (eC : ematrix et rows cols)
  (bm : szp{bm /? rows})
  (bn : szp{bn /? cols})
  (bk : szp{bk /? shared})
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (fA fB : perm)
  preserves
    cpu **
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB)
  requires
    pure (rows/bm * (cols/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    (gC |-> eC)
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel comb gA gB gC bm bn bk slA slB tm tn ());
}
