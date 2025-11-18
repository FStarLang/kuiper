module Kuiper.Poly.GEMM.BlockTiling2D

#lang-pulse

#set-options "--z3rlimit 30"

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix { ematrix }
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy.Vec
open Kuiper.Poly.GEMM.Tiled.Common.Vec
open Kuiper.VArray { varray, varray_pts_to, varray_pts_to_cell }

module B = Kuiper.Barrier
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module FB = Kuiper.Poly.GEMM.FlipFlopBarrier

inline_for_extraction noextract
let ttile
  (#et : Type0)
  (#rows : erased nat)
  (#cols : erased nat)
  (#lC : mlayout rows cols)
  (gC : gpu_matrix et lC)
  (bm : nat{bm > 0 /\ bm /?+ rows})
  (bn : nat{bn > 0 /\ bn /?+ cols})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : gpu_matrix et _
  = thread_tile (block_tile gC bm bn bid) tm tn tid

let ettile
  (#et : Type0)
  (#rows : nat)
  (#cols : nat)
  (em : ematrix et rows cols)
  (bm : nat{bm > 0 /\ bm /?+ rows})
  (bn : nat{bn > 0 /\ bn /?+ cols})
  (tm : nat{tm > 0 /\ tm /?+ bm})
  (tn : nat{tn > 0 /\ tn /?+ bn})
  (bid : natlt ((rows/bm) * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : ematrix et tm tn
  = ematrix_subtile (ematrix_subtile em bm bn (bid/(cols/bn)) (bid%(cols/bn))) tm tn (tid/(bn/tn)) (tid%(bn/tn))

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
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  =
  gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA **
  gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB **
  ttile gC bm bn tm tn bid tid |-> ettile eC bm bn tm tn bid tid **
  pure (SZ.fits (rows * cols)) **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB))

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
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (bid : enatlt (rows/bm * (cols/bn)))
  (tid : enatlt (bm/tm * (bn/tn)))
  : slprop
  =
  gA |-> Frac (fA /. (rows/tm * (cols/tn))) eA **
  gB |-> Frac (fB /. (rows/tm * (cols/tn))) eB **
  ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid

let barrier_tok
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #shared #cols : pos)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (l1 : full_mlayout bm bk)
  (l2 : full_mlayout bk bn)
  (sar1 : gpu_array et (bm * bk))
  (sar2 : gpu_array et (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : slprop
  =
  B.barrier_tok (FB.barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid)
                (FB.barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid)

unfold
let kpre
  (#et : Type0) {| scalar et, v : has_vec_cpy et |}
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
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid **
  live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))) **
  barrier_tok #_ #_ #v eA eB slA slB (fst sh) (fst (snd sh)) nthr bid **
  B.barrier_state 0

instance kpre_block_sendable
  (#et : Type0) (_:scalar et) (v : has_vec_cpy et)
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nblk: SZ.t { SZ.v nblk == rows/bm * (cols/bn) })
  (nthr: SZ.t { SZ.v nthr == bm/tm * (bn/tn) })
  (sh : c_shmems (shmems_desc et bm bn bk))
  (pf : c_shmems_inv sh)
  (i : natlt nblk)
  (j : natlt nthr)
: is_send_across block_of
  (kpre comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh i j)
= solve

unfold
let kpost
  (#et : Type0) {| scalar et, v : has_vec_cpy et |}
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
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt (rows/bm * (cols/bn)))
  (tid : natlt (bm/tm * (bn/tn)))
  : slprop
  =
  kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid **
  live_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))) **
  barrier_tok #_ #_ #v eA eB slA slB (fst sh) (fst (snd sh)) nthr bid **
  B.barrier_state (2 * (shared / bk))

instance kpost_block_sendable
  (#et : Type0) (_:scalar et) (v : has_vec_cpy et)
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (fA fB : perm)
  (nblk: SZ.t { SZ.v nblk == rows/bm * (cols/bn) })
  (nthr: SZ.t { SZ.v nthr == bm/tm * (bn/tn) })
  (sh : c_shmems (shmems_desc et bm bn bk))
  (pf : c_shmems_inv sh)
  (i : natlt nblk)
  (j : natlt nthr)
: is_send_across block_of
  (kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh i j)
= solve

let flat_matrix (#et : Type0) (#rows #cols : nat) (m : ematrix et rows cols)
  : GTot (lseq et (rows * cols))
  = Seq.init_ghost (rows * cols) (fun idx -> macc m (idx / cols) (idx % cols))

inline_for_extraction noextract
fn subproducts2d
  (#et : Type0) {| scalar et |}
  (bm bn bk: szp)
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (rchProd: larray et (tm * tn))
  (#vrchProd : erased (seq et))
  (#l1 : mlayout bm bk) {| clayout l1 |}
  (#l2 : mlayout bk bn) {| clayout l2 |}
  (gA : gpu_matrix et l1)
  (gB : gpu_matrix et l2)
  (#eA : ematrix et bm bk)
  (#eB : ematrix et bk bn)
  (#f : perm)
  (arow: szlt (bm/tm))
  (bcol : szlt (bn/tn))
  (#_ : squash (len vrchProd == tm * tn))
  preserves
    gA |-> Frac f eA **
    gB |-> Frac f eB
  requires
    rchProd |-> vrchProd
  ensures
    rchProd |-> Seq.init_ghost (tm * tn)
      (fun idx ->
        let i = idx / tn in
        let j = idx % tn in
        (vrchProd @! idx) `add` macc (MS.matmul eA eB) i j)
{
  open Pulse.Lib.Array;

  let mut dotIdx : sz = 0sz;
  while (!dotIdx <^ bk)
    invariant live dotIdx ** pure (!dotIdx <= bk)
    invariant live rchProd
  {
    (* register caches *)
    let mut rAcol : Pulse.Lib.Array.array et = [| zero #et #_ ; tm |];
    let mut rBrow : Pulse.Lib.Array.array et = [| zero #et #_ ; tn |];

    let mut j0 = 0sz;
    while (!j0 <^ tm)
      invariant exists* (vj0 : sz{SZ.v vj0 <= tm}). j0 |-> vj0
      invariant exists* (vrAcol : lseq et tm).
        rAcol |-> vrAcol **
        pure (forall (k : natlt (!j0)).
                vrAcol @! k == macc eA (tm * arow + k) !dotIdx)
    {
      pts_to_len rAcol;
      let va = gpu_matrix_read gA (tm *^ arow +^ !j0) !dotIdx;
      rAcol.(!j0) <- va;
      j0 := !j0 +^ 1sz;
    };
    with vrAcol. assert rAcol |-> vrAcol;
    assert
        pure (forall (k : natlt tm).
                vrAcol @! k == macc eA (tm * arow + k) !dotIdx);

    let mut j1 = 0sz;
    while (!j1 <^ tn)
      invariant exists* (vj1 : sz{SZ.v vj1 <= tn}). j1 |-> vj1
      invariant exists* (vrBrow : lseq et tn).
        rBrow |-> vrBrow **
        pure (forall (k : natlt (!j1)).
                vrBrow @! k == macc eB !dotIdx (tn * bcol + k))
    {
      pts_to_len rBrow;
      let vb = gpu_matrix_read gB !dotIdx (tn *^ bcol +^ !j1);
      rBrow.(!j1) <- vb;
      j1 := !j1 +^ 1sz;
    };
    with vrBrow. assert rBrow |-> vrBrow;
    assert
        pure (forall (k : natlt tn).
                vrBrow @! k == macc eB !dotIdx (tn * bcol + k));

    let mut resIdxM = 0sz;
    while (!resIdxM <^ tm)
      invariant live resIdxM ** pure (!resIdxM <= tm)
      invariant live rchProd
    {
      let mut resIdxN = 0sz;
      while (!resIdxN <^ tn)
        invariant live resIdxN ** pure (!resIdxN <= tn)
        invariant live rchProd
      {
        pts_to_len rAcol;
        pts_to_len rBrow;
        pts_to_len rchProd;

        (* works on arrays and therefore does not have the nice matrix abstraction *)
        let ra = rAcol.(!resIdxM);
        let rb = rBrow.(!resIdxN);
        let idx = !resIdxM *^ tn +^ !resIdxN;
        let old = rchProd.(idx);
        let mad = old `add` (ra `mul` rb);
        rchProd.(idx) <- mad;

        resIdxN := !resIdxN +^ 1sz;
      };

      resIdxM := !resIdxM +^ 1sz;
    };

    dotIdx := !dotIdx +^ 1sz;
  };

  with v. assert rchProd |-> v;
  assume pure (v == Seq.init_ghost (tm * tn)
    (fun idx ->
      let i = idx / tn in
      let j = idx % tn in
      (vrchProd @! idx) `add` macc (MS.matmul eA eB) i j));
}

inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #cols : sz)
  (#shared : sz)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (rchProd: larray et (tm * tn))
  (#lC : mlayout rows cols)
  {| clayout lC |}
  (gC : gpu_matrix et lC)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  // (#_ : squash (SZ.fits (bm/tm * (bn/tn))))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  requires
    ttile gC bm bn tm tn bid tid |-> ettile eC bm bn tm tn bid tid **
    live rchProd
  ensures
    ttile gC bm bn tm tn bid tid |-> ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid **
    live rchProd
{
  let t_tile = ttile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid);
  assert (rewrites_to t_tile (ttile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) (SZ.v bid) (SZ.v tid)));

  let mut resIdxM = 0sz;
  while (!resIdxM <^ tm)
    invariant live resIdxM ** pure (!resIdxM <= tm)
    invariant live rchProd
    invariant exists* v. t_tile |-> v
  {
    let mut resIdxN = 0sz;
    while (!resIdxN <^ tn)
      invariant live resIdxN ** pure (!resIdxN <= tn)
      invariant live rchProd
      invariant exists* v. t_tile |-> v
    {
      open Pulse.Lib.Array;
      pts_to_len rchProd;

      (* Combine the new result in the register cache to the value from gC and
      overwrite the the cell in gC *)
      let v0 = gpu_matrix_read t_tile !resIdxM !resIdxN;
      let v1 = rchProd.(!resIdxM *^ tn +^ !resIdxN);
      let v' = comb v0 v1;
      gpu_matrix_write t_tile !resIdxM !resIdxN v';

      resIdxN := !resIdxN +^ 1sz;
    };

    resIdxM := !resIdxM +^ 1sz;
  };

  with m. assert gpu_matrix_pts_to t_tile m;
  assume pure (m == ettile (MS.mmcomb comb eC eA eB) bm bn tm tn bid tid);

  ()
}

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : gpu_matrix et lA)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB)
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC)
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#fA #fB : perm)
  (nthr : nat {nthr == bm/tm * (bn/tn)})
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt (bm/tm * (bn/tn)))
  ()
  norewrite
  requires
    gpu **
    kpre comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (rows/bm * (cols/bn)) bid
  ensures
    gpu **
    kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid **
    thread_id (bm/tm * (bn/tn)) tid **
    block_id (rows/bm * (cols/bn)) bid
{
  unfold_c_shmems sh (`%shmems_desc);
  let (sarA, (sarB, _)) = sh;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;
  gpu_matrix_pts_to_ref gA;
  gpu_matrix_pts_to_ref gB;

  unfold barrier_tok eA eB slA slB sarA sarB nthr bid;

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

  (* register caches *)
  assert pure (tm <= rows);
  assert pure (tn <= cols);
  assert pure (tm * tn <= rows * cols);
  assert pure (SZ.fits (tm * tn)); // should be obvious
  let mut rchProd : Pulse.Lib.Array.array et = [| zero #et #_ ; tm*^tn |];

  with x. fold FB.bp_sharing sA x nthr;
  with x. fold FB.bp_sharing sB x nthr;

  let mut bkIdx  : sz = 0sz;
  while (!bkIdx <^ num_k_tiles)
    invariant live bkIdx ** pure (!bkIdx <= num_k_tiles)
    invariant live rchProd
    invariant B.barrier_state (2 * !bkIdx) **
        (exists* (x : ematrix _ _ _). FB.bp_sharing sA x nthr) **
        (exists* (x : ematrix _ _ _). FB.bp_sharing sB x nthr)
  {
    even_2x !bkIdx;
    #set-options "--z3rlimit 60" {
    rewrite
        (exists* emA. FB.bp_sharing sA emA nthr) **
        (exists* emB. FB.bp_sharing sB emB nthr)
      as FB.barrier_p eA eB sA sB nthr bid (2 * !bkIdx) tid;
    };

    B.barrier_wait ();

    rewrite (FB.barrier_q eA eB sA sB nthr bid (2 * !bkIdx) tid)
        as live_strided_chunks sA nthr tid **
           live_strided_chunks sB nthr tid;

    copy_tiles_out_of_matrices_vec bm bn bk sA sB gA gB mrow !bkIdx mcol (bm/^tm*^(bn/^tn)) tid;

    assert own_strided_chunks sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid;
    assert own_strided_chunks sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid;

    odd_2x1 !bkIdx;
    assert (pure (odd (2 * !bkIdx + 1)));
    rewrite own_strided_chunks sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid **
            own_strided_chunks sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid
         as FB.barrier_p eA eB sA sB nthr bid (2 * !bkIdx + 1) tid;

    B.barrier_wait ();

    even_2x (!bkIdx + 1);
    assert (pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 2));
    assert (pure (even (2 * !bkIdx + 2)));
    assert (pure (odd (2 * !bkIdx + 1)));
    assert pure ((2 * !bkIdx + 1) < (2 * (shared /^ bk)));
    assert pure ((2 * !bkIdx + 1) / 2 == !bkIdx);
    #set-options "--z3rlimit 60" {
    rewrite FB.barrier_q eA eB sA sB nthr bid (2 * !bkIdx + 1) tid
    as
      FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr **
      FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;
    };

    unfold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    unfold FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    pts_to_len rchProd;
    subproducts2d bm bn bk tm tn rchProd sA sB threadRow threadCol;

    fold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    fold FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    // What the hell.
    assert (pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 1 + 1));

    bkIdx := !bkIdx +^ 1sz;
  };

  with emA. unfold FB.bp_sharing sA emA nthr;
  with emB. unfold FB.bp_sharing sB emB nthr;

  let vbkIdx = !bkIdx;
  assert pure (vbkIdx <= num_k_tiles);
  assert pure (not (vbkIdx < num_k_tiles));
  assert pure (vbkIdx == num_k_tiles); // Somehow this is flaky.

  epilogue comb bm bn tm tn rchProd gC eA eB eC bid tid;

  gpu_matrix_concr sA; rewrite each core sA as sarA;
  gpu_matrix_concr sB; rewrite each core sB as sarB;

  rewrite
    B.barrier_tok (FB.barrier_p eA eB sA sB nthr bid)
      (FB.barrier_q eA eB sA sB nthr bid)
  as
    B.barrier_tok (FB.barrier_p eA eB (from_array slA sarA)
          (from_array slB sarB)
          nthr bid)
      (FB.barrier_q eA eB (from_array slA sarA)
          (from_array slB sarB)
          nthr bid);
  fold barrier_tok eA eB slA slB sarA sarB nthr bid;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  fold_c_shmems sh #(1.0R /. (bm/tm * (bn/tn))) (`%shmems_desc);
  ()
}

ghost
fn setup
  (#et : Type0) {| scalar et, has_vec_cpy et |}
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
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
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
      kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid) **
    emp (* frame *)
{
  admit(); // Incomplete as mentioned in paper
}

ghost
fn block_setup
  (#et : Type0) {| scalar et, has_vec_cpy et |}
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
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    can_create_barrier nthr **
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid)
  ensures
    consumed_can_create_barrier **
    (forall+ (tid : natlt nthr).
      kpre comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid) **
    emp (* frame *)
{
  admit(); // Incomplete as mentioned in paper
}

ghost
fn block_teardown
  (#et : Type0) {| scalar et, has_vec_cpy et |}
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
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  // (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_: squash (SZ.fits (rows * cols)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  (sh : c_shmems (shmems_desc et bm bn bk))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB nthr sh bid tid) **
    emp (* frame *)
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid)
{
  admit(); // Incomplete as mentioned in paper
}

ghost
fn teardown
  (#et : Type0) {| scalar et, has_vec_cpy et |}
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
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_: squash (SZ.fits (rows * cols)))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk)
             (tid : natlt nthr).
      kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  admit(); // Incomplete as mentioned in paper
}

#push-options "--z3rlimit_factor 4 --split_queries no --fuel 1 --ifuel 1"
#restart-solver
inline_for_extraction noextract
let mk_kernel
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/tm * (bn/tn)})
  (#_ : squash (chunk et * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (rows/bm * (cols/bn) <= max_blocks
               /\ (bm/tm * (bn/tn)) <= max_threads))
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
= {
  nblk;// = rows/^bm *^ (cols/^bn);
  nthr;// = (bm /^ tm *^ (bn /^ tn));

  shmems_desc = shmems_desc et bm bn bk;

  frame = emp;
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 comb gA eA gB eB gC eC bm bn bk tm tn fA fB bid tid);

  setup      = setup    comb gA eA gB eB gC eC bm bn bk tm tn nblk nthr fA fB;
  teardown   = teardown comb gA eA gB eB gC eC bm bn bk tm tn nblk nthr fA fB;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup comb gA eA gB eB gC eC bm bn bk slA slB tm tn nblk nthr fA fB;
  block_teardown = block_teardown comb gA eA gB eB gC eC bm bn bk slA slB tm tn nblk nthr fA fB ;

  kpre      = kpre  comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB (SZ.v nthr);
  kpost     = kpost comb gA eA gB eB gC eC bm bn bk slA slB tm tn fA fB (SZ.v nthr);

  f = kf comb gA #eA gB #eB gC bm bn bk slA slB tm tn #() #() #() #() #fA #fB (SZ.v nthr);

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=solve;
  kpost_sendable=solve;
}
#pop-options

inline_for_extraction noextract
fn mmcomb_gpu
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#eA : ematrix et rows shared)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#eB : ematrix et shared cols)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (rows/bm * (cols/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  (* fixed the inner layouts, or we'd have to propagate this everywhere? *)
  launch_sync (mk_kernel comb gA gB gC bm bn bk slA slB tm tn (rows/^bm *^ (cols/^bn)) (bm/^tm *^ (bn/^tn)) ());
}
