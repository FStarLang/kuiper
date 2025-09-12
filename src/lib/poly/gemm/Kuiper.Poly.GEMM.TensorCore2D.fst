module Kuiper.Poly.GEMM.TensorCore2D

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

open Kuiper.Poly.GEMM.Copy
open Kuiper.Poly.GEMM.Tiled.Common

open Pulse.Lib.Array

let array_fragment_pts_to
  (#et : Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#l : fragment_layout)
  ([@@@mkey] farr: array (fragment et knd m n k l))
  (ems : seq (value_for et knd m n k))
  : slprop = 
    exists* (s: lseq (fragment et knd m n k l) (len ems)).
      farr |-> s **
      forall+ (i : natlt (Seq.length s)).
        (s @! i) |-> (ems @! i)

ghost
fn gpu_array_fragment_extract
  (#et:Type0)
  (#knd : fragment_kind)
  (#m #n #k : nat)
  (#l : fragment_layout)
  ([@@@mkey] farr: array (fragment et knd m n k l))
  (ems : seq (value_for et knd m n k))
  (i : natlt (len ems))
  (#em : (let o, p = (dims_for knd m n k) in ematrix et o p))
  (#f : perm)
  requires
    array_fragment_pts_to 
  ensures
    factored
      (gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)



// inline_for_extraction noextract
// fn populate_fragments
//   (#et : Type0)
//   (#m #n #k : nat)
//   (#knd : fragment_kind)
//   (#m #n #k : nat)
//   (#l : fragment_layout)
//   (arr: array (fragment et knd m n k l))
//   (#ems : erased (seq (value_for et knd m n k)))
//   (gm : gpu_matrix et (dims_for knd m n k)._0 (dims_for knd m n k)._1)
//   preserves
//     gpu **
//     gm |-> Frac f em 
//   requires 
//     array_fragment_pts_to arr_frags ems
//   ensures
//     exists* ems'. array_fragment_pts_to arr_frags ems'
// {

// }

#push-options "--debug SMTFail --split_queries always"
#push-options "--print_implicits"
inline_for_extraction noextract
fn subproducts_tc_2d
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk: szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  (wm : szp{wm * tm /? bm})
  (wn : szp{wn * tn /? bn})
  (aFrags : array (fragment et_ab FragA tm tn tk FragLRM))
  (#emAFrags : erased (lseq (ematrix et_ab tm tk) wm))
  (bFrags : array (fragment et_ab FragB tm tn tk FragLRM))
  (#emBFrags : erased (seq (ematrix et_ab tk tn)))
  (accumFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (#emAccumFrags : erased (seq (ematrix et_acc tm tn)))
  (gA : gpu_matrix et_ab (R.row_major bm bk))
  (gB : gpu_matrix et_ab (R.row_major bk bn))
  (#eA : ematrix et_ab bm bk)
  (#eB : ematrix et_ab bk bn)
  (#fA #fB : perm)
  (arow: szlt (bm/(wm*tm)))
  (bcol : szlt (bn/(wn*tn)))
  preserves
    gpu **
    pure (Seq.length emAFrags == wm /\
          Seq.length emBFrags == wn /\
          Seq.length emAccumFrags == wm * wn /\
          SZ.fits (tm * tn)) **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    array_fragment_pts_to aFrags emAFrags **
    array_fragment_pts_to bFrags emBFrags **
    array_fragment_pts_to accumFrags emAccumFrags
  ensures
    exists* emAFrags' emBFrags' emAccumFrags'.
      array_fragment_pts_to aFrags emAFrags' **
      array_fragment_pts_to bFrags emBFrags' **
      array_fragment_pts_to accumFrags emAccumFrags'
{
  let mut dotIdx : sz = 0sz;
  while (SZ.(!dotIdx <^ (bk/^tk)))
    invariant
      exists*
        (vdotIdx : sz{vdotIdx <= bk})
        (emAFrags : erased (lseq (ematrix et_ab tm tk) wm))
        (emBFrags : erased (lseq (ematrix et_ab tk tn) wn))
        (emAccumFrags : erased (lseq (ematrix et_acc tm tn) (wm*wn))).
          dotIdx |-> vdotIdx **
          array_fragment_pts_to aFrags emAFrags **
          array_fragment_pts_to bFrags emBFrags **
          array_fragment_pts_to accumFrags emAccumFrags
  {
    open Pulse.Lib.Array;
    // Would like to write this here, but then the loop does not type check. Whye does it want to prove that gA |-> eA?
    // let tile_for_tc_tiles = gpu_matrix_extract_tile_ro' gA (wm*tm) (SZ.v tk) (SZ.v arow) (SZ.v !dotIdx);

    let mut i0 = 0sz;
    while (SZ.(!i0 <^ wm))
      invariant
        exists*
          (vi : sz{vi <= wm})
          (emAFrags : erased (lseq (ematrix et_ab tm tk) wm)).
            i0 |-> vi **
            array_fragment_pts_to aFrags emAFrags
    {
      // create tile for tensor core tiles that belong to the warp
      let tile_for_tc_tiles = gpu_matrix_extract_tile_ro' gA (wm*tm) (SZ.v tk) (SZ.v arow) (SZ.v !dotIdx);
      let a_tile = gpu_matrix_extract_tile_ro' tile_for_tc_tiles (SZ.v tm) (SZ.v tk) (SZ.v !i0) 0;
      // Expected are only nats, but later on when the tile is used we need to concretize.
      // In this case wm*tm and 0 must be concretizable which means that either we have to write (SZ.v (wm*^tm)) and (SZ.v 0sz),
      // which is odd, because a nat is expected, or there must be type classes that can resolve this.
      assert (rewrites_to a_tile (
        gpu_matrix_subtile (
          gpu_matrix_subtile gA (wm*tm) (SZ.v tk) (SZ.v arow) (SZ.v !dotIdx))
          (SZ.v tm) (SZ.v tk) (SZ.v !i0) 0));

      unfold array_fragment_pts_to aFrags;
      with saFrags. assert aFrags |-> saFrags;
      let a_frag = aFrags.(!i0);
      forevery_extract' #(natlt (Seq.length saFrags)) !i0 _;

      mma_loadA a_frag a_tile;

      // TODO purification does not seem to work under lambdas
      let vi = !i0;
      let vdotIdx = !dotIdx;
      Pulse.Lib.Forall.elim_forall
        // type annotation is required, although it is not required in the RefArray example
        (fun (i : natlt (Seq.length saFrags)) ->
            if i = (SZ.v vi)
            then ((saFrags @! i) |-> (ematrix_subtile (ematrix_subtile eA (wm*tm) tk arow vdotIdx) tm tk vi 0))
            else ((saFrags @! i) |-> (emAFrags @! i)));

      assume 
      pure (forall (x: natlt (Seq.Base.length saFrags) {~(x == SZ.v vi)}).
        (match x = SZ.v vi with
          | true ->
            Seq.Base.index saFrags x |->
            ematrix_subtile (ematrix_subtile eA
                  (SZ.v wm * SZ.v tm)
                  (SZ.v tk)
                  (SZ.v arow)
                  (SZ.v vdotIdx))
              (SZ.v tm)
              (SZ.v tk)
              (SZ.v vi)
              0
          | _ -> Seq.Base.index saFrags x |-> Seq.Base.index emAFrags x) ==
      fragment_pts_to (Seq.Base.index saFrags x) (Seq.Base.index emAFrags x));

      Pulse.Lib.Trade.elim_trade _ (forall+ (x: natlt (Seq.Base.length saFrags)).
          match x = SZ.v vi with
          | true ->
            Seq.Base.index saFrags x |->
            ematrix_subtile (ematrix_subtile eA
                  (SZ.v wm * SZ.v tm)
                  (SZ.v tk)
                  (SZ.v arow)
                  (SZ.v vdotIdx))
              (SZ.v tm)
              (SZ.v tk)
              (SZ.v vi)
              0
          | _ -> Seq.Base.index saFrags x |-> Seq.Base.index emAFrags x);
      admit();



      i0 := !i0 +^ 1sz;
    };

  //   let mut i1 = 0sz;
  //   while (SZ.(!i1 <^ wn))
  //     invariant
  //       exists*
  //         (vi : sz{vi <= wn})
  //         (cbFrags : erased (lseq (ematrix et_ab tk tn) wn)).
  //           i1 |-> vi **
  //           bFrags |-> vbFrags
  //   {
  //     let tile_for_tc_tiles = gpu_matrix_extract_tile_ro' gB (SZ.v tk) (wn*tn) (SZ.v !dotIdx) (SZ.v bcol);
  //     let b_tile = gpu_matrix_extract_tile_ro' tile_for_tc_tiles (SZ.v tk) (SZ.v tn) 0 (SZ.v bcol);
  //     let i1' = !i1;
  //     assert (rewrites_to b_tile (
  //       gpu_matrix_subtile (
  //         gpu_matrix_subtile gB (SZ.v tk) (wn*tn) (SZ.v didx) (SZ.v bcol))
  //         (SZ.v tk) (SZ.v tn) 0 (SZ.v i1')
  //       )
  //     );

  //     mma_loadB bFrags.(!i1) b_tile;

  //     i1 := !i1 +^ 1sz;
  //   };

  //   let mut resIdxM = 0sz;
  //   while (SZ.(!resIdxM <^ tm))
  //     invariant
  //       exists*
  //         (vresIdxM : sz{vresIdxM <= tm})
  //         (vaccumFrags : erased (lseq (ematrix et_acc tm tn) (wm*wn))).
  //           resIdxM |-> vresIdxM **
  //           accumFrags |-> vaccumFrags
  //   {
  //     let mut resIdxN = 0sz;
  //     while (SZ.(!resIdxN <^ tn))
  //       invariant
  //         exists*
  //           (vresIdxN : sz{vresIdxN <= tn})
  //           (vaccumFrags : erased (lseq (ematrix et_acc tm tn) (wm*wn))).
  //             resIdxN |-> vresIdxN **
  //             accumFrags |-> vaccumFrags
  //     {
  //       let aFrag = aFrags.(!resIdxM);
  //       let bFrag = bFrags.(!resIdxN);
  //       let accFrag = accumFrags.(!resIdxM *^ tn +^ !resIdxN);

  //       // can this assertion be removed?
  //       let iM = !resIdxM;
  //       let iN = !resIdxN;
  //       assert(pure(SZ.fits(iM *^ tn +^ iN)));

  //       mma_sync' aFrag bFrag accFrag;
  //       resIdxN := !resIdxN +^ 1sz;
  //     };

  //     resIdxM := !resIdxM +^ 1sz;
  //   };

  //   dotIdx := !dotIdx +^ 1sz;
  }
}

let live_thread_tile
  (#et : Type0) {| scalar et |}
  // Since this is an slprop, I would like to not erase the nat.
  // Unfortunately, when unfolding live_thread_tile, after passing
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
  // see comment at live_thread_tile for why we explicitly
  // have to convert (so that the reaveal coercion can kick in)
  live_thread_tile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) bid tid

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
  live_thread_tile gC (SZ.v bm) (SZ.v bn) (SZ.v tm) (SZ.v tn) bid tid

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
      live_tile_stride_cells m1 nthr tid **
      live_tile_stride_cells m2 nthr tid

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
fn epilogue
  (#et : Type0) {| scalar et |}
  // (comb : binop et)

  // rows should be an erased nat because not concrete value is required, but
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
    // see comment in live_thread_tile for why SZ.v
    live_thread_tile gC bm bn tm tn bid tid **
    (exists* vaccumFrag.
      accumFrag |-> vaccumFrag)
  ensures
    gpu **
    live_thread_tile gC bm bn tm tn bid tid **
    (exists* vaccumFrag.
      accumFrag |-> vaccumFrag)
{
  unfold live_thread_tile gC bm bn tm tn bid tid;

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
  fold live_thread_tile gC bm bn tm tn bid tid;
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
  unfold live_thread_tile gC bm bn tm tn bid tid;
  let t_tile = thread_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (SZ.v tm) (SZ.v tn) (SZ.v tid);
  assert (rewrites_to t_tile (thread_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (SZ.v tm) (SZ.v tn) (SZ.v tid)));
  mma_loadAccum accumFrag t_tile;
  fold live_thread_tile gC bm bn tm tn bid tid;

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
        (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. (bm/tm * (bn/tn))) x) **
        (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. (bm/tm * (bn/tn))) x)
      as barrier_p sA sB (bm/tm * (bn/tn)) (2 * vbkIdx) tid;

    B.barrier_wait ();
    rewrite (barrier_q sA sB (bm/tm * (bn/tn)) (2 * vbkIdx) tid)
        as live_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
           live_tile_stride_cells sB (bm/tm * (bn/tn)) tid;

    populate_shmem bm bn bk tm tn sA sB gA gB mrow !bkIdx mcol tid;

    assert (B.barrier_tok (barrier_p sA sB (bm/tm * (bn/tn))) (barrier_q sA sB (bm/tm * (bn/tn))) (2 * vbkIdx + 1) tid);
    odd_2x1 vbkIdx;
    assert (pure (odd (2 * vbkIdx + 1)));
    rewrite live_tile_stride_cells sA (bm/tm * (bn/tn)) tid **
            live_tile_stride_cells sB (bm/tm * (bn/tn)) tid
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
