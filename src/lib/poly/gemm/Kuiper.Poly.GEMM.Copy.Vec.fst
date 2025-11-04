module Kuiper.Poly.GEMM.Copy.Vec

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.Matrix.Vectorized
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module SZ = Kuiper.SizeT
module GR = Pulse.Lib.GhostReference
open Pulse.Lib.Trade { trade }

let in_chunk_covers_all
  (chunk : pos)
  (rows cols : nat)
  (nthr : pos)
  (ij : (natlt rows & natlt cols))
  : Lemma (exists tid. in_chunk chunk rows cols nthr tid ij)
  =
  let flat_idx = ij._1 * cols + ij._2 <: nat in
  let chunk_idx = flat_idx / chunk in
  let tid = chunk_idx % nthr in
  assert (in_chunk chunk rows cols nthr tid ij);
  ()

let in_chunk_no_overlap
  (chunk : pos)
  (rows cols : nat)
  (nthr : pos)
  (ij : (natlt rows & natlt cols))
  (tid1 tid2 : natlt nthr)
  : Lemma (requires in_chunk chunk rows cols nthr tid1 ij /\
                    in_chunk chunk rows cols nthr tid2 ij)
          (ensures tid1 == tid2)
  =
  ()

ghost
fn own_strided_chunks_rw
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (m : gpu_matrix et lm)
  (nthr : nat)
  (tid : natlt nthr)
  (em1 em2 : ematrix et rows cols)
  requires
    pure (
      forall (i : natlt rows) (j : natlt cols).
        in_chunk (chunk et) rows cols nthr tid (i,j) ==>
        macc em1 i j == macc em2 i j)
  requires
    own_strided_chunks m em1 nthr tid
  ensures
    own_strided_chunks m em2 nthr tid
{
  unfold own_strided_chunks m em1 nthr tid;
  forevery_map #(ij : (natlt rows & natlt cols){in_chunk (chunk et) rows cols nthr tid ij})
    (fun ij -> gpu_matrix_pts_to_cell m ij._1 ij._2 (macc em1 ij._1 ij._2))
    (fun ij -> gpu_matrix_pts_to_cell m ij._1 ij._2 (macc em2 ij._1 ij._2))
    fn ij {
      assert pure (in_chunk (chunk et) rows cols nthr tid ij);
      rewrite
        gpu_matrix_pts_to_cell m ij._1 ij._2 (macc em1 ij._1 ij._2)
      as
        gpu_matrix_pts_to_cell m ij._1 ij._2 (macc em2 ij._1 ij._2);
    };
  fold own_strided_chunks m em2 nthr tid;
}

ghost
fn split_matrix_into_strided_chunks
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (m : gpu_matrix et lm)
  (#em : ematrix et rows cols)
  (nthr : pos)
  requires
    m |-> em
  ensures
    pure (SZ.fits (mlayout_size lm))
  ensures
    forall+ (tid : natlt nthr).
      own_strided_chunks m em nthr tid
{
  gpu_matrix_pts_to_ref m;
  gpu_matrix_explode m;
  forevery_flatten _;
  Classical.forall_intro (in_chunk_covers_all (chunk et #_ #hvc) rows cols nthr);
  forevery_refine_ext #_ #(fun _ -> True)
    (fun (ij : (natlt rows & natlt cols)) ->
      exists tid. in_chunk (chunk et #_ #hvc) rows cols nthr tid ij)
    _;
  Classical.forall_intro_3 (fun ij tid1 -> Classical.move_requires
                             (in_chunk_no_overlap (chunk et #_ #hvc) rows cols nthr ij tid1));
  forevery_split_or_n _ _;
  ghost
  fn aux (tid : natlt nthr)
    requires
      forall+ (ij : (natlt rows & natlt cols){in_chunk (chunk et #_ #hvc) rows cols nthr tid ij}).
        gpu_matrix_pts_to_cell m ij._1 ij._2 (macc em ij._1 ij._2)
    ensures
      own_strided_chunks m em nthr tid
  {
    fold own_strided_chunks m em nthr tid;
  };
  forevery_map _ _ aux;
}

ghost
fn join_matrix_from_strided_chunks
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (m : gpu_matrix et lm)
  (#em : ematrix et rows cols)
  (nthr : pos)
  requires
    pure (SZ.fits (mlayout_size lm))
  requires
    forall+ (tid : natlt nthr).
      own_strided_chunks m em nthr tid
  ensures
    m |-> em
{
  assert pure (SZ.fits (mlayout_size lm));
  forevery_map
    (fun tid -> own_strided_chunks m em nthr tid)
    (fun tid -> forall+ (ij : (natlt rows & natlt cols){in_chunk (chunk et #_ #hvc) rows cols nthr tid ij}).
        gpu_matrix_pts_to_cell m ij._1 ij._2 (macc em ij._1 ij._2))
    fn tid { unfold own_strided_chunks m em nthr tid };
  forevery_join_or_n (fun (tid : natlt nthr) ij -> in_chunk (chunk et #_ #hvc) rows cols nthr tid ij)
    (fun ij -> gpu_matrix_pts_to_cell m ij._1 ij._2 (macc em ij._1 ij._2));
  Classical.forall_intro (in_chunk_covers_all (chunk et #_ #hvc) rows cols nthr);
  Classical.forall_intro_3 (fun ij tid1 -> Classical.move_requires
                             (in_chunk_no_overlap (chunk et #_ #hvc) rows cols nthr ij tid1));
  forevery_refine_ext #_
    #(fun (ij : (natlt rows & natlt cols)) ->
      exists tid. in_chunk (chunk et #_ #hvc) rows cols nthr tid ij)
    (fun _ -> True)
    _;
  forevery_unflatten' _;
  gpu_matrix_implode m;
}

ghost
fn join_matrix_from_strided_chunks_underspec
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (m : gpu_matrix et lm)
  (nthr : pos)
  requires
    pure (SZ.fits (mlayout_size lm))
  requires
    forall+ (tid : natlt nthr).
      live_strided_chunks m nthr tid
  ensures
    live m
{
  forevery_map
    (fun (tid : natlt nthr) -> live_strided_chunks m nthr tid)
    (fun (tid : natlt nthr) -> exists* em. own_strided_chunks m em nthr tid)
    fn tid { unfold live_strided_chunks m nthr tid };

  (* Combine the matrices into a single matrix coincinding for every stride. *)
  let ff = forevery_exists #(natlt nthr) _;
  let em' : ematrix et rows cols =
    (mkM fun i j ->
       let flat_idx : nat = i * cols + j in
       let chunk_idx = flat_idx / chunk et in
       let tid = chunk_idx % nthr in
       macc (ff tid) i j);

  forevery_map
    (fun (tid : natlt nthr) -> own_strided_chunks m (ff tid) nthr tid)
    (fun (tid : natlt nthr) -> own_strided_chunks m em' nthr tid)
    fn tid {
      unfold own_strided_chunks m (ff tid) nthr tid;
      forevery_map
        #(ij : (natlt rows & natlt cols){in_chunk (chunk et #_ #hvc) rows cols nthr tid ij})
        (fun ij -> gpu_matrix_pts_to_cell m ij._1 ij._2 (macc (ff tid) ij._1 ij._2))
        (fun ij -> gpu_matrix_pts_to_cell m ij._1 ij._2 (macc em' ij._1 ij._2))
        fn ij { () };
      fold own_strided_chunks m em' nthr tid;
    };

  join_matrix_from_strided_chunks m nthr;
  assert m |-> em';
}

let freeze (p : slprop) : slprop = p

let mul_inv_2 (a b c d:nat)
: Lemma (a == b * c * d /\ c<>0 /\ d<>0 ==> b == (a / d) / c)
= ()

let add_helper
  (i git nthr chunk_et : int)
  : Lemma (requires i == git * nthr * chunk_et)
          (ensures i + nthr * chunk_et == (git + 1) * nthr * chunk_et)
  = ()

let divides_helper
  (d : pos)
  (a b r c : nat)
  : Lemma (requires d /? a /\ d /? b /\ d /? c)
          (ensures d /? (a + b * r + c))
  = lemma_divides_product_l d b r;
    lemma_divides_sum d a (b * r);
    lemma_divides_sum d (a + b * r) c;
    ()

(* A matrix representing em1 "fading into" em2 as we copy chunks. *)
let em_fade
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#rows #cols : pos)
  (em1 : ematrix et rows cols)
  (em2 : ematrix et rows cols)
  (nthr : pos)
  (it : nat)
  : ematrix et rows cols
  = mkM (fun i j ->
      let flat_idx = i * cols + j <: nat in
      let chunk_idx = flat_idx / (chunk et) in
      if chunk_idx / nthr < it
      then macc em2 i j
      else macc em1 i j)

let nop_tactic () : Tactics.Tac unit = ()

#push-options "--z3rlimit 45 --fuel 0 --ifuel 1"
// NB: The scalar constraint is only here so we can use 'zero' as an initializer
// for a local array... would be gone if we had uninitialized local arrays.
inline_for_extraction noextract
fn cp_matrix_vec
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (rows cols: sz)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  {| src_str : strided_row_major lsrc |}
  (src : gpu_matrix et lsrc)
  (#f : perm)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (#edst : ematrix et rows cols)
  (nthr : szp)
  (tid : szlt nthr)
  preserves gpu
  preserves
    src |-> Frac f esrc
  requires
    pure (SZ.fits (rows * cols + nthr - 1)) **
    pure (chunk et /?+ cols) **
    pure (chunk et * nthr /?+ (rows * cols)) **
    pure (aligned 16 (core src)) **
    pure (rows * cols > 0)
  requires
    own_strided_chunks dst edst nthr tid
  ensures
    own_strided_chunks dst esrc nthr tid
{
  // These should be exposed to the precondition and bubbled up.
  assume pure (chunk et /?+ src_str.stride);
  assume pure (chunk et /?+ src_str.offset);
  open FStar.SizeT;
  let mlen = rows *^ cols;

  assert pure (SZ.fits (tid * chunk et)); // ?
  let offset : sz = tid *^ chunk et;
  let mut i : sz = 0sz;

  (* It's very important to make the initializer and bound of this loop
  independent of the tid, this way NVCC can see clearly that it can be unrolled
  (when rows/cols are defined constants). Note, something like:
        uint32_t i2 = threadIdx.x * 8U;
        for (; i2 < 1024U; i2 += 256U) { ... }
  is NOT unrolled, even if NVCC could see statically that threadIdx.x < 32U, and
  therefore this is always 4 iterations. *)

  assert pure (Kuiper.EMatrix.equal edst (em_fade edst esrc nthr 0));
  rewrite own_strided_chunks dst edst nthr tid
       as own_strided_chunks dst (em_fade edst esrc nthr 0) nthr tid;

  let git = Pulse.Lib.GhostReference.alloc #nat 0;
  while ((!i <^ mlen))
    invariant
      live i ** live git **
      pure (SZ.v !i == GR.read git * nthr * chunk et) **
      pure (GR.read git <= (rows*cols) / (nthr * chunk et)) **
      own_strided_chunks dst (em_fade edst esrc nthr (GR.read git)) nthr tid
  {
    assert pure (GR.read git < (rows*cols) / (chunk et * nthr));
    let vi = !i;
    assert pure (vi + offset < mlen); // prove this, it follows from rounding down, it works some times
    let mut local = [| zero #et #_; chunk et |];

    assert pure (SZ.fits (!i + nthr * chunk et));
    let row = (!i +^ offset) /^ cols; assert (rewrites_to row ((!i +^ offset) /^ cols));
    let col = (!i +^ offset) %^ cols; assert (rewrites_to col ((!i +^ offset) %^ cols));
    assert pure (chunk et /?+ cols);
    assert pure (chunk et /?+ offset);
    assert pure (chunk et /?+ !i);
    lemma_divides_mod_op (chunk et) (!i +^ offset) cols;
    assert pure (chunk et /?+ col);
    assert (pure (col + chunk et <= cols));
    assert pure (SZ.v offset == tid * chunk et);
    assert pure (row < rows);
    assert pure (col < cols - chunk et + 1);

    assert pure (chunk et * size #et == 16);
    src_str.pf row col;
    divides_helper (chunk et) src_str.offset src_str.stride row col;
    assert pure (chunk et /? cell_of_pos lsrc row col);
    assert pure (16 /?+ (cell_of_pos lsrc row col * size #et));

    gpu_matrix_vec_read src row col local;

    let ite : erased nat = GR.read git;
    mul_inv_2 ite (!i) nthr (chunk et);

    let mut k = 0sz;
    while ((!k <^ chunk et))
      invariant live k ** pure (!k <= chunk et)
      invariant
        exists* em'. // TODO: functional spec
          own_strided_chunks dst em' nthr tid
    {
      with em'. unfold own_strided_chunks dst em' nthr tid;
      assume pure (col + !k < cols);
      let ecell : erased (natlt rows & natlt cols) = Mktuple2 #(natlt rows) #(natlt cols) row (col +^ !k);
      assume pure (in_chunk (chunk et) rows cols nthr tid ecell);
      forevery_remove'
        #(natlt rows & natlt cols)
        (fun (ij : (natlt rows & natlt cols)) -> in_chunk (chunk et) rows cols nthr tid ij)
        _ ecell;

      assert (gpu_matrix_pts_to_cell dst (reveal ecell)._1 (reveal ecell)._2
                  (macc em' (reveal ecell)._1 (reveal ecell)._2));
      rewrite
        gpu_matrix_pts_to_cell dst (reveal ecell)._1 (reveal ecell)._2
                  (macc em' (reveal ecell)._1 (reveal ecell)._2)
      as
        gpu_matrix_pts_to_cell dst row (col + !k)
                  (macc em' row (col + !k));

      with s. assert local |-> s;
      let v = Pulse.Lib.Array.op_Array_Access local !k #_ #s;
      gpu_matrix_write_cell dst row (col +^ !k) v;

      let em'' : ematrix et rows cols = mupd em' row (col +^ !k) v;

      rewrite
        gpu_matrix_pts_to_cell dst row (col + !k) v
      as
        gpu_matrix_pts_to_cell dst (reveal ecell)._1 (reveal ecell)._2
                  (macc em'' (reveal ecell)._1 (reveal ecell)._2);

      forevery_ext
        #(ij : (natlt rows & natlt cols){in_chunk (chunk et) rows cols nthr tid ij /\ ij =!= ecell})
        (fun ij ->
          gpu_matrix_pts_to_cell dst ij._1 ij._2 (macc em' ij._1 ij._2))
        (fun ij ->
          gpu_matrix_pts_to_cell dst ij._1 ij._2 (macc em'' ij._1 ij._2));

      forevery_insert
        #(natlt rows & natlt cols)
        #(fun ij -> in_chunk (chunk et) rows cols nthr tid ij /\ ij =!= ecell)
        _ ecell;
      forevery_refine_ext
        #(natlt rows & natlt cols)
        #(fun ij -> (in_chunk (chunk et) rows cols nthr tid ij /\ ij =!= ecell) \/ reveal ecell == ij)
        (fun ij -> in_chunk (chunk et) rows cols nthr tid ij)
        (fun ij -> gpu_matrix_pts_to_cell dst ij._1 ij._2 (macc em'' ij._1 ij._2));

      fold own_strided_chunks dst em'' nthr tid;

      k := !k +^ 1sz;
    };

    let vi = !i;
    let vgit = GR.read git;
    assert pure (SZ.fits (nthr * chunk et));
    assert pure (SZ.fits (vi + nthr * chunk et));
    i := vi +^ nthr *^ chunk et;
    GR.write git (vgit  + 1);

    assert pure (SZ.v vi == vgit * nthr * chunk et);
    add_helper vi vgit nthr (chunk et); // sigh
    assert pure (SZ.v !i == GR.read git * nthr * chunk et);

    with em'. assert own_strided_chunks dst em' nthr tid;
    assume pure (em' == em_fade edst esrc nthr (vgit + 1));
    rewrite each em' as em_fade edst esrc nthr (vgit + 1) by nop_tactic ();
    (* ^ FIXME: Without using a tactic, we get a goal where em_fade is
         unfolded, which then fails... why does it unfold!? *)
    ()
  };

  assert pure (Kuiper.EMatrix.equal esrc (em_fade edst esrc nthr (GR.read git)));
  rewrite own_strided_chunks dst (em_fade edst esrc nthr (GR.read git)) nthr tid
       as own_strided_chunks dst esrc nthr tid;

  drop_ (git |-> _);

  ()
}
#pop-options
