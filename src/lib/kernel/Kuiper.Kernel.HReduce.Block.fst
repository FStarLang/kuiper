module Kuiper.Kernel.HReduce.Block

friend Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }
open Kuiper.Kernel.HReduce
open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2

(* Per-element step lemma for the strided reduction in [sum_stride_map_2d]:
   appending the [k]-th strided element to the running sum of the first [k]
   elements yields the running sum of the first [k+1] elements. Analogous to
   [rsum_seq_take_next_] used by the legacy [Kuiper.Kernel.HReduce.sum_stride_map]. *)
private let rsum_seq_stride_step
  (rs : seq real)
  (stride : pos)
  (off : nat{off < stride})
  (k : nat)
  : Lemma (requires k < seq_stride_length rs stride off /\
                    k * stride + off < Seq.length rs)
          (ensures
            rsum (seq_take k (seq_stride rs stride off)) +. (rs @! (k * stride + off)) ==
            rsum (seq_take (k + 1) (seq_stride rs stride off)))
  = let ss = seq_stride rs stride off in
    let a = seq_take k ss in
    let single = Seq.slice ss k (k+1) in
    let v = rs @! (off + k * stride) in
    Kuiper.Seq.Common.lem_append_slice ss 0 k (k+1);
    assert (Seq.equal (seq_take (k+1) ss) (Seq.append a single));
    assert (Seq.length single == 1);
    assert (Seq.index single 0 == Seq.index ss k);
    assert (Seq.index ss k == v);
    Kuiper.Seq.Common.lem_one_elem single v;
    Kuiper.Approximates.rsum_append a single

(* Per-thread input-side reduction: like sum_stride_map but reads from a
   single row of an Array2. *)
(* Wrapper around [Array2.read] that takes the column index already
   refined as [szlt cols], so the [cit_fits] precondition is discharged
   by the parameter type. (Inlining the read in [sum_stride_map_2d]'s
   body makes Pulse fail to discharge the [cit_fits] obligation in this
   module's SMT context — moving the call into a helper sidesteps the
   issue.) *)
inline_for_extraction noextract
fn read_at
  (#et:Type0) {| scalar et |}
  (rows : szp)
  (cols : szp)
  (#lin : Array2.layout (SZ.v rows) (SZ.v cols)) {| ctlayout lin |}
  (x : Array2.t et lin)
  (row : szlt rows)
  (col : szlt cols)
  (#sx : ematrix et (SZ.v rows) (SZ.v cols))
  (#f : perm)
  preserves
    x |-> Frac f sx
  returns
    res : et
  ensures
    pure (res == macc sx (SZ.v row) (SZ.v col))
{
  let r : sz = row;
  let c : sz = col;
  let ij : Array2.raw_cit = (r, c);
  assert pure (Array2.cit_fits (SZ.v rows) (SZ.v cols) ij);
  Array2.read x ij
}

(* Drop a per-element [pure] clause from a [forevery] predicate. The
   global fact [forall x. q x] should typically be extracted beforehand
   via [forevery_extract_pure]; this helper just discards the per-cell
   [pure (q x)] clause so a subsequent [forevery_ext] only needs to
   match the residual slprop. *)
ghost
fn forevery_drop_pure
  (#a:Type0)
  (p : a -> slprop)
  (q : a -> prop)
  requires
    forall+ (x:a). p x ** pure (q x)
  ensures
    forall+ (x:a). p x
{
  forevery_map
    (fun (x:a) -> p x ** pure (q x))
    p
    fn x { drop_ (pure (q x)) }
}

// If k <= (n - off + stride - 1) / stride, k * stride + off >= n, and off < stride,
// then k == (n - off + stride - 1) / stride.
private let stride_length_exact (k n stride off : nat)
  : Lemma
      (requires
        stride > 0 /\ off < stride /\
        k <= (n - off + stride - 1) / stride /\
        k * stride + off >= n)
      (ensures k == (n - off + stride - 1) / stride)
  = Math.Lemmas.lemma_div_le (k * stride) (n - off + stride - 1) stride;
    Math.Lemmas.cancel_mul_div k stride

#push-options "--fuel 4 --ifuel 8 --z3rlimit 200"
inline_for_extraction noextract
fn sum_stride_map_2d
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp)
  (cols : szp)
  (#lin : Array2.layout (SZ.v rows) (SZ.v cols)) {| ctlayout lin |}
  (x : Array2.t et lin)
  (row : szlt rows)
  (stride : szp)
  (off : szlt stride)
  (#sx : ematrix et (SZ.v rows) (SZ.v cols))
  (vr_row : erased (lseq real (SZ.v cols)))
  (#f : perm)
  preserves
    gpu ** x |-> Frac f sx **
    pure (forall (j:nat). j < SZ.v cols ==> macc sx (SZ.v row) j %~ (vr_row @! j)) **
    pure (SZ.fits (SZ.v cols + stride))
  returns
    res : et
  ensures
    pure (res %~ rsum (seq_stride (lseq_map pre_map_r vr_row) stride off))
{
  let mut acc : et = zero;
  let mut idx : sz = off;
  let gidx = galloc #nat 0;

  while (!idx <^ cols)
    invariant
      live acc ** live gidx ** live idx **
      pure (gread gidx <= seq_stride_length (lseq_map pre_map_r vr_row) stride off /\
            !idx < cols + stride /\
            !acc %~ rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr_row) stride off)) /\
            SZ.v !idx == gread gidx * stride + off
      ) **
      emp
    decreases (cols + stride - !idx)
  {
    assert pure (gread gidx < seq_stride_length vr_row stride off);

    let idx_raw : sz = !idx;
    assert pure (SZ.v idx_raw < SZ.v cols);
    let idx_v : szlt cols = idx_raw;
    let v = read_at rows cols x row idx_v;
    let v' = pre_map v;
    (**)assert (pure (v == macc sx (SZ.v row) (SZ.v idx_v)));
    (**)assert (pure (v %~ (vr_row @! SZ.v idx_v)));
    (**)assert (pure (v' %~ (lseq_map pre_map_r vr_row @! SZ.v idx_v)));

    a_add !acc v'
      (rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr_row) stride off)))
      ((lseq_map pre_map_r vr_row) @! SZ.v idx_v);
    rsum_seq_stride_step (lseq_map pre_map_r vr_row) stride off (gread gidx);

    let vgidx = gread gidx;
    Math.Lemmas.distributivity_add_left vgidx 1 stride;

    acc := !acc `add` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);
    ()
  };

  stride_length_exact (gread gidx) (Seq.length (lseq_map pre_map_r vr_row)) (SZ.v stride) (SZ.v off);
  assert pure (gread gidx == seq_stride_length (lseq_map pre_map_r vr_row) stride off);
  assert pure (seq_take (seq_stride_length (lseq_map pre_map_r vr_row) stride off)
                       (seq_stride (lseq_map pre_map_r vr_row) stride off)
              == seq_stride (lseq_map pre_map_r vr_row) stride off);
  drop_ (gidx |-> _);
  !acc
}
#pop-options

(* ── Per-thread predicates for the per-block kernel ────────────────────── *)

unfold
let kpre_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols))
  (#lout : Array1.layout (SZ.v rows))
  (x      : Array2.t et lin)
  (output : Array1.t et lout)
  (sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr   : ematrix real (SZ.v rows) (SZ.v cols))
  (sout : lseq et (SZ.v rows))
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt (SZ.v rows))
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    if_ (op_Equality #nat tid 0) (Cell output (bid <: natlt (SZ.v rows)) |-> (sout @! bid)) **
    exists* (v : et). Cell (Array1.from_array (l1_forward nth) shmem._1) tid |-> v

unfold
let kpost_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols))
  (#lout : Array1.layout (SZ.v rows))
  (x      : Array2.t et lin)
  (output : Array1.t et lout)
  (sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr   : ematrix real (SZ.v rows) (SZ.v cols))
  (sout : lseq et (SZ.v rows))
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt (SZ.v rows))
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    if_ (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output (bid <: natlt (SZ.v rows)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))
    )

(* ── Per-thread kernel function ────────────────────────────────────────── *)

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn kf_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : Array1.layout (SZ.v rows))             {| ctlayout lout |}
  (x      : Array2.t et lin)
  (output : Array1.t et lout)
  (sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr   : ematrix real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : erased (lseq et (SZ.v rows)))
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt rows)
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid) **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (Array1.from_array (l1_forward nth) shmem._1)
                       (vr_partial pre_map_r (ematrix_row vr (SZ.v bid)) nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid) **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (Array1.from_array (l1_forward nth) shmem._1)
                       (vr_partial pre_map_r (ematrix_row vr (SZ.v bid)) nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  unfold kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);

  let (gsa, _) = shmem;
  let sa = Array1.from_array (l1_forward nth) gsa;
  rewrite each Array1.from_array (l1_forward nth) gsa as sa;

  (* Row of vr at bid, as an lseq real cols. *)
  let vr_row : erased (lseq real (SZ.v cols)) = hide (ematrix_row (reveal vr) (SZ.v bid));
  let vr_s : erased (lseq real nth) = vr_partial pre_map_r vr_row nth;

  (* Bridge from (sx %~ vr) to row-level approximation. *)
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 (vr_row @! j) == macc (reveal vr) (SZ.v bid) j);
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 macc sx (SZ.v bid) j %~ (vr_row @! j));

  (* Compute partial sum over stride and write to shmem. *)
  let psum : et = sum_stride_map_2d pre_map pre_map_r rows cols x bid nth tid vr_row;
  Array1.write_cell sa tid psum;

  (* Set up tree reduction state. *)
  let mut n : szlt 32 = 0sz;

  forevery_singleton_intro'
    #(x:nat{tid <= x /\ x < tid + 1})
    (fun x -> Cell sa (x <: natlt nth) |-> (seq![psum] @! (x - tid)))
    tid;
  fold array1_pts_to_slice sa tid (tid+1) seq![psum];

  (**)fold (array1_pts_to_slice_sum sa tid (tid + 1) vr_s);
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) vr_s);

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) vr_s) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration nth sa vr_s tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 it) nth) vr_s))
  as
    (if_ (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s);
    if_elim_true' (op_Equality #nat tid 0) (Cell output (SZ.v bid <: natlt (SZ.v rows)) |-> (reveal sout @! SZ.v bid));
    unfold array1_pts_to_slice_sum sa 0 nth vr_s;
    (**)strided_sum_is_sum pre_map_r vr_row nth;

    let res = array1_read_from_slice sa 0sz;
    Array1.write_cell output bid res;

    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let bij : Kuiper.Bijection.bijection (k:nat{0 <= k /\ k < nth}) (Array1.ait nth) =
      Kuiper.Bijection.Mkbijection
        #(k:nat{0 <= k /\ k < nth})
        #(Array1.ait nth)
        (fun k -> k)
        (fun k -> k)
        ez ez;
    forevery_iso bij _;
    forevery_ext _ (fun (k : natlt nth) -> Cell sa k |-> (ss @! k));
    Array1.implode sa;
    rewrite each sa as Array1.from_array (l1_forward nth) shmem._1;
    if_intro_true' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output (SZ.v bid <: natlt (SZ.v rows)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr (SZ.v bid))))
    );
    fold kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);
  } else {
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_sum sa 0 nth vr_s);
    if_elim_false' (op_Equality #nat tid 0) (Cell output (SZ.v bid <: natlt (SZ.v rows)) |-> (reveal sout @! SZ.v bid));
    if_intro_false' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output (SZ.v bid <: natlt (SZ.v rows)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr (SZ.v bid))))
    );
    fold kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);
    ()
  };
}
#pop-options

(* ── Block-level setup/teardown ────────────────────────────────────────── *)

ghost
fn block_setup_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols))
  (#lout : Array1.layout (SZ.v rows))
  (x      : Array2.t et lin)
  (output : Array1.t et lout)
  (sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr   : ematrix real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : erased (lseq et (SZ.v rows)))
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt (SZ.v rows))
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     Cell output (bid <: natlt (SZ.v rows)) |-> (reveal sout @! bid))
  ensures
    (forall+ (i : natlt nth). kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  (* share input fractional permission across nth threads *)
  Array2.share_n x nth;

  (* tid 0 gets the output cell *)
  forevery_if_intro #(natlt nth) 0 (fun _ -> Cell output (bid <: natlt (SZ.v rows)) |-> (reveal sout @! bid));
  forevery_ext
    (fun tid -> if_ (op_Equality #(natlt nth) tid 0) (Cell output (bid <: natlt (SZ.v rows)) |-> (reveal sout @! bid)))
    (fun tid -> if_ (op_Equality #nat tid 0) (Cell output (bid <: natlt (SZ.v rows)) |-> (reveal sout @! bid)));

  forevery_zip (fun _ -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx) _;

  (* View shmem array as Array1. Explode it. *)
  Array1.raise' (l1_forward nth) gsa;
  Array1.explode (Array1.from_array (l1_forward nth) gsa);

  forevery_zip #(natlt nth)
    (fun tid -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
                if_ (op_Equality #nat tid 0) (Cell output (bid <: natlt (SZ.v rows)) |-> (reveal sout @! bid)))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
       if_ (op_Equality #nat tid 0) (Cell output (bid <: natlt (SZ.v rows)) |-> (reveal sout @! bid))) **
      Cell (Array1.from_array (l1_forward nth) gsa) tid |-> (Array1.from_seq (l1_forward nth) vgsa @! tid)
    )
    (fun (tid : natlt nth) -> kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };
  ()
}

ghost
fn block_teardown_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols))
  (#lout : Array1.layout (SZ.v rows))
  (x      : Array2.t et lin)
  (output : Array1.t et lout)
  (sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr   : ematrix real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : erased (lseq et (SZ.v rows)))
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt (SZ.v rows))
  ()
  norewrite
  requires
    (forall+ (i : natlt nth). kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     exists* (v : et).
       Cell output (bid <: natlt (SZ.v rows)) |-> v **
       pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))))
{
  forevery_unzip _ _;

  Array2.gather_n x nth;

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (Array1.from_array (l1_forward nth) shmem._1) **
        exists* (v : et).
          Cell output (bid <: natlt (SZ.v rows)) |-> v **
          pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (Array1.from_array (l1_forward nth) shmem._1) **
        exists* (v : et).
          Cell output (bid <: natlt (SZ.v rows)) |-> v **
          pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output (bid <: natlt (SZ.v rows)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))
  );

  Array1.lower (Array1.from_array (l1_forward nth) shmem._1);
  rewrite each Array1.core (Array1.from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

(* ── Outer setup/teardown: share x across blocks, explode output ─────── *)

ghost
fn setup_block_outer
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : Array1.layout (SZ.v rows))             {| ctlayout lout |}
  (x      : Array2.t et lin)
  (output : Array1.t et lout)
  (sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr   : ematrix real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : erased (lseq et (SZ.v rows)))
  ()
  norewrite
  requires
    x |-> sx ** output |-> sout
  ensures
    (forall+ (bid : natlt (SZ.v rows)).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       Cell output (bid <: natlt (SZ.v rows)) |-> (reveal sout @! bid)) **
    pure (SZ.fits (Array1.layout_size lout))
{
  Array1.pts_to_ref output;
  Array2.share_n x (SZ.v rows);
  Array1.explode output;

  forevery_zip #(natlt (SZ.v rows))
    (fun (_ : natlt (SZ.v rows)) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt (SZ.v rows)) -> Cell output bid |-> (reveal sout @! bid));
  ()
}

#push-options "--z3rlimit 100 --fuel 4 --ifuel 4"
ghost
fn teardown_block_outer
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : Array1.layout (SZ.v rows))             {| ctlayout lout |}
  (x      : Array2.t et lin)
  (output : Array1.t et lout)
  (sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr   : ematrix real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : erased (lseq et (SZ.v rows)))
  ()
  norewrite
  requires
    (forall+ (bid : natlt (SZ.v rows)).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       exists* (v : et).
         Cell output (bid <: natlt (SZ.v rows)) |-> v **
         pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))) **
    pure (SZ.fits (Array1.layout_size lout))
  ensures
    exists* (sout' : lseq et (SZ.v rows)).
      x |-> sx ** output |-> sout' **
      pure (forall (r : nat). r < SZ.v rows ==>
            (sout' @! r) %~ rsum (lseq_map pre_map_r (ematrix_row vr r)))
{
  forevery_unzip
    (fun (_ : natlt (SZ.v rows)) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt (SZ.v rows)) ->
       exists* (v : et).
         Cell output (bid <: natlt (SZ.v rows)) |-> v **
         pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))));

  Array2.gather_n x (SZ.v rows);

  (* Skolemize the existential: get a function bid -> et naming each cell value *)
  let f =
    forevery_exists
      (fun (bid : natlt (SZ.v rows)) (v : et) ->
         Cell output (bid <: natlt (SZ.v rows)) |-> v **
         pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))));

  (* Build a concrete lseq carrying the cell values. *)
  let sout' : erased (lseq et (SZ.v rows)) =
    hide (Seq.init_ghost (SZ.v rows) (fun (bid : natlt (SZ.v rows)) -> f bid));

  (* Extract the per-row pure approximation fact across all bids. *)
  forevery_extract_pure
    (fun (bid : natlt (SZ.v rows)) ->
       Cell output (bid <: natlt (SZ.v rows)) |-> f bid **
       pure (f bid %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))))
    (fun (bid : natlt (SZ.v rows)) ->
       (Seq.index (reveal sout') bid) %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))
    fn _ {};

  (* Drop the per-cell pure now that we extracted the global fact. *)
  forevery_drop_pure
    (fun (bid : natlt (SZ.v rows)) -> Cell output (bid <: natlt (SZ.v rows)) |-> f bid)
    (fun (bid : natlt (SZ.v rows)) ->
       f bid %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)));

  forevery_ext
    (fun (bid : natlt (SZ.v rows)) ->
       Cell output (bid <: natlt (SZ.v rows)) |-> f bid)
    (fun (bid : natlt (SZ.v rows)) ->
       Cell output (bid <: natlt (SZ.v rows)) |-> Seq.index (reveal sout') bid);

  Array1.implode output;
  ()
}
#pop-options

(* ── Kernel descriptor ─────────────────────────────────────────────────── *)

inline_for_extraction noextract
let kdesc_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : Array1.layout (SZ.v rows))             {| ctlayout lout |}
  (x      : Array2.t et lin  { Array2.is_global x      })
  (output : Array1.t et lout { Array1.is_global output })
  (sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr   : ematrix real (SZ.v rows) (SZ.v cols) { sx %~ vr })
  (sout : erased (lseq et (SZ.v rows)))
  : kernel_desc
      (x |-> sx ** output |-> sout)
      (exists* (sout' : lseq et (SZ.v rows)).
         x |-> sx ** output |-> sout' **
         pure (forall (r : nat). r < SZ.v rows ==>
               (sout' @! r) %~ rsum (lseq_map pre_map_r (ematrix_row vr r))))
  = {
    nblk = rows;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (Array1.from_array _ shmem._1)
                          (vr_partial pre_map_r (ematrix_row vr bid) nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth)
                          (Array1.from_array _ shmem._1)
                          (vr_partial pre_map_r (ematrix_row vr bid) nth)));

    f = kf_block pre_map pre_map_r rows cols nth x output sx vr sout;

    block_pre  = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      Cell (output <: Array1.t et lout) (bid <: natlt (SZ.v rows)) |-> (reveal sout @! bid));
    block_post = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      exists* (v : et).
        Cell (output <: Array1.t et lout) (bid <: natlt (SZ.v rows)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))));

    setup    = setup_block_outer    pre_map pre_map_r rows cols nth x output sx vr sout;
    teardown = teardown_block_outer pre_map pre_map_r rows cols nth x output sx vr sout;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup_block    pre_map pre_map_r rows cols nth x output sx vr sout;
    block_teardown = block_teardown_block pre_map pre_map_r rows cols nth x output sx vr sout;

    kpre  = kpre_block  pre_map pre_map_r rows cols nth x output sx vr sout;
    kpost = kpost_block pre_map pre_map_r rows cols nth x output sx vr sout;
    frame = pure (SZ.fits (Array1.layout_size lout));

    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

(* ── Entry point ──────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : Array2.layout (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : Array1.layout (SZ.v rows))             {| ctlayout lout |}
  (x      : Array2.t et lin  { Array2.is_global x      })
  (output : Array1.t et lout { Array1.is_global output })
  (#sx   : ematrix et   (SZ.v rows) (SZ.v cols))
  (vr    : ematrix real (SZ.v rows) (SZ.v cols))
  (#sout : erased (lseq et (SZ.v rows)))
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (sx %~ vr)
  ensures
    exists* (sout' : lseq et (SZ.v rows)).
      on gpu_loc (output |-> sout') **
      pure (forall (r : nat). r < SZ.v rows ==>
            (sout' @! r) %~ rsum (Kuiper.Seq.Common.lseq_map pre_map_r (ematrix_row vr r)))
{
  launch_sync (kdesc_block pre_map pre_map_r rows cols nth x output sx vr sout);
}
