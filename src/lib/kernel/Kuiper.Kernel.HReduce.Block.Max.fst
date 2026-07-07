module Kuiper.Kernel.HReduce.Block.Max

friend Kuiper.Kernel.HReduce.Max

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Tensor
open Kuiper.Chest1.Helpers
open Kuiper.Bijection { ( =~ ), bij_sym }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }
// Re-open after Kuiper.Tensor so the seq-level `@!`/`seq![..]`/`@+` notations
// shadow the shape-indexing `@!` pulled in via Kuiper.Shape.
open Kuiper.Seq.Common
open Kuiper.Math.OnlineSoftmax { seq_max, seq_max_cons_lem }
open Kuiper.Kernel.HReduce.Max

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier

(* ── Ported MAX seq lemmas ─────────────────────────────────────────────────
   Local copies of the [private] helpers in [Kuiper.Kernel.HReduce.Max] that
   are needed in this module's own proofs (their SMTPats / facts are not visible
   across the [friend] boundary). They are direct ports from the 1D Max file. *)

(* A strided bucket is non-empty as soon as its offset is in range. Stated as an
   SMTPat so the well-typedness of [seq_max (seq_stride ...)] is automatic. *)
let stride_nonempty (#a:Type) (s : seq a) (stride : pos) (off : natlt stride)
  : Lemma (requires off < Seq.length s)
          (ensures seq_stride_length s stride off > 0 /\
                   Seq.length (seq_stride s stride off) == seq_stride_length s stride off /\
                   Seq.length (seq_stride s stride off) > 0)
          [SMTPat (seq_stride s stride off)]
  = let n = Seq.length s in
    FStar.Math.Lemmas.lemma_div_le stride (n - off + stride - 1) stride;
    FStar.Math.Lemmas.cancel_mul_div 1 stride

(* One step of a left-to-right running max over a non-empty prefix: extending
   [seq_take k s] by one element [s @! k] maxes that element in. *)
let seq_max_take_step (s : seq real) (k : nat { 1 <= k /\ k < len s })
  : Lemma (ensures seq_max (seq_take (k + 1) s) == rmax (seq_max (seq_take k s)) (s @! k))
  = let pre  = seq_take k s in
    let pre1 = seq_take (k + 1) s in
    assert (Seq.equal pre1 (pre @+ seq![s @! k]));
    Seq.lemma_eq_elim pre1 (pre @+ seq![s @! k]);
    seq_max_cons_lem pre (s @! k);             // seq_max pre1 == rmax (s@!k) (seq_max pre)
    lem_rmax_comm (s @! k) (seq_max pre)

(* Extracted into a clean (quantifier-free) context: the trivial step
   [min (tid + 1) nth == tid + 1] when [tid < nth] is fragile when proved
   inline under the ambient row-approximation quantifiers. *)
let min_tid_pow2_step (tid nth k : nat)
  : Lemma (requires tid < nth /\ pow2 k == 1)
          (ensures min (tid + pow2 k) nth == tid + 1)
  = ()

(* Wrapper around [tensor_read] that takes the row/col already refined as
   [szlt rows]/[szlt cols], so the [conc] index is well-typed from the
   parameter types. *)
inline_for_extraction noextract
fn read_at
  (#et:Type0) {| scalar et |}
  (rows : szp)
  (cols : szp)
  (#lin : layout2 rows cols) {| ctlayout lin |}
  (x : array2 et lin)
  (row : szlt rows)
  (col : szlt cols)
  (#sx : chest2 et rows cols)
  (#f : perm)
  preserves
    x |-> Frac f sx
  returns
    res : et
  ensures
    pure (res == acc2 sx (SZ.v row) (SZ.v col))
{
  tensor_read x (cidx2 row col)
}

(* Drop a per-element [pure] clause from a [forevery] predicate. *)
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

(* Per-thread input-side max reduction: like [max_stride_map] but reads from a
   single row of a 2-D tensor. Initializes its accumulator with its first strided
   element (index [off]) so the running max is over a non-empty prefix. *)
(* ── Strided-bucket arithmetic ─────────────────────────────────────────────
   [max_stride_map_2d] below runs in a context carrying the ambient quantified
   hypothesis [forall j. acc2 sx row j %~ vr_row @! j], which makes inline stride
   arithmetic pathologically slow.  The pure-nat / generic-seq lemmas it needs
   ([stride_step_arith], [stride_idx_in_bounds], [stride_bucket_index],
   [max_stride_post_arith]) are shared from [Kuiper.Kernel.HReduce.Max]. *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn max_stride_map_2d
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp)
  (cols : szp)
  (#lin : layout2 rows cols) {| ctlayout lin |}
  (x : array2 et lin)
  (row : szlt rows)
  (stride : szp)
  (off : szlt stride { SZ.v off < SZ.v cols })
  (#sx : chest2 et rows cols)
  (vr_row : erased (lseq real cols))
  (#f : perm)
  preserves
    gpu ** x |-> Frac f sx **
    pure (forall (j:nat). j < SZ.v cols ==> acc2 sx (SZ.v row) j %~ (vr_row @! j)) **
    pure (SZ.fits (SZ.v cols + stride))
  returns
    res : et
  ensures
    pure (res %~ seq_max (seq_stride (lseq_map pre_map_r vr_row) stride off))
{
  (* off < cols, so the strided bucket at `off` is non-empty. *)
  stride_nonempty (lseq_map pre_map_r vr_row) stride off;

  (* Initialize the accumulator with the first strided element (column off). *)
  let off_raw : sz = off;
  assert pure (SZ.v off_raw < SZ.v cols);
  let off_c : szlt cols = off_raw;
  let v0 = read_at rows cols x row off_c;
  let acc0 = pre_map v0;
  (**)assert (pure (v0 == acc2 sx (SZ.v row) (SZ.v off_c)));
  (**)assert (pure (v0 %~ (vr_row @! SZ.v off_c)));
  (**)assert (pure (acc0 %~ (lseq_map pre_map_r vr_row @! SZ.v off)));
  (**)assert (pure (seq_stride (lseq_map pre_map_r vr_row) stride off @! 0 == (lseq_map pre_map_r vr_row) @! (off + 0 * stride)));
  (**)assert (pure (acc0 %~ (seq_stride (lseq_map pre_map_r vr_row) stride off @! 0)));
  (**)seq_max_singleton (seq_stride (lseq_map pre_map_r vr_row) stride off @! 0);
  (**)assert (pure (Seq.equal (seq_take 1 (seq_stride (lseq_map pre_map_r vr_row) stride off))
  (**)                        (seq![seq_stride (lseq_map pre_map_r vr_row) stride off @! 0])));
  (**)assert (pure (acc0 %~ seq_max (seq_take 1 (seq_stride (lseq_map pre_map_r vr_row) stride off))));

  let mut acc : et = acc0;
  let mut idx : sz = off +^ stride;
  let gidx = galloc #nat 1;

  while (!idx <^ cols)
    invariant
      live acc ** live gidx **
      live idx **
      pure (SZ.v !idx == gread gidx * stride + off /\
            gread gidx >= 1 /\
            off <= SZ.v !idx /\ SZ.v !idx < cols + stride /\
            gread gidx <= seq_stride_length (lseq_map pre_map_r vr_row) stride off /\
            !acc %~ seq_max (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr_row) stride off))) **
      emp
    decreases (cols + stride - !idx)
  {
    stride_idx_in_bounds (SZ.v cols) (SZ.v off) (SZ.v stride) (gread gidx) (SZ.v !idx);
    assert pure (gread gidx < seq_stride_length vr_row stride off);

    let idx_raw : sz = !idx;
    assert pure (SZ.v idx_raw < SZ.v cols);
    let idx_v : szlt cols = idx_raw;
    let v = read_at rows cols x row idx_v;
    let v' = pre_map v;
    (**)assert (pure (v == acc2 sx (SZ.v row) (SZ.v idx_v)));
    (**)assert (pure (v %~ (vr_row @! SZ.v idx_v)));
    (**)assert (pure (v' %~ (lseq_map pre_map_r vr_row @! SZ.v idx_v)));

    assert pure (!acc %~ seq_max (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr_row) stride off)));
    stride_bucket_index (lseq_map pre_map_r vr_row) (SZ.v stride) (SZ.v off) (gread gidx) (SZ.v !idx);
    assert pure (seq_stride (lseq_map pre_map_r vr_row) stride off @! gread gidx == (lseq_map pre_map_r vr_row) @! (SZ.v !idx));

    (* seq_take (k+1) maxes in the bucket's k-th element; combine with fmax. *)
    (**)seq_max_take_step (seq_stride (lseq_map pre_map_r vr_row) stride off) (gread gidx);

    let vgidx = gread gidx;
    assert (pure (SZ.v !idx == vgidx * stride + off));
    stride_step_arith (SZ.v !idx) vgidx (SZ.v stride) (SZ.v off);
    assert (pure (SZ.v !idx + stride == (vgidx + 1) * stride + off));

    Math.Lemmas.add_div_mod_1 (SZ.v !idx) stride;

    acc := !acc `fmax` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);

    assert pure (SZ.v !idx == gread gidx * stride + off);
    ()
  };

  assert pure (SZ.v !idx == gread gidx * stride + off);
  max_stride_post_arith (lseq_map pre_map_r vr_row) (SZ.v off) (SZ.v stride) (gread gidx) (SZ.v !idx);
  assert pure (gread gidx == seq_stride_length (lseq_map pre_map_r vr_row) stride off);
  assert pure (seq_take (seq_stride_length (lseq_map pre_map_r vr_row) stride off)
                       (seq_stride (lseq_map pre_map_r vr_row) stride off)
              == seq_stride (lseq_map pre_map_r vr_row) stride off);
  assert pure (!acc %~ seq_max (seq_stride (lseq_map pre_map_r vr_row) stride off));

  drop_ (gidx |-> _);
  !acc
}
#pop-options

(* ── Per-thread predicates for the per-block kernel ────────────────────── *)

unfold
let kpre_block
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols)
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    if_ (op_Equality #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)) **
    exists* (v : et). tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ()) v

unfold
let kpost_block
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols)
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    if_ (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid)))
    )

(* ── Per-thread kernel function ────────────────────────────────────────── *)

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn kf_block
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : erased (chest1 et rows))
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt rows)
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid) **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1)
                       (vr_partial_max pre_map_r (ematrix_row vr (SZ.v bid)) nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid) **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1)
                       (vr_partial_max pre_map_r (ematrix_row vr (SZ.v bid)) nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  unfold kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);

  let (gsa, _) = shmem;
  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  (* Row of vr at bid, as an lseq real cols. *)
  let vr_row : erased (lseq real cols) = hide (ematrix_row (reveal vr) (SZ.v bid));
  let vr_s : erased (lseq real nth) = vr_partial_max pre_map_r vr_row nth;

  (* Bridge from (sx %~ vr) to row-level approximation. *)
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 (vr_row @! j) == acc2 (reveal vr) (SZ.v bid) j);
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 acc2 sx (SZ.v bid) j %~ (vr_row @! j));

  (* Compute partial max over stride and write to shmem. *)
  let psum : et = max_stride_map_2d pre_map pre_map_r rows cols x bid nth tid vr_row;
  tensor_write_cell sa (tid, ()) psum;

  (* Set up tree reduction state. *)
  let mut n : szlt 32 = 0sz;

  let psum_chest : chest1 et 1 = mk1 #et #1 (fun _ -> psum);
  slice_singleton sa (SZ.v tid) psum psum_chest;

  (* psum %~ vr_s @! tid == seq_max (single-element slice [tid, tid+1)) *)
  (**)stride_nonempty (lseq_map pre_map_r vr_row) nth tid;
  (**)assert (pure (psum %~ (reveal vr_s @! SZ.v tid)));
  (**)seq_max_singleton (reveal vr_s @! SZ.v tid);
  (**)assert (pure (Seq.equal (Seq.slice (reveal vr_s) tid (tid + 1)) (seq![reveal vr_s @! SZ.v tid])));
  (**)assert (pure (acc1 psum_chest 0 %~ seq_max (Seq.slice (reveal vr_s) tid (tid + 1))));
  (**)fold (array1_pts_to_slice_max sa tid (tid + 1) vr_s);
  (**)assert (pure (pow2 (SZ.v !n) == 1));
  (**)min_tid_pow2_step (SZ.v tid) (SZ.v nth) (SZ.v !n);
  (**)rewrite (array1_pts_to_slice_max sa tid (tid + 1) vr_s)
  (**)     as (array1_pts_to_slice_max sa tid (min (tid + pow2 !n) nth) vr_s);
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_max sa tid (min (tid + pow2 !n) nth) vr_s);

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_max sa tid (min (tid + pow2 !n) nth) vr_s) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration nth sa vr_s tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_max sa tid (min (tid + pow2 it) nth) vr_s))
  as
    (if_ (op_Equality #nat tid 0) (array1_pts_to_slice_max sa 0 nth vr_s));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (array1_pts_to_slice_max sa 0 nth vr_s);
    if_elim_true' (op_Equality #nat tid 0) (Cell output (((SZ.v bid <: natlt rows), ()) <: abs (rows @| INil)) |-> (acc1 sout (SZ.v bid)));
    unfold array1_pts_to_slice_max sa 0 nth vr_s;
    (**)strided_max_is_max pre_map_r vr_row nth;
    (**)assert (pure (Seq.equal (Seq.slice (reveal vr_s) 0 nth) (reveal vr_s)));

    let res = array1_read_from_slice sa 0sz;
    tensor_write_cell output (bid, ()) res;

    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let css : erased (chest1 et nth) = hide (mk1 #et #nth (fun (k:natlt nth) -> acc1 ss k));
    forevery_refine_ext' #nat #(fun (k:nat) -> 0 <= k /\ k < nth) (fun (k:nat) -> k < nth) _;
    forevery_ext
      (fun (k:natlt nth) -> tensor_pts_to_cell sa ((k <: natlt nth), ()) (acc1 ss (k - 0)))
      (fun (k:natlt nth) -> tensor_pts_to_cell sa (abs_bij.gg k) (acc (reveal css) (abs_bij.gg k)));
    forevery_iso_back (abs_bij #nth)
      (fun (i : abs (nth @| INil)) -> tensor_pts_to_cell sa i (acc (reveal css) i));
    tensor_implode sa #1.0R #(reveal css);
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    if_intro_true' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output (((SZ.v bid <: natlt rows), ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr (SZ.v bid))))
    );
    fold kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);
  } else {
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_max sa 0 nth vr_s);
    if_elim_false' (op_Equality #nat tid 0) (Cell output (((SZ.v bid <: natlt rows), ()) <: abs (rows @| INil)) |-> (acc1 sout (SZ.v bid)));
    if_intro_false' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output (((SZ.v bid <: natlt rows), ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr (SZ.v bid))))
    );
    fold kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem (SZ.v bid) (SZ.v tid);
    ()
  };
}
#pop-options

(* ── Block-level setup/teardown ────────────────────────────────────────── *)

ghost
fn block_setup_block
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : erased (chest1 et rows))
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid))
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
  tensor_share_n x nth;

  (* tid 0 gets the output cell *)
  forevery_if_intro #(natlt nth) 0 (fun _ -> Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid));
  forevery_ext
    (fun tid -> if_ (op_Equality #(natlt nth) tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)))
    (fun tid -> if_ (op_Equality #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)));

  forevery_zip (fun _ -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx) _;

  (* View shmem array as a tensor and explode it into per-cell ownership. *)
  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
    (fun tid -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
                if_ (op_Equality #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
       if_ (op_Equality #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid))) **
      Cell (from_array (l1_forward nth) gsa) (abs_bij.gg (tid <: natlt nth))
        |-> (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
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
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : erased (chest1 et rows))
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth). kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     exists* (v : et).
       Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
       pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid))))
{
  forevery_unzip _ _;

  tensor_gather_n x nth;

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et).
          Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
          pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid)))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et).
          Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
          pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid)))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid)))
  );

  tensor_concr (from_array (l1_forward nth) shmem._1);
  rewrite each core (from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

(* ── Outer setup/teardown: share x across blocks, explode output ─────── *)

ghost
fn setup_block_outer
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : erased (chest1 et rows))
  ()
  norewrite
  requires
    x |-> sx ** output |-> sout
  ensures
    (forall+ (bid : natlt rows).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_pts_to_ref output;
  tensor_share_n x (SZ.v rows);
  tensor_explode output;
  forevery_iso abs_bij _;

  forevery_ext
    (fun (bid : natlt rows) -> Cell output (abs_bij.gg (bid <: natlt rows)) |-> acc sout (abs_bij.gg (bid <: natlt rows)))
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid));

  forevery_zip #(natlt rows)
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid));
  ()
}

#push-options "--z3rlimit 100 --fuel 4 --ifuel 4"
ghost
fn teardown_block_outer
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : erased (chest1 et rows))
  ()
  norewrite
  requires
    (forall+ (bid : natlt rows).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       exists* (v : et).
         Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
         pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid)))) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    exists* (sout' : chest1 et rows).
      x |-> sx ** output |-> sout' **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ seq_max (lseq_map pre_map_r (ematrix_row vr r)))
{
  forevery_unzip
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt rows) ->
       exists* (v : et).
         Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
         pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid))));

  tensor_gather_n x (SZ.v rows);

  (* Skolemize the existential: get a function bid -> et naming each cell value *)
  let f =
    forevery_exists
      (fun (bid : natlt rows) (v : et) ->
         Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
         pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid))));

  (* Build a concrete chest carrying the cell values. *)
  let sout' : erased (chest1 et rows) =
    hide (mk1 #et #(SZ.v rows) (fun (bid : natlt rows) -> f bid));

  (* Extract the per-row pure approximation fact across all bids. *)
  forevery_extract_pure
    (fun (bid : natlt rows) ->
       Cell output ((bid, ()) <: abs (rows @| INil)) |-> f bid **
       pure (f bid %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid))))
    (fun (bid : natlt rows) ->
       (acc1 (reveal sout') bid) %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid)))
    fn _ {};

  (* Drop the per-cell pure now that we extracted the global fact. *)
  forevery_drop_pure
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (rows @| INil)) |-> f bid)
    (fun (bid : natlt rows) ->
       f bid %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid)));

  forevery_ext
    (fun (bid : natlt rows) ->
       Cell output ((bid, ()) <: abs (rows @| INil)) |-> f bid)
    (fun (bid : natlt rows) ->
       Cell output (abs_bij.gg (bid <: natlt rows)) |-> acc (reveal sout') (abs_bij.gg (bid <: natlt rows)));

  forevery_iso_back (abs_bij #rows)
    (fun (i : abs (rows @| INil)) -> Cell output i |-> acc (reveal sout') i);

  tensor_implode output;
  ()
}
#pop-options

(* ── Kernel descriptor ─────────────────────────────────────────────────── *)

inline_for_extraction noextract
let kdesc_block
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : erased (chest1 et rows))
  : kernel_desc
      (x |-> sx ** output |-> sout)
      (exists* (sout' : chest1 et rows).
         x |-> sx ** output |-> sout' **
         pure (forall (r : nat). r < SZ.v rows ==>
               (acc1 sout' r) %~ seq_max (lseq_map pre_map_r (ematrix_row vr r))))
  = {
    nblk = rows;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (from_array _ shmem._1)
                          (vr_partial_max pre_map_r (ematrix_row vr bid) nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth)
                          (from_array _ shmem._1)
                          (vr_partial_max pre_map_r (ematrix_row vr bid) nth)));

    f = kf_block pre_map pre_map_r rows cols nth x output sx vr sout;

    block_pre  = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      Cell (output <: array1 et lout) ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid));
    block_post = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      exists* (v : et).
        Cell (output <: array1 et lout) ((bid, ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ seq_max (lseq_map pre_map_r (ematrix_row vr bid))));

    setup    = setup_block_outer    pre_map pre_map_r rows cols nth x output sx vr sout;
    teardown = teardown_block_outer pre_map pre_map_r rows cols nth x output sx vr sout;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup_block    pre_map pre_map_r rows cols nth x output sx vr sout;
    block_teardown = block_teardown_block pre_map pre_map_r rows cols nth x output sx vr sout;

    kpre  = kpre_block  pre_map pre_map_r rows cols nth x output sx vr sout;
    kpost = kpost_block pre_map pre_map_r rows cols nth x output sx vr sout;
    frame = pure (SZ.fits (tlayout_ulen lout));

    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

(* ── Entry point ──────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_block_max
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (#sx   : chest2 et   rows cols)
  (vr    : chest2 real rows cols)
  (#sout : erased (chest1 et rows))
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (sx %~ vr)
  ensures
    exists* (sout' : chest1 et rows).
      on gpu_loc (output |-> sout') **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ seq_max (Kuiper.Seq.Common.lseq_map pre_map_r (ematrix_row vr r)))
{
  launch_sync (kdesc_block pre_map pre_map_r rows cols nth x output sx vr sout);
}
