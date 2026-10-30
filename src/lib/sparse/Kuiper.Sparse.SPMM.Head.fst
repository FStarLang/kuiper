module Kuiper.Sparse.SPMM.Head

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.Approx
open Kuiper.Sparse.LoadStore
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Sparse.SPMM.Barrier

open Kuiper.Tensor
open Kuiper.Array.Vectorized
open Pulse.Lib.Pledge
open Kuiper.Kernel.Base
open Kuiper.EMatrix
open Kuiper.Bijection { ( |~> ) }
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Array2.Strided

module Math = Kuiper.Sparse.Math
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module Compute = Kuiper.Sparse.SPMM.Compute

open Kuiper.Sparse.SPMM.Ownership

inline_for_extraction noextract
fn sparse_load_main
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  // matriz sparse gA
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#relems : erased (lseq real gA.nnz))
  (#rA : chest2 real p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p (reveal col_ind) (reveal row_off)))
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (reveal row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == reveal row_off @! (brow p bid |~> row_perm) + 1})
  (idx : sz { idx > 0 })
  (tid : szlt p.blockWidth)
  (#_ : squash(ri + idx * p.blockItemsK + p.blockItemsK <= re))
  norewrite
  preserves
    gpu **
    smatrix_approx' gA #fA (reveal elems) col_ind row_off relems rA **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures
    B.barrier_state ((idx + 1) * 2) **
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice elems
        (ri + idx * p.blockItemsK) (ri + idx * p.blockItemsK + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind
        (ri + idx * p.blockItemsK) (ri + idx * p.blockItemsK + p.blockItemsK))
{
  assert pure (1.0R /. p.blockWidth >. 0.0R);
  let off : (o:sz{SZ.v o == SZ.v ri + SZ.v idx * SZ.v p.blockItemsK})
    = ri +^ idx *^ p.blockItemsK;

  barrier_in_fold_main_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri re idx tid;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin (idx * 2) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout (idx * 2) tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid;

  barrier_out_unfold_main_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri re idx tid;

  let ri_ = hide (row_off @! (brow p bid |~> row_perm));

  assert pure (chunk et /? p.blockItemsK);
  Math.rounded_offset_aligned sz gA.elems (SZ.v ri_);
  Math.step_aligned gA.elems (SZ.v ri) p.blockItemsK (SZ.v idx);
  assert pure (aligned' 16 gA.elems off);
  load_array_vec elems_tile gA.elems off p.blockWidth tid;

  Math.rounded_offset_aligned et gA.col_ind (SZ.v ri_);
  Math.step_aligned gA.col_ind (SZ.v ri) p.blockItemsK (SZ.v idx);
  assert pure (aligned' 16 gA.col_ind off);
  load_array_vec col_ind_tile gA.col_ind off p.blockWidth tid;

  barrier_in_fold_main_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri re idx tid;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin (idx * 2 + 1) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout (idx * 2 + 1) tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid;

  barrier_out_unfold_main_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri re idx tid;

  barrier_step_lemma (SZ.v idx);
  ();

}

inline_for_extraction noextract
fn sparse_load_residue
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  // matriz sparse gA
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#relems : erased (lseq real gA.nnz))
  (#rA : chest2 real p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : szlt p.blockWidth)
  (idx residue : sz { residue_pred p.blockItemsK ri ri' re idx residue })
  norewrite
  preserves
    gpu **
    smatrix_approx' gA #fA elems col_ind row_off relems rA **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    barrier_in p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid (idx * 2) tid
  ensures
    B.barrier_state ((idx + 1) * 2) **
    pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice elems (re - residue) re) **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice col_ind (re - residue) re) **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK
{
  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin (idx * 2) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout (idx * 2) tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid;

  barrier_out_unfold_residue_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid idx residue;

  load2_array
    elems_tile col_ind_tile
    residue
    gA.elems gA.col_ind (re -^ residue)
    p.blockWidth tid;

  barrier_in_fold_residue_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid idx residue;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin (idx * 2 + 1) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout (idx * 2 + 1) tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid;

  barrier_phase_succ (SZ.v idx);
  barrier_out_unfold_residue_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid idx residue;
}

noextract
let barrier_count
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (#nnz : nat)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (bid : natlt (nblocks p))
: Ghost nat
  (requires
    valid_smatrix p.rows p.shared (cast_pos col_ind) (cast_pos row_off)
  )
  (ensures fun _ -> true)
=
  let ri = row_off @! (brow p bid |~> row_perm) in
  let re = row_off @! (brow p bid |~> row_perm) + 1 in
  let ri' = round2 (max (chunk et) (chunk sz)) ri in
  assert (re >= ri');
  ((re - ri') / p.blockItemsK + 1) * 2

let residue_index
  (block_items : pos)
  (ri ri' : nat)
  (re : nat { ri' <= re })
  (idx residue : nat)
  : Lemma
      (requires residue_pred block_items ri ri' re idx residue)
      (ensures idx == (re - ri') / block_items)
  = if re - ri' < block_items then
      FStar.Math.Lemmas.small_division_lemma_1 (re - ri') block_items

// Defs de algignment
// TODO mover a otro lado
noextract inline_for_extraction
let align_sz (size : szp) (x : sz) : sz =
  (x /^ size) *^ size

noextract inline_for_extraction
let align_offset
  (et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (off : sz)
: Pure sz
  (requires true)
  (ensures fun r -> SZ.v r == round2 (max (chunk et) (chunk sz)) off)
  (* chunk sz y chunk et son potencias de dos, así que alinear
     a la mayor es alinear a ambas. *)
= if chunk sz <^ chunk et
    then align_sz (chunk et) off
    else align_sz (chunk sz) off

open Kuiper.Seq.Common { (@+) }
open Kuiper.Sparse.Mask

let lslice
  (#a : Type0)
  (#n : nat)
  (s : lseq a n)
  (i k : nat { i + k <= n })
: lseq a k
= Seq.slice s i (i + k)

let lslice'
  (#a : Type0)
  (#n : nat)
  (s : lseq a n)
  (i j : natle n { i <= j })
: lseq a (j - i)
= Seq.slice s i j

[@@"opaque_to_smt"]
let seq_mask
  (#et : Type0) {| scalar et |}
  (k : nat)
  (#n : nat)
  (s : lseq et n)
: lseq et (k + n)
= Seq.create k zero @+ s
//

(* Padding is justified in the approximation relation, not by an exact
   floating-point zero-multiplication law. *)
let zero_prefix_approx
  (#et : Type0) {| scalar et, real_like et |}
  (k : nat) (#n : nat) (s : lseq et n) (r : lseq real n)
  : Lemma (requires s %~ r)
          (ensures (Seq.create k (zero #et) @+ s) %~ (Seq.create k 0.0R @+ r))
  = ()

let seq_mask_approx
  (#et : Type0) {| scalar et, real_like et |}
  (k : nat) (#n : nat) (s : lseq et n) (r : lseq real n)
  : Lemma (requires s %~ r)
          (ensures seq_mask k s %~ seq_mask k r)
  = reveal_opaque (`%seq_mask) (seq_mask k s);
    reveal_opaque (`%seq_mask) (seq_mask k r);
    zero_prefix_approx k s r

let seq_mask_prefix
  (#et : Type0) {| scalar et |}
  (k : nat)
  (#n : nat)
  (s : lseq et n)
  (m : nat { m <= n })
: Lemma
  (ensures
    Seq.equal
      (Seq.slice (seq_mask k s) 0 (k + m))
      (Seq.create k zero @+ Seq.slice s 0 m))
=
  let prefix = Seq.create k zero in
  reveal_opaque (`%seq_mask) (seq_mask k s);
  FStar.Seq.Properties.append_slices prefix s;
  Kuiper.Seq.Common.lem_append_slice (prefix @+ s) 0 k (k + m)

(* [seq_mask_prefix] stated through [lslice], with an explicitly equal stop.
   Keeping this conversion in a small pure context avoids asking SMT to
   unfold and normalize the same sequence expression in the full kernel
   proof state. *)
let seq_mask_lslice_prefix
  (#et : Type0) {| scalar et |}
  (k : nat)
  (#n : nat)
  (s : lseq et n)
  (masked : lseq et (k + n) {masked == seq_mask k s})
  (m : nat {m <= n})
  (stop : nat {stop <= k + n})
  (_ : squash (stop == k + m))
  : Lemma
      (Seq.equal
        (lslice masked 0 stop)
        (Seq.create k zero @+ lslice s 0 m))
  = seq_mask_prefix k s m;
    ()

let seq_mask_slice
  (#et : Type0) {| scalar et |}
  (k : nat)
  (#n : nat)
  (s : lseq et n)
  (i j : nat { k <= i /\ i <= j /\ j <= k + n })
: Lemma
  (ensures
    Seq.equal
      (Seq.slice (seq_mask k s) i j)
      (Seq.slice s (i - k) (j - k)))
=
  let prefix = Seq.create k zero in
  reveal_opaque (`%seq_mask) (seq_mask k s);
  FStar.Seq.Properties.append_slices prefix s

let seq_mask_source_slice
  (#et : Type0) {| scalar et |}
  (#n : nat)
  (elems : lseq et n)
  (ri' ri re : nat { ri' <= ri /\ ri <= re /\ re <= n })
  (row_elems_ : lseq et (re - ri) {
    row_elems_ == Seq.slice elems ri re })
  (row_elems : lseq et (re - ri') {
    row_elems == seq_mask (ri - ri') row_elems_ })
  (start stop : nat {
    ri - ri' <= start /\ start <= stop /\ stop <= re - ri' })
  : Lemma
      (Seq.equal
        (Seq.slice elems (ri' + start) (ri' + stop))
        (Seq.slice row_elems start stop))
  = seq_mask_slice (ri - ri') row_elems_ start stop;
    Seq.slice_slice elems ri re
      (start - (ri - ri')) (stop - (ri - ri'));
    ()

let cast_pos_slice
  (#n : nat)
  (s : lseq sz n)
  (i j : nat { i <= j /\ j <= n })
: Lemma
  (ensures
    Seq.equal
      (cast_pos #(j - i) (Seq.slice s i j))
      (Seq.slice (cast_pos s) i j))
=
  assert (
    Seq.equal
      (cast_pos #(j - i) (Seq.slice s i j))
      (Seq.slice (cast_pos s) i j)
  )

(* tcol y la fila de un thread caen bien debajo de 2^32 por size_req. *)
let tcol_fits
  (#et : Type0) {| d : scalar et, sized et, hvc : has_vec_cpy et, real_like et |}
  (p : parameters et)
  (bid : natlt (nblocks_ p))
  (tid : natlt p.blockWidth)
: Lemma (fits (tcol p bid tid + p.blockItemsX / p.blockWidth * p.blockWidth))
=
  FStar.Math.Lemmas.lemma_mult_lt_right (chunk et) tid p.blockWidth;
  assert (bcol p bid < p.cols);
  Kuiper.Divides.lemma_divides_le (p.blockWidth * chunk et) p.blockItemsX;
  assert (p.blockWidth * chunk et <= p.blockItemsX);
  FStar.Math.Lemmas.multiply_fractions p.blockItemsX p.blockWidth;
  assert (tcol p bid tid + p.blockItemsX / p.blockWidth * p.blockWidth < 30000);
  assert_norm (pow2 16 == 65536);
  FStar.SizeT.fits_at_least_16 (tcol p bid tid + p.blockItemsX / p.blockWidth * p.blockWidth)

(* The masked first tile: zero-padding followed by the in-row prefix is the
   [k]-prefix of the padded row. *)
let masked_tile_prefix
  (#a : Type0) (z : a) (s : Seq.seq a) (ri' ri re k : nat)
: Lemma
  (requires ri' <= ri /\ ri <= ri' + k /\ ri' + k <= re /\ re <= Seq.length s)
  (ensures
    Seq.equal
      (Seq.create (ri - ri') z @+ Seq.slice s ri (ri' + k))
      (Seq.slice (Seq.create (ri - ri') z @+ Seq.slice s ri re) 0 k))
= ()

(* Check the length of a padded row without the kernel's approximation facts. *)
let padded_row
  (#a : Type0) (z : a)
  (#n : nat) (s : lseq a n)
  (start first last : nat { start <= first /\ first <= last /\ last <= n })
  : GTot (lseq a (last - start))
  = Seq.create (first - start) z @+ Seq.slice s first last

let masked_row
  (#et : Type0) {| scalar et |}
  (start first last : nat { start <= first /\ first <= last })
  (s : lseq et (last - first))
  : GTot (lseq et (last - start))
  = seq_mask (first - start) s

(* Loading and masking the first tile needs bounds, but not column sortedness.
   Expanding sorted_slice here causes costly quantifier instantiations even
   for sequence-length and slice-bound queries. *)
#push-options "--z3rlimit 60 --using_facts_from '* -Kuiper.Sparse.Common.sorted_slice'"
inline_for_extraction noextract
fn kf_head
  (#et : Type0) {| d : scalar et, sized et, hvc : has_vec_cpy et, real_like et |}
  (p : parameters et)
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#lB : layout2 p.shared p.cols) {| ctlB : ctlayout lB, srmB : strided_row_major lB |}
  (gB : array2 et lB)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#relems : erased (lseq real gA.nnz))
  (rA : chest2 real p.rows p.shared)
  // matriz densa gb
  (#eB : chest2 et p.shared p.cols)
  (#rB : chest2 real p.shared p.cols { eB %~ rB })
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  // TODO creo tid : natlt p.blockWidth esta bien
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == tcol p bid tid })
  (#_ : squash (ri' + p.blockItemsK <= re))
  norewrite
  preserves gpu
  preserves smatrix_approx' gA #fA elems col_ind row_off relems rA
  requires  pure (aligned 16 gA.elems)
  requires  pure (aligned 16 gA.col_ind)
  preserves gB |-> Frac (fB /. allthreads p) eB
  requires  pure (aligned 16 (core gB))
  requires  pure (aligned_strided_row_major (chunk et) srmB)
  preserves
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    )
  preserves thread_id p.blockWidth tid
  requires  thread_live_chunks elems_tile p.blockWidth tid
  requires  thread_live_chunks col_ind_tile p.blockWidth tid
  requires  out |-> Seq.create (p.blockItemsX / p.blockWidth) d.zero
  requires  B.barrier_state 0
  ensures   exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s
  ensures   exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s
  ensures   B.barrier_state 2
  ensures   exists* vout.
    out |-> (vout <: seq et) **
    pure (
      Compute.tile_mm_result #_ #_ #_ #hvc #_
        (Seq.create (threadItemsX p) 0.0R <: lseq real (threadItemsX p))
        (Seq.create (ri - ri') zero @+ Seq.slice relems ri (ri' + p.blockItemsK))
        (lslice (cast_pos col_ind) ri' p.blockItemsK)
        rB n_idx p.blockWidth vout
    )
{
  row_prefix_bound p (SZ.v ri);
  assert pure (ri' <= ri /\ ri <= re /\ re <= gA.nnz);
  assert pure (ri <= ri' + p.blockItemsK /\ ri' + p.blockItemsK <= gA.nnz);

  Math.rounded_offset_aligned sz gA.elems ri;
  assert pure (aligned' 16 gA.elems ri');
  load_array_vec elems_tile gA.elems ri' p.blockWidth tid;

  Math.rounded_offset_aligned et gA.col_ind ri;
  assert pure (aligned' 16 gA.col_ind ri');
  load_array_vec col_ind_tile gA.col_ind ri' p.blockWidth tid;

  barrier_in_fold_mask_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin 0 tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout 0 tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid;

  barrier_out_unfold_mask_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid;

  mask_array elems_tile (ri -^ ri') zero p.blockWidth tid;

  barrier_in_fold_mask_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin 1 tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout 1 tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid;

  barrier_out_unfold_mask_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid;

  // TODO mejores nombres
  let elems' : erased (lseq et (re - ri')) = padded_row zero elems ri' ri re;
  let relems' : erased (lseq real (re - ri')) = padded_row 0.0R relems ri' ri re;
  zero_prefix_approx (ri - ri') #(re - ri) (Seq.slice elems ri re) (Seq.slice relems ri re);
  assert pure (elems' %~ relems');
  let col_ind' : erased (lseq sz (re - ri')) = Seq.slice col_ind ri' re;

  let out0 : erased (lseq et (p.blockItemsX / p.blockWidth)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;
  let rout0 : erased (lseq real (p.blockItemsX / p.blockWidth)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  Math.prod_cancel_divides p.blockItemsX (chunk et) (p.blockWidth);

  Compute.tile_mm_lemma0
    // #_ #_ #_ #_
    // #(p.blockItemsX / p.blockWidth)
    rout0
    rB n_idx p.blockWidth out0;


  Seq.lemma_empty (Seq.slice elems' 0 0);
  Seq.lemma_empty (Seq.slice col_ind' 0 0);

  assert pure (
    Compute.tile_mm_result
      rout0
      (Seq.slice relems' 0 0 <: lseq real 0)
      (Seq.slice (cast_pos col_ind') 0 0)
      rB
      n_idx p.blockWidth
      out0
  );

  masked_tile_prefix #et zero elems ri' ri re p.blockItemsK;
  masked_tile_prefix #real 0.0R relems ri' ri re p.blockItemsK;
  assert pure (
    Seq.equal
      (Seq.create (ri - ri') zero @+ Seq.slice elems ri (ri' + p.blockItemsK))
      (Seq.slice elems' 0 p.blockItemsK)
  );
  assert pure (
    Seq.equal
      (Seq.create (ri - ri') zero @+ Seq.slice relems ri (ri' + p.blockItemsK))
      (Seq.slice relems' 0 p.blockItemsK)
  );
  assert pure (
    Seq.equal
      (Seq.slice (cast_pos col_ind') 0 p.blockItemsK)
      (lslice (cast_pos col_ind) ri' p.blockItemsK)
  );

  tcol_fits p bid tid;

  Compute.tile_mm
    out
    rout0
    elems_tile col_ind_tile #(1.0R /. p.blockWidth)
    #(re - ri')
    elems' col_ind' relems'
    gB #_ #_ #rB
    n_idx p.blockWidth
    0 (SZ.v p.blockItemsK) p.blockItemsK;

  ();
}
#pop-options
