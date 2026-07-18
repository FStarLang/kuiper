module Kuiper.Kernel.GEMM.Naive1

(* Native batched matrix multiplication (GEMM) over rank-3 tensors:
   (batch, m, k) and (batch, k, n) to produce (batch, m, n).

   Like Naive2, but batched: a *single* kernel launch spawns
   [batch * m * n] independent blocks (one thread each, via
   [kernel_desc_m_1] -- intentionally silly and slow), each computing
   one output cell of one page with a full dot product.  The block index
   [gid] is decomposed in *page-minor* (batch-innermost) order,
     page = gid % batch
     rest = gid / batch
     r    = rest / n
     c    = rest % n
   and block [gid] computes C[page][r][c].  Page-minor order is chosen so
   that the batch=1 specialization extracts cleanly: [gid % 1] folds to 0
   and [gid / 1] folds to [gid], leaving no batch arithmetic in the rank-2
   entry point, with no runtime branch in the batched one. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor { tensor_pts_to_cell as pts_to_cell }
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Bijection
open Pulse.Lib.Trade
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module C = Kuiper.Matrix.Casts

(* Reshaping bridge between the 2-D nested index
   [abs (m @| n @| INil)] and the flat pair, mirroring Naive.abs_bij. *)
let abs_bij2 (#m #n : nat)
  : (abs (m @| n @| INil) =~ (natlt m & natlt n)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
  }

(* Reshaping bridge between the 3-D nested index and the flat triple.
   [abs (b @| m @| n @| INil) == natlt b & abs (m @| n @| INil)]
   definitionally, so we reuse [abs_bij2] for the inner two dims. *)
let abs_bij3 (#b #m #n : nat)
  : (abs (b @| m @| n @| INil) =~ (natlt b & (natlt m & natlt n))) =
  bij_prod (bij_self (natlt b)) (abs_bij2 #m #n)

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#b #m #n #k : nat)
  (#lA : layout3 b m k)
  (#lB : layout3 b k n)
  (#lC : layout3 b m n)
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest3 et b m k)
  (eB : chest3 et b k n)
  (eC : chest3 et b m n)
  (fA fB : perm)
  (gid : natlt (b * (m * n)))
  : slprop
  =
  gA |-> Frac (fA /. (b * (m * n))) eA **
  gB |-> Frac (fB /. (b * (m * n))) eB **
  pts_to_cell gC
    (gid % b, ((gid / b) / n, ((gid / b) % n, ())))
    (acc eC (gid % b, ((gid / b) / n, ((gid / b) % n, ()))))

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#b #m #n #k : nat)
  (#lA : layout3 b m k)
  (#lB : layout3 b k n)
  (#lC : layout3 b m n)
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest3 et b m k)
  (eB : chest3 et b k n)
  (eC : chest3 et b m n)
  (fA fB : perm)
  (gid : natlt (b * (m * n)))
  : slprop
  =
  gA |-> Frac (fA /. (b * (m * n))) eA **
  gB |-> Frac (fB /. (b * (m * n))) eB **
  pts_to_cell gC
    (gid % b, ((gid / b) / n, ((gid / b) % n, ())))
    (MS.gemm_single comb
      (slice_page eA (gid % b))
      (slice_page eB (gid % b))
      (slice_page eC (gid % b))
      ((gid / b) / n)
      ((gid / b) % n))

#push-options "--z3rlimit 40 --split_queries always"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#b #m #n #k : SZ.t)
  (#lA : layout3 b m k)
  (#lB : layout3 b k n)
  (#lC : layout3 b m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (#eA : chest3 et b m k)
  (#eB : chest3 et b k n)
  (#eC : chest3 et b m n)
  (#fA #fB : perm)
  (gid : szlt (b * (m * n)))
  ()
  norewrite
  requires
    gpu **
    kpre comb gA gB gC eA eB eC fA fB gid **
    block_id (b *^ (m *^ n)) gid
  ensures
    gpu **
    kpost comb gA gB gC eA eB eC fA fB gid **
    block_id (b *^ (m *^ n)) gid
{
  (* Page-minor (batch-innermost) decomposition.  For a statically-one
     batch this extracts to [page = 0], [rest = gid] with no modulus, so
     the rank-2 entry point contains no batch arithmetic; for a dynamic
     batch it is a plain modulus/division with no runtime branch. *)
  let page : szlt b = gid %^ b; assert (rewrites_to page (gid %^ b));
  let rest : szlt (m *^ n) = gid /^ b; assert (rewrites_to rest (gid /^ b));
  let trow : szlt m = rest /^ n; assert (rewrites_to trow (rest /^ n));
  let tcol : szlt n = rest %^ n; assert (rewrites_to tcol (rest %^ n));

  rewrite pts_to_cell gC
    ((SZ.v gid % SZ.v b <: natlt (SZ.v b)),
      ((SZ.v gid / SZ.v b / SZ.v n <: natlt (SZ.v m)),
        ((SZ.v gid / SZ.v b % SZ.v n <: natlt (SZ.v n)), ())))
    (acc eC
      ((SZ.v gid % SZ.v b <: natlt (SZ.v b)),
        ((SZ.v gid / SZ.v b / SZ.v n <: natlt (SZ.v m)),
          ((SZ.v gid / SZ.v b % SZ.v n <: natlt (SZ.v n)), ()))))
  as pts_to_cell gC
    ((SZ.v page <: natlt (SZ.v b)),
      ((SZ.v trow <: natlt (SZ.v m)),
        ((SZ.v tcol <: natlt (SZ.v n)), ())))
    (acc3 eC (SZ.v page) (SZ.v trow) (SZ.v tcol));

  (* Slice out the [page]-th pages of A and B (read-only). *)
  tensor_extract_slice_ro gA 0 (SZ.v page);
  tensor_extract_slice_ro gB 0 (SZ.v page);

  let s = Kuiper.DotProd.matmul_dotprod
            (sliceof gA 0 (SZ.v page))
            (sliceof gB 0 (SZ.v page))
            trow tcol;

  (* Restore A and B. *)
  elim_trade
    (sliceof gA 0 (SZ.v page) |-> Frac (fA /. (b * (m * n))) (chest_slice 0 (SZ.v page) eA))
    (gA |-> Frac (fA /. (b * (m * n))) eA);
  elim_trade
    (sliceof gB 0 (SZ.v page) |-> Frac (fB /. (b * (m * n))) (chest_slice 0 (SZ.v page) eB))
    (gB |-> Frac (fB /. (b * (m * n))) eB);

  let v0 = tensor_read_cell gC (page, (trow, (tcol, ())));
  let v1 = comb v0 s;
  tensor_write_cell gC (page, (trow, (tcol, ()))) v1;

  rewrite pts_to_cell gC
    ((SZ.v page <: natlt (SZ.v b)),
      ((SZ.v trow <: natlt (SZ.v m)),
        ((SZ.v tcol <: natlt (SZ.v n)), ())))
    v1
  as pts_to_cell gC
    ((SZ.v gid % SZ.v b <: natlt (SZ.v b)),
      ((SZ.v gid / SZ.v b / SZ.v n <: natlt (SZ.v m)),
        ((SZ.v gid / SZ.v b % SZ.v n <: natlt (SZ.v n)), ())))
    (MS.gemm_single comb
      (slice_page eA (SZ.v gid % SZ.v b))
      (slice_page eB (SZ.v gid % SZ.v b))
      (slice_page eC (SZ.v gid % SZ.v b))
      (SZ.v gid / SZ.v b / SZ.v n)
      (SZ.v gid / SZ.v b % SZ.v n));
  ()
}
#pop-options

ghost
fn setup
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#b #m #n #k : szp)
  (#lA : layout3 b m k)
  (#lB : layout3 b k n)
  (#lC : layout3 b m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (#fA : perm)
  (gB : tensor et lB)
  (#fB : perm)
  (gC : tensor et lC)
  (#eA : chest3 et b m k)
  (#eB : chest3 et b k n)
  (#eC : chest3 et b m n)
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    (forall+ (gid : natlt (b *^ (m *^ n))).
      kpre comb gA gB gC eA eB eC fA fB gid) **
    emp (* frame *)
{
  // Split the read-only input tensors once per output cell.
  tensor_share_n gA (b *^ (m *^ n));
  tensor_share_n gB (b *^ (m *^ n));

  // Explode the output tensor into cells and reshape the rank-3 index.
  tensor_explode gC;
  forevery_iso (abs_bij3 #b #m #n) _;
  forevery_ext _
    (fun (prc : natlt b & (natlt m & natlt n)) ->
      pts_to_cell gC
        (fst prc, (fst (snd prc), (snd (snd prc), ())))
        (acc eC (fst prc, (fst (snd prc), (snd (snd prc), ())))));
  forevery_unflatten' _;

  // Expose row and column as separate indices on every batch page.
  forevery_map #(natlt b)
    (fun page ->
      forall+ (rc : natlt m & natlt n).
        pts_to_cell gC
          (page, (fst rc, (snd rc, ())))
          (acc eC (page, (fst rc, (snd rc, ())))))
    (fun page ->
      forall+ (row : natlt m) (col : natlt n).
        pts_to_cell gC
          (page, (row, (col, ())))
          (acc eC (page, (row, (col, ())))))
    fn page {
      forevery_unflatten' _;
    };

  // Flatten row/column into a single page-local index.
  forevery_map #(natlt b)
    (fun page ->
      forall+ (row : natlt m) (col : natlt n).
        pts_to_cell gC
          (page, (row, (col, ())))
          (acc eC (page, (row, (col, ())))))
    (fun page ->
      forall+ (q : natlt (m *^ n)).
        pts_to_cell gC
          (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
          (acc eC
            (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))))
    fn page {
      forevery_unfactor' (m *^ n) m n (fun row col ->
        pts_to_cell gC
          (page, (row, (col, ())))
          (acc eC (page, (row, (col, ())))));
    };

  // Reshape the replicated input permissions to the same page/local-cell space.
  forevery_factor (b *^ (m *^ n)) b (m *^ n)
    (fun _ -> gA |-> Frac (fA /. (b *^ (m *^ n))) eA);
  forevery_factor (b *^ (m *^ n)) b (m *^ n)
    (fun _ -> gB |-> Frac (fB /. (b *^ (m *^ n))) eB);

  // Attach A and B to every output cell before flattening to the global gid.
  forevery_zip_2 #(natlt b) #(natlt (m *^ n))
    (fun _ _ -> gB |-> Frac (fB /. (b *^ (m *^ n))) eB)
    (fun page q ->
      pts_to_cell gC
        (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
        (acc eC
          (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))));
  forevery_zip_2 #(natlt b) #(natlt (m *^ n))
    (fun _ _ -> gA |-> Frac (fA /. (b *^ (m *^ n))) eA)
    (fun page q ->
      gB |-> Frac (fB /. (b *^ (m *^ n))) eB **
      pts_to_cell gC
        (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
        (acc eC
          (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))));

  // Transpose to (local-cell, page) nesting and flatten in *page-minor*
  // (batch-innermost) order: gid = rest * batch + page, matching the kf's
  // [page = gid % batch], [rest = gid / batch] decomposition.
  forevery_commute #(natlt b) #(natlt (m *^ n))
    (fun page q ->
      gA |-> Frac (fA /. (b *^ (m *^ n))) eA **
      gB |-> Frac (fB /. (b *^ (m *^ n))) eB **
      pts_to_cell gC
        (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
        (acc eC
          (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))));

  forevery_unfactor' (b *^ (m *^ n)) (m *^ n) b (fun q page ->
    gA |-> Frac (fA /. (b *^ (m *^ n))) eA **
    gB |-> Frac (fB /. (b *^ (m *^ n))) eB **
    pts_to_cell gC
      (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
      (acc eC
        (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))));

  forevery_ext #(natlt (b *^ (m *^ n))) _
    (kpre comb gA gB gC eA eB eC fA fB);
  ()
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#b #m #n #k : szp)
  (#lA : layout3 b m k)
  (#lB : layout3 b k n)
  (#lC : layout3 b m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (#fA : perm)
  (gB : tensor et lB)
  (#fB : perm)
  (gC : tensor et lC)
  (#eA : chest3 et b m k)
  (#eB : chest3 et b k n)
  (#eC : chest3 et b m n)
  ()
  norewrite
  requires
    (forall+ (gid : natlt (b *^ (m *^ n))).
      kpost comb gA gB gC eA eB eC fA fB gid) **
    emp (* frame *)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.bmmcomb comb eC eA eB
{
  // First recover page-local cell index and page from the flat gid, in
  // *page-minor* order (page = gid % batch, rest = gid / batch), then
  // transpose back to (page, cell) nesting for the rest of the teardown.
  forevery_factor'
    (b *^ (m *^ n)) (SZ.v m * SZ.v n) (SZ.v b)
    (fun q page ->
    gA |-> Frac (fA /. (b * (m * n))) eA **
    gB |-> Frac (fB /. (b * (m * n))) eB **
    pts_to_cell gC
      (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
      (MS.gemm_single comb
        (slice_page eA page)
        (slice_page eB page)
        (slice_page eC page)
        (q / n)
        (q % n)));

  forevery_commute #(natlt (SZ.v m * SZ.v n)) #(natlt b)
    (fun q page ->
    gA |-> Frac (fA /. (b * (m * n))) eA **
    gB |-> Frac (fB /. (b * (m * n))) eB **
    pts_to_cell gC
      (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
      (MS.gemm_single comb
        (slice_page eA page)
        (slice_page eB page)
        (slice_page eC page)
        (q / n)
        (q % n)));

  // Separate the replicated inputs from the output cells in the nested space.
  forevery_unzip_2 #(natlt b) #(natlt (SZ.v m * SZ.v n))
    (fun _ _ -> gA |-> Frac (fA /. (b * (m * n))) eA)
    (fun page q ->
      gB |-> Frac (fB /. (b * (m * n))) eB **
      pts_to_cell gC
        (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
        (MS.gemm_single comb
          (slice_page eA page)
          (slice_page eB page)
          (slice_page eC page)
          (q / n)
          (q % n)));
  forevery_unzip_2 #(natlt b) #(natlt (SZ.v m * SZ.v n))
    (fun _ _ -> gB |-> Frac (fB /. (b * (m * n))) eB)
    (fun page q ->
      pts_to_cell gC
        (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
        (MS.gemm_single comb
          (slice_page eA page)
          (slice_page eB page)
          (slice_page eC page)
          (q / n)
          (q % n)));

  // Flatten the input permissions again so tensor_gather_n can recombine them.
  forevery_unfactor
    (b *^ (m *^ n)) (SZ.v b) (SZ.v m * SZ.v n)
    (fun _ -> gA |-> Frac (fA /. (b * (m * n))) eA);
  forevery_unfactor
    (b *^ (m *^ n)) (SZ.v b) (SZ.v m * SZ.v n)
    (fun _ -> gB |-> Frac (fB /. (b * (m * n))) eB);

  forevery_rw_type
    (natlt (SZ.v (b *^ (m *^ n))))
    (natlt (SZ.v b * (SZ.v m * SZ.v n)))
    (fun _ -> gA |-> Frac (fA /. (SZ.v b * (SZ.v m * SZ.v n))) eA);
  forevery_rw_type
    (natlt (SZ.v (b *^ (m *^ n))))
    (natlt (SZ.v b * (SZ.v m * SZ.v n)))
    (fun _ -> gB |-> Frac (fB /. (SZ.v b * (SZ.v m * SZ.v n))) eB);

  tensor_gather_n gA _;
  tensor_gather_n gB _;

  // Recover page-local row/column indices.
  forevery_map #(natlt b)
    (fun page ->
      forall+ (q : natlt (SZ.v m * SZ.v n)).
        pts_to_cell gC
          (page, ((q / n <: natlt m), ((q % n <: natlt n), ())))
          (MS.gemm_single comb
            (slice_page eA page)
            (slice_page eB page)
            (slice_page eC page)
            (q / n)
            (q % n)))
    (fun page ->
      forall+ (row : natlt m) (col : natlt n).
        pts_to_cell gC
          (page, (row, (col, ())))
          (MS.gemm_single comb
            (slice_page eA page)
            (slice_page eB page)
            (slice_page eC page)
            row col))
    fn page {
      forevery_factor' (SZ.v m * SZ.v n) (SZ.v m) (SZ.v n)
        (fun row col ->
        pts_to_cell gC
          (page, (row, (col, ())))
          (MS.gemm_single comb
            (slice_page eA page)
            (slice_page eB page)
            (slice_page eC page)
            row col));
    };

  // Reinterpret each computed value as the corresponding bmmcomb cell.
  forevery_map #(natlt b)
    (fun page ->
      forall+ (row : natlt m) (col : natlt n).
        pts_to_cell gC
          (page, (row, (col, ())))
          (MS.gemm_single comb
            (slice_page eA page)
            (slice_page eB page)
            (slice_page eC page)
            row col))
    (fun page ->
      forall+ (row : natlt m) (col : natlt n).
        pts_to_cell gC
          (page, (row, (col, ())))
          (acc (MS.bmmcomb comb eC eA eB)
            (page, (row, (col, ())))))
    fn page {
      forevery_map_2
        (fun row col ->
          pts_to_cell gC
            (page, (row, (col, ())))
            (MS.gemm_single comb
              (slice_page eA page)
              (slice_page eB page)
              (slice_page eC page)
              row col))
        (fun row col ->
          pts_to_cell gC
            (page, (row, (col, ())))
            (acc (MS.bmmcomb comb eC eA eB)
              (page, (row, (col, ())))))
        fn row col {
          ()
        };
    };

  // Reassemble the rank-3 output tensor.
  forevery_map #(natlt b)
    (fun page ->
      forall+ (row : natlt m) (col : natlt n).
        pts_to_cell gC
          (page, (row, (col, ())))
          (acc (MS.bmmcomb comb eC eA eB)
            (page, (row, (col, ())))))
    (fun page ->
      forall+ (rc : natlt m & natlt n).
        pts_to_cell gC
          (page, (fst rc, (snd rc, ())))
          (acc (MS.bmmcomb comb eC eA eB)
            (page, (fst rc, (snd rc, ())))))
    fn page {
      forevery_flatten'
        (fun (rc : natlt m & natlt n) ->
          pts_to_cell gC
            (page, (fst rc, (snd rc, ())))
            (acc (MS.bmmcomb comb eC eA eB)
              (page, (fst rc, (snd rc, ())))));
    };

  forevery_flatten'
    (fun (prc : natlt b & (natlt m & natlt n)) ->
      pts_to_cell gC
        (fst prc, (fst (snd prc), (snd (snd prc), ())))
        (acc (MS.bmmcomb comb eC eA eB)
          (fst prc, (fst (snd prc), (snd (snd prc), ())))));
  forevery_iso (bij_sym (abs_bij3 #b #m #n)) _;
  forevery_ext _ (fun (i : abs (b @| m @| n @| INil)) ->
    pts_to_cell gC i (acc (MS.bmmcomb comb eC eA eB) i));
  tensor_implode gC;
  ()
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#b #m #n #k : szp)
  (#lA : layout3 b m k)
  (#lB : layout3 b k n)
  (#lC : layout3 b m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA : chest3 et b m k)
  (#eB : chest3 et b k n)
  (#eC : chest3 et b m n)
  (#fA #fB : perm)
  (#_ : squash (bsize_req b m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.bmmcomb comb eC eA eB)
=
{
  nblk = b *^ (m *^ n);

  frame = emp;

  setup    = setup    comb gA gB gC;
  teardown = teardown comb gA gB gC;

  kpre  = kpre  comb gA gB gC eA eB eC fA fB;
  kpost = kpost comb gA gB gC eA eB eC fA fB;

  f = kf comb gA gB gC;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_m_1 _ _

inline_for_extraction noextract
fn bmmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (batch m n k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (a : tensor et lA { is_global a })
  (b : tensor et lB { is_global b })
  (c : tensor et lC { is_global c })
  (#eA #eB #eC : chest3 et batch _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA eA ** b |-> Frac fB eB)
  requires
    pure (bsize_req batch m n k) **
    on gpu_loc (c |-> eC)
  ensures
    on gpu_loc (c |-> MS.bmmcomb comb eC eA eB)
{
  launch_sync (kdesc comb a b c #eA #eB #eC);
}

(* GEMM-specific: lifting a rank-2 [mmcomb] through the batch-one
   embedding agrees with running the batched [bmmcomb] and lowering the
   single page back to rank-2.  Kept here because it is specific to the
   GEMM spec (the rank2<->batch-1-rank3 casts themselves live in
   [Kuiper.Matrix.Casts]). *)
let batch1_bmmcomb
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (rows shared cols : szp)
  (eC : chest2 et rows cols)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  : Lemma (
      C.c3_to_c2 rows cols
        (MS.bmmcomb comb
          (C.c2_to_c3 rows cols eC)
          (C.c2_to_c3 rows shared eA)
          (C.c2_to_c3 shared cols eB))
      == MS.mmcomb comb eC eA eB)
  =
  C.c2_to_c3_slice_page rows cols eC;
  C.c2_to_c3_slice_page rows shared eA;
  C.c2_to_c3_slice_page shared cols eB;
  Kuiper.Chest.lemma_equal_intro
    (C.c3_to_c2 rows cols
      (MS.bmmcomb comb
        (C.c2_to_c3 rows cols eC)
        (C.c2_to_c3 rows shared eA)
        (C.c2_to_c3 shared cols eB)))
    (MS.mmcomb comb eC eA eB);
  Kuiper.Chest.ext
    (C.c3_to_c2 rows cols
      (MS.bmmcomb comb
        (C.c2_to_c3 rows cols eC)
        (C.c2_to_c3 rows shared eA)
        (C.c2_to_c3 shared cols eB)))
    (MS.mmcomb comb eC eA eB)

inline_for_extraction noextract
fn mmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (a : tensor et lA { is_global a })
  (b : tensor et lB { is_global b })
  (c : tensor et lC { is_global c })
  (#eA : chest2 et m k)
  (#eB : chest2 et k n)
  (#eC : chest2 et m n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA eA ** b |-> Frac fB eB)
  requires
    pure (bsize_req 1 m n k) **
    on gpu_loc (c |-> eC)
  ensures
    on gpu_loc (c |-> MS.mmcomb comb eC eA eB)
{
  map_loc gpu_loc (fun () -> C.t2_to_t3 m k a);
  map_loc gpu_loc (fun () -> C.t2_to_t3 k n b);
  map_loc gpu_loc (fun () -> C.t2_to_t3 m n c);

  bmmcomb_gpu_exact comb 1sz m n k
    (from_array (C.l2_to_l3 m k #lA) (core a))
    (from_array (C.l2_to_l3 k n #lB) (core b))
    (from_array (C.l2_to_l3 m n #lC) (core c))
    #(C.c2_to_c3 m k eA)
    #(C.c2_to_c3 k n eB)
    #(C.c2_to_c3 m n eC)
    #fA #fB;

  map_loc gpu_loc (fun () -> C.t3_to_t2 m k a);
  C.c2_to_c3_roundtrip m k eA;
  rewrite each C.c3_to_c2 m k (C.c2_to_c3 m k eA) as eA;

  map_loc gpu_loc (fun () -> C.t3_to_t2 k n b);
  C.c2_to_c3_roundtrip k n eB;
  rewrite each C.c3_to_c2 k n (C.c2_to_c3 k n eB) as eB;

  map_loc gpu_loc
    #(from_array (C.l2_to_l3 m n #lC) (core c)
      |-> MS.bmmcomb comb
            (C.c2_to_c3 m n eC)
            (C.c2_to_c3 m k eA)
            (C.c2_to_c3 k n eB))
    #(c |-> MS.mmcomb comb eC eA eB)
    fn _ {
      C.t3_to_t2 m n c;
      batch1_bmmcomb comb m k n eC eA eB;
      rewrite
        (c |->
          C.c3_to_c2 m n
            (MS.bmmcomb comb
              (C.c2_to_c3 m n eC)
              (C.c2_to_c3 m k eA)
              (C.c2_to_c3 k n eB)))
      as
        (c |-> MS.mmcomb comb eC eA eB);
    };
}
