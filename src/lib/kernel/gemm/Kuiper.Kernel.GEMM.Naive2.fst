module Kuiper.Kernel.GEMM.Naive2

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor { tensor_pts_to_cell as pts_to_cell }
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module C = Kuiper.Matrix.Casts
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Bijection
open Pulse.Lib.Trade

(* The reshaping bridge between the nested tensor index
   [abs (m @| n @| INil) = natlt m & (natlt n & unit)]
   and the flat pair [natlt m & natlt n], mirroring
   [Kuiper.Array2.abs_bij]. *)
let abs_bij (#m #n : nat)
  : (abs (m @| n @| INil) =~ (natlt m & natlt n)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
  }

(* ------------------------------------------------------------------ *)
(* Batched rank-3 GEMM over tensors (batch, m, k) * (batch, k, n)       *)
(* = (batch, m, n).  A single launch spawns [batch * m * n] threads     *)
(* (via [kernel_desc_n], full blocks of threads), each computing one    *)
(* output cell of one page with a full dot product.  The thread index   *)
(* [gid] is decomposed in *page-minor* (batch-innermost) order,         *)
(*   page = gid % batch                                                 *)
(*   rest = gid / batch                                                 *)
(*   r    = rest / n                                                     *)
(*   c    = rest % n                                                     *)
(* and thread [gid] computes C[page][r][c].                             *)
(* ------------------------------------------------------------------ *)

(* Reshaping bridge between the 3-D nested index and the flat triple.
   [abs (b @| m @| n @| INil) == natlt b & abs (m @| n @| INil)]
   definitionally, so we reuse [abs_bij] for the inner two dims. *)
let abs_bij3 (#b #m #n : nat)
  : (abs (b @| m @| n @| INil) =~ (natlt b & (natlt m & natlt n))) =
  bij_prod (bij_self (natlt b)) (abs_bij #m #n)

unfold
let bkpre
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
let bkpost
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
fn bkf
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
    bkpre comb gA gB gC eA eB eC fA fB gid
  ensures
    gpu **
    bkpost comb gA gB gC eA eB eC fA fB gid
{
  (* Page-minor (batch-innermost) decomposition. *)
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
fn bsetup
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
      bkpre comb gA gB gC eA eB eC fA fB gid) **
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
  // (batch-innermost) order: gid = rest * batch + page, matching the bkf's
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
    (bkpre comb gA gB gC eA eB eC fA fB);
  ()
}

ghost
fn bteardown
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
      bkpost comb gA gB gC eA eB eC fA fB gid) **
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
let bkdesc
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
  nthr = b *^ (m *^ n);

  frame = emp;

  setup    = bsetup    comb gA gB gC;
  teardown = bteardown comb gA gB gC;

  kpre  = bkpre  comb gA gB gC eA eB eC fA fB;
  kpost = bkpost comb gA gB gC eA eB eC fA fB;

  f = bkf comb gA gB gC;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

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
  launch_sync (bkdesc comb a b c #eA #eB #eC);
}


(* ------------------------------------------------------------------ *)
(* Rank-2 GEMM, derived from the batched kernel at batch one.           *)
(*                                                                      *)
(* Rather than re-implementing the thread-level kernel, the rank-2      *)
(* descriptor reuses the batched thread function/pre/post ([bkf],       *)
(* [bkpre], [bkpost]) and the batched [bsetup]/[bteardown], run over a  *)
(* single-page ([batch = 1]) view of the rank-2 matrices obtained by    *)
(* the ownership casts in [Kuiper.Matrix.Casts].  Only the outermost    *)
(* [setup]/[teardown] are wrapped to bridge between the rank-2 global   *)
(* ownership and its single-page rank-3 relayout.                       *)
(* ------------------------------------------------------------------ *)

(* Lowering a one-page batched [mmcomb] yields the rank-2 [mmcomb]. *)
let batch1_mmcomb
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (eC : chest2 et m n)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  : Lemma (
      C.c3_to_c2 m n
        (MS.bmmcomb comb
          (C.c2_to_c3 m n eC)
          (C.c2_to_c3 m k eA)
          (C.c2_to_c3 k n eB))
      == MS.mmcomb comb eC eA eB)
  =
  C.c2_to_c3_slice_page m n eC;
  C.c2_to_c3_slice_page m k eA;
  C.c2_to_c3_slice_page k n eB;
  assert (equal
      (C.c3_to_c2 m n
        (MS.bmmcomb comb
          (C.c2_to_c3 m n eC)
          (C.c2_to_c3 m k eA)
          (C.c2_to_c3 k n eB)))
      (MS.mmcomb comb eC eA eB))

(* ------------------------------------------------------------------ *)
(* Rank-2 GEMM, obtained by *casting* the single batched descriptor     *)
(* [bkdesc] at batch one.  The thread-level kernel and its per-thread    *)
(* pre/post/setup/teardown are not re-implemented: [kdesc] is literally  *)
(* [bkdesc] over a single-page ([batch = 1]) rank-3 relayout of the      *)
(* rank-2 matrices, with two ghost steps composed onto its outer         *)
(* pre/post to bridge the rank-2 ownership and the batched one.          *)
(* ------------------------------------------------------------------ *)

(* [bsize_req] at batch one follows from the rank-2 [size_req]. *)
let size_req_bsize1 (m n k : szp)
  : Lemma (requires size_req m n k) (ensures bsize_req 1 m n k)
  = let _ = max_blocks_explicit in
    assert_norm (SZ.v max_threads == 1024)

(* Generic reframing of a kernel descriptor: pre-compose [fwd] onto the
   setup and post-compose [bwd] onto the teardown, leaving the kernel
   itself (threads, per-thread pre/post, block machinery) untouched. *)
ghost
fn kd_pre_compose
  (#pre #post #pre' : slprop)
  (fwd : unit -> stt_ghost unit emp_inames pre' (fun _ -> pre))
  (kd : kernel_desc pre post)
  ()
  requires pre'
  ensures (forall+ (bid : natlt kd.nblk). kd.block_pre bid) ** kd.frame
{
  fwd ();
  let setup = kd.setup;
  setup ();
}

ghost
fn kd_post_compose
  (#pre #post #post' : slprop)
  (bwd : unit -> stt_ghost unit emp_inames post (fun _ -> post'))
  (kd : kernel_desc pre post)
  ()
  requires (forall+ (bid : natlt kd.nblk). kd.block_post bid) ** kd.frame
  ensures post'
{
  let teardown = kd.teardown;
  teardown ();
  bwd ();
}

inline_for_extraction noextract
let kdreframe
  (#pre #post #pre' #post' : slprop)
  (fwd : unit -> stt_ghost unit emp_inames pre' (fun _ -> pre))
  (bwd : unit -> stt_ghost unit emp_inames post (fun _ -> post'))
  (kd : kernel_desc pre post)
  : kernel_desc pre' post'
  = { kd with
        setup    = kd_pre_compose  fwd kd;
        teardown = kd_post_compose bwd kd; }

#push-options "--z3rlimit 40 --split_queries always"

(* Ghost step raising the rank-2 global ownership to the batch-one
   rank-3 view that [bkdesc]'s setup expects. *)
ghost
fn cast_in
  (#et : Type0) {| scalar et |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (fA fB : perm)
  ()
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> eC
  ensures
    relay gA (C.l2_to_l3 m k #lA) |-> Frac fA (C.c2_to_c3 m k eA) **
    relay gB (C.l2_to_l3 k n #lB) |-> Frac fB (C.c2_to_c3 k n eB) **
    relay gC (C.l2_to_l3 m n #lC) |-> C.c2_to_c3 m n eC
{
  C.t2_to_t3 m k gA;
  C.t2_to_t3 k n gB;
  C.t2_to_t3 m n gC;
}

(* Inverse ghost step lowering the batched result back to the rank-2
   [mmcomb] postcondition. *)
ghost
fn cast_out
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA)
  (gB : tensor et lB)
  (gC : tensor et lC)
  (eA : chest2 et m k)
  (eB : chest2 et k n)
  (eC : chest2 et m n)
  (fA fB : perm)
  ()
  norewrite
  requires
    relay gA (C.l2_to_l3 m k #lA) |-> Frac fA (C.c2_to_c3 m k eA) **
    relay gB (C.l2_to_l3 k n #lB) |-> Frac fB (C.c2_to_c3 k n eB) **
    relay gC (C.l2_to_l3 m n #lC) |->
      MS.bmmcomb comb (C.c2_to_c3 m n eC) (C.c2_to_c3 m k eA) (C.c2_to_c3 k n eB)
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> MS.mmcomb comb eC eA eB
{
  C.t3_to_t2 m k gA;
  C.t3_to_t2 k n gB;
  C.t3_to_t2 m n gC;
  C.c2_to_c3_roundtrip m k eA;
  C.c2_to_c3_roundtrip k n eB;
  batch1_mmcomb comb eC eA eB;
  rewrite
    (gA |-> Frac fA (C.c3_to_c2 m k (C.c2_to_c3 m k eA)))
  as
    (gA |-> Frac fA eA);
  rewrite
    (gB |-> Frac fB (C.c3_to_c2 k n (C.c2_to_c3 k n eB)))
  as
    (gB |-> Frac fB eB);
  rewrite
    (gC |-> C.c3_to_c2 m n
              (MS.bmmcomb comb
                (C.c2_to_c3 m n eC)
                (C.c2_to_c3 m k eA)
                (C.c2_to_c3 k n eB)))
  as
    (gC |-> MS.mmcomb comb eC eA eB);
}

(* Rank-2 kernel descriptor, exposed so callers can launch it directly
   (e.g. asynchronously, as in [Kuiper.Example.Async.GEMM]).  It is the
   batched [bkdesc] at batch one, reframed by [cast_in]/[cast_out]. *)
inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA #eB #eC : chest2 et _ _)
  (#fA #fB : perm)
  (#_ : squash (size_req m n k))
  : kernel_desc
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
    (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.mmcomb comb eC eA eB)
=
  size_req_bsize1 m n k;
  kdreframe
    (cast_in gA gB gC eA eB eC fA fB)
    (cast_out comb gA gB gC eA eB eC fA fB)
    (bkdesc comb #1sz #m #n #k
      #(C.l2_to_l3 m k #lA) #(C.l2_to_l3 k n #lB) #(C.l2_to_l3 m n #lC)
      (relay gA (C.l2_to_l3 m k #lA))
      (relay gB (C.l2_to_l3 k n #lB))
      (relay gC (C.l2_to_l3 m n #lC))
      #(C.c2_to_c3 m k eA) #(C.c2_to_c3 k n eB) #(C.c2_to_c3 m n eC)
      #fA #fB)
#pop-options

(* Rank-2 GEMM: a single launch spawns [m * n] threads (via
   [kernel_desc_n]), each computing one output cell with a full dot
   product. *)
inline_for_extraction noextract
fn mmcomb_gpu_exact
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA #eB #eC : chest2 et _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  launch_sync (kdesc comb gA gB gC #eA #eB #eC);
}
