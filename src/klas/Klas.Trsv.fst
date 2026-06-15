module Klas.Trsv

(* cuBLAS trsv: solve a triangular system  A * x = b  in place.

   This first version handles a LOWER-triangular, NON-unit-diagonal A, no
   transpose, by forward substitution:

       x[i] = ( b[i] - sum_{j<i} A[i][j] * x[j] ) / A[i][i].

   A is an n x n matrix stored row-major as a flat length-(n*n) array; b and the
   result x are length-n vectors. The solve is inherently sequential, so the
   kernel runs on a single thread. We require the (real) diagonal to be nonzero
   so the system has a unique solution.

   This module: the pure real-valued specification and its prefix-stability
   lemma. The verified kernel (which approximates this spec) follows. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT

(* ----------------------------------------------------------------------- *)
(* Pure real specification (forward substitution)                            *)
(* ----------------------------------------------------------------------- *)

(* Flat row-major index bound: for an n x n matrix stored in an array of length
   na >= n*n, every (i,j) with i,j < n lands in range. *)
let idx_bound (n i j : nat)
  : Lemma (requires i < n /\ j < n) (ensures i * n + j < n * n)
  = FStar.Math.Lemmas.lemma_mult_le_right n i (n - 1)

(* A is an n x n row-major matrix held in the first n*n cells of a length-[na]
   array (na >= n*n; the kernel uses na = n*n exactly). *)
let diag_nonzero (n:nat) (na:nat{n * n <= na}) (sA : lseq real na) : prop =
  forall (i:nat). i < n ==> (idx_bound n i i; Seq.index sA (i * n + i) =!= 0.0R)

(* [rtrsv_at i] is the i-th solution entry; [rsub_dot_at i m] is the partial dot
   sum_{j<m} A[i][j] * x[j] of row i with the (already-solved) entries below it.
   They are mutually recursive: x[i] divides (b[i] - partial dot) by A[i][i],
   and the partial dot references the earlier solution entries x[j], j<i.
   Termination: lexicographic on (i, then the partial-dot tag, then m). *)
let rec rtrsv_at (n:nat) (na:nat{n * n <= na}) (sA : lseq real na) (sb : lseq real n)
  (nz : squash (diag_nonzero n na sA)) (i:nat{i < n})
  : Tot real (decreases %[i; 1; 0])
  = idx_bound n i i;
    let dii = Seq.index sA (i * n + i) in
    assert (dii =!= 0.0R);
    ((Seq.index sb i) -. rsub_dot_at n na sA sb nz i i) /. dii

and rsub_dot_at (n:nat) (na:nat{n * n <= na}) (sA : lseq real na) (sb : lseq real n)
  (nz : squash (diag_nonzero n na sA)) (i:nat{i < n}) (m:nat{m <= i})
  : Tot real (decreases %[i; 0; m])
  = if m = 0 then 0.0R
    else begin
      idx_bound n i (m - 1);
      rsub_dot_at n na sA sb nz i (m - 1)
      +. (Seq.index sA (i * n + (m - 1))) *. (rtrsv_at n na sA sb nz (m - 1))
    end

(* ----------------------------------------------------------------------- *)
(* Float specification: the value the kernel actually computes (total, since  *)
(* float division is total). [ftrsv] gives a clean output for the kernel; the *)
(* approximation [ftrsv_approx] below relates it to the real solution.        *)
(* ----------------------------------------------------------------------- *)

let rec ftrsv_at (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sb : lseq et n) (i:nat{i < n})
  : Tot et (decreases %[i; 1; 0])
  = idx_bound n i i;
    div (srow n na sA sb i i) (Seq.index sA (i * n + i))

(* [srow i m] is the running accumulator b[i] - sum_{j<m} A[i][j]*x[j]. *)
and srow (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sb : lseq et n) (i:nat{i < n}) (m:nat{m <= i})
  : Tot et (decreases %[i; 0; m])
  = if m = 0 then Seq.index sb i
    else begin
      idx_bound n i (m - 1);
      (srow n na sA sb i (m - 1))
      `sub` ((Seq.index sA (i * n + (m - 1))) `mul` (ftrsv_at n na sA sb (m - 1)))
    end

let ftrsv (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sb : lseq et n)
  : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> ftrsv_at n na sA sb i)

let ftrsv_index (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sb : lseq et n) (k:nat{k < n})
  : Lemma (ensures Seq.index (ftrsv n na sA sb) k == ftrsv_at n na sA sb k)
          [SMTPat (Seq.index (ftrsv n na sA sb) k)]
  = ()

(* The float solution approximates the real solution, entry by entry: this is
   the actual solve guarantee. Mutual induction mirroring the two specs; each
   step is closed by the a_mul / sub_approx / div_approx SMT-pattern lemmas. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 80"
let rec ftrsv_at_approx
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:nat) (na:nat{n * n <= na})
  (sA : lseq et na) (sb : lseq et n) (rA : lseq real na) (rb : lseq real n)
  (nz : squash (diag_nonzero n na rA)) (i:nat{i < n})
  : Lemma (requires sA %~ rA /\ sb %~ rb)
          (ensures ftrsv_at n na sA sb i %~ rtrsv_at n na rA rb nz i)
          (decreases %[i; 1; 0])
  = idx_bound n i i;
    srow_approx n na sA sb rA rb nz i i;
    assert ((sA @! (i * n + i)) %~ (rA @! (i * n + i)));
    assert (Seq.index rA (i * n + i) =!= 0.0R)

and srow_approx
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:nat) (na:nat{n * n <= na})
  (sA : lseq et na) (sb : lseq et n) (rA : lseq real na) (rb : lseq real n)
  (nz : squash (diag_nonzero n na rA)) (i:nat{i < n}) (m:nat{m <= i})
  : Lemma (requires sA %~ rA /\ sb %~ rb)
          (ensures srow n na sA sb i m %~ ((rb @! i) -. rsub_dot_at n na rA rb nz i m))
          (decreases %[i; 0; m])
  = if m = 0 then assert ((sb @! i) %~ (rb @! i))
    else begin
      idx_bound n i (m - 1);
      srow_approx n na sA sb rA rb nz i (m - 1);
      ftrsv_at_approx n na sA sb rA rb nz (m - 1);
      assert ((sA @! (i * n + (m - 1))) %~ (rA @! (i * n + (m - 1))))
    end
#pop-options

(* ----------------------------------------------------------------------- *)
(* Per-row computation (the sequential inner dot + the division).            *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
inline_for_extraction noextract
fn trsv_row
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (i : szlt n)
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fb #fx : perm)
  preserves
    gpu **
    gA |-> Frac fa sA **
    gb |-> Frac fb sb **
    gx |-> Frac fx sx
  requires
    pure (forall (k:nat). k < SZ.v i ==>
            Seq.index sx k == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA sb k)
  returns
    xi : et
  ensures
    pure (xi == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA sb (SZ.v i))
{
  let mut s : et = Array1.(gb.(i));
  let mut j : szle i = 0sz;

  while (!j <^ i)
    invariant live s
    invariant live j
    invariant pure (SZ.v !j <= SZ.v i /\
                    !s == srow (SZ.v n) (SZ.v (n *^ n)) sA sb (SZ.v i) (SZ.v !j))
    decreases (SZ.v i - SZ.v !j)
  {
    let vj = !j;
    let aij = Array1.(gA.((i *^ n) +^ vj));
    let xj = Array1.(gx.(vj));
    s := !s `sub` (aij `mul` xj);
    j := !j +^ 1sz;
  };

  let aii = Array1.(gA.((i *^ n) +^ i));
  !s `div` aii
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: forward substitution row by row.                    *)
(* ----------------------------------------------------------------------- *)

(* Writing the freshly-solved x[vi] keeps the earlier entries and adds the new
   one (case split on the updated index). *)
let upd_preserves_eq
  (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sb : lseq et n)
  (sxpre : lseq et n) (vi : nat{vi < n}) (xi : et)
  : Lemma (requires (forall (k:nat). k < vi ==> Seq.index sxpre k == ftrsv_at n na sA sb k) /\
                    xi == ftrsv_at n na sA sb vi)
          (ensures (forall (k:nat). k < vi + 1 ==>
                      Seq.index (Seq.upd sxpre vi xi) k == ftrsv_at n na sA sb k))
  = introduce forall (k:nat). k < vi + 1 ==>
                Seq.index (Seq.upd sxpre vi xi) k == ftrsv_at n na sA sb k
    with introduce _ ==> _
    with _. (if k < vi
             then Seq.lemma_index_upd2 sxpre vi xi k
             else Seq.lemma_index_upd1 sxpre vi xi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 80"
inline_for_extraction noextract
fn trsv_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx0 : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0
  ensures
    gpu ** gA |-> Frac fa sA ** gb |-> Frac fb sb **
    gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sx : lseq et (SZ.v n)). gx |-> sx **
      pure (SZ.v !i <= SZ.v n /\
            (forall (k:nat). k < SZ.v !i ==>
               Seq.index sx k == ftrsv_at (SZ.v n) (SZ.v (n *^ n)) sA sb k))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let xi = trsv_row gA gb gx vi;
    with sxpre. assert (gx |-> sxpre);
    Array1.(gx.(vi) <- xi);
    upd_preserves_eq (SZ.v n) (SZ.v (n *^ n)) sA sb sxpre (SZ.v vi) xi;
    i := !i +^ 1sz;
  };

  with sxf. assert (gx |-> sxf);
  Seq.lemma_eq_intro sxf (ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Amax).           *)
(* ----------------------------------------------------------------------- *)

(* The whole computed vector approximates the real solution. *)
let ftrsv_approx_all
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:nat) (na:nat{n * n <= na})
  (sA : lseq et na) (sb : lseq et n) (rA : lseq real na) (rb : lseq real n)
  (nz : squash (diag_nonzero n na rA))
  : Lemma (requires sA %~ rA /\ sb %~ rb)
          (ensures forall (k:nat). k < n ==>
                     ((ftrsv n na sA sb) @! k) %~ (rtrsv_at n na rA rb nz k))
  = introduce forall (k:nat). k < n ==>
                ((ftrsv n na sA sb) @! k) %~ (rtrsv_at n na rA rb nz k)
    with introduce _ ==> _
    with _. (ftrsv_at_approx n na sA sb rA rb nz k; ftrsv_index n na sA sb k)

ghost
fn trsv_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx0 : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0
  ensures
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0);
}

ghost
fn trsv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sb : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gb |-> Frac fb sb **
       gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb) ** emp
  ensures
    gA |-> Frac fa sA ** gb |-> Frac fb sb **
    gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb **
                          gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb);
}

inline_for_extraction noextract
let kamtrsv
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gb /\ Array1.is_global gx))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx0 : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0)
      (ensures  gA |-> Frac fa sA ** gb |-> Frac fb sb **
                gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> trsv_kf gA gb gx #sA #sb #sx0 #fa #fb);

    frame    = emp;
    teardown = trsv_teardown gA gb gx;
    setup    = trsv_setup gA gb gx;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb **
                                     gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn trsv_gen
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)) { Array1.is_global gA })
  (gb : array1 et (l1_forward n) { Array1.is_global gb })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx0 : erased (lseq et (SZ.v n)))
  (rA : erased (lseq real (SZ.v (n *^ n))))
  (rb : erased (lseq real (SZ.v n)))
  (nz : squash (diag_nonzero (SZ.v n) (SZ.v (n *^ n)) rA))
  (#fa #fb : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gb |-> Frac fb sb)
  requires
    on gpu_loc (gx |-> sx0) **
    pure (sA %~ rA /\ sb %~ rb)
  ensures
    on gpu_loc (gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb) **
    pure (forall (k:nat). k < SZ.v n ==>
            ((ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb) @! k)
            %~ (rtrsv_at (SZ.v n) (SZ.v (n *^ n)) rA rb nz k))
{
  on_star_eq gpu_loc (gb |-> Frac fb sb) (gx |-> sx0);
  rewrite (on gpu_loc (gb |-> Frac fb sb) ** on gpu_loc (gx |-> sx0))
       as (on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> sx0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gb |-> Frac fb sb) ** (gx |-> sx0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> sx0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gb |-> Frac fb sb) ** (gx |-> sx0))));

  launch_sync (kamtrsv n gA gb gx #() #sA #sb #sx0 #fa #fb);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gb |-> Frac fb sb) ** (gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gb |-> Frac fb sb) ** (gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb)));
  on_star_eq gpu_loc (gb |-> Frac fb sb) (gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb);
  rewrite (on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb)))
       as (on gpu_loc (gb |-> Frac fb sb) ** on gpu_loc (gx |-> ftrsv (SZ.v n) (SZ.v (n *^ n)) sA sb));

  ftrsv_approx_all (SZ.v n) (SZ.v (n *^ n)) sA sb rA rb nz;
  ()
}

let trsv_f32 = trsv_gen #f32
let trsv_f64 = trsv_gen #f64
