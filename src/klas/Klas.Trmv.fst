module Klas.Trmv

(* cuBLAS trmv (lower-triangular, non-unit, no transpose): triangular
   matrix-vector product  y := A * x, where A is lower-triangular so

       y[i] = sum_{j<=i} A[i][j] * x[j].

   A is n x n row-major in a flat length-(n*n) array; x, y are length n. Each
   y[i] is independent (it reads x, not y), so unlike trsv there is no solve and
   no division; we still run a single thread for simplicity. The index-bound
   helper idx_bound is reused from Klas.Trsv. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Trsv { idx_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT

(* ----------------------------------------------------------------------- *)
(* Real and float specs: the triangular row dot sum_{j<m} A[i][j] x[j].      *)
(* ----------------------------------------------------------------------- *)

let rec rtrmv_dot (n:nat) (na:nat{n * n <= na}) (sA : lseq real na) (rx : lseq real n)
  (i:nat{i < n}) (m:nat{m <= i + 1})
  : Tot real (decreases m)
  = if m = 0 then 0.0R
    else (idx_bound n i (m - 1);
          rtrmv_dot n na sA rx i (m - 1)
          +. (Seq.index sA (i * n + (m - 1))) *. (Seq.index rx (m - 1)))

let rtrmv_at (n:nat) (na:nat{n * n <= na}) (sA : lseq real na) (rx : lseq real n) (i:nat{i < n}) : real
  = rtrmv_dot n na sA rx i (i + 1)

let rec ftrmv_dot (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n)
  (i:nat{i < n}) (m:nat{m <= i + 1})
  : Tot et (decreases m)
  = if m = 0 then zero
    else (idx_bound n i (m - 1);
          (ftrmv_dot n na sA sx i (m - 1))
          `add` ((Seq.index sA (i * n + (m - 1))) `mul` (Seq.index sx (m - 1))))

let ftrmv_at (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n) (i:nat{i < n}) : et
  = ftrmv_dot n na sA sx i (i + 1)

let ftrmv (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n) : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> ftrmv_at n na sA sx i)

let ftrmv_index (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n) (k:nat{k < n})
  : Lemma (ensures Seq.index (ftrmv n na sA sx) k == ftrmv_at n na sA sx k)
          [SMTPat (Seq.index (ftrmv n na sA sx) k)]
  = ()

(* The float row dot approximates the real one (a_mul / a_add). *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
let rec ftrmv_dot_approx
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n)
  (rA : lseq real na) (rx : lseq real n) (i:nat{i < n}) (m:nat{m <= i + 1})
  : Lemma (requires sA %~ rA /\ sx %~ rx)
          (ensures ftrmv_dot n na sA sx i m %~ rtrmv_dot n na rA rx i m)
          (decreases m)
  = if m = 0 then ()
    else (idx_bound n i (m - 1);
          ftrmv_dot_approx n na sA sx rA rx i (m - 1);
          assert ((sA @! (i * n + (m - 1))) %~ (rA @! (i * n + (m - 1))));
          assert ((sx @! (m - 1)) %~ (rx @! (m - 1))))
#pop-options

let ftrmv_at_approx
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n)
  (rA : lseq real na) (rx : lseq real n) (i:nat{i < n})
  : Lemma (requires sA %~ rA /\ sx %~ rx)
          (ensures ftrmv_at n na sA sx i %~ rtrmv_at n na rA rx i)
  = ftrmv_dot_approx n na sA sx rA rx i (i + 1)

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one triangular row dot per output entry.            *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 80"
inline_for_extraction noextract
fn trmv_row
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (i : szlt n)
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx
  returns
    yi : et
  ensures
    pure (yi == ftrmv_at (SZ.v n) (SZ.v (n *^ n)) sA sx (SZ.v i))
{
  let mut s : et = zero;
  let mut j : sz = 0sz;

  while (!j <^ (i +^ 1sz))
    invariant live s
    invariant live j
    invariant pure (SZ.v !j <= SZ.v i + 1 /\
                    !s == ftrmv_dot (SZ.v n) (SZ.v (n *^ n)) sA sx (SZ.v i) (SZ.v !j))
    decreases (SZ.v i + 1 - SZ.v !j)
  {
    let vj = !j;
    let aij = Array1.(gA.((i *^ n) +^ vj));
    let xj = Array1.(gx.(vj));
    s := !s `add` (aij `mul` xj);
    j := !j +^ 1sz;
  };
  !s
}
#pop-options

(* Writing the freshly-computed y[i] keeps the earlier entries (each y[k] is
   independent of the others). *)
let trmv_upd
  (#et:Type0) {| floating et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n)
  (syp : lseq et n) (vi : nat{vi < n}) (yi : et)
  : Lemma (requires (forall (k:nat). k < vi ==> Seq.index syp k == ftrmv_at n na sA sx k) /\
                    yi == ftrmv_at n na sA sx vi)
          (ensures (forall (k:nat). k < vi + 1 ==>
                      Seq.index (Seq.upd syp vi yi) k == ftrmv_at n na sA sx k))
  = introduce forall (k:nat). k < vi + 1 ==>
                Seq.index (Seq.upd syp vi yi) k == ftrmv_at n na sA sx k
    with introduce _ ==> _
    with _. (if k < vi
             then Seq.lemma_index_upd2 syp vi yi k
             else Seq.lemma_index_upd1 syp vi yi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 80"
inline_for_extraction noextract
fn trmv_kf
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0
  ensures
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sy : lseq et (SZ.v n)). gy |-> sy **
      pure (SZ.v !i <= SZ.v n /\
            (forall (k:nat). k < SZ.v !i ==>
               Seq.index sy k == ftrmv_at (SZ.v n) (SZ.v (n *^ n)) sA sx k))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let yi = trmv_row gA gx vi;
    with syp. assert (gy |-> syp);
    Array1.(gy.(vi) <- yi);
    trmv_upd (SZ.v n) (SZ.v (n *^ n)) sA sx syp (SZ.v vi) yi;
    i := !i +^ 1sz;
  };

  with syf. assert (gy |-> syf);
  Seq.lemma_eq_intro syf (ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trsv).           *)
(* ----------------------------------------------------------------------- *)

(* The whole computed vector approximates the real triangular product. *)
let ftrmv_approx_all
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n:nat) (na:nat{n * n <= na}) (sA : lseq et na) (sx : lseq et n)
  (rA : lseq real na) (rx : lseq real n)
  : Lemma (requires sA %~ rA /\ sx %~ rx)
          (ensures forall (k:nat). k < n ==>
                     ((ftrmv n na sA sx) @! k) %~ (rtrmv_at n na rA rx k))
  = introduce forall (k:nat). k < n ==>
                ((ftrmv n na sA sx) @! k) %~ (rtrmv_at n na rA rx k)
    with introduce _ ==> _
    with _. (ftrmv_at_approx n na sA sx rA rx k; ftrmv_index n na sA sx k)

ghost
fn trmv_setup
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0
  ensures (forall+ (tid : natlt 1sz).
             gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
}

ghost
fn trmv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx) ** emp
  ensures
    gA |-> Frac fa sA ** gx |-> Frac fx sx **
    gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
       gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx);
}

inline_for_extraction noextract
let kamtrmv
  (#et:Type0) {| floating et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gx /\ Array1.is_global gy))
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0)
      (ensures  gA |-> Frac fa sA ** gx |-> Frac fx sx **
                gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> trmv_kf gA gx gy #sA #sx #sy0 #fa #fx);
    frame    = emp;
    teardown = trmv_teardown gA gx gy;
    setup    = trmv_setup gA gx gy;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx ** gy |-> sy0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gx |-> Frac fx sx **
                                     gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn trmv_gen
  (#et:Type0) {| floating et, real_like et, floating_real_like et |}
  (n : szp { SZ.fits (SZ.v n * SZ.v n) })
  (gA : array1 et (l1_forward (n *^ n)) { Array1.is_global gA })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (gy : array1 et (l1_forward n) { Array1.is_global gy })
  (#sA : erased (lseq et (SZ.v (n *^ n))))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy0 : erased (lseq et (SZ.v n)))
  (rA : erased (lseq real (SZ.v (n *^ n))))
  (rx : erased (lseq real (SZ.v n)))
  (#fa #fx : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gx |-> Frac fx sx)
  requires
    on gpu_loc (gy |-> sy0) **
    pure (sA %~ rA /\ sx %~ rx)
  ensures
    on gpu_loc (gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx) **
    pure (forall (k:nat). k < SZ.v n ==>
            ((ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx) @! k)
            %~ (rtrmv_at (SZ.v n) (SZ.v (n *^ n)) rA rx k))
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> sy0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> sy0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gx |-> Frac fx sx) ** (gy |-> sy0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> sy0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> sy0))));

  launch_sync (kamtrmv n gA gx gy #() #sA #sx #sy0 #fa #fx);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gx |-> Frac fx sx) ** (gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gx |-> Frac fx sx) ** (gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gy |-> ftrmv (SZ.v n) (SZ.v (n *^ n)) sA sx));

  ftrmv_approx_all (SZ.v n) (SZ.v (n *^ n)) sA sx rA rx;
  ()
}

let trmv_f32 = trmv_gen #f32
let trmv_f64 = trmv_gen #f64
