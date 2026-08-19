module Kuiper.Tensor.Layout.Alg

module T = Kuiper.Tensor
(* Constructing tensor layouts algebraically. *)
open Kuiper
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Chest
open FStar.Tactics.Typeclasses { no_method }
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Layout

(* Injectivity of [major_on_f].  This used to be discharged by the [easy_fill]
   default of [injection.is_inj], but now that F* emits one SMT query per proof
   obligation the Euclidean argument no longer goes through in one shot, so we
   spell it out: [maj * sizeof d + sub.f min] determines [maj] as the quotient
   and [sub.f min] as the remainder modulo [sizeof d]. *)
let major_on_inj (#n:nat)
  (i : natlt (n+1))
  (k : nat)
  (#d : shape n)
  (sub : layout_f_for d)
  (x : abs (insert_i i k d))
  (y : abs (insert_i i k d) { major_on_f i k sub x == major_on_f i k sub y })
  : squash (x == y)
  = let bij = abs_bring_forward_bij i (insert_i i k d) in
    let (majx, minx) = bij.ff x in
    let (majy, miny) = bij.ff y in
    let rx = sub.f minx in
    let ry = sub.f miny in
    let m : pos = sizeof d in
    assert (majx * m + rx == majy * m + ry);
    FStar.Math.Lemmas.lemma_div_plus rx majx m;
    FStar.Math.Lemmas.lemma_div_plus ry majy m;
    FStar.Math.Lemmas.small_div rx m;
    FStar.Math.Lemmas.small_div ry m;
    assert (majx == majy);
    assert (rx == ry);
    sub.is_inj minx miny;
    assert (bij.ff x == bij.ff y);
    bij.gg_ff x;
    bij.gg_ff y

let major_on (#n:nat)
  (i : natlt (n+1))
  (k : nat)
  (#d : shape n)
  (sub : layout_f_for d)
  : layout_f_for (insert_i i k d)
  = {
    f = major_on_f i k sub;
    is_inj = major_on_inj i k sub;
  }


inline_for_extraction noextract
instance csizeof_INil : csizeof INil = { v = 1sz; }

inline_for_extraction noextract
instance csizeof_ICons
  (#n : erased nat)
  (d0 : SZ.t)
  (d1 : shape n)
  (c_d1 : csizeof d1)
  (#_ : squash (SZ.fits (d0 * c_d1.v)))
  : csizeof (ICons d0 d1) =
  { v = SZ.mul d0 c_d1.v; }

inline_for_extraction noextract
instance csizeof_insert_i
  (#n : erased nat)
  (i : erased nat{i < n+1})
  (k : sz)
  (d : shape n)
  (c_d : csizeof d)
  (#_ : squash (SZ.fits (k * c_d.v)))
  : csizeof (insert_i i k d)
  = { v = SZ.mul k c_d.v; }

inline_for_extraction noextract
instance cunit : auto_cinj lunit = {
  ff = (fun _ -> 0sz);
}

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let c_major_on_f
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : shape n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (#_ : squash (SZ.fits (k * sizeof d)))
  (c_sub : auto_cinj sub)
  (idx:conc (insert_i i k d))
  : r : szlt (sizeof (insert_i i k d)) { SZ.v r == major_on_f i k sub (up idx) }
  = // let maj, min = c_bring_forward_ff i (insert_i i k d) idx in
    // ^ Still unclear why something like this does not extract, but the below does.
    match c_bring_forward_ff (SZ.v i) (insert_i i k d) idx with maj, min ->
    assert (maj < (insert_i i k d) @! i);
    assert ((insert_i i k d) @! i == reveal k);
    assert (maj < k);
    match c_sub with { ff } ->
    let sub_i : szlt (sizeof d) = ff min in
    lemma_c_bring_forward_ff_ok i (insert_i i k d) idx;
    bring_forward_commute2 i (insert_i i k d) maj min;
    assert (maj * cs.v <= (k-1) * sizeof d);
    let offset : SZ.t = maj *^ cs.v in
    assert (sub_i < sizeof d);
    assert (maj * cs.v + sub_i < k * sizeof d);
    let r = offset +^ sub_i in
    r
#pop-options

inline_for_extraction noextract
instance c_major_on
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : shape n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (#_ : squash (SZ.fits (k * sizeof d)))
  (c_sub : auto_cinj sub)
  : auto_cinj (major_on i k sub) =
  { ff = c_major_on_f i k c_sub; }

inline_for_extraction noextract
instance c_pack (#n : erased nat) (#d: shape n)
  (#f : layout_f_for d) (c_f : auto_cinj f)
  (#_ : squash (SZ.fits (sizeof d)))
  (#_ : squash (all_fit d))
  : ctlayout (pack f) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap   = (fun x -> c_f.ff x);
  }

inline_for_extraction noextract
instance c_l1_forward (m : erased nat{SZ.fits m}) : T.ctlayout (l1_forward m) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Shape.conc (m @| INil)) ->
              match idx with
              | (i, ()) -> i);
  }

inline_for_extraction noextract
instance c_l2_row_major (m : erased nat{SZ.fits m}) (n : SZ.t{SZ.fits (m * n)}) : T.ctlayout (l2_row_major m n) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Shape.conc (m @| n @| INil)) ->
              match idx with
              | (i, (j, ())) -> SZ.add (SZ.mul i n) j)
  }

inline_for_extraction noextract
instance c_r2_row_major : ctrepr2 l2_row_major = {
  inst = (fun m n #_ -> c_l2_row_major (SZ.v m) n);
}

inline_for_extraction noextract
instance c_l2_col_major (m : sz) (n : erased nat{SZ.fits n /\ SZ.fits (m * n)}) : T.ctlayout (l2_col_major m n) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Shape.conc (m @| n @| INil)) ->
              match idx with
              | (i, (j, ())) -> SZ.add (SZ.mul j m) i)
  }

inline_for_extraction noextract
instance c_r2_col_major : ctrepr2 l2_col_major = {
  inst = (fun m n #_ -> c_l2_col_major m (SZ.v n));
}

#push-options "--z3rlimit 80"
inline_for_extraction noextract
instance c_l3_batched_row_major
  (r : erased nat{SZ.fits r})
  (m : SZ.t{SZ.fits (r * m)})
  (n : SZ.t{SZ.fits (m * n) /\ SZ.fits (r * (m * n))})
  : T.ctlayout (l3_batched_row_major r m n) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Shape.conc (r @| m @| n @| INil)) ->
              match idx with
              | (i, (j, (k, ()))) ->
                SZ.add (SZ.mul i (SZ.mul m n)) (SZ.add (SZ.mul j n) k))
  }
#pop-options

#push-options "--fuel 2 --ifuel 2 --z3rlimit 80"
let l3_batched_row_major_imap
  (r : erased nat{SZ.fits r})
  (m : SZ.t{SZ.fits (r * m)})
  (n : SZ.t{SZ.fits (m * n) /\ SZ.fits (r * (m * n))})
  (i : szlt r) (j : szlt m) (k : szlt n)
  : Lemma (
      (l3_batched_row_major r m n).imap.f
        (SZ.v i, (SZ.v j, (SZ.v k, ()))) ==
      SZ.v (
        SZ.add (SZ.mul i (SZ.mul m n)) (SZ.add (SZ.mul j n) k)))
  = ()
#pop-options

#push-options "--z3rlimit 80"
inline_for_extraction noextract
instance c_l3_batched_col_major
  (r : erased nat{SZ.fits r})
  (m : SZ.t{SZ.fits (r * m)})
  (n : SZ.t{SZ.fits (m * n) /\ SZ.fits (r * (m * n))})
  : T.ctlayout (l3_batched_col_major r m n) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Shape.conc (r @| m @| n @| INil)) ->
              match idx with
              | (i, (j, (k, ()))) ->
                SZ.add (SZ.mul i (SZ.mul m n)) (SZ.add (SZ.mul k m) j))
  }
#pop-options

#push-options "--fuel 2 --ifuel 4 --z3rlimit 80"
let l3_batched_col_major_imap
  (r : erased nat{SZ.fits r})
  (m : SZ.t{SZ.fits (r * m)})
  (n : SZ.t{SZ.fits (m * n) /\ SZ.fits (r * (m * n))})
  (i : szlt r) (j : szlt m) (k : szlt n)
  : Lemma (
      (l3_batched_col_major r m n).imap.f
        (SZ.v i, (SZ.v j, (SZ.v k, ()))) ==
      SZ.v (
        SZ.add (SZ.mul i (SZ.mul m n)) (SZ.add (SZ.mul k m) j)))
  = ()
#pop-options

(* One "digit" of a mixed-radix (row-major) offset: if [a < b] and the
   already-accumulated offset [rest] is below the weight [q], then
   [a * q + rest < b * q].  *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let mul_bound_step (q a b rest : nat)
  : Lemma (requires a < b /\ rest < q)
          (ensures 0 <= a * q /\ a * q + rest < b * q)
  = FStar.Math.Lemmas.nat_times_nat_is_nat a q;
    FStar.Math.Lemmas.lemma_mult_le_right q (a + 1) b;
    FStar.Math.Lemmas.distributivity_add_left a 1 q

(* The rank-4 row-major offset is below the total size.  All the [fits] side
   conditions of the multiplications and additions in [c_l4_batched_row_major]
   below follow from this (together with [SZ.fits (r1 * (r2 * (m * n)))]).
   Now that F* emits one SMT query per proof obligation, Z3 no longer finds
   this nonlinear monotonicity chain on its own, so we spell it out and hand
   every intermediate bound to each leaf query. *)
let l4_row_major_offset_bound (r1 r2 m n i j k l : nat)
  : Lemma (requires i < r1 /\ j < r2 /\ k < m /\ l < n)
          (ensures
            0 <= k * n /\
            k * n + l < m * n /\
            0 <= j * (m * n) /\
            j * (m * n) + (k * n + l) < r2 * (m * n) /\
            0 <= i * (r2 * (m * n)) /\
            i * (r2 * (m * n)) + (j * (m * n) + (k * n + l))
              < r1 * (r2 * (m * n)))
  = FStar.Math.Lemmas.nat_times_nat_is_nat m n;
    FStar.Math.Lemmas.nat_times_nat_is_nat r2 (m * n);
    mul_bound_step n k m l;
    mul_bound_step (m * n) j r2 (k * n + l);
    mul_bound_step (r2 * (m * n)) i r1 (j * (m * n) + (k * n + l))
#pop-options

#push-options "--z3rlimit 80"
inline_for_extraction noextract
instance c_l4_batched_row_major
  (r1: erased nat{SZ.fits r1})
  (r2: SZ.t{SZ.fits (r1 * r2)})
  (m : SZ.t{SZ.fits (r2 * m) /\ SZ.fits (r1 * (r2 * m))})
  (n : SZ.t{SZ.fits (m * n) /\ SZ.fits (r2 * (m * n)) /\ SZ.fits (r1 * (r2 * (m * n)))})
  : T.ctlayout (l4_batched_row_major r1 r2 m n) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Shape.conc (r1 @| r2 @| m @| n @| INil)) ->
              match idx with
              | (i, (j, (k, (l, ())))) ->
                l4_row_major_offset_bound r1 (SZ.v r2) (SZ.v m) (SZ.v n)
                                          (SZ.v i) (SZ.v j) (SZ.v k) (SZ.v l);
                SZ.add (SZ.mul i (SZ.mul r2 (SZ.mul m n))) (SZ.add (SZ.mul j (SZ.mul m n)) (SZ.add (SZ.mul k n) l)))
  }
#pop-options

#push-options "--fuel 2 --ifuel 2 --z3rlimit 80"
let l4_batched_row_major_imap
  (r1: erased nat{SZ.fits r1})
  (r2: SZ.t{SZ.fits (r1 * r2)})
  (m : SZ.t{SZ.fits (r2 * m) /\ SZ.fits (r1 * (r2 * m))})
  (n : SZ.t{
    SZ.fits (m * n) /\
    SZ.fits (r2 * (m * n)) /\
    SZ.fits (r1 * (r2 * (m * n)))})
  (i : szlt r1) (j : szlt r2) (k : szlt m) (l : szlt n)
  : Lemma (
      (l4_batched_row_major r1 r2 m n).imap.f
        (SZ.v i, (SZ.v j, (SZ.v k, (SZ.v l, ())))) ==
      SZ.v (
        SZ.add (SZ.mul i (SZ.mul r2 (SZ.mul m n)))
          (SZ.add (SZ.mul j (SZ.mul m n)) (SZ.add (SZ.mul k n) l))))
  = ()
#pop-options
