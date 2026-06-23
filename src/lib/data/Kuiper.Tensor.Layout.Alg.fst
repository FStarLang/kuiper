module Kuiper.Tensor.Layout.Alg

(* Constructing tensor layouts algebraically. *)
open Kuiper
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Chest
open FStar.Tactics.Typeclasses { no_method }
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Layout

#push-options "--z3rlimit 40"
let major_on (#n:nat)
  (i : natlt (n+1))
  (k : nat)
  (#d : shape n)
  (sub : layout_f_for d)
  : layout_f_for (insert_i i k d)
  = {
    f = major_on_f i k sub;
    is_inj = ez;
  }
#pop-options


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