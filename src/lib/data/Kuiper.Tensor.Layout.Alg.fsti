module Kuiper.Tensor.Layout.Alg

(* Constructing tensor layouts algebraically. *)

open Kuiper
open Kuiper.Injection
open Kuiper.Index
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Layout
type layout_f_for (#n:nat) (d : idesc n) =
  abs d @~> natlt (sizeof d)

let lunit : layout_f_for INil =
  mk_injection #_ #(natlt 1) (fun () -> 0) ez

(* From a layout_f to a full layout *)
let pack (#n:nat) (#d:idesc n) (f : layout_f_for d) : full_tlayout d =
  let ulen = sizeof d in
  let imap = f in
  { ulen; imap }

let major_on_f (#n:nat)
  (i : natlt (n+1))
  (k : nat)
  (#d : idesc n)
  (sub : layout_f_for d)
  : abs (insert_i i k d) -> GTot (natlt (sizeof (insert_i i k d)))
  = fun (idx : abs (insert_i i k d)) ->
      let maj, min = (abs_bring_forward_bij i (insert_i i k d)).ff idx in
      maj * sizeof d + sub.f min

(* Hiding the injection proof ...  but revealing the function *)
val major_on (#n:nat)
  (i : natlt (n+1))
  (k : nat)
  (#d : idesc n)
  (sub : layout_f_for d)
  : l : layout_f_for (insert_i i k d)
     { l.f == major_on_f i k sub }

(* Some examples *)

let l1_forward (m : nat) : tlayout (m @| INil) =
  pack <|
  major_on 0 m <|
  lunit

// FIXME: SZ.t -> nat ?
let l2_row_major (m n : nat) : tlayout (m @| n @| INil) =
  pack <|
  major_on 0 m <|
  major_on 0 n <|
  lunit

let l2_col_major (m n : nat) : tlayout (m @| n @| INil) =
  pack <|
  major_on 1 n <|
  major_on 0 m <|
  lunit

let l3_batched_row_major (r m n : nat) : tlayout (r @| m @| n @| INil) =
  pack <|
  major_on 0 r <|
  major_on 0 m <|
  major_on 0 n <|
  lunit

let l3_batched_col_major (r m n : nat) : tlayout (r @| m @| n @| INil) =
  pack <|
  major_on 0 r <|
  major_on 1 n <|
  major_on 0 m <|
  lunit

(* Constructing a concrete size for a given description.
TODO: use concrete_sz. *)
inline_for_extraction noextract
class csizeof (#n : erased nat) (d : idesc n) =
  {
    v : (v : SZ.t{SZ.v v == sizeof d});
  }

inline_for_extraction noextract
instance val csizeof_INil : csizeof INil

inline_for_extraction noextract
instance val csizeof_ICons
  (#n : erased nat)
  (d0 : SZ.t)
  (d1 : idesc n)
  (c_d1 : csizeof d1)
  (#_ : squash (SZ.fits (d0 * c_d1.v)))
  : csizeof (ICons d0 d1)

inline_for_extraction noextract
instance val csizeof_insert_i
  (#n : erased nat)
  (i : erased nat{i < n+1})
  (k : sz)
  (d : idesc n)
  (c_d : csizeof d)
  (#_ : squash (SZ.fits (k * c_d.v)))
  : csizeof (insert_i i k d)

(* Constructing a concrete function for a given ghost injection. *)
inline_for_extraction noextract
class auto_cinj (#n : erased nat) (#d : erased (idesc n)) (#k : erased nat)
  (f : abs d @~> natlt k) =
  {
    ff : (x:conc d -> y:SZ.t{SZ.v y == f.f (up x)});
  }

inline_for_extraction noextract
instance val cunit : auto_cinj lunit

inline_for_extraction noextract
instance val c_major_on
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : idesc n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (#_ : squash (SZ.fits (k * sizeof d)))
  (c_sub : auto_cinj sub)
  : auto_cinj (major_on i k sub)

(* Constructing a ctlayout automatically if we can concretize the size and the
injection.  FIXME: this does not seem to work. *)
inline_for_extraction noextract
instance val c_pack (#n : erased nat) (#d: idesc n)
  (#f : layout_f_for d) (c_f : auto_cinj f)
  (#_ : squash (SZ.fits (sizeof d)))
  (#_ : squash (all_fit d))
  : ctlayout (pack f)

(* A helper.... should not be needed. *)
inline_for_extraction noextract
instance c_major_on_i_0'
  (#n: erased nat{n > 0})
  (k : erased nat)
  (#d : idesc (n-1))
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (#_ : squash (SZ.fits (k * sizeof d)))
  (c_sub : auto_cinj sub)
  : auto_cinj #n (major_on #(n-1) 0 k sub) =
  c_major_on 0sz k c_sub

#push-options "--z3rlimit 40"
module T = Kuiper.Tensor

// This should just work from instances in Tensor.Layout.Alg

inline_for_extraction noextract
instance c_l1_forward (m : erased nat{SZ.fits m}) : T.ctlayout (l1_forward m) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Index.conc (m @| INil)) ->
              match idx with
              | (i, ()) -> i);
  }

inline_for_extraction noextract
instance c_l2_row_major (m : erased nat{SZ.fits m}) (n : SZ.t{SZ.fits (m * n)}) : T.ctlayout (l2_row_major m n) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Index.conc (m @| n @| INil)) ->
              match idx with
              | (i, (j, ())) -> SZ.add (SZ.mul i n) j)
  }

inline_for_extraction noextract
instance c_l2_col_major (m : sz) (n : erased nat{SZ.fits n /\ SZ.fits (m * n)}) : T.ctlayout (l2_col_major m n) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx : Kuiper.Index.conc (m @| n @| INil)) ->
              match idx with
              | (i, (j, ())) -> SZ.add (SZ.mul j m) i)
  }
#pop-options
