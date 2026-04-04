module Kuiper.TensorLayout

open Kuiper
open Kuiper.Injection
open Kuiper.Index
open Kuiper.Chest
open FStar.Tactics.Typeclasses { no_method }
module V = Kuiper.View
module SZ = Kuiper.SizeT

[@@erasable]
noeq
type tlayout (#r : erased nat) (d : idesc r) = {
  (* Underlying length of base array (Kuiper.Array) *)
  ulen : nat;
  (* Injection from (abstract) index space into base array. *)
  imap : abs d @~> natlt ulen;
}

(* Alias for .ulen *)
let tlayout_size (#d : idesc 'r) (l : tlayout d) : GTot nat = l.ulen

inline_for_extraction
class ctlayout (#r : erased nat) (#d : idesc r) (l : tlayout d) = {
  [@@@no_method]
  culen : (x : SZ.t { SZ.v x == l.ulen });

  [@@@no_method]
  all_fit : squash (all_fit d);

  [@@@no_method]
  cimap : i:conc d -> r:SZ.t{SZ.v r == l.imap.f (up i)};
}

let tensor_aview (et : Type) (#r : nat) (#d : idesc r) (l : tlayout d)
  : V.aview et (chest d et)
  = {
      iview = {
        len = l.ulen;
        ait = abs d;
        step = { imap = l.imap; };
      };
      ctn = solve;
    }

let lunit : (abs INil @~> natlt 1) =
  mk_injection #(abs INil) #(natlt 1) (fun () -> 0) ez

(* Constructing tensor layouts. *)

type layout_f_for (#n:nat) (d : idesc n) =
  abs d @~> natlt (sizeof d)

let pack (#n:nat) (#d:idesc n) (f : layout_f_for d) : tlayout d =
  let ulen = sizeof d in
  let imap = f in
  { ulen; imap }

let g_grouped_by_f (#n:nat)
  (i : natlt (n+1))
  (k : nat)
  (#d : idesc n)
  (sub : layout_f_for d)
  : abs (insert_i i k d) -> GTot (natlt (sizeof (insert_i i k d)))
  = fun (idx : abs (insert_i i k d)) ->
      modulo_insert i k d;
      let maj, min = (abs_bring_forward_bij i (insert_i i k d)).ff idx in
      let sub_i : natlt (sizeof d) = sub.f min in
      let offset = maj * sizeof d in
      offset + sub_i

#push-options "--z3rlimit 20"
let g_grouped_by (#n:nat)
  (i : natlt (n+1))
  (k : nat)
  (#d : idesc n)
  (sub : layout_f_for d)
  : layout_f_for (insert_i i k d)
  = {
    f = g_grouped_by_f i k sub;
    is_inj = ez;
  }
#pop-options

let row_major' (m n : SZ.t) : tlayout (m @| n @| INil) =
  pack <|
  g_grouped_by 0 m <|
  g_grouped_by 0 n <|
  lunit

let col_major' (m n : SZ.t) : tlayout (m @| n @| INil) =
  pack <|
  g_grouped_by 1 n <|
  g_grouped_by 0 m <|
  lunit

let batched_row_major' (r m n : SZ.t) : tlayout (r @| m @| n @| INil) =
  pack <|
  g_grouped_by 0 r <|
  g_grouped_by 0 m <|
  g_grouped_by 0 n <|
  lunit

let batched_col_major' (r m n : SZ.t) : tlayout (r @| m @| n @| INil) =
  pack <|
  g_grouped_by 0 r <|
  g_grouped_by 1 n <|
  g_grouped_by 0 m <|
  lunit

inline_for_extraction noextract
class auto_cinj (#n : erased nat) (#d : erased (idesc n)) (#k : erased nat)
  (f : abs d @~> natlt k) =
  {
    ff : (x:conc d -> y:SZ.t{SZ.v y == f.f (up x)});
  }

inline_for_extraction noextract
class csizeof (#n : erased nat) (d : (idesc n)) =
  {
    v : (v : SZ.t{SZ.v v == sizeof d});
  }

inline_for_extraction noextract
instance csizeof_INil : csizeof INil = {
  v = 1sz;
}

inline_for_extraction noextract
instance csizeof_ICons
  (#n : erased nat)
  (d0 : SZ.t)
  (d1 : idesc n)
  (c_d1 : csizeof d1)
  (#_ : squash (SZ.fits (d0 * c_d1.v)))
  : csizeof (ICons d0 d1) =
  {
    v = SZ.mul d0 c_d1.v;
  }

inline_for_extraction noextract
instance cunit : auto_cinj lunit = {
  ff = (fun _ -> 0sz);
}

#restart-solver // work around crash
inline_for_extraction noextract
let c_grouped_by_f
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : idesc n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (c_sub : auto_cinj sub)
  (idx:conc (insert_i i k d))
  : szlt (sizeof (insert_i i k d))
  = modulo_insert i k d;
    let maj, min = c_bring_forward_ff i (insert_i i k d) idx in
    let { ff } = c_sub in
    let sub_i : szlt (sizeof d) = ff min in
    assume (SZ.fits (maj * cs.v));
    let offset : SZ.t = maj *^ cs.v in
    assume (SZ.fits (offset + sub_i));
    let r = offset +^ sub_i in
    r

let lem_c_grouped_by_f_ok
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : idesc n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (c_sub : auto_cinj sub)
  (idx:conc (insert_i i k d))
  : Lemma (SZ.v (c_grouped_by_f i k c_sub idx) == g_grouped_by_f i k sub (up idx))
          [SMTPat (c_grouped_by_f i k c_sub idx)]
  = let amaj, amin = (abs_bring_forward_bij i (insert_i i k d)).ff (up idx) in
    let maj, min = c_bring_forward_ff i (insert_i i k d) idx in
    assume (SZ.fits (maj * cs.v));
    lemma_c_bring_forward_ff_ok i (insert_i i k d) idx;
    bring_forward_commute2 i (insert_i i k d) maj min;
    assert (SZ.v maj == amaj);
    let asub_i : natlt (sizeof d) = sub.f amin in
    let { ff } = c_sub in
    let sub_i = ff min in
    let aoffset = amaj * sizeof d in
    assume (SZ.fits (aoffset + sub_i));
    let offset = maj *^ cs.v in
    ()

inline_for_extraction noextract
instance c_grouped_by
  (#n: erased nat)
  (i : szlt (n+1))
  (k : erased nat)
  (#d : idesc n)
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (c_sub : auto_cinj sub)
  : auto_cinj (g_grouped_by i k sub) =
  {
    ff = (fun idx -> c_grouped_by_f i k c_sub idx);
  }

inline_for_extraction noextract
instance close (#n : erased nat) (#d: idesc n) (f : layout_f_for d) (c_f : auto_cinj f)
  (#_ : squash (all_fit d))
  : ctlayout (pack f) =
  {
    culen   = magic();
    all_fit = ();
    cimap   = c_f.ff;
  }


inline_for_extraction noextract
instance c_grouped_by_concrete_i_0'
  (#n: erased nat{n > 0})
  (k : erased nat)
  (#d : idesc (n-1))
  {| cs : csizeof d |}
  (#sub : layout_f_for d)
  (c_sub : auto_cinj sub)
  : auto_cinj #n (g_grouped_by #(n-1) 0 k sub) =
  c_grouped_by 0sz k c_sub