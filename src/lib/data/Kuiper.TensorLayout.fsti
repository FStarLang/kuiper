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

#push-options "--z3rlimit 20"
let g_grouped_by (#n:nat)
  (i : natlt (n+1))
  (k : nat)
  (#d : idesc n)
  (sub : layout_f_for d)
  : layout_f_for (insert_i i k d)
  =
  mk_injection #(abs (insert_i i k d)) #(natlt (sizeof (insert_i i k d)))
    (fun (idx : abs (insert_i i k d)) ->
      modulo_insert i k d;
      let maj, min = (abs_bring_forward_bij i (insert_i i k d)).ff idx in
      let sub_i : natlt (sizeof d) = sub.f min in
      let offset = maj * sizeof d in
      offset + sub_i)
    (fun _ _ -> ())
#pop-options

let row_major' (m n : SZ.t) : tlayout (m @| n @| INil) =
  pack <|
  g_grouped_by 0 m  <|
  g_grouped_by 0 n  <|
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

(* Another approach, with worse type inference. *)

// #push-options "--z3rlimit 20"
// let f_grouped_by (#n:nat)
//   (#d : idesc n)
//   (i : natlt n)
//   (sub : abs (modulo_i i d) @~> natlt (sizeof (modulo_i i d)))
//   : (abs d @~> natlt (sizeof d))
//   =
//   mk_injection #(abs d) #(natlt (sizeof d))
//     (fun (idx : abs d) ->
//       let maj, min = (abs_bring_forward_bij i d).ff idx in
//       let sub_i : natlt (sizeof (modulo_i i d)) = sub.f min in
//       let offset = maj * sizeof (modulo_i i d) in
//       offset + sub_i)
//     (fun _ _ -> ())
// #pop-options

// let row_major (m n : nat) : (abs (m @| n @| INil) @~> natlt (m*n)) =
//   f_grouped_by #2 #(m @| n @| INil) 0 <|
//   f_grouped_by #1 #(n @| INil)            0 <|
//   lunit

// let col_major (m n : nat) : (abs (m @| n @| INil) @~> natlt (m*n)) =
//   f_grouped_by #2 #(m @| n @| INil) 1 <|
//   f_grouped_by #1 #(m @| INil)            0 <|
//   lunit

// let batched_row_major (r m n : nat) :
//   (abs (r @| m @| n @| INil) @~> natlt (r*m*n))
// =
//   f_grouped_by #_ #(r @| m @| n @| INil) 0 <|
//   f_grouped_by #_ #(m @| n @| INil)            0 <|
//   f_grouped_by #_ #(n @| INil)                       0 <|
//   lunit
