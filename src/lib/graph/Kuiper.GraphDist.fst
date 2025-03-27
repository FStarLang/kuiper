module Kuiper.GraphDist

#lang-pulse
open Kuiper
open Kuiper.Scalars
module P = Kuiper.Poly.GEMM.Naive2 (* cannot use tiled ones without making dimensions a multiple of tile size *)
module R = Kuiper.Matrix.Reprs
module M = Kuiper.Matrix
open Kuiper.EMatrix { ematrix }

// inline_for_extraction noextract
// class ord_raw (a : Type) = {
//   cmp : a -> a -> Order.order;
// }

// inline_for_extraction noextract
// instance ord_raw_u16 : ord_raw u16 = {
//   cmp = (fun x y -> let open FStar.UInt16 in
//     if x <^ y then Order.Lt
//     else if x >^ y then Order.Gt
//     else Order.Eq);
// }

// inline_for_extraction noextract
// let ( < ) (#t:Type) {| ord_raw t |} (x : t) (y : t) : bool =
//   match  cmp x y with
//   | Order.Lt -> true
  // | _ -> false
  // Order.Lt? (cmp x y)

// inline_for_extraction noextract
// let min (#t:Type) {| ord_raw t |} (x : t) (y : t) : t =
//   if x < y then x else y


(* 0 is to be interpreted as "no distance" == infinity.
Every other number is a concrete distance. *)

inline_for_extraction
type dist = | D : v:u16 -> dist

// [@@CPrologue "__device__"]
inline_for_extraction noextract
let mindist (x y : dist) : GTot dist =
  let open FStar.UInt16 in
  if x.v `lt` y.v then x else y

// [@@CPrologue "__device__"]
let add (x y : dist) : GTot dist =
  if UInt16.eq x.v 0us then y
  else if UInt16.eq y.v 0us then x
  else mindist x y

[@@CPrologue "__device__"; "KrmlPrivate"]
let add' (x y : dist) : d:dist{d == add x y} =
  if UInt16.eq x.v 0us || (not (UInt16.eq y.v 0us) && UInt16.lt y.v x.v)
  then y
  else x

[@@CPrologue "__device__"; "KrmlPrivate"]
let mult (x y : dist) : dist =
  if UInt16.eq x.v 0us || UInt16.eq y.v 0us then D 0us
  else D (x.v `Scalars.add` y.v)

inline_for_extraction noextract
instance scalar_dist : scalar dist = {
  is_sized = { size = 2sz };
  zero = D 0us;
  one = D 1us;
  add = add';
  mul = mult;
}

// let mult_dist = P.mmcomb_gpu #dist #scalar_dist

fn matmul_dist_gpu
  (#size : szp)
  (a : M.gpu_matrix dist (R.row_major size size))
  (b : M.gpu_matrix dist (R.row_major size size))
  (#ea : ematrix dist size size)
  (#eb : ematrix dist size size)
  preserves
    cpu ** (a |-> ea)
  requires
    pure (size * size <= max_blocks) **
    (b |-> eb)
  ensures
    exists* eb'. b |-> eb'
{
  assert (a |-> ea);
  M.gpu_matrix_share_2 a;

  P.mmcomb_gpu add' a a b;

  M.gpu_matrix_gather_2 a;
}
