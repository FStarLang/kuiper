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

inline_for_extraction noextract
type dist = u16

[@@CPrologue "__device__"]
inline_for_extraction noextract
let mindist (x : dist) (y : dist) : dist =
  let open FStar.UInt16 in
  if x `lt` y then x else y

[@@CPrologue "__device__"]
let add (x y : dist) : dist =
  if UInt16.eq x 0us then y
  else if UInt16.eq y 0us then x
  else mindist x y

[@@CPrologue "__device__"]
let add' (x y : dist) : d:dist{d == add x y} =
  if UInt16.eq x 0us || (not (UInt16.eq y 0us) && UInt16.lt y x)
  then y
  else x

[@@CPrologue "__device__"]
let mult x y : dist =
  if UInt16.eq x 0us || UInt16.eq y 0us then 0us
  else x `Scalars.add` y

inline_for_extraction noextract
instance scalar_dist : scalar dist = {
  is_sized = { size = 2sz };
  zero = 0us;
  one = 1us;
  add = add;
  mul = mult;
}

// let mult_dist = P.matmul_gpu #dist #scalar_dist

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
  assume (a |-> ea); (* assume another one... cannot call matmul on a fraction, yet. *)

  P.matmul_gpu #dist #scalar_dist add' a a b;

  drop_ (a |-> ea);
}
