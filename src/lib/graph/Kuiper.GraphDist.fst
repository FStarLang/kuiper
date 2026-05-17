module Kuiper.GraphDist

#lang-pulse
open Kuiper
open Kuiper.Scalars
module K = Kuiper.Kernel.GEMM.Naive2
open Kuiper.Array2
open Kuiper.Tensor.Layout.Alg
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

inline_for_extraction noextract
let mindist (x y : dist) : GTot dist =
  let open FStar.UInt16 in
  if x.v `lt` y.v then x else y

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
instance scalar_dist : scalar dist =
  {
    is_sized = { size = 2sz; default = D 0us };
    zero = D 0us;
    one = D 1us;
    add = add';
    mul = mult;
    lt  = (fun _ _ -> false); // fake, not used by GEMM
    lte = (fun _ _ -> false); // fake, not used by GEMM
    eq  = (fun _ _ -> false); // fake, not used by GEMM
  }

ghost
fn m_share_2
  (#et : Type)
  (#m #n : nat)
  (#l : layout m n)
  (a : array2 et l)
  (#em : ematrix et m n)
  requires
    a |-> em
  ensures
    a |-> Frac 0.5R em ** a |-> Frac 0.5R em
{
  share_n a 2;
  forevery_natlt_pop _ _;
  forevery_natlt_pop _ _;
  forevery_elim_empty _;
}

ghost
fn m_gather_2
  (#et : Type)
  (#m #n : nat)
  (#l : layout m n)
  (a : array2 et l)
  (#em : ematrix et m n)
  requires
    a |-> Frac 0.5R em ** a |-> Frac 0.5R em
  ensures
    a |-> em
{
  forevery_intro_empty #(natlt 0) (fun _ -> a |-> Frac (1.0R /. 2) em);
  forevery_natlt_push 1 _;
  forevery_natlt_push 2 _;
  gather_n a 2;
  ()
}

fn matmul_dist_gpu
  (#size : szp)
  (a : array2 dist (l2_row_major size size) { is_global a })
  (b : array2 dist (l2_row_major size size) { is_global b })
  (#ea : ematrix dist size size)
  (#eb : ematrix dist size size)
  preserves
    cpu ** on gpu_loc (a |-> ea)
  requires
    pure (size * size <= max_blocks) **
    on gpu_loc (b |-> eb)
  ensures
    exists* eb'. on gpu_loc (b |-> eb')
{
  map_loc gpu_loc (fun () -> m_share_2 a);
  K.mmcomb_gpu_exact add' a a b;
  map_loc gpu_loc (fun () -> m_gather_2 a);
}
