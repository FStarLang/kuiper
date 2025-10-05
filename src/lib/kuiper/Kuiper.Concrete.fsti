module Kuiper.Concrete

open Kuiper.SizeT
module SZ = Kuiper.SizeT
open FStar.Ghost
open FStar.SizeT
open FStar.Mul

// class concretizable_as (st mt : Type) (s : st) = {
//   evidence : mt;
// }

// instance _ : concretizable_as int SZ.t 42 = {
//   evidence = 42sz;
// }

(* Can this be a reasonable general solution? *)
inline_for_extraction noextract
class concrete_sz (n : int) = {
  x : (x : SZ.t {SZ.v x == reveal n});
}

inline_for_extraction noextract
let concr (x : erased int) {| d:concrete_sz x |} : sz =
  match d with
  | {x} -> x

inline_for_extraction noextract
let concr' (#x : erased int) (d:concrete_sz x) : sz =
  concr x #d

inline_for_extraction noextract instance _ : concrete_sz 0 = { x = 0sz; }
inline_for_extraction noextract instance _ : concrete_sz 1 = { x = 1sz; }
inline_for_extraction noextract instance _ : concrete_sz 2 = { x = 2sz; }
inline_for_extraction noextract instance _ : concrete_sz 4 = { x = 4sz; }
inline_for_extraction noextract instance _ : concrete_sz 8 = { x = 8sz; }
inline_for_extraction noextract instance _ : concrete_sz 16 = { x = 16sz; }
inline_for_extraction noextract instance _ : concrete_sz 32 = { x = 32sz; }
inline_for_extraction noextract instance _ : concrete_sz 64 = { x = 64sz; }
inline_for_extraction noextract instance _ : concrete_sz 128 = { x = 128sz; }

inline_for_extraction noextract
instance concrete_sz_sz (x : SZ.t) : concrete_sz (SZ.v x) = { x; }

// inline_for_extraction noextract
// instance concrete_sz_erased (x : erased SZ.t) (d : concrete_sz (reveal x)) : concrete_sz (SZ.v x) = { x = concr (hide (SZ.v x)) #d; }

inline_for_extraction noextract
instance concrete_sz_mul (x y : int)
  {| xx : concrete_sz x, yy : concrete_sz y |}
  (#_ : squash (FStar.SizeT.fits (x * y)))
  : concrete_sz (x * y) = {
    x = xx.x *^ yy.x;
  }

inline_for_extraction noextract
instance concrete_sz_div (x y : int { y =!= 0 })
  {| xx : concrete_sz x, yy : concrete_sz y |}
  : concrete_sz (x / y) = {
    x = xx.x /^ yy.x;
  }

inline_for_extraction noextract
instance concrete_sz_rem (x y : int { y =!= 0 })
  {| xx : concrete_sz x, yy : concrete_sz y |}
  : concrete_sz (x % y) = {
    x = xx.x %^ yy.x;
  }
