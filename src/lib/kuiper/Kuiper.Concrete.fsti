module Kuiper.Concrete

open Kuiper.SizeT
module SZ = FStar.SizeT
open FStar.Ghost
open FStar.SizeT { (/^), (%^) }

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

inline_for_extraction noextract
instance concrete_sz_sz (x : SZ.t) : concrete_sz (SZ.v x) = { x; }

// inline_for_extraction noextract
// instance concrete_sz_erased (x : erased SZ.t) (d : concrete_sz (reveal x)) : concrete_sz (SZ.v x) = { x = concr (hide (SZ.v x)) #d; }

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
