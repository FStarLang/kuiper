module Kuiper.Matrix.Reprs
#lang-pulse

open Kuiper
open Kuiper.Bijection
include Kuiper.Matrix.Reprs.Type

let row_major : mrepr =
  fun rows cols -> {
    len = rows * cols;
    map = inj_bij bij_nat_prod;
  }

inline_for_extraction noextract
instance val crepr_row_major : crepr row_major

let col_major : mrepr =
  fun rows cols -> {
    len = rows * cols;
    map = inj_bij (bij_flip `bij_comp` bij_nat_prod #cols #rows);
  }

inline_for_extraction noextract
instance val crepr_col_major : crepr col_major

let bij_mirror (#rows #cols : nat) : (natlt rows & natlt cols =~ natlt rows & natlt cols) =
  Mkbijection
   #(natlt rows & natlt cols) #(natlt rows & natlt cols)
   (fun (x, y) -> (rows-1-x, cols-1-y))
   (fun (x, y) -> (rows-1-x, cols-1-y))
   ez
   ez

let row_major_mirror : mrepr =
  fun rows cols -> {
    len = rows * cols;
    map = inj_bij (bij_mirror `bij_comp` bij_nat_prod);
  }

inline_for_extraction noextract
instance val crepr_row_major_mirror : crepr row_major_mirror

#push-options "--z3rlimit_factor 4"
inline_for_extraction noextract
instance strided_row_major_base (#rows #cols : erased nat)
  {| d : concrete_sz cols |}
  : strided_row_major (row_major rows cols) =
{
  offset = 0sz;
  stride = concr' d;
  pf = ez;
}

inline_for_extraction noextract
instance strided_col_major_base (#rows #cols : erased nat)
  {| d : concrete_sz rows |}
  : strided_col_major (col_major rows cols) =
{
  offset = 0sz;
  stride = concr' d;
  pf = ez;
}
#pop-options