module Kuiper.Matrix.Reprs
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.Matrix.Poly
module SZ = FStar.SizeT

(* Explicit constructor, helps with figuring out what is erased or not. *)
inline_for_extraction noextract
let mk_clayout (#rows #cols : _) (l : erased (mlayout rows cols))
  (c_to    : ((i:SZ.t{i < rows}) -> (j:SZ.t{j < cols}) -> r:SZ.t{SZ.v r == l.bij.ff (SZ.v i, SZ.v j)}))
  (c_from1 : ((idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == fst (l.bij.gg (SZ.v idx))}))
  (c_from2 : ((idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == snd (l.bij.gg (SZ.v idx))}))
  : clayout l =
  {
    c_to = c_to;
    c_from1 = c_from1;
    c_from2 = c_from2;
  }

inline_for_extraction
let row_major : mrepr =
  fun #rows #cols ->
    { bij = bij_nat_prod }

inline_for_extraction noextract
let clayout_row_major (rows : SZ.t) (cols : SZ.t{SZ.fits (rows * cols)}) : clayout (row_major #rows #cols) =
  let open FStar.SizeT in
    mk_clayout #_ #_ (row_major #(SZ.v rows) #(SZ.v cols))
      (fun i j -> i *^ cols +^ j)
      (fun idx -> idx `div` cols)
      (fun idx -> idx %^ cols)

inline_for_extraction noextract
instance crepr_row_major : crepr row_major = {
  map = clayout_row_major;
}

inline_for_extraction
let col_major : mrepr =
  fun #rows #cols ->
    { bij = bij_flip `bij_comp` bij_nat_prod #cols #rows }

inline_for_extraction noextract
let clayout_col_major (rows : SZ.t) (cols : SZ.t{SZ.fits (rows * cols)}) : clayout (col_major #rows #cols) =
  let open FStar.SizeT in
    mk_clayout #_ #_ (col_major #(SZ.v rows) #(SZ.v cols))
      (fun i j -> j *^ rows +^ i)
      (fun idx -> idx %^ rows)
      (fun idx -> idx `div` rows)

inline_for_extraction noextract
instance crepr_col_major : crepr col_major = {
  map = clayout_col_major;
}

let bij_mirror (#rows #cols : nat) : (natlt rows & natlt cols =~ natlt rows & natlt cols) =
  Mkbijection
   #(natlt rows & natlt cols) #(natlt rows & natlt cols)
   (fun (x, y) -> (rows-1-x, cols-1-y))
   (fun (x, y) -> (rows-1-x, cols-1-y))
   ez
   ez

inline_for_extraction
let row_major_mirror : mrepr =
  fun #rows #cols ->
    { bij = bij_mirror `bij_comp` bij_nat_prod }

inline_for_extraction noextract
let clayout_row_major_mirror (rows : SZ.t) (cols : SZ.t{SZ.fits (rows * cols)}) : clayout (row_major_mirror #rows #cols) =
  let open FStar.SizeT in
    mk_clayout #_ #_ (row_major_mirror #(SZ.v rows) #(SZ.v cols))
      (fun i j -> rows *^ cols -^ 1sz -^ i *^ cols -^ j)
      (fun idx -> rows -^ 1sz -^ idx `div` cols)
      (fun idx -> cols -^ 1sz -^ idx %^ cols)

inline_for_extraction noextract
instance crepr_row_major_mirror : crepr row_major_mirror = {
  map = clayout_row_major_mirror;
}
