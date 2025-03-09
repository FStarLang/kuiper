module Kuiper.Matrix.Reprs
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.Matrix.Poly
module SZ = FStar.SizeT

(* Explicit constructor, helps with figuring out what is erased or not. *)
inline_for_extraction
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

let bij_row_major (rows cols : nat)
  : ((natlt rows & natlt cols) =~ natlt (rows * cols))
  = bij_nat_prod

let row_major : mrepr =
  fun #rows #cols ->
    { bij = bij_row_major rows cols }

inline_for_extraction
let c_row_major : crepr row_major =
  fun rows cols ->
    assume (SZ.fits (rows * cols)); (* state and carry around *)
    let open FStar.SizeT in
    mk_clayout #_ #_ (row_major #(SZ.v rows) #(SZ.v cols))
      (fun i j -> i *^ cols +^ j)
      (fun idx -> idx `div` cols)
      (fun idx -> idx %^ cols)

let bij_col_major (rows cols : nat)
  : ((natlt rows & natlt cols) =~ natlt (rows * cols))
  = bij_flip `bij_comp` bij_nat_prod

let col_major : mrepr =
  fun #rows #cols ->
    { bij = bij_col_major rows cols }

inline_for_extraction
let c_col_major : crepr col_major =
  fun rows cols ->
    assume (SZ.fits (rows * cols)); (* state and carry around *)
    let open FStar.SizeT in
    mk_clayout #_ #_ (col_major #(SZ.v rows) #(SZ.v cols))
      (fun i j -> j *^ rows +^ i)
      (fun idx -> idx %^ rows)
      (fun idx -> idx `div` rows)
