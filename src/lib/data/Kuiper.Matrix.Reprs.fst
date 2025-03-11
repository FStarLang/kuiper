module Kuiper.Matrix.Reprs
#lang-pulse

open Kuiper
open Kuiper.Bijection
module SZ = FStar.SizeT
open FStar.SizeT { div as (/^), (%^), (+^), (-^), ( *^ )  }

inline_for_extraction noextract
instance crepr_row_major : crepr row_major = {
  map = fun rows cols -> {
      c_to    = (fun i j -> i *^ cols +^ j);
      c_from1 = (fun idx -> idx /^ cols);
      c_from2 = (fun idx -> idx %^ cols);
  }
}

inline_for_extraction noextract
instance crepr_col_major : crepr col_major = {
  map = fun rows cols -> {
      c_to    = (fun i j -> j *^ rows +^ i);
      c_from1 = (fun idx -> idx %^ rows);
      c_from2 = (fun idx -> idx /^ rows);
  }
}

inline_for_extraction noextract
instance crepr_row_major_mirror : crepr row_major_mirror = {
  map = fun rows cols -> {
      c_to    = (fun i j -> rows *^ cols -^ 1sz -^ i *^ cols -^ j);
      c_from1 = (fun idx -> rows -^ 1sz -^ idx /^ cols);
      c_from2 = (fun idx -> cols -^ 1sz -^ idx %^ cols);
  }
}
