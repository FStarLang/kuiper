module Kuiper.Matrix.Reprs
#lang-pulse

open Kuiper
open Kuiper.Bijection
open FStar.SizeT { (/^), (%^), (+^), (-^), ( *^ )  }

inline_for_extraction noextract
instance crepr_row_major : crepr row_major = {
  map = fun rows cols -> {
      m_len  = rows *^ cols;
      m_cols = cols;
      m_rows = rows;
      c_to    = (fun i j -> i *^ cols +^ j);
  }
}

inline_for_extraction noextract
instance crepr_col_major : crepr col_major = {
  map = fun rows cols -> {
      m_len  = rows *^ cols;
      m_cols = cols;
      m_rows = rows;
      c_to    = (fun i j -> j *^ rows +^ i);
  }
}

inline_for_extraction noextract
instance crepr_row_major_mirror : crepr row_major_mirror = {
  map = fun rows cols -> {
      m_len  = rows *^ cols;
      m_cols = cols;
      m_rows = rows;
      c_to    = (fun i j -> rows *^ cols -^ 1sz -^ i *^ cols -^ j);
  }
}
