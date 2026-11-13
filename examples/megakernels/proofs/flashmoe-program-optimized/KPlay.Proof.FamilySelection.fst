module KPlay.Proof.FamilySelection

open Kuiper

module S = FStar.Seq
module F = Kuiper.Floating
module I32 = FStar.Int32
module Steps = KPlay.Proof.RouteSteps
module Family = KPlay.Proof.FamilyArgmax
module Shared = KPlay.Proof.Shared.Argmax

inline_for_extraction noextract
let rec scan_prefix_receipt (row:Steps.probability_row) (n:nat{n <= 8})
  : Lemma
      (ensures Family.prefix_result row n
        (I32.v (Steps.scan_prefix row n).chosen) (Steps.scan_prefix row n).best)
      (decreases n)
  =
  if n = 0 then Family.prefix_initial row
  else (
    scan_prefix_receipt row (n - 1);
    let before = Steps.scan_prefix row (n - 1) in
    if F.lt before.best (S.index row (n - 1)) then
      Family.prefix_update row (n - 1) (I32.v before.chosen) before.best
    else Family.prefix_skip row (n - 1) (I32.v before.chosen) before.best
  )

inline_for_extraction noextract
let source_selection (row:Steps.probability_row) (selected:i32) (best:f32)
  : Lemma
      (requires selected == (Steps.scan_prefix row 8).chosen /\
        best == (Steps.scan_prefix row 8).best)
      (ensures Family.row_result row selected best /\
        I32.v selected == Shared.top1_index (Family.row_input row) /\
        best == S.index row (Shared.top1_index (Family.row_input row)))
  =
  scan_prefix_receipt row 8;
  Family.prefix_complete row (I32.v selected) selected best;
  Family.row_refines row selected best
