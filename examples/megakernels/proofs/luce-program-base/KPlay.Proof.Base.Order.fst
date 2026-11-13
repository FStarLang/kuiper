module KPlay.Proof.Base.Order

module S = FStar.Seq
module F = Kuiper.Floating
module FP = Kuiper.Float32
module D = KPlay.Proof.Shared.Selection

(* These pure order definitions and lemmas retain the accepted argmax
   interface's bodies; only module/type aliases are changed. *)
inline_for_extraction noextract
let ordered (x:FP.t) : GTot bool = not (F.is_nan x)

inline_for_extraction noextract
let priority_le (x y:FP.t) : GTot bool =
  if F.is_nan x then F.is_nan y
  else F.is_nan y || not (F.lt y x)

inline_for_extraction noextract
let native_order (x y:FP.t)
  : Lemma (
      (F.lt x y ==> ordered x /\ ordered y) /\
      (F.lt y x ==> ordered x /\ ordered y) /\
      (F.ieee_eq x y ==> ordered x /\ ordered y) /\
      (F.ieee_eq x y <==> F.ieee_eq y x) /\
      (ordered x /\ ordered y ==>
        (F.lt x y <==> ~(F.lt y x \/ F.ieee_eq x y)) /\
        (F.ieee_eq x y <==> ~(F.lt x y \/ F.lt y x)))) =
  F.eq_spec x y;
  F.eq_spec y x;
  if F.lt x y then FP.lt_ordered x y;
  if F.lt y x then FP.lt_ordered y x;
  if ordered x && ordered y then (
    F.lte_is_lt_or_eq x y;
    F.lte_is_lt_or_eq y x;
    F.negate_lt_is_lte x y;
    F.negate_lt_is_lte y x
  )

inline_for_extraction noextract
let native_irreflexive (x:FP.t) : Lemma (~(F.lt x x)) =
  native_order x x;
  F.eq_spec x x

inline_for_extraction noextract
let ieee_equal_transitive (x y z:FP.t)
  : Lemma
      (requires F.ieee_eq x y /\ F.ieee_eq y z)
      (ensures F.ieee_eq x z) =
  F.eq_spec x y;
  F.eq_spec y z;
  F.eq_spec x z

inline_for_extraction noextract
let ieee_equal_comparisons (x y z:FP.t)
  : Lemma
      (requires F.ieee_eq x y)
      (ensures F.lt z x == F.lt z y /\ F.lt x z == F.lt y z) =
  native_order x y;
  native_order x z;
  native_order y z;
  if F.lt x z && F.lt z y then FP.lt_transitive x z y;
  if F.lt y z && F.lt z x then FP.lt_transitive y z x;
  if F.ieee_eq y z then ieee_equal_transitive x y z;
  if F.ieee_eq x z then ieee_equal_transitive y x z

inline_for_extraction noextract
let priority_transitive (x y z:FP.t)
  : Lemma
      (requires priority_le x y /\ priority_le y z)
      (ensures priority_le x z) =
  native_order x y;
  native_order y z;
  native_order x z;
  if F.lt z x && F.lt x y then FP.lt_transitive z x y;
  if F.ieee_eq x y then ieee_equal_comparisons x y z

inline_for_extraction noextract
let priority_native (x y:FP.t)
  : Lemma (
      priority_le x y <==>
        F.is_nan y \/ (~(F.is_nan x) /\ F.lte x y)) =
  if ordered x && ordered y then F.negate_lt_is_lte y x

inline_for_extraction noextract
noeq type selection = {
  best: FP.t;
  token: nat;
}

(* kernel.cu:378-382, including the strict comparison and NaN priority. *)
inline_for_extraction noextract
let update (before:selection) (candidate:FP.t) (index:nat) : GTot selection =
  if F.lt before.best candidate ||
     (F.is_nan candidate && not (F.is_nan before.best))
  then { best = candidate; token = index }
  else before

inline_for_extraction noextract
let update_dominates (before:selection) (candidate:FP.t) (index:nat)
  : Lemma (
      priority_le before.best (update before candidate index).best /\
      priority_le candidate (update before candidate index).best /\
      ((update before candidate index == before) \/
       ((update before candidate index).best == candidate /\
        (update before candidate index).token == index))) =
  native_order before.best candidate;
  native_irreflexive before.best;
  native_irreflexive candidate

inline_for_extraction noextract
let rec scan (#n:pos) (values:D.vec FP.t n) (row:nat) (before:selection)
  (count:nat{count <= n}) : GTot selection (decreases count) =
  if count = 0 then before
  else update (scan values row before (count - 1))
    (S.index values (count - 1)) (row + count - 1)

inline_for_extraction noextract
let rec scan_dominates (#n:pos) (values:D.vec FP.t n) (row:nat)
  (before:selection) (count:nat{count <= n})
  : Lemma
      (ensures
        priority_le before.best (scan values row before count).best /\
        (forall (i:nat{i < count}).
          priority_le (S.index values i) (scan values row before count).best) /\
        ((scan values row before count == before) \/
         (exists (i:nat{i < count}).
           (scan values row before count).best == S.index values i /\
           (scan values row before count).token == row + i)))
      (decreases count) =
  if count = 0 then native_irreflexive before.best
  else (
    scan_dominates values row before (count - 1);
    let previous = scan values row before (count - 1) in
    let current = S.index values (count - 1) in
    update_dominates previous current (row + count - 1);
    priority_transitive before.best previous.best
      (update previous current (row + count - 1)).best;
    let each (i:nat{i < count})
      : Lemma (priority_le (S.index values i) (scan values row before count).best) =
      if i < count - 1 then
        priority_transitive (S.index values i) previous.best
          (update previous current (row + count - 1)).best
    in
    FStar.Classical.forall_intro each
  )

inline_for_extraction noextract
let scan_special_values (#n:pos) (values:D.vec FP.t n)
  (row:nat) (before:selection)
  : Lemma (
      (F.is_nan before.best ==> F.is_nan (scan values row before n).best) /\
      (forall (i:D.token n).
        F.is_nan (S.index values i) ==> F.is_nan (scan values row before n).best) /\
      (forall (i:D.token n).
        ~(F.is_nan (scan values row before n).best) ==>
          ~(F.is_nan (S.index values i)) /\
          F.lte (S.index values i) (scan values row before n).best)) =
  scan_dominates values row before n;
  let each (i:D.token n)
    : Lemma (
        F.is_nan (S.index values i) ==> F.is_nan (scan values row before n).best) =
    ()
  in
  FStar.Classical.forall_intro each;
  let numerical (i:D.token n)
    : Lemma (
        ~(F.is_nan (scan values row before n).best) ==>
          ~(F.is_nan (S.index values i)) /\
          F.lte (S.index values i) (scan values row before n).best) =
    priority_native (S.index values i) (scan values row before n).best
  in
  FStar.Classical.forall_intro numerical
