module KPlay.SharedSpec

module F32 = Kuiper.Float32
module FP = Kuiper.Floating
module A = Kuiper.Approximates.Base
module R = FStar.Real

(* The argsort/first-column/gather stage only. Approximation is not an
   order embedding: the discrete choice uses native FP32 comparisons. *)

inline_for_extraction noextract
let index (n:nat) : Type0 = i:nat{i < n}

inline_for_extraction noextract
let finite32 : Type0 = x:F32.t{FP.kind #F32.t x == FP.Finite}

inline_for_extraction noextract
let probability : Type0 =
  x:finite32{FP.lte #F32.t FP.zero x /\ FP.lte #F32.t x FP.one}

inline_for_extraction noextract
let row (e:pos) : Type0 = index e -> probability

inline_for_extraction noextract
let probabilities (s:nat) (e:pos) : Type0 = index s -> row e

inline_for_extraction noextract
noeq type outputs (s:nat) (e:pos) = {
  selected_expert: index s -> index e;
  selected_probability: index s -> R.real
}

inline_for_extraction noextract
let rec best_prefix (#e:pos) (p:row e) (n:pos{n <= e})
  : Tot (index n) (decreases n)
  = if n = 1 then 0
    else
      let previous = best_prefix p (n - 1) in
      let next = n - 1 in
      if FP.lt #F32.t (p previous) (p next) then next else previous

inline_for_extraction noextract
let top1_index (#e:pos) (p:row e) : index e =
  best_prefix p e

inline_for_extraction noextract
let select (#s:nat) (#e:pos) (p:probabilities s e) : outputs s e = {
  selected_expert = (fun t -> top1_index (p t));
  selected_probability =
    (fun t -> A.to_real #F32.t (p t (top1_index (p t))))
}

inline_for_extraction noextract
let transition (#s:nat) (#e:pos) (p:probabilities s e)
  : probabilities s e & outputs s e =
  (p, select p)

inline_for_extraction noextract
let post (#s:nat) (#e:pos)
  (before after:probabilities s e) (out:outputs s e) : prop =
  (forall (t:index s) (j:index e). after t j == before t j) /\
  (forall (t:index s).
    out.selected_expert t == top1_index (before t) /\
    out.selected_probability t ==
      A.to_real #F32.t (before t (top1_index (before t))))

inline_for_extraction noextract
let stable_maximum (#e:pos) (p:row e)
  (n:pos{n <= e}) (chosen:index n) : prop =
  forall (j:index n).
    FP.lte #F32.t (p j) (p chosen) /\
    (j < chosen ==> FP.lt #F32.t (p j) (p chosen))

(* These are consequences of the existing finite native comparison laws,
   including numerical equality of the two zero signs. *)
inline_for_extraction noextract
let comparison (x y:finite32)
  : Lemma (
      (FP.lte #F32.t x y <==> ~(FP.lt #F32.t y x)) /\
      (FP.lte #F32.t y x <==> ~(FP.lt #F32.t x y)) /\
      (FP.lte #F32.t x y <==> FP.lt #F32.t x y \/ FP.eq #F32.t x y) /\
      (~(FP.lt #F32.t x y) ==> FP.lt #F32.t y x \/ FP.eq #F32.t x y) /\
      (FP.eq #F32.t x y <==> FP.eq #F32.t y x) /\
      (FP.eq #F32.t x y ==> ~(FP.lt #F32.t x y) /\ ~(FP.lt #F32.t y x)) /\
      (FP.lt #F32.t x y ==> ~(FP.lt #F32.t y x)) /\
      FP.lte #F32.t x x)
  = FP.eq_spec #F32.t x y;
    FP.eq_spec #F32.t y x;
    FP.eq_spec #F32.t x x;
    FP.lte_is_lt_or_eq #F32.t x y;
    FP.lte_is_lt_or_eq #F32.t y x;
    FP.lte_is_lt_or_eq #F32.t x x;
    FP.negate_lt_is_lte #F32.t x y;
    FP.negate_lt_is_lte #F32.t y x

inline_for_extraction noextract
let numerical_eq_transitive (x y z:finite32)
  : Lemma
      (requires FP.eq #F32.t x y /\ FP.eq #F32.t y z)
      (ensures FP.eq #F32.t x z)
  = FP.eq_spec #F32.t x y;
    FP.eq_spec #F32.t y z;
    FP.eq_spec #F32.t x z

inline_for_extraction noextract
let le_lt_transitive (x y z:finite32)
  : Lemma
      (requires FP.lte #F32.t x y /\ FP.lt #F32.t y z)
      (ensures FP.lt #F32.t x z)
  = comparison x y;
    comparison x z;
    comparison y z;
    if not (FP.lt #F32.t x z) then (
      if FP.lt #F32.t z x then F32.lt_transitive y z x
      else if FP.lt #F32.t x y then F32.lt_transitive x y z
      else numerical_eq_transitive y x z
    )

inline_for_extraction noextract
let rec best_prefix_correct (#e:pos) (p:row e) (n:pos{n <= e})
  : Lemma
      (ensures stable_maximum p n (best_prefix p n))
      (decreases n)
  = if n = 1 then comparison (p 0) (p 0)
    else (
      best_prefix_correct p (n - 1);
      let previous = best_prefix p (n - 1) in
      let next = n - 1 in
      if FP.lt #F32.t (p previous) (p next) then (
        let each (j:index n)
          : Lemma (
              FP.lte #F32.t (p j) (p next) /\
              (j < next ==> FP.lt #F32.t (p j) (p next)))
          = if j = next then comparison (p next) (p next)
            else (
              assert (j < next);
              le_lt_transitive (p j) (p previous) (p next);
              comparison (p j) (p next)
            )
        in
        FStar.Classical.forall_intro each
      ) else (
        let each (j:index n)
          : Lemma (
              FP.lte #F32.t (p j) (p previous) /\
              (j < previous ==> FP.lt #F32.t (p j) (p previous)))
          = if j = next then comparison (p previous) (p next)
            else assert (j < next)
        in
        FStar.Classical.forall_intro each
      )
    )

inline_for_extraction noextract
let select_correct (#s:nat) (#e:pos)
  (p:probabilities s e) (t:index s)
  : Lemma (
      stable_maximum (p t) e ((select p).selected_expert t) /\
      A.v_approximates #F32.t
        (p t ((select p).selected_expert t))
        ((select p).selected_probability t))
  = best_prefix_correct (p t) e;
    A.to_real_ok #F32.t (p t (top1_index (p t)))

inline_for_extraction noextract
let all_equal_selects_first (#e:pos) (p:row e)
  : Lemma
      (requires forall (j:index e). FP.eq #F32.t (p j) (p 0))
      (ensures top1_index p == 0)
  = best_prefix_correct p e;
    let chosen = top1_index p in
    comparison (p 0) (p chosen)

inline_for_extraction noextract
let singleton_selects_only (p:row 1)
  : Lemma (top1_index p == 0)
  = ()

inline_for_extraction noextract
let transition_satisfies_post (#s:nat) (#e:pos) (p:probabilities s e)
  : Lemma (let after, out = transition p in post p after out)
  = ()
