module Kuiper.Sum

open Kuiper.Common
open Kuiper.Functions

class semigroup (a : Type) = {
  id : a;
  op : a -> a -> a;

  assoc : is_associative op;
  neut : is_neutral_for id op;
}

class commutative_semigroup (a : Type) = {
  [@@@Tactics.Typeclasses.tcinstance]
  is_semi : semigroup a;

  comm : is_commutative is_semi.op;
}

(* [i,j) *)
val sum (#a:Type) {| semigroup a |} (i j : nat) (f : between i j -> GTot a) : GTot a
let rec sum #a #_ i j f : GTot a (decreases j-i) =
  if i >= j
  then id
  else f i `op` sum (i + 1) j f

let sum_pop_left (#a:Type) {| semigroup a |}
  (i j : nat{i < j}) (f : between i j -> GTot a)
  : Lemma (sum i j f == f i `op` sum (i + 1) j f)
  = ()

let rec sum_pop_right (#a:Type) {| d : semigroup a |}
  (i j : nat{i < j}) (f : between i j -> GTot a)
  : Lemma (ensures sum i j f == sum i (j - 1) f `op` f (j - 1))
          (decreases j - i)
= if i < j-1 then
    calc (==) {
      sum i j f;
    == {}
      f i `op` sum (i + 1) j f;
    == { sum_pop_right (i + 1) j f }
      f i `op` (sum (i + 1) (j - 1) f `op` f (j - 1));
    == { let _ = d.assoc in () }
      (f i `op` sum (i + 1) (j - 1) f) `op` f (j - 1);
    == {}
      sum i (j - 1) f `op` f (j - 1);
    }
  else (
    assert (i == j - 1);
    calc (==) {
      sum i j f;
    == {}
      f i `op` sum j j f;
    == { let _ = d.neut in () }
      f i;
    == {}
      f (j-1);
    == { let _ = d.neut in () }
      sum i (j - 1) f `op` f (j - 1);
    }
  )

let rec sum_split (#a:Type) {| d : semigroup a |}
  (i j : nat) (f : between i j -> GTot a) (k : nat{i <= k /\ k <= j})
  : Lemma (ensures sum i j f == sum i k f `op` sum k j f)
          (decreases j - i)
  = if i = k then
      calc (==) {
        sum i j f;
      == { let _ = d.neut in () }
        id `op` sum i j f;
      == {}
        sum i k f `op` sum k j f;
      }
    else
      calc (==) {
        sum i j f;
      == { sum_pop_left i j f }
        f i `op` sum (i + 1) j f;
      == { sum_split (i + 1) j f k }
        f i `op` (sum (i + 1) k f `op` sum k j f);
      == { let _ = d.assoc in () }
        (f i `op` sum (i + 1) k f) `op` sum k j f;
      == {}
        sum i k f `op` sum k j f;
      }
