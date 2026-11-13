module Kuiper.Floating.Bounds

module F = Kuiper.Floating
module I64 = FStar.Int64

let limit : nat = 16777216
let bound = n:nat{n <= limit}

(* Spec-only integer endpoints, exact for binary32 throughout this range. *)
let integer (#t:Type) {| F.floating t |} (n:bound) : GTot t =
  F.of_int (I64.int_to_t n)

let magnitude (#t:Type) {| F.floating t |} (n:bound) (x:t) : prop =
  F.is_finite x /\ F.lte (F.abs x) (integer n)

let nonnegative (#t:Type) {| F.floating t |} (n:bound) (x:t) : prop =
  F.is_finite x /\ F.lte F.zero x /\ F.lte x (integer n)

let unit (#t:Type) {| F.floating t |} (x:t) : prop =
  F.is_finite x /\ F.lte F.zero x /\ F.lte x F.one

let integer_zero (#t:Type) {| F.floating t |} ()
  : Lemma (integer #t 0 == F.zero)
  = I64.v_inj (I64.int_to_t 0) 0L

let integer_one (#t:Type) {| F.floating t |} ()
  : Lemma (integer #t 1 == F.one)
  = I64.v_inj (I64.int_to_t 1) 1L

let nonnegative_magnitude (#t:Type) {| F.floating t |} (n:bound) (x:t)
  : Lemma (requires nonnegative n x)
          (ensures magnitude n x)
  = ()

let nonnegative_unit (#t:Type) {| F.floating t |} (x:t)
  : Lemma (nonnegative 1 x <==> unit x)
  = integer_one #t ()

let sub_post (#t:Type) {| F.floating t |} (x y result:t) : prop =
  (forall (a:bound) (b:bound{a + b <= limit}).
    {:pattern (magnitude a x); (magnitude b y)}
    magnitude a x /\ magnitude b y ==>
    magnitude (a + b) result /\
    (F.lte x y ==> F.lte result F.zero)) /\
  (F.is_finite x /\ F.is_finite y /\ F.eq x y ==> F.is_zero result)

let fma_post (#t:Type) {| F.floating t |} (x y z result:t) : prop =
  forall (a b:bound) (c:bound{a * b + c <= limit}).
    {:pattern (magnitude a x); (magnitude b y); (magnitude c z)}
    magnitude a x /\ magnitude b y /\ magnitude c z ==>
    magnitude (a * b + c) result /\
    (F.lte F.zero x /\ F.lte F.zero y /\ F.lte F.zero z ==>
      F.lte F.zero result /\ F.lte z result /\
      (y == F.one ==> F.lte x result))

let exp_post (#t:Type) {| F.floating t |} (x result:t) : prop =
  ((F.is_finite x /\ F.lte x F.zero) \/ x == F.sub F.zero F.infinity ==>
    unit result) /\
  (F.is_zero x ==> result == F.one) /\
  (x == F.sub F.zero F.infinity ==> result == F.zero)

let divide_post (#t:Type) {| F.floating t |} (x y result:t) : prop =
  unit x /\ nonnegative 8 y /\ F.lte F.one y ==> unit result
