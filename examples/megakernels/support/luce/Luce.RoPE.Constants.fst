module Luce.RoPE.Constants

open FStar.Real
module Pow = FStar.Math.Pow
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module A = Kuiper.Approximates.Base
module CF = Kuiper.Float32.ConstantFold
module Cert = Kuiper.Float32.ConstantFold.Certificates
module Data = Luce.RoPE.Constants.Data
module Power = Luce.RoPE.Constants.Power

noextract
let denominator : pos = 1000000000000

noextract
let root_certificate (j:nat{j < 32}) : bool =
  let entry = Data.data j in
  Power.ipow entry.lo 32 < Power.ipow 10000000 j * Power.ipow denominator 32 &&
  Power.ipow 10000000 j * Power.ipow denominator 32 < Power.ipow entry.hi 32

noextract
let literal_certificate (j:nat{j < 32}) : bool =
  let e = Data.data j in
  Cert.exact_interval e.input e.lo denominator e.hi denominator e.im e.ie &&
  Cert.reciprocal e.input e.output e.im e.ie e.om e.oe

let certificates_check (j:nat{j < 32})
  : Lemma (root_certificate j /\ literal_certificate j) =
  match j with
  | 0 -> assert_norm (root_certificate 0 /\ literal_certificate 0)
  | 1 -> assert_norm (root_certificate 1 /\ literal_certificate 1)
  | 2 -> assert_norm (root_certificate 2 /\ literal_certificate 2)
  | 3 -> assert_norm (root_certificate 3 /\ literal_certificate 3)
  | 4 -> assert_norm (root_certificate 4 /\ literal_certificate 4)
  | 5 -> assert_norm (root_certificate 5 /\ literal_certificate 5)
  | 6 -> assert_norm (root_certificate 6 /\ literal_certificate 6)
  | 7 -> assert_norm (root_certificate 7 /\ literal_certificate 7)
  | 8 -> assert_norm (root_certificate 8 /\ literal_certificate 8)
  | 9 -> assert_norm (root_certificate 9 /\ literal_certificate 9)
  | 10 -> assert_norm (root_certificate 10 /\ literal_certificate 10)
  | 11 -> assert_norm (root_certificate 11 /\ literal_certificate 11)
  | 12 -> assert_norm (root_certificate 12 /\ literal_certificate 12)
  | 13 -> assert_norm (root_certificate 13 /\ literal_certificate 13)
  | 14 -> assert_norm (root_certificate 14 /\ literal_certificate 14)
  | 15 -> assert_norm (root_certificate 15 /\ literal_certificate 15)
  | 16 -> assert_norm (root_certificate 16 /\ literal_certificate 16)
  | 17 -> assert_norm (root_certificate 17 /\ literal_certificate 17)
  | 18 -> assert_norm (root_certificate 18 /\ literal_certificate 18)
  | 19 -> assert_norm (root_certificate 19 /\ literal_certificate 19)
  | 20 -> assert_norm (root_certificate 20 /\ literal_certificate 20)
  | 21 -> assert_norm (root_certificate 21 /\ literal_certificate 21)
  | 22 -> assert_norm (root_certificate 22 /\ literal_certificate 22)
  | 23 -> assert_norm (root_certificate 23 /\ literal_certificate 23)
  | 24 -> assert_norm (root_certificate 24 /\ literal_certificate 24)
  | 25 -> assert_norm (root_certificate 25 /\ literal_certificate 25)
  | 26 -> assert_norm (root_certificate 26 /\ literal_certificate 26)
  | 27 -> assert_norm (root_certificate 27 /\ literal_certificate 27)
  | 28 -> assert_norm (root_certificate 28 /\ literal_certificate 28)
  | 29 -> assert_norm (root_certificate 29 /\ literal_certificate 29)
  | 30 -> assert_norm (root_certificate 30 /\ literal_certificate 30)
  | _ -> assert_norm (root_certificate 31 /\ literal_certificate 31)

let power_enclosure (j:nat{j < 32})
  : Lemma (
      let e = Data.data j in
      of_int e.lo /. of_int denominator <=. Pow.pow 10000000.0R (of_int j /. 32.0R) /\
      Pow.pow 10000000.0R (of_int j /. 32.0R) <=. of_int e.hi /. of_int denominator) =
  certificates_check j;
  let e = Data.data j in
  Power.enclosure 10000000 j 32 e.lo e.hi denominator

let inverse_frequency_approx (j:nat{j < 32})
  : Lemma (A.v_approximates (inverse_frequency j)
      (1.0R /. Pow.pow 10000000.0R (of_int j /. 32.0R))) =
  certificates_check j;
  power_enclosure j;
  let e = Data.data j in
  let ideal = Pow.pow 10000000.0R (of_int j /. 32.0R) in
  CF.literal_approx e.input e.lo denominator e.hi denominator e.im e.ie ideal;
  A.div_approx (F.one #FP.t) (F.of_literal #FP.t e.input) 1.0R ideal;
  CF.reciprocal_literal_eq e.input e.output e.im e.ie e.om e.oe
