module FStar.Math.LogSeries

open FStar.Real

#set-options "--fuel 2 --ifuel 0 --z3rlimit 60"

type rpos = x:real{x >. 0.0R}
type rnonneg = x:real{x >=. 0.0R}

let rec ninth_power (n:nat) : Tot rpos (decreases n) =
  if n = 0 then 1.0R else ninth_power (n - 1) /. 9.0R

let term (n:nat) : rpos =
  2.0R *. ninth_power n /. (3.0R *. of_int (2 * n + 1))

let tail (n:nat) : rpos = 9.0R *. term n /. 8.0R

let rec partial (n:nat) : Tot rnonneg (decreases n) =
  if n = 0 then 0.0R else partial (n - 1) +. term (n - 1)

let partial_step (n:nat)
  : Lemma (partial (n + 1) == partial n +. term n) = ()

let term_ratio (n:nat)
  : Lemma (term (n + 1) <=. term n /. 9.0R) =
  assert (ninth_power (n + 1) == ninth_power n /. 9.0R);
  assert (of_int (2 * n + 1) >. 0.0R);
  assert (of_int (2 * (n + 1) + 1) >=. of_int (2 * n + 1))

let tail_step (n:nat)
  : Lemma (term n +. tail (n + 1) <=. tail n) =
  term_ratio n

let rec partial_mono (m n:nat{m <= n})
  : Lemma (partial m <=. partial n) (decreases n) =
  if m < n then (
    partial_mono m (n - 1);
    partial_step (n - 1)
  )

let rec upper_mono (n m:nat{n <= m})
  : Lemma (partial m +. tail m <=. partial n +. tail n)
      (decreases (m - n)) =
  if n < m then (
    partial_step n;
    tail_step n;
    upper_mono (n + 1) m
  )

let every_partial_bounded (n m:nat)
  : Lemma (partial m <=. partial n +. tail n) =
  if n <= m then upper_mono n m
  else partial_mono m n

let series_set (y:real) : prop =
  exists (n:nat). y == partial n

let partial_in_set (n:nat)
  : Lemma (series_set (partial n)) =
  introduce exists (m:nat). partial n == partial m with n and ()

let series_upper (n:nat)
  : Lemma (is_upper_bound series_set (partial n +. tail n)) =
  introduce forall (y:real).
    series_set y ==> y <=. partial n +. tail n
  with introduce series_set y ==> y <=. partial n +. tail n
  with (
    eliminate exists (m:nat). y == partial m
    with every_partial_bounded n m
  )

let series_ready ()
  : Lemma (is_nonempty series_set /\ is_bounded_above series_set) =
  partial_in_set 0;
  introduce exists (y:real). series_set y with (partial 0) and ();
  series_upper 0;
  introduce exists (b:real). is_upper_bound series_set b
  with (partial 0 +. tail 0) and ()

(* The convergent atanh series at 1/3 defines the natural-logarithm scale.
   Its positive terms admit a geometric tail bound; no analytic axiom is used. *)
[@@"opaque_to_smt"]
let log_two () : real =
  series_ready ();
  lub series_set

let log_two_lub ()
  : Lemma (is_lub series_set (log_two ())) =
  reveal_opaque (`%log_two) log_two;
  series_ready ()

let log_two_bounds (n:nat)
  : Lemma (partial n <=. log_two () /\ log_two () <=. partial n +. tail n) =
  partial_in_set n;
  series_upper n;
  log_two_lub ()

let log_two_positive ()
  : Lemma (log_two () >. 0.0R) =
  log_two_bounds 1;
  partial_step 0

type fraction = nat & pos

let fraction_value (q:fraction) : real =
  of_int (fst q) /. of_int (snd q)

let fraction_add (x y:fraction) : fraction =
  (fst x * snd y + fst y * snd x, snd x * snd y)

let fraction_add_value (x y:fraction)
  : Lemma (fraction_value (fraction_add x y) ==
      fraction_value x +. fraction_value y) = ()

let fraction_lt (x y:fraction)
  : Lemma
      (requires fst x * snd y < fst y * snd x)
      (ensures fraction_value x <. fraction_value y) = ()

let rec ninth_denominator (n:nat) : Tot pos (decreases n) =
  if n = 0 then 1 else 9 * ninth_denominator (n - 1)

let rec ninth_fraction (n:nat)
  : Lemma (ninth_power n == 1.0R /. of_int (ninth_denominator n))
      (decreases n) =
  if n > 0 then ninth_fraction (n - 1)

let term_fraction (n:nat) : fraction =
  (2, 3 * (2 * n + 1) * ninth_denominator n)

let term_fraction_value (n:nat)
  : Lemma (term n == fraction_value (term_fraction n)) =
  ninth_fraction n

let tail_fraction (n:nat) : fraction =
  (9 * fst (term_fraction n), 8 * snd (term_fraction n))

let tail_fraction_value (n:nat)
  : Lemma (tail n == fraction_value (tail_fraction n)) =
  term_fraction_value n

let rec partial_fraction (n:nat) : Tot fraction (decreases n) =
  if n = 0 then (0, 1)
  else fraction_add (partial_fraction (n - 1)) (term_fraction (n - 1))

let rec partial_fraction_value (n:nat)
  : Lemma (partial n == fraction_value (partial_fraction n)) (decreases n) =
  if n > 0 then (
    partial_fraction_value (n - 1);
    term_fraction_value (n - 1);
    fraction_add_value (partial_fraction (n - 1)) (term_fraction (n - 1))
  )

let upper_fraction (n:nat) : fraction =
  fraction_add (partial_fraction n) (tail_fraction n)

let upper_fraction_value (n:nat)
  : Lemma (partial n +. tail n == fraction_value (upper_fraction n)) =
  partial_fraction_value n;
  tail_fraction_value n;
  fraction_add_value (partial_fraction n) (tail_fraction n)

let decimal_certificate ()
  : Lemma (
      69314718055994530 * snd (partial_fraction 20) <
        fst (partial_fraction 20) * 100000000000000000 /\
      fst (upper_fraction 20) * 100000000000000000 <
        69314718055994532 * snd (upper_fraction 20)) =
  assert_norm (
      69314718055994530 * snd (partial_fraction 20) <
        fst (partial_fraction 20) * 100000000000000000 /\
      fst (upper_fraction 20) * 100000000000000000 <
        69314718055994532 * snd (upper_fraction 20))

#push-options "--fuel 1 --z3rlimit 60"
let log_two_decimal_bounds ()
  : Lemma (
      69314718055994530.0R /. 100000000000000000.0R <. log_two () /\
      log_two () <. 69314718055994532.0R /. 100000000000000000.0R) =
  log_two_bounds 20;
  partial_fraction_value 20;
  upper_fraction_value 20;
  decimal_certificate ();
  fraction_lt (69314718055994530, 100000000000000000) (partial_fraction 20);
  fraction_lt (upper_fraction 20) (69314718055994532, 100000000000000000)
#pop-options
