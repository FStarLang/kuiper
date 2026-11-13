module Kuiper.Real.Trigonometry

open FStar.Real

(* Trusted real trigonometry in radians. The remainder laws characterize the
   ordinary sine/cosine power series, not arbitrary rotary coefficients.
   They are mathematical assumptions, not floating-point error bounds. *)
inline_for_extraction noextract
let rec divided_power (x:real) (n:nat) : Tot real (decreases n) =
  if n = 0 then 1.0R
  else divided_power x (n - 1) *. x /. of_int n

inline_for_extraction noextract
let absolute (x:real) : real =
  if t2b (x <. 0.0R) then 0.0R -. x else x

inline_for_extraction noextract
let rec sine_sum (x:real) (n:nat) : Tot real (decreases n) =
  if n = 0 then 0.0R
  else
    let term = divided_power x (2 * n - 1) in
    sine_sum x (n - 1) +. (if (n - 1) % 2 = 0 then term else 0.0R -. term)

inline_for_extraction noextract
let rec cosine_sum (x:real) (n:nat) : Tot real (decreases n) =
  if n = 0 then 0.0R
  else
    let term = divided_power x (2 * n - 2) in
    cosine_sum x (n - 1) +. (if (n - 1) % 2 = 0 then term else 0.0R -. term)

val sin (x:real) : real
val cos (x:real) : real

val sin_remainder (x:real) (n:nat)
  : Lemma (
      let error = sin x -. sine_sum x n in
      let bound = divided_power (absolute x) (2 * n + 1) in
      0.0R -. bound <=. error /\ error <=. bound)

val cos_remainder (x:real) (n:nat)
  : Lemma (
      let error = cos x -. cosine_sum x n in
      let bound = divided_power (absolute x) (2 * n) in
      0.0R -. bound <=. error /\ error <=. bound)

let sin_zero () : Lemma (sin 0.0R == 0.0R) =
  sin_remainder 0.0R 0

let cos_zero () : Lemma (cos 0.0R == 1.0R) =
  cos_remainder 0.0R 1
