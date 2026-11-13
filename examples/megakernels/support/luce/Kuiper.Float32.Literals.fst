module Kuiper.Float32.Literals

(* Exact rational certificates for unsigned fixed-point decimal literals.
   Unrecognized syntax gets no certificate; this is not a float parser. *)
noextract
let digit (c : FStar.Char.char) : option nat =
  match c with
  | '0' -> Some 0 | '1' -> Some 1 | '2' -> Some 2 | '3' -> Some 3
  | '4' -> Some 4 | '5' -> Some 5 | '6' -> Some 6 | '7' -> Some 7
  | '8' -> Some 8 | '9' -> Some 9 | _ -> None

noextract
let rec fraction (cs : list FStar.Char.char) (n : nat) (d : pos)
  : Tot (option (nat & pos)) (decreases cs) =
  match cs with
  | [] -> Some (n, d)
  | c :: rest ->
    match digit c with
    | None -> None
    | Some v -> fraction rest (10 * n + v) (10 * d)

noextract
let rec integral (cs : list FStar.Char.char) (n : nat) (seen : bool)
  : Tot (option (nat & pos)) (decreases cs) =
  match cs with
  | '.' :: rest ->
    if seen then (match rest with [] -> None | _ -> fraction rest n 1)
    else None
  | c :: rest ->
    (match digit c with
    | None -> None
    | Some v -> integral rest (10 * n + v) true)
  | [] -> None

noextract
let positive_normal_decimal (s : string) : bool =
  match integral (FStar.String.list_of_string s) 0 false with
  | None -> false
  | Some (n, d) ->
    n * 85070591730234615865843651857942052864 >= d &&
    n <= 340282346638528859811704183484516925440 * d

noextract
let at_most_one_decimal (s : string) : bool =
  match integral (FStar.String.list_of_string s) 0 false with
  | None -> false
  | Some (n, d) -> n <= d

noextract
let rec binary_power (n : nat) : Tot pos (decreases n) =
  if n = 0 then 1 else 2 * binary_power (n - 1)

(* Conservative normal rounding cells: stay within three eighths ULP of the
   center, leaving one eighth ULP before the nearest-rounding boundary and
   room for an intermediate binary64 literal conversion.
   Power-of-two centers are excluded to keep spacing uniform on both sides. *)
noextract
let common_normal_cell
  (s : string) (lo_num : nat) (lo_den : pos)
  (hi_num : nat) (hi_den : pos) (mantissa : nat) (exponent : int) : bool =
  if not (8388608 < mantissa && mantissa < 16777216 &&
          -149 <= exponent && exponent <= 104) then false
  else
    match integral (FStar.String.list_of_string s) 0 false with
    | None -> false
    | Some (n, d) ->
      let unit_num = if exponent >= 0 then binary_power exponent else 1 in
      let unit_den = if exponent >= 0 then 1 else binary_power (-exponent) in
      let cell_lo = (8 * mantissa - 3) * unit_num in
      let cell_hi = (8 * mantissa + 3) * unit_num in
      let cell_den = 8 * unit_den in
      lo_num * hi_den <= hi_num * lo_den &&
      cell_lo * d < n * cell_den && n * cell_den < cell_hi * d &&
      cell_lo * lo_den < lo_num * cell_den &&
      hi_num * cell_den < cell_hi * hi_den
