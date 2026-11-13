module Luce.RoPE

open FStar.Real
module Pow = FStar.Math.Pow
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module Lit = Kuiper.Float32.Literals

#set-options "--fuel 2 --ifuel 1 --z3rlimit 120"

noextract
let precision : pos = 1000000000000
type interval = nat & nat

noextract
let value (n:nat) : real = of_int n /. of_int precision
noextract
let ratio (m n:nat) : real = of_int m /. of_int (Lit.binary_power n)

let exp2_add (x y:real)
  : Lemma (Pow.exp2 (x +. y) == Pow.exp2 x *. Pow.exp2 y) =
  Exp.exp2_as_exp (x +. y);
  Exp.exp2_as_exp x;
  Exp.exp2_as_exp y;
  Exp.exp_add (x *. Exp.log 2.0R) (y *. Exp.log 2.0R)

let half_power (x:real)
  : Lemma (Pow.exp2 (x /. 2.0R) == Sqrt.sqrt (Pow.exp2 x)) =
  Pow.pow_pow 2.0R x 0.5R;
  Pow.pow_half (Pow.exp2 x)

let square_enclosure (x:Pow.rpos) (lo hi:nat)
  : Lemma
      (requires value lo *. value lo <=. x /\ x <=. value hi *. value hi)
      (ensures value lo <=. Sqrt.sqrt x /\ Sqrt.sqrt x <=. value hi) =
  Sqrt.sqrt_square x;
  if Sqrt.sqrt x <. value lo then (
    assert (Sqrt.sqrt x *. Sqrt.sqrt x <. value lo *. value lo)
  );
  if value hi <. Sqrt.sqrt x then (
    assert (value hi *. value hi <. Sqrt.sqrt x *. Sqrt.sqrt x)
  )

noextract
let rec certificate (m n:nat) (steps:list interval)
  : Tot bool (decreases n) =
  match steps with
  | [] -> false
  | (lo,hi) :: rest ->
    if n = 0 then
      m = 0 && lo = precision && hi = precision &&
      (match rest with [] -> true | _ -> false)
    else
      let d = Lit.binary_power (n - 1) in
      let upper = m >= d in
      let rem = if upper then m - d else m in
      let factor = if upper then 2 else 1 in
      match rest with
      | [] -> false
      | (previous_lo,previous_hi) :: _ ->
        m < 2 * d &&
        certificate rem (n - 1) rest &&
        lo * lo <= factor * previous_lo * precision &&
        factor * previous_hi * precision <= hi * hi

let rec certificate_sound (m n:nat) (steps:list interval)
  : Lemma
      (requires certificate m n steps)
      (ensures (match steps with
        | [] -> False
        | (lo,hi) :: _ ->
          value lo <=. Pow.exp2 (ratio m n) /\
          Pow.exp2 (ratio m n) <=. value hi))
      (decreases n) =
  match steps with
  | [] -> ()
  | (lo,hi) :: rest ->
    if n = 0 then (
      Pow.pow_zero 2.0R
    ) else (
      let d = Lit.binary_power (n - 1) in
      let upper = m >= d in
      let rem = if upper then m - d else m in
      let factor = if upper then 2 else 1 in
      certificate_sound rem (n - 1) rest;
      let x = ratio rem (n - 1) in
      if upper then (
        exp2_add 1.0R x;
        Pow.pow_one 2.0R;
        half_power (1.0R +. x);
        square_enclosure (2.0R *. Pow.exp2 x) lo hi
      ) else (
        half_power x;
        square_enclosure (Pow.exp2 x) lo hi
      )
    )

noextract
let lower_certificate : list interval = [
  (1192092881187,1192092881189); (1421085437378,1421085437380);
  (1009741910164,1009741910166); (1019578725142,1019578725144);
  (1039540776763,1039540776765); (1080645026554,1080645026556);
  (1167793673417,1167793673419); (1363742063675,1363742063677);
  (1859792416239,1859792416241); (1729413915751,1729413915753);
  (1495436245997,1495436246000); (1118164782922,1118164782924);
  (1250292481769,1250292481771); (1563231289970,1563231289973);
  (1221846032972,1221846032974); (1492907728290,1492907728293);
  (1114386742595,1114386742597); (1241857812073,1241857812074);
  (1542210825407,1542210825409); (1189207115002,1189207115004);
  (1414213562373,1414213562374); (1000000000000,1000000000000);
  (1000000000000,1000000000000); (1000000000000,1000000000000);
  (1000000000000,1000000000000)
]

noextract
let upper_certificate : list interval = [
  (1192092930438,1192092930440); (1421085554801,1421085554804);
  (1009742077033,1009742077035); (1019579062132,1019579062133);
  (1039541463938,1039541463940); (1080646455247,1080646455249);
  (1167796761240,1167796761241); (1363749275563,1363749275564);
  (1859812086599,1859812086601); (1729450498730,1729450498732);
  (1495499513780,1495499513782); (1118259397859,1118259397860);
  (1250504080900,1250504080902); (1563760456350,1563760456352);
  (1222673382423,1222673382425); (1494930200086,1494930200089);
  (1117408151566,1117408151568); (1248600977188,1248600977190);
  (1559004400236,1559004400239); (1215247359979,1215247359982);
  (1476826145938,1476826145941); (1090507732664,1090507732666);
  (1189207115002,1189207115004); (1414213562373,1414213562374);
  (1000000000000,1000000000000)
]

let certificates_check ()
  : Lemma (certificate 4252968 24 lower_certificate /\
           certificate 4252969 24 upper_certificate) =
  assert_norm (certificate 4252968 24 lower_certificate /\
               certificate 4252969 24 upper_certificate)

let corrupted_certificates_rejected ()
  : Lemma (
      not (certificate 0 0 [(1000000000000,1000000000001)]) /\
      not (certificate 1 1 [(1414213562373,1414213562373);
                           (1000000000000,1000000000000)]) /\
      (match lower_certificate with
       | [] -> false
       | _ :: rest -> not (certificate 4252968 24 ((0,0) :: rest)))) =
  assert_norm (
      not (certificate 0 0 [(1000000000000,1000000000001)]) /\
      not (certificate 1 1 [(1414213562373,1414213562373);
                           (1000000000000,1000000000000)]) /\
      (match lower_certificate with
       | [] -> false
       | _ :: rest -> not (certificate 4252968 24 ((0,0) :: rest))))

let rec integer_power (n:nat)
  : Lemma (Pow.exp2 (of_int n) == of_int (Lit.binary_power n))
      (decreases n) =
  if n = 0 then Pow.pow_zero 2.0R
  else (
    integer_power (n - 1);
    Pow.pow_succ 2.0R (n - 1)
  )

let exponential_bracket ()
  : Lemma (
      Pow.exp2 (390128936.0R /. 16777216.0R) <. 10000000.0R /\
      10000000.0R <. Pow.exp2 (390128937.0R /. 16777216.0R)) =
  certificates_check ();
  certificate_sound 4252968 24 lower_certificate;
  certificate_sound 4252969 24 upper_certificate;
  integer_power 23;
  assert_norm (Lit.binary_power 23 == 8388608 /\ Lit.binary_power 24 == 16777216);
  exp2_add 23.0R (ratio 4252968 24);
  exp2_add 23.0R (ratio 4252969 24)

inline_for_extraction noextract
let coefficient () : real =
  Exp.log_two_positive ();
  Exp.log 10000000.0R /. Exp.log 2.0R

let coefficient_exponential ()
  : Lemma (Pow.exp2 (coefficient ()) == 10000000.0R) =
  Exp.log_two_positive ();
  Exp.exp2_as_exp (coefficient ());
  Exp.exp_log 10000000.0R

let coefficient_bounds ()
  : Lemma (
      390128936.0R /. 16777216.0R <. coefficient () /\
      coefficient () <. 390128937.0R /. 16777216.0R) =
  exponential_bracket ();
  coefficient_exponential ();
  if coefficient () <=. 390128936.0R /. 16777216.0R then
    Pow.pow_mono 2.0R (coefficient ()) (390128936.0R /. 16777216.0R);
  if 390128937.0R /. 16777216.0R <=. coefficient () then
    Pow.pow_mono 2.0R (390128937.0R /. 16777216.0R) (coefficient ())

let literal_cell_check ()
  : Lemma (
      Lit.common_normal_cell "23.2534961700439453125"
        390128936 16777216 390128937 16777216 12191529 (-19)) =
  assert_norm (
      Lit.common_normal_cell "23.2534961700439453125"
        390128936 16777216 390128937 16777216 12191529 (-19))

let inverse_frequency_bridge (j:nat{j < 32})
  : Lemma (
      Pow.exp2 (0.0R -. (of_int j /. 32.0R) *. coefficient ()) ==
      1.0R /. Pow.pow 10000000.0R (of_int j /. 32.0R)) =
  coefficient_exponential ();
  Pow.pow_pow 2.0R (coefficient ()) (0.0R -. of_int j /. 32.0R);
  Pow.pow_neg 10000000.0R (of_int j /. 32.0R)
