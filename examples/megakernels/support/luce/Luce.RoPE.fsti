module Luce.RoPE

open FStar.Real
module Pow = FStar.Math.Pow

inline_for_extraction noextract
val coefficient : unit -> real

val coefficient_bounds ()
  : Lemma (
      390128936.0R /. 16777216.0R <. coefficient () /\
      coefficient () <. 390128937.0R /. 16777216.0R)

val literal_cell_check ()
  : Lemma (
      Kuiper.Float32.Literals.common_normal_cell "23.2534961700439453125"
        390128936 16777216 390128937 16777216 12191529 (-19))

val inverse_frequency_bridge (j:nat{j < 32})
  : Lemma (
      Pow.exp2 (0.0R -. (of_int j /. 32.0R) *. coefficient ()) ==
      1.0R /. Pow.pow 10000000.0R (of_int j /. 32.0R))
