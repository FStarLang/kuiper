module KPlay.Proof.RoPE.Source

open Kuiper
open Kuiper.Float.Casts
module S = FStar.Seq.Base
module I64 = FStar.Int64

inline_for_extraction noextract let rotary_half = 32
inline_for_extraction noextract let rotary_cells = 64
inline_for_extraction noextract type position = p:I64.t{0 <= I64.v p /\ I64.v p < 2048}

inline_for_extraction noextract
let inverse_frequency (i:nat{i < rotary_half}) : f32 =
  match i with
  | 0  -> of_literal "1.0"
  | 1  -> of_literal "0.604296386241912841796875"
  | 2  -> of_literal "0.3651741445064544677734375"
  | 3  -> of_literal "0.22067339718341827392578125"
  | 4  -> of_literal "0.13335214555263519287109375"
  | 5  -> of_literal "0.08058421313762664794921875"
  | 6  -> of_literal "0.0486967526376247406005859375"
  | 7  -> of_literal "0.02942727319896221160888671875"
  | 8  -> of_literal "0.01778279431164264678955078125"
  | 9  -> of_literal "0.01074607856571674346923828125"
  | 10 -> of_literal "0.0064938166178762912750244140625"
  | 11 -> of_literal "0.0039241896010935306549072265625"
  | 12 -> of_literal "0.00237137381918728351593017578125"
  | 13 -> of_literal "0.001433012657798826694488525390625"
  | 14 -> of_literal "0.000865964335389435291290283203125"
  | 15 -> of_literal "0.000523299095220863819122314453125"
  | 16 -> of_literal "0.000316227786242961883544921875"
  | 17 -> of_literal "0.000191095299669541418552398681640625"
  | 18 -> of_literal "0.000115478193038143217563629150390625"
  | 19 -> of_literal "0.000069783054641447961330413818359375"
  | 20 -> of_literal "0.0000421696531702764332294464111328125"
  | 21 -> of_literal "0.000025482966520939953625202178955078125"
  | 22 -> of_literal "0.000015399264157167635858058929443359375"
  | 23 -> of_literal "0.00000930572059587575495243072509765625"
  | 24 -> of_literal "0.00000562341347176698036491870880126953125"
  | 25 -> of_literal "0.00000339820826411596499383449554443359375"
  | 26 -> of_literal "0.00000205352489501819945871829986572265625"
  | 27 -> of_literal "0.00000124093776321387849748134613037109375"
  | 28 -> of_literal "0.00000074989424092564149759709835052490234375"
  | 29 -> of_literal "0.000000453158378377338522113859653472900390625"
  | 30 -> of_literal "0.00000027384197665014653466641902923583984375"
  | _  -> of_literal "0.00000016548170833630138076841831207275390625"

inline_for_extraction noextract
let rotary_angle (p:position) (i:nat{i < rotary_half}) : f32 =
  mul (inverse_frequency i) (of_int p)

inline_for_extraction noextract
let rotary_component (p:position) (d:nat{d < rotary_cells}) : bf16 =
  if d < rotary_half
  then fcast (cos (rotary_angle p d))
  else fcast (sin (rotary_angle p (d - rotary_half)))

inline_for_extraction noextract
let rotary_post (p:position) (r:lseq bf16 rotary_cells) : prop =
  forall (d:nat{d < rotary_cells}).
    S.index r d == rotary_component p d

