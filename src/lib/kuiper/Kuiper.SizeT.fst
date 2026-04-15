module Kuiper.SizeT

let fits_iff_u32 (x:nat)
  : Lemma (FStar.SizeT.fits x <==> FStar.UInt.fits x 32)
          [SMTPat (FStar.SizeT.fits x)]
  = admit() (* assumption *)

(* This is extracted primitely, it must not be marked inline. *)
let sizet_to_u32 (x: SZ.t)
  : Pure U32.t (requires FStar.UInt.fits (SZ.v x) 32)
               (ensures fun r -> U32.v r == SZ.v x)
  = U32.uint_to_t (SZ.v x)

(* This is extracted primitively, it must not be marked inline. *)
let sizet_and (x y : SZ.t) : SZ.t =
  magic ()

let sizet_and_div_pow2 (x:SZ.t) (y:SZ.t) (n:nat)
  : Lemma (requires SZ.v y == pow2 n)
          (ensures SZ.v (sizet_and x SZ.(y -^ 1sz)) == SZ.v x % (pow2 n))
  = admit() (* assumption, should prove it! *)

let s_divmod_inv_1 (j:szp) (i:sz)
  : Lemma (s_undivmod j (s_divmod j i) == i)
          [SMTPat (s_undivmod j (s_divmod j i))]
  = ()

let s_divmod_inv_2 (j:szp) (dm : sz & szlt j {SZ.fits (dm._1 * j + dm._2)})
  : Lemma (s_divmod j (s_undivmod j dm) == dm)
          [SMTPat (s_divmod j (s_undivmod j dm))]
  = ()

let lem_sdivup (x:sz) (y:szp{SZ.fits (x+y)})
  : Lemma (SZ.v (sdivup x y) == divup (SZ.v x) (SZ.v y))
          [SMTPat (sdivup x y)]
  = ()

inline_for_extraction noextract
let spow2 (s : sz{s < 32}) : r:sz{SZ.v r == pow2 s} =
  SZ.uint32_to_sizet (U32.shift_left 1ul (sizet_to_u32 s))

inline_for_extraction noextract
let sdiv_pow2 (i:sz{i < 32}) (tid: sz) : bool =
  // SZ.rem tid (spow2 i) = 0sz
  sizet_and tid SZ.(spow2 i -^ 1sz) = 0sz

let sdiv_pow2_ok (i:sz{i < 32}) (tid:sz) :
  Lemma (sdiv_pow2 i tid == div_pow2 i tid)
        [SMTPat (sdiv_pow2 i tid)]
= sizet_and_div_pow2 tid (spow2 i) i;
  calc (==) {
    SZ.v (SZ.rem tid (spow2 i));
    == {}
    SZ.v tid - ((SZ.v tid / SZ.v (spow2 i)) * SZ.v (spow2 i));
    == { FStar.Math.Lemmas.euclidean_division_definition (SZ.v tid) (SZ.v (spow2 i)) }
    SZ.v tid % SZ.v (spow2 i);
    == {}
    SZ.v tid % pow2 (SZ.v i);
}

noextract inline_for_extraction
let smin (a b : sz): sz =
  let open FStar.SizeT in
  if a <^ b then a else b
