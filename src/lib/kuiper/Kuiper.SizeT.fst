module Kuiper.SizeT

module SZ = FStar.SizeT

let fits_iff_u32 (x:nat)
  : Lemma (FStar.SizeT.fits x <==> FStar.UInt.fits x 32)
          [SMTPat (FStar.SizeT.fits x)]
  = admit() (* assumption *)

(* This is extracted primitely, it must not be marked inline. *)
let sizet_to_u32 (x: SZ.t)
  : Pure U32.t (requires FStar.UInt.fits (SZ.v x) 32)
               (ensures fun r -> U32.v r == SZ.v x)
  = U32.uint_to_t (SZ.v x)

(* This is extracted primitively, it must not be marked inline. The
implementation here is just a model. *)
let sizet_and (x y : SZ.t) : SZ.t =
  FStar.SizeT.uint_to_t <|
  FStar.SizeT.v x `FStar.UInt.logand #32` FStar.SizeT.v y

(* This should probably be in F*'s library *)
let rec from_vec_zero (#n:nat) (vec : FStar.BitVector.bv_t n)
  : Lemma (requires forall (i : nat{i < n}). Seq.index vec i = false)
          (ensures UInt.from_vec #n vec = 0)
  = if n = 0 then () else from_vec_zero #(n-1) (Seq.slice vec 0 (n-1))

let sizet_and_div_pow2 (x:SZ.t) (y:SZ.t) (n:nat)
  : Lemma (requires SZ.v y == pow2 n)
          (ensures SZ.v (sizet_and x (y -^ 1sz)) == SZ.v x % (pow2 n))
  = if n = 0 then (
      (* jeez *)
      calc (==) {
        SZ.v (sizet_and x (y -^ 1sz));
        == {}
        FStar.SizeT.v x `FStar.UInt.logand #32` 0;
        == { from_vec_zero #32 (UInt.to_vec #32 (FStar.SizeT.v x `FStar.UInt.logand #32` 0)) }
        0;
      }
    ) else
      FStar.UInt.logand_mask #32 (SZ.v x) n

let s_divmod_inv_1 (j:szp) (i:sz)
  : Lemma (s_undivmod j (s_divmod j i) == i)
          [SMTPat (s_undivmod j (s_divmod j i))]
  = ()

let s_divmod_inv_2 (j:szp) (dm : sz & szlt j {SZ.fits (dm._1 * j + dm._2)})
  : Lemma (s_divmod j (s_undivmod j dm) == dm)
          [SMTPat (s_divmod j (s_undivmod j dm))]
  = ()

inline_for_extraction noextract
let sdivup (x : sz) (y : szp) : sz =
  (* The idiomatic way to round up would be
       (x +^ (y -^ 1sz)) `div` y
     which also generates nice code like (x + 7) / 8 after partial evaluation.
     But, it requires that x + (y-1) does not overflow, which means
     propagating this precondition everywhere, even though it's essentially
     unfalsifiable for a SizeT... The implementation below
     is more verbose but does not have this requirement. *)
  (x /^ y) +^ (if SZ.rem x y <> 0sz then 1sz else 0sz)

let lem_sdivup (x:sz) (y:szp)
  : Lemma (SZ.v (sdivup x y) == divup (SZ.v x) (SZ.v y))
          [SMTPat (sdivup x y)]
  = ()

inline_for_extraction noextract
let sdivup' (x : sz) (y : szp{SZ.fits (x+y-1)}) : sz =
  (x +^ (y -^ 1sz)) /^ y

let lem_sdivup' (x : sz) (y : szp{SZ.fits (x+y-1)})
  : Lemma (SZ.v (sdivup' x y) == divup x y)
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
