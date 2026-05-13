module Kuiper.SizeT

(* F*'s SizeT + some more definitions. We also
redefine the fits predicate to make it more amenable
to prove by normalization. *)

(* Must be before the include, or it inherits an assume qualifier. *)
unfold let my_fits (x:int) : prop =
  0 <= x /\ x < 0x100000000
unfold let fits = my_fits

// We don't have DISallow lists, so we must
// list all identifiers we want to use from FStar.SizeT
include FStar.SizeT {
  t, v, add, mul, rem, uint_to_t, uint32_to_sizet,
  ( <^ ), ( <=^ ), ( >^ ), ( >=^ ),
  lt, lte, gt, gte,
  ( /^ ), ( %^ ), ( +^ ), ( -^ ), ( *^ ),
}

open Kuiper.Divides
open Kuiper.Math

module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

type sz  = SZ.t
type szp = x:sz{SZ.v x > 0}
// Note: making this argument int instead of nat prevents
// queries about non-negativity from appearing in the well-formedness
// of types.
type szlt (n:int) = i:sz{SZ.v i <  n}
type szle (n:int) = i:sz{SZ.v i <= n}

type szpmultiple (k:pos) = x:szp{k /? SZ.v x}

(* Throughout this repo we would like to assume a 64bit machine, and use
   size_t for array indices and whatnot, BUT, size_t has very poor
   performance on the GPU compared to a 32-bit integer, mostly due
   to increasing register pressure! So, our fork of karamel extracts
   size_t to uint32_t, which means we should
   NOT assume that a size_t can fit a u64, lest we could get overflow.

   The right thing to do is use FStar.UInt32.t instead of SZ.t where
   this matters, but this is a pervasive change. *)
assume SizeTFitsU32 : SZ.fits_u32

(* We also assume that those are the ONLY values of size_t that we will
   ever encounter. *)
val fits_iff_u32 (x:nat)
  : Lemma (FStar.SizeT.fits x <==> FStar.UInt.fits x 32)
          [SMTPat (FStar.SizeT.fits x)]

let fits_ok (x:nat)
  : Lemma (requires my_fits x)
          (ensures SZ.fits x)
          [SMTPat (SZ.fits x)]
  = assert (x < pow2 32);
    assert_norm (pow2 32 == 0x100000000);
    SZ.fits_u32_implies_fits x

[@@coercion] unfold let sizet_to_nat  (x: SZ.t)  : GTot nat = SZ.v x
[@@coercion] unfold let u32_to_nat    (x: U32.t) : GTot nat = U32.v x
[@@coercion] unfold let u64_to_nat    (x: U64.t) : GTot nat = U64.v x
// [@@coercion] unfold let sizet_to_enat (x: SZ.t)  : erased nat = SZ.v x
// [@@coercion] unfold let u32_to_enat   (x: U32.t) : erased nat = U32.v x
// [@@coercion] unfold let u64_to_enat   (x: U64.t) : erased nat = U64.v x

(* assumption, add to F*? *)
val sizet_to_u32 (x: SZ.t)
  : Pure U32.t (requires FStar.UInt.fits (SZ.v x) 32)
               (ensures fun r -> U32.v r == SZ.v x)

val sizet_and (x y : SZ.t) : SZ.t

(* We should extend FStar.SizeT to allow bitoperations.
This is very specialized. *)
val sizet_and_div_pow2 (x:SZ.t) (y:SZ.t) (n:nat)
  : Lemma (requires SZ.v y == pow2 n)
          (ensures SZ.v (sizet_and x (y -^ 1sz)) == SZ.v x % (pow2 n))

inline_for_extraction noextract
let s_divmod (j:szp) (i:sz) : dm:(sz & szlt j){SZ.fits (dm._1 * j + dm._2)} =
  (i /^ j, i %^ j)

inline_for_extraction noextract
let s_undivmod (j:szp) (dm : sz & szlt j {SZ.fits (dm._1 * j + dm._2)}) : sz =
  [@@inline_let] let (d, m) = dm in
  d *^ j +^ m

val s_divmod_inv_1 (j:szp) (i:sz)
  : Lemma (s_undivmod j (s_divmod j i) == i)
          [SMTPat (s_undivmod j (s_divmod j i))]

val s_divmod_inv_2 (j:szp) (dm : sz & szlt j {SZ.fits (dm._1 * j + dm._2)})
  : Lemma (s_divmod j (s_undivmod j dm) == dm)
          [SMTPat (s_divmod j (s_undivmod j dm))]

(* Round up division. Extracts to something like
     (x/y) + (x%y ? 1 : 0) *)
inline_for_extraction noextract
val sdivup (x : sz) (y : szp) : sz

val lem_sdivup (x : sz) (y : szp)
  : Lemma (SZ.v (sdivup x y) == divup x y)
          [SMTPat (sdivup x y)]

(* Also round up division. This version extracts to
      (x + y - 1) / y
    but requires the essentially-unfalsifiable precondition that
    x+y does not overflow. *)
inline_for_extraction noextract
val sdivup' (x : sz) (y : szp{SZ.fits (x+y-1)}) : sz
val lem_sdivup' (x : sz) (y : szp{SZ.fits (x+y-1)})
  : Lemma (SZ.v (sdivup' x y) == divup x y)
          [SMTPat (sdivup x y)]

inline_for_extraction noextract
val spow2 (s : sz{s < 32}) : r:sz{SZ.v r == pow2 s}

inline_for_extraction noextract
val sdiv_pow2 (i:sz{i < 32}) (tid: sz) : bool

val sdiv_pow2_ok (i:sz{i < 32}) (tid:sz) :
  Lemma (sdiv_pow2 i tid == div_pow2 i tid)
        [SMTPat (sdiv_pow2 i tid)]

noextract inline_for_extraction
let smin (a b : sz): sz =
  if a <^ b then a else b
