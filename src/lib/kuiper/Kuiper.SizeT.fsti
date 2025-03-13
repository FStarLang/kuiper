module Kuiper.SizeT

open FStar.Ghost
open Pulse.Lib.Core
open FStar.Mul
open Kuiper.Divides

module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

unfold type sz  = FStar.SizeT.t
unfold type szp = x:sz{FStar.SizeT.v x > 0}
unfold type szlt (n:nat) = i:sz{SZ.v i < n}
unfold type szpmultiple (k:pos) = x:szp{k /? SZ.v x}

(* Throughout this repo we assume a 64bit machine. This
simplifies reasoning about overflow a bit. *)
assume SizeTFitsU64 : FStar.SizeT.fits_u64
assume SizeTFitsU32 : FStar.SizeT.fits_u32

let fits_sizet (x:nat)
  : Lemma (requires x < 0x10000000000000000)
          (ensures FStar.SizeT.fits x)
          [SMTPat (FStar.SizeT.fits x)]
  = assert_norm (pow2 64 == 0x10000000000000000);
    FStar.SizeT.fits_u64_implies_fits x

[@@coercion; pulse_unfold] unfold let sizet_to_nat  (x: SZ.t)  : GTot nat = SZ.v x
[@@coercion; pulse_unfold] unfold let u32_to_nat    (x: U32.t) : GTot nat = U32.v x
[@@coercion; pulse_unfold] unfold let u64_to_nat    (x: U64.t) : GTot nat = U64.v x
[@@coercion; pulse_unfold] unfold let sizet_to_enat (x: SZ.t)  : erased nat = SZ.v x
[@@coercion; pulse_unfold] unfold let u32_to_enat   (x: U32.t) : erased nat = U32.v x
[@@coercion; pulse_unfold] unfold let u64_to_enat   (x: U64.t) : erased nat = U64.v x

(* assumption, add to F*? *)
val sizet_to_u32 (x: SZ.t)
  : Pure U32.t (requires FStar.UInt.fits (SZ.v x) 32)
               (ensures fun r -> U32.v r == SZ.v x)

val sizet_and (x y : SZ.t) : SZ.t

(* We should extend FStar.SizeT to allow bitoperations.
This is very specialized. *)
val sizet_and_div_pow2 (x:SZ.t) (y:SZ.t) (n:nat)
  : Lemma (requires SZ.v y == pow2 n)
          (ensures SZ.v (sizet_and x SZ.(y -^ 1sz)) == SZ.v x % (pow2 n))

(* This can be assumed to skip overflow checking locally. *)
val sizet_does_not_overflow : prop

val overflow_lem () : Lemma (sizet_does_not_overflow ==> (forall n. SZ.fits n))

unfold
let between (n:int) (lohi: int & int)
  : prop = lohi._1 <= n /\ n <= lohi._2

let s_divmod (j:szp) (i:sz) : dm:(sz & szlt j){SZ.fits (dm._1 * j + dm._2)} =
  let open FStar.SizeT in
  (i `div` j, i %^ j)

let s_undivmod (j:szp) (dm : sz & szlt j {SZ.fits (dm._1 * j + dm._2)}) : sz =
  let open FStar.SizeT in
  dm._1 *^ j +^ dm._2

let s_divmod_inv_1 (j:szp) (i:sz)
  : Lemma (s_undivmod j (s_divmod j i) == i)
          [SMTPat (s_undivmod j (s_divmod j i))]
  = ()

let s_divmod_inv_2 (j:szp) (dm : sz & szlt j {SZ.fits (dm._1 * j + dm._2)})
  : Lemma (s_divmod j (s_undivmod j dm) == dm)
          [SMTPat (s_divmod j (s_undivmod j dm))]
  = ()

let three_fits (a b c : nat) : prop =
  SZ.fits (a * b) /\
  SZ.fits (b * c) /\
  SZ.fits (a * c)

let sdivup (x:sz) (y:szp{SZ.fits (x+y)}) : sz =
  let open FStar.SizeT in
  (x +^ y -^ 1sz) `div` y

let lem_sdivup (x:sz) (y:szp{SZ.fits (x+y)})
  : Lemma (SZ.v (sdivup x y) == divup (SZ.v x) (SZ.v y))
          [SMTPat (sdivup x y)]
  = ()
