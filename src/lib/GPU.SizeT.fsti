module GPU.SizeT

open FStar.Ghost
module SZ = FStar.SizeT
module U32 = FStar.UInt32
module U64 = FStar.UInt64

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

[@@coercion; Pulse.Lib.Core.pulse_unfold] unfold let sizet_to_nat  (x: SZ.t)  : GTot nat = SZ.v x
[@@coercion; Pulse.Lib.Core.pulse_unfold] unfold let u32_to_nat    (x: U32.t) : GTot nat = U32.v x
[@@coercion; Pulse.Lib.Core.pulse_unfold] unfold let u64_to_nat    (x: U64.t) : GTot nat = U64.v x
[@@coercion; Pulse.Lib.Core.pulse_unfold] unfold let sizet_to_enat (x: SZ.t)  : erased nat = SZ.v x
[@@coercion; Pulse.Lib.Core.pulse_unfold] unfold let u32_to_enat   (x: U32.t) : erased nat = U32.v x
[@@coercion; Pulse.Lib.Core.pulse_unfold] unfold let u64_to_enat   (x: U64.t) : erased nat = U64.v x

(* assumption, add to F*? *)
val sizet_to_u32 (x: SZ.t) 
  : Pure U32.t (requires FStar.UInt.fits (SZ.v x) 32)
               (ensures fun r -> U32.v r == SZ.v x)
