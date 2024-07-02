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

[@@coercion] unfold let sizet_to_int  (x: SZ.t)  : GTot int = SZ.v x
[@@coercion] unfold let u32_to_int    (x: U32.t) : GTot int = U32.v x
[@@coercion] unfold let u64_to_int    (x: U64.t) : GTot int = U64.v x
[@@coercion] unfold let sizet_to_eint (x: SZ.t)  : erased int = SZ.v x
[@@coercion] unfold let u32_to_eint   (x: U32.t) : erased int = U32.v x
[@@coercion] unfold let u64_to_eint   (x: U64.t) : erased int = U64.v x

(* assumption, add to F*? *)
val sizet_to_u32 (x: SZ.t) 
  : Pure U32.t (requires FStar.UInt.fits (SZ.v x) 32)
               (ensures fun r -> U32.v r == SZ.v x)
