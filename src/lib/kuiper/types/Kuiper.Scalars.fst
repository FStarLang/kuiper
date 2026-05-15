module Kuiper.Scalars

(* The default proofs here are very flaky. *)
#set-options "--z3rlimit 1 --retry 4"

inline_for_extraction noextract
instance is_scalar_u8 : scalar UInt8.t =
  let open FStar.UInt8 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; eq;
    valid = (fun _ -> true);
    min_val = 0uy;
    max_val = 255uy;
  }

inline_for_extraction noextract
instance is_scalar_u16 : scalar UInt16.t =
  let open FStar.UInt16 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; eq;
    valid = (fun _ -> true);
    min_val = 0us;
    max_val = 65535us;
  }

inline_for_extraction noextract
instance is_scalar_u32 : scalar UInt32.t =
  let open FStar.UInt32 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; eq;
    valid = (fun _ -> true);
    min_val = 0ul;
    max_val = 4294967295ul;
  }

inline_for_extraction noextract
instance is_scalar_u64 : scalar UInt64.t =
  let open FStar.UInt64 in
  assert_norm (pow2 64 - 1 == 18446744073709551615);
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; eq;
    valid = (fun _ -> true);
    min_val = 0uL;
    max_val = 18446744073709551615uL;
  }
