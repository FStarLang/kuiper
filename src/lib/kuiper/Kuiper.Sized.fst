module Kuiper.Sized

module SZ = Kuiper.SizeT

module U8  = FStar.UInt8
module U16 = FStar.UInt16
module U32 = FStar.UInt32
module U64 = FStar.UInt64

inline_for_extraction
class sized (t:Type) = {
  size : SZ.t;
  default: t;
}

inline_for_extraction noextract instance _ : sized U8.t  = { size = 1sz; default = 0uy }
inline_for_extraction noextract instance _ : sized U16.t = { size = 2sz; default = 0us }
inline_for_extraction noextract instance _ : sized U32.t = { size = 4sz; default = 0ul }
inline_for_extraction noextract instance _ : sized U64.t = { size = 8sz; default = 0uL }
